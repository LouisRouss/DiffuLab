from typing import cast

import torch
from torch import Tensor

from diffulab.diffuse.modelizations.gaussian_diffusion import MeanType, ModelVarType
from diffulab.diffuse.samplers.common import StepResult
from diffulab.diffuse.samplers.gaussian_diffusion.common import Sampler
from diffulab.diffuse.utils import extract_into_tensor


class DDPM(Sampler):
    name = "ddpm"

    def __init__(self, mean_type: str = "epsilon", var_type: str = "fixed_small") -> None:
        """
        Denoising Diffusion Probabilistic Model (DDPM) sampler.

        Args:
            mean_type (str, optional): Type of mean prediction. Options are "epsilon", "x0", or "v".
                Defaults to "epsilon".
            var_type (str, optional): Type of variance prediction. Options are "fixed_small", "fixed_large",
                "learned", or "learned_range". Defaults to "fixed_small".
        """
        super().__init__()
        self.mean_type = mean_type
        self.var_type = var_type

    def set_steps(self, betas: Tensor) -> None:
        """
        Precomputes and sets various parameters required for the DDPM sampling process based on the provided
        beta schedule. This includes calculations for alphas, cumulative products of alphas, posterior variance,
        and coefficients used in the reverse diffusion steps.
        Args:
            betas (Tensor): A 1D tensor representing the beta schedule for the diffusion process.
        Raises:
            ValueError: If the provided betas tensor is not 1-dimensional.
        """
        self.betas = betas
        self.alphas = torch.ones_like(self.betas) - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alphas_bar[:-1]])
        self.alphas_bar_next = torch.cat([self.alphas_bar[1:], torch.tensor([0.0], dtype=torch.float64)])

        # utils for computation
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.posterior_variance = (
            self.betas
            * (torch.ones_like(self.alphas_bar_prev) - self.alphas_bar_prev)
            / (torch.ones_like(self.alphas_bar) - self.alphas_bar)
        )

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * (self.alphas_bar_prev).sqrt() / (torch.ones_like(self.alphas_bar) - self.alphas_bar)
        )
        self.posterior_mean_coef2 = (
            (torch.ones_like(self.alphas_bar_prev) - self.alphas_bar_prev)
            * self.alphas.sqrt()
            / (torch.ones_like(self.alphas_bar) - self.alphas_bar)
        )

    def _get_x_start_from_x_prev(self, x_prev: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """
        Computes the initial state (x_0) given the previous state (x_{t-1}) and current state (x_t).
        It uses precomputed coefficients to solve for x_0 based on the Gaussian diffusion model's
        posterior distribution parameters.
        Args:
            x_prev (Tensor): The state at the previous timestep (x_{t-1}).
            x (Tensor): The state at the current timestep (x_t).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The inferred initial clean state (x_0) based on the provided states.
        """
        x_start = (1.0 / extract_into_tensor(self.posterior_mean_coef1, t, x_prev.shape)) * x_prev - (
            extract_into_tensor(self.posterior_mean_coef2, t, xt.shape)
            / extract_into_tensor(self.posterior_mean_coef1, t, xt.shape)
        ) * xt
        return x_start

    def _get_x_start_from_eps(self, eps: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """
        Computes the initial state (x_0) from the current state (x_t) and the predicted noise (epsilon).
        This method uses the diffusion model's forward process parameters to invert the noise
        addition and recover the original clean data, given the current noisy state and the
        predicted noise component.
        Args:
            eps (Tensor): The predicted noise component (epsilon).
            x (Tensor): The current noisy state at timestep t (x_t).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The inferred initial clean state (x_0) based on the current state and predicted noise.
        """
        x_start = (1.0 / extract_into_tensor(self.sqrt_alphas_bar, t, xt.shape)) * xt - (
            (torch.ones_like(eps) - extract_into_tensor(self.alphas_bar, t, eps.shape)).sqrt()
            / extract_into_tensor(self.sqrt_alphas_bar, t, xt.shape)
        ) * eps
        return x_start

    def get_x_start(self, model_output: Tensor, xt: Tensor, t: Tensor, clamp_x: bool = False) -> Tensor:
        """
        Infers the initial state (x_0) from the model's output based on the specified mean prediction type.
        This method determines how to interpret the model's output (whether as noise, x_0, or x_{t-1})
        and computes the corresponding x_0. It also optionally clamps the inferred x_0 to a valid range
        for numerical stability.
        Args:
            model_output (Tensor): The raw output from the denoising model.
            x (Tensor): The current noisy state at timestep t (x_t).
            t (Tensor): The current timestep indices as a tensor.
            clamp_x (bool, optional): Whether to clamp the predicted x_start to [-1, 1] range
                for numerical stability. Defaults to False.
        Returns:
            Tensor: The inferred initial clean state (x_0) based on the model's output and current state.
        Raises:
            ValueError: If an unknown mean type is specified.
        """
        dispatch = {
            MeanType.XPREV.value: lambda: self._get_x_start_from_x_prev(model_output, xt, t),
            MeanType.XSTART.value: lambda: model_output,
            MeanType.EPSILON.value: lambda: self._get_x_start_from_eps(model_output, xt, t),
        }

        try:
            x_start = dispatch[self.mean_type]()
        except KeyError:
            raise ValueError(f"Unknown mean type: {self.mean_type}")

        if clamp_x:
            x_start = torch.clamp(x_start, -1, 1)
        return x_start

    def _get_mean_from_x_start(self, xt: Tensor, x_start: Tensor, t: Tensor) -> Tensor:
        """
        Computes the mean of the posterior distribution q(x_{t-1} | x_t, x_0) given the current state and inferred initial state.
        This function calculates the mean of the posterior distribution using pre-computed coefficients that
        depend on the noise schedule. The posterior mean is used during the reverse diffusion process to sample
        from q(x_{t-1} | x_t, x_0).
        Args:
            x (Tensor): The current state at timestep t (x_t).
            x_start (Tensor): The inferred or predicted initial clean state (x_0).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The mean of the posterior distribution q(x_{t-1} | x_t, x_0).
        """

        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        return mean

    def get_variance(
        self,
        t: Tensor,
        x_shape: tuple[int, ...],
        log_var: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Computes the variance and log variance for the posterior distribution q(x_{t-1} | x_t, x_0).
        This method determines the variance type specified during initialization and computes
        the corresponding variance and log variance values. It supports fixed small, fixed large,
        learned, and learned range variance types.
        Args:
            t (Tensor): The current timestep indices as a tensor.
            x_shape (tuple[int, ...]): The shape of the input tensor for which variance is computed.
            log_var (Tensor | None, optional): The predicted log variance from the model, required
                if the variance type is "learned" or "learned_range". Defaults to None.
        Returns:
            tuple[Tensor, Tensor | None]: A tuple containing:
                - var (Tensor): The computed variance for the posterior distribution.
                - log_var (Tensor | None): The computed log variance, or None if not applicable.
        Raises:
            ValueError: If an unknown variance type is specified or if log_var is required but not provided.
        """

        def _fixed_small():
            v = extract_into_tensor(self.posterior_variance, t, x_shape)
            lv = extract_into_tensor(self.posterior_log_variance_clipped, t, x_shape)
            return v, lv

        def _fixed_large():
            v_seq = torch.cat([self.posterior_variance[1:2], self.betas[1:]])
            lv_seq = torch.log(v_seq)
            v = extract_into_tensor(v_seq, t, x_shape)
            lv = extract_into_tensor(lv_seq, t, x_shape)
            return v, lv

        def _learned():
            assert log_var is not None, "log_var must be provided for LEARNED"
            return log_var.exp(), log_var

        def _learned_range():
            assert log_var is not None, "log_var must be provided for LEARNED_RANGE"
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x_shape)
            max_log = extract_into_tensor(self.betas, t, x_shape).log()
            w = (log_var + 1) / 2
            lv = cast(Tensor, w * max_log + (1 - w) * min_log)
            return lv.exp(), lv

        dispatch = {
            ModelVarType.FIXED_SMALL.value: _fixed_small,
            ModelVarType.FIXED_LARGE.value: _fixed_large,
            ModelVarType.LEARNED.value: _learned,
            ModelVarType.LEARNED_RANGE.value: _learned_range,
        }

        try:
            var, lv = dispatch[self.var_type]()
        except KeyError:
            raise ValueError(f"Unknown model var type: {self.var_type}")

        return var, lv

    def _get_p_mean_var(
        self, prediction: Tensor, xt: Tensor, t: Tensor, clamp_x: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes the posterior mean and variance for the reverse diffusion step.
        This core method calculates the parameters of the posterior distribution
        p(x_{t-1} | x_t, x_0) used in the reverse diffusion process. It processes the model's
        prediction and current noisy state to extract the mean, variance, log variance,
        and the predicted clean state.
        Args:
            prediction (Tensor): The raw output from the denoising model. Could be noise prediction,
                x_start prediction, or x_prev prediction depending on self.mean_type.
            x (Tensor): The current noisy state at timestep t (x_t).
            t (Tensor): The current timestep indices as a tensor.
            clamp_x (bool, optional): Whether to clamp the predicted x_start to [-1, 1] range
                for numerical stability. Defaults to False.
        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - mean (Tensor): The mean of the posterior distribution p(x_{t-1} | x_t).
                - var (Tensor): The variance of the posterior distribution.
                - log_var (Tensor | None): The log variance of the posterior distribution.
                - x_start (Tensor): The predicted clean data (x_0) based on current state and prediction.
        """
        assert self.var_type in [
            ModelVarType.FIXED_SMALL.value,
            ModelVarType.FIXED_LARGE.value,
            ModelVarType.LEARNED.value,
            ModelVarType.LEARNED_RANGE.value,
        ]

        model_output = prediction
        log_var: Tensor | None = None
        if self.var_type in [ModelVarType.LEARNED.value, ModelVarType.LEARNED_RANGE.value]:
            assert model_output.shape[1] % 2 == 0
            model_output, log_var = torch.chunk(model_output, 2, dim=1)

        # extract x_start
        x_start = self.get_x_start(model_output, xt, t, clamp_x)
        # extract mean
        mean = self._get_mean_from_x_start(xt, x_start, t)
        # extract variance
        var, log_var = self.get_variance(t, xt.shape, log_var)
        assert log_var is not None  # for pyright

        return mean, var, log_var, x_start

    def _get_x_prev_from_mean_var(self, mean: Tensor, log_var: Tensor, t: Tensor) -> Tensor:
        """
        Computes the previous state (x_{t-1}) from the posterior mean and variance.
        This method samples the previous state from the posterior distribution p(x_{t-1}|x_t)
        using the provided mean and variance parameters. For all timesteps except the final one (t=0),
        it adds random Gaussian noise scaled by the square root of the variance. For the final timestep,
        it returns the mean without adding noise.
        Args:
            mean (Tensor): The posterior mean of p(x_{t-1}|x_t).
            var (Tensor): The posterior variance of p(x_{t-1}|x_t).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The sampled previous state (x_{t-1}) from the posterior distribution.
        """
        # Create a mask where t > 0 is True, and t == 0 is False
        t_mask = (t > 0).float().view(-1, *([1] * (mean.dim() - 1)))

        # Generate noise with the same shape as mean
        noise = torch.randn_like(mean)

        # Apply noise only where t > 0, using the mask to conditionally add noise
        return mean + t_mask * noise * torch.exp(0.5 * log_var)

    def _get_eps_from_xstart(self, x_start: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """
        Computes the noise component (epsilon) from the initial state and current state.
        This method is the inverse of _get_x_start_from_eps, calculating the noise that would
        have been added to x_start to produce the current noisy state x at timestep t.
        Args:
            x_start (Tensor): The inferred or known initial clean state (x_0).
            x (Tensor): The current noisy state at timestep t (x_t).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            StepResult: A dictionary containing the results of the step, including:
                - x_prev (Tensor): The updated state tensor at the previous timestep.
                - estimated_x0 (Tensor, optional): Estimated original data.
                - x_prev_mean (Tensor, optional): The mean of the updated state before adding noise.
                - x_prev_std (Tensor, optional): The standard deviation of the noise added.
                - logprob (Tensor, optional): The log probability of the transition.
        """
        eps = ((1 / extract_into_tensor(self.sqrt_alphas_bar, t, xt.shape)) * xt - x_start) / (
            1 / extract_into_tensor(self.alphas_bar, t, xt.shape) - 1
        ).sqrt()

        return eps

    def step(self, model_prediction: Tensor, timesteps: Tensor, xt: Tensor, clamp_x: bool = False) -> StepResult:
        mean, var, log_var, x_start = self._get_p_mean_var(model_prediction, xt, timesteps, clamp_x)
        x_prev = self._get_x_prev_from_mean_var(mean, log_var, timesteps)

        var_safe = var.clamp_min(1e-20)
        const = torch.log(2 * torch.pi * var_safe) * 0.5
        elem = -((x_prev.detach() - mean) ** 2) / (2.0 * var_safe) - const
        # zero out at t==0 to avoid degenerate edge-case
        t_mask = (timesteps > 0).float().view(-1, *([1] * (elem.dim() - 1)))
        logprob = elem * t_mask

        return StepResult(
            x_prev=x_prev, estimated_x0=x_start, x_prev_mean=mean, x_prev_std=var_safe.sqrt(), logprob=logprob
        )
