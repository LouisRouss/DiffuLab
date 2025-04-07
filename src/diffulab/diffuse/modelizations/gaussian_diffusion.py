import enum
import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diffulab.diffuse.modelizations.diffusion import Diffusion
from diffulab.diffuse.modelizations.utils import space_timesteps
from diffulab.diffuse.utils import extract_into_tensor
from diffulab.networks.denoisers.common import Denoiser, ModelInput


class MeanType(enum.Enum):
    EPSILON = "epsilon"
    XSTART = "xstart"
    XPREV = "xprev"


class ModelVarType(enum.Enum):
    LEARNED = "learned"
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED_RANGE = "learned_range"


class GaussianDiffusion(Diffusion):
    """
    Gaussian Diffusion model implementation.
    This class implements the diffusion model described in 'Denoising Diffusion Probabilistic Models'
    (Ho et al., 2020) and builds upon techniques from various follow-up works. It provides a
    framework for training and sampling from diffusion models using Gaussian noise. It is vastly
    inspired by the implementation in the OpenAI Guided Diffusion repository.
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
    under MIT LICENSE as of 2025-03-02
    Args:
        n_steps (int, optional): Number of diffusion steps. Default is 1000.
        sampling_method (str, optional): Method used for sampling. Currently supports "ddpm"
            (standard diffusion) and "ddim" (deterministic diffusion). Default is "ddpm".
        schedule (str, optional): Noise schedule to use. Options include "linear" and "cosine".
            Default is "linear".
        mean_type (str, optional): Type of parameterization used for the model's output.
            Options include:
            - "epsilon": Model predicts the noise added.
            - "xstart": Model predicts the clean data directly.
            - "xprev": Model predicts the previous timestep.
            Default is "epsilon".
        variance_type (str, optional): Type of variance computation to use. Options include:
            - "fixed_small": Use the exact posterior variance.
            - "fixed_large": Use larger variance
            - "learned": Model learns the variance.
            - "learned_range": Model learns interpolation between min/max variance.
            Default is "fixed_small".
    Attributes:
        mean_type (str): The selected mean parameterization type.
        var_type (str): The selected variance type.
        training_steps (int): Number of steps used for training.
        timestep_map (list[int]): Mapping from sampling steps to training steps when using
            fewer sampling steps than training steps.
        betas (Tensor): Beta schedule for noise levels.
        alphas (Tensor): 1 - betas.
        alphas_bar (Tensor): Cumulative product of alphas.
        sqrt_alphas_bar (Tensor): Square root of alphas_bar.
        posterior_variance (Tensor): Variance of the posterior distribution.
        posterior_log_variance_clipped (Tensor): Log of the posterior variance, clipped for numerical stability.
    Methods:
        set_steps(n_steps, schedule, section_counts): Sets the number and spacing of diffusion steps.
        draw_timesteps(batch_size): Samples random timesteps for training.
        compute_loss(model, model_inputs, timesteps, noise): Computes the training loss.
        denoise(model, data_shape, model_inputs, use_tqdm, clamp_x, guidance_scale):
            Performs the reverse diffusion process to generate samples.
        add_noise(x, timesteps, noise): Adds noise to inputs according to the forward process.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        sampling_method: str = "ddpm",
        schedule: str = "linear",
        mean_type: str = "epsilon",
        variance_type: str = "fixed_small",
    ):
        if mean_type not in MeanType._value2member_map_:
            raise ValueError(f"mean_type must be one of {[e.value for e in MeanType]}")
        if variance_type not in ModelVarType._value2member_map_:
            raise ValueError(f"variance_type must be one of {[e.value for e in ModelVarType]}")
        if sampling_method not in ["ddpm", "ddim"]:
            raise ValueError("sampling method must be one of ['ddpm', 'ddim']")

        self.mean_type = mean_type
        self.var_type = variance_type
        self.training_steps = n_steps
        super().__init__(n_steps=n_steps, sampling_method=sampling_method, schedule=schedule)

    def set_diffusion_parameters(self, betas: Tensor) -> None:
        """
        Sets up the diffusion parameters for the Gaussian diffusion process.
        This method computes various coefficients and parameters needed for the diffusion process
        after the noise schedule (betas) has been defined. It calculates alpha values, their
        cumulative products, and various coefficients needed for the forward and reverse processes.
        Args:
            betas (Tensor): The noise schedule tensor defining beta values for each timestep
                in the diffusion process.
        """
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alphas_bar[:-1]])
        self.alphas_bar_next = torch.cat([self.alphas_bar[1:], torch.tensor([0.0], dtype=torch.float64)])

        # utils for computation
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.posterior_variance = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * (self.alphas_bar_prev).sqrt() / (1.0 - self.alphas_bar)
        self.posterior_mean_coef2 = (1.0 - self.alphas_bar_prev) * self.alphas.sqrt() / (1.0 - self.alphas_bar)

    def set_steps(
        self,
        n_steps: int,
        schedule: str = "linear",
        section_counts: int | str | None = None,
    ) -> None:
        """
        Sets the number of diffusion steps and configures the variance schedule.
        This method updates the diffusion process to use a specific number of steps and
        schedule type. It allows changing from the training configuration to a potentially
        different sampling configuration by remapping the timesteps.
        Args:
            n_steps (int): The number of diffusion steps to use for sampling.
            schedule (str, optional): The type of schedule to use . Options include linear or cosine
            section_counts (int | str | None, optional): Controls how timesteps are selected when
                using fewer sampling steps than training steps:
                - When an integer, selects that many timesteps evenly spaced in the original range
                - When "ddim", uses DDIM-style timestep selection
                - When None and n_steps differs from training_steps, defaults to n_steps
                - When None and n_steps equals training_steps, uses all training steps
                Defaults to None.
        Note:
            When using fewer steps for sampling than were used for training, this method
            computes new beta values that preserve the same overall noise schedule but with
            fewer steps. It also creates a timestep_map that maps from sampling indices to
            the original training timesteps.
        """
        if n_steps != self.training_steps:
            section_counts = section_counts or n_steps
        self.steps = n_steps

        betas = self._get_variance_schedule(self.training_steps, schedule)
        self.set_diffusion_parameters(betas)
        self.timestep_map: list[int] = []

        if section_counts:
            timesteps_to_use = space_timesteps(
                num_timesteps=self.training_steps, section_counts=section_counts, ddim=self.sampling_method == "ddim"
            )
            last_alpha_bar = torch.tensor(1.0)
            new_betas: list[Tensor] = []
            for i, alpha_bar in enumerate(self.alphas_bar):
                if i in timesteps_to_use:
                    new_betas.append(1 - alpha_bar / last_alpha_bar)
                    last_alpha_bar = alpha_bar
                    self.timestep_map.append(i)
            self.set_diffusion_parameters(torch.tensor(new_betas))

    def _get_variance_schedule(self, n_steps: int, variance_schedule: str = "linear") -> Tensor:
        """
        Creates a variance schedule for the diffusion process.
        This function generates the beta values that define the noise schedule for the diffusion process.
        Different schedules produce different noise patterns over time and can affect model performance.
        Args:
            n_steps (int): Number of diffusion steps for which to generate the variance schedule.
            variance_schedule (str, optional): The type of schedule to use. Options include:
                - "linear": Linear schedule from Ho et al. (2020), scaled to work for any number of steps.
                - "cosine": Cosine schedule that smoothly increases the noise variance.
                Defaults to "linear".
        Returns:
            Tensor: A 1D tensor containing beta values for each timestep, with shape (n_steps,).
        Raises:
            NotImplementedError: If an unsupported variance schedule type is specified.
        References:
            Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
            https://arxiv.org/abs/2006.11239
            For cosine schedule: Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (2021)
            https://arxiv.org/abs/2102.09672
        """

        if variance_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64, requires_grad=False)
        elif variance_schedule == "cosine":
            return self._betas_for_alpha_bar(
                n_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {variance_schedule}")

    def _betas_for_alpha_bar(
        self, n_steps: int, alpha_bar: Callable[[float], float], max_beta: float = 0.999
    ) -> Tensor:
        """
        Computes beta values for a given alpha bar function.
        This helper method generates a sequence of beta values based on a provided alpha bar function.
        The beta values are derived to achieve the desired noise schedule defined by alpha_bar.
        Args:
            n_steps (int): Number of diffusion steps to generate betas for.
            alpha_bar (Callable[[float], float]): A function that takes a normalized timestep (t/n_steps)
                and returns the corresponding cumulative product of alphas (alpha_bar) at that timestep.
            max_beta (float, optional): Maximum allowed value for any beta to ensure numerical stability.
                Defaults to 0.999.
        Returns:
            Tensor: A tensor of beta values with shape (n_steps,) and dtype torch.float64,
                representing the variance schedule for the diffusion process.
        """
        betas: list[float] = []
        for i in range(n_steps):
            t1 = i / n_steps
            t2 = (i + 1) / n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float64, requires_grad=False)

    def _get_x_start_from_x_prev(self, x_prev: Tensor, x: Tensor, t: Tensor) -> Tensor:
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
        x_start = (1.0 / extract_into_tensor(self.posterior_mean_coef1, t, x_prev.shape)) * x_prev + (
            1.0 / extract_into_tensor(self.posterior_mean_coef2, t, x.shape)
        ) * x
        return x_start

    def _get_x_start_from_eps(self, eps: Tensor, x: Tensor, t: Tensor) -> Tensor:
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

        x_start = (1.0 / extract_into_tensor(self.sqrt_alphas_bar, t, x.shape)) * x - (
            (1.0 - extract_into_tensor(self.alphas_bar, t, eps.shape)).sqrt() / self.sqrt_alphas_bar[t]
        ) * eps
        return x_start

    def _get_mean_from_x_start(self, x: Tensor, x_start: Tensor, t: Tensor) -> Tensor:
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
            + extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )
        return mean

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

    def _get_eps_from_xstart(self, x_start: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes the noise component (epsilon) from the initial state and current state.
        This method is the inverse of _get_x_start_from_eps, calculating the noise that would
        have been added to x_start to produce the current noisy state x at timestep t.
        Args:
            x_start (Tensor): The inferred or known initial clean state (x_0).
            x (Tensor): The current noisy state at timestep t (x_t).
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The noise component (epsilon) that was added to x_start to produce x.
        """
        eps = (1.0 / (1 - extract_into_tensor(self.alphas_bar, t, x.shape)).sqrt()) * (
            x - extract_into_tensor(self.sqrt_alphas_bar, t, x_start.shape) * x_start
        )
        return eps

    def _get_p_mean_var(
        self, prediction: Tensor, x: Tensor, t: Tensor, clamp_x: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes the posterior mean and variance for the reverse diffusion step.
        This core method calculates the parameters of the posterior distribution
        p(x_{t-1} | x_t) used in the reverse diffusion process. It processes the model's
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
                - log_var (Tensor): The log variance of the posterior distribution.
                - x_start (Tensor): The predicted clean data (x_0) based on current state and prediction.
        """
        if self.var_type in [ModelVarType.FIXED_SMALL.value, ModelVarType.FIXED_LARGE.value]:
            model_output = prediction
        elif self.var_type in [ModelVarType.LEARNED.value, ModelVarType.LEARNED_RANGE.value]:
            assert prediction.shape[1] % 2 == 0
            model_output, log_var = torch.chunk(prediction, 2, dim=1)
        else:
            raise ValueError(f"Unknown model var type: {self.var_type}")

        # extract mean
        if self.mean_type == MeanType.XPREV.value:
            x_start = self._get_x_start_from_x_prev(model_output, x, t)
        elif self.mean_type == MeanType.XSTART.value:
            x_start = model_output
        elif self.mean_type == MeanType.EPSILON.value:
            x_start = self._get_x_start_from_eps(model_output, x, t)
        else:
            raise ValueError(f"Unknown mean type: {self.mean_type}")
        if clamp_x:
            x_start = torch.clamp(x_start, -1, 1)
        mean = self._get_mean_from_x_start(x, x_start, t)

        # extract variance
        if self.var_type == ModelVarType.FIXED_SMALL.value:
            var, log_var = (
                extract_into_tensor(self.posterior_variance, t, x.shape),
                extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape),
            )
        elif self.var_type == ModelVarType.FIXED_LARGE.value:
            var, log_var = (
                torch.cat([self.posterior_variance[1:2], self.betas[1:]]),
                torch.cat([self.posterior_log_variance_clipped[1:2], self.betas[1:]]),
            )
            var, log_var = extract_into_tensor(var, t, x.shape), extract_into_tensor(log_var, t, x.shape)
        elif self.var_type == ModelVarType.LEARNED.value:
            var = log_var.exp()  # type: ignore
        elif self.var_type == ModelVarType.LEARNED_RANGE.value:
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(self.betas, t, x.shape).log()
            log_var = (log_var + 1) / 2  # type: ignore
            log_var = log_var * max_log + (1 - log_var) * min_log
            var = log_var.exp()
        else:
            raise ValueError(f"Unknown model var type: {self.var_type}")

        return mean, var, log_var, x_start  # type: ignore

    def _get_mean_for_ddim_guidance(
        self, x: Tensor, x_start: Tensor, t: Tensor, grad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the adjusted mean for DDIM with gradient guidance.
        This method computes a modified mean for the next timestep in the reverse diffusion process
        when using classifier guidance with DDIM sampling. It adjusts the predicted noise (epsilon)
        based on the gradient input, then recalculates the predicted clean image and mean accordingly.
        Args:
            x (Tensor): The current noisy state at timestep t.
            x_start (Tensor): The predicted initial clean state.
            t (Tensor): The current timestep indices as a tensor.
            grad (Tensor): The gradient used for guidance, typically from a classifier.
        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - mean (Tensor): The adjusted mean for the next state in the reverse diffusion process.
                - x_start (Tensor): The updated predicted clean state based on the guided noise.
                - eps (Tensor): The adjusted noise prediction after applying the guidance.
        """
        eps = self._get_eps_from_xstart(x, x_start, t)
        eps = eps - (1 - extract_into_tensor(self.alphas_bar, t, grad.shape)).sqrt() * grad

        x_start = self._get_x_start_from_eps(eps, x, t)
        mean = self._get_mean_from_x_start(x, x_start, t)

        return mean, x_start, eps

    def _sample_x_prev_ddim(self, x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """
        Performs a deterministic DDIM update from state x_t to x_{t-1}.
        This method implements the Denoising Diffusion Implicit Models (DDIM) update rule,
        which allows for deterministic sampling from the diffusion model. It computes x_{t-1}
        given x_t and the predicted noise (epsilon) using coefficients from the diffusion process.
        Args:
            x (Tensor): The current state x_t at timestep t.
            eps (Tensor): The predicted noise component (epsilon) at timestep t.
            t (Tensor): The current timestep indices as a tensor.
        Returns:
            Tensor: The previous state x_{t-1} after the deterministic DDIM update.
        References:
            Song, J. et al. "Denoising Diffusion Implicit Models" (2020)
            https://arxiv.org/abs/2010.02502
        """
        x_prev = (
            extract_into_tensor(self.alphas_bar_prev, t, x.shape).sqrt()
            * (
                (x - (1 - extract_into_tensor(self.alphas_bar, t, x.shape)).sqrt() * eps)
                / extract_into_tensor(self.sqrt_alphas_bar, t, x.shape)
            )
            + (1 - extract_into_tensor(self.alphas_bar_prev, t, eps.shape)).sqrt() * eps
        )
        return x_prev

    def classifier_grad(
        self, x: Tensor, y: Tensor, t: Tensor, classifier: Callable[[Tensor, Tensor], Tensor]
    ) -> Tensor:
        """
        Computes the gradient of a classifier's output with respect to its input.
        This function calculates the gradient of the log probability of the target class
        with respect to the input. This gradient can be used for classifier guidance in
        diffusion models, steering the generation process toward samples that the classifier
        associates with the target class.
        Args:
            x (Tensor): The input tensor for which to compute gradients. Will be detached
                from its computation graph and have requires_grad set to True.
            y (Tensor): The target class indices, with shape (batch_size,).
            t (Tensor): The timestep tensor passed to the classifier.
            classifier (Callable[[Tensor, Tensor], Tensor]): A function that takes the input x
                and timestep t, and returns logits for each class.
        Returns:
            Tensor: The gradient of the log probability of the target class with respect to
                the input x, with the same shape as x.
        """
        x = x.detach().requires_grad_()
        logits = classifier(x, t)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs_y = logprobs[torch.arange(y.size(0)), y]
        return torch.autograd.grad(logprobs_y.sum(), x)[0]

    def draw_timesteps(self, batch_size: int) -> Tensor:
        """
        Draws random timesteps for training the diffusion model.
        This function samples timesteps uniformly from the range [0, steps-1] for each item
        in a batch. These timesteps are used in the diffusion process to determine how much
        noise to add to the input data.
        Args:
            batch_size (int): The number of timesteps to draw, typically matching
                the batch size of the training data.
        Returns:
            Tensor: A tensor of shape (batch_size,) containing integer timestep indices
                uniformly sampled from [0, steps-1].
        """

        return torch.randint(0, self.steps, (batch_size,), dtype=torch.int32)

    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        t: int,
        clamp_x: bool = False,
        guidance_scale: float = 0.0,
        classifier_free: bool = True,
        classifier: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Performs one step of the reverse diffusion process.
        This method denoises the input data by one timestep using the trained model.
        It supports both classifier-free guidance and classifier-based guidance for
        conditional generation.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including
                the current noisy data tensor keyed as 'x' and conditional information.
            t (int): The current timestep index.
            clamp_x (bool, optional): Whether to clamp x_start predictions to a valid range.
                Defaults to False.
            guidance_scale (float, optional): Scale factor for guidance. Values greater than 0
                enable guidance, with higher values emphasizing the conditional prediction more.
                Defaults to 0.0 (no guidance).
            classifier_free (bool, optional): Whether to use classifier-free guidance (True) or
                classifier-based guidance (False). Defaults to True.
            classifier (Callable[[Tensor, Tensor], Tensor] | None, optional): A classifier function
                for gradient-based guidance when classifier_free is False. Defaults to None.
        Returns:
            Tensor: The denoised data tensor after taking one reverse diffusion step.
        Notes:
            - When using classifier-free guidance, the model is run twice: once with and once
              without conditioning inputs, and the results are interpolated.
            - When using classifier-based guidance, gradients from a classifier are used to
              guide the sampling process toward the desired class or condition.
            - The timestep mapping is applied if a different number of sampling steps than
              training steps is being used.
        """
        device = next(model.parameters()).device
        timesteps = torch.full((model_inputs["x"].shape[0],), t, device=device, dtype=torch.int32)
        if self.timestep_map:
            map_tensor = torch.tensor(self.timestep_map, device=timesteps.device, dtype=timesteps.dtype)
            timesteps = map_tensor[timesteps]
        prediction = model(**{**model_inputs, "p": 0}, timesteps=timesteps)

        if classifier_free and guidance_scale > 0:
            prediction_uncond = model(**{**model_inputs, "p": 1}, timesteps=timesteps)
            prediction = prediction + guidance_scale * (prediction - prediction_uncond)

        mean, _, log_var, x_start = self._get_p_mean_var(prediction, model_inputs["x"], timesteps, clamp_x)

        if not classifier_free and guidance_scale > 0:
            assert classifier is not None
            if "y" in model_inputs:
                grad = guidance_scale * self.classifier_grad(prediction, model_inputs["y"], timesteps, classifier)
            elif "context" in model_inputs:
                grad = guidance_scale * self.classifier_grad(prediction, model_inputs["context"], timesteps, classifier)
            else:
                raise ValueError("No context or label provided for the classifier")

            if self.sampling_method == "ddpm":
                mean = mean + grad
                x_prev = self._get_x_prev_from_mean_var(mean, log_var, timesteps)
            elif self.sampling_method == "ddim":
                mean, x_start, eps = self._get_mean_for_ddim_guidance(model_inputs["x"], x_start, timesteps, grad)
                x_prev = self._sample_x_prev_ddim(model_inputs["x"], eps, timesteps)
            else:
                raise NotImplementedError(
                    f"Classifier guidance not implemented for sampling method: {self.sampling_method}"
                )
        else:
            x_prev = self._get_x_prev_from_mean_var(mean, log_var, timesteps)

        return x_prev

    # maybe change it to avoid inplace change of the dict
    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 0,
        classifier_free: bool = True,
        classifier: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Generates a sample by running the entire reverse diffusion process.
        This method implements the full reverse process of the diffusion model, iteratively
        applying the one_step_denoise function from the most noisy state to progressively
        cleaner states. It can generate samples unconditionally or with guidance.
        Args:
            model (Denoiser): The neural network model used for denoising.
            data_shape (tuple[int, ...]): Shape of the data to generate, typically (batch_size, channels, height, width).
            model_inputs (ModelInput): A dictionary containing the model inputs, potentially including:
                - 'x': Initial noise tensor. If not provided, random noise will be generated.
                - conditional inputs like context embeddings, class labels, etc.
            use_tqdm (bool, optional): Whether to display a progress bar during generation. Defaults to True.
            clamp_x (bool, optional): Whether to clamp the generated values to a valid range (typically [-1, 1]).
                Defaults to True.
            guidance_scale (float, optional): Scale factor for guidance. If greater than 0, applies guidance
                to steer the generation. Defaults to 0.
            classifier_free (bool, optional): Whether to use classifier-free guidance. If True, uses the conditional
                and unconditional model outputs for guidance. If False, uses a separate classifier. Defaults to True.
            classifier (Callable[[Tensor, Tensor], Tensor] | None, optional): External classifier function to use
                for guidance when classifier_free is False. Accepts input tensor and timesteps, returns gradients.
                Defaults to None.
        """
        if "x" not in model_inputs:
            model_inputs["x"] = torch.randn(
                data_shape, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
            )
        for t in tqdm(
            list(range(self.steps))[::-1],
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            model_inputs["x"] = self.one_step_denoise(
                model=model,
                model_inputs=model_inputs,
                t=t,
                clamp_x=clamp_x,
                guidance_scale=guidance_scale,
                classifier_free=classifier_free,
                classifier=classifier,
            )
        return model_inputs["x"]

    ### Need to add compute loss for different parameterization + variance learned
    def compute_loss(
        self, model: Denoiser, model_inputs: ModelInput, timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        """
        Computes the loss for training the Gaussian diffusion model.
        This method calculates the mean squared error loss between the model's prediction
        and the noise added during the forward diffusion process. It first applies noise to
        the input data according to the specified timesteps, then gets the model's prediction
        of that noise and computes the MSE loss between the predicted and true noise.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including
                the clean data tensor keyed as 'x' and any conditional information.
            timesteps (Tensor): A tensor of shape (batch_size,) containing timestep indices.
            noise (Tensor | None, optional): Pre-generated noise to add to the inputs.
                If None, random noise will be generated. Defaults to None.
        Returns:
            Tensor: The computed mean squared error loss as a scalar tensor.
        Note:
            When using a different number of sampling steps than training steps,
            this method maps the timestep indices through the timestep_map to ensure
            the model sees the correct noise level for each step.
        """
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        if self.timestep_map:
            map_tensor = torch.tensor(self.timestep_map, device=timesteps.device, dtype=timesteps.dtype)
            timesteps = map_tensor[timesteps]
        prediction = model(**model_inputs, timesteps=timesteps)
        loss = nn.functional.mse_loss(prediction, noise, reduction="mean")
        return loss

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Adds noise to the input data according to the Gaussian diffusion process.
        This method implements the forward diffusion process by adding noise to the input
        data according to a specific timestep, following the equation:
        x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * noise
        where:
        - α̅_t is the cumulative product of (1 - β_t) up to timestep t
        - β_t is the noise schedule parameter at timestep t
        Args:
            x (Tensor): The clean input data tensor to be noised, typically representing x_0.
            timesteps (Tensor): A tensor of shape (batch_size,) containing timestep indices
                for each example in the batch.
            noise (Tensor | None, optional): Pre-defined noise to add to the input.
                If None, random Gaussian noise will be generated. Defaults to None.
        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - x_t (Tensor): The noised input at the specified timesteps.
                - noise (Tensor): The noise used in the forward process.
        """

        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        x_t = (
            extract_into_tensor(self.sqrt_alphas_bar, timesteps, x.shape) * x
            + (1 - extract_into_tensor(self.alphas_bar, timesteps, noise.shape)).sqrt() * noise
        )
        return x_t, noise
