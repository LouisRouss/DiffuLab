import torch
from torch import Tensor

from diffulab.diffuse.samplers.common import StepResult
from diffulab.diffuse.samplers.gaussian_diffusion.ddpm import DDPM
from diffulab.diffuse.utils import extract_into_tensor


class DDIM(DDPM):
    name = "ddim"

    def __init__(self, mean_type: str = "epsilon", var_type: str = "fixed_small") -> None:
        """
        Denoising Diffusion Implicit Models (DDIM) sampler.
        This class implements the DDIM sampling method, which allows for deterministic sampling
        from a diffusion model. It extends the DDPM sampler and overrides the step method to
        perform the DDIM update.
        Args:
            betas (Tensor): A 1D tensor containing the beta values for each timestep in the diffusion process.
            mean_type (str, optional): The type of mean prediction used by the model. Defaults to "epsilon".
            var_type (str, optional): The type of variance prediction used by the model. Defaults to "fixed_small".
        References:
            Song, J. et al. "Denoising Diffusion Implicit Models" (2020)
            https://arxiv.org/abs/2010.02502
        """
        super().__init__(mean_type, var_type)

    def _sample_x_prev_ddim(
        self, xt: Tensor, eps: Tensor, x_start: Tensor, t: Tensor, eta: float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - x_prev (Tensor): The updated state tensor at the previous timestep x_{t-1}.
                - mean_pred (Tensor): The mean prediction before adding noise.
                - sigma (Tensor): The standard deviation of the noise added.
        """
        sigma = (
            eta
            * (
                (torch.ones_like(xt) - extract_into_tensor(self.alphas_bar_prev, t, xt.shape))
                / (torch.ones_like(xt) - extract_into_tensor(self.alphas_bar, t, xt.shape))
            ).sqrt()
            * (
                torch.ones_like(xt)
                - extract_into_tensor(self.alphas_bar, t, xt.shape)
                / extract_into_tensor(self.alphas_bar_prev, t, xt.shape)
            ).sqrt()
        )
        mean_pred = (
            x_start * extract_into_tensor(self.alphas_bar_prev, t, xt.shape).sqrt()
            + (torch.ones_like(xt) - extract_into_tensor(self.alphas_bar_prev, t, xt.shape) - sigma**2).sqrt() * eps
        )

        noise = torch.randn_like(mean_pred)
        t_mask = (t > 0).float().view(-1, *([1] * (mean_pred.dim() - 1)))
        x_prev = mean_pred + t_mask * sigma * noise
        return x_prev, mean_pred, sigma

    def step(
        self, model_prediction: Tensor, timesteps: Tensor, xt: Tensor, clamp_x: bool = False, eta: float = 0.0
    ) -> StepResult:
        """
        Perform one step of the reverse diffusion process using the DDIM update rule.
        Args:
            model_prediction (Tensor): The model's prediction at the current timestep.
            timesteps (Tensor): The current timestep in the diffusion process.
            xt (Tensor): The current state tensor at the current timestep.
            clamp_x (bool, optional): Whether to clamp the predicted x0 to a valid range
            eta (float, optional): The noise scale for the DDIM update. Defaults to 0.0 for deterministic sampling.
            *args: Additional positional arguments specific to the sampler.
            **kwargs: Additional keyword arguments specific to the sampler.
        Returns:
            StepResult: A dictionary containing the results of the step, including:
                - x_prev (Tensor): The updated state tensor at the previous timestep.
                - estimated_x0 (Tensor, optional): Estimated original data.
                - x_prev_mean (Tensor, optional): The mean of the updated state before adding noise.
                - x_prev_std (Tensor, optional): The standard deviation of the noise added.
                - logprob (Tensor, optional): The log probability of the transition.
        """
        _, _, _, x_start = self._get_p_mean_var(model_prediction, xt, timesteps, clamp_x)
        eps = self._get_eps_from_xstart(x_start, xt, timesteps)
        x_prev, ddim_mean, ddim_std = self._sample_x_prev_ddim(xt, eps, x_start, timesteps, eta)

        out = StepResult(x_prev=x_prev, estimated_x0=x_start, x_prev_mean=ddim_mean)
        if eta > 0:
            logprob = -(
                (x_prev.detach() - ddim_mean) ** 2 / (2 * ddim_std**2)
                + torch.log(ddim_std)
                + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x_prev.device))
            )
            out["x_prev_std"] = ddim_std
            out["logprob"] = logprob

        return out
