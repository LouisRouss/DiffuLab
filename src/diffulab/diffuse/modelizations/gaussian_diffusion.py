import math
from typing import Any, Callable, cast

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diffulab.diffuse.modelizations.diffusion import Diffusion
from diffulab.diffuse.modelizations.utils import space_timesteps
from diffulab.diffuse.samplers.common import StepResult
from diffulab.diffuse.samplers.gaussian_diffusion import DDIM, DDPM
from diffulab.diffuse.utils import SamplingOutput, extract_into_tensor
from diffulab.networks.denoisers.common import Denoiser, ModelInput
from diffulab.training.losses.common import LossFunction


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
        n_steps (int, optional): Number of diffusion steps. Default is 1000. Set to match training steps.
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

    sampler_registry = {
        "ddpm": DDPM,
        "ddim": DDIM,
    }

    def __init__(
        self,
        n_steps: int = 1000,
        sampling_method: str = "ddpm",
        schedule: str = "linear",
        latent_diffusion: bool = False,
        sampler_parameters: dict[str, Any] = {},
    ):
        if sampling_method not in ["ddpm", "ddim"]:
            raise ValueError("sampling method must be one of ['ddpm', 'ddim']")

        self.training_steps = n_steps
        super().__init__(
            n_steps=self.training_steps,
            sampling_method=sampling_method,
            schedule=schedule,
            latent_diffusion=latent_diffusion,
            sampler_parameters=sampler_parameters,
        )

    def set_diffusion_parameters(self, betas: Tensor) -> None:
        self.betas = betas
        self.alphas = torch.ones_like(self.betas) - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sampler.set_steps(betas)

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
                    new_betas.append(torch.ones_like(alpha_bar) - alpha_bar / last_alpha_bar)
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
        sampler_args: dict[str, Any] = {},
    ) -> StepResult:
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
            sampler_args (dict[str, Any], optional): Additional arguments to pass to the sampler's step method.
                Defaults to an empty dictionary.
        Returns:
            Tensor: The denoised data tensor after taking one reverse diffusion step.
        Notes:
            - The timestep mapping is applied if a different number of sampling steps than
              training steps is being used.
        """
        device = next(model.parameters()).device
        timesteps = torch.full((model_inputs["x"].shape[0],), t, device=device, dtype=torch.int32)
        if self.timestep_map:
            map_tensor = torch.tensor(self.timestep_map, device=timesteps.device, dtype=timesteps.dtype)
            timesteps_model = map_tensor[timesteps]
        else:
            timesteps_model = timesteps
        prediction = model(**{**model_inputs, "p": 0}, timesteps=timesteps_model)["x"]

        if guidance_scale > 0:
            prediction_uncond = model(**{**model_inputs, "p": 1}, timesteps=timesteps_model)["x"]
            prediction = prediction_uncond + guidance_scale * (prediction - prediction_uncond)

        step_output = self.sampler.step(
            model_prediction=prediction,
            timesteps=timesteps,
            xt=model_inputs["x"],
            clamp_x=clamp_x,
            **sampler_args,
        )

        return step_output

    ### TODO: Need to add compute loss for different parameterization + variance learned
    def compute_loss(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        timesteps: Tensor,
        noise: Tensor | None = None,
        extra_losses: list[LossFunction] = [],
        extra_args: dict[str, Any] = {},
    ) -> dict[str, Tensor]:
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
            dict[str, Tensor]: A dictionary containing the loss value and any additional losses
        Note:
            When using a different number of sampling steps than training steps,
            this method maps the timestep indices through the timestep_map to ensure
            the model sees the correct noise level for each step.
        """
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        if self.timestep_map:
            map_tensor = torch.tensor(self.timestep_map, device=timesteps.device, dtype=timesteps.dtype)
            timesteps = map_tensor[timesteps]
        prediction = model(**model_inputs, timesteps=timesteps)["x"]
        loss = nn.functional.mse_loss(prediction, noise, reduction="mean")
        loss_dict = {"loss": loss}
        for extra_loss in extra_losses:
            e_loss = cast(Tensor, extra_loss(**extra_args))
            loss_dict[extra_loss.name] = e_loss
        return loss_dict

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
            + (torch.ones_like(noise) - extract_into_tensor(self.alphas_bar, timesteps, noise.shape)).sqrt() * noise
        )
        return x_t, noise

    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = False,
        guidance_scale: float = 0,
        sampler_args: dict[str, Any] = {},
        return_intermediates: bool = False,
    ) -> SamplingOutput:
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
            sampler_args (dict[str, Any], optional): Additional arguments to pass to the sampler's step method.
                Defaults to an empty dictionary.
            return_intermediates (bool, optional): Whether to return intermediate states during denoising.
                Defaults to False.
        Returns:
            SamplingOutput: A dictionary containing:
                - x (Tensor): The final generated sample tensor.
                - xt (Tensor, optional): If `return_intermediates` is True, a tensor of shape
                  (batch_size, steps+1, ...) containing the sample at each timestep.
                - estimated_x0 (Tensor, optional): If `return_intermediates` is True, a tensor
                  of shape (batch_size, steps, ...) containing the estimated original data at
                  each timestep.
                - xt_mean (Tensor, optional): If `return_intermediates` is True and the sampler
                  provides mean estimates, a tensor of shape (batch_size, steps, ...) containing
                  the mean at each timestep.
                - xt_std (Tensor, optional): If `return_intermediates` is True and the sampler
                  provides std estimates, a tensor of shape (batch_size, steps, ...) containing
                  the standard deviation at each timestep.
                - logprob (Tensor, optional): If `return_intermediates` is True and the sampler
                  provides log probabilities, a tensor of shape (batch_size, steps, ...) containing
                  the log probabilities at each timestep.
        Notes:
            The model_inputs dictionary is modified in place with the current sample state.
        """
        if "x" not in model_inputs:
            model_inputs["x"] = torch.randn(
                data_shape, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
            )

        all_x0: list[Tensor] = []
        all_xt: list[Tensor] = [model_inputs["x"]]
        all_xt_mean: list[Tensor] = []
        all_xt_std: list[Tensor] = []
        all_logprobs: list[Tensor] = []

        for t in tqdm(
            list(range(self.steps))[::-1],
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            step_output = self.one_step_denoise(
                model=model,
                model_inputs=model_inputs,
                t=t,
                clamp_x=clamp_x,
                guidance_scale=guidance_scale,
            )

            model_inputs["x"] = step_output["x_prev"]
            if return_intermediates:
                all_xt.append(step_output["x_prev"])
                all_x0.append(step_output["estimated_x0"])
                if "x_prev_mean" in step_output:
                    all_xt_mean.append(step_output["x_prev_mean"])
                if "x_prev_std" in step_output:
                    all_xt_std.append(step_output["x_prev_std"])
                if "logprob" in step_output:
                    all_logprobs.append(step_output["logprob"])

        out: SamplingOutput = {"x": model_inputs["x"]}
        if return_intermediates:
            out["xt"] = torch.stack(all_xt, dim=1)
            out["estimated_x0"] = torch.stack(all_x0, dim=1)
            if len(all_xt_mean) > 0:
                out["xt_mean"] = torch.stack(all_xt_mean, dim=1)
            if len(all_xt_std) > 0:
                out["xt_std"] = torch.stack(all_xt_std, dim=1)
            if len(all_logprobs) > 0:
                out["logprob"] = torch.stack(all_logprobs, dim=1)

        return out
