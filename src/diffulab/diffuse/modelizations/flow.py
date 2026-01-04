import random
from typing import Any, Literal, cast

import torch
from torch import Tensor
from tqdm import tqdm

from diffulab.diffuse.modelizations.diffusion import Diffusion
from diffulab.diffuse.samplers import StepResult
from diffulab.diffuse.samplers.flow import Euler, EulerMaruyama
from diffulab.diffuse.utils import SamplingOutput
from diffulab.networks.denoisers.common import Denoiser, ModelInput, ModelOutput
from diffulab.training.losses.common import LossFunction


class Flow(Diffusion):
    """
    Flow-based diffusion model.
    This class implements a diffusion model based on continuous-time flow matching.
    The model maps a noise distribution to a data distribution through a continuous
    transformation defined by a neural network.
    Args:
        n_steps (int, optional): Number of steps for the diffusion process. Defaults to 50.
        sampling_method (str, optional): Method used for sampling. Defaults to "euler".
        schedule (str, optional): Schedule for time discretization. Currently only "linear"
            is supported. Defaults to "linear".
        latent_diffusion (bool, optional): Whether the diffusion operates in a latent space.
            Defaults to False.
        logits_normal (bool, optional): Whether to use logit-normal distribution for timestep
            sampling during training. When True, timesteps are drawn from a sigmoid-transformed
            normal distribution. Defaults to False.
        sampler_parameters (dict[str, Any], optional): Additional parameters for the sampler.
            Defaults to an empty dictionary.
    Methods:
        set_steps(n_steps, schedule): Sets the number of timesteps and their spacing.
        at(timesteps): Computes the "a(t)" coefficient for noise mixing.
        bt(timesteps): Computes the "b(t)" coefficient for noise mixing.
        draw_timesteps(batch_size): Samples random timesteps for training.
        get_v(model, model_inputs, t_curr): Computes the velocity field at current time.
        one_step_denoise(model, model_inputs, t_prev, t_curr, guidance_scale, sampler_args): Performs one step
            of the reverse diffusion process.
        compute_loss(model, model_inputs, timesteps, noise, extra_losses, extra_args): Computes the loss for training.
        compute_loss_grpo(model, model_inputs, sampling, advantages, kl_beta, eps, timestep_fraction, guidance_scale):
            Computes the GRPO loss for reinforcement learning.
        add_noise(x, timesteps, noise): Adds noise to the input according to the timesteps.
        denoise(model, data_shape, model_inputs, use_tqdm, clamp_x, guidance_scale, sampler_args, return_intermediates):
            Generates samples by running the reverse diffusion process.
    References:
        Lipman, Y., et al. (2022). "Flow Matching for Generative Modeling."
        https://arxiv.org/abs/2210.02747

    """

    sampler_registry = {
        "euler": Euler,
        "euler_maruyama": EulerMaruyama,
    }

    def __init__(
        self,
        n_steps: int = 50,
        sampling_method: Literal["euler", "euler_maruyama"] = "euler",
        schedule: Literal["linear"] = "linear",
        latent_diffusion: bool = False,
        logits_normal: bool = False,
        shift: float | None = None,
        sampler_parameters: dict[str, Any] = {},
        prediction_type: Literal["v", "x"] = "v",
    ) -> None:
        assert prediction_type in ["v", "x"], (
            "prediction_type must be either 'v' or 'x', noise prediction not supported yet for flow models"
        )
        super().__init__(
            n_steps=n_steps,
            sampling_method=sampling_method,
            schedule=schedule,
            latent_diffusion=latent_diffusion,
            sampler_parameters=sampler_parameters,
        )
        self.logits_normal = logits_normal
        self.shift = shift
        self.x_prediction = prediction_type == "x"

    @staticmethod
    def _shift_timestep(t: Tensor | float, alpha: float) -> Tensor | float:
        """
        Applies the time-shifting function s(α, t) = αt / (1 + (α - 1)t).

        This shifts the timestep distribution to concentrate more samples
        at certain noise levels during training.

        Args:
            t: The original timestep(s) in [0, 1].
            alpha: The shift parameter. alpha > 1 shifts toward higher noise levels.

        Returns:
            The shifted timestep(s).
        """
        return alpha * t / (1 + (alpha - 1) * t)

    def set_steps(self, n_steps: int, schedule: str = "linear", shift: float | None = None) -> None:
        """
        Update the number of steps and schedule for the flow-based diffusion model.
        This method configures the timesteps sequence based on the specified number of steps
        and schedule type. Currently, only a linear schedule is supported, which creates
        an evenly spaced sequence of timesteps from 1 to 0.
        Args:
            n_steps (int): The number of diffusion steps to use.
            schedule (str, optional): The scheduling algorithm to use for timestep generation.
                Currently only "linear" is implemented. Defaults to "linear".
            shift (float | None, optional): If provided, applies time-shifting to the timesteps
        Raises:
            NotImplementedError: If a schedule other than "linear" is specified.
        Note:
            The timesteps are set in descending order from 1 to 0, with n_steps+1 values
            (including both endpoints).
        Example:
            ```python
            flow = Flow(n_steps=100)
            # Later, change to use fewer steps
            flow.set_steps(50)  # Will create a new linear schedule with 50 steps
            ```
        """
        self.shift = shift
        if schedule == "linear":
            self.schedule = schedule
            timesteps: list[float] = torch.linspace(1, 0, n_steps + 1).tolist()  # type: ignore
            if self.shift is not None:
                timesteps = [self._shift_timestep(t, self.shift) for t in timesteps]  # type: ignore
            self.timesteps = timesteps
            self.steps = n_steps
        else:
            raise NotImplementedError("Only linear schedule is supported for the moment")

        self.sampler.set_steps(self.timesteps)

    def at(self, timesteps: Tensor) -> Tensor:
        """
        Computes the coefficient a(t) for the flow model equation xt = a(t) * x0 + b(t) * eps.

        In the flow-based diffusion model, a(t) = 1 - t, which represents the weight of the
        original signal (x0) at timestep t in the forward process.

        Args:
            timesteps (Tensor): A tensor containing timestep values in the range [0, 1].

        Returns:
            Tensor: The a(t) coefficient tensor of the same shape as the input,
               evaluated as (1 - timesteps).
        """

        return torch.ones_like(timesteps) - timesteps

    def bt(self, timesteps: Tensor) -> Tensor:
        """
        Computes the coefficient b(t) for the flow model equation xt = a(t) * x0 + b(t) * eps.

        In the flow-based diffusion model, b(t) = t, which represents the weight of the
        noise (eps) at timestep t in the forward process.
        Args:
            timesteps (Tensor): Tensor containing timestep values, typically in the range [0, 1].
        Returns:
            Tensor: The calculated beta coefficient values corresponding to each input timestep.
        """

        return timesteps

    def draw_timesteps(self, batch_size: int) -> Tensor:
        """
        Generates timesteps for training.
        This method samples timesteps for the diffusion process according to the
        configured distribution. It can use either a uniform distribution (default)
        or a logit-normal distribution based on the `logits_normal` flag.
        Args:
            batch_size (int): The number of timesteps to generate, typically matching
                the batch size of the data.
        Returns:
            Tensor: A tensor of shape (batch_size,) containing timestep values in the range [0, 1].
                If `logits_normal` is False, timesteps are drawn from a uniform distribution.
                If `logits_normal` is True, timesteps are drawn from a sigmoid-transformed
                normal distribution which concentrates samples around 0.5.
        """

        if self.logits_normal:
            nt = torch.randn((batch_size), dtype=torch.float32)
            t: Tensor = torch.sigmoid(nt)
        else:
            t = torch.rand((batch_size), dtype=torch.float32)

        # Apply time-shifting if shift_value is set
        if self.shift is not None:
            t = self._shift_timestep(t, self.shift)  # type: ignore

        if self.x_prediction:
            t = t.clamp(min=0.05)

        return t

    def get_v(self, model: Denoiser, model_inputs: ModelInput, t_curr: float) -> Tensor:
        """
        Computes the velocity field (v) at the current timestep.
        This method evaluates the denoising model to obtain the velocity vector field
        at the specified timestep, which is used to guide the reverse diffusion process.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including
                the current state tensor keyed as 'x' and any conditional information.
            t_curr (float): The current timestep value, typically in the range [0, 1].
        Returns:
            Tensor: The predicted velocity field from the model at the current timestep,
                which is used to update the state in the reverse diffusion process.
        """

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t_curr, device=device, dtype=dtype)
        prediction = model(**model_inputs, timesteps=timesteps)["x"]
        if self.x_prediction:
            return (model_inputs["x"] - prediction) / max(t_curr, 0.05)

        return prediction

    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        t_prev: float,
        t_curr: float,
        guidance_scale: float,
        sampler_args: dict[str, Any] = {},
    ) -> StepResult:
        """
        Performs one step of denoising in the reverse diffusion process.
        This method implements a single step of the reverse flow-based diffusion process,
        moving from the current timestep t_curr toward the previous timestep t_prev.
        It computes the velocity field at the current timestep and uses it to update
        the state according to the specified sampling method.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including
                the current state tensor keyed as 'x' and any conditional information.
            t_prev (float): The previous (target) timestep value to move toward.
            t_curr (float): The current timestep value.
            guidance_scale (float): Scale factor for classifier-free guidance. If greater than 0,
                the method computes both conditional and unconditional predictions and
                interpolates between them, with higher values emphasizing the conditional prediction.
            sampler_args (dict[str, Any], optional): Additional arguments to pass to the sampler's
                step method. Defaults to an empty dictionary.
        Returns:
            Tensor: The updated state tensor after one step of the reverse diffusion process.
        Note:
            Currently, only the "euler" sampling method is implemented, which approximates
            the ODE solution with a simple Euler step. Additional sampling methods may be
            implemented in the future for improved accuracy or efficiency.
        """
        v = self.get_v(model, ModelInput({**model_inputs, "p": 0}), t_curr)
        if guidance_scale > 0:
            v_dropped = self.get_v(model, {**model_inputs, "p": 1}, t_curr)
            v = v_dropped + guidance_scale * (v - v_dropped)
        return self.sampler.step(model_inputs["x"], v, t_curr, t_prev, **sampler_args)

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
        Computes the loss for training the flow-based diffusion model.
        This method implements the loss function for training the flow-based diffusion model,
        which measures the difference between the predicted velocity field and the true velocity
        field (noise - x_t). The loss is computed as the mean squared error between these values.
        Additional losses can be included by passing them in the `extra_losses` list
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including
                the clean data tensor keyed as 'x' and any conditional information.
            timesteps (Tensor): A tensor of shape (batch_size,) containing timestep values
                in the range [0, 1].
            noise (Tensor | None, optional): Pre-generated noise to add to the inputs.
                If None, random noise will be generated. Defaults to None.
            extra_losses (list[LossFunction], optional): Additional loss functions to compute
                alongside the main loss. Defaults to an empty list.
            extra_args (dict[str, Any], optional): Additional arguments for the loss computation.
                Defaults to an empty dictionary.
        Returns:
            dict[str, Tensor]: A dictionary containing the loss value and any additional losses
        Note:
            The function first adds noise to the input data according to the specified timesteps,
            then computes the model's prediction. The loss is calculated as the mean squared error
            between the true velocity (noise - x_t) and the predicted velocity, averaged over
            all dimensions and then over the batch.
        """
        x_0 = model_inputs["x"].clone()
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction: ModelOutput = model(**model_inputs, timesteps=timesteps)
        if self.x_prediction:
            prediction["x"] = (model_inputs["x"] - prediction["x"]) / timesteps.view(
                -1, *([1] * (model_inputs["x"].dim() - 1))
            )  # get v from x prediction

        # Compute flow matching loss
        losses = ((noise - x_0) - prediction["x"]) ** 2
        losses = losses.reshape(losses.shape[0], -1).mean(dim=-1)
        loss = losses.mean()
        loss_dict = {"loss": loss}

        # Compute extra losses if any
        for extra_loss in extra_losses:
            e_loss = cast(Tensor, extra_loss(**extra_args))
            loss_dict[extra_loss.name] = e_loss
        return loss_dict

    def compute_loss_grpo(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        sampling: SamplingOutput,
        advantages: Tensor,
        kl_beta: float = 0,
        eps: float = 1e-4,
        timestep_fraction: float = 0.6,
        guidance_scale: float = 4,
    ) -> dict[str, Tensor]:
        """
        Computes the Preference Reward-based GRPO loss for preference alignment with flow-based diffusion models.
        reference : https://arxiv.org/abs/2508.20751. The sampler needs to be set to Euler-Maruyama for GRPO.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing the model inputs, including the current state tensor
                keyed as 'x' and any conditional information.
            sampling (SamplingOutput): The output from the sampling process, containing intermediate samples
                transitions distribution related parameters
            advantages (Tensor): A tensor of shape (batch_size,) containing the advantage values for each sample in the batch.
            kl_beta (float, optional): Coefficient for the KL divergence term in the loss. Defaults to 0.
            eps (float, optional): Clipping parameter for the policy loss. Defaults to 1e-4.
            timestep_fraction (float, optional): Fraction of timesteps to consider for loss computation. Defaults to 0.6.
            guidance_scale (float, optional): Scale for classifier-free guidance during denoising. Defaults to 4.
        Returns:
            dict[str, Tensor]: A dictionary containing the computed GRPO loss.
        """
        assert isinstance(self.sampler, EulerMaruyama), "GRPO only works with the Euler-Maruyama sampler"
        assert "xt" in sampling, "sampling output should contain all intermediate samples"
        assert "logprob" in sampling, "sampling output should contain all logprobs"
        assert "xt_mean" in sampling, "sampling output should contain all intermediate means"

        indices = random.sample(range(0, self.steps), k=round(self.steps * timestep_fraction))
        losses: list[Tensor] = []
        for idx in indices:
            model_inputs["x"] = sampling["xt"][:, idx]
            step_result = self.one_step_denoise(
                model=model,
                model_inputs=cast(ModelInput, model_inputs),  # We added "x" to model_inputs
                t_curr=self.timesteps[idx],
                t_prev=self.timesteps[idx + 1],
                guidance_scale=guidance_scale,
                sampler_args={"x_prev": sampling["xt"][:, idx + 1]},
            )

            assert "logprob" in step_result, "logprob should be returned by the sampler"
            assert "x_prev_mean" in step_result, "x_prev_mean should be returned by the sampler"
            assert "x_prev_std" in step_result, "std should be returned by the GRPO sampling output"

            prob_ratios = torch.exp(step_result["logprob"] - sampling["logprob"][:, idx])
            unclipped_objective = advantages * prob_ratios
            clipped_objective = advantages * torch.clamp(prob_ratios, 1 - eps, 1 + eps)
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

            diff = (step_result["x_prev_mean"] - sampling["xt_mean"][:, idx]) ** 2
            kl_loss = diff.mean(dim=tuple(range(1, diff.dim()))) / (2 * step_result["x_prev_std"] ** 2)
            kl_loss = kl_loss.mean()

            step_loss = policy_loss + kl_beta * kl_loss
            losses.append(step_loss)

        loss = torch.stack(losses).mean()
        return {"loss": loss}

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Adds noise to the input data according to the flow diffusion process.
        This method applies the forward noising process according to the equation:
        z_t = a(t) * x + b(t) * noise
        Where a(t) = 1-t and b(t) = t are coefficients that control how much of the
        original signal and noise are mixed at timestep t.
        Args:
            x (Tensor): The clean input data tensor to be noised.
            timesteps (Tensor): A tensor of shape (batch_size,) containing timestep values
                in the range [0, 1] for each example in the batch.
            noise (Tensor | None, optional): Pre-defined noise to add to the input.
                If None, random noise will be generated. Defaults to None.
        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - z_t (Tensor): The noised input at the specified timesteps.
                - noise (Tensor): The noise used in the forward process.
        """

        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        at = self.at(timesteps).view(-1, *([1] * (x.dim() - 1))).to(x.device)
        bt = self.bt(timesteps).view(-1, *([1] * (x.dim() - 1))).to(x.device)
        z_t = at * x + bt * noise
        return z_t, noise

    @torch.inference_mode()
    def denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        data_shape: tuple[int, ...] | None = None,
        use_tqdm: bool = True,
        clamp_x: bool = False,
        guidance_scale: float = 0,
        sampler_args: dict[str, Any] = {},
        return_intermediates: bool = False,
    ) -> SamplingOutput:
        """
        Generates samples by running the reverse diffusion process.
        This method implements the sample generation process for flow-based diffusion models
        by iteratively denoising from pure noise to a clean sample. The process starts with
        random noise and progressively transforms it by following the reverse flow defined
        by the velocity field predicted by the neural network.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing inputs for the model, such as
                conditional information or labels. The function will update this dictionary
                with the current sample state during generation.
            data_shape (tuple[int, ...] | None): Shape of the data to generate, typically
                (batch_size, channels, height, width) for images. Needed if 'x' is not in model_inputs.
                Defaults to None.
            use_tqdm (bool, optional): Whether to display a progress bar during generation.
                Defaults to True.
            clamp_x (bool, optional): Whether to clamp the generated values to [-1, 1] range.
                Defaults to False.
            guidance_scale (float, optional): Scale for classifier-free guidance. Values greater
                than 0 enable guidance, with higher values giving stronger adherence to the
                conditioning. Defaults to 0.
            sampler_args (dict[str, Any], optional): Additional arguments to pass to the sampler's
                step method. Defaults to an empty dictionary.
            return_intermediates (bool, optional): Whether to return intermediate results at each step.
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
        Note:
            The function works by starting with random noise and iteratively updating it
            using the velocity field predicted by the model at each timestep. The process
            follows the timestep sequence defined during initialization, moving from t=1
            (pure noise) to t=0 (clean data).
            The dictionary model_inputs is updated in place with the current sample state
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if "x" not in model_inputs:
            assert data_shape is not None, "'data_shape' must be provided if 'x' is not in model_inputs"
            model_inputs["x"] = torch.randn(data_shape, device=device, dtype=dtype)

        all_x0: list[Tensor] = []
        all_xt: list[Tensor] = [model_inputs["x"]]
        all_xt_mean: list[Tensor] = []
        all_xt_std: list[Tensor] = []
        all_logprobs: list[Tensor] = []

        for t_curr, t_prev in tqdm(
            zip(self.timesteps[:-1], self.timesteps[1:]),
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            step_output = self.one_step_denoise(
                model,
                model_inputs,
                t_curr=t_curr,
                t_prev=t_prev,
                guidance_scale=guidance_scale,
                sampler_args=sampler_args,
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

        if clamp_x:
            model_inputs["x"] = model_inputs["x"].clamp(-1, 1)

        out: SamplingOutput = {"x": model_inputs["x"]}
        if return_intermediates:
            out["xt"] = torch.stack(all_xt, dim=1)
            out["estimated_x0"] = torch.stack(all_x0, dim=1)
            if len(all_xt_mean) > 0:
                out["xt_mean"] = torch.stack(all_xt_mean, dim=1)
            if len(all_xt_std) > 0:
                out["xt_std"] = torch.stack(all_xt_std, dim=0)
            if len(all_logprobs) > 0:
                out["logprob"] = torch.stack(all_logprobs, dim=1)

        return out
