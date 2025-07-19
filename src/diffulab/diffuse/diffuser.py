from typing import Any

from torch import Tensor

from diffulab.diffuse.modelizations.diffusion import Diffusion
from diffulab.diffuse.modelizations.flow import Flow
from diffulab.diffuse.modelizations.gaussian_diffusion import GaussianDiffusion
from diffulab.networks.denoisers.common import Denoiser, ModelInput
from diffulab.networks.vision_towers.common import VisionTower
from diffulab.training.losses import LossFunction


class Diffuser:
    """
    A wrapper class for diffusion models, supporting different types of diffusion processes.
    This class provides a unified interface to work with different diffusion model architectures
    by encapsulating the specific diffusion implementation details.
    Args:
        - denoiser (Denoiser): The neural network model used for denoising.
        - sampling_method (str, optional): Method used for sampling.
        - model_type (str, optional): Type of diffusion model to use. Defaults to "rectified_flow".
            Available options:
                - "rectified_flow": Flow-based diffusion model.
                - "gaussian_diffusion": Gaussian diffusion model.
        - n_steps (int, optional): Number of diffusion steps. Defaults to 1000.
        - extra_args (dict[str, Any], optional): Additional arguments to pass to the diffusion model. Defaults to {}.

    Attributes:
        - model_type (str): Type of diffusion model used.
        - denoiser (Denoiser): The neural network model used for denoising.
        - n_steps (int): Number of diffusion steps.
        - diffusion (Diffusion): The diffusion model implementation.

    Methods:
        - __init__: Initialize the diffuser with specified parameters.
        - eval: Set the denoiser model to evaluation mode.
        - train: Set the denoiser model to training mode.
        - draw_timesteps: Draw a batch of timesteps for diffusion.
        - compute_loss: Calculate the loss for the diffusion model.
        - set_steps: Update the number of diffusion steps and related parameters.
        - generate: Generate new samples using the diffusion model.
    """

    model_registry: dict[str, type[Diffusion]] = {"rectified_flow": Flow, "gaussian_diffusion": GaussianDiffusion}

    def __init__(
        self,
        denoiser: Denoiser,
        sampling_method: str,
        model_type: str = "rectified_flow",
        n_steps: int = 1000,
        vision_tower: VisionTower | None = None,
        extra_args: dict[str, Any] = {},
        extra_losses: list[LossFunction] = [],
    ):
        self.model_type = model_type
        self.denoiser = denoiser
        self.n_steps = n_steps
        self.vision_tower = vision_tower
        self.extra_losses = extra_losses
        if self.vision_tower:
            self.latent_scale = self.vision_tower.latent_scale

        if self.model_type in self.model_registry:
            self.diffusion = self.model_registry[self.model_type](
                n_steps=n_steps,
                sampling_method=sampling_method,
                latent_diffusion=self.vision_tower is not None,
                **extra_args,
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not implemented")

    def eval(self) -> None:
        """
        Set the denoiser model to evaluation mode.
        """
        self.denoiser.eval()

    def train(self) -> None:
        """
        Set the denoiser model to training mode.
        """
        self.denoiser.train()

    def draw_timesteps(self, batch_size: int) -> Tensor:
        """
        Draw a batch of timesteps for diffusion.
        Args:
            - batch_size (int): The number of timesteps to draw.
        Returns:
            Tensor: A 1-D tensor of timesteps.
        """
        return self.diffusion.draw_timesteps(batch_size=batch_size)

    def compute_loss(
        self,
        model_inputs: ModelInput,
        timesteps: Tensor,
        noise: Tensor | None = None,
        extra_args: dict[str, Any] = {},
    ) -> dict[str, Tensor]:
        """
        Compute the loss for the diffusion model using the denoiser and diffusion process.
        This method serves as a bridge between the Diffuser class and the underlying
        diffusion implementation by forwarding the loss computation to the diffusion model.
        Args:
            - model_inputs (ModelInput): A dictionary containing the model inputs,
              including the data tensor keyed as 'x' and any conditional information.
            - timesteps (Tensor): A tensor of timesteps for the batch.
            - noise (Tensor | None, optional): Pre-defined noise to add to the input.
              If None, random noise will be generated. Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary containing the loss value and any additional losses
        """
        return self.diffusion.compute_loss(self.denoiser, model_inputs, timesteps, noise, self.extra_losses, extra_args)

    def set_steps(self, n_steps: int, **extra_args: dict[str, Any]) -> None:
        """
        Update the number of diffusion steps and related parameters.
        This method allows changing the number of steps used in the diffusion process
        after the Diffuser has been initialized. It delegates to the underlying diffusion
        model's set_steps method.
        Args:
            n_steps (int): The new number of diffusion steps to use.
            **extra_args (dict[str, Any]): Additional arguments to pass to the diffusion
                model's set_steps method. These may include parameters like 'schedule'
                or 'section_counts' depending on the diffusion model implementation.
        Example:
            ```
            diffuser = Diffuser(denoiser, sampling_method="ddpm", n_steps=1000)
            # Later, change to use fewer steps for faster sampling
            diffuser.set_steps(100, schedule="ddim")
            ```
        """
        self.diffusion.set_steps(n_steps, **extra_args)  # type: ignore

    def generate(
        self,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 0,
        **kwargs: dict[str, Any],
    ) -> Tensor:
        """
        Generates a new sample using the diffusion model.
        This method delegates to the underlying diffusion model's denoise method to
        generate a sample from the diffusion process. It can handle conditional generation
        with guidance when appropriate settings are provided.
        Args:
            - data_shape (tuple[int, ...]): Shape of the data to generate, typically (batch_size, channels, height, width).
                If a vision tower is used, this should be the shape of the latent space.
            - model_inputs (ModelInput): A dictionary containing inputs for the model, such as initial noise,
                conditional information, or labels. If 'x' is not provided, random noise will be generated.
            - use_tqdm (bool, optional): Whether to display a progress bar during generation. Defaults to True.
            - clamp_x (bool, optional): Whether to clamp the generated values to [-1, 1] range. Defaults to True.
            - guidance_scale (float, optional): Scale for classifier or classifier-free guidance.
                Values greater than 0 enable guidance. Defaults to 0.
            **kwargs (dict[str, Any]): Additional arguments to pass to the diffusion model's denoise method.
                These may include parameters like 'classifier', 'classifier_free', etc.
        Returns:
            Tensor: The generated data tensor.
        Example:
            ```python
            # Generate an unconditional sample
            sample = diffuser.generate(
                data_shape=(1, 3, 256, 256),
                model_inputs={"x": initial_noise}
            # Generate a conditional sample with classifier-free guidance
            sample = diffuser.generate(
                data_shape=(1, 3, 256, 256),
                model_inputs={"context": text_embedding, "x": initial_noise},
                guidance_scale=7.5
            ```
        """
        if self.vision_tower:
            z = self.diffusion.denoise(
                self.denoiser,
                data_shape,
                model_inputs,
                use_tqdm=use_tqdm,
                clamp_x=clamp_x,
                guidance_scale=guidance_scale,
                **kwargs,
            )
            z = z / self.latent_scale
            return self.vision_tower.decode(z)

        return self.diffusion.denoise(
            self.denoiser,
            data_shape,
            model_inputs,
            use_tqdm=use_tqdm,
            clamp_x=clamp_x,
            guidance_scale=guidance_scale,
            **kwargs,
        )
