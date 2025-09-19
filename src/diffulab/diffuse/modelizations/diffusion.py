from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from diffulab.diffuse.modelizations.utils import GRPOSamplingOutput
from diffulab.networks.denoisers.common import Denoiser, ModelInput, ModelInputGRPO
from diffulab.training.losses import LossFunction


class Diffusion(ABC):
    """
    Abstract base class for diffusion models.
    This class defines the interface for all diffusion model implementations, providing
    a common structure for forward and reverse diffusion processes. Subclasses must implement
    specific methods to define how noise is added and removed from data.
    Args:
        n_steps (int): Number of steps in the diffusion process.
        sampling_method (str, optional): Method used for the reverse process (sampling).
            Defaults to "euler".
        schedule (str, optional): Schedule for time discretization in the diffusion process.
            Defaults to "linear".
    Attributes:
        timesteps (list[float]): List of timestep values for the diffusion process.
        steps (int): Number of steps in the diffusion process.
        sampling_method (str): Method used for sampling/reverse process.
        schedule (str): Schedule type used for timestep spacing.
    Methods:
        set_steps: Configure the timestep sequence for the diffusion process.
        one_step_denoise: Perform a single step of the reverse diffusion process.
        compute_loss: Calculate the loss for training the diffusion model.
        add_noise: Add noise to input data according to the forward process.
        denoise: Generate samples by running the complete reverse diffusion process.
        draw_timesteps: Sample random timesteps for training.
    """

    def __init__(
        self,
        n_steps: int,
        sampling_method: str = "euler",
        schedule: str = "linear",
        latent_diffusion: bool = False,
        **kwargs: Any,
    ):
        self.timesteps: list[float] = []
        self.steps: int = n_steps
        self.sampling_method = sampling_method
        self.schedule = schedule
        self.latent_diffusion = latent_diffusion
        self.set_steps(n_steps, schedule=schedule, **kwargs)

    @abstractmethod
    def set_steps(self, n_steps: int, schedule: str) -> None:
        """
        Configure the timestep sequence for the diffusion process.
        This method updates the number of steps and the schedule type used in the diffusion process.
        Args:
            n_steps (int): The number of steps to use in the diffusion process.
            schedule (str): The schedule type to use for timestep spacing (e.g., "linear", "cosine").
        Note:
            Implementations should update the internal timesteps and related parameters
            based on the new configuration.
        """
        pass

    @abstractmethod
    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        guidance_scale: float,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Perform a single step of the reverse diffusion process.
        This method implements a single step of denoising in the reverse diffusion process,
        moving from the current noisy state toward a less noisy state based on the model's
        predictions.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing model inputs, including the
                current noisy data tensor and any conditional information.
            guidance_scale (float): Scale factor for classifier or classifier-free guidance.
                If greater than 0, combines conditional and unconditional predictions.
            *args (Any): Additional positional arguments specific to the diffusion implementation.
            **kwargs (Any): Additional keyword arguments specific to the diffusion implementation.
        Returns:
            Tensor: The updated data tensor after one step of denoising.
        Note:
            The specific implementation details, such as how the model prediction is used to
            update the state, depend on the concrete diffusion model subclass.
        """
        pass

    def one_step_denoise_grpo(
        self,
        model: Denoiser,
        model_inputs: ModelInputGRPO,
        guidance_scale: float,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Tensor | float, ...]:
        raise NotImplementedError("This model does not implement GRPO.")

    @abstractmethod
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
        Calculate the loss for training the diffusion model.
        This abstract method computes the loss used to train the model based on the difference
        between the model's predictions and the true values at specific timesteps. Different
        diffusion models may implement this differently, but generally involve adding noise
        to clean inputs and comparing the model's denoising predictions.
        Args:
            model (Denoiser): The neural network model used for denoising.
            model_inputs (ModelInput): A dictionary containing model inputs, including the
                clean data tensor keyed as 'x' and any conditional information.
            timesteps (Tensor): A tensor of shape (batch_size,) containing the timesteps at
                which to compute the loss for each example in the batch.
            noise (Tensor | None, optional): Pre-defined noise to add to the inputs.
                If None, random noise will be generated. Defaults to None.
            extra_losses (list[LossFunction], optional): Additional loss functions to compute
                alongside the main loss. Defaults to an empty list.
            extra_args (dict[str, Any], optional): Additional arguments for the loss computation.
                Defaults to an empty dictionary.
        Returns:
            dict[str, Tensor]: A dictionary containing the loss value and any additional losses
        """
        pass

    def compute_loss_grpo(
        self,
        model: Denoiser,
        model_inputs: ModelInputGRPO,
        grpo_sampling_output: GRPOSamplingOutput,
        advantages: Tensor,
        kl_beta: float = 0,
        eps: float = 1e-4,
        timestep_fraction: float = 0.6,
        guidance_scale: float = 4,
        eta: float = 0.7,
    ) -> dict[str, Tensor]:
        raise NotImplementedError("This modelization does not implement GRPO.")

    @abstractmethod
    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Add noise to input data according to the forward process.
        This method implements the forward diffusion process by adding noise to clean data
        according to the specified timesteps. The specific noise application method depends on
        the concrete diffusion model subclass implementation.
        Args:
            x (Tensor): The clean input data tensor to be noised.
            timesteps (Tensor): A tensor of shape (batch_size,) containing timestep values for each example in the batch.
            noise (Tensor | None, optional): Pre-defined noise to add to the input.
                If None, random noise will be generated with the same shape as x. Defaults to None.
        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - The noised input data at the specified timesteps.
                - The noise that was used in the forward process.
        Note:
            Different diffusion models have different methods for adding noise based on their
            specific mathematical formulations (e.g., Gaussian diffusion, flow-based, etc.).
        """

        pass

    @abstractmethod
    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = False,
        guidance_scale: float = 0,
    ) -> Tensor:
        """
        Generate samples by running the reverse diffusion process.
        This method implements the complete reverse diffusion process to generate new samples,
        starting from random noise and iteratively denoising until reaching the final output.
        Args:
            model (Denoiser): The neural network model used for denoising.
            data_shape (tuple[int, ...]): Shape of data to generate (batch_size, channels, height, width).
            model_inputs (ModelInput): A dictionary containing model inputs, such as initial noise
                or conditional information. If 'x' is not provided, random noise will be generated.
            use_tqdm (bool, optional): Whether to show a progress bar during generation.
                Defaults to True.
            clamp_x (bool, optional): Whether to clamp output values to [-1, 1] range.
                Defaults to False.
            guidance_scale (float, optional): Scale factor for classifier or classifier-free guidance.
                Values greater than 0 enable guidance. Defaults to 0.
        Returns:
            Tensor: The generated data tensor after completing the reverse diffusion process.
        """
        pass

    def denoise_grpo(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInputGRPO,
        use_tqdm: bool = True,
        clamp_x: bool = False,
        guidance_scale: float = 0,
        *args: Any,
        **kwargs: Any,
    ) -> GRPOSamplingOutput:
        raise NotImplementedError("This modelization does not implement GRPO.")

    @abstractmethod
    def draw_timesteps(self, batch_size: int) -> Tensor:
        """
        Sample random timesteps for training the diffusion model.
        This method generates random timesteps that are used during training to sample
        points in the diffusion process. The specific sampling distribution depends on
        the concrete implementation in subclasses.
        Args:
            batch_size (int): Number of timesteps to generate, typically matching the
                batch size of the training data.
        Returns:
            Tensor: A tensor of shape (batch_size,) containing the sampled timestep
                values. The range and distribution of these values depends on the specific
                diffusion model implementation.
        Note:
            Different diffusion model implementations may use different distributions
            for sampling timesteps. For example:
            - Gaussian diffusion typically samples integers from [0, num_diffusion_steps-1]
            - Flow-based diffusion typically samples continuous values from [0, 1]
        """
        pass
