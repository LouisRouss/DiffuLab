import torch
from torch import Tensor
from tqdm import tqdm

from diffulab.diffuse.diffusion import Diffusion
from diffulab.networks.denoisers.common import Denoiser, ModelInput


# replace function at, bt etc ... By actually precomputing the values and storing them for every timestep
# more efficient
class Flow(Diffusion):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler", schedule: str = "linear"):
        super().__init__(n_steps=n_steps, sampling_method=sampling_method, schedule=schedule)

    def set_steps(self, n_steps: int, schedule: str = "linear") -> None:
        if schedule == "linear":
            self.schedule = schedule
            self.timesteps: list[float] = torch.linspace(1, 0, n_steps + 1).tolist()  # type: ignore
            self.steps = n_steps
        else:
            raise NotImplementedError("Only linear schedule is supported for the moment")

    def at(self, timesteps: Tensor) -> Tensor:
        return 1 - timesteps

    def bt(self, timesteps: Tensor) -> Tensor:
        return timesteps

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)

    def get_v(self, model: Denoiser, model_inputs: ModelInput, t_curr: float) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t_curr, device=device, dtype=dtype)
        prediction = model(**model_inputs, timesteps=timesteps)
        return prediction

    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        t_prev: float,
        t_curr: float,
        guidance_scale: float,
        clamp_x: bool,
    ) -> Tensor:
        v = self.get_v(model, ModelInput({**model_inputs, "p": 0}), t_curr)
        if guidance_scale > 0:
            v_dropped = self.get_v(model, {**model_inputs, "p": 1}, t_curr)
            v = v + guidance_scale * (v - v_dropped)
        if self.sampling_method == "euler":
            x_t_minus_one: Tensor = model_inputs["x"] - v * (t_curr - t_prev)
        else:  # different methods to be implemented maybe in the generic class instead
            raise NotImplementedError
        return x_t_minus_one

    def compute_loss(
        self, model: Denoiser, model_inputs: ModelInput, timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction: torch.Tensor = model(**model_inputs, timesteps=timesteps)
        losses = (prediction - (model_inputs["x"] - noise)) ** 2
        losses = losses.reshape(losses.shape[0], -1).mean(dim=-1)
        loss = losses.mean()
        return loss

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
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
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = False,
        guidance_scale: float = 0,
        n_steps: int | None = None,
    ) -> Tensor:
        if n_steps:
            original_steps = self.steps
            self.set_steps(n_steps, schedule=self.schedule)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = torch.randn(data_shape, device=device, dtype=dtype)
        for t_curr, t_prev in tqdm(
            zip(self.timesteps[:-1], self.timesteps[1:]),
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            model_inputs["x"] = x
            x = self.one_step_denoise(
                model,
                model_inputs,
                t_curr=t_curr,
                t_prev=t_prev,
                guidance_scale=guidance_scale,
                clamp_x=clamp_x,
            )
        if n_steps:
            self.set_steps(original_steps, schedule=self.schedule)  # reset the steps # type: ignore
        return x
