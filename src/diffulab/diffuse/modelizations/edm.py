from diffulab.diffuse.modelizations.diffusion import Diffusion


class EDM(Diffusion):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler", schedule: str = "linear"):
        super().__init__(n_steps=n_steps, sampling_method=sampling_method, schedule=schedule)
