from typing import Any

from accelerate import Accelerator  # type: ignore


class Trainer:
    def __init__(
        self,
        n_epoch: int,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_step: int = 1,
        precision_type: str = "no",
        project_name: str = "my_project",
        run_config: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision=precision_type,
            gradient_accumulation_steps=gradient_accumulation_step,
            log_with="wandb",
        )
        self.accelerator.init_trackers(project_name=project_name, config=run_config, init_kwargs=init_kwargs)  # type: ignore
