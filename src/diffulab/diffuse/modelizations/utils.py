from typing import TypedDict

from jaxtyping import Float
from torch import Tensor


def space_timesteps(num_timesteps: int, section_counts: str | int, ddim: bool = False) -> set[int]:
    """
    Schedules the timesteps to be sampled for generation, by dividing the total number
    of timesteps into sections with specific counts.

    Args:
        num_timesteps (int): The total number of timesteps in the diffusion process used
        during training.
        section_counts (str | int): If a string, a comma-separated list of integers
            specifying how many steps to take in each section. If an integer, uniform
            sectioning is applied.
        ddim (bool, optional): If True, uses uniformly spaced steps for DDIM sampling.
            Defaults to False.

    Returns:
        set[int]: A set of timestep indices to be sampled.

    Raises:
        ValueError: If the requested section counts cannot be achieved with the given
            number of timesteps.

    Example:
        >>> space_timesteps(1000, 10, ddim=True)
        {0, 100, 200, 300, 400, 500, 600, 700, 800, 900}
    """
    if ddim:
        assert isinstance(section_counts, int)
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == section_counts:
                return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")

    if isinstance(section_counts, str):
        section_counts_list = [int(x) for x in section_counts.split(",")]
    else:
        section_counts_list = [section_counts]

    size_per = num_timesteps // len(section_counts_list)
    extra = num_timesteps % len(section_counts_list)
    start_idx = 0
    all_steps: list[int] = []
    for i, section_count in enumerate(section_counts_list):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps: list[int] = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class GRPOSamplingOutput(TypedDict):
    x: Float[Tensor, "B C H W"]
    x_0_original: Float[Tensor, "B C H W"]
    x_t: Float[Tensor, "B T+1 C H W"]  # T = number of sampled timesteps
    log_probs: Float[Tensor, "B T"]
    x_t_minus_one_mean: Float[Tensor, "B T C H W"]
