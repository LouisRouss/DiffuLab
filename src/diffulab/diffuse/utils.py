from typing import NotRequired, Required, TypedDict

from torch import Tensor


def extract_into_tensor(arr: Tensor, timesteps: Tensor, broadcast_shape: tuple[int, ...]) -> Tensor:
    """
    Extract values from a 1-D tensor for a batch of indices.

    :param arr: the 1-D tensor.
    :param timesteps: a tensor of indices into the tensor to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class SamplingOutput(TypedDict, total=False):
    x: Required[Tensor]  # final sample
    estimated_x0: NotRequired[Tensor]  # estimated x0 at each step
    xt: NotRequired[Tensor]  # samples at each step
    xt_mean: NotRequired[Tensor]  # mean of samples at each step for SDE based methods
    xt_std: NotRequired[Tensor]  # std of samples at each step for SDE based methods
    logprob: NotRequired[Tensor]  # logprob at each step for flow based methods
