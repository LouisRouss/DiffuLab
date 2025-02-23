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
