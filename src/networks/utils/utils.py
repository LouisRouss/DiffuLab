from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return (val) if exists(val) else (d)


def checkpoint(func: Callable[..., Tensor], inputs: Any, params: Any, flag: bool) -> Any:
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)  # type: ignore
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[..., tuple[Tensor, ...]],
        length: int,
        *args: Tensor,
    ) -> tuple[Tensor, ...]:
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx: Any, *output_grads: Tensor):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
