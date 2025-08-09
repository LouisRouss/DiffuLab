from typing import Any

import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint  # type: ignore


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return (val) if exists(val) else (d)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
