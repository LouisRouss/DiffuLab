from abc import ABC

import torch.nn as nn


class LossFunction(ABC, nn.Module):  # to be completed
    def __init__(self) -> None:
        super().__init__()  # type: ignore
