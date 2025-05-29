from abc import ABC, abstractmethod
from typing import Any


class LossFunction(ABC):  # to be completed
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """
        Forward pass of the loss function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: Computed loss value.
        """
        ...
