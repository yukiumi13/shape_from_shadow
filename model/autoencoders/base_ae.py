from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from model.types import OptimizationVariables, OccVolume

T = TypeVar("T")


class AutoEncoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: OptimizationVariables,
        deterministic: bool,
    ) -> OccVolume:
        pass
