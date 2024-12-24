from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from jaxtyping import Float
from torch import Tensor

from torch import nn

from model.types import OccVolume, RenderOutputs

T = TypeVar("T")


class RenderPipeline(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        light_position: Float[Tensor, "*batch 3"],
        occ_volume: OccVolume,
    ) -> RenderOutputs:
        pass

