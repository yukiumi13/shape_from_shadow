from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


T_cfg = TypeVar("T_cfg")


class Loss(nn.Module, ABC, Generic[T_cfg]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name

    @abstractmethod
    def forward(
        self,
        pred,
        gt,
    ) -> Float[Tensor, ""]:
        pass
