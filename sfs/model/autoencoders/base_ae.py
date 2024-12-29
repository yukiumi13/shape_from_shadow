# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Base 3D VAE
# --------------------------------------------------------
# 
# Created on Sun Dec 29 2024.

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from sfs.model.types import OptimizationVariables, OccVolume

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

