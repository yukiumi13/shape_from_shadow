from dataclasses import dataclass

from jaxtyping import Float
from typing import Literal
from torch import Tensor

from .loss import Loss
    
@dataclass
class LossMSECfg:
    name: Literal["mse"]
    weight: float = 1.0


@dataclass
class LossMSECfgWrapper:
    mse: LossMSECfg


class LossMSE(Loss[LossMSECfg, LossMSECfgWrapper]):
    def forward(
        self,
        pred: Float[Tensor, "*batch C H W"],
        gt: Float[Tensor, "*batch C H W"]
    ) -> Float[Tensor, ""]:
        delta = pred - gt
        return self.cfg.weight * (delta**2).mean()
