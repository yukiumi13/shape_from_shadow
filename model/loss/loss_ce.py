from dataclasses import dataclass

from jaxtyping import Float
from typing import Literal
from torch import Tensor
import torch.nn.functional as F

from .loss import Loss

@dataclass
class LossCrossEntropyCfg:
    name: Literal["cross_entropy"]
    weight: float = 1.0


class LossCrossEntropy(Loss):
    def forward(
        self,
        pred: Float[Tensor, "*batch C H W"],
        gt: Float[Tensor, "*batch C H W"]
    ) -> Float[Tensor, ""]:
        delta = F.cross_entropy(pred, gt)
        return self.cfg.weight * delta.mean()
