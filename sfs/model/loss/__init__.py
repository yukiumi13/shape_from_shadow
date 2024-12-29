from .loss import Loss
from .loss_mse import LossMSE, LossMSECfg
from .loss_ce import LossCrossEntropy, LossCrossEntropyCfg
from typing import Union

LOSSES = {
    "mse": LossMSE,
    "cross_entropy": LossCrossEntropyCfg
}

LossCfg = Union[LossMSECfg , LossCrossEntropyCfg]

def get_losses(cfg: LossCfg) -> Loss:
    return LOSSES[cfg.name](cfg)
