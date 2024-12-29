from .loss import Loss
from .loss_mse import LossMSE, LossMSECfgWrapper
from .loss_ce import LossCrossEntropy, LossCrossEntropyCfgWrapper
from typing import Union

LOSSES = {
    LossMSECfgWrapper: LossMSE,
    LossCrossEntropyCfgWrapper: LossCrossEntropy
}

LossCfgWrapper = Union[LossMSECfgWrapper , LossCrossEntropyCfgWrapper]


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]