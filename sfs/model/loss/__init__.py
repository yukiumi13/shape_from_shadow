from .loss import Loss
from .loss_mse import LossMSE, LossMSECfgWrapper
from .loss_ce import LossCrossEntropy, LossCrossEnctropyCfgWrapper
from typing import Union

LOSSES = {
    LossMSECfgWrapper: LossMSE,
    LossMSECfgWrapper: LossCrossEntropy
}

LossCfgWrapper = Union[LossMSECfgWrapper , LossCrossEnctropyCfgWrapper]


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]