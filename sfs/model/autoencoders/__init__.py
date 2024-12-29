from typing import Optional, Union
from .base_ae import AutoEncoder
from .s2vs_ae import Shape2VecSetAutoEncoder, Shape2VecSetAutoEncoderCfg

AUTO_ENCODERS = {
    "shape2vecset": Shape2VecSetAutoEncoder,
}

AutoEncoderCfg = Union[Shape2VecSetAutoEncoderCfg]
ImplementedAutoEncoders = Union[Shape2VecSetAutoEncoder] # For Linting

def get_autoencoder(cfg: AutoEncoderCfg) -> ImplementedAutoEncoders:
    autoencoder = AUTO_ENCODERS[cfg.name]
    autoencoder = autoencoder(cfg)
    return autoencoder