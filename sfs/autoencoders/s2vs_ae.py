from dataclasses import dataclass
from typing import Literal, Optional
import torch
from torch import Tensor
import numpy as np

from global_config import get_cfg
from sfs.types import OptimizationVariables, OccVolume
from .base_ae import AutoEncoder
from .vec2shape import models_ae


@dataclass
class Shape2VecSetAutoEncoderCfg:
    name: Literal["shape2vecset"]
    latent_dim: int = 512
    latent_num: int = 8
    occ_resolution: int = 128
    pretrained_weight: Optional[str] = None

class Shape2VecSetAutoEncoder(AutoEncoder[Shape2VecSetAutoEncoderCfg]):
    
    def __init__(self, cfg: Shape2VecSetAutoEncoderCfg) -> None:
        super().__init__(cfg)
        
        # Set up Shape2VecSet
        model_name = f"kl_d512_m{cfg.latent_dim}_l{cfg.latent_num}"
        self.auto_encoder = models_ae.__dict__[model_name]()
        
        if cfg.pretrained_weight is not None:
            ckpt = torch.load(cfg.pretrained_weight)
            self.auto_encoder.load_state_dict(ckpt[
                          'model'], strict=True)
            del ckpt
            
        # Register Constants
        # 3D Grid Generation from Shape2VecSet/eval.py
        density = cfg.occ_resolution
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
            np.float32)).view(3, -1).transpose(0, 1)[None]
        
        self.register_buffer("grid_queries", grid)
            
    def forward(self, 
                context: OptimizationVariables,
                deterministic: bool = True) -> OccVolume:
        
        # LatentSet-to-Occ
        
        