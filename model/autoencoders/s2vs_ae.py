# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Implementation of Shape2VecSet
# --------------------------------------------------------
# 
# --------------------------------------------------------
# References:
#   - https://arxiv.org/abs/2301.11445
# --------------------------------------------------------
# Created on Mon Dec 23 2024.



from dataclasses import dataclass
from typing import Literal, Optional
from jaxtyping import Float
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, repeat

from global_config import get_cfg
from model.types import OptimizationVariables, OccVolume
from .base_ae import AutoEncoder
from .vec2shape.models_ae import make_s2vs_ae

from utils.logger import std_logger, cyan


@dataclass
class Shape2VecSetAutoEncoderCfg:
    name: Literal["shape2vecset"]
    latent_num: int = 512
    latent_dim: int = 8
    occ_resolution: int = 128
    pretrained_weight: Optional[str] = None

class Shape2VecSetAutoEncoder(AutoEncoder[Shape2VecSetAutoEncoderCfg]):
    
    def __init__(self, cfg: Shape2VecSetAutoEncoderCfg) -> None:
        super().__init__(cfg)
        
        # Set up Shape2VecSet
        self.auto_encoder = make_s2vs_ae(latent_num=cfg.latent_num, latent_dim=cfg.latent_dim)
        
        std_logger.info(f"Model: {cyan(f'kl_d512_m{cfg.latent_num}_l{cfg.latent_dim}')}")
        std_logger.info(f"Pretrain: {cyan(cfg.pretrained_weight)}")
        
        if cfg.pretrained_weight is not None:
            ckpt = torch.load(cfg.pretrained_weight)
            self.auto_encoder.load_state_dict(ckpt[
                          'model'], strict=True)
            del ckpt
            
        # Register Constants
        # ! Grid convention here differs from the original Shape2VecSet
        # ! Original Grid is generated with xy indexing, which means grid[i][j][k] = i * y + j * x + k * z.
        # ! It is inconvenient for F.grid_sample, which assumes the input volume with the shape of X Y Z C and 
        # ! the sampling grid is (x y z), i.e., grid[i][j][k] = i * x + j * y + k * z
        self.density = cfg.occ_resolution
        x = np.linspace(-1, 1, self.density)
        y = np.linspace(-1, 1, self.density)
        z = np.linspace(-1, 1, self.density)
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
            np.float32)).view(3, -1).transpose(0, 1)[None]
        
        self.register_buffer("grid_queries", grid)
        self.grid_queries: Tensor[Float, "1 N_queries 3"]
            
    def forward(self, 
                context: OptimizationVariables,) -> OccVolume:
        
        batch = context['latent_set'].size(0)
        
        # LatentSet-to-Logits
        latent_set = context['latent_set'] # (b, k, l)
        occ_logits = self.auto_encoder.decode(latent_set, self.grid_queries)
        
        # Logits-to-OccVolume
        occ_logits_volume = rearrange(occ_logits, "batch (x y z) () -> batch x y z", x=self.density, y=self.density, z=self.density)
        xyz_grid = repeat(self.grid_queries, "() (x y z) coord -> b x y z coord", b = batch, x=self.density, y=self.density, z=self.density)
        
        occ_logits_volume, xyz_grid = self._postprocess(occ_logits_volume, xyz_grid)
        
        return OccVolume(grid=xyz_grid, occ_logits=occ_logits_volume)
    
    def _postprocess(self, 
                     occ_logits_volume: Float[Tensor, "*batch L W H"],
                     xyz_grid: Float[Tensor, "*batch L W H 3"]):
        """postprocess specified for Shape2VecSet

        Args & Returns:
            occ_logits_volume (Float[Tensor, ): 
            xyz_grid (Float[Tensor, ): 
        """
        
        # Axis swapping and flipping: x y z -> x -z y 
        occ_logits_volume = rearrange(occ_logits_volume, "... x y z -> ... x z y")
        occ_logits_volume = torch.flip(occ_logits_volume, [-2])
        xyz_grid = rearrange(xyz_grid, "... L W H xyz -> ... L H W xyz")
        xyz_grid = torch.flip(xyz_grid, [-3])
        xyz_grid = xyz_grid[..., [0, 2, 1]]
        xyz_grid[..., -2:-1] *= -1
        
        return occ_logits_volume, xyz_grid
    
        