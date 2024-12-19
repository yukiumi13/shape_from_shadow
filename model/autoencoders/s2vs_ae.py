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
        # model_name = f"kl_d512_m{cfg.latent_num}_l{cfg.latent_dim}"
        self.auto_encoder = make_s2vs_ae(latent_num=cfg.latent_num, latent_dim=cfg.latent_dim)
        
        if cfg.pretrained_weight is not None:
            ckpt = torch.load(cfg.pretrained_weight)
            self.auto_encoder.load_state_dict(ckpt[
                          'model'], strict=True)
            del ckpt
            
        # Register Constants
        # 3D Grid Generation from Shape2VecSet/eval.py
        self.density = cfg.occ_resolution
        x = np.linspace(-1, 1, self.density+1)
        y = np.linspace(-1, 1, self.density+1)
        z = np.linspace(-1, 1, self.density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
            np.float32)).view(3, -1).transpose(0, 1)[None]
        
        self.register_buffer("grid_queries", grid)
        self.grid_queries: Tensor[Float, "1 N_queries 3"]
            
    def forward(self, 
                context: OptimizationVariables,) -> OccVolume:
        
        batch = context['latent_set'].size(0)
        
        # LatentSet-to-Logits
        latent_set = context['latent_set'] # (b, k, l)
        occ_logits = self.auto_encoder.decode(latent_set, self.grid_queries)['logits']
        
        # Logits-to-OccVolume
        occ_logits_volume = rearrange(occ_logits, "batch (x y z) -> batch y x z", x=self.density+1, y=self.density+1, z=self.density+1) # Axis swapping as Shape2VecSet/eval.py
        xyz_grid = repeat(self.grid_queries, "() (x y z) coord -> b x y z coord", b = batch)
        return OccVolume(grid=xyz_grid, occ_logits=occ_logits_volume)