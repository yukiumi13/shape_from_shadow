from dataclasses import dataclass
from typing import Literal, Optional
from jaxtyping import Float
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, repeat

from global_config import get_cfg
from model.types import OptimizationVariables, RenderOutputs, OccVolume
from .base_render import RenderPipeline

from utils.logger import std_logger, cyan


@dataclass
class ShadowRenderCfg:
    name: Literal["simple_shadow_render"]

class ShadowRender(RenderPipeline[ShadowRenderCfg]):
    
    def __init__(self, cfg: ShadowRenderCfg) -> None:
        super().__init__(cfg)
        
        
            
    def forward(self, 
                scene_context: OptimizationVariables,
                occ_volume: OccVolume,) -> RenderOutputs:
        
        batch = scene_context["latent_set"].size(0)
        
        # LatentSet-to-Logits
        return RenderOutputs(grid=xyz_grid, occ_logits=occ_logits_volume)