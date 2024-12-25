from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from jaxtyping import Float
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, repeat, einsum

from global_config import get_cfg
from model.types import OptimizationVariables, RenderOutputs, OccVolume
from .base_render import RenderPipeline
from .geometry.projection import gen_xy_from_aabb, homogeneous, gen_plane_rays, sRT_from_pose_and_scale, geo_trf
from .vol_render import raymarching

from utils.logger import std_logger, cyan

# ! Canonical Space aabb is fixed to align AE
OBJECT_AABB = [-1., -1., -1., 1., 1., 1.]


@dataclass
class ShadowRenderCfg:
    name: Literal["simple_shadow_render"]
    rays_sample: int = -1
    sample_per_ray: int = 128
    ground_aabb: Tuple[float] = (-1., -1., 1., 1.)
    ground_res: int = 128
    

class ShadowRender(RenderPipeline[ShadowRenderCfg]):
    
    def __init__(self, cfg: ShadowRenderCfg) -> None:
        super().__init__(cfg)
        plane_xy = gen_xy_from_aabb(torch.tensor(cfg.ground_aabb), cfg.ground_res)
        self.register_buffer("plane_xyz", homogeneous(plane_xy, z=-1.))
        self.plane_xyz: Float[Tensor, "*batch h w 3"]
        
    def _composite_sample_along_ray(self, rays_value:Float[Tensor, "*batch C H W D"]) -> Float[Tensor, "*batch C H W"]:
        return rays_value.amax(dim=-1)
    
    def _sample_volume(self, 
                        xyzs:Float[Tensor, "*batch H W D 3"],
                        volume3d:Float[Tensor, "*batch C Z Y X"])-> Float[Tensor, "*batch C H W D"]:
        return torch.nn.functional.grid_sample(volume3d, xyzs, align_corners=False)
        
    def forward(self, 
                scene_context: OptimizationVariables,
                occ_volume: OccVolume,)-> RenderOutputs:
        
        H, W, D = self.cfg.ground_res, self.cfg.ground_res, self.cfg.sample_per_ray
        
        # World Space Rays
        rays_o, rays_d = gen_plane_rays(scene_context["light_position"], self.plane_xyz)
        
        # World Space to Canonical Space (Object)
        object_pose = scene_context["object_pose"]
        object_scale = scene_context["object_scale"]
        obj_to_world, rot_o2w = sRT_from_pose_and_scale(object_pose, object_scale)
        world_to_obj, rot_w2o = obj_to_world.inverse(), rearrange(rot_o2w, "... i j -> ... j i")
        rays_o = geo_trf(world_to_obj, rays_o)
        rays_d = einsum(rot_w2o, rays_d, "... i j, ... n j -> ... n i") 

        # Efficient Ray-marching
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, torch.tensor(OBJECT_AABB).type_as(rays_o), 0.05)
        nears, fars = rearrange(nears, "... (h w) -> ... h w ()", h = H, w = W), rearrange(fars, "... (h w) -> ... h w ()", h = H, w = W) 
        ts = nears + torch.linspace(0,1,D).type_as(nears) * (fars - nears)

        xyzs = repeat(rays_o, "... (h w) i -> ... h w d i", h = H, w = W, d = D) + \
                repeat(ts , "... h w d -> ... h w d ()") * repeat(rays_d, "... (h w) i -> ... h w d i", h = H, w = W, d = D) # [B, H, W, D, 3]
              
        # Query Occ
        occ_vol = rearrange((occ_volume.occ_logits).sigmoid(), "... Z Y X -> ... () Z Y X")  
        rays_occ = self._sample_volume(xyzs, occ_vol)
        ground_shadow = self._composite_sample_along_ray(rays_occ)
        shadow_map = rearrange(ground_shadow, "... () H W -> ... H W")
        
        
        return RenderOutputs(shadow_map=shadow_map, queries_coords=xyzs, occ=rays_occ)
    
    

    