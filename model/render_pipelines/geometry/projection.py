#
# Created on Sun Dec 22 2024
# 3D Transforms 
# Copyright (c) Yang Li Microsoft 2024
#

from jaxtyping import Float
from typing import Optional, Tuple
import torch
from torch import Tensor
from einops import rearrange, repeat, einsum
import numpy as np
from roma import unitquat_to_rotmat

def homogeneous(xyz: Float[Tensor, "... dim"], z:float = 1.) -> Float[Tensor, "... dim+1"]:
    Z = torch.ones_like(xyz[...,:1]) * z
    return torch.cat([xyz, Z], dim = -1)

def gen_plane_rays(origin: Float[Tensor, "*batch 3"],
             target: Float[Tensor, "*batch H W 3"]) -> Tuple[Float[Tensor, "*batch N 3"], Float[Tensor, "*batch N 3"]]:
    
    H, W = target.shape[-3:-1]
    target = rearrange(target, "... h w xyz -> ... (h w) xyz")
    
    rays_o = repeat(origin, "... xyz -> ... N xyz", N = H * W).clone()
    rays_d = target - rays_o
    rays_d /= rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d
    
def gen_plane_pts(canonical_xyz: Float[Tensor, "*batch H W 3"],
                  transform: Optional[Float[Tensor, "*batch 4 4"]] = None):
    
    c_xyz1 = homogeneous(canonical_xyz)
    
    if transform is not None:
        return einsum(transform, c_xyz1, "... i j, ... h w j -> ... h w i")
    else:
        return canonical_xyz

def gen_xy_from_aabb(aabb:Float[Tensor, "4"],
                    res:int = 128)-> Float[Tensor, "RES RES 2"]:
    device = aabb.device
    xmin, ymin, xmax, ymax = aabb.tolist()
    x = np.linspace(xmin, xmax, res)
    y = torch.linspace(ymin, ymax, res)
    xp, yp = np.meshgrid(x, y)
    xy_plane = torch.tensor(rearrange([xp, yp], "i h w -> h w i"), device=device).float()
    return xy_plane

def sRT_from_pose_and_scale(pose: Float[Tensor, "*batch 7"], # [xyzw, translation] 
                                                            # ! canonical-to-world
                           scale: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch 4 4"]:
    '''Rotaion->scale->translation.
        sRt = sR @ xyz + rt    '''
    R = unitquat_to_rotmat(pose[...,:4])
    T = pose[..., 4:]
    sRT = torch.eye(4, device=R.device).expand(*R.shape[:-2], 4, 4).clone()
    sR = scale * R
    sRT[...,:3,:3] = sR
    sRT[...,:3,3] = T
    return sRT, R

def geo_trf(trf: Float[Tensor, "*batch 4 4"],
            pts: Float[Tensor, "*batch N 3"]):
    pts_homo = homogeneous(pts)
    return einsum(trf, pts_homo, "... i j, ... n j -> ... n i")[...,:3]

