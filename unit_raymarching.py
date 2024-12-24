# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
#  Unit_test:
#       - Ray Marching
# --------------------------------------------------------
# 
# Created on Tue Dec 24 2024.

from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from model.autoencoders import get_autoencoder, Shape2VecSetAutoEncoderCfg
    from model.render_pipelines.vol_render import raymarching
    from model.render_pipelines.vol_render.nerf.renderer import near_far_from_aabb

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange, repeat, reduce
import math

RES = 64 # Grid Resolution
CAS = 1 # Number of Cascades of Multi-resolution Occ Grid
SIGMA_THR = 1e-2 # Determine whether skip the queried cell
RENDER_RES =64 
ITVL = 100

device = torch.device("cuda:0")

# Load sample occ grid
gap = 2. / 128
cfg = Shape2VecSetAutoEncoderCfg(name = 'shape2vecset', pretrained_weight='/home/yangli/sfs/checkpoints/pretrained/ae_kl_d512_m512_l8.pth')
shape2vecset_autoencoder = get_autoencoder(cfg).cuda()

# Load sample mesh 
sample_path = Path('/mntdata/shapenet/sample')
hash_id = 'a6d282a360621055614d73f24792753f'
sample_surface_path = sample_path / 'surface' / f'{hash_id}.npz'
surface = np.load(sample_surface_path)["points"]

ind = np.random.default_rng().choice(
                    surface.shape[0], 2048, replace=False)
surface2048 = torch.from_numpy(surface[ind][None]).cuda().float()

# Mesh to Occ
occ = shape2vecset_autoencoder.auto_encoder(surface2048, shape2vecset_autoencoder.grid_queries)['logits'] 
occ_res = shape2vecset_autoencoder.density + 1
occ_grid = rearrange(occ, "() (x y z) -> () x y z", x = occ_res, y = occ_res, z = occ_res)[:, :(occ_res-1), :(occ_res-1), :(occ_res-1)]


# Plot occ
from trimesh import PointCloud
occ_pts = shape2vecset_autoencoder.grid_queries[occ>0].cpu()
occ_pts = PointCloud(occ_pts)
occ_pts.export('occ_pred.ply')


# Occ -> Density Grid
density_grid = reduce(occ_grid, "b (x x1) (y y1) (z z1) -> b x y z", 'max', b=CAS, x=RES,y=RES, z=RES )
density_bitfield = raymarching.packbits(density_grid, SIGMA_THR) # [CAS, H ** 3] -> [CAS, H ** 3 // 8]

# Efficient Ray Marching
# Init Render Context
light_pos = torch.tensor([1.,1.,2.]).to(device)

# Generate Rays
ground_xy = torch.meshgrid(torch.linspace(-1,1, RENDER_RES), torch.linspace(-1,1,RENDER_RES), indexing = 'xy')
ground_xyz = rearrange([*ground_xy, -torch.ones(RENDER_RES, RENDER_RES)], "xyz h w -> (h w) xyz").to(device)
rays_o = repeat(light_pos, "xyz -> b xyz", b=RENDER_RES**2).contiguous()
rays_d = ground_xyz - light_pos
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True).contiguous()
nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, torch.tensor([-1.,-1.,-1.,1.,1.,1.]).to(device), 0.05)
n_alive = RENDER_RES ** 2
n_step = 100
rays_alive = torch.arange(n_alive, dtype=torch.int32).to(device)
rays_t = nears.clone()
xyzs, dirs, ts = raymarching.march_rays(RENDER_RES ** 2, 
                                        n_step, 
                                        rays_alive, 
                                        rays_t, 
                                        rays_o, 
                                        rays_d, 
                                        1., 
                                        True, 
                                        density_bitfield, 
                                        1, 
                                        RES, 
                                        nears, 
                                        fars, 
                                        )

print(f"{xyzs.shape=}",f"{dirs.shape=}", f"{ts.shape=}")
print(ts)

data = xyzs.cpu()
ray_xyzs_pts = PointCloud(data)
ray_xyzs_pts.export('rays.ply')
 