# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Unit Test:
#       - CUDA-based Efficient Ray Marching
# --------------------------------------------------------
# Reference:
#       - https://github.com/NVlabs/instant-ngp
# --------------------------------------------------------
# Created on Wed Dec 25 2024.

from sfs.model.render_pipelines.vol_render import raymarching
from sfs.model.render_pipelines.vol_render.nerf.renderer import near_far_from_aabb, custom_meshgrid
from sfs.model.autoencoders import get_autoencoder, Shape2VecSetAutoEncoderCfg
from sfs.model.types import OptimizationVariables

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange, repeat

RES = 128 # Grid Resolution
CAS = 1 # Number of Multi-resolution Occ Grid
SIGMA_THR = 0.5 # Determine whether skip the queried cell
RENDER_RES =64 

gap = 2. / 128
device = 'cuda:0'

# ! There may be some bugs in CUDA implementation of Morton 3D Encoding, pytorch instead
def morton3D(coords):
    def part1by2(n):
        n &= 0x1fffff  # we only look at the first 21 bits
        n = (n | n << 32) & 0x1f00000000ffff
        n = (n | n << 16) & 0x1f0000ff0000ff
        n = (n | n << 8) & 0x100f00f00f00f00f
        n = (n | n << 4) & 0x10c30c30c30c30c3
        n = (n | n << 2) & 0x1249249249249249
        return n

    def _morton3D(x, y, z):
        return (part1by2(x) | part1by2(y) << 1 | part1by2(z) << 2)

    morton_codes = []
    for coord in coords:
        x, y, z = coord
        morton_code = _morton3D(x.item(), y.item(), z.item())
        morton_codes.append(morton_code)
    return torch.tensor(morton_codes, dtype=torch.int32).type_as(coords)

cfg = Shape2VecSetAutoEncoderCfg(name = 'shape2vecset', pretrained_weight='/home/yangli/sfs/checkpoints/pretrained/ae_kl_d512_m512_l8.pth', occ_resolution=RES)
shape2vecset_autoencoder = get_autoencoder(cfg).cuda()

# Load sample mesh 
sample_path = Path('/mntdata/shapenet/sample')
hash_id = 'a6d282a360621055614d73f24792753f'
sample_surface_path = sample_path / 'surface' / f'{hash_id}.npz'
surface = np.load(sample_surface_path)["points"]

ind = np.random.default_rng().choice(
                    surface.shape[0], 2048, replace=False)
surface2048 = torch.from_numpy(surface[ind][None]).cuda().float()
latent = shape2vecset_autoencoder.auto_encoder(surface2048, shape2vecset_autoencoder.grid_queries)['latent_set']
output = shape2vecset_autoencoder({'latent_set':latent})
occ = (output.occ_logits > 0).float()

# plot
fig, ax = plt.subplots(1,2,subplot_kw={'projection': '3d'})
data = output.grid[output.occ_logits>0].cpu()
ax[0].scatter(data[:, 0], data[:, 1], data[:, 2], s=1)
ax[0].set_xlim([-1, 1])
ax[0].set_ylim([-1, 1])
ax[0].set_zlim([-1, 1])


# ! raymarching.density_grid is morton-ordered
# construct density grid
xx, yy, zz = custom_meshgrid(torch.arange(RES, dtype=torch.int32), 
                             torch.arange(RES, dtype=torch.int32), 
                             torch.arange(RES, dtype=torch.int32)) # ! int32 MUST be specified since Morton3D involves bitwise operations

coords = rearrange([xx, yy, zz], "xyz X Y Z -> () X Y Z xyz") 
indices = morton3D(rearrange(coords, "() X Y Z xyz -> (X Y Z) xyz")).long() # [N]
occ = rearrange(occ, "() Z Y X -> () () Z Y X ")
density_grid = torch.zeros(CAS, RES * RES * RES).type_as(occ)
density_grid[0, indices] = rearrange(occ, "() () Z Y X -> (Y X Z)")
density_bitfield = raymarching.packbits(density_grid, SIGMA_THR) # [CAS, H ** 3] -> [CAS, H ** 3 // 8]


# Efficient Ray Marching
# Init Render Context
light_pos = torch.tensor([0,0,1.]).to(device)

# Generate Rays
ground_xy = torch.meshgrid(torch.linspace(-1,1, RENDER_RES), torch.linspace(-1,1,RENDER_RES), indexing = 'xy')
ground_xyz = rearrange([*ground_xy, -torch.ones(RENDER_RES, RENDER_RES)], "xyz h w -> (h w) xyz").to(device)
rays_o = repeat(light_pos, "xyz -> b xyz", b=RENDER_RES**2).contiguous()
rays_d = ground_xyz - light_pos
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True).contiguous()
nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, torch.tensor([-1.,-1.,-1.,1.,1.,1.]).to(device), 0.01)
print(nears, fars)
n_alive = RENDER_RES ** 2
n_step = 100
rays_alive = torch.arange(n_alive, dtype=torch.int32).to(device)
rays_t = nears.clone()

# CUDA Implementation of sampling along rays
xyzs, dirs, ts = raymarching.march_rays(
                                        n_alive,
                                        n_step,
                                        torch.arange(n_alive, device = device).int(),
                                        rays_t,
                                        rays_o, 
                                        rays_d, 
                                        2., 
                                        True, 
                                        density_bitfield, 
                                        1, 
                                        RES, 
                                        nears, 
                                        fars, 
                                        )


data = xyzs.cpu()
print(xyzs.min(dim=0))
print(xyzs.max(dim=0))
print(xyzs.shape)
ground_xyz = ground_xyz.cpu()
# ax[1].scatter(ground_xyz[:,0], ground_xyz[:,1],ground_xyz[:,2], color = 'red')
ax[1].scatter(data[::50, 0], data[::50, 1], data[::50, 2], s=1)
light_pos= light_pos.cpu()
ax[1].scatter(light_pos[0], light_pos[1],light_pos[2], color = 'green')
ax[1].set_xlim([-1, 1])
ax[1].set_ylim([-1, 1])
ax[1].set_zlim([-1, 1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_zlabel('z')

view_func = lambda x: x.view_init(elev=45, azim=45) # View Control Handle
list(map(view_func, ax))
plt.show()