# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Unit Test:
#       - Render ground shadow map from 3D mesh and single light
# --------------------------------------------------------
# 
# Created on Tue Dec 24 2024.

from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from sfs.model.autoencoders import get_autoencoder, Shape2VecSetAutoEncoderCfg
    from sfs.model.render_pipelines import get_renderer, ShadowRenderCfg
    from sfs.model.types import OptimizationVariables

from sfs.utils.sys_monitor import PeakCUDAMemoryTracker
from sfs.utils.logger import std_logger
from sfs.utils.export import export_img

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image



RES = 128 # Grid Resolution
CAS = 1 # Number of Cascades of Multi-resolution Occ Grid
SIGMA_THR = 1e-2 # Determine whether skip the queried cell
RENDER_RES =128
ITVL = 100
BOUND = 2.

device = torch.device("cuda:0")

cfg = Shape2VecSetAutoEncoderCfg(name = 'shape2vecset', pretrained_weight='/home/yangli/sfs/checkpoints/pretrained/ae_kl_d512_m512_l8.pth', occ_resolution=RES)
shape2vecset_autoencoder = get_autoencoder(cfg).cuda()

cfg = ShadowRenderCfg(name = 'simple_shadow_render', ground_res=RENDER_RES, ground_aabb= (-1.,-1.,1.,1.), sample_per_ray= 128)
shadow_renderer = get_renderer(cfg).cuda()

# Load sample mesh 
sample_path = Path('/mntdata/shapenet/sample')
hash_id = 'a6d282a360621055614d73f24792753f'
sample_surface_path = sample_path / 'surface' / f'{hash_id}.npz'
surface = np.load(sample_surface_path)["points"]

# Plot
fig = plt.figure(figsize=(12, 6))

ax_occ = fig.add_subplot(121, projection='3d')
ax_occ.set_title('Occupancy Grid')
ax_occ.set_xlabel("X")
ax_occ.set_ylabel("Y")
ax_occ.set_zlabel("Z")
ax_occ.set_xlim(-BOUND, BOUND)
ax_occ.set_ylim(-BOUND, BOUND)
ax_occ.set_zlim(-BOUND, BOUND)

ax_shadow = fig.add_subplot(122)
ax_shadow.set_title('Shadow Map')

ind = np.random.default_rng().choice(
                    surface.shape[0], 2048, replace=False)
surface2048 = torch.from_numpy(surface[ind][None]).cuda().float()
latent = shape2vecset_autoencoder.auto_encoder(surface2048, shape2vecset_autoencoder.grid_queries)['latent_set']
occ_vol = shape2vecset_autoencoder({'latent_set':latent})

pts3d = occ_vol.grid[ occ_vol.occ_logits > 0].view(-1,3).cpu()
ax_occ.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]) 

light_position = torch.randn(1,3)
light_position[:, 2] = (light_position[:, 2] + 2 ) / 2

# light_position = torch.tensor([[0.,0.,2.]])

ax_occ.scatter(*light_position[0], color='r', s=1) 
ax_occ.text(*light_position[0], 'Light', color='red') 


scene = OptimizationVariables(latent_set=torch.randn(1, 512).cuda(),
                              light_position=light_position.cuda(),
                              object_pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).cuda(),
                              object_scale=torch.tensor([1.0]).cuda())

with PeakCUDAMemoryTracker() as tracker:
    shadow_pred = shadow_renderer(scene, occ_vol)

ax_shadow.imshow(shadow_pred["shadow_map"][0].detach().cpu(), cmap='viridis', aspect='auto')
view_func = lambda x: x.view_init(elev=45, azim=45)
list(map(view_func, [ax_occ]))
fig.savefig("shadow_plot.png")

save_image(shadow_pred["shadow_map"], "assets/sample_shadow.png")

std_logger.info(f"Peak CUDA Memoy Allocated: {tracker.peak_memory / 1024**3:.2f} GB")



