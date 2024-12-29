# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Unit_test:
#       - Encode and Reconstruct 3D Meshes 
# --------------------------------------------------------
# 
# Created on Tue Dec 24 2024.


from sfs.model.autoencoders import get_autoencoder, Shape2VecSetAutoEncoderCfg
from sfs.model.types import OptimizationVariables

import mcubes
import trimesh
import torch
import numpy as np 
from pathlib import Path
from einops import rearrange
import matplotlib.pyplot as plt

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
latent = shape2vecset_autoencoder.auto_encoder(surface2048, shape2vecset_autoencoder.grid_queries)['latent_set']
output = shape2vecset_autoencoder({'latent_set':latent})

data = output.grid[output.occ_logits>0].cpu()
data = rearrange(data, "... c -> (...) c")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('vae.png')