from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from model.autoencoders import get_autoencoder, Shape2VecSetAutoEncoderCfg
    from model.render_pipelines.vol_render import raymarching
    from model.render_pipelines.shadow_render import ShadowRenderCfg, ShadowRender
    from model.types import OccVolume, OptimizationVariables

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
occ_pred = OccVolume(grid = shape2vecset_autoencoder.grid_queries, occ_logits=occ_grid)

render_cfg = ShadowRenderCfg(name='simple_shadow_render')
shadow_render = ShadowRender(render_cfg).cuda()

scene = OptimizationVariables(latent_set=torch.randn(1, 512).cuda(),
                              light_position=torch.tensor([[0., 0., 2.]]).cuda(),
                              object_pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).cuda(),
                              object_scale=torch.tensor([1.0]).cuda())

out = shadow_render(scene, occ_pred)
xyzs = out['queries_coords']

from utils.export import export_pts
export_pts(xyzs[::100], "queried_xyzs.ply")
export_pts(shape2vecset_autoencoder.grid_queries[occ>0], "occ.ply")

