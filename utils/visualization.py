# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Plot functions
# --------------------------------------------------------
# 
# Created on Thu Dec 26 2024.


import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import Tensor
from torchvision.utils import make_grid
from einops import rearrange

def visualize_pred_gt_comparison(
                                pred: Float[Tensor, "C H W"],
                                gt: Float[Tensor, "C H W"]
):
    H, W = pred.shape[1:]
    
    comp = make_grid([pred, gt], nrow=1, ncol=2).cpu().numpy()
    # comp = rearrange([ray_pred, ray_gt], "n c h w -> c h (n w)").cpu().numpy()
    
    fig, ax = plt.figure(figsize=(10, 20))
    ax.imshow(comp)
    ax.axis('off')
    ax.text(0.25, -0.025, 'Pred', fontsize=20, va='center', ha='center', transform=ax.transAxes)
    ax.text(0.75, -0.025, 'GT', fontsize=20, va='center', ha='center', transform=ax.transAxes)
    ax.set_title('Comparison')
    ax.set_axi
    
    return fig