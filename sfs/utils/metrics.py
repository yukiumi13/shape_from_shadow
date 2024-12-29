# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Calculate Metrics
# --------------------------------------------------------
# 
# Created on Thu Dec 26 2024.


import torch
from torch import Tensor
from jaxtyping import Float

def compute_IoU(tensor1:Float[Tensor, "*batch C H W"], 
                tensor2:Float[Tensor, "*batch C H W"]):

    tensor1_flat = tensor1.view(-1) > 0.5
    tensor2_flat = tensor2.view(-1) > 0.5

    intersection = (tensor1_flat.bool() & tensor2_flat.bool()).sum().item()
    union = (tensor1_flat.bool() | tensor2_flat.bool()).sum().item()

    iou = intersection / union if union != 0 else 0.0
    
    return iou