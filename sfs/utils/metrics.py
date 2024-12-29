# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# Calculate Metrics
# --------------------------------------------------------
# 
# Created on Thu Dec 26 2024.


import torch

def compute_IoU(tensor1, tensor2):

    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)

    intersection = (tensor1_flat & tensor2_flat).sum().item()
    union = (tensor1_flat | tensor2_flat).sum().item()

    iou = intersection / union if union != 0 else 0.0
    
    return iou