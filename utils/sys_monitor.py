# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# CUDA Memory Logger
# --------------------------------------------------------
# 
# Created on Tue Dec 24 2024.

import torch

class PeakCUDAMemoryTracker:
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.peak_memory = torch.cuda.max_memory_allocated()