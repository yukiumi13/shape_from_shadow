# Copyright (c) 2024 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# System Performance Benchmarker
# --------------------------------------------------------
# 
# Created on Tue Dec 24 2024.

import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch

from .logger import std_logger, cyan

class PeakCUDAMemoryTracker:
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):     
        self.peak_memory = torch.cuda.max_memory_allocated()

        
class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.peak_memory_allocation = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = time()
            yield
        finally:
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)
                
    @contextmanager
    def track_peak_memory(self, tag:str ):
        try:
            torch.cuda.reset_peak_memory_stats()
            yield
        finally:
            self.peak_memory_allocation[tag] = torch.cuda.max_memory_allocated() / 1024**3
            

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        system_report = {'execution_time': dict(self.execution_times), 'peak_memory': dict(self.peak_memory_allocation)}
        with path.open("w") as f:
            json.dump(system_report, f)
            

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            std_logger.info(f"{cyan(tag)}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
        for tag, gb in self.peak_memory_allocation.items():
            std_logger.info(f"{cyan(tag)}: {len(gb)} calls, peak memory {np.mean(gb)} GB allocated")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
        self.peak_memory_allocation = defaultdict(list)