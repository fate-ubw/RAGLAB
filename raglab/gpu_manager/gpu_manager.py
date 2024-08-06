import os
import torch
from typing import List

class GPUAllocationError(Exception):
    pass

class GPUManager:
    def __init__(self, available_gpu_ids: List[int]):
        self.available_gpu_ids = available_gpu_ids
        self.current_gpu_id = None
        self.used_gpu_ids = []

    def allocate_gpu(self) -> int:
        if not self.available_gpu_ids:
            raise GPUAllocationError("No available GPUs")
        
        self.current_gpu_id = self.available_gpu_ids.pop(0)
        self.used_gpu_ids.append(self.current_gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.current_gpu_id}"
        return self.current_gpu_id
