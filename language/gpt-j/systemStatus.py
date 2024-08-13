import psutil
import torch

def get_gpu_memory_info():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
    return free_memory / (1024**3)

def get_cpu_memory_info():
    mem = psutil.virtual_memory()
    return mem.total / (1024**3)