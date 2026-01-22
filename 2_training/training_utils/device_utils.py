import torch
import os

def get_num_workers():
    """Set the appropriate number of workers to use based on core count"""
    return int(os.cpu_count())

def get_device(verbose=0):
    """Detect and return device (CUDA or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if cuda, determine which gpu by seeing which has more free VRAM
    most_free_vram = -1
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free, _ = torch.cuda.mem_get_info()
        if free > most_free_vram:
            device = torch.device(f"cuda:{i}")
            most_free_vram = free

    if verbose == 1:
        print(f"*** DEVICE: {device} SELECTED FOR TRAINING WITH {(most_free_vram / 1e9):.2f} GB AVAILABLE ***")

    return device

def _is_cuda_oom(e: BaseException):
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and "out of memory" in msg

