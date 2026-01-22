import torch
import json
import os
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import traceback

from configs.model_config import model_config
from configs.path_config import path_config
from training_utils.device_utils import get_device

@contextmanager
def vram_ctx(
    tag,
    log_fp=model_config.cur_training_out_dir / 'vram-log.jsonl',
    step=None,
    epoch=None,
    device=get_device(),
    log_on_any_exception=True,
    empty_cache_on_oom=True,
    allow_async=False,          # will reduce performance impact if True, but may get incorrect metrics
                                # if False, will ensure that all parallel operations have exec order maintained
):
    """
    Context manager designed to wrap any cuda-based functions, and log VRAM stats.
    Especially useful when encountering cuda oom issues to help find potential vram leaks or areas needing garbage collection
    """
    log_fp = Path(log_fp)
    log_fp.parent.mkdir(exist_ok=True, parents=True)
    is_oom = False
    err_str = ""
    tb_str = ""

    # get cuda usage stats before executing what is under the ctx manager
    if not allow_async:
        torch.cuda.synchronize(device)
    alloc_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    peak_before = torch.cuda.max_memory_allocated(device)

    try:
        yield # execute what the ctx manager is wrapping

    except BaseException as e: # catch exception
        if _is_cuda_oom(e): # check if exception is cuda oom
            is_oom = True
            err_str = str(e)
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            if empty_cache_on_oom: # empty cache if crashed
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        elif log_on_any_exception: # optionally log non-cuda exceptions
            err_str = str(e)
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise

    finally: # if exec successfully, get and log cuda info
        if not allow_async:
            torch.cuda.synchronize(device)

        alloc_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        peak_after = torch.cuda.max_memory_allocated(device)

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
            "epoch": epoch,
            "step": step,
            "is_oom": is_oom,
            "alloc_mb_before": alloc_before / 1024**2,
            "reserved_mb_before": reserved_before / 1024**2,
            "peak_mb_before": peak_before / 1024**2,
            "alloc_mb_after": alloc_after / 1024**2,
            "reserved_mb_after": reserved_after / 1024**2,
            "peak_mb_after": peak_after / 1024**2,
            "delta_alloc_mb": (alloc_after - alloc_before) / 1024**2,
            "delta_reserved_mb": (reserved_after - reserved_before) / 1024**2,
            "delta_peak_mb": (peak_after - peak_before) / 1024**2,
            "error": err_str,
            "traceback": tb_str,
        }

        with open(log_fp, "a") as f:
            f.write(json.dumps(row) + "\n")