import inspect
import random
from typing import Any, Mapping, Optional

import numpy as np
import torch


def random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_optimization(use_optim):
    if not use_optim:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "opt_einsum"):
            torch.backends.opt_einsum.enabled = False


def get_kwargs(fn, config_dict: Mapping[str, Any]):
    params = inspect.signature(fn).parameters
    return {
        k: v
        for k, v in config_dict.items()
        if k in params
    }
