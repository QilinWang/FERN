from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List, Tuple, Union, Callable, Dict, Optional, Any, Self, Literal,
    Annotated,
    Iterable,
    Set,
    Sequence,
)
from torch import Tensor  
import torch.nn.utils.parametrizations as parametrizations
import torch.nn.utils.parametrize as parametrize 
from metrics import rand_proj
import einops
from einops import rearrange 
import torch.nn.utils.spectral_norm as spectral_norm
from torch.cuda.amp import autocast 
import study.configs as configs 
from pydantic import (
    BaseModel, PositiveInt, ConfigDict, Field, computed_field, model_validator
)
import abc  

import copy
import numpy as np

import study.FERN_core as fcore  


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1,].repeat(1, 1,(self.kernel_size - 1) // 2)
        end = x[:, :, -1:,].repeat(1, 1,(self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        # x = x.permute(0, 2, 1)
        return x

def dynamic_size(data: torch.Tensor, last_dim_size: int) -> List[int]:
    return list(data.shape[:-1]) + [last_dim_size]

_EULER_GAMMA = 0.5772156649015329
_GUMBEL_STD  = np.pi / np.sqrt(6.0)

def sample_base(shape, device, dtype, kind="gauss", *, gen: torch.Generator|None=None):
    if kind == "gauss":
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)
    elif kind == "gumbel":
        # Standard Gumbel(0,1): g = -log(-log U)  (equivalently -log Exp(1))
        g = -torch.empty(shape, device=device, dtype=dtype).exponential_(generator=gen).log()
        # Optional: standardize to ~N(0,1)-like scale if you want parity with Gaussian
        g = (g - _EULER_GAMMA) / _GUMBEL_STD
        return g
    elif kind == "laplace":
        return torch.distributions.Laplace(loc=torch.tensor(0., device=device, dtype=dtype),
            scale=torch.tensor(1., device=device, dtype=dtype)).sample(shape, generator=gen)
    else:
        raise ValueError(f"Unknown base kind: {kind}")

# -------------------------------
# region bootstrap
def _indices_from_mode(T: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode not in _MODE_TO_SLICE:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {list(_MODE_TO_SLICE.keys())}")
    start, step = _MODE_TO_SLICE[mode]
    return torch.arange(start, T, step, device=device)

def _complement_indices(T: int, keep_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(T, dtype=torch.bool, device=keep_idx.device)
    mask[keep_idx] = False
    return mask.nonzero(as_tuple=False).squeeze(-1)

_MODE_TO_SLICE = {
    'odd':         (1, 2),  # 1,3,5,...
    'even':        (0, 2),  # 0,2,4,...
    'even_evens':  (0, 4),  # 0,4,8,...
    'even_odds':   (2, 4),  # 2,6,10,...
    'odd_evens':   (1, 4),  # 1,5,9,...
    'odd_odds':    (3, 4),  # 3,7,11,...
}

def make_keep_mask(x: torch.Tensor, start: int, step: int) -> torch.Tensor:
    """Binary mask with ones at positions [start::step] along the last dim."""
    if not isinstance(start, int) or not isinstance(step, int) or step <= 0:
        raise ValueError("`start` must be int and `step` a positive int.")
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., start::step] = True
    return mask


def keep_positions(
    x: torch.Tensor,
    keep_mode: Literal['odd','even','even_evens','even_odds','odd_odds','odd_evens'],
    return_mask: bool = False,
) -> torch.Tensor:
    """Keep ONLY the selected positions; others are zeroed. Optionally return mask."""
    if keep_mode not in _MODE_TO_SLICE:
        raise ValueError(f"keep_mode must be one of {list(_MODE_TO_SLICE)}")
    start, step = _MODE_TO_SLICE[keep_mode]
    mask = make_keep_mask(x, start, step)
    if return_mask:
        return mask
    return x * mask.to(dtype=x.dtype)

def drop_positions(
    x: torch.Tensor,
    drop_mode: Literal['odd','even','even_evens','even_odds','odd_odds','odd_evens'],
) -> torch.Tensor:
    """Zero OUT the selected positions; keep the rest."""
    if drop_mode not in _MODE_TO_SLICE:
        raise ValueError(f"drop_mode must be one of {list(_MODE_TO_SLICE)}")
    start, step = _MODE_TO_SLICE[drop_mode]
    y = x.clone()
    y[..., start::step] = 0
    return y

def bootstrap_keep(
    x: torch.Tensor,
    *,
    keep_mode: str = 'even',
    resample_from: Optional[str] = None,   # e.g. 'odd'; default = complement of keep_mode
    generator: Optional[torch.Generator] = None,
    patch_size: int = 24,
) -> torch.Tensor:
    """
    Bootstrap selected positions along the last dim, keeping others unchanged.

    Args:
        x: Tensor of shape (..., T). Works with any dtype/device.
        keep_mode: which positions to keep as-is (see _MODE_TO_SLICE).
        resample_from: which positions form the bootstrap pool and are
            *also* the positions to be replaced. If None, use the complement
            of keep_mode. Typical pairs: keep='even', resample_from='odd'.
        generator: optional torch.Generator for reproducible sampling.

    Returns:
        y: same shape as x. Positions in `keep_mode` are copied from x.
           Positions in `resample_from` are resampled (with replacement)
           from x restricted to that same set.
    """
    *lead, T = x.shape
    device = x.device

    keep_idx = _indices_from_mode(T, keep_mode, device)

    if resample_from is None:
        res_idx = _complement_indices(T, keep_idx)   # default: complement
    else:
        res_idx = _indices_from_mode(T, resample_from, device)

    if res_idx.numel() == 0:
        return x.clone()

    y = x.clone()

    # Process in contiguous blocks
    num_blocks = T // patch_size
    for b in range(num_blocks):
        start = b * patch_size
        end = start + patch_size
        block_res_idx = res_idx[(res_idx >= start) & (res_idx < end)]
        if block_res_idx.numel() == 0:
            continue

        pool = x.index_select(-1, block_res_idx)        # (..., K)
        K = block_res_idx.numel()

        draw = torch.randint(0, K, (K,), device=device, generator=generator)
        boot = pool.index_select(-1, draw)              # (..., K)

        y.index_copy_(-1, block_res_idx, boot)

    return y

# Convenience alias matching your original function:
def bootstrap_odds_keep_evens(x: torch.Tensor) -> torch.Tensor:
    return bootstrap_keep(x, keep_mode='even', resample_from='odd')
def bootstrap_even_keep_odd(x: torch.Tensor) -> torch.Tensor:
    return bootstrap_keep(x, keep_mode='odd', resample_from='even')
# endregion bootstrap

