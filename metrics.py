from typing import TYPE_CHECKING, Optional, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn, Tensor
from typing import Tuple, Callable, List, Optional
import numpy as np
from pydantic import (
    BaseModel, Field, ConfigDict, model_validator, field_validator, computed_field,
)
from colored import fg, attr
import study.configs as configs
import study.utils as utils

class MetricMeter(BaseModel):
    """Tracks a single metric's statistics at batch level."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    )

    total: float = Field(default=0.0, ge=0.0)
    count: int = Field(default=0, ge=0)

    def add(self, x: float) -> None:
        if not np.isfinite(x):  # Don't add NaNs or Infs, but maybe log a warning
            print(f"Warning: {x} is not finite")
        self.total += x
        self.count += 1

    @computed_field
    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class MetricTracker(BaseModel):
    """Tracks a single metric's statistics at batch level""" 
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    )

    name: str
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    weight_backward: float
    weight_validate: float
    meter: MetricMeter = Field(default_factory=MetricMeter)

    def acc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw_result = self.fn(pred, target)
        self.meter.add(raw_result.item())
        return raw_result

    @property
    def mean(self) -> float:
        return self.meter.mean

    def reset(self):
        self.meter.reset()

    @field_validator("weight_backward", "weight_validate")
    def check_weights_are_non_negative(cls, v):
        if v < 0:
            raise ValueError("Metric weights cannot be negative")
        return v

    def __str__(self) -> str:
        return f"MetricMeter(mean={self.mean:.4f}, count={self.count})"


class ModeMetrics(BaseModel):
    """Manages all metric calculations for a single mode (e.g., 'train').""" 
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    )

    # --- INPUTS ---
    mode: str
    cfg: configs.BaseTrainingConfig
    global_std: np.ndarray
    is_training_mode: bool
    seed: int

    # --- STATE (Initialized by model_validator) ---
    mse_tracker: MetricTracker = None
    mae_tracker: MetricTracker = None
    huber_tracker: MetricTracker = None
    swd_tracker: MetricTracker = None
    ept_tracker: MetricTracker = None
    quantile70_tracker: MetricTracker = None
    quantile30_tracker: MetricTracker = None
    quantile85_tracker: MetricTracker = None
    quantile15_tracker: MetricTracker = None
    all_trackers: List[MetricTracker] = None
    running_obj: MetricMeter = None

    @model_validator(mode="after")
    def setup_trackers(self) -> "ModeMetrics":
        """Pydantic's equivalent of __init__ logic."""
        self.mse_tracker = MetricTracker(
            name="mse",
            fn=nn.MSELoss(),
            weight_backward=self.cfg.mse_weight_backward,
            weight_validate=self.cfg.mse_weight_validate,
        )
        self.mae_tracker = MetricTracker(
            name="mae",
            fn=nn.L1Loss(),
            weight_backward=self.cfg.mae_weight_backward,
            weight_validate=self.cfg.mae_weight_validate,
        )
        self.huber_tracker = MetricTracker(
            name="huber",
            fn=nn.HuberLoss(delta=1),
            weight_backward=self.cfg.huber_weight_backward,
            weight_validate=self.cfg.huber_weight_validate,
        )
        self.swd_tracker = MetricTracker(
            name="swd",
            # fn=SWDMetric(dim=self.cfg.pred_len, num_proj=self.cfg.num_proj_swd, seed=self.seed).to(self.cfg.device),
            fn=SWDMetric(
                feature_dim=self.cfg.channels,      # D, channels
                num_proj=self.cfg.num_proj_swd,     # e.g., 512 or 1500
                seed=self.seed,
                feature_axis=2,                     # D
                point_axis=1,                       # S
                use_proj=False,                  #  
                return_type="sq_swd2",
            ).to(self.cfg.device),
            weight_backward=self.cfg.swd_weight_backward,
            weight_validate=self.cfg.swd_weight_validate,
        )
        self.ept_tracker = MetricTracker(
            name="ept",
            fn=EPTMetric(global_std=self.global_std, device=self.cfg.device),
            weight_backward=0.0,
            weight_validate=0.0,
        )
        self.quantile85_tracker = MetricTracker(
            name="quantile85",
            fn=PinballLoss(quantile=0.85, reduction="mean"),
            weight_backward=self.cfg.quantile85_weight_backward,
            weight_validate=self.cfg.quantile85_weight_validate,
        )
        self.quantile15_tracker = MetricTracker(
            name="quantile15",
            fn=PinballLoss(quantile=0.15, reduction="mean"),
            weight_backward=self.cfg.quantile15_weight_backward,
            weight_validate=self.cfg.quantile15_weight_validate,
        )
        self.quantile70_tracker = MetricTracker(
            name="quantile70",
            fn=PinballLoss(quantile=0.70, reduction="mean"),
            weight_backward=self.cfg.quantile70_weight_backward,
            weight_validate=self.cfg.quantile70_weight_validate,
        )
        self.quantile30_tracker = MetricTracker(
            name="quantile30",
            fn=PinballLoss(quantile=0.30, reduction="mean"),
            weight_backward=self.cfg.quantile30_weight_backward,
            weight_validate=self.cfg.quantile30_weight_validate,
        )
        self.all_trackers = [
            self.mse_tracker,
            self.mae_tracker,
            self.huber_tracker,
            self.swd_tracker,
            self.ept_tracker,
            self.quantile85_tracker,
            self.quantile15_tracker,
            self.quantile70_tracker,
            self.quantile30_tracker,
        ]
        self.running_obj = MetricMeter()
        return self

    def update_and_calc_objective(self, pred: torch.Tensor, targets: torch.Tensor, cv=False
    ) -> torch.Tensor:
        if cv == True:
            if not self.is_training_mode and self.cfg.use_cross_validation_for_val:
                num_folds = self.cfg.val_cv_num_folds
                num_samples = self.cfg.val_cv_num_samples
                batch_size, seq_len, channels = pred.shape

                if seq_len >= num_folds:
                    generator = torch.Generator(device=pred.device)
                    # Use a consistent seed for reproducibility within the same validation run
                    generator.manual_seed(self.seed) 
                    
                    # 1. Create a binary mask of zeros with the same sequence length
                    mask = torch.zeros(seq_len, device=pred.device, dtype=pred.dtype)
                    
                    # 2. Get the indices for each fold
                    fold_indices = torch.chunk(torch.arange(seq_len, device=pred.device), chunks=num_folds)
                    
                    # 3. Randomly select which folds to KEEP
                    selected_fold_keys = torch.randperm(num_folds, generator=generator,device=pred.device)[:num_samples]
                    
                    # 4. Set the mask to 1 at the indices of the selected folds
                    for key in selected_fold_keys:
                        mask[fold_indices[key]] = 1
                    
                    # 5. Reshape mask to (1, seq_len, 1) to broadcast and apply it
                    mask = mask.view(1, seq_len, 1)
                    pred = pred * mask
                    targets = targets * mask
                    
        # --- Main objective calculation logic ---
        objective_terms = []
        for tracker in self.all_trackers:
            # Check if the metric is computationally expensive
            # is_costly = tracker.name in ("swd", "ept") #TODO
            is_costly = False

            # During training, skip expensive metrics if their weight is 0
            if self.is_training_mode and is_costly and tracker.weight_backward <= 0:
                raw_loss = torch.tensor(0.0, device=pred.device)
            else:
                raw_loss = tracker.acc_loss(pred, targets)

            # Determine the appropriate weight based on the mode (train/val)
            weight = (
                tracker.weight_backward
                if self.is_training_mode
                else tracker.weight_validate
            )

            if weight > 0:
                objective_terms.append(weight * raw_loss)

        if not objective_terms:
            return torch.tensor(
                0.0, device=pred.device, requires_grad=self.is_training_mode
            )

        total_objective = sum(objective_terms)
        self.running_obj.add(total_objective.item())
        return total_objective

    def report_running_obj(self) -> float:
        obj = self.running_obj.mean
        return obj

    def collect_metrics(self) -> "MetricsReport":
        metrics_report = MetricsReport(
            mode=self.mode,
            mse=self.mse_tracker.mean,
            mae=self.mae_tracker.mean,
            ept=self.ept_tracker.mean,
            huber=self.huber_tracker.mean,
            swd=self.swd_tracker.mean,
            quantile85=self.quantile85_tracker.mean,
            quantile15=self.quantile15_tracker.mean,
            quantile70=self.quantile70_tracker.mean,
            quantile30=self.quantile30_tracker.mean,
            running_obj=self.running_obj.mean,
        )
        return metrics_report

    def reset(self):
        for tracker in self.all_trackers:
            tracker.reset()
        self.running_obj.reset()


class MetricManager(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    )

    # --- INPUTS ---
    cfg: configs.BaseTrainingConfig
    global_std: np.ndarray
    seed: int

    # --- STATE (Initialized by model_validator) ---
    train: ModeMetrics = None
    val: ModeMetrics = None
    test: ModeMetrics = None

    best_val_metrics: Optional["MetricsReport"] = None
    best_test_metrics: Optional["MetricsReport"] = None

    @model_validator(mode="after")
    def setup_modes(self) -> "MetricManager":
        self.train = ModeMetrics(
            mode="train",
            cfg=self.cfg,
            global_std=self.global_std,
            is_training_mode=True,
            seed=self.seed,
        )
        self.val = ModeMetrics(
            mode="val",
            cfg=self.cfg,
            global_std=self.global_std,
            is_training_mode=False,
            seed=self.seed,
        )
        self.test = ModeMetrics(
            mode="test",
            cfg=self.cfg,
            global_std=self.global_std,
            is_training_mode=False,
            seed=self.seed,
        )
        return self

    def reset_all(self):
        self.train.reset()
        self.val.reset()
        self.test.reset()


class MetricsReport(BaseModel):
    """A structured container for the results of one epoch for one mode.""" 
    mode: str
    mse: float
    mae: float
    ept: float
    huber: float
    swd: float
    quantile70: float
    quantile30: float
    quantile85: float
    quantile15: float
    running_obj: float
    display_digits: int = 2
    display_digits_aux: int = 2

    def __str__(self) -> str:
        mse_str = f"{self.mse:.{self.display_digits}f}"
        mae_str = f"{self.mae:.{self.display_digits}f}"
        huber_str = f"{self.huber:.{self.display_digits}f}"
        swd_str = f"{self.swd:.{self.display_digits}f}"
        ept_str = f"{self.ept:.0f}"
        quantile70_str = f"{self.quantile70:.{self.display_digits_aux}f}"
        quantile30_str = f"{self.quantile30:.{self.display_digits_aux}f}"
        quantile85_str = f"{self.quantile85:.{self.display_digits_aux}f}"
        quantile15_str = f"{self.quantile15:.{self.display_digits_aux}f}"
        running_obj_str = f"{self.running_obj:.{self.display_digits_aux}f}"

        # 2. Then, pass the pre-formatted string to cprint for styling.
        parts = [
            f"{utils.cprint(self.mode.upper(), color='cyan', bold=True)}",
            f"MSE={utils.cprint(mse_str, color='light_magenta', bold=True)}",
            f"MAE={utils.cprint(mae_str, color='light_magenta', bold=True)}",
            f"HUB={utils.cprint(huber_str, color='light_magenta', bold=True)}",
            f"SWD={utils.cprint(swd_str, color='light_magenta', bold=True)}",
            f"EPT={utils.cprint(ept_str, color='light_magenta', bold=True)}",
            f"Q70={utils.cprint(quantile70_str, color='light_magenta', bold=True)}",
            f"Q30={utils.cprint(quantile30_str, color='light_magenta', bold=True)}",
            f"Q85={utils.cprint(quantile85_str, color='light_magenta', bold=True)}",
            f"Q15={utils.cprint(quantile15_str, color='light_magenta', bold=True)}",
            f"OBJ={utils.cprint(running_obj_str, color='light_magenta', bold=True)}",
        ]
        return " | ".join(parts)

import torch
import torch.nn as nn
from typing import Literal






# ---------------------------
# Projections
# ---------------------------

def rand_proj(input_dim: int, num_proj: int, seed: int, requires_grad: bool = False) -> torch.Tensor:
    """Generate a (input_dim x num_proj) matrix whose columns are unit-length projection directions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device) ; g.manual_seed(seed)
    proj = torch.randn(input_dim, num_proj, device=device, requires_grad=requires_grad, generator=g)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-9)  # normalize columns (unit vectors)
    return proj  # (F, L)

# ---------------------------
# Core SWD slice + sort
# ---------------------------


def _project_along_axis(x: torch.Tensor, proj_mat: torch.Tensor, feature_axis: int) -> torch.Tensor:
    """
    Project x along 'feature_axis' with proj_mat (F x L), returning a tensor where that axis is replaced by L.
    Works by moving 'feature_axis' to the last dim, matmul, then moving L back to that position.
    Intuitively, 
    Always ask yourself: "What is the set of things I'm trying to compare?" 
    The answer defines your "point cloud."
    The axis you sort over represents the different points in the cloud.
    The axis you destroy (project) represents the dimensions/features of each individual point.
    """
    # Bring 'feature_axis' to last
    x_feat_last = torch.moveaxis(x, feature_axis, -1)       # (..., F)
    z = x_feat_last @ proj_mat                              # (..., L)
    # Put L back where 'feature_axis' was
    z = torch.moveaxis(z, -1, feature_axis)                 # (..., L, ...)
    return z



class SWDMetric(nn.Module):
    """ 
    Sliced Wasserstein with axis control.

    Args:
      feature_dim:  size of the axis to project (F)
      num_proj:     number of projection directions (L)
      seed:         RNG seed for projections
      feature_axis: which axis in (y_pred, y_real) is the feature space to project (e.g., D)
      point_axis:   which axis indexes the set of points to sort (e.g., S, R, or B)
      use_proj: if False, use identity projections (no cross-feature mixing);
                    if True, use random unit directions in feature space
      return_type:  "sq_swd2" (mean squared), "swd2" (sqrt of mean squared), or "swd1" (mean abs)
 
    Typical choices:
      * Horizon-shape (per-sample): feature_axis=D, point_axis=S
      * Dataset cloud:               feature_axis=(S*D) after flatten, point_axis=B
      * Multi-realization per-x:     feature_axis=(S*D) or D, point_axis=R
    """
    def __init__(
        self,
        feature_dim: int,
        num_proj: int,
        seed: int,
        feature_axis: int,   # which axis in input tensors is the feature space to project
        point_axis: int,     # which axis in input tensors is the set of points to sort
        use_proj: bool = True, 
        return_type: Literal["sq_swd2", "swd2", "swd1"] = "sq_swd2",
    ):
        super().__init__()
        self.train_seed   = seed
        self.eval_seed    = seed
        self.num_proj     = num_proj
        self.feature_axis = feature_axis
        self.point_axis   = point_axis
        self.return_type  = return_type
        self.use_proj     = use_proj
        self.feature_dim  = feature_dim
        
        if use_proj:
            proj_mat = rand_proj(feature_dim, num_proj,
                                 seed=self.eval_seed, requires_grad=False) 
        else: 
            # assert num_proj == feature_dim, (
            #     f"In identity mode, num_proj ({num_proj}) must equal "
            #     f"feature_dim ({feature_dim})."
            # )
            # print(f"In identity mode, num_proj is reset to be equal to feature_dim.")
            proj_mat = torch.eye(feature_dim, feature_dim)
        self.register_buffer("proj_mat", proj_mat)
 
    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_real: same shape tensors (e.g., (B, S, D) etc.).
        We project along 'feature_axis' and sort along 'point_axis'.
        """
        assert y_pred.shape == y_real.shape, "Shapes of y_pred and y_real must match."
        # optionally resample during training
        if self.training and self.use_proj:
            self.train_seed += 1
            newP = rand_proj(self.feature_dim, self.num_proj,
                             seed=self.train_seed, requires_grad=False)
            self.proj_mat.copy_(newP)
  
        # 1) project along 'feature_axis'
        z_pred = _project_along_axis(y_pred, self.proj_mat, self.feature_axis)   # (..., L, ...)
        z_real = _project_along_axis(y_real, self.proj_mat, self.feature_axis)   # (..., L, ...)
        
        """
        Generic sliced 1-D OT:
        1) project along 'feature_axis' with proj_mat (F x L),
        2) sort along 'point_axis' (the set index),
        3) return sorted difference (pred - real).
        """
        # After projection, 'feature_axis' has been replaced by L (or F in identity mode).
        proj_axis  = self.feature_axis % z_pred.ndim
        point_axis = self.point_axis  % z_pred.ndim
        # guard: don't sort along the projections axis
        assert point_axis != proj_axis, "point_axis must index the set of points, not the projections."

        # 2) sort along the 'point_axis' (quantile matching)
        z_pred_sorted, _ = torch.sort(z_pred, dim=point_axis)
        z_real_sorted, _ = torch.sort(z_real, dim=point_axis)
        diff = z_pred_sorted - z_real_sorted

        # 3) reduce: mean over points, over projections, then over remaining batch-like axes
        # adjust projection axis index if point_axis < proj_axis (sorting removes/permutes nothing but keep indices aligned)
        if self.return_type == "sq_swd2":
            val = diff.pow(2).mean(dim=point_axis)
            # if proj_axis was after point_axis, its index shifts left by 1 after the reduction
            # pa = point_axis
            # pr = proj_axis - (proj_axis > pa)
            # val = val.mean(dim=pr)
            swd = val.mean()
        elif self.return_type == "swd2":
            val = diff.pow(2).mean(dim=point_axis).sqrt()
            # pr = proj_axis - (proj_axis > point_axis)
            # val = val.mean(dim=pr)
            swd = val.mean()
        elif self.return_type == "swd1":
            val = diff.abs().mean(dim=point_axis)
            # pr = proj_axis - (proj_axis > point_axis)
            # val = val.mean(dim=pr)
            swd = val.mean()
        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        return swd

"""
Imagine diff has axes [ ... point_axis ..., ... proj_axis ... ].

First, you reduce (.mean) over point_axis.
Now every axis that comes after point_axis in the order shifts left by one.

If proj_axis was to the right of point_axis, its index is now off by -1.

So proj_axis - (proj_axis > point_axis) is just a quick way to fix the index.

Example:

Suppose diff.shape = (B, S, L) with axes (0=B, 1=S=point_axis, 2=L=proj_axis).

You reduce over axis=1 (S). New shape (B, L).

Now the projections axis is at index 1, not 2.

proj_axis=2, point_axis=1 ⇒ proj_axis > point_axis = True ⇒ 2-1=1. Perfect.
"""
# usage
# Inputs y_* shaped (B, S, D). Project channels (feature_axis=2), sort horizon (point_axis=1):
# metric = SWDMetric(feature_dim=D, num_proj=1500, seed=0, feature_axis=2, point_axis=1, return_type="sq_swd2")

# Univariate (D=1), exact 1D OT (no need for many projections)
# metric = SWDMetric(feature_dim=1, num_proj=1, seed=0, feature_axis=2, point_axis=1, return_type="sq_swd2")
# score = metric(y_pred, y_true)



class TargetStdMetric(nn.Module):
    """Compute the standard deviation of target data."""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        # y_real shape: [batch_size, seq_len, channels] or [batch_size, channels, seq_len]
        return torch.std(y_real, dim=-1).mean()  # Average across batch and channels

class EPTMetric(nn.Module):
    """
    Effective Prediction Time:
        • global_std: 1D tensor [D]  (per-channel threshold)
        • y_pred, y_true: [B, S, D]
    Returns a scalar: mean T_{b,d} over all sequences and channels.
    """
    def __init__(self, global_std: np.ndarray, device: torch.device):
        super().__init__()
        
        # shape to [1, D, 1] so it broadcasts against [B, D, S]
        # print(f"EPT global_std shape: {global_std.shape}")
        self.register_buffer("thr", torch.tensor(global_std, dtype=torch.float32, device=device)[None, :, None])

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.permute(0, 2, 1)
        y_real = y_real.permute(0, 2, 1)
        # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_real.shape}, thr shape: {self.thr.shape}")
        # print(f"self.thr: {self.thr}")
        err = (y_pred - y_real).abs()                 # [B,D,S]
        crossed = err > self.thr                      # [B,D,S] bool
        # first index along S where crossed is True; if never, returns 0
        first = (crossed.float()).argmax(-1)          # [B,D] int64
        # first index along S where value==1 (PyTorch argmax returns first max)
        never = (~crossed.any(-1))                    # [B,D] bool
        # If any time‑step is True, it returns True; otherwise False. → [B,D].
        first[never] = y_pred.size(-1)                # set to S when never crossed
        return first.float().mean()                   # scalar like MSE/MAE

import torch
import torch.nn as nn
from typing import Iterable, Literal
class PinballLoss(nn.Module):
    r"""
    Quantile (pinball) loss for data shaped (B, S, D).

    Canonical definition (always ≥ 0):
        Let u = y_true - y_pred and τ ∈ (0,1).
        ρ_τ(u) = max( τ·u, (τ-1)·u )

    Equivalent piecewise with error = y_pred - y_true:
        if error ≥ 0 (over-predict):    (1-τ)·error
        else (under-predict):           -τ·error

    Args:
        quantile: τ in (0, 1)
        reduction: "mean" | "sum" | "none"
    """
    def __init__(self, quantile: float, reduction: Literal["mean","sum","none"]="mean"):
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile τ must be in (0, 1).")
        self.register_buffer("q", torch.tensor(quantile, dtype=torch.float32), persistent=False)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape or y_pred.ndim != 3:
            raise ValueError(f"y_pred and y_true must both be (B, S, D); got {y_pred.shape=} {y_true.shape=}")
        q = self.q.to(device=y_pred.device, dtype=y_pred.dtype)

        # canonical form (diff = y_true - y_pred)
        diff = y_true - y_pred                     # (B, S, D)
        loss = torch.maximum(q * diff, (q - 1) * diff)

        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss  # (B, S, D)
    
class CRPSApprox(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1, 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: [B, num_quantiles, S, D]  predicted quantiles
        y_true: [B, S, D]                 true values
        """
        y_true = y_true.unsqueeze(1)  # [B,1,S,D]
        error = y_true - y_pred       # note: consistent sign with pinball def
        loss = torch.maximum(self.quantiles * error, (self.quantiles - 1) * error)
        return loss.mean()            # scalar CRPS estimate
