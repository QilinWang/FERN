from typing import TYPE_CHECKING

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import importlib
from colored import fg, attr
from pydantic import (
    BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator, PositiveInt,
)
from typing import (
    Callable, Optional, Union, List, Dict
)
import numpy as np
from pathlib import Path
import study.metrics as metrics 
from safetensors.torch import save_file, load_file 

import study.data_gen as data_gen
import study.data_mgr as data_mgr 
import study.configs as configs 

def cprint(text, color=None, bold=False):
    parts = []
    if color:
        parts.append(fg(color))
    if bold:
        parts.append(attr("bold"))
    parts.append(text)
    parts.append(attr("reset"))
    return "".join(parts)
 
 
class Checkpoint(BaseModel):
    """
    A Pydantic model for managing checkpoint configurations using modern,
    idiomatic patterns for computed fields and side effects.
    """

    experiment_name: str = Field(
        ..., min_length=1, description="The name of the experiment."
    )
    model_type: str = Field(
        ..., min_length=1, description="The type of the model (e.g., 'ResNet', 'BERT')."
    )
    base_results_dir: Path = Field(
        Path("results"), description="The base directory for all results."
    )

    # 1. Use @computed_field for clean, declarative derived data
    @computed_field
    def base_path(self) -> Path:
        """The derived base path for the specific experiment."""
        return self.base_results_dir / self.experiment_name

    @computed_field
    def models_path(self) -> Path:
        """The derived path for storing model checkpoints."""
        return self.base_path / "models"

    @computed_field
    def predictions_path(self) -> Path:
        """The derived path for storing model predictions."""
        return self.base_path / "predictions"

    # 2. Use model_post_init for logic that runs after the model is created
    def model_post_init(self, __context) -> None:
        """
        A method called automatically after Pydantic initialization to perform side effects,
        such as creating directories on the filesystem.
        """
        print(
            f"Post-init: Creating directory at {self.models_path} and {self.predictions_path}"
        )
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.predictions_path.mkdir(parents=True, exist_ok=True)

    @field_validator("experiment_name", "model_type")
    def name_must_not_be_whitespace(cls, v: str) -> str:
        """Ensure that string fields are not just whitespace."""
        if not v.strip():
            raise ValueError("Field must not be empty or contain only whitespace.")
        return v

def save_model(settings: Checkpoint, seed: int, model: nn.Module) -> Path:
    """Saves the model's state dictionary to a .safetensors file."""
    save_path = settings.models_path / f"{settings.model_type}_seed_{seed}.safetensors"
    save_file(model.state_dict(), save_path)
    print(f"    ✅ Model saved successfully to: {save_path}")
    return save_path


def load_model(
    settings: Checkpoint, seed: int, model: nn.Module, device: str
) -> Path:
    """Loads a model's state dictionary from a .safetensors file."""
    load_path = settings.models_path / f"{settings.model_type}_seed_{seed}.safetensors"
    if not load_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {load_path}.")

    state_dict = load_file(load_path, device=device)  # Good practice to specify device
    model.load_state_dict(state_dict)

    print(f"✅ Model loaded successfully from: {load_path}")
    return load_path

def save_predictions(
    settings: Checkpoint,
    seed: int,
    data_dict: Dict[str, torch.Tensor],
) -> Path:
    filename = f"predictions_seed_{seed}.safetensors"
    save_path = settings.predictions_path / filename
    save_file(data_dict, save_path)

    print(f"✅ Predictions saved successfully in safetensors format to: {save_path}")
    return save_path


def load_predictions(
    settings: "Checkpoint",
    seed: int,
    device: str, 
) -> Dict[str, torch.Tensor]:
    filename = f"predictions_seed_{seed}.safetensors"
    load_path = settings.predictions_path / filename

    if not load_path.exists():
        raise FileNotFoundError(f"No predictions file found at {load_path}")

    device = torch.device(device)
    loaded_tensors = load_file(load_path, device=device)  # Safest to load to CPU first

    print(f"✅ Predictions loaded successfully from: {load_path}")
    return loaded_tensors


def tde_to_seq(
    windows: torch.Tensor,  # Shape: (B, D, S) -> (Batch, Dimensions/Features, Sequence Length)
    stride: int = 1,
    offset: int = 0,
) -> np.ndarray:
    """
    Reconstructs a full time series from a batch of overlapping windows.
    Seq_full  = reconstruct_series_from_windows_torch(seq_windows,  stride=1, offset=0)
    Pred_full = reconstruct_series_from_windows_torch(pred_windows, stride=1, offset=seq_len)

    Args:
        windows: A tensor of sliding windows. Shape (B, D, S).
        stride: The step size between the start of each window.
        offset: An initial offset to apply to the timeline, useful for
                stitching prediction windows onto the end of input windows.

    Returns:
        A tensor of the reconstructed series. Shape (D, T).

    Example:
        window = [1, 2, 3]  ; start = 0 * stride = 0, end = 0 + 3 = 3
        It adds the window to acc at slice [0:3].
        acc is now [1, 2, 3, 0, 0]
        It increments count at slice [0:3].
        count is now [1, 1, 1, 0, 0]
    """
    B, D, S = windows.shape
    T = offset + (B - 1) * stride + S  # Calculate the full length of the series
    acc = np.zeros((D, T), dtype=windows.dtype)
    count = np.zeros((D, T), dtype=np.int32)

    for b in range(B):
        start = offset + b * stride
        end = start + S
        acc[:, start:end] += windows[b]
        count[:, start:end] += 1

    # Clamp count to 1 to avoid division by zero where there are no windows
    count = count  # .clamp(min=1)
    reconstructed_series = acc / count

    return reconstructed_series

def reconstruct_series_from_windows_torch(
        windows: torch.Tensor,   # (B, D, S)
        stride: int = 1,
        offset: int = 0,         # shift into the full timeline
        mode: str = 'average'    # 'average' or 'sum'
    ) -> torch.Tensor:
        """
        Reconstruct a series of length T = offset + (B-1)*stride + S from overlapping windows.

        For inputs (seq windows) use offset=0.
        For preds (forecast windows) use offset=seq_len.

        windows[b,:, :] covers positions
        [ offset + b*stride : offset + b*stride + S ]  

        Returns:
            full: Tensor of shape (D, T)

        Example: 
        x = [0, 1 ,2 ,3 ,4 ,5]
        seq_len = 3, pred_len = 2, stride = 1. Window = x[b: b+3]
        b | window = x[b: b+3]
        b = 0 -> [0,1,2]
        b = 1 -> [1,2,3]
        b = 2 -> [2,3,4]
        b = 3 -> [3,4,5]
        #B (batch size) = 4, S (sequence length) = 3.
        Length = offset(set to 0) + (#B - 1) * stride + S = 0 + (4 - 1) * 1 + 3 = 6

        Predictions:
        b | pred window = x[b+3 : b+3+2]
        b = 0 -> [3,4]
        b = 1 -> [4,5]
        #B_pred (batch size of predictions) = seq_len - pred_len + 1 = 2
        
        Length = offset(set to seq_len=3) + (#B_pred - 1) * stride + pred_len 
              = 3 + (2 - 1) * 1 + 2 = 6
        
        # reconstruct inputs and preds separately
        Seq_full  = reconstruct_series_from_windows_torch(seq_windows,  stride=1, offset=0,          mode='average')
        Pred_full = reconstruct_series_from_windows_torch(pred_windows, stride=1, offset=seq_len,    mode='average')

        # now stitch them: use the true inputs up to seq_len, then the preds
        full = Seq_full.clone()
        full[:, seq_len:] = pred_full[:, seq_len:]
        """
        B, D, S = windows.shape
        T = offset + (B - 1) * stride + S
        acc   = torch.zeros((D, T), dtype=windows.dtype, device=windows.device)
        count = torch.zeros((D, T), dtype=torch.int32, device=windows.device)
        for b in range(B):
            start = offset + b * stride
            acc[:, start:start+S]    += windows[b]
            count[:, start:start+S]  += 1

        if mode == 'average':
            count = count.clamp(min=1)
            return acc / count.to(windows.dtype)
        elif mode == 'sum':
            return acc
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    
class EarlyStopper(BaseModel):
    """A Pydantic model to manage early stopping logic."""

    patience: PositiveInt
    best_val: float = float("inf")
    counter: int = 0

    def check(self, val_obj: float) -> bool:
        """Checks the validation objective and updates the counter."""
        improved = val_obj < self.best_val

        log_parts = [cprint("Val obj", "light_magenta")]

        if improved:
            log_parts.extend(
                [
                    cprint("improved", "light_green", bold=True),
                    cprint(f"{self.best_val:.4f}", "white"),
                    cprint("→", "light_green"),
                    cprint(f"{val_obj:.4f}", "light_green", bold=True),
                    cprint("(saving checkpoint)", "light_blue"),
                ]
            )
            self.best_val = val_obj
            self.counter = 0

        else:
            log_parts.extend(
                [
                    cprint("did not improve", "light_red"),
                    cprint(f"({val_obj:.4f} > {self.best_val:.4f})", "white"),
                    cprint(
                        f"counter {self.counter + 1}/{self.patience}", "light_yellow"
                    ),
                ]
            )
            self.counter += 1

        final_log_message = " ".join(log_parts)
        print(f"    -> {final_log_message}")

        return improved

    @property
    def should_stop(self) -> bool:
        """Determines if training should stop."""
        return self.counter >= self.patience

    def reset(self):
        """Resets the stopper to its initial state."""
        self.best_val = float("inf")
        self.counter = 0

def plot_lorenz96_3d(trajectory, dim_to_plot=[0, 1, 2]):
    """
    Plots the Lorenz-96 system dynamics in 3D phase space.

    Args:
        trajectory (torch.Tensor): Trajectory of the Lorenz-96 system (shape: [steps, dim]).
        dim_to_plot (list): Dimensions to visualize in 3D (e.g., [0, 1, 2]).
    """
    if len(dim_to_plot) != 3:
        raise ValueError("dim_to_plot must contain exactly 3 dimensions for a 3D plot.")
    
    trajectory = trajectory.cpu().numpy()  # Convert to NumPy array for plotting
    x, y, z = trajectory[:, dim_to_plot[0]], trajectory[:, dim_to_plot[1]], trajectory[:, dim_to_plot[2]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, lw=0.8)
    ax.set_xlabel(f'Dimension {dim_to_plot[0]}')
    ax.set_ylabel(f'Dimension {dim_to_plot[1]}')
    ax.set_zlabel(f'Dimension {dim_to_plot[2]}')
    ax.set_title('3D Phase Space Trajectory')
    
    plt.show()
    
#region Parametrizations 

class SymmetricParametrization(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)
 
class MatrixExponentialParametrization(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)
    
class CayleyMapParametrization(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X): # (I + X)(I - X)^{-1}
        Id = self.Id.to(X.device)
        return torch.linalg.solve(Id - X, Id + X) 
    
class SkewParametrization(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)
    
class LowerTriangularParametrization(nn.Module):
    def forward(self, X):
        return torch.tril(X, diagonal=-1) + torch.eye(X.size(-1), device=X.device)
#endregion Parametrizations 

#region embedding 
def patch_embed(data, patch_len, stride):
    data = torch.nn.functional.pad(data, ( 0,  patch_len - stride,  ),  'replicate') # self.padding
    return data.unfold(dimension=-1, size=patch_len, step=stride)

def shift_embed(x, order=1):
    x = torch.nn.functional.pad(x, (order,0),  'replicate')
    return x[..., :-order]

def deri_embed(x, order=1):
    """ Compute discrete derivative of order n using rolling differences.  """
    x = torch.nn.functional.pad(x, (order,0),  'replicate')
    y = x.roll(-order, dims=-1)
    return (y - x)[..., :-order]
#endregion embedding

class SchedulerFactory:
    """
    A factory class for creating PyTorch learning rate schedulers based on a configuration object.
    """

    @staticmethod
    def build(cfg: configs.BaseTrainingConfig, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Builds and returns a learning rate scheduler based on the provided configuration.

        Args:
            cfg: A configuration object (e.g., a Pydantic model or an argparse Namespace)
                    that contains scheduler settings. It must have attributes like
                    'scheduler_type', 'learning_rate', 'warmup_epochs', etc.
            optimizer: The PyTorch optimizer to which the scheduler will be attached.

        Returns:
            A PyTorch learning rate scheduler instance, or None if the scheduler_type is 'none'
            or not recognized.
        """
        scheduler_type = getattr(cfg, 'scheduler_type', 'none')

        if scheduler_type == "plateau":
            print(
                f"[*] ReduceLROnPlateau scheduler enabled with patience={cfg.lr_scheduler_patience} and factor={cfg.lr_scheduler_factor}"
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=cfg.lr_scheduler_factor,
                patience=cfg.lr_scheduler_patience,
            )

        elif scheduler_type == "cosine":
            print(
                f"[*] Cosine scheduler with {cfg.warmup_epochs}-epoch warmup enabled."
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001, # Start LR at 0.1% of the initial LR
                total_iters=cfg.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs - cfg.warmup_epochs,
                eta_min=cfg.eta_min
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.warmup_epochs]
            )

        elif scheduler_type == "none":
            print("[*] No learning rate scheduler will be used.")
            return None

        else:
            print(f"[!] Warning: Scheduler type '{scheduler_type}' not recognized. No scheduler will be used.")
            return None