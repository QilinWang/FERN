from typing import TYPE_CHECKING, Optional 
import torch
import torch.nn as nn
from metrics import SWDMetric, EPTMetric
from FERN import FERN 
import numpy as np
import study.other_models as other_models
from other_models.PatchTST import Model as PatchTST
from other_models.DLinear import Model as DLinear
from other_models.TimeMixer import Model as TimeMixer
import time
from typing import Optional, List, Dict, Tuple, Callable, Union
import study.configs as configs
import study.data_mgr as data_mgr 
from pydantic import (
    Field, BaseModel, computed_field, model_validator, field_validator, ConfigDict,PositiveInt,
)
from pathlib import Path
from safetensors.torch import save_file, load_file
from colored import fg, attr
import matplotlib.pyplot as plt

import study.metrics as metrics 
import study.utils as utils  

from torch.optim.lr_scheduler import (
    _LRScheduler,  ReduceLROnPlateau, LinearLR, CosineAnnealingLR, SequentialLR,
)
import pandas as pd
pd.set_option("display.float_format", "{:.2f}".format)
from loguru import logger
from IPython.display import display, Markdown  

def bootstrap_odds_keep_evens(x: torch.Tensor) -> torch.Tensor:
    """
    x : (..., T) tensor
        Last (time) axis length T may be even or odd.
        Works on any device / dtype; gradients flow through the gather.

    returns
    -------
    y : same shape as x
        Even indices are copied from x; odd indices are resampled
        (with replacement) from the set of odd positions.
    """
    *lead, T = x.shape
    device = x.device

    # 1) build even / odd index tensors once ---------------------------------
    even_idx = torch.arange(0, T, 2, device=device)           # 0,2,4,...
    odd_idx  = torch.arange(1, T, 2, device=device)           # 1,3,5,...

    # 2) gather even positions exactly --------------------------------------
    even_part = x.index_select(-1, even_idx)                  # (...,  ceil(T/2))

    # 3) bootstrap (with replacement) from odd positions --------------------
    k = odd_idx.numel()                                       # = ⌈T/2⌉ - 1
    draw_idx = torch.randint(0, k, (k,), device=device)       # shape (k,)
    odd_part = x.index_select(-1, odd_idx)                    # original odds
    odd_boot = odd_part.index_select(-1, draw_idx)            # resampled odds

    # 4) interleave even and bootstrapped odd back together -----------------
    # create an empty tensor and scatter parts into the correct slots
    y = torch.empty_like(x)
    y.index_copy_(-1, even_idx, even_part)
    y.index_copy_(-1, odd_idx,  odd_boot)
    return y

class ModelTrainer:
    def __init__(self, config: configs.BaseTrainingConfig, 
                 data_bundle: data_mgr.DataBundle, seed: int):
        self.cfg = config
        assert isinstance(self.cfg, configs.BaseTrainingConfig), (
            f"Config must be a ModelConfigType, got {type(self.cfg)}"
        )
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = ModelFactory(self.cfg)
        
        use_fused = torch.cuda.is_available() and self.cfg.device.startswith('cuda')

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate, fused=use_fused , weight_decay=0.000, # 0.01
        )
        # torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.cfg.learning_rate,
        #     weight_decay=0.0,
        # )
        
        
        self.scheduler = utils.SchedulerFactory.build(self.cfg, self.opt) 
        
        if data_bundle.no_scale_val_test:
            global_std = data_bundle.primary_global_std_dev
        else:
            global_std = np.array([1.0] * data_bundle.primary_global_std_dev.shape[0])
        self.metric_mgr = metrics.MetricManager(
            cfg=self.cfg, global_std=global_std, seed=self.seed
        )

        self.early_stopper = utils.EarlyStopper(patience=self.cfg.patience)
        
        self.checkpoint = utils.Checkpoint(
            experiment_name=data_bundle.experiment_name,
            model_type=self.cfg.model_type,
        )

    def train_model(self, data_bundle):  
            
        self.early_stopper.reset()  # work across epochs

        for epoch in range(self.cfg.epochs):
            
            # --- TRAINING PHASE ---
            self.model.train()
            self.metric_mgr.train.reset()
            # self.metric_mgr.train.swd_tracker.fn.resample_projections()
            with torch.set_grad_enabled(True):
                for batch_x, batch_y in data_bundle.dataloaders.train_loader:
                    b = BatchBundle(
                        model=self.model,   
                        opt=self.opt,   
                        mgr=self.metric_mgr,   
                        checkpoint=self.checkpoint,
                        earlystopper=self.early_stopper,
                        mode="train",
                        epoch=epoch,
                        x=batch_x.to(data_bundle.device, non_blocking=True),  # Assign new data
                        y=batch_y.to(data_bundle.device, non_blocking=True),  # Assign new data
                        seed=self.seed,
                        scheduler=self.scheduler,
                        no_scale=data_bundle.no_scale, 
                        no_scale_val_test=data_bundle.no_scale_val_test,
                        scalar_mean=data_bundle.scalar_mean,
                        scalar_std=data_bundle.scalar_std,
                    )
                    b = (
                        b | batch_train_forward | batch_train_metrics 
                        | batch_train_backward | batch_train_report
                    )
            b = b | phase_train

            # --- VALIDATION PHASE ---
            self.model.eval()
            self.metric_mgr.val.reset()
            with torch.set_grad_enabled(False):
                for batch_x, batch_y in data_bundle.dataloaders.val_loader:
                    b = BatchBundle(
                        model=self.model,  # Pass by reference (fast) 
                        opt=self.opt,  # Pass by reference (fast)
                        mgr=self.metric_mgr,  # Pass by reference (fast)
                        checkpoint=self.checkpoint,
                        earlystopper=self.early_stopper,
                        mode="val",
                        epoch=epoch,
                        x=batch_x.to(data_bundle.device, non_blocking=True),  # Assign new data
                        y=batch_y.to(data_bundle.device, non_blocking=True),  # Assign new data
                        seed=self.seed,
                        scheduler=self.scheduler,
                        no_scale=data_bundle.no_scale, 
                        no_scale_val_test=data_bundle.no_scale_val_test,
                        scalar_mean=data_bundle.scalar_mean,
                        scalar_std=data_bundle.scalar_std,
                    )
                    
                    b = b | batch_val_forward | batch_val_metrics | batch_val_report
            b = b | phase_val
            if b.earlystopper.should_stop:
                print(f"Early stopping at epoch {b.epoch}")
                break

            # --- TESTING PHASE ---
            self.model.eval()
            self.metric_mgr.test.reset()
            with torch.set_grad_enabled(False):
                for batch_x, batch_y in data_bundle.dataloaders.test_loader:
                    b = BatchBundle(
                        model=self.model,   
                        opt=self.opt,   
                        mgr=self.metric_mgr,  
                        checkpoint=self.checkpoint,
                        earlystopper=self.early_stopper,
                        mode="test",
                        epoch=epoch,
                        x=batch_x.to(data_bundle.device, non_blocking=True),  # Assign new data
                        y=batch_y.to(data_bundle.device, non_blocking=True),  # Assign new data
                        seed=self.seed,
                        scheduler=self.scheduler,
                        no_scale=data_bundle.no_scale,  
                        no_scale_val_test=data_bundle.no_scale_val_test,
                        scalar_mean=data_bundle.scalar_mean,
                        scalar_std=data_bundle.scalar_std,
                    )
                    b = b | batch_test_forward | batch_test_metrics | batch_test_report
            b = b | phase_test
            
        # --- FINAL TESTING PHASE ---
        print(" ---- TESTING FINAL MODEL ----") 
        load_path = utils.load_model(
            self.checkpoint, self.seed, self.model, device=self.cfg.device
        )
        self.model.eval()
        self.metric_mgr.test.reset()
        pred_lst = []
        target_lst = []
        with torch.set_grad_enabled(False):
            for batch_x, batch_y in data_bundle.dataloaders.test_loader:
                b = BatchBundle(
                    model=self.model,   
                    opt=self.opt,   
                    mgr=self.metric_mgr,  
                    checkpoint=self.checkpoint,
                    earlystopper=self.early_stopper,
                    mode="test",
                    epoch=epoch,
                    x=batch_x.to(data_bundle.device, non_blocking=True),  
                    y=batch_y.to(data_bundle.device, non_blocking=True),  
                    seed=self.seed,
                    scheduler=self.scheduler,
                    no_scale=data_bundle.no_scale, 
                    no_scale_val_test=data_bundle.no_scale_val_test,
                    scalar_mean=data_bundle.scalar_mean,
                    scalar_std=data_bundle.scalar_std,
                )
                b = b | batch_test_forward | batch_test_metrics | batch_test_report
                pred_lst.append(b.pred.cpu().detach())
                target_lst.append(b.y.cpu().detach())

        b.pred_tde = torch.cat(pred_lst, dim=0).permute(0, 2, 1).numpy()
        b.truth_tde = torch.cat(target_lst, dim=0).permute(0, 2, 1).numpy()

        b = b | phase_final

        return b

#region BatchBundle
class BatchBundle(BaseModel):
    def __or__(self, fn):
        return fn(self)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    )

    model: torch.nn.Module
    opt: torch.optim.Optimizer
    mgr: metrics.MetricManager
    earlystopper: BaseModel
    checkpoint: utils.Checkpoint
    mode: str
    epoch: int 
    no_scale: bool 
    no_scale_val_test: bool
    scalar_mean: np.ndarray | torch.Tensor | None = None
    scalar_std:  np.ndarray | torch.Tensor | None = None
    scalar_min: np.ndarray | torch.Tensor | None = None
    scalar_max: np.ndarray | torch.Tensor | None = None
    
    x: torch.Tensor = Field(..., description="batch-level inputs in [B S D] form")
    y: torch.Tensor = Field(..., description="batch-level targets in [B S D] form")
    
    pred: torch.Tensor = Field(None, description="batch-level preds in [B S D] form")
    stage_objective: torch.Tensor = None
     
    
    seed: int
    scheduler: _LRScheduler | ReduceLROnPlateau | LinearLR | CosineAnnealingLR | SequentialLR | None = None

    

    train_metrics: metrics.MetricsReport = None
    val_metrics: metrics.MetricsReport = None
    test_metrics: metrics.MetricsReport = None

    pred_tde: Optional[np.ndarray] = Field(None, description="Test preds in [B S D] form")
    truth_tde: Optional[np.ndarray] = Field(None, description="Test targets in [B S D] form")

    pred_seq: Optional[np.ndarray] = Field(None, description="Test preds in [D, Full_length] form")
    truth_seq: Optional[np.ndarray] = Field(None, description="Test targets in [D, Full_length] form")
    
     

    def check_pred_exists(self):
        """Raises an error if the 'pred' field has not been populated."""
        if self.pred is None:
            raise ValueError(
                "'pred' field is None. Ensure 'stage_forward' has been run "
                "before this step in the pipeline."
            )
#endregion BatchBundle

#region BatchBundle Functions
def _assert_lengths(b):
    exp_seq = b.model.config.seq_len
    exp_pred = b.model.config.pred_len
    assert b.x.shape[1] == exp_seq,  f"x length {b.x.shape[1]} ≠ cfg.seq_len {exp_seq}"
    assert b.y.shape[1] == exp_pred, f"y length {b.y.shape[1]} ≠ cfg.pred_len {exp_pred}"
    assert b.pred.shape[1] == exp_pred, f"pred length {b.pred.shape[1]} ≠ cfg.pred_len {exp_pred}"


def batch_train_forward(b: BatchBundle) -> BatchBundle:
    model_output = b.model(b.x)
    b.pred = b.model._extract_prediction(model_output)
    b.check_pred_exists()
    _assert_lengths(b)        # <--- add
    return b

def inv_standardize(x: torch.Tensor, scalar_mean, scalar_std) -> torch.Tensor:
    if scalar_mean is None or scalar_std is None:
        raise RuntimeError("inv_standardize called without scalar_mean/std.")
    mean = torch.as_tensor(scalar_mean, dtype=x.dtype, device=x.device)
    std  = torch.as_tensor(scalar_std,  dtype=x.dtype, device=x.device)
    # [D] -> [..., D] broadcasting
    while mean.dim() < x.dim():
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return x * std + mean
    
def batch_val_forward(b: BatchBundle) -> BatchBundle:
    model_output = b.model(b.x, update=False)
    b.pred = b.model._extract_prediction(model_output)
    if b.no_scale_val_test and not b.no_scale:
        b.pred = inv_standardize(b.pred, b.scalar_mean, b.scalar_std)
        b.y = inv_standardize(b.y, b.scalar_mean, b.scalar_std)
    b.check_pred_exists() 
    _assert_lengths(b)        # <--- add
    return b


def batch_test_forward(b: BatchBundle) -> BatchBundle:
    b.model.eval()
    model_output = b.model(b.x, update=False)
    b.pred = b.model._extract_prediction(model_output)
    if b.no_scale_val_test and not b.no_scale:
        b.pred = inv_standardize(b.pred, b.scalar_mean, b.scalar_std) #TODO
        b.y = inv_standardize(b.y, b.scalar_mean, b.scalar_std)
    b.check_pred_exists()
    _assert_lengths(b)        # <--- add
     
    return b


def batch_train_metrics(b: BatchBundle) -> BatchBundle:
    b.stage_objective = b.mgr.train.update_and_calc_objective(b.pred, b.y, cv=False)
    return b


def batch_val_metrics(b: BatchBundle) -> BatchBundle:
    b.stage_objective = b.mgr.val.update_and_calc_objective(b.pred, b.y, cv=True) 
    return b


def batch_test_metrics(b: BatchBundle) -> BatchBundle:
    b.stage_objective = b.mgr.test.update_and_calc_objective(b.pred, b.y, cv=False)
    return b


def batch_train_backward(b: BatchBundle) -> BatchBundle:
    b.opt.zero_grad(set_to_none=True) #!!! close for optimization
    b.stage_objective.backward()
    # max_norm = 2.0
    # total_norm = torch.nn.utils.clip_grad_norm_(b.model.parameters(), max_norm)
    # print(f"Total norm: {total_norm}")

    # print(f"b.model.model.fixed_params['complex_a'].grad is not None: {b.model.model.fixed_params['complex_a'].grad is not None}")
    # print(f"b.model.model.fixed_params['complex_b'].grad is not None: {b.model.model.fixed_params['complex_b'].grad is not None}")
    # print(f"b.model.model.k.a is b.model.model.fixed_params['complex_a']: {b.model.model.k.a is b.model.model.fixed_params['complex_a']}")  # should be True

    b.opt.step()
    return b


def batch_train_report(b: BatchBundle) -> BatchBundle:
    b.train_metrics = b.mgr.train.collect_metrics()
    return b


def batch_val_report(b: BatchBundle) -> BatchBundle:
    b.val_metrics = b.mgr.val.collect_metrics()
    return b


def batch_test_report(b: BatchBundle) -> BatchBundle:
    b.test_metrics = b.mgr.test.collect_metrics()
    # torch.cuda.empty_cache() #!!! close for optimization
    return b


def phase_train(b: BatchBundle) -> BatchBundle:
    b.train_metrics = b.mgr.train.collect_metrics()
    running_obj = b.mgr.train.report_running_obj()
    print(f"Epoch {b.epoch + 1} | {b.train_metrics}")
    return b


def phase_val(b: BatchBundle) -> BatchBundle:
    b.val_metrics = b.mgr.val.collect_metrics()
    running_obj = b.mgr.val.report_running_obj()
    print(f"Epoch {b.epoch + 1} | {b.val_metrics}")
    if b.scheduler and not isinstance(b.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        b.scheduler.step()
    elif b.scheduler:
        b.scheduler.step(b.stage_objective)
    else:
        pass
    # if b.scheduler:
    #     b.scheduler.step(b.stage_objective)
    if b.epoch > 2:
        improved: bool = b.earlystopper.check(b.mgr.val.report_running_obj())
        if improved:
            save_path = utils.save_model(b.checkpoint, b.seed, b.model)
            # b.save_model()
            b.mgr.best_val_metrics = b.mgr.val.collect_metrics()

    return b


def phase_test(b: BatchBundle) -> BatchBundle:
    b.test_metrics = b.mgr.test.collect_metrics()
    running_obj = b.mgr.test.report_running_obj()
    print(f"Epoch {b.epoch + 1} | {b.test_metrics}")
    return b


def phase_final(b: BatchBundle) -> BatchBundle:
    b.pred_seq = utils.tde_to_seq(b.pred_tde, stride=1, offset=0)
    b.truth_seq = utils.tde_to_seq(b.truth_tde, stride=1, offset=0)
    b.test_metrics = b.mgr.test.collect_metrics()
    running_obj = b.mgr.test.report_running_obj()
    b.mgr.best_test_metrics = b.mgr.test.collect_metrics()
    print(f"Best Test ReRun | {b.model.config.model_type} | {b.test_metrics}")
    return b
#endregion BatchBundle Functions

#region ModelAdapter
class ModelAdapter(nn.Module):
    """A single, generic adapter that wraps any model."""

    def __init__(self, config: configs.BaseTrainingConfig, underlying_model: nn.Module):
        super().__init__()
        self.config = config
        self.model = underlying_model

    def _extract_prediction(
        self, raw_output: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """Internal method to handle the different output formats."""
        sig = self.config.output_signature
        if sig == "tensor":
            if not isinstance(raw_output, torch.Tensor):
                raise TypeError(
                    f"Model was configured for 'tensor' output but returned {type(raw_output)}"
                )
            return raw_output
        elif sig == "dict:pred":
            if not isinstance(raw_output, dict):
                raise TypeError(
                    f"Model was configured for 'dict:pred' output but returned {type(raw_output)}"
                )
            pred = raw_output.get("pred")
            if pred is None:
                raise ValueError("Model output is dict but has no 'pred' key")
            return pred
        # Add other cases like 'obj.pred' or 'tuple[0]' if needed
        else:
            raise NotImplementedError(f"Output signature '{sig}' is not implemented.")

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Calls the underlying model's forward pass using the signature
        specified in the Pydantic configuration.
        """
        sig = self.config.forward_signature

        if sig == "x,update":  # For models like FERN
            return self.model(x, update=update)

        elif sig == "x,none,none,none":  # For models like PatchTST
            return self.model(x, None, None, None)

        elif sig == "x":  # For models like DLinear
            return self.model(x)

        elif sig == "naive_repeat":  # The model itself is just this adapter.
            last = x[:, -1:, :]
            return last.repeat(1, self.config.pred_len, 1)

        else:
            raise NotImplementedError(f"Forward signature '{sig}' is not implemented.")
#endregion ModelAdapter

#region ModelFactory
def ModelFactory(config: configs.BaseTrainingConfig) -> ModelAdapter:
    """wraps underlying model for a unified interface"""

    model_type = config.model_type

    if model_type == "FERN":
        underlying_model = FERN(config)
    elif model_type == "PatchTST":
        underlying_model = PatchTST(config)
    elif model_type == "DLinear":
        underlying_model = DLinear(config)
    elif model_type == "TimeMixer":
        underlying_model = TimeMixer(config)
    # elif model_type == "HoloMorph":
    #     underlying_model = HoloMorph(config)
    elif model_type == "Attraos":
        from study.other_models.Attraos import Model as Attraos
        underlying_model = Attraos(config) 
    elif model_type == "Koopa":
        from study.other_models.Koopa import Model as Koopa
        underlying_model = Koopa(config)
    elif model_type == "naive":  # For the naive case, the adapter IS the model.
        underlying_model = (
            nn.Identity()
        )  # Note: We pass nn.Identity() as a placeholder, it won't be used.
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelAdapter(config=config, underlying_model=underlying_model).to(
        config.device
    )
#endregion ModelFactory

def debug_check_model_gradients(model):
    grads = [
        p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
    ]
    if not grads or any(g is None for g in grads):
        raise RuntimeError(
            "[Train Batch] No gradients found on any parameters after backward!"
        )
    print(f"  [Train Batch] Max abs grad among params: {max(grads):.6f}")



#region SingleRunResults
class SingleRunResults(BaseModel):
    """A lightweight "handle" representing the results of a single training run."""

    seed: int
    training_time: float = None

    data_bundle: data_mgr.DataBundle
    final_bundle: BatchBundle = None

    @property
    def best_val_metrics(self) -> torch.Tensor:
        return self.final_bundle.mgr.best_val_metrics

    @property
    def best_test_metrics(self) -> torch.Tensor:
        return self.final_bundle.mgr.best_test_metrics

    def save_tdes(self):
        filename_pred = f"pred_tde_seed_{self.seed}_pred"
        filename_truth = f"pred_tde_seed_{self.seed}_truth"
        save_path_pred = self.final_bundle.checkpoint.predictions_path / filename_pred
        save_path_truth = self.final_bundle.checkpoint.predictions_path / filename_truth
        np.save(save_path_pred, self.final_bundle.pred_tde)
        np.save(save_path_truth, self.final_bundle.truth_tde)
        print(f"✅ TDE predictions saved as NumPy array to: {save_path_pred}")
        print(f"✅ TDE targets saved as NumPy array to: {save_path_truth}")

    def save_raw_seqs(self):
        filename_pred = f"pred_seq_seed_{self.seed}_pred"
        filename_truth = f"pred_seq_seed_{self.seed}_truth"
        save_path_pred = self.final_bundle.checkpoint.predictions_path / filename_pred
        save_path_truth = self.final_bundle.checkpoint.predictions_path / filename_truth
        np.save(save_path_pred, self.final_bundle.pred_seq)
        np.save(save_path_truth, self.final_bundle.truth_seq)
        print(f"✅ Sequences predictions saved as NumPy array to: {save_path_pred}")
        print(f"✅ Sequences targets saved as NumPy array to: {save_path_truth}")

    def load_tdes(self) -> Tuple[np.ndarray, np.ndarray]:
        filename_pred = f"pred_tde_seed_{self.seed}_pred.npy"
        filename_truth = f"pred_tde_seed_{self.seed}_truth.npy"
        load_path_pred = self.final_bundle.checkpoint.predictions_path / filename_pred
        load_path_truth = self.final_bundle.checkpoint.predictions_path / filename_truth

        if not load_path_pred.exists():
            raise FileNotFoundError(
                f"No TDE predictions file found at {load_path_pred}"
            )
        if not load_path_truth.exists():
            raise FileNotFoundError(f"No TDE targets file found at {load_path_truth}")

        data_pred = np.load(load_path_pred)
        data_truth = np.load(load_path_truth)
        print(f"✅ TDE predictions loaded from: {load_path_pred}")
        print(f"✅ TDE targets loaded from: {load_path_truth}")
        return data_pred, data_truth

    def load_raw_seqs(self) -> Tuple[np.ndarray, np.ndarray]:
        filename_pred = f"pred_seq_seed_{self.seed}_pred.npy"
        filename_truth = f"pred_seq_seed_{self.seed}_truth.npy"
        load_path_pred = self.final_bundle.checkpoint.predictions_path / filename_pred
        load_path_truth = self.final_bundle.checkpoint.predictions_path / filename_truth

        if not load_path_pred.exists():
            raise FileNotFoundError(
                f"No sequence predictions file found at {load_path_pred}"
            )
        if not load_path_truth.exists():
            raise FileNotFoundError(
                f"No sequence targets file found at {load_path_truth}"
            )

        data_pred = np.load(load_path_pred)
        data_truth = np.load(load_path_truth)
        print(f"✅ Sequences predictions loaded from: {load_path_pred}")
        print(f"✅ Sequences targets loaded from: {load_path_truth}")
        return data_pred, data_truth

    def save_model(self):
        utils.save_model(self.final_bundle.checkpoint, self.seed, self.final_bundle.model)

    def load_model(self):
        utils.load_model(
            self.final_bundle.checkpoint,
            self.seed,
            self.final_bundle.model,
            device=self.data_bundle.device,
        )

    def report_training_time(self):  # Changed to a method
        minutes = int(self.training_time // 60)
        seconds = self.training_time % 60
        print(f"Time taken: {minutes}m {seconds:.2f}s")

    def plot_seq(self, feature_idx: int = 0):
        # plt.style.use("seaborn-v0_8-whitegrid")
        # plt.rcParams.update({
        #     "axes.edgecolor": "0.7",
        #     "axes.labelcolor": "0.5",
        #     "xtick.color":   "0.5",
        #     "ytick.color":   "0.5",
        #     "grid.color":    "0.8",
        #     "text.color":    "0.4",
        #     "font.size":      9,        # ≈ 9-pt in the PDF (AAAI minimum)
        # })

        # # ---------- figure ----------
        # fig, ax = plt.subplots(figsize=(6.9, 3.5))   # AAAI single-column width

        # # Prediction (dashed)
        # ax.plot(
        #     self.final_bundle.pred_seq[feature_idx],
        #     label="Reconstructed Prediction",
        #     color="#1f77b4",          # CVD-safe blue (Matplotlib tab:blue)
        #     linewidth=1.0,
        #     linestyle="--", alpha=0.5,
        # )

        # # Ground-truth series – red, solid with sparse markers
        # ax.plot(
        #     self.final_bundle.truth_seq[feature_idx],
        #     label="True Target",
        #     color="#d62728",          # CVD-safe red (Matplotlib tab:red)
        #     linewidth=1.2,
        #     linestyle="-",
        #     marker="o",
        #     markersize=1,
        #     markevery=200,
        # )

        # # ---------- labels ----------
        # ax.set_title(f"Full Series (Feature {feature_idx}, Seed {self.seed})", pad=2)
        # ax.set_xlabel("Time Step")
        # ax.set_ylabel("Value")

        # # ---------- pack layout, then add legend below ----------
        # fig.tight_layout(pad=0.1)          # pack axes first
        # fig.subplots_adjust(bottom=0.18)   # reserve ~18 % height for legend

        # ax.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.15),   # centred, 15 % below x-axis
        #     ncol=2,
        #     frameon=False,
        #     fontsize=8,
        # )

        # fig.savefig("seq_plot.png", dpi=300, bbox_inches="tight")  # vector PDF export
        # plt.show()
        fig, ax = plt.subplots(figsize=(6.9, 3.0))   # 17.8 cm × 7.6 cm

        # --- Plot true series (red, solid) ---
        ax.plot(self.final_bundle.truth_seq[feature_idx],
                color="#d62728",            # tab10 red
                linewidth=2.0,
                label="True (x)")

        # --- Plot predicted series (blue, solid, slightly thinner) ---
        ax.plot(self.final_bundle.pred_seq[feature_idx],
                color="#1f77b4",            # tab10 blue
                linewidth=1.8,
                label="Predicted (x)")
 

        # --- Axes cosmetics ---
        ax.set_xlabel("Time Step", labelpad=6)
        ax.set_ylabel("Value",     labelpad=4)
        ax.set_title(f"Full Series (Feature {feature_idx}, Seed {self.seed})", pad=8, fontsize=16, weight="semibold")
        ax.grid(True, linewidth=0.5, alpha=0.6)

        # Legend in the upper-left corner inside the axes
        ax.legend(loc="upper left", frameon=False, fontsize=11)

        # Tight layout & CMYK export
        fig.tight_layout(pad=0.3)
        fig.savefig('seq_plot.png',
                    format="png",
                    backend="png",
                    cmyk=True,               # CMYK colour model for AAAI
                    bbox_inches="tight")
        fig.savefig("seq_plot.png", dpi=300, bbox_inches="tight")  # vector PDF export

        plt.show()

    def plot_tde(self, feature_idx: int = 0, batch_idx: int = 0):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "axes.edgecolor": "0.7",
            "axes.labelcolor": "0.5",
            "xtick.color":  "0.5",
            "ytick.color":  "0.5",
            "grid.color":   "0.8",
            "text.color":   "0.4",
            "font.size":     9,     # ≈ 9-pt in final PDF (AAAI minimum)
        })
        fig, ax = plt.subplots(figsize=(3.3, 2.5))  # AAAI single-column width

        # ----- data series -----
        ax.plot(
            self.final_bundle.pred_tde[batch_idx, feature_idx, :],
            label="Reconstructed Prediction",
            color="black",
            linewidth=1.0,
            linestyle="--",
        )
        ax.plot(
            self.final_bundle.truth_tde[batch_idx, feature_idx, :],
            label="True Target",
            color="black",
            linewidth=1.5,
            linestyle="-",
            marker="o",
            markersize=2,
            markevery=10,
        )

        # ----- labels -----
        ax.set_title(f"Reconstruction (Feature {feature_idx}, Seed {self.seed})", pad=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")

        # ----- layout, then legend just below -----
        fig.tight_layout(pad=0.1)              # pack axes
        fig.subplots_adjust(bottom=0.18)       # reserve 18 % of fig height for legend

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),       # centred, 15 % below x-axis
            ncol=2,
            frameon=False,
            fontsize=8,
        )

        fig.savefig("tde_plot.png", dpi=300, bbox_inches="tight")  # vector preferred
        plt.show()
        # plt.style.use("seaborn-v0_8-whitegrid")
        # plt.rcParams["axes.edgecolor"] = "0.7"
        # plt.rcParams["axes.labelcolor"] = "0.5"
        # plt.rcParams["xtick.color"] = "0.5"
        # plt.rcParams["ytick.color"] = "0.5"
        # plt.rcParams["grid.color"] = "0.8"
        # plt.rcParams["text.color"] = "0.4"
        # fig, ax = plt.subplots(
        #     figsize=(3.3, 2.5)
        
        # ) #(12, 5))

        # ax.plot(
        #     self.final_bundle.pred_tde[batch_idx, feature_idx, :],
        #     label="Reconstructed Prediction",
        #     color='black',  # Use black for grayscale.
        #     linewidth=1.0,  # Thinner lines can look cleaner.
        #     linestyle="--", # Dashed line for predictions.
        # )
        # ax.plot(
        #     self.final_bundle.truth_tde[batch_idx, feature_idx, :],
        #     label="True Target",
        #     color='black',  # Use black for grayscale.
        #     linewidth=1.5,  # A slightly thicker line for the ground truth.
        #     linestyle="-",  # Solid line for the true target.
        #     marker='o',     # Add circular markers.
        #     markersize=3,   # Make markers small to avoid clutter.
        #     markevery=10,   # Plot a marker only every 10 data points.
        # )

        # ax.set_title(
        #     f"Full Reconstructed Series (Feature {feature_idx}, Seed {self.seed})"
        # )
        # ax.set_xlabel("Time Step")
        # ax.set_ylabel("Value")
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
#endregion SingleRunResults