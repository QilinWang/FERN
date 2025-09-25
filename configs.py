import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Dict, Optional, List, Literal, Union
from pydantic import (
    BaseModel, Field, computed_field, model_validator, ConfigDict,
)
from enum import Enum, StrEnum, auto 
from functools import cached_property  
import math
import study.FERN_core as fcore

 

#region BaseTrainingConfig
class BaseTrainingConfig(BaseModel):
    """A Pydantic model for shared, common training parameters."""

    # --- Essential Identifiers ---
    seq_len: int
    pred_len: int
    label_len: int = 0
    channels: int
    batch_size: int = 128

    # --- Core Training Hyperparameters ---
    learning_rate: float = 9e-4
    epochs: int = 50
    patience: int = 5
    device: str = "cuda"
    # dtype: torch.dtype = torch.float32
    num_proj_swd: int = 500
    seeds: Optional[List[int]] = Field(default_factory=lambda: [1955, 7, 20])
    task_name: str = "long_term_forecast"

    scheduler_type: Literal["none", "plateau", "cosine"] = Field(
        default="none",
        description="Type of LR scheduler to use ('none', 'plateau', 'cosine').",
    )
    lr_scheduler_patience: int = Field(
        default=2,
        description="Patience (in epochs) for ReduceLROnPlateau before reducing LR.",
    )
    lr_scheduler_factor: float = Field(
        default=0.7,
        description="Factor by which to reduce LR for ReduceLROnPlateau (new_lr = lr * factor).",
    )
    warmup_epochs: int = Field(
        default=3,
        description="Number of epochs for warmup.",
    )
    eta_min: float = Field(
        default=1e-5,
        description="Minimum LR for cosine scheduler.",
    )

    # --- Losses ---
    mse_weight_backward: float = 0.0
    mae_weight_backward: float = 0.0
    huber_weight_backward: float = 1.0
    swd_weight_backward: float = 0.0
    quantile85_weight_backward: float = 0.0
    quantile70_weight_backward: float = 0.0
    quantile30_weight_backward: float = 0.0
    quantile15_weight_backward: float = 0.0

    mse_weight_validate: float = 0.0
    mae_weight_validate: float = 0.0
    huber_weight_validate: float = 1.0
    swd_weight_validate: float = 0.0
    quantile85_weight_validate: float = 0.0
    quantile70_weight_validate: float = 0.0
    quantile30_weight_validate: float = 0.0
    quantile15_weight_validate: float = 0.0

    use_cross_validation_for_val: bool = True
    val_cv_num_folds: int = 10
    val_cv_num_samples: int = 6
# endregion

 
DIM_BY_LETTER = {"x": "seq_len",   # length of x-series
                 "y": "pred_len",  # horizon
                 "z": "dim_augment"}
 
SCHEMA_BY_FLAVOUR = {"ss": fcore.CoefSchema,
                     "koo": fcore.CoefSchema,
                     "hid": fcore.HiddenSchema,
                     "hid_im": fcore.HiddenSchema}


#region FERNConfig
class FERNConfig(BaseTrainingConfig):
    """FLAT configuration for FERN model. Contains ALL base and specific fields."""
    # --- Meta ---
    model_type: Literal["FERN"] = "FERN"
    forward_signature: Literal["x,update"] = "x,update"
    output_signature: Literal["tensor"] = "tensor"

    # --- FERN Specific Fields ---
    dim_augment: int = 128
    dim_hidden: int = 128
    householder_reflects_latent: int = 2
    householder_reflects_data: int = 4

    factory_schemas: Dict[str, Union[fcore.CoefSchema, fcore.HiddenSchema]] = Field(
        default_factory=dict, repr=False
    )

    # Ablations
    class DecoderMode(StrEnum):
        FULL_ROTATION = auto()
        NO_ROTATION = auto()
        SHIFT_ONLY = auto()
        SCALE_ONLY = auto()
        SHIFT_INSIDE_SCALE = auto()

    decoder_mode: DecoderMode = DecoderMode.FULL_ROTATION

    enable_koopman: bool = True
    ablate_no_rotation: bool = False
    use_complex_eigenvalues: bool = True
    enable_rotate_back_Koopman: bool = True 
    # ablate_cascaded_model: bool = False
    ablate_single_encoding_layer: bool = False
    ablate_deterministic_y0: bool = False  # must used togetehr with shift_inside_scale
    use_data_shift_in_z_push: bool = True
 
    factory_schemas: Dict[str, Union[fcore.CoefSchema, fcore.HiddenSchema]] = Field(
        default_factory=dict, repr=False
    )
  
    # NEW: A model validator to build and populate the factory_schemas dictionary.
    @model_validator(mode='after')
    def _build_factory_schemas(self) -> "FERNConfig":
        """ 
        Iterates through the fcore.LegoBricks enum and 
        populates the `factory_schemas` dictionary.
        """ 
        for fid in fcore.LegoBricks:  
            if fid.flavour in {"hid","hid_im"}:
                in_dim = getattr(self, DIM_BY_LETTER[fid.src]) 
                out_dim = self.dim_hidden
                schema = fcore.HiddenSchema(
                channels=self.channels, in_dim=in_dim, 
                out_dim=out_dim, hid_dim=self.dim_hidden,
                device=self.device,  
            )

            elif fid.flavour in {"ss", "koo"}:
                in_dim = self.dim_hidden
                out_dim = getattr(self, DIM_BY_LETTER[fid.dst])
                if fid == fcore.LegoBricks.KOO_Z_GIVEN_X_V0:
                    in_dim = self.seq_len
                    out_dim = self.dim_hidden
                    
                #----Further Specific SS Configs ----
                if fid.flavour in ["ss"] and fid.dst in ["y"]:
                    use_patch = False #TODO
                else:
                    use_patch = False 
                
                
                if fid.flavour in ["ss"] and fid.dst in ["x", 'z'] and not fid.version in ["v0",'v3']:
                    structure = "complex"   # tri_anti complex
                elif fid.flavour in ["ss"] and fid.dst in ["y"]:
                    structure = "diagonal"
                else:
                    structure = "diagonal"  # global default
                # print(f'check: condition is --if fid.flavour in ["ss"] and fid.dst in ["x",]:--',flush=True)
                # print(f"then structure is --structure = 'complex' # tri_anti complex--",flush=True)
                # print(f"fid.flavour: {fid.flavour}",flush=True)
                # print(f"fid.dst: {fid.dst}",flush=True)
                # print(f"Let us check the structure: {structure}",flush=True)
                #--------------------------------
                schema = fcore.CoefSchema(
                channels=self.channels, in_dim=in_dim,
                out_dim=out_dim, hid_dim=self.dim_hidden,
                use_patch=use_patch, structure=structure,
            )   
                # if structure == 'complex':
                #     print(f"schema structure: {schema.structure}",flush=True)
                # print(f"schema: {schema}",flush=True)
            elif fid.flavour == "rot":
                schema = fcore.RotationSchema(
                    channels=self.channels, in_dim=self.dim_hidden,
                    out_dim=getattr(self, DIM_BY_LETTER[fid.dst]),
                    num_reflects=self.householder_reflects_latent,
                    adapt_params=False
                )
            else:
                continue
            
            self.factory_schemas[fid] = schema
        return self
# endregion

#region PatchTSTConfig
class PatchTSTConfig(BaseTrainingConfig):
    """FLAT configuration for PatchTST model."""

    # --- Meta ---
    model_type: Literal["PatchTST"] = "PatchTST"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

    # --- PatchTST Specific Fields ---
    d_model: int = 128
    e_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    patch_len: int = 16
    stride: int = 8
    factor: int = 3

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def dec_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels


class DLinearConfig(BaseTrainingConfig):
    """FLAT configuration for DLinear model."""

    # --- Meta ---
    model_type: Literal["DLinear"] = "DLinear"
    forward_signature: Literal["x"] = "x"
    output_signature: Literal["tensor"] = "tensor"

    # --- DLinear Specific Fields ---
    individual: bool = True


class TimeMixerConfig(BaseTrainingConfig):
    """FLAT configuration for TimeMixer model."""

    # --- Meta ---
    model_type: Literal["TimeMixer"] = "TimeMixer"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

    # --- TimeMixer Specific Fields ---
    embed: Optional[str] = None  # must keep None
    freq: Optional[str] = None  # must keep None
    use_norm: Optional[bool] = None  # must keep None
    channel_independence: bool = True
    e_layers: int = 2
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    d_model: int = 16
    d_ff: int = 32
    dropout: float = 0.1
    decomp_method: str = "moving_avg"
    moving_avg: int = 25
    down_sampling_method: str = "avg"

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def dec_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels


class NaiveConfig(BaseTrainingConfig):
    """FLAT configuration for Naive baseline model."""

    # --- Meta ---
    model_type: Literal["naive"] = "naive"
    forward_signature: Literal["naive_repeat"] = "naive_repeat"
    output_signature: Literal["tensor"] = "tensor"

class AttraosConfig(BaseTrainingConfig):
    """FLAT configuration for Attraos model."""

    # --- Meta ---
    model_type: Literal["Attraos"] = "Attraos"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

class KoopaConfig(BaseTrainingConfig):
    """FLAT configuration for Koopa model."""

    # --- Meta ---
    model_type: Literal["Koopa"] = "Koopa"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

ModelConfigType = Union[
    FERNConfig,
    TimeMixerConfig,
    PatchTSTConfig,
    DLinearConfig,
    NaiveConfig,
]
