
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from pydantic import (
    BaseModel, ConfigDict, model_validator, computed_field, Field,
    PositiveInt,
)
from typing import (
    List, Tuple, Union, Callable, Dict, Optional, Any, Self, Literal,
    Annotated,
    Iterable,
    Set,
    Sequence,
)
from enum import StrEnum, auto
import math
 
"""
# Base abstractions and fundamental operations
"""

#region LegoBricks
class LegoBricks(StrEnum):  
    """
    The fundamental, swappable components that *generate* parameters 
    for a geometric transformation. Each entry is a composable transformation brick 
    
    # syntax:    <flavour> : <dst> : <src> : <version?>
    SS_X_GIVEN_Z_V0 = "ss:x:z:v0"  
    means "Scale and shift x given z version 0"
    
    ROT_IN_Y    = "rot:y"       # Rotation in data space
    HID_IM_GIVEN_Z = "hid_im:z"   # Hidden layer given z 
    """
       
    SS_Z_GIVEN_X_V0 = "ss:z:x:v0"   
    SS_X_GIVEN_Z_V0 = "ss:x:z:v0"   
    
    SS_Z_GIVEN_X_V2 = "ss:z:x:v2"   
    SS_X_GIVEN_Z_V2 = "ss:x:z:v2"   
    
    SS_Z_GIVEN_X_V3 = "ss:z:x:v3"   
    SS_X_GIVEN_Z_V3 = "ss:x:z:v3"   
    
    SS_Z_GIVEN_X_V4 = "ss:z:x:v4"   
    SS_X_GIVEN_Z_V4 = "ss:x:z:v4"   
    
    SS_Z_GIVEN_X_V5 = "ss:z:x:v5"   
    SS_X_GIVEN_Z_V5 = "ss:x:z:v5"   
    
    SS_Y_GIVEN_X_V0  = "ss:y:x:v0"      
    SS_X_GIVEN_Y_V0  = "ss:x:y:v0"       
    SS_Y_GIVEN_X_V2 = "ss:y:x:v2"   
    SS_X_GIVEN_Y_V2 = "ss:x:y:v2"   
      
    
    SS_Y_GIVEN_Z_V0 = "ss:y:z:v0"  
    SS_Z_GIVEN_Y_V0 = "ss:z:y:v0" 
    
    SS_Y_GIVEN_Z_V2  = "ss:y:z:v2"     
    SS_Z_GIVEN_Y_V2  = "ss:z:y:v2"     
    
    SS_Y_GIVEN_Z_V3  = "ss:y:z:v3"     
    SS_Z_GIVEN_Y_V3  = "ss:z:y:v3"     
    
    SS_Y_GIVEN_Z_V4  = "ss:y:z:v4"     
    SS_Z_GIVEN_Y_V4  = "ss:z:y:v4"     
    
    SS_Z_GIVEN_Z_V0  = "ss:z:z:v0"     
    
    SS_X_GIVEN_X_V0  = "ss:x:x:v0"     
    
    SS_Y_GIVEN_Y_V0  = "ss:y:y:v0"     
     
    KOO_Z_GIVEN_Z_V0  = "koo:z:z:v0"   
    KOO_Z_GIVEN_X_V0  = "koo:z:x:v0"   
    KOO_Z_GIVEN_Y_V0  = "koo:z:y:v0"   
    KOO_Y_GIVEN_Z_V0  = "koo:y:z:v0"   

    HID_GIVEN_X_V0    = "hid:x:v0"     
    HID_GIVEN_Y_V0    = "hid:y:v0"     
    HID_GIVEN_Z_V0    = "hid:z:v0" 
        
    HID_IM_GIVEN_Y_V0 = "hid_im:y:v0"  
    HID_IM_GIVEN_Z_V0 = "hid_im:z:v0"  

    ROT_IN_X_V0    = "rot:x:v0"      
    ROT_IN_Z_V0    = "rot:z:v0"      
    ROT_IN_Y_V0    = "rot:y:v0"      
    ROT_IN_Y_V1    = "rot:y:v1"      

    # --- helper properties ----------------------------------
    @property
    def flavour(self) -> str:
        return self.value.split(":")[0]

    @property
    def dst(self) -> str:
        return self.value.split(":")[1]

    @property
    def src(self) -> str:
        return self.value.split(":")[2] if self.flavour not in {"hid","hid_im","rot"} else self.value.split(":")[1]

    @property
    def version(self) -> str:
        return self.value.split(":")[3] if len(self.value.split(":")) > 3 else None


#endregion LegoBricks
class CoefName(StrEnum):
    SCALE     = 'scale'
    SHIFT     = 'shift'
    OFF_SCALE = 'off_scale'
    OFF_SIGN = 'off_sign'
    MAGNITUDE = 'magnitude'
    MOMENTUM = 'momentum'
    MOBIUS_A  = 'mobius_a'
    MOBIUS_B  = 'mobius_b'
    MOBIUS_C  = 'mobius_c'
    MOBIUS_D  = 'mobius_d'



#region DynamicTanh

def bounded_tanh(x, low, high, tau=None, eps=1e-4, hard=True):
    """
    Maps R -> [low, high] via tanh. Choose tau so center slope is well-scaled.
    If tau is None, set tau = range to get ~unit slope at 0.
    """
    m = 0.5 * (low + high)
    r = 0.5 * (high - low)
    if tau is None:
        # unit slope at x=0: dy/dx = r*(1/tau) -> set tau=r
        # tau = torch.clamp(r.detach(), min=eps)
        tau = r
    if hard:
        return m + r * F.hardtanh(x , min_val=low, max_val=high) # / tau
    else:
        return m + r * torch.tanh(x ) # / tau

class BoundedScale(nn.Module):
    def __init__(self, low=0.5, high=2.0, init_val=1.0):
        super().__init__()
        self.low = low
        self.high = high           # learnable param

    def forward(self, x):
        # Sigmoid + affine transform → [low, high]
        return self.low + (self.high - self.low) * torch.sigmoid(x)
    
class DynamicTanh(nn.Module):
    def __init__(self, channels: int, dim: int, use_hardtanh: bool = False, 
                 min_val: float = -3.0, max_val: float = 3.0, tau: float = 2, 
                 ):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.ones(channels, self.dim)*0.5)
        self.b = nn.Parameter(torch.zeros(channels,self.dim)*(1.0))  
        
        self.c = nn.Parameter(torch.ones(channels, self.dim)*1.0)
        self.d = nn.Parameter(torch.zeros(channels, self.dim))
        
        self.use_hardtanh = use_hardtanh
        self.min_val = min_val
        self.max_val = max_val
        self.tau = tau
        self.shift = nn.Parameter(torch.randn(channels, self.dim))
        self.scale_low = 0.70
        self.scale_high = 1.1
        self.f = nn.Parameter(torch.zeros(channels,self.dim))
        self.bound = BoundedScale(low=self.scale_low, high=self.scale_high, 
                                  init_val=1.0)
         

    def forward(self, x):
        core_in =  x * self.a  #+ self.b#    #+ self.b
        # core = self.bound(core_in)
        # core = F.hardtanh(core_in, min_val=self.min_val, max_val=self.max_val)
        core = torch.tanh(core_in)
        # core = (torch.clamp(core_in, -2., 2.) if self.use_hardtanh else torch.tanh(core_in))
        # core2 = -self.a2 * x #+ self.b2
        # ReLU6 is a great choice as it's simple and bounded.
        # c = F.relu6(self.c_raw)
        y = core* self.c + self.shift# + self.d
        # y = core * (self.scale_low + (self.scale_high - self.scale_low) * torch.sigmoid(self.c)) + self.shift #+ self.d
        return y
        # soft two-sided cap
        # return softclip_leaky(y, self.min_val, self.max_val, beta=3, leaky=0.1) 

class ResidualBlock(nn.Module):
    """
    A custom nn.Module that implements the core ResNet idea: y = x + F(x).

    This block wraps the transformation function `F(x)` and adds its output
    back to the original input `x`.
    """
    def __init__(self, fx_module: nn.Module, shortcut: nn.Module = None,
                 dim: int = None, channels: int = None):
        """
        Args:
            fx_module (nn.Module): The sequence of layers that define the
                transformation `F(x)`. This is often a small nn.Sequential.
            shortcut (nn.Module, optional): A module (like a Linear layer or
                1x1 Conv) to make the dimensions of `x` match the dimensions
                of `F(x)`. This is only needed if `F(x)` changes the shape.
                Defaults to None, which means an identity connection.
        """
        super().__init__()
        self.fx_module = fx_module
        self.dim = dim
        self.channels = channels
        # If no shortcut is provided, the identity function is used.
        self.shortcut = shortcut if shortcut is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass: applies the transformation and adds the
        skip connection.
        """
        # The shortcut connection is applied to the original input x
        identity = self.shortcut(x)
        
        # The main transformation is applied to x
        fx = self.fx_module(x)
        
        # The core of the residual block: add the identity to the transformation
        return identity + fx
#endregion DynamicTanh

class ActivationType(StrEnum): 
    RELU = auto()
    RELU6 = auto()
    LEAKY_RELU = auto()
    SOFTPLUS = auto()
    TANH = auto()
    HARDTANH = auto() #TODO: test
    DYNAMIC_TANH = auto()
    SIGMOID = auto()
    LOGSIGMOID = auto()
    GELU = auto()
    ELU = auto()
    MISH = auto()
    IDENTITY = auto()
    SOFTSIGN = auto()
    SOFTMAX = auto()
    LOGSOFTMAX = auto()
    SOFTSHRINK = auto()
    HARDSHRINK = auto()
    RRELU = auto()
    SELU = auto()
    CELU = auto()
    TANHSHRINK = auto()
    HARDSWISH = auto()
    SILU = auto()
    RESBLOCK = auto()

ACTI_MAP = {
    ActivationType.IDENTITY: nn.Identity,
    ActivationType.RELU: nn.ReLU,
    ActivationType.RELU6: nn.ReLU6,
    ActivationType.TANH: nn.Tanh,
    ActivationType.LOGSIGMOID: nn.LogSigmoid,
    ActivationType.ELU: nn.ELU,
    ActivationType.GELU: nn.GELU,
    ActivationType.MISH: nn.Mish,
    ActivationType.HARDTANH: nn.Hardtanh,
    ActivationType.DYNAMIC_TANH: DynamicTanh,
    ActivationType.LEAKY_RELU: nn.LeakyReLU,
    ActivationType.SOFTPLUS: nn.Softplus,
    ActivationType.SIGMOID: nn.Sigmoid,
    ActivationType.SOFTSIGN: nn.Softsign,
    ActivationType.SOFTMAX: nn.Softmax,
    ActivationType.LOGSOFTMAX: nn.LogSoftmax,
    ActivationType.SOFTSHRINK: nn.Softshrink,
    ActivationType.HARDSHRINK: nn.Hardshrink,
    ActivationType.RRELU: nn.RReLU,
    ActivationType.SELU: nn.SELU,
    ActivationType.CELU: nn.CELU,
    ActivationType.TANHSHRINK: nn.Tanhshrink,
    ActivationType.HARDSWISH: nn.Hardswish,
    ActivationType.SILU: nn.SiLU,
    ActivationType.RESBLOCK: ResidualBlock,
}

class ActivConfig(BaseModel):
    """Declarative config for one activation function."""
    activation_type: ActivationType
    # Only needed for DynamicTanh; optional otherwise
    dim: int | None = None
    channels: int | None = None

    # Optional tunables (use when relevant) 
    leaky_relu_negative_slope: float | None = None         # LeakyReLU
    softplus_beta: float | None = None                   # Softplus
    softplus_threshold: float | None = None              # Softplus
    gelu_approximate: Literal["none","tanh"] | None = None  # GELU
    elu_alpha: float | None = None                  # ELU
    celu_alpha: float | None = None                  # CELU
    dynamic_hardtanh: bool | None = None
    dynamic_tanh_tau: float | None = None
    dynamic_tanh_min: float | None = None
    dynamic_tanh_max: float | None = None
    hard_tanh_min: float | None = None
    hard_tanh_max: float | None = None
    softshrink_lambd: float | None = None
    hardshrink_lambd: float | None = None

    def finalized(self, *, dim: int, channels: int) -> "ActivConfig":
        """Return a copy with dim/channels filled only if needed."""
        if self.activation_type == ActivationType.DYNAMIC_TANH:
            return self.model_copy(update={
                "dim": self.dim or dim,
                "channels": self.channels or channels,
            })
        return self
    
    def build(self) -> nn.Module:
        t = self.activation_type
        if t == ActivationType.DYNAMIC_TANH:
            if self.channels is None or self.dim is None:
                raise ValueError("DynamicTanh requires channels and dim.")
            # Change this order if your class signature is (dim, channels)
            return DynamicTanh(self.channels, self.dim, 
                               use_hardtanh=self.dynamic_hardtanh or False, 
                               min_val=self.dynamic_tanh_min or -1.0, 
                               max_val=self.dynamic_tanh_max or 1.0)

        if t == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(self.leaky_relu_negative_slope or 0.07,)
        if t == ActivationType.RELU:
            return nn.ReLU()
        if t == ActivationType.RELU6:
            return nn.ReLU6()
        if t == ActivationType.SOFTPLUS:
            return nn.Softplus(beta=self.softplus_beta or 1.0, threshold=self.softplus_threshold or 20.0)
        if t == ActivationType.GELU:
            return nn.GELU(approximate=self.gelu_approximate or "none")
        if t == ActivationType.ELU:
            return nn.ELU(alpha=self.elu_alpha or 1.0)
        if t == ActivationType.HARDTANH:
            return nn.Hardtanh(min_val=self.hard_tanh_min or -1.0, max_val=self.hard_tanh_max or 1.0)
        if t == ActivationType.CELU:
            return nn.CELU(alpha=self.celu_alpha or self.elu_alpha or 1.0)
        if t == ActivationType.SOFTSHRINK:
            return nn.Softshrink(lambd=self.softshrink_lambd or 0.5)
        if t == ActivationType.HARDSHRINK:
            return nn.Hardshrink(lambd=self.hardshrink_lambd or 0.5) 
        if t == ActivationType.DYNAMIC_TANH:
            return DynamicTanh(self.channels, self.dim, 
                               use_hardtanh=self.dynamic_hardtanh or False, 
                                min_val=self.dynamic_tanh_min or -1.0, 
                                max_val=self.dynamic_tanh_max or 1.0, 
                                tau=self.dynamic_tanh_tau or 0.1)
        if t == ActivationType.RESBLOCK:
            return ResidualBlock(
                fx_module=nn.Mish())

        cls = ACTI_MAP.get(t)
        if cls:
            return cls()
        raise ValueError(f"Invalid activation: {t}")

#region LinearSpec
class LinearSpec(BaseModel):
    out_dim: int
    use_spec_norm: bool = False
    use_bias: bool =False# True

#endregion LinearSpec

#region BlockSpec
class BlockSpec(BaseModel):
    pre: LinearSpec
    act: ActivConfig | ActivationType
    norm: Literal["layer", "rms", "none"] | None = None
    drop_out: float | None = None
    post: LinearSpec | None = None
    
    @property
    def out_dim(self) -> int:
        """The block's final output width."""
        return self.post.out_dim if self.post is not None else self.pre.out_dim
 

    @model_validator(mode='before')
    @classmethod
    def _promote_activation(cls, data: dict) -> dict:
        act = data.get('act')
        if isinstance(act, ActivationType):
            data['act'] = ActivConfig(
                activation_type=act,
                # # Only DynamicTanh will assert these:
                # dim=data.get('out_dim'),
                # channels=data.get('channels'),
            )
        return data
    
    def build(self, in_dim: int, channels: int) -> nn.Module:
        layers: list[nn.Module] = []
        
        # pre linear: in_dim -> pre.out_dim
        lin1 = nn.Linear(in_dim, self.pre.out_dim, bias=self.pre.use_bias)
        if self.pre.use_spec_norm: lin1 = spectral_norm(lin1)
        layers.append(lin1) 
        
        if self.norm is not None:
            if self.norm == "layer":
                layers.append(nn.LayerNorm(self.pre.out_dim, elementwise_affine=False)) 
            elif self.norm == "rms":
                layers.append(nn.RMSNorm(self.pre.out_dim, elementwise_affine=False)) 
            elif self.norm == "none":
                pass
        
        act_cfg: ActivConfig = (
            self.act if isinstance(self.act, ActivConfig)
            else ActivConfig(activation_type=self.act)
        ).finalized(dim=self.pre.out_dim, channels=channels)
        layers.append(act_cfg.build())
        
        if self.drop_out is not None and self.drop_out > 0.0:
            layers.append(nn.Dropout(self.drop_out))
        
        # optional post linear: pre.out_dim -> post.out_dim
        if self.post is not None:
            lin2 = nn.Linear(self.pre.out_dim, self.post.out_dim, bias=self.post.use_bias)
            if self.post.use_spec_norm: lin2 = spectral_norm(lin2)
            layers.append(lin2)

        return nn.Sequential(*layers)
#endregion BlockSpec
#region HiddenSchema  

class HiddenSchema(BaseModel):
    """
    Builds an MLP mapping (B, C, in_dim) → (B, C, hid_dim).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    channels: int
    in_dim: int
    hid_dim: int
    out_dim: int  # optional bookkeeping for downstream; not used by build() 
    
    architecture: list[BlockSpec] | None = None
    use_conv: bool = False # False
    use_params: bool = False # True
    device: str = 'cuda'
    # dtype: torch.dtype = torch.float32
    use_history: bool = False
  
    @model_validator(mode="after")
    def build_default_architecture(self) -> "HiddenSchema":
        # Default: in -> hid, then hid -> hid, hid -> hid
        if self.architecture is None:
            self.architecture = [
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.IDENTITY, elu_alpha=0.5), norm='none', drop_out=0.0    ), # IDENTITY
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.ELU, elu_alpha=1.0), norm='none', drop_out=0.0), 
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.ELU, elu_alpha=1.0), norm='none', drop_out=0.0),
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.ELU, softshrink_lambd=0.07), norm='none', drop_out=0.0),  
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.LOGSIGMOID, elu_alpha=1.0), norm='none', drop_out=0.0),
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActivConfig(activation_type=ActivationType.ELU, elu_alpha=1.0), norm='none', drop_out=0.0,
                        #   post=LinearSpec(out_dim=self.hid_dim)
                          ),
            ]
            assert len(self.architecture) >= 1, "Hidden architecture must have ≥1 block"
            
            first = self.architecture[0]
            assert first.pre.out_dim == self.hid_dim, \
                f"first block must map to hid_dim={self.hid_dim}, got {first.pre.out_dim}"

            for i, blk in enumerate(self.architecture[1:], start=1):
                assert blk.out_dim == self.hid_dim, \
                    f"block {i} must preserve hid_dim={self.hid_dim}, got out_dim={blk.out_dim}"

        return self

    def build_convs(self) -> nn.Conv1d:
        return nn.Conv1d(self.channels, self.channels, kernel_size=1, groups=self.channels, bias=True)

    def build_params(self) -> nn.ParameterDict:
        # params applied POST-MLP on hidden space (C, H)
        params = nn.ParameterDict()
        params["scale"] = nn.Parameter(torch.ones(self.channels, self.in_dim))
        params["shift"] = nn.Parameter(torch.zeros(self.channels, self.in_dim))
        return params

    def build(self) -> nn.Sequential:
        # requires LinearBlockConfig to have the validator that promotes ActivationType→ActivConfig
        layers = []
        in_w = self.in_dim
        for blk in self.architecture:
            seq = blk.build(in_dim=in_w, channels=self.channels)
            layers.append(seq)
            in_w = blk.out_dim                  # <- becomes hid_dim after block 1
        return nn.Sequential(*layers)
    
    def build_factory(self) -> nn.Module:
        from study.FERN_gen import HiddenFactory
        return HiddenFactory(schema=self)
#endregion HiddenSchema

#region CoefSchema 

#region HeadSpec
class HeadPostSpec(BaseModel):
    # lightweight post-processing knobs (same as today)
    zero_threshold: float | None = None
    clip_lower_bound: float | None = None
    clip_upper_bound: float | None = None
    stage: Literal["pre","post"] = "pre"  # where affine params apply

class HeadSpec(BaseModel):
    """One coefficient head = core block."""
    core: list[BlockSpec]                   # H -> out_elems 
    post: HeadPostSpec | None = None  # threshold/clip/stage 
     
    @classmethod
    def scale_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.ELU, leaky_relu_negative_slope=0.1),
                        #   post=LinearSpec(out_dim=out_elems)
                        ),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig( activation_type=ActivationType.ELU,
                            dynamic_tanh_min=-2.0, dynamic_tanh_max=2.0, dynamic_tanh_tau=3.0), #DYNAMIC_TANH
                        #   post=LinearSpec(out_dim=out_elems),
                #         #   act_2=ActivConfig( activation_type=ActivationType.SOFTSHRINK, softshrink_lambd=0.1),
                        )
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
        
    @classmethod
    def complex_scale_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.ELU, leaky_relu_negative_slope=0.1),
                        #   post=LinearSpec(out_dim=out_elems),
                          ),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig( activation_type=ActivationType.ELU, dynamic_hardtanh=False,
                                          dynamic_tanh_min=-2.0, dynamic_tanh_max=2.0, dynamic_tanh_tau=0.2),
                          post=LinearSpec(out_dim=out_elems),
                          ),
                        #   act_2=ActivConfig( activation_type=ActivationType.SOFTSHRINK, softshrink_lambd=0.1),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
    
    @classmethod
    def shift_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.ELU), leaky_relu_negative_slope=0.1,
                          norm='none', drop_out=0.0,
                          ),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.ELU, dynamic_hardtanh=False,
                                          dynamic_tanh_min=-2.0, dynamic_tanh_max=2.0, dynamic_tanh_tau=2),
                          norm='none', drop_out=0.0,
                          post=LinearSpec(out_dim=out_elems),
                          ),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
    
    @classmethod
    def off_scale_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                # BlockSpec(pre=LinearSpec(out_dim=out_elems),
                #           act=ActivConfig(activation_type=ActivationType.SILU, hard_tanh_min=-10.0, hard_tanh_max=10.0)),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.DYNAMIC_TANH, dynamic_hardtanh=False,
                                          dynamic_tanh_min=-12.0, dynamic_tanh_max=12.0, dynamic_tanh_tau=0.1),
                        #   post=LinearSpec(out_dim=out_elems),
                          ),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
    
    @classmethod
    def off_sign_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.RELU6)),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
    
    @classmethod
    def mag_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.RELU6)),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
        
    @classmethod
    def momentum_default(cls, out_elems: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),
                          act=ActivConfig(activation_type=ActivationType.RELU6)),
            ], 
            post=HeadPostSpec(zero_threshold=None, clip_lower_bound=None, clip_upper_bound=None, stage="pre")
        )
#endregion HeadSpec

#region PatchSpec
def compute_patch(out_dim: int, patch_len: int, stride: int) -> 'PatchSpec':
    """
    Factory function to calculate and create a PatchSpec configuration.

    Args:
        out_dim: The target output dimension.
        patch_len: The length of each individual patch.
        stride: The step size between the start of consecutive patches.

    Returns:
        A validated PatchSpec object with all derived values calculated.
    """
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive integers.")
    if out_dim < patch_len:
        raise ValueError("out_dim must be >= patch_len")
    if patch_len < 1 or stride < 1:
        raise ValueError("patch_len and stride must be > 0")
    if stride >= patch_len: 
        raise ValueError("stride must be < patch_len")

    # minimal patches to reach/end beyond out_dim 
    print(f"[compute_patch] out_dim={out_dim}, patch_len={patch_len}, stride={stride}")
    num_patches = math.ceil(max(0, out_dim - patch_len) / stride) + 1 
    covered_len = (num_patches - 1) * stride + patch_len
    if covered_len < out_dim:
        raise ValueError("covered_len must be >= out_dim")
    out_elems     = num_patches * patch_len 
    print(f"  -> num_patches={num_patches}, covered_len={covered_len}, out_elems(raw)={out_elems}")
    print(f"num_patches: {num_patches}, covered_len: {covered_len}, out_elems: {out_elems}")

    return PatchSpec(out_dim=out_dim, patch_len=patch_len, stride=stride, 
        num_patches=num_patches, covered_len=covered_len, out_elems=out_elems)

class PatchSpec(BaseModel):
    out_dim: PositiveInt 
    patch_len: PositiveInt = 48
    stride: PositiveInt = 24 
    out_elems: PositiveInt = Field(..., description="The total tensor length the model must generate (num_patches * patch_len).")
    num_patches: PositiveInt = Field(..., description="The minimum number of patches needed to cover out_dim.")
    covered_len: PositiveInt = Field(..., description="The actual length of the sequence after stitching (>= out_dim).")
    
    # Convenience: stitch policy
    window: Optional[str] = "ones"   # "ones" | "hann" | None
    normalize: str = "avg"           # "avg" | "sum"
   
    @computed_field
    @property
    def overlap(self) -> int:
        return max(0, self.patch_len - self.stride)
   

class CoefSchema(BaseModel): 
    """
    Bill of Materials for coefficient heads mapping hidden (H) → output (D_out)
    with optional patching. Assumed input to heads is (B, C, H).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True) 
    in_dim: int  # usually = hid_dim of HiddenSchema
    out_dim: int # target field dimension (e.g., seq_len/pred_len)
    hid_dim: int # explicit for clarity if needed elsewhere
    channels: int 
      
    specs: Dict[CoefName, HeadSpec] = None 
    structure: Literal['diagonal', 'complex', 'tri_sym', 'tri_anti', 
                       'tri_free_sym', 'tri_free_anti'] = 'diagonal'
    
    use_patch: bool = False
    patch: PatchSpec | None = None
    
    use_params: bool = False # False
    use_convs: bool = False # True
    
    def build_params(self) -> nn.ParameterDict:
        """
        Per-head, per-channel affine params; shape depends on stage:
          - 'pre': (C, hid_dim)  (applied to hidden features)
          - 'post': (C, out_dim) (applied to head output)
        """
        params = nn.ParameterDict()
        for head, spec in self.specs.items():
            if head == CoefName.OFF_SIGN:
                continue
            if spec.post and spec.post.stage == "pre":
                shape = (self.channels, self.hid_dim)
            else:  # "post"
                shape = (self.channels, self.out_dim)
            # default neutral affine
            if head in [CoefName.SCALE, CoefName.OFF_SCALE]:
                params[head.name] = nn.Parameter(torch.ones(*shape))
            else:
                params[head.name] = nn.Parameter(torch.zeros(*shape))
        return params
    
    def build_convs(self) -> nn.ModuleDict:
        """
        Depthwise 1x1 convs for per-channel alignment (optional).
        Not the main head projection; that lives in the factory MLP/linear.
        """
        convs = nn.ModuleDict()
        for head, spec in self.specs.items():
            convs[head.name] = nn.Conv1d(
                in_channels=self.channels, 
                out_channels=self.channels,
                kernel_size=1, groups=self.channels, bias=True,
            )
        return convs

    def build(self) -> nn.ModuleDict:
        # from study.FERN_core import ActivConfig
        foundry = nn.ModuleDict() 
        for head, spec in self.specs.items(): 
            layers_core = []
            in_w = self.hid_dim
            for blk in spec.core:
                seq = blk.build(in_dim=in_w, channels=self.channels)
                layers_core.append(seq)
                in_w = blk.out_dim         
            foundry[head.name] = nn.Sequential(*layers_core) 
        return foundry
     
    @computed_field
    @property
    def out_elems(self) -> int:
        """
        Raw width emitted by each head BEFORE stitch.
        - no patch: equals out_dim
        - patch   : equals num_patches * patch_len (concatenated raw width)
        """
        return self.patch.out_elems if self.patch else self.out_dim
     
    @computed_field
    @property
    def patch_len(self) -> int:
        """Width a single patch segment should produce (L if patching, else out_dim)."""
        return self.patch.patch_len if self.patch else self.out_dim
    
    @model_validator(mode="after") 
    def _build_default_specs(self) -> "CoefSchema":
         
        if self.use_patch and self.patch is None:
            self.patch = compute_patch(self.out_dim, 48, 24)
            assert self.patch is not None
            assert self.patch.num_patches * self.patch.patch_len == self.out_elems  # raw
            assert (self.patch.num_patches - 1) * self.patch.stride + self.patch.patch_len >= self.out_dim
            assert self.patch.stride <= self.patch.patch_len  # no gaps
        
        out_elems = self.out_elems
        if self.specs is None:
            self.specs: dict[str, HeadSpec] = { 
                CoefName.SHIFT: HeadSpec.shift_default(out_elems),
            }
        if self.structure == 'complex':
            self.specs[CoefName.SCALE] = HeadSpec.complex_scale_default(out_elems)
        else:
            self.specs[CoefName.SCALE] = HeadSpec.scale_default(out_elems)
            
        if self.structure == 'tri_sym':
            self.specs[CoefName.OFF_SCALE] = HeadSpec.off_scale_default(out_elems-1)
        elif self.structure == 'tri_anti':
            self.specs[CoefName.OFF_SCALE] = HeadSpec.off_scale_default(out_elems-1)
        elif self.structure == 'tri_free_sym':
            self.specs[CoefName.OFF_SCALE] = HeadSpec.off_scale_default(out_elems-1)
            self.specs[CoefName.OFF_SIGN] = HeadSpec.off_sign_default(out_elems-1) 
        elif self.structure == 'tri_free_anti':
            self.specs[CoefName.OFF_SCALE] = HeadSpec.off_scale_default(out_elems-1)
            self.specs[CoefName.OFF_SIGN] = HeadSpec.off_sign_default(out_elems-1)
        
        # required heads
        if self.structure == 'diagonal':
            required_heads = (CoefName.SCALE, CoefName.SHIFT)
        elif self.structure == 'complex':
            required_heads = (CoefName.SCALE, CoefName.SHIFT)
        elif self.structure == 'tri_sym':
            required_heads = (CoefName.SCALE, CoefName.SHIFT, CoefName.OFF_SCALE)
        elif self.structure == 'tri_anti':
            required_heads = (CoefName.SCALE, CoefName.SHIFT, CoefName.OFF_SCALE)
        elif self.structure in ['tri_free_anti', 'tri_free_sym']:
            required_heads = (CoefName.SCALE, CoefName.SHIFT, CoefName.OFF_SCALE, CoefName.OFF_SIGN)
        
        for required in required_heads:
            if required not in self.specs:
                raise ValueError(f"Coef Schema validation failed: Missing required head: {required} with structure: {self.structure}") 
            
        return self
      
    def build_factory(self) -> nn.Module:
        from study.FERN_gen import CoefFactory
        return CoefFactory(schema=self)
#endregion CoefSchema

#region RotationSchema
class RotationSchema(BaseModel):
    """Schema for data-dependent Householder rotations (Koopman/OT blocks)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
  
    channels: int
    in_dim: int
    out_dim: int
    num_reflects: int = 2
    # If you need these toggles later, keep them here so you don’t revive adapters:
    adapt_params: bool = False, 
    patch_size: int | None = 24#None
    with_scale_generation: bool = False

    def build_factory(self) -> nn.Module:
        from study.FERN_gen import RotateFactory  # avoid top-level circular import
        return RotateFactory(schema=self)
#endregion RotationSchema
 