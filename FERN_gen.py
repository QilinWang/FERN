from enum import StrEnum, auto

from typing import Union, Literal, List, Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from pydantic import ConfigDict, model_validator
from pydantic import BaseModel, PositiveInt, ConfigDict, Field, computed_field, model_validator

import study.FERN_core as fcore
import study.FERN_util as fru

import study.configs as configs
    
# ------------------------------------------------------------
# region Op
class Op(BaseModel):
    model_config = ConfigDict(
        extra='forbid',  # <--- Add this line 
        arbitrary_types_allowed=True,      # accept tensors
        json_encoders={                    # compact dumps
            torch.Tensor: lambda t: f"Tensor{tuple(t.shape)}"
        },
    )
    name: str = "Op"

    # ----- core algebra -----
    def __call__(self, x):        return self.apply(x)
    def __ror__(self, left):      return self._pipe(left, self)
    def __or__(self, right):      return self._pipe(self, right)

    # subclasses supply these
    def apply(self, x):    raise NotImplementedError
    def inverse(self, x):  raise NotImplementedError

    # ----- printing ----------
    def _summary(self, k=0): return f'{"  "*k}- {self.name}\n'
    def __str__(self):     return self._summary()

    # helper
    @staticmethod
    def _pipe(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, Op):
            return b.apply(a)
        if isinstance(a, Op) and isinstance(b, torch.Tensor):
            return a.apply(b)
        if isinstance(a, Op) and isinstance(b, Op):
            return Chain([a, b])
        print(f"type(a): {type(a)}, type(b): {type(b)}")
        raise TypeError
    
    @property
    def inv(self) -> "InverseOp":
        return InverseOp(base=self)

    def __invert__(self) -> "InverseOp":  
        """enables ~op so  x | self.shift.inv = x | ~self.shift""" 
        return self.inv


class Chain(Op):
    name: str = "Chain"
    parts: list[Op] = Field(default_factory=list)

    # constructor flattens nested composites
    def __init__(self, parts: Iterable[Op]):
        flat = []
        for p in parts:
            flat.extend(p.parts) if isinstance(p, Chain) else flat.append(p)
        super().__init__(parts=flat)

    # container proxy helpers ------------------------------------------
    def __iter__(self):          return iter(self.parts)
    def __len__(self):           return len(self.parts)
    def __getitem__(self, i):    return self.parts[i]

    # algebra -----------------------------------------------------------
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.parts:
            x = op.apply(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for op in reversed(self.parts):
            x = op.inverse(x)
        return x
    
    @property
    def inv(self):
        # reverse order & invert each part
        # Then (~Chain) works and is equivalent to chaining each part’s inverse in reverse order.
        return Chain([p.inv for p in reversed(self.parts)])

    # pretty print
    def _summary(self, k=0):
        inner = "".join(p._summary(k+1) for p in self.parts)
        return f'{"  "*k}* {self.name}:\n{inner}'


class InverseOp(Op):
    base: Op

    @property
    def name(self):  # override for nicer print
        return f"{self.base.name}^-1"

    def apply(self, x):
        return self.base.inverse(x)

    def inverse(self, x):
        # inverse of the inverse is the forward
        return self.base.apply(x)

    def _summary(self, k=0):
        return f'{"  "*k}- {self.name}\n'
# endregion Op

"""
In this section, we define a list of convient wrapper of tensors 
that allows you to 'mutate' tensors with coefficients 
like x | scale | shift | ... or x | rotation | ... 
This is the 'micro use' of Op and Chain DSL.
It allows us to reuse the same API in G and T (where we send States 
to States), the 'macro use'.
"""
# region Scale

def apply_block_diagonal(x, scale):
    """
    Apply a block-diagonal transformation with 2×2 blocks to a vector.
    # [a  b] [x_even]
    # [-b a] [x_odd ]
    
    Args:
        x: Input tensor of shape [..., dim]
        a: Scaling parameters of shape [..., dim//2]
        b: Rotation parameters of shape [..., dim//2]
        
    Returns:
        Transformed tensor of same shape as x
    """
    dim = x.shape[-1]
    assert dim % 2 == 0, "Dimension must be even for 2×2 blocks" 
    # For example, if x has shape [32, 128], it becomes [32, 64, 2], 64 is the number of 2×2 blocks, 2 is the number of elements per block
    x_reshaped = einops.rearrange(x, 'b d (sn e) -> b d sn e', sn=dim//2, e=2) # Reshape tensor to group adjacent pairs [..., dim//2, 2]
    x_even = x_reshaped[..., 0]  # [..., dim//2]
    x_odd = x_reshaped[..., 1]   # [..., dim//2]
    scale_reshaped = einops.rearrange(scale, 'b d (sn e) -> b d sn e', sn=dim//2, e=2)
    scale_even = scale_reshaped[..., 0]
    scale_odd = scale_reshaped[..., 1]       
    y_even = scale_even * x_even - scale_odd * x_odd       # [..., dim//2]
    y_odd = scale_odd * x_even + scale_even * x_odd       # [..., dim//2]
    
    return einops.rearrange(torch.stack([y_even, y_odd], dim=-1), 'b d sn e -> b d (sn e)')



def apply_tridiagonal(
    x: torch.Tensor,
    diag_lower: torch.Tensor,
    diag_main: torch.Tensor,
    diag_upper: torch.Tensor
) -> torch.Tensor:
    """
    Applies a tridiagonal matrix transformation without materializing the matrix.

    This function computes `y = T @ x` where T is a tridiagonal matrix defined
    by the three diagonal coefficient tensors. It is highly efficient as it
    only uses slicing, padding, and element-wise multiplication.

    The operation for each element `y[i]` is:
    y[i] = diag_lower[i-1]*x[i-1] + diag_main[i]*x[i] + diag_upper[i]*x[i+1]

    Args:
        x (torch.Tensor): The input tensor of shape (..., S).
        diag_lower (torch.Tensor): The lower diagonal coefficients, shape (..., S-1).
        diag_main (torch.Tensor): The main diagonal coefficients, shape (..., S).
        diag_upper (torch.Tensor): The upper diagonal coefficients, shape (..., S-1).

    Returns:
        torch.Tensor: The transformed tensor `y` of the same shape as `x`.
    """
    S = x.shape[-1]
    if diag_main.shape[-1] != S:
        raise ValueError(f"Shape mismatch: x last dim is {S} but diag_main is {diag_main.shape[-1]}")
    if diag_lower.shape[-1] != S - 1 or diag_upper.shape[-1] != S - 1:
        raise ValueError(f"Shape mismatch: Off-diagonals must have length {S-1}")

    # 1. Main diagonal term (no shift needed)
    # y_i += main_i * x_i
    main_term = diag_main * x

    # 2. Lower diagonal term (applies to x_{i-1})
    # y_i += lower_{i-1} * x_{i-1}
    x_shifted_right = x[..., :-1]  # x_0, x_1, ..., x_{S-2}
    lower_prod = diag_lower * x_shifted_right
    # Pad on the left to align with y_1, y_2, ...
    lower_term = F.pad(lower_prod, (1, 0))

    # 3. Upper diagonal term (applies to x_{i+1})
    # y_i += upper_i * x_{i+1}
    x_shifted_left = x[..., 1:]  # x_1, x_2, ..., x_{S-1}
    upper_prod = diag_upper * x_shifted_left
    # Pad on the right to align with y_0, y_1, ...
    upper_term = F.pad(upper_prod, (0, 1))

    # 4. Sum all contributions
    y = main_term + lower_term + upper_term

    return y


    # def inverse(self, x: torch.Tensor) -> torch.Tensor:
    #     return apply_tridiagonal(x=x, diag_lower=self.lower_coef, diag_main=self.coef, diag_upper=self.upper_coef)

#region softclip functions
def softclip_hinge(x, low, high, beta=0.8):
    """
    Smoothly approaches clamp(low, high).
    Inside: ~identity. Near edges: smooth knee. Outside: tends toward clamp.
    Derivative inside ~1, outside -> 0 (without leaky mix-in).
    """
    return x - F.softplus(x - high, beta=beta) + F.softplus(low - x, beta=beta) # , beta=beta

def softclip_leaky(x, low, high, beta=0.8, leaky=0.05):
    """
    Same as softclip_hinge but mixes in a leaky identity to keep nonzero grad outside.
    Gradient: ~1 inside, ~leaky outside.
    """
    y = softclip_hinge(x, low, high, beta=beta)
    return leaky * x + (1.0 - leaky) * y

def range_penalty(x, low, high):
    # Quadratic penalty outside [low, high]; zero inside
    under = F.relu(low - x)
    over  = F.relu(x - high)
    return (under.pow(2) + over.pow(2)).mean()

def relu6_softcap_leaky(x, high: float = 6.0, beta: float = 1.0, leak: float = 0.05):
    x0    = F.softplus(x)
    y_cap = x0 - F.softplus(x0 - high, beta=beta)
    return y_cap + leak * (x0 - y_cap)           # (1-leak)*cap + leak*identity

def positive_log_exp(x): 
    max_log = 3.0
    log_output = max_log * torch.tanh(x)
    output = torch.expm1(log_output) + 1.0
    return output

#region Scale Op 
class Scale(Op):  
    coef: torch.Tensor 
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "relu6_softcap_leaky",
        "tanh", "hardtanh", 'none', 'softclip_hinge', 'softclip_leaky', 'bounded_tanh', 'positive_log_exp',
    ] = "none"           # "relu6" | "softplus" | "none"
    low: float = 0.0
    high: float = 6.0 
    factor: float = 1.0
    tau: float | None = None
    # leakt coef
    leak: float = 0.05
    beta: float = 1.0
    
    def process(self, s: torch.Tensor) -> torch.Tensor: 
        if self.post_process == "relu6":  return F.relu6(s) * self.factor
        if self.post_process == "softplus":  return F.softplus(s) * self.factor
        if self.post_process == "sigmoid":  return torch.sigmoid(s) * self.factor
        if self.post_process == "tanh":  return torch.tanh(s) * self.factor
        if self.post_process == "hardtanh":  return F.hardtanh(s, min_val=self.low, max_val=self.high) * self.factor
        if self.post_process == "relu6_softcap_leaky":  return relu6_softcap_leaky(
            s, high=self.high,  beta=self.beta, leak=self.leak) * self.factor
        if self.post_process == "softclip_hinge":  return softclip_hinge(s, low=self.low, high=self.high, beta=self.beta) * self.factor
        if self.post_process == "softclip_leaky":  return softclip_leaky(s, low=self.low, high=self.high, beta=self.beta, leaky=self.leak) * self.factor
        if self.post_process == "bounded_tanh":  return fcore.bounded_tanh(s, low=self.low, high=self.high, tau=self.tau) * self.factor
        if self.post_process == "positive_log_exp":  return positive_log_exp(s) * self.factor
        return s
 
    # def apply(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.process(self.coef)
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the apply method.")
    
    # def inverse(self, x: torch.Tensor) -> torch.Tensor:
    #     return x / (self.coef + 1e-9) 
class DiagScale(Scale):  
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "relu6_softcap_leaky",
        "tanh", "hardtanh", 'none', 'softclip_hinge', 'softclip_leaky', 'bounded_tanh', 'positive_log_exp',
    ] = "relu6_softcap_leaky"           # "relu6" | "softplus" | "none"
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return x * processed_coef
        # return x * self.coef
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return x / (processed_coef + 1e-9)

class ComplexScale(Scale):  
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "relu6_softcap_leaky",
        "tanh", "hardtanh", 'none', 'softclip_hinge', 'softclip_leaky', 'bounded_tanh', 'positive_log_exp',
    ] = "softclip_leaky"           # "relu6" | "softplus" | "none"
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return apply_block_diagonal(x, processed_coef)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return x / (processed_coef + 1e-9)
    
class TriScale(Scale):
    upper_coef: torch.Tensor 
    lower_coef: torch.Tensor
    
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "relu6_softcap_leaky",
        "tanh", "hardtanh", 'none', 'softclip_hinge', 'softclip_leaky', 'bounded_tanh', 'positive_log_exp',
    ] = "relu6_softcap_leaky"           # "relu6" | "softplus" | "none"
    
    @model_validator(mode="after")
    def validate_coefs(self) -> "TriScale":
        if self.upper_coef.shape[-1] != self.coef.shape[-1]-1:
            raise ValueError("upper_coef shape should be one less than coef shape")
        if self.lower_coef.shape[-1] != self.coef.shape[-1]-1:
            raise ValueError("lower_coef shape should be one less than coef shape")
        
        return self
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return apply_tridiagonal(x=x, diag_lower=self.lower_coef, diag_main=processed_coef, diag_upper=self.upper_coef)
 
# endregion Scale Op

#region Shift
class Shift(Op):  
    coef: torch.Tensor
    magnitude: torch.Tensor | None = None
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "relu6_softcap_leaky",
        "tanh", "hardtanh", 'none', 'softclip_hinge', 'softclip_leaky', 'bounded_tanh', 'positive_log_exp',
    ] = "none"     # softclip_leaky       # "relu6" | "softplus" | "none"
    low: float = -20.0
    high: float = 20.0 
    factor: float = 1.0
    beta: float = 3.0
    leak: float = 0.05
    tau: float | None = None
    
    def process(self, s: torch.Tensor) -> torch.Tensor: 
        if self.post_process == "relu6":  return F.relu6(s) * self.factor
        if self.post_process == "softplus":  return F.softplus(s) * self.factor
        if self.post_process == "sigmoid":  return torch.sigmoid(s) * self.factor
        if self.post_process == "tanh":  return torch.tanh(s) * self.factor
        if self.post_process == "hardtanh":  return F.hardtanh(s, min_val=self.low, max_val=self.high) * self.factor
        if self.post_process == "relu6_softcap_leaky":  return relu6_softcap_leaky(
            s, high=self.high,  beta=self.beta, leak=self.leak) * self.factor
        if self.post_process == "softclip_hinge":  return softclip_hinge(s, low=self.low, high=self.high, beta=self.beta) * self.factor
        if self.post_process == "softclip_leaky":  return softclip_leaky(s, low=self.low, high=self.high, beta=self.beta, leaky=self.leak) * self.factor
        if self.post_process == "bounded_tanh":  return fcore.bounded_tanh(s, low=self.low, high=self.high, tau=self.tau) * self.factor
        if self.post_process == "positive_log_exp":  return positive_log_exp(s) * self.factor
        return s
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return x + processed_coef
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        processed_coef = self.process(self.coef)
        return x - processed_coef
 

# region Rotation
class Rotation(Op): 
    """ vecs for data-dependent rotations """
    unit_vecs: List[torch.Tensor] 
    patch_size: int | None = None
    scale_lst: List[Op] | None = None
    data_lst: List[torch.Tensor] | None = None

    def apply(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.rotate(x, update=update)
    
    def inverse(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.rotate_back(x, update=update)
    
    def apply_reflect(self, unit_vec: torch.Tensor, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Apply single Householder reflection: H = I - 2vv^T."""   
        with torch.set_grad_enabled(update):
            if self.patch_size is not None:
                B, C, out_dim = x.shape 
                G = out_dim // self.patch_size  # number of patches
                assert out_dim % self.patch_size == 0, "out_dim must be divisible by patch_size"
                x = x.contiguous().view(B, C, G, self.patch_size) 
                unit_vec = unit_vec.contiguous().view(B, C, G, self.patch_size)
                
                dot_prod = torch.sum(unit_vec * x, dim=-1, keepdim=True) # (v/||v||)ᵀx ; [B,D,1]
                result = x - 2 * unit_vec * dot_prod # x - 2(v/||v||)((v/||v||)ᵀx) ; [B,D,S]
                result = result.view(B, C, G * self.patch_size)
            else:
                dot_prod = torch.sum(unit_vec * x, dim=-1, keepdim=True) # (v/||v||)ᵀx ; [B,D,1]
                result = x - 2 * unit_vec * dot_prod # x - 2(v/||v||)((v/||v||)ᵀx) ; [B,D,S]
        return result

    def rotate(self, data: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Apply sequence of Householder reflections H_n(...H_2(H_1(x)))."""   
        with torch.set_grad_enabled(update):
            if self.scale_lst is None:
                for i, unit_vec in enumerate(self.unit_vecs):  
                    data = self.apply_reflect(unit_vec=unit_vec, x=data, update=update)  
            else:
                # SUM of blocks: y = Σ_b R_b S_b R_b^T x,  with 2 reflections per R_b
                nR = len(self.unit_vecs)
                assert nR % 2 == 0, "num_reflects must be even when using scales (2 reflections per rotation)."
                assert len(self.scale_lst) == nR // 2, "scale_lst must have length num_reflects//2."

                x0 = data                               # keep original input x
                contribs = []                           # collect R_b S_b R_b^T x

                # Walk forward; at each even index i (i%2==1), form a block
                for i, unit_vec in enumerate(self.unit_vecs):
                    data = self.apply_reflect(unit_vec=unit_vec, x=data, update=update)  # forward apply

                    if (i % 2 == 1):
                        # Build R_b^T x by applying the first (i..0) reflections in REVERSE to x0
                        w = x0
                        for j in range(i, -1, -1):
                            w = self.apply_reflect(unit_vec=self.unit_vecs[j], x=w, update=update)

                        # Apply the block scale in the local coords (your custom op)
                        z = (w | self.scale_lst[i // 2])

                        # Map back with R_b: apply the first (0..i) reflections in FORWARD order
                        yb = z
                        for j in range(0, i + 1):
                            yb = self.apply_reflect(unit_vec=self.unit_vecs[j], x=yb, update=update)

                        contribs.append(yb)

                data = torch.stack(contribs, dim=0).sum(dim=0)
 
        return data 

    def rotate_back(self, data: torch.Tensor, update: bool = True) -> torch.Tensor:  
        """Apply reverse sequence H_1(H_2(...H_n(x)))."""  
        nR = len(self.unit_vecs)
        with torch.set_grad_enabled(update): 
            for i, unit_vec in enumerate(reversed(self.unit_vecs)):   
                data = self.apply_reflect(unit_vec=unit_vec, x=data, update=update)    
        return data 
# endregion Rotation

# region Koopman
class Koopman(Op):
    a: torch.Tensor        # shape [..., d//2]
    b: torch.Tensor        # shape [..., d//2]
    mode: Literal['complex', 'real'] = "complex"

    def _split(self, x):
        d = x.shape[-1]
        z = einops.rearrange(x, '... (n e) -> ... n e', e=2)
        return z[..., 0], z[..., 1]               # real, imag

    def _merge(self, re, im):
        return einops.rearrange(
            torch.stack([re, im], dim=-1), '... n e -> ... (n e)')

    # forward: (a+ib)·(x_re+ix_im)
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'complex':
            x_even, x_odd = self._split(x)
            y_re =  self.a * x_even - self.b * x_odd
            y_im =  self.b * x_even + self.a * x_odd
            return self._merge(y_re, y_im)
        elif self.mode == 'real':
            return self.scale * x
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
  
    # inverse: divide by (a+ib)
    def inverse(self, x):
        if self.mode == 'complex':
            denom = self.a**2 + self.b**2 + 1e-9
            x_even, x_odd = self._split(x)
            y_re =  ( self.a * x_even + self.b * x_odd) / denom
            y_im =  (-self.b * x_even + self.a * x_odd) / denom
            return self._merge(y_re, y_im)
        elif self.mode == 'real':
            return x / (self.scale + 1e-9)
        else:
            raise ValueError(f"Invalid mode: {self.mode}") 
# endregion Koopman
    
# region States 
class States(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
    )
    x: torch.Tensor 
    z: torch.Tensor
    y: Optional[torch.Tensor] = None
    
    h_re: Optional[torch.Tensor] = None
    h_im: Optional[torch.Tensor] = None
    
    scale: Optional['Op'] = None  
    shift: Optional['Op'] = None
    rotation: Optional['Rotation'] = None 
    koopman: Optional['Koopman'] = None 
    
    resid_factor: Optional[nn.Parameter] = None
    temp_save_pre_scale: Optional[torch.Tensor] = None
     
    is_training: bool = True
      
    # --- an | pipe adapter ---------------------------------------
    def __or__(self, op):          # allow states | op style
        return op(self) 
    
    def set_field(self, field_name: str, field_value: torch.Tensor | Op) -> "States":
        return self.model_copy(update={field_name: field_value})
    
    def get_field(self, field_name: str) -> torch.Tensor | Op | None:
        return getattr(self, field_name)
    
    def set_x(self, x: torch.Tensor) -> "States":
         return self.model_copy(update={"x": x})
    
    def set_y(self, y: torch.Tensor) -> "States":
        return self.model_copy(update={"y": y})
    
    def set_z(self, z: torch.Tensor) -> "States":
        return self.model_copy(update={"z": z})
# endregion States


#region build_factories 
def build_factories(
    cfg: configs.FERNConfig,
    needed: set[fcore.LegoBricks] | None = None
) -> nn.ModuleDict:
    # If not provided, build everything
    needed = needed or set(cfg.factory_schemas.keys())

    modules = {
        fid.name: schema.build_factory()   # use .name for nn.ModuleDict key
        for fid, schema in cfg.factory_schemas.items()
        if fid in needed
    }
    return nn.ModuleDict(modules)
# endregion FERN Factory


# region ChannelMod
class ChannelMod(nn.Module):
    def __init__(self, channels, num_reflects, dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(channels, num_reflects, dim))
        self.b = nn.Parameter(torch.zeros(channels, num_reflects, dim))

    def forward(self, x):
        return x * self.a[None, :, :, :] + self.b[None, :, :, :]
# endregion ChannelMod

# region RotateFactory
class RotateFactory(nn.Module):
    """ Real-valued Data-dependent rotation matrix via Householder reflections """ 
    def __init__(
        self,
        schema: fcore.RotationSchema,
    ):
        super().__init__()   
        self.num_reflects = schema.num_reflects    
        self.channel_mod = (
            ChannelMod(schema.channels, schema.num_reflects, schema.out_dim) 
            if schema.adapt_params else None
        )
        self.schema = schema
             
        self.direction_nets  =  nn.Sequential( 
            # nn.Linear(schema.in_dim, schema.in_dim, bias=True),
            # nn.ReLU(),
            nn.Linear(schema.in_dim, schema.out_dim*schema.num_reflects, bias=False),
            # nn.Dropout(0.1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.1), 
            # fcore.DynamicTanh(channels=schema.channels, dim=schema.out_dim*schema.num_reflects, ),  
        ) 
        
        # for m in self.direction_nets.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if schema.with_scale_generation:
            self.scale_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(schema.in_dim, schema.out_dim, bias=False),
                    # nn.RMSNorm(schema.out_dim, elementwise_affine=True),
                    nn.ELU(),
                    nn.Linear(schema.out_dim, schema.out_dim, bias=False),
                    # nn.RMSNorm(schema.out_dim, elementwise_affine=True),
                    fcore.DynamicTanh(channels=schema.channels, dim=schema.out_dim, ),
                    nn.Linear(schema.out_dim, schema.out_dim, bias=False),
                )
                for i in range(schema.num_reflects // 2)
            ])
        else:
            self.scale_nets = None
            
        # self.params = nn.ParameterList([
        #         nn.Parameter(torch.ones(schema.channels, schema.out_dim))
        #         for i in range(schema.num_reflects // 2)
        #     ])
  
    def gen_unit_vecs(self, src: torch.Tensor) -> List[torch.Tensor]:
        raw = self.direction_nets(src) # (B,C, out_dim * R) 
        raw = raw.view(*raw.shape[:-1], self.num_reflects, -1) # (B, C, R, out_dim)
        if self.channel_mod is not None:
            raw = self.channel_mod(raw) 
        
        if self.schema.patch_size is not None:
            B, C, R, out_dim = raw.shape 
            # print(f"out_dim: {out_dim}, patch_size: {self.schema.patch_size}")
            assert out_dim % self.schema.patch_size == 0, "out_dim must be divisible by patch_size"
            G = out_dim // self.schema.patch_size  # number of patches
            raw = raw.contiguous().view(B, C, R, G, self.schema.patch_size) 
            raw = F.normalize(raw, p=2, dim=-1, eps=1e-9)  # unit per patch_size-d patch
            raw = raw.view(B, C, R, G * self.schema.patch_size)
        else:
            raw = F.normalize(raw, p=2, dim=-1, eps=1e-9)
        # raw = F.normalize(raw, p=2, dim=-1, eps=1e-9)
        # if self.channel_mod is not None:
        #     raw = self.channel_mod(raw)               
        #     raw = F.normalize(raw, p=2, dim=-1, eps=1e-9)
        return [raw[..., r, :] for r in range(self.num_reflects)]
    
    def gen_scale_lst(self, src: torch.Tensor) -> List[Op] | None:
        if self.scale_nets is not None:
            return [DiagScale(coef=self.scale_nets[i](src),
                             post_process='relu6_softcap_leaky',
                            low=-0.0,
                            high=8.0,
                            factor=1.0,
                            beta=0.8,
                            leak=0.1,
                            tau=None,
                             ) # relu6_softcap_leaky
                     for i in range(self.num_reflects // 2)]
            # return [DiagScale(coef=self.params[i],
            #                  post_process='relu6_softcap_leaky',
            #                  low=-0.0,
            #                  high=6.0,
            #                  factor=1.0,
            #                  beta=3.0,
            #                  leak=0.1,
            #                  tau=None,
            #                  ) # relu6_softcap_leaky
            #          for i in range(self.num_reflects // 2)]
        else:
            return None

    def forward(self, source: torch.Tensor) -> Rotation:
        return Rotation(unit_vecs=self.gen_unit_vecs(source), patch_size=self.schema.patch_size,
                        scale_lst=self.gen_scale_lst(source)) 
# endregion RotateFactory

# region HiddenFactory
class HiddenFactory(nn.Module):
    def __init__(self, schema: fcore.HiddenSchema):
        super().__init__()
        self.schema = schema 
        self.channels = schema.channels
        self.hidden_net_re = self.schema.build() 
        self.conv = nn.Conv1d(self.channels, self.channels, 
            kernel_size=1, groups=self.channels) if schema.use_conv else None
        self.params = self.schema.build_params() if schema.use_params else None
        if self.schema.use_history:
            self.register_buffer("history", torch.ones(1, self.channels, self.schema.hid_dim, dtype=torch.float32))
        else:
            self.history = None

        self.context_net = nn.Linear(self.schema.hid_dim*2, self.schema.hid_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.params is not None:
            x = x * self.params["scale"] + self.params["shift"]
        if self.conv is not None:
            x = self.conv(x)

        B = x.size(0)
        if self.history is not None:
            hist = self.history.expand(B, -1, -1)          # (B, C, H)
        else:
            hist = None
        if hist is not None:
            temp_h = self.hidden_net_re(x)
            temp_h = self.context_net(torch.cat([temp_h, hist], dim=-1))
        else:
            temp_h = self.hidden_net_re(x)
        h_re = temp_h                                   # (B, C, H)

        # --------- EMA update of buffer (NO GRAD, REDUCE OVER B) ----------
        if self.history is not None:
            with torch.no_grad():
                new_hist = h_re.mean(dim=0, keepdim=True)   # (1, C, H)
                self.history.mul_(0.9).add_(0.1 * new_hist) # in-place keeps it a buffer
                # alternatively: self.history.copy_(0.9*self.history + 0.1*new_hist)
            # -------------------------------------------------------------------
        # -------------------------------------------------------------------

        return h_re

class HiddenImagFactory(nn.Module):
    def __init__(self, schema: fcore.HiddenSchema):
        super().__init__()
        self.schema = schema
        self.hidden_net_im = self.schema.build()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_im = self.hidden_net_im(x)
        return h_im 
# endregion HiddenFactory 

# region CoefFactory
class CoefFactory(nn.Module):
    """ 
    The architecture is defined in the Pydantic config, and this class builds it.
    """
    def __init__(self, schema: fcore.CoefSchema):
        super().__init__()
        self.schema = schema   
        self.specs = self.schema.specs # dict[fcore.CoefName, HeadSpec] 
         
        self.foundry = self.schema.build() # Dict[str, nn.Module]
        self.params = self.schema.build_params() if self.schema.use_params else None 
        self.convs = self.schema.build_convs() if self.schema.use_convs else None
        # self.momentum_net = nn.Sequential(
        #     nn.Linear(self.schema.out_dim, self.schema.out_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.schema.out_dim, self.schema.out_dim),
        # )
        # print(f"schema.heads: [{self.schema.specs['scale']}]")
        
    #region Build Scale Op
      
    # (1-leak)*cap + leak*identity
    def _build_shift_op(self, raw_coefs: dict) -> Shift:
        return Shift(coef=raw_coefs[fcore.CoefName.SHIFT],
                     post_process='softclip_leaky',
                     low=-9.0,
                     high=9.0,
                     factor=1.0,
                     beta=0.8,
                     leak=0.1,
                     tau=None,
                     ) # softclip_leaky bounded_tanh none
     
    def _build_scale_op(self, raw_coefs: dict) -> Scale:
        structure = self.schema.structure
        # print(f"structure: {structure}",flush=True)
        coef_dim = raw_coefs[fcore.CoefName.SCALE].shape[-1]
        if fcore.CoefName.OFF_SCALE in raw_coefs:
            off_dim = raw_coefs[fcore.CoefName.OFF_SCALE].shape[-1] 
            if off_dim != coef_dim-1:
                raise ValueError(f"Off-scale coef shape mismatch: coef dim is {coef_dim}, off-scale coef dim should be {coef_dim-1}, but is {off_dim}")
        if fcore.CoefName.OFF_SIGN in raw_coefs:
            off_dim = raw_coefs[fcore.CoefName.OFF_SIGN].shape[-1]
            if off_dim != coef_dim-1:
                raise ValueError(f"Off-sign coef shape mismatch: coef dim is {coef_dim}, off-sign coef dim should be {coef_dim-1}, but is {off_dim}")
        
        scale = raw_coefs[fcore.CoefName.SCALE]
        
        
        if structure == 'diagonal':
            
            return DiagScale(coef=scale,
                             post_process='softclip_leaky',
                            low=-1.0,
                            high=2.5,
                            factor=1.0,
                            beta=0.8,
                            leak=0.1,
                            tau=None,
                             ) # relu6_softcap_leaky
 
        elif structure == 'complex':
            return ComplexScale(coef=scale,
                                 low=-2.5,
                                high=2.5,
                                factor=1.0,
                                beta=0.8,
                                leak=0.1,
                                tau=None,
                                post_process='softclip_leaky'
                                ) # softclip_leaky
  
        elif structure == 'tri_sym':
            return TriScale(
                coef=scale,
                lower_coef=raw_coefs[fcore.CoefName.OFF_SCALE],
                upper_coef=raw_coefs[fcore.CoefName.OFF_SCALE],
                low=-5.0,
                high=5.0,
                factor=1.0,
                beta=0.8,
                leak=0.1,
                tau=None,
                post_process='softclip_leaky'
            )

        elif structure == 'tri_anti':
            return TriScale(
                coef=scale,
                lower_coef=-raw_coefs[fcore.CoefName.OFF_SCALE],
                upper_coef=raw_coefs[fcore.CoefName.OFF_SCALE],
                low=-5.0,
                high=5.0,
                factor=1.0,
                beta=0.8,
                leak=0.1,
                tau=None,
                post_process='softclip_leaky'
            )
        
        elif structure == 'tri_free_sym':
            upper_coef = raw_coefs[fcore.CoefName.OFF_SCALE] * raw_coefs[fcore.CoefName.OFF_SIGN]
            return TriScale(
                coef=scale,
                lower_coef=upper_coef,
                upper_coef=upper_coef,
                low=-5.0,
                high=5.0,
                factor=1.0,
                beta=0.8,
                leak=0.1,
                tau=None,
                post_process='softclip_leaky'
            )
            
        elif structure == 'tri_free_anti':
            upper_coef = raw_coefs[fcore.CoefName.OFF_SCALE] * raw_coefs[fcore.CoefName.OFF_SIGN]
            return TriScale(
                coef=scale,
                lower_coef=-upper_coef,
                upper_coef=upper_coef,
                low=-5.0,
                high=5.0,
                factor=1.0,
                beta=0.8,
                leak=0.1,
                tau=None,
                post_process='softclip_leaky'
            )
        else:
            raise NotImplementedError(f"Unknown scaling structure: {structure}")
        
    # endregion _build_scaling_op
    def forward(
        self,
        h_re: torch.Tensor,                       # (B, C, H)
        *,
        want: tuple[fcore.CoefName | str, ...] | None = None,
        update: bool = True,
    ) -> Tuple[Scale, Shift]:  
        # ---- normalize 'want' to a set of fcore.CoefName ----
        if want is None:
            requested = {fcore.CoefName.SCALE, fcore.CoefName.SHIFT}
        else:
            requested = {
                w if isinstance(w, fcore.CoefName) else fcore.CoefName[w.upper()]
                for w in want
            }

        # ---- schema dims ----
        B, C, H = h_re.shape
        assert C == self.schema.channels and H == self.schema.hid_dim, \
            f"h_re (B,{C},{H}) != schema (C={self.schema.channels}, H={self.schema.hid_dim})"
  
        with torch.set_grad_enabled(update):
            raw_coefs = {}
            for head, spec in self.specs.items():
                # capacity + runtime gates
                # if hasattr(spec, "enabled") and spec.enabled is False:
                #     continue
                # if head not in requested:
                #     continue

                key = head.name  # Enum-centric string key
                net = self.foundry[key]
                if self.schema.use_params: 
                    if self.params is not None:
                        h_re = h_re - self.params[key][None, :, :]
                    else:
                        raise ValueError(f"Params are not built for {key}")
                if self.schema.use_convs:
                    if self.convs is not None:
                        h_re = self.convs[key](h_re)
                    else:
                        raise ValueError(f"Convs are not built for {key}")
                coef = net(h_re)
                # if head == fcore.CoefName.SCALE:
                #     coef_momentum = self.momentum_net(coef)
                #     coef_momentum = F.normalize(coef_momentum, p=2, dim=-1)
                #     dot_prod = torch.sum(coef_momentum * coef, dim=-1, keepdim=True)
                #     coef_momentum = coef_momentum * dot_prod
                #     coef = coef + 0.01*F.softplus(coef_momentum)
                
                
                # print(f"num_patches: {self.schema.patch.num_patches}")

                if self.schema.patch:
                    coef = stitch(
                        coef,
                        patch_len=self.schema.patch.patch_len,
                        stride=self.schema.patch.stride,
                        input_len=self.schema.out_dim,             
                        num_patches=self.schema.patch.num_patches,
                        window=self.schema.patch.window,       # "ones" | "hann" | None
                        normalize=self.schema.patch.normalize,  # "avg" | "sum"
                     )

                # ---- optional per-head conv/affine ----
                # if key in self.convs:
                #     coef = self.convs[key](coef)  # Conv1d depthwise over length axis
                # if key in self.params and self.params[key].shape[-1] == final_len:
                #     coef = coef * self.params[key][None, :, :]

                # ---- post thresholds/clips if present on spec.post ----
                post = getattr(spec, "post", None)
                if post is not None:
                    zthr = getattr(post, "zero_threshold", None)
                    clip_upper = getattr(post, "clip_upper_bound", None)  # FIXED: was clip_lower_bound
                    clip_lower = getattr(post, "clip_lower_bound", None)

                    if zthr is not None and zthr > 0.0:
                        # Keep your original semantics: zero values <= zthr
                        coef = F.threshold(coef, zthr, 0.0)

                    # Only clamp if at least one bound is provided
                    if clip_lower is not None or clip_upper is not None:
                        if clip_lower is None:
                            coef = torch.clamp(coef, max=float(clip_upper))
                        elif clip_upper is None:
                            coef = torch.clamp(coef, min=float(clip_lower))
                        else:
                            coef = torch.clamp(coef, min=float(clip_lower), max=float(clip_upper))

                     
                raw_coefs[head] = coef
                
                    
            scale = self._build_scale_op(raw_coefs)
            shift = self._build_shift_op(raw_coefs)
        return (scale, shift)
# endregion CoefFactory
 
#region patch_stitch  
def make_window(kind: Optional[Literal["ones","hann"]], L: int, device, dtype) -> torch.Tensor:
    if kind in (None, "ones"):
        w = torch.ones(L, device=device, dtype=dtype)
    elif kind == "hann":
        w = torch.hann_window(L, periodic=False, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown window kind: {kind}")
    return w.view(1, 1, 1, L)  # (1,1,1,L) for easy broadcasting
  

def stitch(
    patches: torch.Tensor,               # (B, C, P, L)
    *,
    patch_len: int,
    stride: int,
    input_len: int,
    num_patches: int,
    window: Optional[Literal["ones","hann"]] = "ones",
    normalize: Literal["avg","sum"] = "avg",
) -> torch.Tensor:
    """
    1D stitch implemented via F.fold by pretending height=1.
    This is compact and fast when P is large, but requires an exact sliding grid.
    """
    # print(patches.shape, num_patches, patch_len, input_len)
    # print()
    B, C, _ = patches.shape 
    patches = patches.view(B, C, num_patches, patch_len)
    P = num_patches
    L = patch_len

    # Compute the number of sliding blocks F.fold expects and enforce exact cover.
    expected_P = (input_len - patch_len) // stride + 1
    if (input_len - patch_len) % stride != 0 or expected_P != P:
        raise ValueError(
            f"fold requires exact grid: got P={P}, but input_len={input_len}, "
            f"patch_len={patch_len}, stride={stride} imply P={expected_P}."
        )

    dev, dt = patches.device, patches.dtype
    win = make_window(window, patch_len, dev, dt).squeeze(0).squeeze(0).squeeze(0)  # (L,)

    # Prepare the 'im2col' style tensor for fold: (N, C*kernel_h*kernel_w, n_blocks)
    # Our kernel_h=1, kernel_w=L. We multiply by window before folding.
    num = (patches * win)                                   # (B,C,P,L) * (L,)
    num = num.permute(0, 1, 3, 2).reshape(B, C * L, P)      # (B, C*L, P)

    out_num = F.fold(num, output_size=(1, input_len),
                     kernel_size=(1, L), stride=(1, stride))  # (B, C, 1, target_len)

    if normalize == "sum":
        return out_num.squeeze(2)

    # Denominator via folding window ones
    den_cols = torch.ones((B, C, P, L), device=dev, dtype=dt) * win  # (B,C,P,L)
    den_cols = den_cols.permute(0, 1, 3, 2).reshape(B, C * L, P)     # (B, C*L, P)
    out_den  = F.fold(den_cols, output_size=(1, input_len),
                      kernel_size=(1, L), stride=(1, stride))        # (B, C, 1, target_len)

    return (out_num / torch.clamp_min(out_den, 1e-12)).squeeze(2)
 
 
 # # --- Example Usage and Verification ---
# if __name__ == '__main__':
#     # Setup a simple case
#     B, C, S = 1, 1, 4  # Batch, Channels, Sequence Length
#     x = torch.tensor([1., 2., 3., 4.]).view(B, C, S)

#     # Define the tridiagonal matrix T:
#     # [ 2  3  0  0]
#     # [-1  4  1  0]
#     # [ 0 -2  5  2]
#     # [ 0  0 -3  6]
#     diag_main = torch.tensor([2., 4., 5., 6.]).view(B, C, S)
#     diag_lower = torch.tensor([-1., -2., -3.]).view(B, C, S - 1)
#     diag_upper = torch.tensor([3., 1., 2.]).view(B, C, S - 1)

#     # Calculate with our efficient function
#     y_efficient = apply_tridiagonal(x, diag_lower, diag_main, diag_upper)

#     # Manually calculate the expected result for verification
#     # y_0 = 2*x_0 + 3*x_1 = 2*1 + 3*2 = 8
#     # y_1 = -1*x_0 + 4*x_1 + 1*x_2 = -1*1 + 4*2 + 1*3 = -1 + 8 + 3 = 10
#     # y_2 = -2*x_1 + 5*x_2 + 2*x_3 = -2*2 + 5*3 + 2*4 = -4 + 15 + 8 = 19
#     # y_3 = -3*x_2 + 6*x_3 = -3*3 + 6*4 = -9 + 24 = 15
#     y_expected = torch.tensor([8., 10., 19., 15.]).view(B, C, S)

#     print("Input x:\n", x)
#     print("\nEfficient function output:\n", y_efficient)
#     print("\nExpected manual output:\n", y_expected)
#     print("\nAre they close?", torch.allclose(y_efficient, y_expected))