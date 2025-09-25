import study.FERN_core as fcore
from enum import StrEnum, auto
from pydantic import ConfigDict, model_validator
from pydantic import BaseModel, PositiveInt, ConfigDict, Field, computed_field, model_validator
from typing import Union, Literal, List, Tuple, Optional, Iterable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import study.FERN_util as fru
import study.FERN_gen as fgen
import sys


#region StrEnum
class ParamKey(StrEnum):
    """Abstract names for which parameter head you need from your factory dict."""
    HID = "HID"    # hidden from src
    SS  = "SS"     # scale/shift params
    ROT = "ROT"    # rotation params
    KOO = "KOO"    # koopman params
# endregion StrEnum

# region G and T 
class Fields(StrEnum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    H_RE = 'h_re'
    H_IM = 'h_im'



class G(fgen.Op): 
    
    field: Fields
    fac: Union[fgen.CoefFactory, fgen.RotateFactory, fgen.HiddenFactory] = None
    augment: Literal['none','odd','even','even_evens','even_odds','odd_odds',
        'odd_evens', 'boot_odd_keep_even', 'boot_even_keep_odd'] = 'none'
     
    def apply(self, state: fgen.States)->fgen.States:
        # --- 1. Get the source tensor ---
        source = getattr(state, self.field)

        # --- 2. Apply augmentation if specified ---
        if self.augment == 'boot_odd_keep_even':
            source = fru.bootstrap_odds_keep_evens(source)
        elif self.augment == 'boot_even_keep_odd':
            source = fru.bootstrap_even_keep_odd(source)
        elif self.augment != 'none':
            source = fru.keep_positions(x=source, keep_mode=self.augment)
            
            
        if isinstance(self.fac, fgen.CoefFactory): 
            scale, shift = self.fac(source)          # unpack (h_re)
            state.scale = scale
            state.shift = shift
            return state
        elif isinstance(self.fac, fgen.RotateFactory):
            rotation = self.fac(source)
            state.rotation = rotation
            return state
        elif isinstance(self.fac, fgen.HiddenFactory): 
            h_re =  self.fac(source) 
            state.h_re = h_re
            # state.h_im = h_im
            return state
        else:
            raise ValueError(f"Unknown factory: {type(self.fac)}")


class TOp(StrEnum):
    """The actions your T(...) op can apply to the destination field."""
    ROTATION = auto()
    SCALE = auto() 
    SHIFT = auto()
    KOOPMAN = auto()   
   
   
class T(fgen.Op):
    """ APPLIES a pre-generated operator from the States to a data field. """
    field: Fields
    op_name: TOp
    inverse: bool = False    # Whether to apply the inverse operation

    def apply(self, state: fgen.States) -> fgen.States:   
        op   = getattr(state,   self.op_name)
        target = getattr(state, self.field)
        new_target = target | op
        
        if self.op_name in [TOp.SCALE]:
            # if state.resid_factor is None:
            #     final_target = new_target + target
            # else:
            #     final_target = new_target + state.resid_factor * target
            final_target = new_target + target + 1e-5 #TODO
            # factor = 1.0
            # final_target = factor * new_target + (1 - factor) * target # new_target
            # state.temp_save_pre_scale = target
        # elif self.op_name in [TOp.SHIFT]:
        #     factor = 0.9
        #     if state.temp_save_pre_scale is None:
        #         final_target = new_target
        #     else:
        #         final_target = factor * new_target + (1 - factor) * state.temp_save_pre_scale 
        #         state.temp_save_pre_scale = None
        else:
            final_target = new_target
        setattr(state, self.field, final_target)
        return state
# endregion G and T
class I(fgen.Op):
    """Identity operator"""
    def apply(self, state):  # just returns the states unchanged
        return state

#region ProbG
class ProbG(fgen.Op):
    """A Op operator that probabilistically chooses one of several G operators to apply.
    A Op operator that probabilistically chooses one of several G operators to apply.
    During evaluation (is_training=False), it always chooses the first operator in the list.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    choices: List[Tuple[G, float]]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ProbabilisticG":
        """Ensures the probabilities sum to 1.0."""
        total_prob = sum(prob for _, prob in self.choices)
        if not torch.isclose(torch.tensor(total_prob), torch.tensor(1.0)):
            raise ValueError(f"Probabilities must sum to 1.0, but got {total_prob}")
        return self

    def apply(self, state: fgen.States) -> fgen.States:
        """
        Randomly selects and applies a G operator during training.
        Applies the first G operator during evaluation.
        """
        # During inference/evaluation, always use the first (default) option for reproducibility.
        if not state.is_training:
            default_op, _ = self.choices[0]
            return default_op.apply(state)

        # During training, randomly select an operator based on the probabilities.
        p = torch.rand(1).item()
        cumulative_prob = 0.0
        for g_op, prob in self.choices:
            cumulative_prob += prob
            if p < cumulative_prob:
                return g_op.apply(state)
        
        # Fallback to the last operator in case of floating point inaccuracies
        last_op, _ = self.choices[-1]
        return last_op.apply(state)

#region CoefMonitorOp
class CoefMonitorOp(fgen.Op):
    """
    Unified coefficient monitor for SCALE or SHIFT.

    - Zero-fraction: uses abs(x) <= zero_eps for both scale and shift.
    - Bucketed fractions: uses 'bins' (right edges) for |x|.
      e.g., bins=(1,10,30,50) => <1, 1–10, 10–30, 30–50, ≥50
    - Prints batch, run (cumulative), and EMA stats.
    """
    message: Optional[str] = None
    coef: Literal["scale", "shift"] = "scale"

    # How to fetch coefficients
    use_processed: bool = True          # call .process(coef) if available

    # Zero detection
    zero_eps: float = 1e-4              # abs(x) <= zero_eps counts as "zero"

    # Bucketization (right edges)
    bins: Sequence[float] = (1.0, 10.0, 30.0, 50.0)

    # EMA smoothing
    ema_momentum: float = 0.9

    # ---------- running stats ----------
    calls: int = 0
    total_elems_seen: int = 0

    zero_count: int = 0
    running_zero_frac: float = 0.0
    ema_zero_frac: float = 0.0

    total_seen_for_bins: int = 0
    running_counts: List[int] = Field(default_factory=list)
    running_fracs: List[float] = Field(default_factory=list)
    ema_fracs: List[float] = Field(default_factory=list)

    # ---------- helpers ----------
    def flush_print(self, text: str):
        print(text, flush=True)
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def format_one_line(self, v: torch.Tensor, head: int = 3, tail: int = 1) -> str:
        v = v.flatten()
        d = v.numel()
        if d == 0:
            return "[]"
        if d <= head + tail:
            vals = ", ".join(f"{x:.3g}" for x in v.tolist())
            return f"[{vals}]"
        head_vals = ", ".join(f"{x:.3g}" for x in v[:head].tolist())
        tail_vals = ", ".join(f"{x:.3g}" for x in v[-tail:].tolist())
        return f"[{head_vals}, …, {tail_vals}]"

    def reduce_to_D(self, t: torch.Tensor) -> torch.Tensor:
        # mean over all dims except last -> [D]
        if t.dim() >= 2:
            return t.mean(dim=tuple(range(t.dim() - 1)))
        return t

    def fmt_edge(self, e: float) -> str:
        ei = int(round(float(e)))
        return str(ei) if abs(e - ei) < 1e-6 else f"{e:g}"

    def build_labels(self) -> List[str]:
        labels: List[str] = []
        prev = 0.0
        for e in self.bins:
            left = self.fmt_edge(prev)
            right = self.fmt_edge(e)
            labels.append(f"{left}–{right}" if prev > 0 else f"<{right}")
            prev = float(e)
        labels.append(f"≥{self.fmt_edge(self.bins[-1])}")
        return labels

    def get_coef_tensor(self, s: fgen.States) -> torch.Tensor:
        if self.coef == "scale":
            if s.scale is None:
                raise RuntimeError("CoefMonitorOp: scale not available in state.")
            raw = s.scale.coef
            if self.use_processed and hasattr(s.scale, "process"):
                return s.scale.process(raw)
            return raw
        if self.coef == "shift":
            if s.shift is None:
                raise RuntimeError("CoefMonitorOp: shift not available in state.")
            raw = s.shift.coef
            if self.use_processed and hasattr(s.shift, "process"):
                return s.shift.process(raw)
            return raw
        raise NotImplementedError(self.coef)

    # ---------- main ----------
    def apply(self, state: fgen.States) -> fgen.States:
        try:
            with torch.no_grad():
                msg = self.message or f"{self.coef.upper()}_MON"

                t = self.get_coef_tensor(state).detach()
                total = t.numel()
                if total == 0:
                    self.flush_print(f"{msg}: <empty tensor>")
                    return state

                # One-line feature preview
                v_mean = self.reduce_to_D(t).to("cpu", dtype=torch.float32)
                one_line = self.format_one_line(v_mean, head=3, tail=1)

                # ----- zero fraction (shared for scale & shift) -----
                zeros = (t.abs() <= self.zero_eps).sum().item()
                zero_frac = zeros / total

                # ----- bucketize |coef| with shared bins -----
                edges = torch.as_tensor(self.bins, device=t.device, dtype=t.dtype)
                bidx = torch.bucketize(t.abs().reshape(-1), edges, right=False)          # 0..len(edges)
                counts = torch.bincount(bidx, minlength=edges.numel() + 1)
                fracs = (counts.float() / total).tolist()
                labels = self.build_labels()

                # init running containers on first call
                if not self.running_counts:
                    self.running_counts = [0]   * len(labels)
                    self.running_fracs  = [0.0] * len(labels)
                    self.ema_fracs      = [0.0] * len(labels)

                # update totals
                self.calls += 1
                self.total_elems_seen += total
                self.total_seen_for_bins += total

                self.zero_count += zeros
                self.running_zero_frac = self.zero_count / max(self.total_elems_seen, 1)

                m = self.ema_momentum
                self.ema_zero_frac = m * self.ema_zero_frac + (1 - m) * zero_frac

                for i in range(len(labels)):
                    c_i = int(counts[i].item())
                    self.running_counts[i] += c_i
                    self.running_fracs[i]  = self.running_counts[i] / max(self.total_seen_for_bins, 1)
                    self.ema_fracs[i]      = m * self.ema_fracs[i] + (1 - m) * fracs[i]

                batch_bins = " ".join(f"{lab}={100*f:.1f}%" for lab, f in zip(labels, fracs))
                run_bins   = " ".join(f"{lab}={100*f:.1f}%" for lab, f in zip(labels, self.running_fracs))
                ema_bins   = " ".join(f"{lab}={100*f:.1f}%" for lab, f in zip(labels, self.ema_fracs))

                self.flush_print(
                    f"{msg}: {one_line} "
                    f"| batch 0={zero_frac*100:.1f}% {batch_bins} "
                    f"| run 0={self.running_zero_frac*100:.1f}% {run_bins} "
                    f"| ema 0={self.ema_zero_frac*100:.1f}% {ema_bins} "
                    f"(calls={self.calls})"
                )

        except Exception as e:
            self.flush_print(f"[CoefMonitorOp ERROR] {e}")

        return state
     
# endregion CoefMonitorOp
# endregion ProbG

#region Geom
class Geom(StrEnum):
    """A enum for the geometric operations"""
    SHIFT_ONLY     = auto()
    SCALE_ONLY     = auto()
    SCALE_SHIFT    = auto()        # scale then shift
    PROB_SCALE_SHIFT = auto()      # probabilistic scale then shift
    SHIFT_SCALE    = auto()        # shift then scale  (rare but handy)
    R    = auto()
    R_SCALE_SHIFT_RB = auto()
    R_SCALE_RB_SHIFT = auto()
    R_SCALE_SHIFT = auto()
    R_KOOP_RB = auto()
    R_KOOP = auto()
    MOBIUS         = auto()
    KOOP   = auto()
    R_KOOP_RB_SHIFT = auto() 
    # CASCADE = auto()
# endregion Geom

#region GStep and TStep 
class BaseStep(BaseModel):
    model_config = ConfigDict()
    
class GStep(BaseStep):
    """GStep (“generate”) Says: “Run a generator G(...) on either the source field ('src') 
    or the hidden field ('h_re'), using a particular ParamKey.”"""
    type: Literal["G", "ProbG"] = "G"
    # "src" means: use the source field string passed at compile time
    # "h_re" means: use the hidden real field
    field: Literal["src", "h_re"]
    param: ParamKey
    # Optional per-param overrides; None → inherit from compile_chain(version) → "v0"
    ss_version: str | None = None
    rot_version: str | None = None
    hid_version: str | None = None
    koo_version: str | None = None
    probg_choices: List[Tuple[str, float]] = None
    
    @model_validator(mode="after")
    def validate_probg_choices(self) -> "GStep":
        if self.type == "ProbG":
            if self.probg_choices is None:
                self.probg_choices = [
                    ('none', 0.5), # boot_odd_keep_even
                    ('boot_odd_keep_even', 0.25), # boot_odd_keep_even # odd
                    ('even', 0.25), # boot_odd_keep_even
                ]
        return self

class TStep(BaseStep):
    """TStep (“transform”) Says: “Apply a transform T(...) on the destination field ('dst') 
    with a given TOp, optionally with inverse=True (only valid for rotation).”"""
    type: Literal["T"] = "T"
    field: Literal["dst"]  # we always transform the destination field
    op_name: TOp
    inverse: bool = False
    
class PrintStep(BaseStep): 
    type: Literal["Print"] = "Print"
    message: str | None = None 
    coef: Literal['scale', 'shift', 'koopman', 'rotation'] | None = None
    use_processed: bool = True
    bins: Sequence[float] = (1.0, 10.0, 30.0, 50.0)
    
     
# endregion GStep and TStep

#region StepSpec and ChainSpec
StepSpec = Union[GStep, TStep, PrintStep]

class ChainSpec(BaseModel):
    """ 
    A Pydantic model that wraps: 
    name: for logging, 
    steps: a list of GStep | TStep | PrintStep, 
    optional defaults: version and augment (so you can omit them at call sites).
    """
    name: str
    steps: List[StepSpec]

    # Optional compile-time defaults (don’t have to set here)
    version: Optional[str] = None
    augment: Literal[
        "none","odd","even","even_evens","even_odds",
        "odd_odds","odd_evens","boot_odd_keep_even"
    ] = "none"

# endregion StepSpec and ChainSpec

#region Key builders
# --- key builders (same logic, centralized) ---
def _hid_key(src: str, version: str) -> str: # "HID_GIVEN_{SRC}"
    return f"HID_GIVEN_{src.upper()}_{version.upper()}"

def _ss_key(dst: str, src: str, version: str) -> str: # "SS_{DST}_GIVEN_{SRC}[_V]" 
    return f"SS_{dst.upper()}_GIVEN_{src.upper()}_{version.upper()}"

def _rot_key(dst: str, version: str) -> str: # "ROT_IN_{DST}"
    return f"ROT_IN_{dst.upper()}_{version.upper()}"

def _koo_key(dst: str, src: str, version: str) -> str: # "KOO_{DST}_GIVEN_{SRC}[_V]"
    return f"KOO_{dst.upper()}_GIVEN_{src.upper()}_{version.upper()}"

def _resolve_factory_key(pk: ParamKey, *, src: str, dst: str, version: str) -> str:
    """All the “factory key naming” lives here, so if you change naming or add versioning later, you do it once."""
    if pk is ParamKey.HID: return _hid_key(src, version)
    if pk is ParamKey.SS:  return _ss_key(dst, src, version)
    if pk is ParamKey.ROT: return _rot_key(dst, version)
    if pk is ParamKey.KOO: return _koo_key(dst, src, version)
    raise KeyError(pk)

def pick_ver(pk: ParamKey, step: GStep, default: str | None) -> str:
    if pk is ParamKey.SS:
        return (step.ss_version or default or "v0")
    if pk is ParamKey.ROT:
        return (step.rot_version or default or "v0")
    if pk is ParamKey.HID:
        return (step.hid_version or default or "v0")
    if pk is ParamKey.KOO:
        return (step.koo_version or default or "v0")
    raise KeyError(pk)
# endregion Key builders

#region compile_chain
# --- compile_chain: ChainSpec -> fru.Op ---
def compile_chain(
    spec: ChainSpec,
    *,
    src: str,
    dst: str,
    fac: "nn.ModuleDict",
    version: str | None = None,     # chain-level default
    augment: Optional[str] = None,
) -> "fgen.Op":
    """Takes a ChainSpec + runtime context and returns executable plan:
    Start with plan = I().
    For each step:
    GStep: resolve field ('src' → actual src, otherwise 'h_re'), compute the factory key from ParamKey, and append G(...).
    Augment is applied only when field == src (your current design). 
    TStep: append T(field=dst, op_name=..., inverse=...). 
    Return the composed fgen.Op."""
    
    plan = I()
    aug = augment if augment is not None else spec.augment
     
      
    for step in spec.steps:
        if step.type == "G" or step.type == "ProbG": 
            field = src if step.field == "src" else "h_re"
            v = pick_ver(step.param, step, version)
            key = _resolve_factory_key(step.param, src=src, dst=dst, version=v)
            if field == src:
                if step.type == "G":
                    plan = plan | G(field=field, fac=fac[key], augment=aug)
                elif step.type == "ProbG":
                    plan = plan | ProbG(choices=[
                        (G(field=field, fac=fac[key], augment=aug), prob) for aug, prob in step.probg_choices
                    ]) 
            else:
                plan = plan | G(field=field, fac=fac[key])
        elif step.type == "T":  # T
            assert step.field == "dst"
            plan = plan | (
                T(field=dst, op_name=step.op_name, inverse=True)
                if step.inverse else
                T(field=dst, op_name=step.op_name)
            )
        elif step.type == "Print":  # Print
            plan = plan | CoefMonitorOp(message=step.message, coef=step.coef, 
                                        use_processed=step.use_processed, bins=step.bins)
    return plan
# endregion Compiler

#region FLOW BUILDER
# Geom -> ChainSpec registry (one-liners)
"""A dict mapping your Geom enum → ChainSpec."""
RECIPES: dict[Geom, ChainSpec] = {
    Geom.SCALE_SHIFT: ChainSpec(
        name="SCALE_SHIFT",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0"),
            # PrintStep(message="SCALE_SHIFT's scale", field=None, coef="scale",
            #           use_processed=True, bins=(1.0, 3.0, 6.0, 10.0)),
            # PrintStep(message="SCALE_SHIFT's shift", field=None, coef="shift", 
            #           use_processed=False, bins=(1.0, 2.0, 5.0, 10.0)),
            GStep(field="h_re", param=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.SHIFT), #TODO
        ],
    ),
    Geom.PROB_SCALE_SHIFT: ChainSpec(
        name="PROB_SCALE_SHIFT",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0", type="ProbG"),
            GStep(field="h_re", param=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ), 
    Geom.SCALE_ONLY: ChainSpec(
        name="SCALE_ONLY",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0"),
            GStep(field="h_re", param=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
        ],
    ),
    Geom.SHIFT_ONLY: ChainSpec(
        name="SHIFT_ONLY",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0"),
            GStep(field="h_re", param=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ),
    Geom.R_SCALE_RB_SHIFT: ChainSpec(
        name="R_SCALE_RB_SHIFT",
        steps=[
            GStep(field="src",   param=ParamKey.HID, hid_version="v0", type="G"), # ProbG
            GStep(field="h_re",  param=ParamKey.SS),
            GStep(field="h_re",  param=ParamKey.ROT, rot_version="v0"),
            # PrintStep(message="R_SCALE_RB_SHIFT's scale", field=None, coef="scale", 
            #           use_processed=True, bins=(1.0, 2.0, 5.0, 9.0)),
            # PrintStep(message="R_SCALE_RB_SHIFT's shift", field=None, coef="shift", 
            #           use_processed=False, bins=(1.0, 2.0, 5.0, 10.0)),
            TStep(field="dst",   op_name=TOp.ROTATION),
            TStep(field="dst",   op_name=TOp.SCALE),
            TStep(field="dst",   op_name=TOp.ROTATION, inverse=True),
            TStep(field="dst",   op_name=TOp.SHIFT),
        ],
    ),
    Geom.R_SCALE_SHIFT: ChainSpec(
        name="R_SCALE_SHIFT",
        steps=[
            GStep(field="src",   param=ParamKey.HID, hid_version="v0"),
            GStep(field="h_re",  param=ParamKey.SS),
            GStep(field="h_re",  param=ParamKey.ROT, rot_version="v0"),
            TStep(field="dst",   op_name=TOp.ROTATION),
            TStep(field="dst",   op_name=TOp.SCALE),
            TStep(field="dst",   op_name=TOp.SHIFT),
        ],
    ),
    Geom.R: ChainSpec(
        name="R",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0"),
            GStep(field="h_re", param=ParamKey.ROT),
            TStep(field="dst", op_name=TOp.ROTATION),
        ],
    ),
    Geom.R_KOOP_RB_SHIFT: ChainSpec(
        name="R_KOOP_RB_SHIFT",
        steps=[
            GStep(field="src",  param=ParamKey.HID, hid_version="v0"),
            GStep(field="h_re", param=ParamKey.KOO, koo_version="v0"),
            GStep(field="h_re", param=ParamKey.ROT, rot_version="v0"),
            TStep(field="dst",  op_name=TOp.ROTATION),
            TStep(field="dst",  op_name=TOp.KOOPMAN),
            TStep(field="dst",  op_name=TOp.ROTATION, inverse=True),
            TStep(field="dst",  op_name=TOp.SHIFT),
        ],
    ),
    Geom.KOOP: ChainSpec(
        name="KOOP",
        steps=[
            GStep(field="src", param=ParamKey.HID, hid_version="v0"),
            TStep(field="dst", op_name=TOp.KOOPMAN),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ),
    # Geom.CASCADE: ChainSpec(
    #     name="CASCADE",
    #     steps=[
    #         GStep(field="src", param=ParamKey.HID, hid_version="v0"),
    #         GStep(field="h_re",  param=ParamKey.SS),
    #         GStep(field="h_re", param=ParamKey.ROT, rot_version="v0"),
    #         TStep(field="dst", op_name=TOp.CASCADE),
    #     ],
    # ),
}
#endregion FLOW BUILDER

# Pretty-printer: spec -> "I | G[src:HID] | G[h_re:SS_Y_GIVEN_Z_V2] | R | Sc | Ri | Sh"
def render_chain(spec: ChainSpec, *, src: str, dst: str, version: str | None) -> str:
    parts = ["I"]
    for st in spec.steps:
        if st.type == "G":
            f = "src" if st.field == "src" else "h_re"
            parts.append(f"G[{f}:{st.param.name}]")
        else:
            op = st.op_name
            parts.append("~R" if (op == "rotation" and st.inverse) else
                         {"rotation":"R","scale":"Sc","complex_scale":"CSc","shift":"Sh","koopman":"K"}[op])
    return " | ".join(parts)

# Example in logs:
# print(render_chain(RECIPES[Geom.R_SCALE_RB_SHIFT], src="z", dst="y", version="v2"))
# I | G[src:HID] | G[h_re:SS] | G[h_re:ROT] | R | Sc | ~R | Sh

