# =========================
# file: fern_imperative.py
# =========================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class FERNConfig:
    # Core
    seq_len: int
    pred_len: int
    channels: int
    device: str = "cuda"

    # Model dims
    dim_augment: int = 128
    dim_hidden: int = 128

    # Encoder
    num_encoding_layers: int = 5
    householder_reflects_data: int = 4

    # Which encoder layers use complex (2x2 block) vs diagonal scaling
    # 0-based indices among the 5 encoding steps
    complex_layers_x: List[int] = field(default_factory=lambda: [1, 3])
    complex_layers_z: List[int] = field(default_factory=lambda: [1, 3])

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def softclip_hinge(x: torch.Tensor, low: float, high: float, beta: float = 0.8) -> torch.Tensor:
    return x - F.softplus(x - high, beta=beta) + F.softplus(low - x, beta=beta)

def softclip_leaky(x: torch.Tensor, low: float, high: float, beta: float = 0.8, leaky: float = 0.05) -> torch.Tensor:
    y = softclip_hinge(x, low, high, beta=beta)
    return leaky * x + (1.0 - leaky) * y

def relu6_softcap_leaky(x: torch.Tensor, high: float = 6.0, beta: float = 1.0, leak: float = 0.05) -> torch.Tensor:
    # Always >= 0, capped ~high with soft knee, leaky identity outside cap
    x0 = F.softplus(x)                                       # >= 0
    y_cap = x0 - F.softplus(x0 - high, beta=beta)            # soft cap near 'high'
    return y_cap + leak * (x0 - y_cap)                       # leaky blend

def positive_log_exp(x: torch.Tensor, max_log: float = 3.0) -> torch.Tensor:
    # strictly positive ( > 0 ), smooth cap
    log_output = max_log * torch.tanh(x)
    return torch.expm1(log_output) + 1.0

def sample_base(shape, device, dtype):
    return torch.randn(shape, device=device, dtype=dtype) * 0.1

# ------------------------------------------------------------
# Blocks
# ------------------------------------------------------------
class HiddenNetwork(nn.Module):
    """Space-specific trunk ϕ: R^L -> R^H (shared across heads)."""
    def __init__(self, in_dim: int, hid_dim: int, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias), nn.Identity(),
            nn.Linear(hid_dim, hid_dim, bias=bias), nn.ELU(alpha=1.0),
            nn.Linear(hid_dim, hid_dim, bias=bias), nn.ELU(alpha=1.0),
            nn.Linear(hid_dim, hid_dim, bias=bias), nn.Softshrink(lambd=0.07),
            nn.Linear(hid_dim, hid_dim, bias=bias), nn.LogSigmoid(),
            nn.Linear(hid_dim, hid_dim, bias=bias), nn.ELU(alpha=1.0),
        )

    def forward(self, x):
        return self.net(x)

class ScaleHead(nn.Module):
    """Generates raw scale parameters (pre-processing applied later)."""
    def __init__(self, hid_dim: int, out_dim: int, channels: int, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.ELU(1.0),
            nn.Linear(out_dim, out_dim, bias=bias),
        )

    def forward(self, h):
        return self.net(h)

class ShiftHead(nn.Module):
    """Generates shift parameters."""
    def __init__(self, hid_dim: int, out_dim: int, channels: int, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.ELU(1.0),
            nn.Linear(out_dim, out_dim, bias=bias),
        )

    def forward(self, h):
        return self.net(h)

class RotationGenerator(nn.Module):
    """Householder vectors for U(z): (B,D,H)->(B,D,R,P)."""
    def __init__(self, hid_dim: int, out_len: int, num_reflects: int, bias: bool = False):
        super().__init__()
        self.R = num_reflects
        self.P = out_len
        self.dir = nn.Sequential(
            nn.Linear(hid_dim, out_len * num_reflects, bias=bias),
            nn.ReLU(),
        )

    def forward(self, h_bdh: torch.Tensor) -> List[torch.Tensor]:
        B, D, H = h_bdh.shape
        raw = self.dir(h_bdh.view(B*D, H)).view(B, D, self.R, self.P)
        v = F.normalize(raw, p=2, dim=-1, eps=1e-9)
        return [v[:, :, i, :] for i in range(self.R)]

# ------------------------------------------------------------
# Transform ops (residual SCALE like T.apply, SHIFT additive)
# ------------------------------------------------------------
# bounds
DIAG_S_HIGH = 6.0       # nonnegative via relu6_softcap_leaky
COMPLEX_S_LOW, COMPLEX_S_HIGH = -2.5, 2.5
SHIFT_LOW, SHIFT_HIGH = -9.0, 9.0
BETA, LEAK = 0.8, 0.1

def apply_block_diagonal(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """
    Complex-like 2x2 blocks along the last dim:
      y_even = a*x_even - b*x_odd
      y_odd  = b*x_even + a*x_odd
    where coef alternates (a,b,a,b,...).
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Block-diagonal op requires even last dim; got {dim}")
    x2 = rearrange(x, 'b d (n e) -> b d n e', e=2)
    ce = rearrange(coef, 'b d (n e) -> b d n e', e=2)
    x_even, x_odd = x2[..., 0], x2[..., 1]
    a, b = ce[..., 0], ce[..., 1]
    y_even = a * x_even - b * x_odd
    y_odd  = b * x_even + a * x_odd
    return rearrange(torch.stack([y_even, y_odd], dim=-1), 'b d n e -> b d (n e)')

def apply_reflection(v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = (v * y).sum(dim=-1, keepdim=True)
    return y - 2.0 * v * dot

def apply_rotation(y: torch.Tensor, vecs: List[torch.Tensor]) -> torch.Tensor:
    for v in vecs:
        y = apply_reflection(v, y)
    return y

def apply_rotation_inverse(y: torch.Tensor, vecs: List[torch.Tensor]) -> torch.Tensor:
    for v in reversed(vecs):
        y = apply_reflection(v, y)
    return y

def split_complex(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = rearrange(x, 'b d (n e) -> b d n e', e=2)
    return z[..., 0], z[..., 1]

def merge_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return rearrange(torch.stack([re, im], dim=-1), 'b d n e -> b d (n e)')

def apply_K(y: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    re, im = split_complex(y)
    yre = a[None, :, :] * re - b[None, :, :] * im
    yim = b[None, :, :] * re + a[None, :, :] * im
    return merge_complex(yre, yim)

def apply_K_T(y: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    re, im = split_complex(y)
    yre = a[None, :, :] * re + b[None, :, :] * im
    yim = -b[None, :, :] * re + a[None, :, :] * im
    return merge_complex(yre, yim)

def process_scale_diag_nonneg(raw: torch.Tensor) -> torch.Tensor:
    return relu6_softcap_leaky(raw, high=DIAG_S_HIGH, beta=1.0, leak=LEAK)  # >= 0

def process_scale_complex(raw: torch.Tensor) -> torch.Tensor:
    return softclip_leaky(raw, low=COMPLEX_S_LOW, high=COMPLEX_S_HIGH, beta=BETA, leaky=LEAK)

def process_shift(raw: torch.Tensor) -> torch.Tensor:
    return softclip_leaky(raw, low=SHIFT_LOW, high=SHIFT_HIGH, beta=BETA, leaky=LEAK)

def apply_scale_residual(x: torch.Tensor, raw_scale: torch.Tensor, structure: Literal["diagonal","complex"]) -> torch.Tensor:
    """
    Residual SCALE like your T.apply:
      final = (x | SCALE) + x
    where (x | SCALE) = s_proc ⊙ x  (diag) or R_s(x) (complex block).
    """
    if structure == "diagonal":
        s_proc = process_scale_diag_nonneg(raw_scale)     # >= 0
        scaled = x * s_proc
    else:  # "complex"
        s_proc = process_scale_complex(raw_scale)
        scaled = apply_block_diagonal(x, s_proc)
    return scaled + x  # residual style

def apply_shift_additive(x: torch.Tensor, raw_shift: torch.Tensor) -> torch.Tensor:
    """
    SHIFT op returns x + t_proc; no extra residual in T.apply for SHIFT.
    """
    t_proc = process_shift(raw_shift)
    return x + t_proc

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class FERN(nn.Module):
    def __init__(self, cfg: FERNConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        S = cfg.seq_len
        A = cfg.dim_augment
        P = cfg.pred_len
        H = cfg.dim_hidden

        # Evenness guards for complex ops
        if any(i in cfg.complex_layers_x for i in range(cfg.num_encoding_layers)) and (S % 2 != 0):
            raise ValueError(f"seq_len must be even for complex layers on x; got {S}")
        if any(i in cfg.complex_layers_z for i in range(cfg.num_encoding_layers)) and (A % 2 != 0):
            raise ValueError(f"dim_augment must be even for complex layers on z; got {A}")
        if P % 2 != 0:
            raise ValueError(f"pred_len must be even for K; got {P}")

        # Shared trunks ϕ_x, ϕ_z (space-specific, shared across heads)
        self.phi_x = HiddenNetwork(S, H, bias=False)
        self.phi_z = HiddenNetwork(A, H, bias=False)

        # Encoder heads (distinct per layer)
        self.x2z_scales = nn.ModuleList([ScaleHead(H, A, C, bias=False) for _ in range(cfg.num_encoding_layers)])
        self.x2z_shifts = nn.ModuleList([ShiftHead(H, A, C, bias=False) for _ in range(cfg.num_encoding_layers)])
        self.z2x_scales = nn.ModuleList([ScaleHead(H, S, C, bias=False) for _ in range(cfg.num_encoding_layers)])
        self.z2x_shifts = nn.ModuleList([ShiftHead(H, S, C, bias=False) for _ in range(cfg.num_encoding_layers)])

        # Ellipsoidal transport layer
        self.U = RotationGenerator(H, P, cfg.householder_reflects_data, bias=False)
        N = P // 2
        self.K_a = nn.Parameter(torch.ones(C, N))   # fixed learnable per-channel complex spin (a,b)
        self.K_b = nn.Parameter(torch.zeros(C, N))

        # Decoder heads from ϕ_z(z_final)
        self.scale_y_head = ScaleHead(H, P, C, bias=False)   # Λ(z): must be positive -> enforce below
        self.shift_y_head = ShiftHead(H, P, C, bias=False)   # t(z)

    def forward(self, x_bsd: torch.Tensor) -> torch.Tensor:
        # x: (B,S,C) -> (B,C,S)
        x = x_bsd.permute(0, 2, 1)
        B, C, _ = x.shape
        dev, dt = x.device, x.dtype

        # init latent and base y0
        z = sample_base((B, C, self.cfg.dim_augment), dev, dt)
        y0 = sample_base((B, C, self.cfg.pred_len),   dev, dt)

        # --- Encoder: 5 blocks, residual SCALE then SHIFT on each side ---
        for i in range(self.cfg.num_encoding_layers):
            struct_z = "complex" if i in self.cfg.complex_layers_z else "diagonal"
            struct_x = "complex" if i in self.cfg.complex_layers_x else "diagonal"

            # z <- Scale_residual(x) then SHIFT
            hx = self.phi_x(x)                               # (B,D,H)
            s_z = self.x2z_scales[i](hx)                     # (B,D,A)
            t_z = self.x2z_shifts[i](hx)                     # (B,D,A)
            z  = apply_scale_residual(z, s_z, struct_z)
            z  = apply_shift_additive(z, t_z)

            # x <- Scale_residual(z) then SHIFT
            hz = self.phi_z(z)                               # (B,D,H)
            s_x = self.z2x_scales[i](hz)                     # (B,D,S)
            t_x = self.z2x_shifts[i](hz)                     # (B,D,S)
            x  = apply_scale_residual(x, s_x, struct_x)
            x  = apply_shift_additive(x, t_x)

        # --- Ellipsoidal transport ---
        hz = self.phi_z(z)                                   # (B,D,H) for decoder heads
        rot_vecs = self.U(hz)                                # list of R vectors (B,D,P)

        # 1) rotate+spin: K U(z) y0
        y_u  = apply_rotation(y0, rot_vecs)
        y_ku = apply_K(y_u, self.K_a, self.K_b)

        # 2) scale Λ(z) >= 0  (strictly positive; no leaky mix here)
        lam_raw = self.scale_y_head(hz)                      # (B,D,P)
        lam_pos = positive_log_exp(lam_raw)                  # > 0
        y_s = lam_pos * y_ku

        # 3) unspin+unrotate: U^T K^T y
        y_kT = apply_K_T(y_s, self.K_a, self.K_b)
        y_un = apply_rotation_inverse(y_kT, rot_vecs)

        # 4) translate
        t_y = process_shift(self.shift_y_head(hz))           # (B,D,P)
        y_star = y_un + t_y

        return y_star.permute(0, 2, 1)                       # (B,P,C)
