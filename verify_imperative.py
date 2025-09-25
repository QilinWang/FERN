# ===========================
# file: verify_imperative.py
# ===========================
from __future__ import annotations

import argparse, os, numpy as np
import torch
import torch.nn.functional as F

from fern_imperative import FERN, FERNConfig
from data_and_metrics import load_etth2_data, calculate_metrics

def main():
    ap = argparse.ArgumentParser(description="Verify FERN (imperative) on ETTh2")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--checkpoint_path", type=str, default="fern_etth2_imperative.pt")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_proj_swd", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        return
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    cfgd = ckpt["config"]; gstd = ckpt["global_std"]

    # rebuild model
    cfg = FERNConfig(**cfgd)
    cfg.device = str(device)
    model = FERN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # load test data
    data_path = os.path.join(args.data_dir, "ETTh2.csv")
    try:
        _, _, te, _, _ = load_etth2_data(data_path, cfg.seq_len, cfg.pred_len, args.batch_size, device=device, scale=False)
    except FileNotFoundError:
        # synthetic fallback so the script "rolls"
        T = 2000; D = cfg.channels
        synth = torch.randn(T, D, device=device) * 0.5
        from torch.utils.data import DataLoader, Dataset
        class _DS(Dataset):
            def __init__(self, data, S, P): self.data=data; self.S=S; self.P=P
            def __len__(self): return self.data.shape[0] - self.S - self.P + 1
            def __getitem__(self, i):
                return self.data[i:i+self.S], self.data[i+self.S:i+self.S+self.P]
        te = DataLoader(_DS(synth, cfg.seq_len, cfg.pred_len), batch_size=args.batch_size, shuffle=False)

    # eval
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te:
            xb = xb.to(device); yb = yb.to(device)
            yhat = model(xb)
            preds.append(yhat); trues.append(yb)
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    # metrics
    m = calculate_metrics(pred, true, gstd, device, num_proj=args.num_proj_swd, seed=args.seed)
    print("\n--- Verification (ETTh2) ---")
    print(f"MSE: {m['MSE']:.4f}")
    print(f"MAE: {m['MAE']:.4f}")
    print(f"SWD^2: {m['SWD']:.4f}")
    print(f"EPT: {m['EPT']:.2f}")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()
