"""Neon merge of two IMM pkl checkpoints: theta = base + w * (base - aux).

The merged pkl can be evaluated with imm/eval.py --checkpoint-path.

    python imm/merge.py --base checkpoints/imagenet256_ts_a2.pkl \
        --aux runs/gmo_a0.5/network-snapshot-000060.pkl --w 0.8 --out merged.pkl
"""
import argparse
import copy
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def neon_merge(base_module, aux_module, w):
    b_sd = base_module.state_dict()
    a_sd = aux_module.state_dict()
    merged = {}
    for k, b in b_sd.items():
        if isinstance(b, torch.Tensor) and torch.is_floating_point(b) and k in a_sd:
            a = a_sd[k].to(dtype=b.dtype)
            m = b.clone()
            finite = torch.isfinite(b) & torch.isfinite(a)
            m[finite] = b[finite] + w * (b[finite] - a[finite])
            merged[k] = m
        else:
            merged[k] = b
    out = copy.deepcopy(base_module)
    out.load_state_dict(merged)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base IMM pkl (theta_r)")
    ap.add_argument("--aux", required=True, help="Finetuned IMM pkl (theta_s)")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.base, "rb") as f:
        base = pickle.load(f)
    with open(args.aux, "rb") as f:
        aux = pickle.load(f)
    base["ema"] = neon_merge(base["ema"], aux["ema"], args.w)
    with open(args.out, "wb") as f:
        pickle.dump(base, f)
    print(f"Saved merged checkpoint (w={args.w}) -> {args.out}")


if __name__ == "__main__":
    main()
