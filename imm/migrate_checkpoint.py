"""
Migrate an IMM pickle checkpoint from an older timm version to the current one.

Older timm Attention modules were pickled without attributes that newer timm
added to Attention.__init__:
  - attn_dim   (num_heads * head_dim)  -- used in forward reshape
  - norm        nn.Identity()          -- scale_norm layer, default off

Additionally, fused_attn is forced to False so that forward-mode autodiff (JVP)
works correctly. PyTorch's efficient attention kernel does not support forward AD,
which the GMO spectral correction requires.

Usage:
    python imm/migrate_checkpoint.py checkpoints/imm.pkl checkpoints/imm_patched.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch.nn as nn

# The IMM pickle contains references to modules in imm/ and the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMM_DIR   = os.path.join(_REPO_ROOT, "imm")
for _p in (_REPO_ROOT, _IMM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def patch_attention_modules(net) -> int:
    import torch.nn as nn
    patched = 0
    for m in net.modules():
        if type(m).__name__ != "Attention":
            continue
        changed = False
        if not hasattr(m, "attn_dim"):
            m.attn_dim = m.num_heads * m.head_dim
            changed = True
        if not hasattr(m, "norm"):
            # scale_norm was False when the model was trained; Identity() is the correct default.
            m.norm = nn.Identity()
            changed = True
        if getattr(m, "fused_attn", False):
            # Efficient attention doesn't support forward-mode AD (JVP) needed by GMO.
            m.fused_attn = False
            changed = True
        if changed:
            patched += 1
    return patched


def main():
    parser = argparse.ArgumentParser(description="Patch IMM checkpoint for current timm")
    parser.add_argument("input",  help="Path to the original .pkl checkpoint")
    parser.add_argument("output", help="Path to write the patched .pkl checkpoint")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        sys.exit(f"Input file not found: {src}")
    if dst.exists():
        ans = input(f"{dst} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    print(f"Loading {src} ...")
    with open(src, "rb") as f:
        data = pickle.load(f)

    net = data.get("ema") if isinstance(data, dict) else data
    if net is None:
        sys.exit("Could not find 'ema' key in checkpoint dict.")

    n = patch_attention_modules(net)
    print(f"Patched {n} Attention module(s).")

    if n == 0:
        print("Nothing to patch: checkpoint is already compatible. No file written.")
        return

    print(f"Saving patched checkpoint to {dst} ...")
    with open(dst, "wb") as f:
        pickle.dump(data, f)

    print("Done.")


if __name__ == "__main__":
    main()
