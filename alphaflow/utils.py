"""Shared utilities for AlphaFlow evaluation and GMO scripts."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from src.training.network_utils import load_snapshot
from src.structs import SnapshotConfig


class _Cond:
    """Minimal cond wrapper with .label [B, num_classes] for the SiT backbone."""
    def __init__(self, label: torch.Tensor):
        self.label = label

    def __getitem__(self, idx):
        return _Cond(self.label[idx])


def neon_merge_backbone(base_sd, aux_sd, w, device):
    """Merge SiT backbone state dicts: theta = base + w*(base - aux)."""
    merged = {}
    for k, b in base_sd.items():
        b_t = b.to(device)
        if not isinstance(b_t, torch.Tensor) or not torch.is_floating_point(b_t):
            merged[k] = b_t
            continue
        if k in aux_sd and isinstance(aux_sd[k], torch.Tensor):
            a_t = aux_sd[k].to(device, dtype=b_t.dtype)
            finite = torch.isfinite(b_t) & torch.isfinite(a_t)
            m = b_t.clone()
            m[finite] = b_t[finite] + w * (b_t[finite] - a_t[finite])
            merged[k] = m
        else:
            merged[k] = b_t
    return merged


@torch.no_grad()
def alphaflow_step(net, z_5d, y_oh, device):
    """1-step AlphaFlow generation: x0 = z - u(z, sigma=1, sigma_next=0).

    z_5d: [B, 1, C, H, W]; y_oh: [B, num_classes]
    Returns x0_5d: [B, 1, C, H, W]
    """
    B = z_5d.shape[0]
    sigma = torch.ones(B, device=device)
    sigma_next = torch.zeros(B, device=device)
    cond = _Cond(y_oh)
    nl = sigma.view(B, 1, 1, 1, 1)
    nl_next = (sigma - sigma_next).view(B, 1, 1, 1, 1)
    u, _ = net.model(z_5d, noise_labels=nl, cond=cond, noise_labels_next=nl_next)
    return z_5d - u


def load_base_net(ckpt_path, device, verbose=True):
    """Load base AlphaFlow LatentDiffusion model (EMA weights)."""
    snap_cfg = SnapshotConfig(snapshot_path=ckpt_path, use_ema=True, load_state=True)
    net, _, _ = load_snapshot(snap_cfg, verbose=verbose, device=device)
    return net.eval().to(device)


def load_aux_backbone(ckpt_path):
    """Load finetuned backbone state dict from a train.py checkpoint (EMA preferred)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return raw.get("ema", raw.get("model", raw))
