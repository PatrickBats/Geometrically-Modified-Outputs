"""Finetune an AlphaFlow checkpoint on GMO samples.

Uses the official AlphaFlowLoss from https://github.com/snap-research/alphaflow
(clone it and point AF_ROOT at it). The learning rate is the official 3e-4 / 10.
Saves {"ema": backbone_state_dict}, which alphaflow/eval.py reads via --aux-ckpt/--w
for the Neon merge.

    AF_ROOT=third_party/alphaflow python alphaflow/finetune.py \
        --data data/af_alpha0.1 --base checkpoints/alphaflow_b_2.pt --out theta_s.pt \
        --images 480000 --batch 64 --lr 3e-5 --snap-every 120000
"""
import argparse, copy, sys, time
from pathlib import Path

import os as _os
AF_ROOT = _os.path.abspath(_os.environ.get("AF_ROOT", "third_party/alphaflow"))
sys.path.insert(0, AF_ROOT)
import torch
# Some torch builds lack flash/mem-efficient SDPA *backward*; force the math kernel for training.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.structs import DataSampleType, SnapshotConfig, LossPhase
from src.data.datasets import FolderDataset
from src.data import construct_inputs_from_batch
from src.training.network_utils import load_snapshot
from src.training.loss import AlphaFlowLoss
from src.structs import EasyDict
from src.utils import config_utils


def build_cfg():
    with initialize_config_dir(config_dir=AF_ROOT + "/configs", version_base=None):
        cfg = compose(config_name="train")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=480000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--snap-every", type=int, default=120000)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--cur-step-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")

    cfg = build_cfg()
    use_x_cond = bool(OmegaConf.select(cfg, "model.use_x_cond") or False)
    # EasyDict-ify only the loss branch (whole-cfg resolve trips on unrelated env interpolations);
    # loss's ${model...} refs still resolve against the root cfg.
    loss_cfg = EasyDict.init_recursively(cfg.loss)
    # Our data is precomputed latents -> loss should only-normalize (not re-encode via VAE).
    if "use_precomputed_latents" not in loss_cfg.model:
        loss_cfg.model["use_precomputed_latents"] = True
    loss_scaling = float(OmegaConf.select(cfg, "model.loss_scaling") or 1.0)

    # base net (official loader; same path as bundle load_base_net)
    snap = SnapshotConfig(snapshot_path=args.base, use_ema=True, load_state=True)
    net, _, _ = load_snapshot(snap, verbose=True, device=dev)
    net = net.to(dev).train()
    for p in net.parameters():
        p.requires_grad_(True)

    loss_fn = AlphaFlowLoss(loss_cfg).to(dev)

    ds = FolderDataset(src=args.data, data_type=DataSampleType.IMAGE_LATENT,
                       label_shape=(1000,), metadata_ext=".meta.json")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=4, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99),
                            eps=1e-8, weight_decay=0.01)
    ema = copy.deepcopy(net.model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    def save(seen):
        out = args.out if seen >= args.images else args.out.replace(".pt", f"_seen{seen}.pt")
        torch.save({"ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                    "images_seen": seen}, out)
        print(f"[save] {out} (images_seen={seen})", flush=True)

    seen, step, last_snap, t0 = 0, 0, 0, time.time()
    max_steps = max(1, args.images // args.batch)
    n_smoke = 2
    done = False
    while not done:
        for batch in dl:
            x, cond = construct_inputs_from_batch(batch, use_x_cond=use_x_cond, device=dev)
            losses = loss_fn(net=net, x=x, cond=cond, phase=LossPhase.Gen, cur_step=args.cur_step_base + step)
            if step == 0:
                print("loss keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                                     for k, v in losses.items()}, flush=True)
            # Match official compute_gradients: backward on losses.total.mean() * loss_scaling.
            total = losses["total"].mean() * loss_scaling
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.model.parameters()):
                    pe.mul_(args.ema_decay).add_(pn.detach(), alpha=1 - args.ema_decay)
                for be, bn in zip(ema.buffers(), net.model.buffers()):
                    be.copy_(bn)
            seen += args.batch
            step += 1
            if step % 20 == 0 or args.smoke:
                print(f"step {step}/{max_steps} seen {seen} loss {total.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/it)", flush=True)
            if args.smoke and step >= n_smoke:
                print("SMOKE OK", flush=True); return
            if seen - last_snap >= args.snap_every:
                save(seen); last_snap = seen
            if seen >= args.images:
                done = True; break
    save(seen)
    print("FINETUNE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
