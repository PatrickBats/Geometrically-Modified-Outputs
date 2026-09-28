import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphaflow.utils import alphaflow_step, neon_merge_backbone, load_base_net, load_aux_backbone
from checkpoints.download import DEFAULT_CHECKPOINT_KEY_FOR_MODEL, ensure_checkpoint
from fid_stats.download import DEFAULT_FID_STATS_KEY, ensure_fid_stats


def init_distributed():
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = local_rank = 0
        world = 1
    return rank, local_rank, world


def parse_seed_values(seed_values: str) -> List[int]:
    seed_text = "" if seed_values is None else str(seed_values).strip()
    if not seed_text:
        raise ValueError("--seeds cannot be empty")
    return [int(token) for token in seed_text.replace(",", " ").split()]


def compute_fid_in_memory(net, vae, detector, device, rank, world_size,
                           num_images, fid_stats_path, seed, batch_size):
    """Generate images and compute FID entirely in memory for one seed."""
    per_gpu = math.ceil(num_images / world_size)
    num_classes = net.label_shape[0]
    C, H, W = net.in_channels, net.input_shape[2], net.input_shape[3]

    feat_dim = 2048
    sum_feats = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    sum_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    local_count = 0

    torch.manual_seed(int(seed) + rank)

    pbar = range(0, per_gpu, batch_size)
    if rank == 0:
        pbar = tqdm(pbar, desc=f"Generating+FID (seed={seed})")

    for start in pbar:
        bs = min(batch_size, per_gpu - start)
        z = torch.randn(bs, 1, C, H, W, device=device)
        y_oh = torch.zeros(bs, num_classes, device=device)
        y_oh.scatter_(1, torch.randint(0, num_classes, (bs,), device=device).unsqueeze(1), 1.0)

        with torch.no_grad():
            x0_5d = alphaflow_step(net, z, y_oh, device)
            raw_latents = net.denormalize_latents(x0_5d).squeeze(1).float()
            imgs = vae.decode(raw_latents).sample
            imgs_u8 = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            feats = detector(imgs_u8)[0].to(torch.float64)

        sum_feats += feats.sum(0)
        sum_outer += feats.T @ feats
        local_count += feats.shape[0]

    if dist.is_initialized():
        t_count = torch.tensor(local_count, dtype=torch.float64, device=device)
        dist.all_reduce(t_count)
        dist.all_reduce(sum_feats)
        dist.all_reduce(sum_outer)
        total_count = int(t_count.item())
    else:
        total_count = local_count

    mu = (sum_feats / total_count).cpu().numpy()
    cov = (sum_outer / total_count) - torch.outer(
        sum_feats / total_count, sum_feats / total_count)
    cov = (cov * (total_count / max(total_count - 1, 1))).cpu().numpy()

    if rank == 0:
        ref = np.load(fid_stats_path)
        mu_ref = ref["mu"].astype(np.float64)
        sigma_ref = ref["sigma"].astype(np.float64)
        diff = mu - mu_ref
        s, _ = scipy.linalg.sqrtm(cov @ sigma_ref, disp=False)
        if np.iscomplexobj(s):
            s = s.real
        return float(diff @ diff + np.trace(cov + sigma_ref - 2.0 * s))
    return None


def resolve_checkpoint(checkpoint_path, checkpoint_key, allow_download):
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.exists() and allow_download:
            key = checkpoint_key or DEFAULT_CHECKPOINT_KEY_FOR_MODEL.get("alphaflow-B/2")
            if key:
                return ensure_checkpoint(key)
        if not p.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Set --download-missing or provide a valid --checkpoint-path."
            )
        return str(p)

    key = checkpoint_key
    if key is None:
        raise ValueError("Pass --checkpoint-path or --checkpoint-key.")
    if allow_download:
        return ensure_checkpoint(key)
    candidate = os.path.join("checkpoints", f"{key}.pt")
    if not Path(candidate).exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {candidate}. "
            "Run checkpoints/download.py or set --download-missing."
        )
    return candidate


def resolve_fid_stats(fid_stats, allow_download):
    if fid_stats:
        p = Path(fid_stats)
        if p.exists():
            return str(p)
        if allow_download:
            return ensure_fid_stats(DEFAULT_FID_STATS_KEY)
        raise FileNotFoundError(f"FID stats not found at {fid_stats}.")
    if allow_download:
        return ensure_fid_stats(DEFAULT_FID_STATS_KEY)
    candidate = os.path.join("fid_stats", "adm_in256_stats.npz")
    if not Path(candidate).exists():
        raise FileNotFoundError(
            f"Expected FID stats at {candidate}. Set --download-missing."
        )
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaFlow checkpoints with repeated-seed FID")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-key", type=str, default=None,
                        help="Key from checkpoints/download.py catalog, e.g. alphaflow_b_2_imagenet256")
    parser.add_argument("--aux-ckpt", type=str, default=None,
                        help="Optional finetuned backbone for Neon merge before evaluation")
    parser.add_argument("--w", type=float, default=None,
                        help="Neon merge weight (required when --aux-ckpt is set)")
    parser.add_argument("--fid-stats", type=str, default=None)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.aux_ckpt is not None and args.w is None:
        parser.error("--w is required when --aux-ckpt is set")

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    seeds = parse_seed_values(args.seeds)

    ckpt_path = resolve_checkpoint(args.checkpoint_path, args.checkpoint_key, args.download_missing)
    fid_stats_path = resolve_fid_stats(args.fid_stats, args.download_missing)

    safe_name = Path(ckpt_path).stem
    output_path = args.output or f"fid_results_{safe_name}.json"

    if rank == 0:
        print(f"Checkpoint: {ckpt_path}")
        print(f"FID stats:  {fid_stats_path}")
        print(f"Seeds:      {seeds}")
        print(f"Images:     {args.num_images}")

    net = load_base_net(ckpt_path, device, verbose=(rank == 0))

    if args.aux_ckpt is not None:
        if rank == 0:
            print(f"Applying Neon merge (aux={args.aux_ckpt}, w={args.w}) ...")
        base_sd = {k: v.cpu() for k, v in net.model.state_dict().items()}
        aux_sd = load_aux_backbone(args.aux_ckpt)
        merged_sd = neon_merge_backbone(base_sd, aux_sd, args.w, device)
        net.model.load_state_dict(merged_sd, strict=True)
        net.model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    detector = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    seed_results = []
    for seed in seeds:
        fid_local = compute_fid_in_memory(
            net, vae, detector, device, rank, world_size,
            args.num_images, fid_stats_path, seed, args.batch_size,
        )
        fid_tensor = torch.tensor(-1.0, dtype=torch.float32, device=device)
        if rank == 0:
            fid_tensor.fill_(fid_local if fid_local is not None else float("nan"))
        if dist.is_initialized():
            dist.broadcast(fid_tensor, src=0)
        fid = float(fid_tensor.item())

        if rank == 0:
            print(f"  seed={seed}: FID={fid:.4f}")
            seed_results.append({"seed": int(seed), "fid": fid})

    if rank == 0:
        fid_values = np.array([r["fid"] for r in seed_results], dtype=np.float64)
        mean_fid = float(fid_values.mean()) if len(fid_values) else float("nan")
        std_fid = float(fid_values.std(ddof=1)) if len(fid_values) > 1 else 0.0

        print(f"\nSummary over {len(seed_results)} seeds")
        print(f"  mean FID: {mean_fid:.4f}")
        print(f"  std  FID: {std_fid:.4f}")

        output = {
            "checkpoint": ckpt_path,
            "checkpoint_key": args.checkpoint_key,
            "aux_ckpt": args.aux_ckpt,
            "w": args.w,
            "num_images": int(args.num_images),
            "batch_size": int(args.batch_size),
            "fid_stats": fid_stats_path,
            "seeds": [int(s) for s in seeds],
            "seed_results": seed_results,
            "mean_fid": mean_fid,
            "std_fid": std_fid,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
