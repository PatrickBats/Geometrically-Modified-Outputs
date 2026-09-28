import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meanflow.sampler import meanflow_sampler
from meanflow.sit import SiT_models

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


def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and normalize it to a state dict.

    Supports raw state dicts and common wrappers such as model/state_dict/ema.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            return ckpt["ema"]
        if "model_ema" in ckpt and isinstance(ckpt["model_ema"], dict):
            return ckpt["model_ema"]

        # If keys look like parameter tensors, treat this as the state dict.
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt

    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def compute_fid_in_memory(
    model,
    vae,
    detector,
    device,
    rank,
    world_size,
    num_images,
    cfg_scale,
    fid_stats_path,
    seed,
    num_steps,
    batch_size,
    latent_size,
):
    """Generate images and compute FID entirely in memory for one seed."""
    per_gpu = math.ceil(num_images / world_size)

    feat_dim = 2048
    sum_feats = torch.zeros([feat_dim], dtype=torch.float64, device=device)
    sum_outer = torch.zeros([feat_dim, feat_dim], dtype=torch.float64, device=device)
    local_count = 0

    # Offset by rank so each process has a unique RNG stream for this run.
    torch.manual_seed(int(seed) + rank)

    pbar = range(0, per_gpu, batch_size)
    if rank == 0:
        pbar = tqdm(pbar, desc=f"Generating+FID (seed={seed})")

    num_classes = model.num_classes if hasattr(model, "num_classes") else 1000

    for start in pbar:
        bs = min(batch_size, per_gpu - start)
        z = torch.randn(bs, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)

        with torch.no_grad():
            latents = meanflow_sampler(model, z, y=y, cfg_scale=cfg_scale, num_steps=num_steps)
            imgs = vae.decode(latents / 0.18215).sample
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
    cov = (sum_outer / total_count) - torch.outer(sum_feats / total_count, sum_feats / total_count)
    cov = (cov * (total_count / max(total_count - 1, 1))).cpu().numpy()

    if rank == 0:
        ref = np.load(fid_stats_path)
        mu_ref = ref["mu"].astype(np.float64)
        sigma_ref = ref["sigma"].astype(np.float64)
        diff = mu - mu_ref
        s, _ = scipy.linalg.sqrtm(cov @ sigma_ref, disp=False)
        if np.iscomplexobj(s):
            s = s.real
        fid = float(diff @ diff + np.trace(cov + sigma_ref - 2.0 * s))
        return fid
    return None


def resolve_primary_checkpoint(
    model_name: str,
    checkpoint_path: Optional[str],
    checkpoint_key: Optional[str],
    allow_download: bool,
) -> str:
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.exists() and allow_download:
            key = checkpoint_key or DEFAULT_CHECKPOINT_KEY_FOR_MODEL.get(model_name)
            if key:
                return ensure_checkpoint(key)
        if not p.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Set --download-missing or provide a valid --checkpoint-path."
            )
        return str(p)

    key = checkpoint_key or DEFAULT_CHECKPOINT_KEY_FOR_MODEL.get(model_name)
    if key is None:
        raise ValueError(
            f"No default checkpoint key for model {model_name}. "
            "Pass --checkpoint-path or --checkpoint-key."
        )
    if allow_download:
        return ensure_checkpoint(key)

    candidate = os.path.join("checkpoints", f"{key}.pt")
    if not Path(candidate).exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {candidate}. "
            "Run checkpoints/download.py or set --download-missing."
        )
    return candidate


def resolve_fid_stats_path(fid_stats: Optional[str], allow_download: bool) -> str:
    if fid_stats:
        p = Path(fid_stats)
        if p.exists():
            return str(p)
        if allow_download:
            # If user passed a non-existent path, use default stats key and copy not required.
            return ensure_fid_stats(DEFAULT_FID_STATS_KEY)
        raise FileNotFoundError(
            f"FID stats file not found at {fid_stats}. Set --download-missing or provide a valid path."
        )

    if allow_download:
        return ensure_fid_stats(DEFAULT_FID_STATS_KEY)

    candidate = os.path.join("fid_stats", "adm_in256_stats.npz")
    if not Path(candidate).exists():
        raise FileNotFoundError(
            f"Expected FID stats at {candidate}. "
            "Run fid_stats/download.py or set --download-missing."
        )
    return candidate


def build_output_path(checkpoint_path: str, checkpoint_key: Optional[str]) -> str:
    base_name = checkpoint_key or Path(checkpoint_path).stem
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
    safe_name = safe_name.strip("_") or "checkpoint"
    return f"fid_results_{safe_name}.json"

def main():
    parser = argparse.ArgumentParser(description="Evaluate MeanFlow checkpoints with repeated-seed FID")
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1, help="Sampler steps (1 = single-step MeanFlow)")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=32)

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-key", type=str, default=None)
    parser.add_argument("--fid-stats", type=str, default=None)
    parser.add_argument("--download-missing", action="store_true")

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    seeds = parse_seed_values(args.seeds)

    if args.model not in SiT_models:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(SiT_models.keys())}")

    if args.batch_size <= 0 or args.num_steps <= 0 or args.num_images <= 0:
        raise ValueError("--batch-size, --num-steps, and --num-images must all be positive.")

    fid_stats_path = resolve_fid_stats_path(args.fid_stats, args.download_missing)

    primary_ckpt_path = resolve_primary_checkpoint(
        args.model,
        args.checkpoint_path,
        args.checkpoint_key,
        args.download_missing,
    )
    output_path = build_output_path(primary_ckpt_path, args.checkpoint_key)

    if rank == 0:
        print(f"Checkpoint: {primary_ckpt_path}")
        print(f"Checkpoint key: {args.checkpoint_key or DEFAULT_CHECKPOINT_KEY_FOR_MODEL.get(args.model)}")
        print(f"FID stats: {fid_stats_path}")
        print(f"Seeds: {seeds}")
        print(f"Images per eval: {args.num_images}")
        print(f"Sampler steps: {args.num_steps}")
        print(f"Output file: {output_path}")

    sd = load_checkpoint_state_dict(primary_ckpt_path)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    detector = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[args.model](
        input_size=32,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    seed_results = []
    for seed in seeds:
        fid_local = compute_fid_in_memory(
            model,
            vae,
            detector,
            device,
            rank,
            world_size,
            args.num_images,
            args.cfg_scale,
            fid_stats_path,
            seed,
            args.num_steps,
            args.batch_size,
            args.latent_size,
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

        print(f"\nSummary for checkpoint evaluation over {len(seed_results)} seeds")
        print(f"  mean FID: {mean_fid:.4f}")
        print(f"  std  FID: {std_fid:.4f}")

        output = {
            "model": args.model,
            "checkpoint": primary_ckpt_path,
            "checkpoint_key": args.checkpoint_key,
            "num_images": int(args.num_images),
            "num_steps": int(args.num_steps),
            "batch_size": int(args.batch_size),
            "latent_size": int(args.latent_size),
            "cfg_scale": float(args.cfg_scale),
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
