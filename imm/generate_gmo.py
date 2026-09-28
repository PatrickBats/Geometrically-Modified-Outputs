"""Spectral GMO correction for IMM checkpoints.

Uses the shared correction in gmo/core.py, adapted for IMM's pkl-format checkpoints and latent
normalisation scheme.

Outputs are written to LMDB in the same moments format used by plot_gmo.py so
the same visualisation script works across all pipelines.  IMM latents are
denormalised to raw VAE space before storage (LATENT_SCALE=1, same as AlphaFlow).

Usage (single GPU):
    python imm/generate_gmo.py \
        --checkpoint checkpoints/imagenet256_ts_a2.pkl \
        --output-dir spectral_data_imm \
        --alphas "0.1,0.2"

Distributed (sharded across N GPUs):
    for i in $(seq 0 7); do
        python imm/generate_gmo.py --checkpoint ... --shard $i --n-shards 8 &
    done
    python imm/generate_gmo.py --checkpoint ... --combine
"""

import gc
import os
import sys
# dnnlib / torch_utils must be importable for pkl unpickling via persistence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gmo.core import gmo_correct
from sampler import imm_to_vae_latent


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_imm_checkpoint(ckpt_path: str, device):
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    return data["ema"].eval().requires_grad_(False).to(device)


# ---------------------------------------------------------------------------
# One-step IMM forward + GMO correction
# ---------------------------------------------------------------------------

def make_gen_fn(net, t_cur, t_next, y):
    """z [C, H, W] -> one-step IMM output (flat), differentiable for torch.func."""
    def gen_fn(z):
        x0 = net.cfg_forward(z.unsqueeze(0), t_cur, t_next, class_labels=y, cfg_scale=1.0)
        return x0.reshape(-1)
    return gen_fn


def efficient_spectral_correct(net, z_3d, y, t_cur, t_next, device, alphas,
                                power_iters=20, hutch_probes=4):
    """Compute vanilla + corrected latents for all alpha values.

    Returns (results, sigma1, E, vanilla_vae) where:
      - results      : {alpha: corrected_vae_latent_cpu [C, H, W]}
      - vanilla_vae  : [C, H, W] raw VAE latent (vae.decode-ready)
    """
    C, H, W = z_3d.shape
    gen_fn = make_gen_fn(net, t_cur, t_next, y)
    s, corrected, sigma1, E = gmo_correct(gen_fn, z_3d, alphas, power_iters, hutch_probes)

    def _to_vae(x_flat):
        return imm_to_vae_latent(x_flat.reshape(C, H, W).cpu())

    results = {alpha: _to_vae(x) for alpha, x in corrected.items()}
    return results, sigma1, E, _to_vae(s)


# ---------------------------------------------------------------------------
# LMDB helpers (raw VAE latents, LATENT_SCALE=1: same format as alphaflow)
# ---------------------------------------------------------------------------

def _latent_to_moments(latent_4ch):
    """Store raw VAE latent as moments (mean=latent, logvar=-30)."""
    mean   = np.asarray(latent_4ch, dtype=np.float32)
    logvar = np.full_like(mean, -30.0)
    return np.concatenate([mean, logvar], axis=0)  # [8, H, W]


def _open_shard_writers(output_dir, shard_idx, keys, n_samples_estimate):
    import lmdb
    envs, txns, counts = {}, {}, {}
    for key in keys:
        tag  = "vanilla" if key == "vanilla" else f"alpha_{key}"
        path = output_dir / f"shard_{shard_idx}_{tag}"
        path.mkdir(parents=True, exist_ok=True)
        map_size = max(n_samples_estimate * 260_000, 1 << 24)
        env = lmdb.open(str(path), map_size=map_size)
        with env.begin() as txn:
            existing = txn.get(b"num_samples")
            counts[key] = int(existing.decode()) if existing else 0
        envs[key]  = env
        txns[key]  = env.begin(write=True)
    return envs, txns, counts


def _write_sample(txn, idx, latent_tensor, label):
    import pickle as _pkl
    arr     = latent_tensor.detach().numpy().astype(np.float32)
    moments = _latent_to_moments(arr) if arr.shape[0] == 4 else arr
    data    = {
        "moments":      moments,
        "moments_flip": np.flip(moments, axis=-1).copy(),
        "label":        int(label),
    }
    txn.put(str(idx).encode(), _pkl.dumps(data))


def _flush_shard_writers(envs, txns, counts):
    for key in envs:
        txns[key].put(b"num_samples", str(counts[key]).encode())
        txns[key].commit()
        txns[key] = envs[key].begin(write=True)
    return txns


def _close_shard_writers(envs, txns, counts):
    for key in envs:
        txns[key].put(b"num_samples", str(counts[key]).encode())
        txns[key].commit()
        envs[key].close()
    print("  Saved shard LMDBs: " + ", ".join(f"{k}={counts[k]}" for k in counts))


def _combine_shards(output_dir, alphas):
    import lmdb
    for key in ["vanilla"] + alphas:
        tag        = "vanilla" if key == "vanilla" else f"alpha_{key}"
        shard_dirs = sorted(d for d in output_dir.glob(f"shard_*_{tag}") if d.is_dir())
        if not shard_dirs:
            print(f"  No shards for {tag}, skipping")
            continue

        total = 0
        for sd in shard_dirs:
            env = lmdb.open(str(sd), readonly=True, lock=False)
            with env.begin() as txn:
                n = txn.get(b"num_samples")
                total += int(n.decode()) if n else 0
            env.close()

        out_path = output_dir / f"combined_{tag}"
        out_path.mkdir(parents=True, exist_ok=True)
        out_env    = lmdb.open(str(out_path), map_size=max(total * 260_000, 1 << 24))
        global_idx = 0
        with out_env.begin(write=True) as out_txn:
            for sd in shard_dirs:
                env = lmdb.open(str(sd), readonly=True, lock=False)
                with env.begin() as txn:
                    n_bytes = txn.get(b"num_samples")
                    n = int(n_bytes.decode()) if n_bytes else 0
                    for i in range(n):
                        raw = txn.get(str(i).encode())
                        if raw:
                            out_txn.put(str(global_idx).encode(), raw)
                            global_idx += 1
                env.close()
            out_txn.put(b"num_samples", str(global_idx).encode())
        out_env.close()
        print(f"  Combined {tag}: {global_idx} samples -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectral GMO correction for IMM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to IMM .pkl checkpoint")
    parser.add_argument("--output-dir", type=str, default="spectral_data_imm")
    parser.add_argument("--alphas", type=str, default="0.1,0.2")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--per-class-count", type=int, default=30)
    parser.add_argument("--power-iters", type=int, default=20)
    parser.add_argument("--hutch-probes", type=int, default=20)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true",
                        help="Combine shard LMDBs into one per variant")
    args = parser.parse_args()

    alphas     = [float(a) for a in args.alphas.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        _combine_shards(output_dir, alphas)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading IMM checkpoint from {args.checkpoint} ...")
    net = load_imm_checkpoint(args.checkpoint, device)

    num_classes = net.label_dim
    C = net.img_channels
    H = W = net.img_resolution

    # One-step time points
    t_cur  = torch.as_tensor(net.T,   dtype=torch.float32, device=device)
    t_next = torch.as_tensor(net.eps, dtype=torch.float32, device=device)

    per_class  = args.per_class_count
    all_samples = [(cls, idx) for cls in range(num_classes) for idx in range(per_class)]
    shard_size  = (len(all_samples) + args.n_shards - 1) // args.n_shards
    start       = args.shard * shard_size
    my_samples  = all_samples[start: min(start + shard_size, len(all_samples))]
    print(f"Shard {args.shard}/{args.n_shards}: {len(my_samples)} samples")

    progress_file = output_dir / f"shard_{args.shard}_progress.json"
    completed     = set()
    if progress_file.exists():
        completed = set(
            tuple(x) for x in json.loads(progress_file.read_text()).get("completed", [])
        )
        print(f"Resuming: {len(completed)} done")

    keys = ["vanilla"] + alphas
    shard_envs, shard_txns, shard_counts = _open_shard_writers(
        output_dir, args.shard, keys, len(my_samples)
    )
    FLUSH_EVERY = 50

    t0 = time.time()
    for cls, intra_idx in tqdm(my_samples, desc=f"Shard {args.shard}"):
        if (cls, intra_idx) in completed:
            continue

        torch.manual_seed(cls * per_class + intra_idx + 12345)
        z_4d = net.get_init_noise([1, C, H, W], device)
        z_3d = z_4d.squeeze(0)
        y    = torch.nn.functional.one_hot(
            torch.tensor([cls], device=device),
            num_classes=num_classes,
        ).to(torch.float32)

        try:
            results, sigma1, E, vanilla_vae = efficient_spectral_correct(
                net, z_3d, y, t_cur, t_next, device, alphas,
                args.power_iters, args.hutch_probes,
            )
            _write_sample(shard_txns["vanilla"], shard_counts["vanilla"], vanilla_vae, cls)
            shard_counts["vanilla"] += 1
            for alpha in alphas:
                _write_sample(shard_txns[alpha], shard_counts[alpha], results[alpha], cls)
                shard_counts[alpha] += 1

            completed.add((cls, intra_idx))
            if len(completed) % FLUSH_EVERY == 0:
                shard_txns = _flush_shard_writers(shard_envs, shard_txns, shard_counts)
                progress_file.write_text(
                    json.dumps({"completed": [list(x) for x in sorted(completed)]})
                )
        finally:
            del z_4d, z_3d, y
            for var in ("results", "vanilla_vae", "sigma1", "E"):
                if var in locals():
                    del locals()[var]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"Shard {args.shard} done in {elapsed/60:.1f} min "
        f"({elapsed/max(len(my_samples), 1):.1f} s/sample)"
    )
    _close_shard_writers(shard_envs, shard_txns, shard_counts)
    progress_file.write_text(
        json.dumps({
            "completed": [list(x) for x in sorted(completed)],
            "done": True,
            "elapsed_min": elapsed / 60,
        })
    )


if __name__ == "__main__":
    main()
