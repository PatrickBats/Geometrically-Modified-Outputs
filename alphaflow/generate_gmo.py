"""Spectral GMO correction for AlphaFlow checkpoints.

Runs the shared correction in gmo/core.py, adapted for AlphaFlow's 5-D latent convention and optional Neon merge.

Outputs are written to LMDB in the same moments format used by plot_gmo.py so
the same visualization script works for both pipelines.

Usage (single GPU):
    python alphaflow/generate_gmo.py \
        --base-ckpt checkpoints/alphaflow-B-2-cfg.pt \
        --output-dir spectral_data_alphaflow \
        --alphas "0.1,0.2"

With Neon merge:
    python alphaflow/generate_gmo.py \
        --base-ckpt checkpoints/alphaflow-B-2-cfg.pt \
        --aux-ckpt training_output/finetune_b2/checkpoints/0003500.pt \
        --w 0.1 \
        --output-dir spectral_data_alphaflow

Distributed (sharded across N GPUs):
    for i in $(seq 0 7); do
        python alphaflow/generate_gmo.py ... --shard $i --n-shards 8 &
    done
    python alphaflow/generate_gmo.py ... --combine
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
# Forward-mode AD (JVP) is unsupported by fused SDPA kernels: force math kernel.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from tqdm import tqdm

from gmo.core import gmo_correct
from utils import _Cond, neon_merge_backbone, load_base_net, load_aux_backbone


# ---------------------------------------------------------------------------
# AlphaFlow one-step forward + GMO correction
# ---------------------------------------------------------------------------

def make_gen_fn(backbone, y_oh, device):
    """z [C, H, W] -> one-step AlphaFlow output (flat), differentiable for torch.func.

    Same map as alphaflow_step in utils.py: sigma = 1, sigma_next = 0.
    """
    cond = _Cond(y_oh)
    nl = torch.ones(1, device=device).view(1, 1, 1, 1, 1)
    nl_next = torch.ones(1, device=device).view(1, 1, 1, 1, 1)  # sigma - sigma_next

    def gen_fn(z):
        z_5d = z.unsqueeze(0).unsqueeze(0)          # [1, 1, C, H, W]
        u, _ = backbone(z_5d, noise_labels=nl, cond=cond, noise_labels_next=nl_next)
        return (z_5d - u).reshape(-1)

    return gen_fn


def efficient_spectral_correct(backbone, net, z_3d, y_oh, device, alphas,
                                power_iters=20, hutch_probes=4):
    """Compute vanilla + corrected latents for all alpha values.

    Returns (results, sigma1, E, vanilla_4d) where:
      - results: {alpha: corrected_4d_cpu}
      - vanilla_4d: [C, H, W] denormalized latent (VAE-ready)
    """
    C, H, W = z_3d.shape
    gen_fn = make_gen_fn(backbone, y_oh, device)
    s, corrected, sigma1, E = gmo_correct(gen_fn, z_3d, alphas, power_iters, hutch_probes)

    def _to_raw_latent(x_flat):
        """Convert backbone output -> VAE-ready latent [C, H, W]."""
        x0_5d = x_flat.reshape(C, H, W).unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
        raw = net.denormalize_latents(x0_5d)                        # [1,1,C,H,W]
        return raw.squeeze(0).squeeze(0).cpu()                      # [C,H,W]

    results = {alpha: _to_raw_latent(x) for alpha, x in corrected.items()}
    return results, sigma1, E, _to_raw_latent(s)


# ---------------------------------------------------------------------------
# LMDB helpers (same format as meanflow/generate_gmo.py)
# ---------------------------------------------------------------------------

# AlphaFlow latents are already in VAE-native scale, so LATENT_SCALE=1 means
# plot_gmo.py will call vae.decode(moments[:4]) directly without re-scaling.
_LATENT_SCALE = 1.0


def _latent_to_moments(latent_4ch):
    mean = latent_4ch / _LATENT_SCALE
    logvar = np.full_like(mean, -30.0)
    return np.concatenate([mean, logvar], axis=0)  # [8, H, W]


def _open_shard_writers(output_dir, shard_idx, keys, n_samples_estimate):
    import lmdb
    envs, txns, counts = {}, {}, {}
    for key in keys:
        tag = "vanilla" if key == "vanilla" else f"alpha_{key}"
        path = output_dir / f"shard_{shard_idx}_{tag}"
        path.mkdir(parents=True, exist_ok=True)
        map_size = max(n_samples_estimate * 260_000, 1 << 24)
        env = lmdb.open(str(path), map_size=map_size)
        with env.begin() as txn:
            existing = txn.get(b"num_samples")
            counts[key] = int(existing.decode()) if existing else 0
        envs[key] = env
        txns[key] = env.begin(write=True)
    return envs, txns, counts


def _write_sample(txn, idx, latent_tensor, label):
    arr = latent_tensor.detach().numpy().astype(np.float32)
    moments = _latent_to_moments(arr) if arr.shape[0] == 4 else arr
    data = {
        "moments": moments,
        "moments_flip": np.flip(moments, axis=-1).copy(),
        "label": int(label),
    }
    txn.put(str(idx).encode(), pickle.dumps(data))


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
        tag = "vanilla" if key == "vanilla" else f"alpha_{key}"
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
        out_env = lmdb.open(str(out_path), map_size=max(total * 260_000, 1 << 24))
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
    parser = argparse.ArgumentParser(description="Spectral GMO correction for AlphaFlow")
    parser.add_argument("--base-ckpt", type=str, default="checkpoints/alphaflow-B-2-cfg.pt")
    parser.add_argument("--aux-ckpt", type=str, default=None,
                        help="Optional finetuned backbone for Neon merge")
    parser.add_argument("--w", type=float, default=None,
                        help="Neon merge weight (required when --aux-ckpt is set)")
    parser.add_argument("--output-dir", type=str, default="spectral_data_alphaflow")
    parser.add_argument("--alphas", type=str, default="0.1,0.2")
    parser.add_argument("--per-class-count", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--power-iters", type=int, default=20)
    parser.add_argument("--hutch-probes", type=int, default=20)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true",
                        help="Combine shard LMDBs into one per variant")
    args = parser.parse_args()

    if args.aux_ckpt is not None and args.w is None:
        parser.error("--w is required when --aux-ckpt is set")

    alphas = [float(a) for a in args.alphas.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        _combine_shards(output_dir, alphas)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model from {args.base_ckpt} ...")
    net = load_base_net(args.base_ckpt, device)

    if args.aux_ckpt is not None:
        print(f"Applying Neon merge (aux={args.aux_ckpt}, w={args.w}) ...")
        base_backbone_sd = {k: v.cpu() for k, v in net.model.state_dict().items()}
        aux_backbone_sd = load_aux_backbone(args.aux_ckpt)
        merged_sd = neon_merge_backbone(base_backbone_sd, aux_backbone_sd, args.w, device)
        net.model.load_state_dict(merged_sd, strict=True)
        net.model.eval()

    backbone = net.model
    num_classes = net.label_shape[0]
    C = net.in_channels
    H, W = net.input_shape[2], net.input_shape[3]

    per_class = args.per_class_count
    all_samples = [(cls, idx) for cls in range(num_classes) for idx in range(per_class)]
    shard_size = (len(all_samples) + args.n_shards - 1) // args.n_shards
    start = args.shard * shard_size
    my_samples = all_samples[start:min(start + shard_size, len(all_samples))]
    print(f"Shard {args.shard}/{args.n_shards}: {len(my_samples)} samples")

    progress_file = output_dir / f"shard_{args.shard}_progress.json"
    completed = set()
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
        z_3d = torch.randn(C, H, W, device=device)
        y_oh = torch.zeros(1, num_classes, device=device)
        y_oh[0, cls] = 1.0

        try:
            results, sigma1, E, vanilla_lat = efficient_spectral_correct(
                backbone, net, z_3d, y_oh, device, alphas,
                args.power_iters, args.hutch_probes,
            )
            _write_sample(shard_txns["vanilla"], shard_counts["vanilla"], vanilla_lat, cls)
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
            del z_3d, y_oh
            for var in ("results", "vanilla_lat", "sigma1", "E"):
                if var in locals():
                    del locals()[var]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"Shard {args.shard} done in {elapsed/60:.1f} min "
        f"({elapsed/max(len(my_samples),1):.1f} s/sample)"
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
