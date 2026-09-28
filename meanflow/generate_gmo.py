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
from tqdm import tqdm

from sit import SiT_models
from gmo.core import gmo_correct


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, model_name="SiT-L/2", device="cuda", num_classes=1000):
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[model_name](
        input_size=32, num_classes=num_classes, use_cfg=True, **block_kwargs,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {model_name}: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ---------------------------------------------------------------------------
# MeanFlow forward (with gradients for JVP/VJP)
# ---------------------------------------------------------------------------

def forward_meanflow(model, z, y, cfg_scale, num_steps):
    batch_size = z.shape[0]
    device = z.device
    do_cfg = y is not None and cfg_scale > 1.0
    null_y = torch.full_like(y, model.num_classes) if do_cfg else None

    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        if do_cfg:
            z_c = torch.cat([z, z], 0)
            u_c = model(z_c, torch.cat([r, r], 0), torch.cat([t, t], 0), y=torch.cat([y, null_y], 0))
            u_cond, u_uncond = u_c.chunk(2, 0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(z, r, t, y=y)
        return z - u

    z_cur = z
    time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur, t_next = time_steps[i], time_steps[i + 1]
        t = torch.full((batch_size,), t_cur, device=device)
        r = torch.full((batch_size,), t_next, device=device)
        if do_cfg:
            z_c = torch.cat([z_cur, z_cur], 0)
            u_c = model(z_c, torch.cat([r, r], 0), torch.cat([t, t], 0), y=torch.cat([y, null_y], 0))
            u_cond, u_uncond = u_c.chunk(2, 0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(z_cur, r, t, y=y)
        z_cur = z_cur - (t_cur - t_next) * u
    return z_cur


# ---------------------------------------------------------------------------
# GMO correction
# ---------------------------------------------------------------------------

def make_gen_fn(model, y, cfg_scale, num_steps):
    """z [C, H, W] -> MeanFlow output (flat), differentiable for torch.func."""
    def gen_fn(z):
        return forward_meanflow(model, z.unsqueeze(0), y, cfg_scale, num_steps).reshape(-1)
    return gen_fn


def efficient_spectral_correct(model, z, y, cfg_scale, device, alphas,
                                power_iters=20, hutch_probes=4, num_steps=1):
    z_dev = z.to(device)
    C = model.in_channels
    H = W = z_dev.shape[-1]  # latent spatial size
    gen_fn = make_gen_fn(model, y, cfg_scale, num_steps)
    s, corrected, sigma1, E = gmo_correct(gen_fn, z_dev, alphas, power_iters, hutch_probes)
    results = {alpha: x.reshape(C, H, W).cpu() for alpha, x in corrected.items()}
    return results, sigma1, E, s.reshape(C, H, W).cpu()


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------

def latent_to_moments(latent_4ch):
    LATENT_SCALE = 0.18125
    mean = latent_4ch / LATENT_SCALE
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
    moments = latent_to_moments(arr) if arr.shape[0] == 4 else arr
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Efficient Spectral GMO for MeanFlow")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="SiT-L/2")
    parser.add_argument("--output-dir", type=str, default="spectral_data")
    parser.add_argument("--alphas", type=str, default="0.1,0.2")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--per-class-count", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--power-iters", type=int, default=20)
    parser.add_argument("--hutch-probes", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true", help="Combine shard LMDBs into one per variant")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        _combine_shards(output_dir, alphas)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.model, device, args.num_classes)

    per_class = args.per_class_count
    all_samples = [(cls, idx) for cls in range(args.num_classes) for idx in range(per_class)]
    shard_size = (len(all_samples) + args.n_shards - 1) // args.n_shards
    start = args.shard * shard_size
    my_samples = all_samples[start:min(start + shard_size, len(all_samples))]

    print(f"Shard {args.shard}/{args.n_shards}: {len(my_samples)} samples [{start}, {start + len(my_samples)})")

    progress_file = output_dir / f"shard_{args.shard}_progress.json"
    completed = set()
    if progress_file.exists():
        completed = set(tuple(x) for x in json.loads(progress_file.read_text()).get("completed", []))
        print(f"Resuming: {len(completed)} done")

    keys = ["vanilla"] + alphas
    shard_envs, shard_txns, shard_counts = _open_shard_writers(output_dir, args.shard, keys, len(my_samples))
    FLUSH_EVERY = 50

    t0 = time.time()
    for cls, intra_idx in tqdm(my_samples, desc=f"Shard {args.shard}"):
        if (cls, intra_idx) in completed:
            continue

        torch.manual_seed(cls * per_class + intra_idx + 12345)
        z = torch.randn(1, model.in_channels, 32, 32, device=device)
        y = torch.tensor([cls], device=device, dtype=torch.long)

        try:
            results, sigma1, E, vanilla_lat = efficient_spectral_correct(
                model, z.squeeze(0), y, args.cfg_scale, device, alphas,
                args.power_iters, args.hutch_probes, args.num_steps,
            )
            _write_sample(shard_txns["vanilla"], shard_counts["vanilla"], vanilla_lat, cls)
            shard_counts["vanilla"] += 1
            for alpha in alphas:
                _write_sample(shard_txns[alpha], shard_counts[alpha], results[alpha], cls)
                shard_counts[alpha] += 1

            completed.add((cls, intra_idx))
            if len(completed) % FLUSH_EVERY == 0:
                shard_txns = _flush_shard_writers(shard_envs, shard_txns, shard_counts)
                progress_file.write_text(json.dumps({"completed": [list(x) for x in sorted(completed)]}))
        finally:
            del z, y
            for var in ("results", "vanilla_lat", "sigma1", "E"):
                if var in locals():
                    del locals()[var]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"Shard {args.shard} done in {elapsed/60:.1f} min ({elapsed/len(my_samples):.1f} s/sample)")
    _close_shard_writers(shard_envs, shard_txns, shard_counts)
    progress_file.write_text(json.dumps({
        "completed": [list(x) for x in sorted(completed)],
        "done": True,
        "elapsed_min": elapsed / 60,
    }))


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


if __name__ == "__main__":
    main()
