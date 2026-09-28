#!/usr/bin/env python3
"""Create image grids from AlphaFlow GMO LMDB shard files, decoded with a VAE.

AlphaFlow latents are stored in VAE-native scale (LATENT_SCALE=1), so they are
passed directly to vae.decode() without an additional scale factor.

Example:
    python alphaflow/plot_gmo.py --data-dir spectral_data_alphaflow --n 5 --seed 0
    python alphaflow/plot_gmo.py --data-dir spectral_data_alphaflow --variant combined_alpha_0.25
"""
import argparse
import glob
import os
import pickle
import random

import lmdb
import numpy as np
import torch
import torchvision.utils as vutils

# AlphaFlow latents are stored in VAE-native scale (no additional rescaling needed)
_LATENT_SCALE = 1.0


def find_variants(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "combined_*")))
    return paths or sorted(glob.glob(os.path.join(data_dir, "shard_*")))


def open_env(path):
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


def read_sample(env, idx):
    with env.begin() as txn:
        data = txn.get(str(idx).encode())
        if data is None:
            raise IndexError(idx)
        return pickle.loads(data)


def latent_from_moments(moments):
    return moments[:4] * _LATENT_SCALE  # (4, H, W): no-op at scale=1


def load_vae(device):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()


def decode_with_vae(vae, latents_np, device):
    lat = torch.from_numpy(latents_np).to(torch.float32).to(device)
    with torch.no_grad():
        # Latents are already in VAE-native scale; decode directly.
        out = vae.decode(lat).sample
    return ((out.cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="spectral_data_alphaflow")
    parser.add_argument("--variant", type=str, default=None,
                        help="Specific LMDB folder name inside data-dir")
    parser.add_argument("--n", type=int, default=5, help="Images per variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.variant:
        candidate = os.path.join(args.data_dir, args.variant)
        if not os.path.isdir(candidate):
            raise SystemExit(f"Variant not found: {candidate}")
        variants = [candidate]
    else:
        variants = find_variants(args.data_dir)

    if not variants:
        raise SystemExit(f"No LMDB folders found under {args.data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device)
    print("Loaded VAE for decoding")

    for variant_path in variants:
        env = open_env(variant_path)
        with env.begin() as txn:
            n_total = int(txn.get(b"num_samples").decode())
        if n_total == 0:
            print(f"Skipping {variant_path}: empty LMDB")
            env.close()
            continue

        rng = random.Random(args.seed)
        indices = rng.sample(range(n_total), min(args.n, n_total))
        latents = np.stack(
            [latent_from_moments(read_sample(env, i)["moments"]) for i in indices]
        )
        env.close()

        imgs = decode_with_vae(vae, latents, device)
        base_name = os.path.basename(variant_path.rstrip("/"))
        out_file = args.out or os.path.join(args.data_dir, f"preview_{base_name}.png")
        vutils.save_image(imgs, out_file, nrow=len(indices), normalize=False)
        print(f"Saved preview -> {out_file}")


if __name__ == "__main__":
    main()
