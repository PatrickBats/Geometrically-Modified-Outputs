"""Decode a GMO LMDB (from imm/generate_gmo.py) into PNGs for IMM finetuning.

Files are named label<class>_<index>.png, the layout expected by Neon's
create_labels.py and dataset_tool.py.

    python imm/lmdb_to_png.py spectral_data_imm/combined_alpha_0.5 data/imm_alpha_0.5
"""
import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lmdb_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.get(b"num_samples").decode())
        entries = [pickle.loads(txn.get(str(i).encode())) for i in range(n)]

    counts = {}
    for i in tqdm(range(0, n, args.batch_size)):
        batch = entries[i:i + args.batch_size]
        lat = torch.from_numpy(np.stack([e["moments"][:4] for e in batch])).to(device)
        with torch.no_grad():
            img = vae.decode(lat).sample
        img = ((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        for e, arr in zip(batch, img):
            c = int(e["label"])
            j = counts.get(c, 0)
            counts[c] = j + 1
            Image.fromarray(arr).save(out / f"label{c}_{j:06d}.png", compress_level=1)
    print(f"Wrote {n} PNGs -> {out}")


if __name__ == "__main__":
    main()
