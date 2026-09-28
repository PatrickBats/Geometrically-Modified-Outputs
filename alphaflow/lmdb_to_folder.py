"""Convert a GMO LMDB (from alphaflow/generate_gmo.py) into the AlphaFlow
FolderDataset latent layout used by alphaflow/finetune.py:

    <key>.latents.pkl        pickle({'mean': [4,H,W], 'logvar': [4,H,W]})
    <key>.latents.meta.json  {'class': int}

    python alphaflow/lmdb_to_folder.py spectral_data_alphaflow/combined_alpha_0.1 data/af_alpha0.1
"""
import argparse
import json
import os
import pickle

import lmdb
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lmdb_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False)
    n = 0
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k == b"num_samples":
                continue
            obj = pickle.loads(v)
            m = np.asarray(obj["moments"], dtype=np.float32)
            key = f"{n:07d}"
            with open(os.path.join(args.out_dir, f"{key}.latents.pkl"), "wb") as f:
                pickle.dump({"mean": m[:4].copy(), "logvar": m[4:].copy()}, f)
            with open(os.path.join(args.out_dir, f"{key}.latents.meta.json"), "w") as f:
                json.dump({"class": int(obj["label"])}, f)
            n += 1
    print(f"Wrote {n} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
