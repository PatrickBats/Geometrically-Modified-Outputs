"""Download the released GMO checkpoints (after Neon self-training with GMOs).

Examples:
    python checkpoints/download.py --list
    python checkpoints/download.py --key sit_b_2
    python checkpoints/download.py --all
"""

import argparse
import importlib
from pathlib import Path
from typing import Dict


CHECKPOINT_CATALOG: Dict[str, Dict[str, str]] = {
    # MeanFlow checkpoints
    "sit_b_2": {
        "repo_id": "anon-y-mous/meanflow",
        "filename": "sit_b_2.pt",
        "repo_type": "model",
        "description": "MeanFlow SiT-B/2 after Neon self-training with GMOs (ImageNet-256)",
        "dataset": "imagenet256",
        "model": "SiT-B/2",
        "pipeline": "meanflow",
    },
    "sit_l_2": {
        "repo_id": "anon-y-mous/meanflow",
        "filename": "sit_l_2.pt",
        "repo_type": "model",
        "description": "MeanFlow SiT-L/2 after Neon self-training with GMOs (ImageNet-256)",
        "dataset": "imagenet256",
        "model": "SiT-L/2",
        "pipeline": "meanflow",
    },
    # AlphaFlow checkpoints
    "alphaflow_b_2": {
        "repo_id": "anon-y-mous/alphaflow",
        "filename": "alphaflow_b_2.pt",
        "repo_type": "model",
        "description": "AlphaFlow SiT-B/2 after Neon self-training with GMOs (ImageNet-256)",
        "dataset": "imagenet256",
        "model": "SiT-B/2",
        "pipeline": "alphaflow",
    },
    "alphaflow_xl_2": {
        "repo_id": "anon-y-mous/alphaflow",
        "filename": "alphaflow_xl_2.pt",
        "repo_type": "model",
        "description": "AlphaFlow SiT-XL/2 after Neon self-training with GMOs (ImageNet-256)",
        "dataset": "imagenet256",
        "model": "SiT-XL/2",
        "pipeline": "alphaflow",
    },
    # IMM checkpoint
    "imm": {
        "repo_id": "anon-y-mous/imm",
        "filename": "imm.pkl",
        "repo_type": "model",
        "description": "IMM DiT-XL/2 after Neon self-training with GMOs (ImageNet-256, cfg_scale=1.5)",
        "dataset": "imagenet256",
        "model": "DiT-XL/2",
        "pipeline": "imm",
        "cfg_scale": 1.5,
    },
}

DEFAULT_CHECKPOINT_KEY_FOR_MODEL = {
    # MeanFlow
    "SiT-B/2": "sit_b_2",
    "SiT-L/2": "sit_l_2",
    # AlphaFlow
    "alphaflow-B/2": "alphaflow_b_2",
    "alphaflow-XL/2": "alphaflow_xl_2",
    # IMM
    "DiT-XL/2": "imm",
}


def _download_huggingface(repo_id: str, filename: str, repo_type: str, output_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for checkpoint downloads. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    print(f"Downloading {filename} from Hugging Face Hub ({repo_id})...")
    
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False
    )


def ensure_checkpoint(key: str, output_dir: str = "checkpoints", force: bool = False) -> str:
    if key not in CHECKPOINT_CATALOG:
        raise KeyError(
            f"Unknown checkpoint key '{key}'. "
            f"Available keys: {sorted(CHECKPOINT_CATALOG.keys())}"
        )

    meta = CHECKPOINT_CATALOG[key]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / meta["filename"]

    if output_path.exists() and not force:
        return str(output_path)

    _download_huggingface(
        repo_id=meta["repo_id"],
        filename=meta["filename"],
        repo_type=meta["repo_type"],
        output_dir=out_dir
    )
    
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint download failed for key '{key}'")

    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints")
    parser.add_argument("--key", type=str, default=None, help="Checkpoint key to download")
    parser.add_argument("--all", action="store_true", help="Download all known checkpoints")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available checkpoints:")
        for key, meta in CHECKPOINT_CATALOG.items():
            print(f"- {key}: {meta['filename']} ({meta['description']})")
        return

    if args.all:
        for key in sorted(CHECKPOINT_CATALOG.keys()):
            path = ensure_checkpoint(key, output_dir=args.output_dir, force=args.force)
            print(f"Downloaded {key} -> {path}")
        return

    if args.key:
        path = ensure_checkpoint(args.key, output_dir=args.output_dir, force=args.force)
        print(f"Downloaded {args.key} -> {path}")
        return

    parser.error("Specify one of --list, --key, or --all")


if __name__ == "__main__":
    main()