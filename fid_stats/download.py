import argparse
import importlib
from pathlib import Path
from typing import Dict

FID_STATS_CATALOG: Dict[str, Dict[str, str]] = {
    "adm_in256_stats": {
        "repo_id": "anon-y-mous/fid_stats",
        "filename": "adm_in256_stats.npz",
        "repo_type": "dataset",
        "description": "ADM ImageNet-256 Inception statistics for FID",
    },
}

DEFAULT_FID_STATS_KEY = "adm_in256_stats"


def _download_huggingface(repo_id: str, filename: str, repo_type: str, output_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for FID stats downloads. "
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


def ensure_fid_stats(key: str = DEFAULT_FID_STATS_KEY, output_dir: str = "fid_stats", force: bool = False) -> str:
    if key not in FID_STATS_CATALOG:
        raise KeyError(f"Unknown FID stats key '{key}'. Available keys: {sorted(FID_STATS_CATALOG.keys())}")

    meta = FID_STATS_CATALOG[key]
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
        raise RuntimeError(f"FID stats download failed for key '{key}'")
    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download reference FID statistics")
    parser.add_argument("--key", type=str, default=DEFAULT_FID_STATS_KEY)
    parser.add_argument("--list", action="store_true", help="List available FID stats entries")
    parser.add_argument("--output-dir", type=str, default="fid_stats")
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available FID stats:")
        for key, meta in FID_STATS_CATALOG.items():
            print(f"- {key}: {meta['filename']} ({meta['description']})")
        return

    path = ensure_fid_stats(key=args.key, output_dir=args.output_dir, force=args.force)
    print(f"Downloaded {args.key} -> {path}")


if __name__ == "__main__":
    main()