"""Top-level downloader + benchmark sweep runner for GMO FID.

Routes each checkpoint key to the correct eval script based on its pipeline
(meanflow/eval.py, alphaflow/eval.py, imm/eval.py).

Examples:
    python run_benchmark_sweep.py --download-only

    # MeanFlow sweep
    python run_benchmark_sweep.py \
        --nproc-per-node 8 \
        --checkpoint-keys sit_b_2 \
        --num-images 50000 \
        --seeds "0,1,2,3,4"

    # AlphaFlow sweep
    python run_benchmark_sweep.py \
        --nproc-per-node 8 \
        --checkpoint-keys alphaflow_b_2 \
        --num-images 50000
"""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from checkpoints.download import CHECKPOINT_CATALOG, ensure_checkpoint
from fid_stats.download import DEFAULT_FID_STATS_KEY, ensure_fid_stats


def parse_csv(values: str) -> List[str]:
    text = "" if values is None else str(values).strip()
    if not text:
        return []
    return [token.strip() for token in text.replace(" ", ",").split(",") if token.strip()]


def run_command(cmd: List[str], cwd: Path, verbose: bool) -> int:
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(proc.returncode)


def ensure_assets(checkpoint_keys: List[str], fid_stats_key: str,
                  force_download: bool) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for key in checkpoint_keys:
        resolved[key] = ensure_checkpoint(key=key, force=force_download)
    fid_stats_path = ensure_fid_stats(key=fid_stats_key, force=force_download) if checkpoint_keys else None
    resolved["fid_stats"] = fid_stats_path
    return resolved


def build_meanflow_eval_cmd(torchrun_bin, nproc_per_node, model, checkpoint_path,
                             fid_stats_path, cfg_scale, num_steps, num_images,
                             batch_size, latent_size, seeds) -> List[str]:
    return [
        torchrun_bin, f"--nproc_per_node={nproc_per_node}",
        "meanflow/eval.py",
        "--model", model,
        "--cfg-scale", str(cfg_scale),
        "--num-steps", str(num_steps),
        "--num-images", str(num_images),
        "--batch-size", str(batch_size),
        "--latent-size", str(latent_size),
        "--checkpoint-path", checkpoint_path,
        "--fid-stats", fid_stats_path,
        "--seeds", seeds,
    ]


def build_alphaflow_eval_cmd(torchrun_bin, nproc_per_node, checkpoint_path,
                              fid_stats_path, num_images, batch_size, seeds) -> List[str]:
    return [
        torchrun_bin, f"--nproc_per_node={nproc_per_node}",
        "alphaflow/eval.py",
        "--checkpoint-path", checkpoint_path,
        "--fid-stats", fid_stats_path,
        "--num-images", str(num_images),
        "--batch-size", str(batch_size),
        "--seeds", seeds,
    ]


def build_imm_eval_cmd(torchrun_bin, nproc_per_node, checkpoint_path,
                        fid_stats_path, num_images, batch_size, seeds,
                        cfg_scale, num_steps) -> List[str]:
    return [
        torchrun_bin, f"--nproc_per_node={nproc_per_node}",
        "imm/eval.py",
        "--checkpoint-path", checkpoint_path,
        "--fid-stats", fid_stats_path,
        "--num-images", str(num_images),
        "--batch-size", str(batch_size),
        "--cfg-scale", str(cfg_scale),
        "--num-steps", str(num_steps),
        "--seeds", seeds,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-level download + benchmark sweep for GMO FID")
    parser.add_argument("--torchrun-bin", type=str, default="torchrun")
    parser.add_argument("--nproc-per-node", type=int, default=1)

    parser.add_argument("--checkpoint-keys", type=str, default=None,
                        help="Comma-separated checkpoint keys to evaluate. Defaults to all known keys.")
    parser.add_argument("--fid-stats-key", type=str, default=DEFAULT_FID_STATS_KEY)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")

    # Shared eval args
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # MeanFlow-specific args
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)

    parser.add_argument("--out-dir", type=str, default="benchmark_sweep")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be > 0")

    checkpoint_keys = (
        parse_csv(args.checkpoint_keys) if args.checkpoint_keys
        else list(CHECKPOINT_CATALOG.keys())
    )
    if not checkpoint_keys:
        raise ValueError("No checkpoint keys resolved. Pass --checkpoint-keys explicitly.")

    unknown = [k for k in checkpoint_keys if k not in CHECKPOINT_CATALOG]
    if unknown:
        raise ValueError(f"Unknown checkpoint key(s): {unknown}. Known: {sorted(CHECKPOINT_CATALOG.keys())}")

    workspace = Path(__file__).resolve().parent
    run_name = args.run_name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = workspace / args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = ensure_assets(checkpoint_keys, args.fid_stats_key, args.force_download)
    fid_stats_path = assets["fid_stats"]

    if args.download_only:
        print("Downloads complete. Exiting due to --download-only.")
        print(json.dumps({"checkpoint_keys": checkpoint_keys, "fid_stats": fid_stats_path}, indent=2))
        return

    sweep_summary = {"checkpoint_keys": checkpoint_keys, "runs": []}

    for key in checkpoint_keys:
        checkpoint_path = assets[key]
        catalog_entry = CHECKPOINT_CATALOG[key]
        pipeline = catalog_entry.get("pipeline", "meanflow")
        model = catalog_entry.get("model", "SiT-B/2")

        if pipeline == "alphaflow":
            cmd = build_alphaflow_eval_cmd(
                torchrun_bin=args.torchrun_bin,
                nproc_per_node=args.nproc_per_node,
                checkpoint_path=checkpoint_path,
                fid_stats_path=fid_stats_path,
                num_images=args.num_images,
                batch_size=args.batch_size,
                seeds=args.seeds,
            )
        elif pipeline == "imm":
            cmd = build_imm_eval_cmd(
                torchrun_bin=args.torchrun_bin,
                nproc_per_node=args.nproc_per_node,
                checkpoint_path=checkpoint_path,
                fid_stats_path=fid_stats_path,
                num_images=args.num_images,
                batch_size=args.batch_size,
                seeds=args.seeds,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
            )
        else:
            cmd = build_meanflow_eval_cmd(
                torchrun_bin=args.torchrun_bin,
                nproc_per_node=args.nproc_per_node,
                model=model,
                checkpoint_path=checkpoint_path,
                fid_stats_path=fid_stats_path,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                num_images=args.num_images,
                batch_size=args.batch_size,
                latent_size=args.latent_size,
                seeds=args.seeds,
            )

        if args.dry_run:
            code = 0
            print("DRY RUN:", " ".join(cmd))
        else:
            code = run_command(cmd, cwd=workspace, verbose=args.verbose)

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        sweep_summary["runs"].append({
            "checkpoint_key": key,
            "checkpoint_path": checkpoint_path,
            "pipeline": pipeline,
            "output": str(workspace / f"fid_results_{safe_name}.json"),
            "return_code": code,
        })

    summary_path = out_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)

    print(f"Sweep complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
