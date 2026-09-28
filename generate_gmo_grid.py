import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from checkpoints.download import CHECKPOINT_CATALOG, ensure_checkpoint
from gmo.core import gmo_correct


# ---------------------------------------------------------------------------
# GMO correction of a single latent
# ---------------------------------------------------------------------------

def correct_one(forward_fn, z, alpha, power_iters, hutch_probes):
    """Standard and GMO outputs (flat) for one latent z of shape 1xCxHxW."""
    def gen_fn(z3):
        return forward_fn(z3.unsqueeze(0)).reshape(-1)
    s, results, _, _ = gmo_correct(gen_fn, z.squeeze(0), [alpha], power_iters, hutch_probes)
    return s, results[alpha]


# ---------------------------------------------------------------------------
# ImageNet model setup
# ---------------------------------------------------------------------------

def _setup_imagenet(ckpt_path, model_name, device, cfg_scale, num_classes):
    from meanflow.sit import SiT_models
    from diffusers.models import AutoencoderKL

    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[model_name](
        input_size=32, num_classes=num_classes, use_cfg=True, **block_kwargs
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        ckpt = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    C, H = model.in_channels, 32

    def make_forward(y):
        def forward_fn(z_b):
            B = z_b.shape[0]
            r = torch.zeros(B, device=device)
            t = torch.ones(B, device=device)
            if cfg_scale > 1.0:
                null_y = torch.full_like(y, num_classes)
                u_c = model(
                    torch.cat([z_b, z_b], 0),
                    torch.cat([r, r], 0),
                    torch.cat([t, t], 0),
                    y=torch.cat([y, null_y], 0),
                )
                u_cond, u_uncond = u_c.chunk(2, 0)
                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z_b, r, t, y=y)
            return z_b - u
        return forward_fn

    def to_image(x_flat):
        with torch.no_grad():
            img = vae.decode(x_flat.reshape(1, C, H, H) / 0.18215).sample
        return ((img.squeeze(0) + 1) * 0.5).clamp(0.0, 1.0)

    return make_forward, to_image, (C, H, H), num_classes


# ---------------------------------------------------------------------------
# IMM model setup
# ---------------------------------------------------------------------------

def _setup_imm_imagenet(ckpt_path, device, cfg_scale, num_classes):
    import pickle

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _IMM_DIR   = os.path.join(_repo_root, "imm")
    for _p in (_repo_root, _IMM_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from imm.sampler import IMM_RAW_MEAN, IMM_RAW_STD, IMM_FINAL_STD
    from diffusers.models import AutoencoderKL

    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"].eval().requires_grad_(False).to(device)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    C = net.img_channels    # 4
    H = net.img_resolution  # 32

    t_cur  = torch.as_tensor(net.T,   dtype=torch.float32, device=device)
    t_next = torch.as_tensor(net.eps, dtype=torch.float32, device=device)

    def make_forward(y):
        def forward_fn(z_b):
            with torch.enable_grad():
                x0 = net.cfg_forward(z_b, t_cur, t_next, class_labels=y, cfg_scale=cfg_scale)
            return x0.reshape(-1)
        return forward_fn

    mean = torch.tensor(IMM_RAW_MEAN, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    std  = torch.tensor(IMM_RAW_STD,  device=device, dtype=torch.float32).view(1, 4, 1, 1)

    def to_image(x_flat):
        with torch.no_grad():
            raw = (x_flat.reshape(1, C, H, H).float() / IMM_FINAL_STD) * std + mean
            img = vae.decode(raw).sample
        return ((img.squeeze(0) + 1) * 0.5).clamp(0.0, 1.0)

    def sample_noise():
        return net.get_init_noise([1, C, H, H], device)

    n_classes = num_classes if num_classes else net.label_dim
    return make_forward, to_image, sample_noise, (C, H, H), n_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a GMO sample grid")
    parser.add_argument("--checkpoint-key", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4, help="Images per row in the grid")
    parser.add_argument("--power-iters", type=int, default=20)
    parser.add_argument("--hutch-probes", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale (ImageNet)")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint_key:
        key = args.checkpoint_key
        if args.download_missing:
            ckpt_path = ensure_checkpoint(key)
        else:
            meta = CHECKPOINT_CATALOG.get(key, {})
            filename = meta.get("filename", f"{key}.pt")
            ckpt_path = os.path.join("checkpoints", filename)
        catalog = CHECKPOINT_CATALOG.get(key, {})
    elif args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        key = None
        catalog = {}
    else:
        raise ValueError("Pass --checkpoint-key or --checkpoint-path")

    pipeline   = catalog.get("pipeline", "meanflow")
    model_name = catalog.get("model", "SiT-B/2")
    cfg_scale  = args.cfg_scale if args.cfg_scale != 1.0 else catalog.get("cfg_scale", args.cfg_scale)
    stem = key or Path(ckpt_path).stem
    if args.out:
        out_path = args.out
        p = Path(out_path)
        std_out_path = str(p.parent / f"{p.stem}_std{p.suffix}")
    else:
        out_path     = f"gmo_grid_{stem}_alpha{args.alpha}_seed{args.seed}.png"
        std_out_path = f"std_grid_{stem}_seed{args.seed}.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Pipeline   : {pipeline}  |  alpha={args.alpha}  |  seed={args.seed}  |  n={args.num_images}")

    images      = []
    base_images = []

    if pipeline == "alphaflow":
        raise SystemExit("For AlphaFlow use alphaflow/generate_gmo.py and alphaflow/plot_gmo.py")

    if pipeline == "imm":
        make_forward, to_image, sample_noise, (C, H, W), n_classes = _setup_imm_imagenet(
            ckpt_path, device, cfg_scale, args.num_classes
        )
        for i in range(args.num_images):
            y = torch.nn.functional.one_hot(
                torch.randint(0, n_classes, (1,), device=device),
                num_classes=n_classes,
            ).to(torch.float32)
            z   = sample_noise()
            fwd = make_forward(y)
            s, x = correct_one(fwd, z, args.alpha, args.power_iters, args.hutch_probes)
            base_images.append(to_image(s).cpu())
            images.append(to_image(x).cpu())
            print(f"  [{i+1}/{args.num_images}]", end="\r")

    else:
        make_forward, to_image, (C, H, W), n_classes = _setup_imagenet(
            ckpt_path, model_name, device, cfg_scale, args.num_classes
        )
        for i in range(args.num_images):
            y   = torch.randint(0, n_classes, (1,), device=device)
            z   = torch.randn(1, C, H, W, device=device)
            fwd = make_forward(y)
            s, x = correct_one(fwd, z, args.alpha, args.power_iters, args.hutch_probes)
            base_images.append(to_image(s).cpu())
            images.append(to_image(x).cpu())
            print(f"  [{i+1}/{args.num_images}]", end="\r")

    print()
    grid = make_grid(torch.stack(images), nrow=args.nrow, padding=2)
    save_image(grid, out_path)
    print(f"Saved {args.num_images}-image GMO grid -> {out_path}")

    std_grid = make_grid(torch.stack(base_images), nrow=args.nrow, padding=2)
    save_image(std_grid, std_out_path)
    print(f"Saved {args.num_images}-image std grid -> {std_out_path}")


if __name__ == "__main__":
    main()
