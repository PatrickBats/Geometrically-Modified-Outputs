# SIMS-style output-space negative guidance for IMM, with in-memory FID.
#
# SIMS (Alemohammad et al. 2024) extrapolates the score at inference:
#   s(x,t) = (1+w)*s_base(x,t) - w*s_selftrained(x,t)
# IMM analog: the per-step pushforward output F is combined the same way. Since the
# per-step map (ddim / simple_edm / euler_fm inside cfg_forward) is affine in the raw
# network output with model-independent coefficients, combining cfg_forward outputs is
# algebraically identical to combining the raw predictions (same placement as CFG).
#
# Same protocol and FID pipeline as Neon's generate_fid_neon.py; the only change
# is dual-model output combination instead of the Neon weight merge. sims_w=0 short-circuits
# to the base model, bitwise-identical to a base run of generate_fid_neon.py.
#
# Usage (from the repo root; runs inside Neon's IMM code, see imm/run_sims.sh):
#   bash imm/run_sims.sh <theta_s.pkl> <sims_w> <fid_out.txt> [SEED] [NUM_GPUS]

import os, pickle, functools, warnings, contextlib
from math import ceil
import numpy as np
import scipy.linalg
import torch, torch.distributed as dist
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm
import dnnlib
from torch_utils import misc

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")

DEFAULT_FID_STATS = "fid_stats/adm_in256_stats.npz"
DETECTOR_URL = ('https://api.ngc.nvidia.com/v2/models/nvidia/research/'
                'stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl')

# ----------------------------------------------------------------------------
# IMM-compatible sampling (identical to generate_fid_neon.py)

def generator_fn(*args, name='pushforward_generator_fn', **kwargs):
    return globals()[name](*args, **kwargs)

@torch.no_grad()
def pushforward_generator_fn(net, latents, class_labels=None, discretization=None, mid_nt=None, num_steps=None, cfg_scale=None):
    d = latents.dtype
    dev = latents.device

    if discretization == 'uniform':
        t_steps = torch.linspace(net.T, net.eps, num_steps + 1, dtype=d, device=dev)
    elif discretization == 'edm':
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float32, device=dev)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float32, device=dev)).exp().item()
        rho = 7.0
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=dev)
        nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps).to(d)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt, dtype=torch.float32, device=dev)).item() for nt in mid_nt]
        t_steps = torch.tensor([net.T] + list(mid_t), dtype=d, device=dev)
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])

    x = latents
    for (t_cur, t_next) in zip(t_steps[:-1], t_steps[1:]):
        x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale)
    return x

# -------------------- SIMS dual-model wrapper --------------------
class SIMSNet:
    """Presents the cfg_forward/sampler interface of net_r while extrapolating
    away from net_s in output space: F = (1+w)*F_r - w*F_s."""

    def __init__(self, net_r, net_s, w):
        object.__setattr__(self, "net_r", net_r)
        object.__setattr__(self, "net_s", net_s)
        object.__setattr__(self, "w", float(w))

    def __getattr__(self, name):  # only called when not found on the instance
        return getattr(object.__getattribute__(self, "net_r"), name)

    def cfg_forward(self, *args, **kwargs):
        F_r = self.net_r.cfg_forward(*args, **kwargs)
        if self.w == 0.0:
            return F_r
        F_s = self.net_s.cfg_forward(*args, **kwargs)
        return (1.0 + self.w) * F_r - self.w * F_s

# -------------------- FID helpers (from edm/calculate_fid.py) --------------------
def fid_from_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(sigma @ sigma_ref, disp=False)
    if np.iscomplexobj(s):
        s = s.real
    return float(m + np.trace(sigma + sigma_ref - 2 * s))

# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    # -------------------- DDP setup --------------------
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_cuda else 0
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cpu")

    # -------------------- Config & CLI overrides --------------------
    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    preset_key = str(getattr(config, "preset", "8_steps_cfg1.5_pushforward_uniform"))
    fid_stats_path = str(getattr(config, "fid_stats", DEFAULT_FID_STATS))
    fid_out_path = getattr(config, "fid_out", None)

    per_class_count = int(getattr(config, "per_class_count", 0))
    if per_class_count <= 0 and int(config.label_dim) > 0:
        if rank == 0:
            raise ValueError("Provide +per_class_count=<int> on the CLI.")

    # -------------------- RNG / Backend --------------------
    seed = config.eval.seed if config.eval.seed is not None else 42
    seed = seed + rank
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = bool(config.eval.cudnn_benchmark)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # -------------------- Encoder --------------------
    encoder = dnnlib.util.construct_class_by_name(**config.encoder)

    # -------------------- Networks --------------------
    resume_pkl = config.eval.resume
    if resume_pkl is None:
        if rank == 0:
            raise ValueError("Set eval.resume to your checkpoint path.")

    aux_resume = getattr(config, "aux_resume", None)
    sims_w = float(getattr(config, "sims_w", 0.0))
    if sims_w != 0.0 and aux_resume is None:
        raise ValueError("sims_w != 0 requires +aux_resume=<theta_s.pkl>.")

    with dnnlib.util.open_url(resume_pkl, verbose=(rank == 0)) as f:
        base_data = pickle.load(f)
    base_module = base_data.get('ema', None)
    if base_module is None:
        raise KeyError("Base checkpoint missing 'ema' module.")
    net_r = base_module.eval().requires_grad_(False).to(device)

    net_s = None
    if aux_resume is not None:
        with dnnlib.util.open_url(aux_resume, verbose=(rank == 0)) as f:
            aux_data = pickle.load(f)
        aux_module = aux_data.get('ema', None)
        if aux_module is None:
            raise KeyError("Aux checkpoint missing 'ema' module.")
        net_s = aux_module.eval().requires_grad_(False).to(device)

    net = SIMSNet(net_r, net_s, sims_w) if net_s is not None else net_r

    # -------------------- Inception detector --------------------
    with dnnlib.util.open_url(DETECTOR_URL, verbose=(rank == 0)) as f:
        detector = pickle.load(f).eval().to(device)
    detector_kwargs = dict(return_features=True)

    # -------------------- Sampling preset --------------------
    sample_block = config.get('sampling', {}).get(preset_key, None)
    if sample_block is None:
        available = list(config.get('sampling', {}).keys())
        if rank == 0:
            raise KeyError(f"Sampling preset '{preset_key}' not found. Available: {available}")

    cfg_override = getattr(config, "cfg_scale", None)
    if cfg_override is not None:
        sample_block = dict(sample_block)
        sample_block["cfg_scale"] = float(cfg_override)

    gen_fn = functools.partial(generator_fn, **sample_block)

    # -------------------- Shapes & classes --------------------
    bs = int(config.eval.batch_size)
    H = net.img_resolution
    W = net.img_resolution
    C = net.img_channels
    num_classes = int(net.label_dim)

    # -------------------- Global label sequence, split across ranks --------------------
    if num_classes > 0:
        labels_all = np.repeat(np.arange(num_classes, dtype=np.int64), per_class_count)
        idx_all = np.arange(labels_all.shape[0], dtype=np.int64)
        local_idx = idx_all[rank::world_size]
        local_labels = labels_all[local_idx]
        total_local = local_labels.shape[0]
    else:
        total_images = int(getattr(config, "total_images", 0))
        if total_images <= 0 and rank == 0:
            raise ValueError("For unconditional models set +total_images=<int>.")
        idx_all = np.arange(total_images, dtype=np.int64)
        local_idx = idx_all[rank::world_size]
        local_labels = np.array([], dtype=np.int64)
        total_local = local_idx.shape[0]

    total_batches = ceil(total_local / bs)
    pbar = tqdm(total=total_batches, desc=f"Rank {rank}",
                disable=(world_size > 1 and rank != 0), leave=True)

    one_hot_buf = torch.empty(bs, num_classes, device=device, dtype=torch.float32) if num_classes > 0 else None

    def make_one_hot_np(labels_np, out):
        out.zero_()
        idx = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        out[:idx.numel()].scatter_(1, idx.view(-1, 1), 1.0)
        return out[:idx.numel()]

    # -------------------- Generate + extract Inception features in-memory --------------------
    feat_dim = 2048
    sum_feats = torch.zeros([feat_dim], dtype=torch.float64, device=device)
    sum_outer = torch.zeros([feat_dim, feat_dim], dtype=torch.float64, device=device)
    local_count = 0

    for start in range(0, total_local, bs):
        end = min(start + bs, total_local)
        cur_bs = end - start

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            z = net.get_init_noise([cur_bs, C, H, W], device)
            if num_classes > 0:
                cur_labels = local_labels[start:end]
                c = make_one_hot_np(cur_labels, one_hot_buf)
            else:
                c = None

            lat = gen_fn(net, z, c)
            imgs = encoder.decode(lat).detach()

        # Convert to uint8 [B, 3, 256, 256]
        if imgs.dtype == torch.uint8:
            imgs_u8 = imgs
        else:
            x = imgs.float()
            if x.min() >= 0.0 and x.max() <= 1.0:
                imgs_u8 = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
            else:
                imgs_u8 = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)

        # Extract Inception features (detector expects uint8 NCHW on GPU)
        with torch.no_grad():
            feats = detector(imgs_u8.to(device), **detector_kwargs).to(torch.float64)
        sum_feats += feats.sum(0)
        sum_outer += feats.T @ feats
        local_count += feats.shape[0]

        pbar.update(1)

    pbar.close()

    # -------------------- All-reduce across ranks --------------------
    if dist.is_initialized():
        t_count = torch.tensor(local_count, dtype=torch.float64, device=device)
        dist.all_reduce(t_count)
        dist.all_reduce(sum_feats)
        dist.all_reduce(sum_outer)
        total_count = int(t_count.item())
    else:
        total_count = local_count

    mu = (sum_feats / total_count).cpu().numpy()
    cov = ((sum_outer / total_count) - torch.outer(sum_feats / total_count, sum_feats / total_count))
    cov = (cov * (total_count / max(total_count - 1, 1))).cpu().numpy()  # unbiased

    # -------------------- FID (rank 0) --------------------
    if rank == 0:
        ref = np.load(fid_stats_path)
        mu_ref = ref['mu'].astype(np.float64)
        sigma_ref = ref['sigma'].astype(np.float64)
        fid_val = fid_from_stats(mu, cov, mu_ref, sigma_ref)

        fid_out = str(fid_out_path) if fid_out_path is not None else "fid_result.txt"
        os.makedirs(os.path.dirname(fid_out) if os.path.dirname(fid_out) else '.', exist_ok=True)
        with open(fid_out, "w") as f:
            f.write(f"preset: {preset_key}\n")
            f.write(f"images: {total_count}\n")
            f.write(f"sims_w: {sims_w}\n")
            f.write(f"aux_resume: {aux_resume}\n")
            f.write(f"cfg_scale: {getattr(config, 'cfg_scale', None)}\n")
            f.write(f"seed: {config.eval.seed if config.eval.seed is not None else 42}\n")
            f.write(f"FID: {fid_val:.6f}\n")
        print(f"FID: {fid_val:.6f} | images={total_count} | sims_w={sims_w} | preset={preset_key}")
        print(f"Wrote: {fid_out}")

    # -------------------- Clean DDP teardown --------------------
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        with contextlib.suppress(Exception):
            dist.destroy_process_group()

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
