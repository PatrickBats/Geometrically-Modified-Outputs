"""Shared IMM sampler and latent-decode helpers used by imm/eval.py and generate_gmo.py."""
import torch

# StabilityVAEEncoder normalisation constants (from im256.yaml)
IMM_RAW_MEAN   = [0.86488, -0.27787343,  0.21616915,  0.3738409]
IMM_RAW_STD    = [4.85503674, 5.31922414, 3.93725398, 3.9870003]
IMM_FINAL_STD  = 0.5   # sigma_data in im256.yaml


@torch.no_grad()
def imm_sampler(net, latents, class_labels=None, cfg_scale=1.5, num_steps=1):
    """Multi-step IMM sampler.

    latents      : [B, C, H, W] in IMM normalised space
    class_labels : [B, label_dim] one-hot (or None for unconditional)
    Returns      : [B, C, H, W] in IMM normalised space
    """
    device = latents.device
    t_steps = torch.linspace(
        net.T, net.eps, num_steps + 1, dtype=torch.float64, device=device
    )
    x = latents.to(torch.float64)
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x = net.cfg_forward(
            x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale
        ).to(torch.float64)
    return x


def imm_decode(latents, vae, device):
    """Decode IMM normalised latents to pixel images in [-1, 1].

    latents : [B, 4, H, W] float32 in IMM normalised space
    Returns : [B, 3, H*8, W*8] float32 in [-1, 1]
    """
    mean = torch.tensor(IMM_RAW_MEAN, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    std  = torch.tensor(IMM_RAW_STD,  device=device, dtype=torch.float32).view(1, 4, 1, 1)
    raw  = latents.to(torch.float32) / IMM_FINAL_STD * std + mean
    with torch.no_grad():
        return vae.decode(raw).sample


def imm_to_vae_latent(latent_chw):
    """Convert a single IMM normalised latent [C, H, W] to raw VAE latent [C, H, W] (CPU)."""
    mean = torch.tensor(IMM_RAW_MEAN).view(4, 1, 1)
    std  = torch.tensor(IMM_RAW_STD).view(4, 1, 1)
    return latent_chw.to(torch.float32) / IMM_FINAL_STD * std + mean
