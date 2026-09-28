"""Geometrically Modified Outputs (GMO).

All models share the same correction. A model enters only through ``gen_fn``,
a differentiable map from a single noise tensor ``z`` (any shape) to a flat
output vector. The Jacobian information (sigma1, u1, v1, E, Jz) is computed
once per sample and reused for every alpha.
"""
import math

import torch
from torch.func import jvp, vjp


def jvp_flat(gen_fn, z, v_flat):
    """Return (J v, gen_fn(z)) for a flat tangent v."""
    with torch.enable_grad():
        out, jv = jvp(gen_fn, (z.detach(),), (v_flat.reshape(z.shape),))
    return jv.detach(), out.detach()


def vjp_flat(gen_fn, z, u_flat):
    """Return J^T u as a flat vector."""
    with torch.enable_grad():
        _, pullback = vjp(gen_fn, z.detach())
        (v,) = pullback(u_flat)
    return v.reshape(-1).detach()


def power_iteration(gen_fn, z, n_iters=20):
    """Top singular triplet (sigma1, u1, v1) of the Jacobian of gen_fn at z."""
    v = torch.randn(z.numel(), device=z.device)
    v = v / v.norm()
    for _ in range(n_iters):
        Jv, _ = jvp_flat(gen_fn, z, v)
        u = Jv / (Jv.norm() + 1e-10)
        v = vjp_flat(gen_fn, z, u)
        v = v / (v.norm() + 1e-10)
    Jv, _ = jvp_flat(gen_fn, z, v)
    sigma1 = Jv.norm().item()
    u1 = Jv / (Jv.norm() + 1e-10)
    return sigma1, u1, v


def hutchinson_frobenius_norm(gen_fn, z, n_probes=20):
    """Hutchinson estimate of ||J||_F^2."""
    total = 0.0
    for _ in range(n_probes):
        r = torch.randn(z.numel(), device=z.device)
        Jr, _ = jvp_flat(gen_fn, z, r)
        total += Jr.norm().item() ** 2
    return total / n_probes


def apply_correction(s, Jz, sigma1, u1, v1, E, z_flat, alpha):
    """GMO output for one alpha in [0, 1).

    Shrinks the Jacobian by sqrt(1 - alpha) and moves the freed energy onto the
    top singular direction, so ||J||_F is preserved.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    v1z = (v1 @ z_flat).item()
    gamma = math.sqrt(1 - alpha) - 1
    sigma1_new = math.sqrt((1 - alpha) * sigma1 ** 2 + alpha * E)
    delta = sigma1_new - math.sqrt(1 - alpha) * sigma1
    return s + gamma * Jz + delta * v1z * u1


def gmo_correct(gen_fn, z, alphas, power_iters=20, hutch_probes=20):
    """GMO outputs of gen_fn at z for every alpha, sharing one Jacobian pass.

    Returns (s, results, sigma1, E): the standard output s (flat), a dict
    {alpha: corrected flat output}, the top singular value and ||J||_F^2.
    """
    z_flat = z.reshape(-1)
    with torch.no_grad():
        s = gen_fn(z.detach()).reshape(-1)
    Jz, _ = jvp_flat(gen_fn, z, z_flat)
    sigma1, u1, v1 = power_iteration(gen_fn, z, power_iters)
    E = hutchinson_frobenius_norm(gen_fn, z, hutch_probes)
    results = {a: apply_correction(s, Jz, sigma1, u1, v1, E, z_flat, a) for a in alphas}
    return s, results, sigma1, E
