"""
gradient_select.py - Gradient-based ensemble member selection.

Given a [M, N] tensor of per-token p(ground truth) values for M candidate
models on N fitness tokens, learns a softmax mixture weight per model by
minimizing E_t[-log(Sum_i w_i * p_i(gt))] with AdamW on the alpha logits.
For each requested K, picks the top-K models by learned weight and
renormalizes to get a per-model weight to use at ensemble evaluation time.
"""

import math
import torch
from typing import Dict, List, Sequence


@torch.no_grad()
def _per_model_loss(P, eps=1e-12):
    """Per-row mean -log p (NLL on the fitness slice). P: [M, N]."""
    return (-torch.log(P.clamp(min=eps))).mean(dim=1)


def optimize_alpha(P_fit, opt_steps=300, lr=0.5, weight_decay=0.0,
                   grad_clip=1.0, eps=1e-12, seed=0):
    """Optimize alpha so w = softmax(alpha) minimizes mean -log(w @ P_fit).

    P_fit: [M, N] float tensor on the device the optimization should run on.
    Returns alpha [M] (detached) on the same device.
    """
    device = P_fit.device
    M, _ = P_fit.shape
    g = torch.Generator(device=device).manual_seed(seed)
    alpha = torch.zeros(M, device=device, dtype=torch.float32)
    alpha = alpha + 1e-4 * torch.randn(M, device=device, generator=g)
    alpha.requires_grad_(True)
    opt = torch.optim.AdamW([alpha], lr=lr, weight_decay=weight_decay)

    for _ in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(alpha, dim=0)
        q = (w @ P_fit).clamp_min(eps)
        loss = -torch.log(q).mean()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=grad_clip)
        opt.step()
    return alpha.detach(), float(loss.detach().item())


def run_gradient_selection(
    P_active: torch.Tensor,
    k_values: Sequence[int],
    opt_steps: int = 300,
    lr: float = 0.5,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    eps: float = 1e-12,
    seed: int = 0,
) -> Dict[int, Dict]:
    """Top-K-by-learned-weight selection.

    Args:
        P_active: [M, N] per-token p(gt) for M active candidates over the
            cached fitness slice (replicated across callers).
        k_values: list of K values to evaluate.

    Returns:
        dict {k: {"selected_local": [int...],         # indices into rows of P_active
                  "weights_renorm": [float...],       # one weight per selected, sums to 1
                  "fit_loss": float}}                 # in-sample NLL on the fit slice
        Plus a "_alpha" entry with the raw learned alpha (length M).
    """
    M, _ = P_active.shape
    if M == 0:
        return {"_alpha": []}

    alpha, _ = optimize_alpha(
        P_fit=P_active, opt_steps=opt_steps, lr=lr,
        weight_decay=weight_decay, grad_clip=grad_clip, eps=eps, seed=seed,
    )
    w_all = torch.softmax(alpha, dim=0)

    out: Dict[int, Dict] = {"_alpha": [float(x) for x in alpha.cpu().tolist()]}
    for k in k_values:
        k_eff = min(int(k), M)
        top_idx = torch.topk(w_all, k=k_eff).indices.sort().values
        w_sel = w_all[top_idx]
        w_renorm = w_sel / w_sel.sum().clamp_min(eps)

        p_fit = (w_renorm.unsqueeze(1) * P_active[top_idx]).sum(dim=0)
        fit_loss = float((-torch.log(p_fit.clamp_min(eps))).mean().item())

        out[int(k)] = {
            "selected_local": top_idx.cpu().tolist(),
            "weights_renorm": [float(x) for x in w_renorm.cpu().tolist()],
            "fit_loss": fit_loss,
        }
    return out
