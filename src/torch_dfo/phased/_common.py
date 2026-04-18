"""Shared low-level helpers used by multiple phased submodules.

These utilities are leaf helpers that the orchestrator and phase modules
both rely on.  They live in a dedicated module so that ``_cma_phase`` and
other submodules can import them directly at module load time without
creating a circular import against :mod:`torch_dfo.phased.orchestrator`.

The public ``torch_dfo.phased`` package re-exports the names below so
existing test imports (``from torch_dfo.phased import _normalize_covariance``
etc.) continue to work unchanged.
"""

from __future__ import annotations

import math

import torch

from torch_dfo.phased._config import PhasedConfig

_EPS = 1e-12

_CONFIG_DEFAULTS = PhasedConfig()


def _compute_valley_focus_generation_bounds(
    lambdas: tuple[int, ...] = _CONFIG_DEFAULTS.high_dim_valley_entry_portfolio_lambdas,
) -> tuple[int, int]:
    """Return minimum and maximum incumbent-only generations per full refresh."""
    branch = _CONFIG_DEFAULTS.high_dim_valley_entry_branch
    focus_cycle = _CONFIG_DEFAULTS.high_dim_valley_entry_focus_cycle
    max_focus_cycle = _CONFIG_DEFAULTS.high_dim_valley_entry_max_focus_cycle
    focus_eval_ratio = _CONFIG_DEFAULTS.high_dim_valley_entry_focus_eval_ratio
    max_focus_eval_ratio = _CONFIG_DEFAULTS.high_dim_valley_entry_max_focus_eval_ratio

    valley_lam = max(1, lambdas[branch])
    full_lam = max(1, sum(lambdas))
    historical_min = max(0, focus_cycle - 1)
    historical_max = max(historical_min, max_focus_cycle - 1)
    parity_min = math.ceil(focus_eval_ratio * full_lam / valley_lam)
    parity_max = math.ceil(max_focus_eval_ratio * full_lam / valley_lam)
    min_focus = max(historical_min, parity_min)
    max_focus = max(min_focus, historical_max, parity_max)
    return min_focus, max_focus


def _normalize_covariance(
    C: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize covariance to unit average eigenvalue, preserving orientation.

    Called at CMA-ES
    phase boundaries (not every eigendecomposition) to avoid distorting the
    learned covariance during CMA-ES iterations.
    """
    C = (C + C.T) * 0.5
    dim = C.shape[0]
    eye = torch.eye(dim, device=device, dtype=dtype)
    try:
        if device.type not in ("cpu", "cuda"):
            eigvals, eigvecs = torch.linalg.eigh(C.to("cpu"))
            eigvals = eigvals.to(device)
            eigvecs = eigvecs.to(device)
        else:
            eigvals, eigvecs = torch.linalg.eigh(C)
    except RuntimeError:
        return eye
    eigvals = eigvals.clamp_min(1e-8)
    mean_eig = eigvals.mean().item()
    if not math.isfinite(mean_eig) or mean_eig <= _EPS:
        return eye
    eigvals = eigvals / mean_eig
    result: torch.Tensor = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    return (result + result.T) * 0.5


def _merge_search_pool(
    pool: torch.Tensor | None,
    pool_fit: torch.Tensor | None,
    additions: torch.Tensor | None,
    add_fit: torch.Tensor | None,
    max_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Merge *additions* into *pool*, keeping only the *max_size* best by fitness."""
    if (
        additions is None
        or add_fit is None
        or additions.numel() == 0
        or add_fit.numel() == 0
        or max_size <= 0
    ):
        return pool, pool_fit

    if pool is None or pool_fit is None or pool.numel() == 0 or pool_fit.numel() == 0:
        merged = additions
        merged_fit = add_fit
    else:
        merged = torch.cat([pool, additions], dim=0)
        merged_fit = torch.cat([pool_fit, add_fit], dim=0)

    keep = merged_fit.argsort()[:max_size]
    return merged[keep], merged_fit[keep]
