"""Tests for Oja-style rank-mu V-update in DLR portfolio."""

from __future__ import annotations

import torch

from tests._thresholds import ATOL_F32_TIGHT
from torch_dfo.dlr_cma import DLRPortfolio


def _make_dlr(
    dim: int = 10,
    K: int = 2,
    lambdas: tuple[int, ...] = (12, 12),
    sigma_fracs: tuple[float, ...] = (0.3, 0.1),
) -> DLRPortfolio:
    """Helper to build a DLRPortfolio with standard test settings."""
    device = torch.device("cpu")
    dtype = torch.float64
    lb = torch.full((dim,), -5.0, device=device, dtype=dtype)
    ub = torch.full((dim,), 5.0, device=device, dtype=dtype)
    rng = torch.Generator(device=device).manual_seed(42)
    return DLRPortfolio(
        dim=dim,
        lb=lb,
        ub=ub,
        lambdas=lambdas,
        sigma_fracs=sigma_fracs,
        device=device,
        dtype=dtype,
        rng=rng,
    )


def test_rank_mu_columns_update() -> None:
    """Rank-mu V columns should change after tell() with non-uniform fitness."""
    dlr = _make_dlr(dim=10, K=2, lambdas=(12, 12), sigma_fracs=(0.3, 0.1))
    k = dlr.V.shape[-1]
    rank_mu_start = k // 2
    v_before = dlr.V[:, :, rank_mu_start:].clone()

    candidates = dlr.ask()
    fitness = torch.arange(candidates.shape[0], dtype=torch.float64)
    dlr.tell(candidates, fitness)

    v_after = dlr.V[:, :, rank_mu_start:]
    assert not torch.allclose(v_before, v_after), "Rank-mu V columns did not update"


def test_rank1_columns_still_update() -> None:
    """Rank-1 (p_c) V columns should still update."""
    dlr = _make_dlr(dim=10, K=1, lambdas=(12,), sigma_fracs=(0.3,))
    k = dlr.V.shape[-1]
    rank1_end = k // 2
    v_before = dlr.V[:, :, :rank1_end].clone()

    candidates = dlr.ask()
    fitness = torch.arange(candidates.shape[0], dtype=torch.float64)
    dlr.tell(candidates, fitness)

    v_after = dlr.V[:, :, :rank1_end]
    assert not torch.allclose(v_before, v_after), "Rank-1 V columns did not update"


def test_oja_ema_blending() -> None:
    """V columns should use EMA, not hard overwrite."""
    dlr = _make_dlr(dim=10, K=1, lambdas=(12,), sigma_fracs=(0.3,))
    k = dlr.V.shape[-1]
    col = k // 2  # first rank-mu column

    k_mu = k - k // 2
    # Run enough tells to cycle back to the same column with different fitness
    # After k_mu tells, the counter wraps and column `col` is written again.
    cands = dlr.ask()
    dlr.tell(cands, torch.arange(cands.shape[0], dtype=torch.float64))
    v_after_first = dlr.V[0, :, col].clone()

    # Advance counter through remaining rank-mu columns
    for _ in range(k_mu - 1):
        cands = dlr.ask()
        dlr.tell(cands, torch.arange(cands.shape[0], dtype=torch.float64))

    # Now counter is back at col. Tell with REVERSED fitness.
    cands = dlr.ask()
    fitness_rev = torch.arange(cands.shape[0], dtype=torch.float64).flip(0)
    dlr.tell(cands, fitness_rev)
    v_after_second = dlr.V[0, :, col]

    # With EMA (alpha=0.3), the column retains memory from the first write
    # but blends in the new direction from reversed fitness.
    cos_sim = torch.dot(v_after_first, v_after_second) / (
        v_after_first.norm() * v_after_second.norm() + 1e-30
    )
    assert cos_sim > 0.2, f"EMA blending lost all memory: cos_sim={cos_sim:.3f}"
    assert not torch.allclose(v_after_first, v_after_second, atol=ATOL_F32_TIGHT), (
        "V column unchanged — EMA not updating on reversed fitness"
    )


def test_rank_mu_counter_cycles() -> None:
    """The rank-mu column counter should cycle through all rank-mu columns."""
    dlr = _make_dlr(dim=10, K=1, lambdas=(12,), sigma_fracs=(0.3,))
    k = dlr.V.shape[-1]
    k_half = k // 2
    num_mu_cols = max(k - k_half, 1)

    # Run enough generations to cycle through all rank-mu columns
    for _ in range(num_mu_cols + 1):
        cands = dlr.ask()
        dlr.tell(cands, torch.arange(cands.shape[0], dtype=torch.float64))

    # After num_mu_cols+1 tells, counter should have wrapped around
    assert dlr._rank_mu_col[0] == 1, f"Counter should wrap: expected 1, got {dlr._rank_mu_col[0]}"
