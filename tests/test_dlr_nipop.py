"""Tests for NIPOP restart scheduler in DLR portfolio."""

from __future__ import annotations

import math

import pytest
import torch

from torch_dfo.dlr_cma import DLRPortfolio


def _make_dlr(
    dim: int = 10,
    K: int = 2,
    lambdas: tuple[int, ...] = (12, 12),
    sigma_fracs: tuple[float, ...] = (0.3, 0.3),
    sigma_factors: tuple[float, ...] | None = None,
    cma_budget: int | None = None,
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
        sigma_factors=sigma_factors,
        cma_budget=cma_budget,
    )


def test_nipop_lambda_doubles_on_restart() -> None:
    """Lambda should double at NIPOP level 1."""
    dim = 10
    lam_default = 4 + math.floor(3 * math.log(max(dim, 2)))
    dlr = _make_dlr(dim=dim, K=2, lambdas=(lam_default, lam_default))
    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=1)
    assert dlr.lambdas[0] == min(lam_default * 2, dlr.lam_max)


def test_nipop_sigma_decreases_on_restart() -> None:
    """Sigma should decrease by sigma_factor^nipop_level on restart."""
    dim = 10
    dlr = _make_dlr(
        dim=dim,
        K=2,
        lambdas=(12, 12),
        sigma_fracs=(0.3, 0.3),
        sigma_factors=(1.6, 2.0),
    )
    span = 10.0  # ub - lb mean for [-5, 5]
    sigma_default = 0.3 * span
    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=2)
    expected = sigma_default / (1.6**2)
    assert abs(dlr.sigmas[0].item() - expected) < 1e-6


def test_nipop_lam_max_cap() -> None:
    """Lambda should never exceed lam_max."""
    dim = 10
    dlr = _make_dlr(
        dim=dim,
        K=1,
        lambdas=(12,),
        sigma_fracs=(0.3,),
        cma_budget=1000,
    )
    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=10)
    assert dlr.lambdas[0] <= dlr.lam_max


def test_ask_skips_large_branches() -> None:
    """ask() with remaining_budget should skip branches with lambda > budget."""
    dim = 10
    dlr = _make_dlr(
        dim=dim,
        K=2,
        lambdas=(12, 12),
        sigma_fracs=(0.3, 0.3),
        cma_budget=20000,
    )
    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=5)  # lambda grows large
    candidates = dlr.ask(remaining_budget=20)
    # Should only get branch 1's candidates (12), not branch 0's large lambda
    assert candidates.shape[0] <= 20


def test_ask_returns_empty_when_budget_exhausted() -> None:
    """ask() should return empty tensor when remaining_budget < min(lambdas)."""
    dim = 10
    dlr = _make_dlr(dim=dim, K=2, lambdas=(12, 12), sigma_fracs=(0.3, 0.3))
    candidates = dlr.ask(remaining_budget=5)  # less than min lambda
    assert candidates.shape[0] == 0


def test_nipop_default_sigma_factors() -> None:
    """Default sigma_factors should be (1.4, 1.6, 2.0, 1.2)[:K]."""
    dlr = _make_dlr(dim=10, K=3, lambdas=(12, 12, 12), sigma_fracs=(0.3, 0.3, 0.3))
    assert dlr.sigma_factors == (1.4, 1.6, 2.0)


def test_nipop_level_tracked_per_branch() -> None:
    """Each branch should track its own NIPOP level."""
    dim = 10
    dlr = _make_dlr(dim=dim, K=2, lambdas=(12, 12), sigma_fracs=(0.3, 0.3))
    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=3)
    dlr.restart_branch(1, x0, nipop_level=1)
    assert dlr._nipop_level[0] == 3
    assert dlr._nipop_level[1] == 1


def test_restart_resets_state_with_nipop() -> None:
    """NIPOP restart should still reset all branch state (log_d, V, p_sigma, etc.)."""
    dim = 10
    dlr = _make_dlr(dim=dim, K=1, lambdas=(12,), sigma_fracs=(0.3,))

    # Run a few generations to build up state
    for _ in range(3):
        cands = dlr.ask()
        dlr.tell(cands, torch.arange(cands.shape[0], dtype=torch.float64))

    x0 = torch.zeros(dim, dtype=torch.float64)
    dlr.restart_branch(0, x0, nipop_level=1)

    assert torch.allclose(dlr.log_d[0], torch.zeros(dim, dtype=torch.float64))
    assert torch.allclose(dlr.p_sigma[0], torch.zeros(dim, dtype=torch.float64))
    assert torch.allclose(dlr.p_c[0], torch.zeros(dim, dtype=torch.float64))
    assert dlr.stag_count[0].item() == 0
    assert dlr._rank_mu_col[0] == 0


def test_ask_partial_skip_some_branches_active() -> None:
    """D6: remaining_budget chosen so SOME branches are skipped but not all.

    With K=3 branches at lambdas (8, 12, 20), a remaining_budget of 14 should
    admit branches 0 and 1 (lambdas 8, 12; both ≤ 14) and skip branch 2
    (lambda 20 > 14). Resulting candidate count: 8 + 12 = 20.
    """
    dim = 10
    dlr = _make_dlr(
        dim=dim,
        K=3,
        lambdas=(8, 12, 20),
        sigma_fracs=(0.3, 0.3, 0.3),
    )
    # Budget admits branches 0 and 1, skips branch 2.
    remaining = 14
    candidates = dlr.ask(remaining_budget=remaining)
    expected_pop = 8 + 12
    assert candidates.shape == (expected_pop, dim), (
        f"Partial-skip shape wrong: got {candidates.shape}, expected "
        f"({expected_pop}, {dim}). Active branches should be 0 and 1."
    )
    # All candidates must be within bounds.
    assert (candidates >= dlr.lb).all()
    assert (candidates <= dlr.ub).all()
    # And the active-branch bookkeeping matches.
    assert dlr._last_active == [0, 1], (
        f"_last_active wrong: got {dlr._last_active}, expected [0, 1]"
    )


# ----------------------------------------------------------------------
# Constructor API unification (B4): bounds= / seed= alignment with peers
# ----------------------------------------------------------------------
def test_dlrportfolio_accepts_bounds_kwarg() -> None:
    """Happy path: new unified API (bounds= + seed=) constructs and samples."""
    lambdas = (12, 12)
    # Pin to CPU so float64 works on MPS-equipped test hosts (matches how
    # sibling optimizer tests — CMAES, SHADE — are written).
    dlr = DLRPortfolio(dim=5, bounds=5.0, lambdas=lambdas, seed=42, device="cpu")
    assert dlr.lb.shape == (5,)
    assert dlr.ub.shape == (5,)
    # Bounds resolved symmetrically via normalize_bounds.
    assert torch.allclose(dlr.lb, torch.full((5,), -5.0, dtype=dlr.dtype))
    assert torch.allclose(dlr.ub, torch.full((5,), 5.0, dtype=dlr.dtype))
    candidates = dlr.ask()
    assert candidates.shape == (sum(lambdas), 5)


def test_dlrportfolio_seed_kwarg_produces_deterministic_output() -> None:
    """The seed= kwarg must actually reach the internal generator.

    If __init__ silently ignored seed= (e.g. ``seed or rng`` ordering flipped),
    two optimizers constructed with the same seed would diverge. Check that
    they agree bit-exactly on the first ask.
    """
    kwargs = {"dim": 5, "bounds": 5.0, "lambdas": (12, 12), "device": "cpu"}
    a = DLRPortfolio(seed=1234, **kwargs)
    b = DLRPortfolio(seed=1234, **kwargs)
    assert torch.equal(a.ask(), b.ask()), "same seed produced divergent asks"
    c = DLRPortfolio(seed=9999, **kwargs)
    assert not torch.equal(a.ask(), c.ask()), "different seeds produced identical asks"


def test_dlrportfolio_lb_ub_emits_deprecation_warning() -> None:
    """Back-compat path: lb=/ub= still works but raises DeprecationWarning."""
    dim = 5
    lb = torch.full((dim,), -5.0, dtype=torch.float64)
    ub = torch.full((dim,), 5.0, dtype=torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(7)
    with pytest.warns(DeprecationWarning, match="lb/ub"):
        dlr = DLRPortfolio(
            dim=dim,
            lb=lb,
            ub=ub,
            lambdas=(12, 12),
            sigma_fracs=(0.3, 0.1),
            device=torch.device("cpu"),
            dtype=torch.float64,
            rng=rng,
        )
    candidates = dlr.ask()
    assert candidates.shape == (24, dim)


def test_dlrportfolio_rejects_both_bounds_and_lb() -> None:
    """Passing both bounds= and lb=/ub= must raise ValueError with a clear message."""
    dim = 5
    lb = torch.full((dim,), -5.0, dtype=torch.float64)
    ub = torch.full((dim,), 5.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="bounds"):
        DLRPortfolio(dim=dim, bounds=5.0, lb=lb, ub=ub, seed=0)
