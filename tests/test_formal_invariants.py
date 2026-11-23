"""Tier-2 formal-refinement tests binding Lean 4 / Z3 invariants to PyTorch code."""

from __future__ import annotations

import pytest
import torch

import torch_dfo.dlr_cma as dlr_cma_mod
from torch_dfo.cmaes import CMAES
from torch_dfo.dlr_cma import DLRPortfolio

# ===========================================================================
# Phase 1: CMAES & DLRPortfolio mathematical invariants
# ===========================================================================


class TestCMAESFormalInvariants:
    """Refinement tests for Hansen (2016) CMA-ES rates, active PSD, gauge, and restart."""

    @pytest.mark.parametrize(
        ("dim", "pop_size"),
        [(3, 64), (16, 128), (16, 4096)],
    )
    def test_cmaes_cc_strictly_between_0_and_1_large_pop(self, dim: int, pop_size: int) -> None:
        """Hansen (2016) Table 1 c_c stays in (0, 1) and keeps p_c active at large pop_size."""
        opt = CMAES(
            dim=dim, bounds=5.0, pop_size=pop_size, device="cpu", dtype=torch.float64, seed=0
        )
        assert 0.0 < opt.c_c < 1.0, f"dim={dim}, pop={pop_size}: c_c={opt.c_c} violates 0 < c_c < 1"
        expected_cc = (4.0 + opt.mu_eff / dim) / (dim + 4.0 + 2.0 * opt.mu_eff / dim)
        assert opt.c_c == pytest.approx(expected_cc, rel=1e-12, abs=1e-12)

        x = opt.ask()
        opt.tell(x, (x**2).sum(dim=-1))
        assert torch.linalg.vector_norm(opt.p_c).item() > 0.0, (
            f"dim={dim}, pop={pop_size}: p_c collapsed to zero after tell()"
        )

    def test_cmaes_active_preserves_psd_before_eigendecomp(self) -> None:
        """Active CMA covariance update must remain PSD even under extreme worst-point outliers."""
        dim = 4
        opt = CMAES(
            dim=dim,
            bounds=100.0,
            pop_size=10,
            device="cpu",
            dtype=torch.float64,
            seed=7,
            active=True,
        )
        # Defer eigendecomp so we observe raw updated C before any clamping.
        opt._decomp_frequency = 100
        opt.mean.zero_()
        opt.sigma = 1.0
        opt.C = torch.eye(dim, dtype=torch.float64)
        opt.C_invsqrt = torch.eye(dim, dtype=torch.float64)

        # Best mu=5 near 0; worst mu_neg=5 are extreme outliers along e_0 (norm 100 * sigma).
        candidates = torch.zeros(10, dim, dtype=torch.float64)
        candidates[:5] = 0.01 * torch.randn(5, dim, dtype=torch.float64)
        candidates[5:, 0] = 100.0
        fitness = torch.arange(10, dtype=torch.float64)

        opt.tell(candidates, fitness)
        C_sym = 0.5 * (opt.C + opt.C.T)
        min_eig = torch.linalg.eigvalsh(C_sym).min().item()
        assert min_eig >= -1e-12, f"Active CMA broke PSD before eigendecomp: min_eig={min_eig:.6e}"

    def test_cmaes_normalize_on_decomp_defaults_false_and_gauge_invariant(self) -> None:
        """Default _normalize_on_decomp is False; when normalization triggers, 3-var gauge holds."""
        dim = 4
        opt = CMAES(dim=dim, bounds=5.0, pop_size=10, device="cpu", dtype=torch.float64, seed=1)
        assert opt._normalize_on_decomp is False

        # Explicit normalization preserves sigma^2 * C and sigma * p_c.
        opt._normalize_on_decomp = True
        opt.C = torch.diag(torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float64))
        opt.sigma = 1.5
        opt.p_c = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)

        cov_before = (opt.sigma**2) * opt.C.clone()
        scaled_pc_before = opt.sigma * opt.p_c.clone()

        opt._update_eigensystem()

        cov_after = (opt.sigma**2) * opt.C
        scaled_pc_after = opt.sigma * opt.p_c
        assert (opt.D_diag**2).mean().item() == pytest.approx(1.0, rel=1e-12)
        assert torch.allclose(cov_after, cov_before, rtol=1e-12, atol=1e-12), (
            "Normalization altered sampling covariance sigma^2 * C"
        )
        assert torch.allclose(scaled_pc_after, scaled_pc_before, rtol=1e-12, atol=1e-12), (
            "Normalization broke p_c gauge scaling sigma * p_c"
        )

        # Extreme float32 safeguard: mean_eig > 1e6 triggers gauge-preserving norm even when False.
        opt32 = CMAES(dim=dim, bounds=5.0, pop_size=10, device="cpu", dtype=torch.float32, seed=1)
        assert opt32._normalize_on_decomp is False
        opt32.C = torch.diag(torch.tensor([1e7, 2e7, 3e7, 4e7], dtype=torch.float32))
        opt32.sigma = 1e-3
        opt32.p_c = torch.tensor([100.0, -200.0, 50.0, 300.0], dtype=torch.float32)
        cov32_before = (opt32.sigma**2) * opt32.C.clone()
        pc32_before = opt32.sigma * opt32.p_c.clone()

        opt32._update_eigensystem()

        assert (opt32.D_diag**2).mean().item() == pytest.approx(1.0, rel=1e-5)
        assert torch.allclose((opt32.sigma**2) * opt32.C, cov32_before, rtol=1e-5, atol=1e-5)
        assert torch.allclose(opt32.sigma * opt32.p_c, pc32_before, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("bad_pop", [0, 1, -4])
    def test_cmaes_restart_rejects_pop_size_lt_2(self, bad_pop: int) -> None:
        """restart(new_pop_size < 2) must raise ValueError matching __init__."""
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device="cpu", dtype=torch.float64, seed=0)
        with pytest.raises(ValueError, match="pop_size"):
            opt.restart(new_pop_size=bad_pop)


class TestDLRPortfolioFormalInvariants:
    """Refinement tests for DLRPortfolio c_c, diagonal variance, sigma order, NIPOP, and RNG."""

    def test_dlr_cc_strictly_between_0_and_1_init_and_resize(self) -> None:
        """DLRPortfolio uses Hansen c_c in (0, 1) at init and after NIPOP resize."""
        dim = 16
        dlr = DLRPortfolio(
            dim=dim,
            bounds=5.0,
            lambdas=(128, 256),
            sigma_fracs=(0.2, 0.1),
            cma_budget=200_000,
            seed=0,
            device="cpu",
            dtype=torch.float64,
        )
        for ki in range(dlr.K):
            mue = dlr._mu_eff[ki]
            expected = (4.0 + mue / dim) / (dim + 4.0 + 2.0 * mue / dim)
            assert 0.0 < dlr._c_c[ki] < 1.0
            assert dlr._c_c[ki] == pytest.approx(expected, rel=1e-12)

        dlr.restart_branch(0, torch.zeros(dim, dtype=torch.float64), nipop_level=5)
        mue0 = dlr._mu_eff[0]
        expected0 = (4.0 + mue0 / dim) / (dim + 4.0 + 2.0 * mue0 / dim)
        assert 0.0 < dlr._c_c[0] < 1.0
        assert dlr._c_c[0] == pytest.approx(expected0, rel=1e-12)

    def test_dlr_diagonal_variance_stable_when_stalled(self) -> None:
        """With h_sigma == 0, delta_h must be multiplied by c1 so D does not inflate."""
        dim = 10
        dlr = DLRPortfolio(
            dim=dim,
            bounds=100.0,
            lambdas=(12,),
            sigma_fracs=(0.01,),
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        # Force a large p_sigma so h_sigma == 0 triggers, and place all candidates at mean
        # (diag_1 = 0, diag_mu = 0) to isolate the base coefficient (1 - c1 - cmu + c1 * delta_h).
        dlr.p_sigma[0] = torch.full((dim,), 10.0, dtype=torch.float64)
        dlr.p_c[0] = torch.zeros(dim, dtype=torch.float64)
        dlr.log_d[0] = torch.zeros(dim, dtype=torch.float64)

        cands = dlr.means[0].unsqueeze(0).repeat(12, 1)
        fit = torch.arange(12, dtype=torch.float64)
        dlr.tell(cands, fit)

        c1 = dlr._c_1[0]
        cmu = dlr._c_mu[0]
        cc = dlr._c_c[0]
        delta_h = cc * (2.0 - cc)
        expected_d = 1.0 - c1 - cmu + c1 * delta_h
        actual_d = torch.exp(dlr.log_d[0])
        assert torch.allclose(
            actual_d, torch.full((dim,), expected_d, dtype=torch.float64), rtol=1e-10
        ), f"Expected d={expected_d:.6f} (with c1*delta_h), got {actual_d[0].item():.6f}"
        assert expected_d < 1.0

    def test_dlr_y_sel_uses_pre_update_sigma(self) -> None:
        """y_sel in diagonal rank-mu update must divide by pre-update sigma_old."""
        dim = 6
        dlr = DLRPortfolio(
            dim=dim,
            bounds=100.0,
            lambdas=(8,),
            sigma_fracs=(0.01,),
            seed=0,
            device="cpu",
            dtype=torch.float64,
        )
        # Pre-load a large p_sigma so sigma jumps sharply during tell(), exposing sigma-order bug.
        dlr.p_sigma[0] = torch.full((dim,), 20.0, dtype=torch.float64)
        dlr.p_c[0] = torch.zeros(dim, dtype=torch.float64)
        dlr.log_d[0] = torch.zeros(dim, dtype=torch.float64)
        sigma_old = dlr.sigmas[0].item()

        # Construct symmetric candidates around mean so mean_diff == 0 (y_w == 0, p_c == 0)
        # while selected candidates have non-zero variance (y_sel != 0).
        mean0 = dlr.means[0].clone()
        offsets = torch.tensor(
            [[1.0], [-1.0], [1.0], [-1.0], [2.0], [-2.0], [2.0], [-2.0]],
            dtype=torch.float64,
        ).expand(8, dim)
        cands = mean0.unsqueeze(0) + sigma_old * offsets
        # Make top-4 weights symmetric: indices 0,1 share best fitness, 2,3 next.
        # Compute exact expected diag_mu using sigma_old.
        fit = torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0, 11.0, 12.0, 13.0], dtype=torch.float64)
        mu_k = dlr._mu[0]
        w_k = dlr._weights[0, :mu_k]
        y_sel_expected = (cands[:mu_k] - mean0.unsqueeze(0)) / sigma_old
        diag_mu_expected = (w_k.unsqueeze(1) * y_sel_expected**2).sum(dim=0)

        dlr.tell(cands, fit)

        # Since p_sigma norm is huge, h_sigma == 0 so p_c == 0 and diag_1 == 0.
        c1 = dlr._c_1[0]
        cmu = dlr._c_mu[0]
        cc = dlr._c_c[0]
        delta_h = cc * (2.0 - cc)
        expected_d = (1.0 - c1 - cmu + c1 * delta_h) * 1.0 + cmu * diag_mu_expected
        actual_d = torch.exp(dlr.log_d[0])
        assert torch.allclose(actual_d, expected_d, rtol=1e-9, atol=1e-9), (
            f"diag_mu used post-update sigma: expected {expected_d[0].item():.6f}, "
            f"got {actual_d[0].item():.6f}"
        )

    def test_dlr_restart_branch_resets_generation(self) -> None:
        """restart_branch(ki, ...) must reset _generation[ki] back to 0."""
        dim = 8
        dlr = DLRPortfolio(dim=dim, bounds=5.0, lambdas=(12, 12), seed=1, device="cpu")
        for _ in range(5):
            x = dlr.ask()
            dlr.tell(x, (x**2).sum(dim=-1))
        assert dlr._generation[0].item() == 5
        assert dlr._generation[1].item() == 5

        dlr.restart_branch(0, torch.zeros(dim, dtype=torch.float64), nipop_level=1)
        assert dlr._generation[0].item() == 0
        assert dlr._generation[1].item() == 5

    def test_dlr_supports_k_gt_4_and_enforces_lam_max_ge_2(self) -> None:
        """DLRPortfolio supports K > 4 default sigma_factors and enforces lam_max >= 2."""
        K = 6
        dlr = DLRPortfolio(
            dim=8,
            bounds=5.0,
            lambdas=(8,) * K,
            sigma_fracs=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005),
            seed=0,
            device="cpu",
        )
        assert len(dlr.sigma_factors) == K
        assert all(f > 1.0 for f in dlr.sigma_factors)
        # Must not raise IndexError for ki >= 4
        dlr.restart_branch(5, torch.zeros(8, dtype=torch.float64), nipop_level=1)

        with pytest.raises(ValueError, match="cma_budget"):
            DLRPortfolio(
                dim=8,
                bounds=5.0,
                lambdas=(8, 8),
                sigma_fracs=(0.2, 0.1),
                cma_budget=3,
                seed=0,
                device="cpu",
            )

        tight = DLRPortfolio(
            dim=8,
            bounds=5.0,
            lambdas=(4, 4),
            sigma_fracs=(0.2, 0.1),
            cma_budget=4,
            seed=0,
            device="cpu",
        )
        assert tight.lam_max >= 2

    def test_dlr_cross_device_rng_guard_in_load_state_dict(self) -> None:
        """DLRPortfolio.load_state_dict handles cross-device checkpoints via seed fallback."""
        src = DLRPortfolio(dim=6, bounds=5.0, lambdas=(8, 8), seed=42, device="cpu")
        x = src.ask()
        src.tell(x, (x**2).sum(dim=-1))
        state = src.state_dict()
        assert state["_rng_seed"] == 42
        assert state["_saved_device_type"] == "cpu"

        # Simulate checkpoint saved from a different device type with incompatible RNG bytes.
        fake_cross = dict(state)
        fake_cross["_saved_device_type"] = "cuda"
        fake_cross["_rng_state"] = torch.zeros(8, dtype=torch.uint8)

        dst = DLRPortfolio(dim=6, bounds=5.0, lambdas=(8, 8), seed=99, device="cpu")
        dst.load_state_dict(fake_cross)
        assert dst._rng.initial_seed() == 42

    def test_dlr_no_dead_woodbury_solve(self) -> None:
        """Dead _woodbury_solve helper has been removed."""
        assert not hasattr(dlr_cma_mod, "_woodbury_solve")
