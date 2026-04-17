"""Tests for torch_dfo.cmaes -- CMA-ES optimizer with IPOP restart support."""

from __future__ import annotations

import math

import pytest
import torch

from tests._thresholds import (
    ATOL_F32_DEFAULT,
    ATOL_F64_DEFAULT,
    ATOL_F64_TIGHT,
    CONV_CMAES_RASTRIGIN_10D,
    CONV_SPHERE_1D,
    CONV_SPHERE_10D_TIGHT,
    CONV_SPHERE_HIGH_DIM,
    RTOL_DEFAULT,
    SMOKE_F_INIT_CMAES,
    TOL_CMAES_BEST_F32,
    TOL_CMAES_BEST_F64,
    TOL_CMAES_C_SYMMETRY_F32,
    TOL_CMAES_C_SYMMETRY_F64,
    TOL_CMAES_INVSQRT_IDENTITY_F32,
    TOL_CMAES_INVSQRT_IDENTITY_F64,
)
from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import rastrigin, sphere
from torch_dfo.cmaes import CMAES, _default_pop_size


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------
class TestConstruction:
    """Validate constructor wiring, default pop_size, and tensor shapes."""

    def test_default_pop_size_formula(self) -> None:
        for dim in (2, 5, 10, 30, 100):
            expected = 4 + math.floor(3 * math.log(dim))
            assert _default_pop_size(dim) == expected

    def test_default_pop_size_used(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 10
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt.pop_size == _default_pop_size(dim)

    def test_explicit_pop_size(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        assert opt.pop_size == 20

    def test_state_shapes(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 8
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)

        assert opt.mean.shape == (dim,)
        assert opt.C.shape == (dim, dim)
        assert opt.B.shape == (dim, dim)
        assert opt.D_diag.shape == (dim,)
        assert opt.C_invsqrt.shape == (dim, dim)
        assert opt.p_sigma.shape == (dim,)
        assert opt.p_c.shape == (dim,)

    def test_state_device(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=4, bounds=5.0, device=device, dtype=dtype, seed=42)
        for t in (opt.mean, opt.C, opt.B, opt.D_diag, opt.C_invsqrt, opt.p_sigma, opt.p_c):
            assert t.device.type == device.type

    def test_state_dtype(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=4, bounds=5.0, device=device, dtype=dtype, seed=42)
        for t in (opt.mean, opt.C, opt.B, opt.D_diag, opt.C_invsqrt, opt.p_sigma, opt.p_c):
            assert t.dtype == dtype

    def test_initial_mean_is_center(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=3, bounds=(-2.0, 4.0), device=device, dtype=dtype, seed=42)
        expected = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        assert torch.allclose(opt.mean, expected)

    def test_initial_covariance_is_identity(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        eye = torch.eye(dim, device=device, dtype=dtype)
        assert torch.allclose(opt.C, eye)

    def test_initial_paths_are_zero(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        zeros = torch.zeros(5, device=device, dtype=dtype)
        assert torch.equal(opt.p_sigma, zeros)
        assert torch.equal(opt.p_c, zeros)

    def test_mu_and_weights(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        pop_size = 14
        opt = CMAES(dim=5, bounds=5.0, pop_size=pop_size, device=device, dtype=dtype, seed=42)
        assert opt.mu == pop_size // 2
        assert opt.weights.shape == (opt.mu,)
        assert torch.allclose(opt.weights.sum(), torch.tensor(1.0, dtype=dtype, device=device))
        # Weights are positive and descending
        assert torch.all(opt.weights > 0)
        diffs = opt.weights[1:] - opt.weights[:-1]
        assert torch.all(diffs <= 0)


# ---------------------------------------------------------------------------
# Strategy parameter formulas (Hansen 2016, Table 1)
# ---------------------------------------------------------------------------
class TestStrategyParameters:
    """Verify that strategy parameters follow Hansen 2016 formulas exactly."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_strategy_parameter_formulas(self, device: torch.device) -> None:
        """CMA-ES strategy parameters c_sigma, d_sigma, c_c, c_1, c_mu match Hansen's formulas at dim=10."""
        dtype = torch.float64
        dim = 10
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)

        # Default pop_size = 4 + floor(3*ln(10)) = 4 + 6 = 10 ... actually let's check
        expected_pop = 4 + math.floor(3 * math.log(dim))
        assert opt.pop_size == expected_pop
        mu = expected_pop // 2

        # Compute mu_eff from weights
        raw = torch.log(torch.tensor((expected_pop + 1) / 2, dtype=dtype)) - torch.log(
            torch.arange(1, mu + 1, dtype=dtype),
        )
        weights = raw / raw.sum()
        mu_eff = (1.0 / (weights**2).sum()).item()
        assert opt.mu_eff == pytest.approx(mu_eff, abs=ATOL_F64_DEFAULT)

        # c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        expected_c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        assert opt.c_sigma == pytest.approx(expected_c_sigma, abs=ATOL_F64_DEFAULT)

        # d_sigma = 1 + 2*max(0, sqrt((mu_eff-1)/(dim+1)) - 1) + c_sigma
        expected_d_sigma = (
            1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + expected_c_sigma
        )
        assert opt.d_sigma == pytest.approx(expected_d_sigma, abs=ATOL_F64_DEFAULT)

        # c_c = (mu_eff + 2) / (dim + 4 + 2*mu_eff/dim)
        expected_c_c = (mu_eff + 2) / (dim + 4 + 2 * mu_eff / dim)
        assert opt.c_c == pytest.approx(expected_c_c, abs=ATOL_F64_DEFAULT)

        # c_1 = 2 / ((dim+1.3)^2 + mu_eff)
        expected_c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        assert opt.c_1 == pytest.approx(expected_c_1, abs=ATOL_F64_DEFAULT)

        # c_mu = min(1-c_1, 2*(mu_eff-2+1/mu_eff)/((dim+2)^2+mu_eff))
        expected_c_mu = min(
            1 - expected_c_1,
            2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff),
        )
        assert opt.c_mu == pytest.approx(expected_c_mu, abs=ATOL_F64_DEFAULT)


# ---------------------------------------------------------------------------
# Decomposition frequency
# ---------------------------------------------------------------------------
class TestDecompositionFrequency:
    """Verify eigendecomposition frequency follows the formula."""

    @pytest.mark.parametrize("dim", [5, 20, 50])
    def test_decomposition_frequency(self, dim: int) -> None:
        """``_decomp_frequency`` matches Hansen's schedule across several dimensions."""
        opt = CMAES(
            dim=dim,
            bounds=5.0,
            device=torch.device("cpu"),
            dtype=torch.float64,
            seed=42,
        )
        expected = max(1, math.floor(1 / (10 * dim * (opt.c_1 + opt.c_mu))))
        assert opt._decomp_frequency == expected, (
            f"dim={dim}: got {opt._decomp_frequency}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# ask() output properties
# ---------------------------------------------------------------------------
class TestAsk:
    """Verify ask() returns correctly shaped, bounded, on-device candidates."""

    def test_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim, pop_size = 6, 12
        opt = CMAES(dim=dim, bounds=5.0, pop_size=pop_size, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert candidates.shape == (pop_size, dim)

    def test_within_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=(-3.0, 7.0), pop_size=50, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert torch.all(candidates >= opt.lb)
        assert torch.all(candidates <= opt.ub)

    def test_device_and_dtype(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=4, bounds=5.0, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert candidates.device.type == device.type
        assert candidates.dtype == dtype

    def test_population_stored(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=4, bounds=5.0, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert torch.equal(opt.population, candidates)


# ---------------------------------------------------------------------------
# tell() updates mean and sigma
# ---------------------------------------------------------------------------
class TestTell:
    """Verify tell() updates the distribution parameters correctly."""

    def test_mean_changes(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        old_mean = opt.mean.clone()
        candidates = opt.ask()
        fitness = sphere(candidates)
        opt.tell(candidates, fitness)
        # Mean should have moved
        assert not torch.equal(opt.mean, old_mean)

    def test_mean_moves_toward_better_solutions(self, device: torch.device) -> None:
        """Given enough generations, the mean migrates from the off-center bounds toward the sphere optimum."""
        dtype = best_float_dtype(device)
        # Use asymmetric bounds so center (2.5) is far from the sphere optimum (0)
        opt = CMAES(
            dim=5,
            bounds=(1.0, 8.0),
            pop_size=20,
            device=device,
            dtype=dtype,
            seed=42,
        )
        old_mean = opt.mean.clone()
        # Run 30 generations so the mean has time to converge
        for _ in range(30):
            candidates = opt.ask()
            fitness = sphere(candidates)
            opt.tell(candidates, fitness)
        # New mean should be closer to the origin (sphere optimum) than old mean
        old_dist = torch.linalg.norm(old_mean).item()
        new_dist = torch.linalg.norm(opt.mean).item()
        assert new_dist < old_dist

    def test_generation_increments(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt._generation == 0
        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 1
        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 2

    def test_sigma_decreases_on_sphere(self, device: torch.device) -> None:
        """Sigma is not monotonic; check that it eventually decreases from its initial value on the sphere."""
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        initial_sigma = opt.sigma
        for _ in range(200):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # After 200 generations on sphere, sigma should have decreased significantly
        assert opt.sigma < initial_sigma * 0.1, (
            f"sigma={opt.sigma:.6f} not < 10% of initial={initial_sigma:.6f}"
        )


# ---------------------------------------------------------------------------
# Eigendecomposition stays valid
# ---------------------------------------------------------------------------
class TestEigendecomposition:
    """Verify C stays positive definite through optimization."""

    def test_c_stays_positive_definite(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        for _ in range(50):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # Check eigenvalues of C are all positive (CPU fallback for MPS)
        C_cpu = opt.C.to("cpu") if device.type not in ("cpu", "cuda") else opt.C
        eigvals = torch.linalg.eigvalsh(C_cpu)
        assert torch.all(eigvals > 0), f"Negative eigenvalue found: {eigvals.min().item()}"

    def test_c_is_symmetric(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        for _ in range(50):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # C should be symmetric
        tol = TOL_CMAES_C_SYMMETRY_F32 if dtype == torch.float32 else TOL_CMAES_C_SYMMETRY_F64
        assert torch.allclose(opt.C, opt.C.T, atol=tol)

    def test_c_invsqrt_consistency(self, device: torch.device) -> None:
        """C_invsqrt @ C @ C_invsqrt should approximate I."""
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        for _ in range(20):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # Force fresh decomposition
        opt._update_eigensystem()
        product = opt.C_invsqrt @ opt.C @ opt.C_invsqrt
        eye = torch.eye(opt.dim, device=device, dtype=dtype)
        tol = (
            TOL_CMAES_INVSQRT_IDENTITY_F32
            if dtype == torch.float32
            else TOL_CMAES_INVSQRT_IDENTITY_F64
        )
        assert torch.allclose(product, eye, atol=tol)

    def test_c_invsqrt_consistent_at_natural_decomp(self, device: torch.device) -> None:
        """H11: C_invsqrt stays consistent after a natural decomp boundary.

        Runs past ``_decomp_frequency`` generations so the periodic eigensystem
        refresh happens organically inside ``tell()``, then asserts
        ``C_invsqrt @ C @ C_invsqrt ≈ I`` without calling ``_update_eigensystem``
        manually. Complements ``test_c_invsqrt_consistency`` which forces a
        fresh decomp right before the assertion.
        """
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        # Run enough generations to cross at least two natural decomp boundaries.
        n_gens = max(2 * opt._decomp_frequency + 1, 20)
        for _ in range(n_gens):
            c = opt.ask()
            opt.tell(c, sphere(c))
        product = opt.C_invsqrt @ opt.C @ opt.C_invsqrt
        eye = torch.eye(opt.dim, device=device, dtype=dtype)
        tol = (
            TOL_CMAES_INVSQRT_IDENTITY_F32
            if dtype == torch.float32
            else TOL_CMAES_INVSQRT_IDENTITY_F64
        )
        assert torch.allclose(product, eye, atol=tol), (
            f"C_invsqrt @ C @ C_invsqrt drifted from I after {n_gens} natural gens"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_eigenvalue_clamping(self, device: torch.device) -> None:
        """Near-zero eigenvalue gets clamped to >= 1e-8 before normalization.

        After normalization (eigenvalues / mean), the reconstructed C should
        have eigenvalues that reflect the 1e-8 floor rather than 1e-25.
        """
        dtype = torch.float64
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        # Set C to have a near-zero eigenvalue
        diag_vals = torch.ones(dim, device=device, dtype=dtype)
        diag_vals[0] = 1e-25
        opt.C = torch.diag(diag_vals)
        opt._update_eigensystem()
        # After clamping at 1e-8 and normalizing by mean:
        # pre-norm eigenvalues: [1e-8, 1, 1, 1, 1]
        # mean = (1e-8 + 4) / 5 ~ 0.8
        # normalized: [1e-8/0.8, 1/0.8, ...] ~ [1.25e-8, 1.25, ...]
        # D_diag = sqrt(normalized) ~ [1.12e-4, 1.12, ...]
        # The key check: D_diag should be well above sqrt(1e-25) = ~3.16e-13
        min_d_floor = math.sqrt(1e-8 / (4.0 + 1e-8) * dim)
        assert torch.all(opt.D_diag >= min_d_floor * 0.5), (
            f"D_diag has unexpectedly small value: min={opt.D_diag.min().item():.2e}"
        )
        # All D_diag must be positive and finite
        assert torch.all(opt.D_diag > 0), "D_diag has non-positive values"
        assert torch.isfinite(opt.D_diag).all(), "D_diag has non-finite values"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_eigenvalue_clamp_is_exactly_1e8(self, device: torch.device) -> None:
        """Eigenvalue floor must be exactly 1e-8.

        Sets one eigenvalue to 1e-12 (below 1e-8).  After
        ``_update_eigensystem``, that eigenvalue must be clamped to 1e-8.
        Guards against a looser 1e-20 floor that would let degenerate
        eigenvalues pass through.
        """
        dtype = torch.float64
        dim = 3
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        # Set C with one eigenvalue between old floor (1e-20) and new floor (1e-8)
        opt.C = torch.diag(torch.tensor([1e-12, 1.0, 1.0], device=device, dtype=dtype))
        opt._update_eigensystem()
        # After clamping at 1e-8 and normalizing: pre-norm eigs are [1e-8, 1, 1]
        # The reconstructed C's smallest eigenvalue should reflect the 1e-8 floor
        C_cpu = opt.C.to("cpu") if device.type not in ("cpu", "cuda") else opt.C
        eigvals = torch.linalg.eigvalsh(C_cpu)
        min_eig = eigvals.min().item()
        # With normalization, min_eig = 1e-8 / mean([1e-8, 1, 1]) ≈ 1e-8 / 0.667 ≈ 1.5e-8
        # If clamp were 1e-20, min_eig would be ~1e-12 / 0.667 ≈ 1.5e-12 (much smaller)
        assert min_eig > 1e-9, (
            f"Smallest eigenvalue {min_eig:.2e} suggests clamp floor is below 1e-8"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_eigenvalue_mean_is_one_after_normalization(self, device: torch.device) -> None:
        """After ``_update_eigensystem``, the mean of D_diag^2 should be ~1.0.

        Covariance normalization divides eigenvalues by their mean; if that
        step is dropped, the mean would track the raw covariance scale.
        """
        dtype = torch.float64
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        # Set anisotropic C with known eigenvalues
        opt.C = torch.diag(torch.tensor([0.1, 1.0, 2.0, 5.0, 10.0], device=device, dtype=dtype))
        opt._update_eigensystem()
        # After normalization, D_diag^2 (= normalized eigenvalues) should have mean ~1.0
        eig_mean = (opt.D_diag**2).mean().item()
        assert abs(eig_mean - 1.0) < 0.01, (
            f"Normalized eigenvalue mean={eig_mean:.4f}, expected ~1.0. "
            f"normalization may be missing."
        )


# ---------------------------------------------------------------------------
# Mirrored sampling
# ---------------------------------------------------------------------------
class TestMirroredSampling:
    """Verify that mirrored=True produces antithetic sample pairs."""

    def test_mirrored_z_pairs(self, device: torch.device) -> None:
        """Even pop_size: mirrored pairs must cancel exactly (sum == 0 up to FP eps)."""
        dtype = best_float_dtype(device)
        dim = 4
        pop_size = 10  # even — every sample has an antithetic twin
        opt = CMAES(
            dim=dim,
            bounds=100.0,  # wide bounds to avoid clamping
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            seed=42,
            mirrored=True,
        )
        # Set mean to zero so candidates = sigma * BD @ z. With identity C and
        # mean=0, each pair (z, -z) contributes exactly zero to the sum — the
        # result should be bitwise zero modulo floating-point accumulation.
        opt.mean.zero_()
        candidates = opt.ask()
        candidate_sum = candidates.sum(dim=0)
        zeros = torch.zeros(dim, device=device, dtype=dtype)
        # Tight tolerance — only floating-point addition noise is allowed.
        fp_eps = torch.finfo(dtype).eps
        atol = fp_eps * pop_size * opt.sigma * 10.0
        assert torch.allclose(candidate_sum, zeros, atol=atol), (
            f"Mirrored candidate sum not exactly zero for even pop: {candidate_sum}"
        )

    def test_mirrored_z_pairs_odd_pop(self, device: torch.device) -> None:
        """Odd pop_size: one sample is unpaired, so a small sigma-scaled residual is expected."""
        dtype = best_float_dtype(device)
        dim = 4
        pop_size = 9  # odd — one unpaired sample remains
        opt = CMAES(
            dim=dim,
            bounds=100.0,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            seed=42,
            mirrored=True,
        )
        sigma = opt.sigma
        opt.mean.zero_()
        candidates = opt.ask()
        candidate_sum = candidates.sum(dim=0)
        # With one unpaired ~N(0,1) sample, sum is bounded by ~sigma * few sigmas.
        zeros = torch.zeros(dim, device=device, dtype=dtype)
        assert torch.allclose(candidate_sum, zeros, atol=sigma * 5.0), (
            f"Odd-pop mirrored sum unexpectedly large: {candidate_sum}"
        )

    def test_mirrored_odd_pop_size(self, device: torch.device) -> None:
        """Odd pop_size should still work and return the correct shape."""
        dtype = best_float_dtype(device)
        opt = CMAES(
            dim=3,
            bounds=100.0,
            pop_size=11,
            device=device,
            dtype=dtype,
            seed=42,
            mirrored=True,
        )
        candidates = opt.ask()
        assert candidates.shape == (11, 3)


# ---------------------------------------------------------------------------
# Active CMA
# ---------------------------------------------------------------------------
class TestActiveCMA:
    """Verify that active=True still converges (does not diverge)."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_active_converges_on_sphere(self, device: torch.device) -> None:
        """Active-CMA converges on the sphere to at least 1e-8 best fitness."""
        dtype = torch.float64
        dim = 5
        opt = CMAES(
            dim=dim,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=42,
            active=True,
        )
        for _ in range(200):
            c = opt.ask()
            opt.tell(c, sphere(c))
        _, best_f = opt.best()
        assert best_f.item() < CONV_SPHERE_10D_TIGHT

    def test_active_c_stays_valid(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(
            dim=5,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=42,
            active=True,
        )
        for _ in range(30):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # CPU fallback for MPS (eigvalsh not supported on MPS)
        C_cpu = opt.C.to("cpu") if device.type not in ("cpu", "cuda") else opt.C
        eigvals = torch.linalg.eigvalsh(C_cpu)
        assert torch.all(eigvals > 0)

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_active_cma_parameter_formulas(self, device: torch.device) -> None:
        """``c_mu_neg`` and ``neg_weights`` must match Hansen's active-CMA formulas exactly."""
        dtype = torch.float64
        opt = CMAES(
            dim=10,
            bounds=5.0,
            pop_size=14,
            device=device,
            dtype=dtype,
            seed=42,
            active=True,
        )
        # c_mu_neg = min(0.25 * c_mu, max(0, 1 - c_1 - c_mu))
        expected = min(0.25 * opt.c_mu, max(0.0, 1 - opt.c_1 - opt.c_mu))
        assert opt._c_mu_neg == pytest.approx(expected, abs=ATOL_F64_TIGHT), (
            f"c_mu_neg={opt._c_mu_neg}, expected={expected}"
        )
        # neg_weights should sum to 1.0 and be positive descending
        assert opt._neg_weights.sum().item() == pytest.approx(1.0, abs=ATOL_F64_DEFAULT)
        assert torch.all(opt._neg_weights > 0)
        diffs = opt._neg_weights[1:] - opt._neg_weights[:-1]
        assert torch.all(diffs <= 0), "neg_weights should be non-increasing"


# ---------------------------------------------------------------------------
# restart() / IPOP
# ---------------------------------------------------------------------------
class TestRestart:
    """Verify restart() resets state correctly and supports IPOP doubling."""

    def test_paths_zeroed(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        # Run a few generations to build up paths
        for _ in range(10):
            c = opt.ask()
            opt.tell(c, sphere(c))
        assert torch.any(opt.p_sigma != 0)

        opt.restart()
        zeros = torch.zeros(5, device=device, dtype=dtype)
        assert torch.equal(opt.p_sigma, zeros)
        assert torch.equal(opt.p_c, zeros)

    def test_pop_size_doubles(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        opt.restart(new_pop_size=20)
        assert opt.pop_size == 20
        assert opt.mu == 10
        assert opt.population.shape == (20, 5)
        assert opt.fitness.shape == (20,)
        # Weights should sum to 1 with the new count
        assert torch.allclose(
            opt.weights.sum(),
            torch.tensor(1.0, dtype=dtype, device=device),
        )

    def test_mean_reset_to_center(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=3, bounds=(-2.0, 4.0), device=device, dtype=dtype, seed=42)
        # Run to move mean away from center
        for _ in range(5):
            c = opt.ask()
            opt.tell(c, sphere(c))
        opt.restart()
        expected = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        assert torch.allclose(opt.mean, expected)

    def test_custom_mean_and_sigma(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)
        new_mean = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        opt.restart(mean=new_mean, sigma=0.5)
        assert torch.allclose(opt.mean, new_mean.to(device=device, dtype=dtype))
        assert opt.sigma == 0.5

    def test_generation_reset(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        for _ in range(5):
            c = opt.ask()
            opt.tell(c, sphere(c))
        assert opt._generation > 0
        opt.restart()
        assert opt._generation == 0

    def test_covariance_reset_to_identity(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        for _ in range(20):
            c = opt.ask()
            opt.tell(c, sphere(c))
        opt.restart()
        eye = torch.eye(dim, device=device, dtype=dtype)
        assert torch.allclose(opt.C, eye)

    def test_ask_works_after_restart(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        for _ in range(5):
            c = opt.ask()
            opt.tell(c, sphere(c))
        opt.restart(new_pop_size=20)
        candidates = opt.ask()
        assert candidates.shape == (20, 5)
        fitness = sphere(candidates)
        opt.tell(candidates, fitness)  # should not raise

    def test_restart_with_custom_c_init(self, device: torch.device) -> None:
        """Restarting with an anisotropic C_init preserves its shape after the normalization step."""
        dtype = best_float_dtype(device)
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        for _ in range(5):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # Restart with anisotropic C_init (different eigenvalues)
        c_init = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype))
        opt.restart(C_init=c_init)
        eye = torch.eye(dim, device=device, dtype=dtype)
        # C should NOT be identity (anisotropic shape is preserved by normalization)
        assert not torch.allclose(opt.C, eye), (
            "C should differ from identity after restart with anisotropic C_init"
        )
        # Verify ask/tell loop works after restart with custom C_init
        for _ in range(5):
            candidates = opt.ask()
            fitness = sphere(candidates)
            opt.tell(candidates, fitness)
        _, fit = opt.best()
        assert fit.item() < SMOKE_F_INIT_CMAES, (
            "Optimization should improve from random init after restart with C_init"
        )


# ---------------------------------------------------------------------------
# Sigma clamping
# ---------------------------------------------------------------------------
class TestSigmaClamping:
    """Verify sigma stays within [sigma_min, sigma_max] after tell()."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sigma_stays_within_bounds(self, device: torch.device) -> None:
        """After many tell() calls, sigma should never exceed sigma_max or
        drop below sigma_min when explicit bounds are set.
        """
        dtype = torch.float64
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        span = (opt.ub - opt.lb).mean().item()
        # Set explicit PhasedDFO-style bounds
        opt.sigma_min = 0.01 * span
        opt.sigma_max = 0.10 * span
        for _ in range(200):
            c = opt.ask()
            opt.tell(c, sphere(c))
            assert opt.sigma >= opt.sigma_min, (
                f"sigma={opt.sigma:.2e} < sigma_min={opt.sigma_min:.2e}"
            )
            assert opt.sigma <= opt.sigma_max, (
                f"sigma={opt.sigma:.2e} > sigma_max={opt.sigma_max:.2e}"
            )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sigma_max_widens_after_restart(self, device: torch.device) -> None:
        """After restart(), sigma_max should be 0.30 * span (wider for restart phases)."""
        dtype = torch.float64
        opt = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        span = (opt.ub - opt.lb).mean().item()
        # Set explicit initial-phase bounds
        opt.sigma_min = 0.01 * span
        opt.sigma_max = 0.10 * span
        initial_sigma_max = opt.sigma_max
        opt.restart()
        # restart() widens sigma_max to 0.30 * span
        assert opt.sigma_max == pytest.approx(0.30 * span)
        assert opt.sigma_max > initial_sigma_max

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sigma_defaults_are_noop(self, device: torch.device) -> None:
        """Default sigma bounds should be a no-op range (no clamping)."""
        dtype = torch.float64
        opt = CMAES(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        # sigma_min default is exactly 1e-30 (effective no-op lower bound).
        assert opt.sigma_min == pytest.approx(1e-30)
        assert opt.sigma_max == float("inf")


# ---------------------------------------------------------------------------
# Convergence tests (CPU float64 only)
# ---------------------------------------------------------------------------
class TestConvergence:
    """Convergence tests run only on CPU with float64 for determinism."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sphere_10d(self, device: torch.device, seed: int) -> None:
        """CMA-ES should solve sphere 10d to < 1e-10 within 2000 generations."""
        dim = 10
        opt = CMAES(
            dim=dim,
            bounds=5.0,
            device=device,
            dtype=torch.float64,
            seed=seed,
        )
        for _ in range(2000):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            _, best_f = opt.best()
            if best_f.item() < 1e-10:
                break
        _, best_f = opt.best()
        assert best_f.item() < CONV_SPHERE_HIGH_DIM, (
            f"Did not converge: best_f = {best_f.item():.2e}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_rastrigin_10d(self, device: torch.device, seed: int) -> None:
        """CMA-ES with IPOP restarts should reach < 2.5 on rastrigin 10d."""
        dim = 10
        best_f_global = float("inf")
        pop_size = 4 + math.floor(3 * math.log(dim))
        opt = CMAES(
            dim=dim,
            bounds=5.12,
            pop_size=pop_size,
            device=device,
            dtype=torch.float64,
            seed=seed,
        )
        # IPOP: run multiple restarts with increasing pop_size
        for _restart in range(5):
            for _ in range(1000):
                c = opt.ask()
                f = rastrigin(c)
                opt.tell(c, f)
                _, bf = opt.best()
                best_f_global = min(best_f_global, bf.item())
                if best_f_global < 1.0:
                    break
            if best_f_global < 1.0:
                break
            pop_size *= 2
            opt.restart(new_pop_size=pop_size)
        assert best_f_global < CONV_CMAES_RASTRIGIN_10D, (
            f"Did not converge: best_f = {best_f_global:.2e}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_dim_1_edge_case(self, device: torch.device) -> None:
        """CMA-ES with dim=1 should converge on the 1-D sphere (edge case)."""
        dtype = torch.float64
        opt = CMAES(dim=1, bounds=5.0, device=device, dtype=dtype, seed=42)
        for _ in range(100):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            _, best_f = opt.best()
            if best_f.item() < 1e-6:
                break
        _, best_f = opt.best()
        assert best_f.item() < CONV_SPHERE_1D, (
            f"CMA-ES dim=1 did not converge: best_f = {best_f.item():.2e}"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """Same seed produces identical trajectories."""

    def test_same_seed_same_trajectory(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        kwargs = {
            "dim": 5,
            "bounds": 5.0,
            "pop_size": 10,
            "device": device,
            "dtype": dtype,
            "seed": 123,
        }
        opt1 = CMAES(**kwargs)
        opt2 = CMAES(**kwargs)

        for _ in range(10):
            c1 = opt1.ask()
            c2 = opt2.ask()
            assert torch.equal(c1, c2)
            f1 = sphere(c1)
            f2 = sphere(c2)
            opt1.tell(c1, f1)
            opt2.tell(c2, f2)

        assert torch.equal(opt1.mean, opt2.mean)
        assert opt1.sigma == pytest.approx(opt2.sigma)
        assert torch.equal(opt1.C, opt2.C)

    def test_different_seeds_differ(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=1)
        opt2 = CMAES(dim=5, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=2)
        c1 = opt1.ask()
        c2 = opt2.ask()
        assert not torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# Limited-memory path sampling
# ---------------------------------------------------------------------------
class TestPathMemorySampling:
    """Validate optional LM-CMA-style path memory."""

    def test_path_memory_records_normalized_paths(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(
            dim=6,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=42,
            path_memory=3,
            path_scale=0.5,
        )

        candidates = opt.ask()
        opt.tell(candidates, sphere(candidates))

        assert opt._path_count == 1
        first_norm = torch.linalg.vector_norm(opt._path_vectors[0])
        assert first_norm.item() == pytest.approx(1.0, rel=RTOL_DEFAULT, abs=ATOL_F32_DEFAULT)

    def test_path_memory_restart_clears_history(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(
            dim=6,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=42,
            path_memory=3,
            path_scale=0.5,
        )

        candidates = opt.ask()
        opt.tell(candidates, sphere(candidates))
        assert opt._path_count > 0

        opt.restart(mean=torch.zeros(6, device=device, dtype=dtype), sigma=0.5)

        assert opt._path_count == 0
        assert torch.all(opt._path_vectors == 0)

    def test_path_memory_roundtrip_preserves_next_ask(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = CMAES(
            dim=6,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=42,
            path_memory=3,
            path_scale=0.5,
        )
        candidates = opt1.ask()
        opt1.tell(candidates, sphere(candidates))

        state = opt1.state_dict()
        opt2 = CMAES(
            dim=6,
            bounds=5.0,
            pop_size=12,
            device=device,
            dtype=dtype,
            seed=99,
        )
        opt2.load_state_dict(state)

        assert opt2.path_memory == opt1.path_memory
        assert opt2._path_count == opt1._path_count
        assert torch.allclose(opt1.ask(), opt2.ask())

    def test_path_line_samples_follow_covariance_path(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(
            dim=4,
            bounds=100.0,
            pop_size=6,
            device=device,
            dtype=dtype,
            seed=42,
            sigma0=0.01,
            path_line_samples=2,
            path_line_scale=0.5,
        )
        opt.mean.zero_()
        opt.sigma = 1.0
        opt.p_c = torch.tensor([3.0, 4.0, 0.0, 0.0], device=device, dtype=dtype)

        candidates = opt.ask()

        expected = torch.tensor([0.6, 0.8, 0.0, 0.0], device=device, dtype=dtype)
        assert torch.allclose(candidates[0], expected)
        assert torch.allclose(candidates[1], -expected)

        restored = CMAES(dim=4, bounds=100.0, pop_size=6, device=device, dtype=dtype)
        restored.load_state_dict(opt.state_dict())
        assert restored.path_line_samples == opt.path_line_samples
        assert restored.path_line_scale == pytest.approx(opt.path_line_scale)

    def test_zero_path_line_samples_preserve_rng_path(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        base_kwargs = {
            "dim": 4,
            "bounds": 100.0,
            "pop_size": 6,
            "device": device,
            "dtype": dtype,
            "seed": 42,
            "sigma0": 0.01,
        }
        opt1 = CMAES(**base_kwargs)
        opt2 = CMAES(**base_kwargs, path_line_samples=2, path_line_scale=1.0)

        assert torch.allclose(opt1.ask(), opt2.ask())


# ---------------------------------------------------------------------------
# Multi-device (parametrized via conftest device fixture)
# ---------------------------------------------------------------------------
class TestMultiDevice:
    """Ensure the full ask/tell loop works on all available devices."""

    def test_ask_tell_loop(self, device: torch.device) -> None:
        """Ask/tell loop runs cleanly on all devices and reaches a meaningful best fitness."""
        dtype = best_float_dtype(device)
        opt = CMAES(dim=4, bounds=5.0, pop_size=8, device=device, dtype=dtype, seed=42)
        for _ in range(5):
            c = opt.ask()
            assert c.device.type == device.type
            f = sphere(c)
            opt.tell(c, f)
        sol, fit = opt.best()
        assert sol.device.type == device.type
        assert fit.device.type == device.type
        assert fit.item() < SMOKE_F_INIT_CMAES

    def test_best_tracking(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = CMAES(dim=3, bounds=5.0, pop_size=10, device=device, dtype=dtype, seed=42)
        for _ in range(20):
            c = opt.ask()
            opt.tell(c, sphere(c))
        sol, fit = opt.best()
        assert fit.item() < SMOKE_F_INIT_CMAES
        # best_solution should actually achieve approximately best_fitness
        actual_f = sphere(sol.unsqueeze(0)).item()
        tol = TOL_CMAES_BEST_F32 if dtype == torch.float32 else TOL_CMAES_BEST_F64
        assert actual_f == pytest.approx(fit.item(), abs=tol)


# ---------------------------------------------------------------------------
# h_sigma Heaviside (indirect test)
# ---------------------------------------------------------------------------
class TestHSigma:
    """Verify that h_sigma allows p_c updates during normal optimization."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_h_sigma_one_on_first_tell(self, device: torch.device) -> None:
        """H3: h_sigma == 1 after a single tell with a controlled pre-state.

        Construction: start from fresh init (p_sigma = 0). The h_sigma formula
        in ``cmaes.py::tell`` computes::

            lhs = ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))
            rhs = (1.4 + 2/(dim+1)) * chi_n
            h_sigma = 1.0 if lhs < rhs else 0.0

        After one tell on sphere, p_sigma moves from 0 by a bounded amount; the
        lhs stays well under the rhs, so h_sigma MUST be 1. We verify by
        reproducing the formula directly (so this test is tied to the
        condition, not to the side effect of p_c being nonzero).
        """
        import math as _math
        dtype = torch.float64
        dim = 5
        opt = CMAES(dim=dim, bounds=5.0, pop_size=12, device=device, dtype=dtype, seed=42)
        # Run one controlled tell.
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Compute the exact lhs/rhs using the same formula as cmaes.py line 374.
        gen = opt._generation  # now 1
        ps_norm = torch.linalg.norm(opt.p_sigma).item()
        lhs = ps_norm / _math.sqrt(1 - (1 - opt.c_sigma) ** (2 * gen))
        rhs = (1.4 + 2.0 / (dim + 1)) * opt.chi_n
        h_sigma = 1.0 if lhs < rhs else 0.0
        assert h_sigma == 1.0, (
            f"h_sigma predicted 0 after single tell: lhs={lhs:.4f}, rhs={rhs:.4f}"
        )
        # Cross-check: under h_sigma=1 and mu_eff > 0 with a non-trivial mean
        # step, p_c must have a non-zero entry.
        assert torch.any(opt.p_c != 0), (
            "p_c is all zero after a tell where h_sigma == 1 — formula wiring broke"
        )


# ---------------------------------------------------------------------------
# Zero-span bounds (W2): must fail cleanly at construction, not deep in eigh
# ---------------------------------------------------------------------------
class TestZeroSpanBounds:
    """Zero-span bounds must raise ValueError via ``normalize_bounds``."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_zero_span_scalar_bounds_raises_value_error(
        self, device: torch.device
    ) -> None:
        """``CMAES(bounds=0)`` rejects at construction."""
        with pytest.raises(ValueError, match="positive span"):
            CMAES(dim=3, bounds=0, pop_size=10, device=device,
                  dtype=torch.float64, seed=42)

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_zero_span_tuple_bounds_raises_value_error(
        self, device: torch.device
    ) -> None:
        """``CMAES(bounds=(v, v))`` rejects at construction."""
        with pytest.raises(ValueError, match="positive span"):
            CMAES(dim=3, bounds=(5.0, 5.0), pop_size=10, device=device,
                  dtype=torch.float64, seed=42)
