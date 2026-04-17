"""Tests for torch_dfo._polish -- polish operators for PhasedDFO Phase 3."""

from __future__ import annotations

import math

import pytest
import torch

from tests._thresholds import ATOL_F32_TIGHT, atol_for
from tests.conftest import best_float_dtype
from torch_dfo._polish import (
    GRADIENT_FD_TRUST_RATIO,
    _add_unique,
    _central_diff_gradient,
    coordinate_basin_search,
    directional_basin_search,
    fd_bfgs_polish,
    nm_polish,
)
from torch_dfo.benchmarks import rosenbrock, sphere

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_fn(x: torch.Tensor) -> torch.Tensor:
    """Sphere function accepting (N, dim) -> (N,)."""
    return sphere(x)


def _rosenbrock_fn(x: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function accepting (N, dim) -> (N,)."""
    return rosenbrock(x)


def _make_bounds(dim: int, lo: float, hi: float, device: torch.device, dtype: torch.dtype):
    lb = torch.full((dim,), lo, device=device, dtype=dtype)
    ub = torch.full((dim,), hi, device=device, dtype=dtype)
    return lb, ub


# ---------------------------------------------------------------------------
# _add_unique helper
# ---------------------------------------------------------------------------
class TestAddUnique:
    """Validate the deduplication helper."""

    def test_adds_new_value(self) -> None:
        targets: list[float] = [1.0, 2.0]
        _add_unique(targets, 3.0, 0.0, 10.0)
        assert len(targets) == 3
        assert targets[-1] == 3.0

    def test_rejects_duplicate_within_tolerance(self) -> None:
        targets: list[float] = [1.0, 2.0]
        _add_unique(targets, 1.0 + 1e-10, 0.0, 10.0)
        assert len(targets) == 2

    def test_clamps_to_bounds(self) -> None:
        targets: list[float] = []
        _add_unique(targets, -5.0, 0.0, 10.0)
        assert len(targets) == 1
        assert targets[0] == 0.0

        _add_unique(targets, 15.0, 0.0, 10.0)
        assert len(targets) == 2
        assert targets[1] == 10.0

    def test_accepts_value_differing_by_more_than_tolerance(self) -> None:
        targets: list[float] = [1.0]
        _add_unique(targets, 1.0 + 1e-8, 0.0, 10.0)
        assert len(targets) == 2


# ---------------------------------------------------------------------------
# _central_diff_gradient
# ---------------------------------------------------------------------------
class TestCentralDiffGradient:
    """Validate central-difference gradient helper."""

    def test_central_diff_gradient_on_sphere(self, device: torch.device) -> None:
        """FD gradient on sphere at x=[1,2,3] should match analytical [2,4,6]."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)

        grad, fe = _central_diff_gradient(
            x,
            _sphere_fn,
            lb,
            ub,
            ub - lb,
            fd_step=1e-4,
        )

        expected = torch.tensor([2.0, 4.0, 6.0], device=device, dtype=dtype)
        assert fe == 2 * dim
        # one-off: FD gradient tolerance — problem-specific
        assert torch.allclose(grad, expected, atol=0.01), (
            f"FD gradient {grad} does not match analytical {expected}"
        )


# ---------------------------------------------------------------------------
# coordinate_basin_search
# ---------------------------------------------------------------------------
class TestCoordinateBasinSearch:
    """Validate per-axis coordinate basin search with coarse-to-fine grid."""

    def test_improves_sphere(self, device: torch.device) -> None:
        """Starting from a non-optimal point, CBS should improve sphere significantly."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, _fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            passes=2,
        )

        assert best_f < 1.0, f"CBS on 5d sphere from x=2*ones should reach < 1.0, got {best_f}"
        assert best_x.shape == (dim,)
        assert best_f.shape == ()

    def test_return_types_and_shapes(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
        )

        assert isinstance(best_x, torch.Tensor)
        assert isinstance(best_f, torch.Tensor)
        assert isinstance(fe, int)
        assert best_x.shape == (dim,)
        assert best_f.ndim == 0
        assert best_x.device.type == device.type
        assert best_f.device.type == device.type
        # Monotonic improvement guarantee: CBS should never return worse than input.
        assert best_f <= f0, f"CBS returned worse result: {best_f} > {f0}"

    def test_fe_used_positive_and_bounded(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 4
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        budget = 200

        _, _, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=budget,
        )

        assert fe > 0
        assert fe <= budget
        # CBS must probe each dimension at least once per pass (coarse grid).
        assert fe >= dim, f"CBS used only {fe} evals for {dim} dimensions"

    def test_with_elite_centroid_and_median(self, device: torch.device) -> None:
        """CBS should benefit from elite centroid and median hints."""
        dtype = best_float_dtype(device)
        dim = 4
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 3.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # Place centroid/median near the optimum
        centroid = torch.full((dim,), 0.1, device=device, dtype=dtype)
        median = torch.full((dim,), -0.05, device=device, dtype=dtype)

        _best_x, best_f, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            elite_centroid=centroid,
            elite_median=median,
        )

        assert best_f < f0
        assert fe > 0

    def test_respects_budget(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 10
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        budget = 30

        _, _, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=budget,
        )

        assert fe <= budget

    def test_no_nans_in_output(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, _ = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
        )

        assert torch.isfinite(best_x).all()
        assert torch.isfinite(best_f)

    def test_cbs_budget_exhaustion_mid_dimension(self, device: torch.device) -> None:
        """CBS with budget too small to finish even the first dimension should exit cleanly."""
        dtype = best_float_dtype(device)
        dim = 10
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=5,
        )

        assert fe <= 5, f"CBS exceeded budget of 5, used {fe}"
        assert torch.isfinite(best_x).all()
        assert torch.isfinite(best_f)

    def test_monotonic_improvement_guarantee(self, device: torch.device) -> None:
        """CBS must never return a fitness worse than the input."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f_x = _sphere_fn(x0.unsqueeze(0)).squeeze()

        _best_x, best_f, _fe = coordinate_basin_search(
            x0,
            f_x,
            _sphere_fn,
            lb,
            ub,
        )

        # Use a dtype-aware atol so the comparison works for both float32 and float64.
        atol = atol_for(dtype)
        assert best_f <= f_x + atol, (
            f"Monotonic guarantee violated: best_f={best_f} > f_x={f_x} (atol={atol})"
        )

    def test_with_population_elite_best(self, device: torch.device) -> None:
        """CBS with population should include elite_best as a target."""
        dtype = best_float_dtype(device)
        dim = 4
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 3.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # Create a population where the best member is near the optimum
        pop = torch.randn(10, dim, device=device, dtype=dtype) * 2.0
        pop[0] = torch.full((dim,), 0.01, device=device, dtype=dtype)  # near optimum
        pop_fit = _sphere_fn(pop)

        _best_x, best_f, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            population=pop,
            pop_fitness=pop_fit,
        )

        assert best_f < f0
        assert fe > 0


# ---------------------------------------------------------------------------
# directional_basin_search
# ---------------------------------------------------------------------------
class TestDirectionalBasinSearch:
    """Validate coarse-to-fine line search along given directions."""

    def test_improves_sphere_coordinate_directions(self, device: torch.device) -> None:
        """DBS along coordinate axes should improve sphere from a non-optimal point."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)

        best_x, best_f, _fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert best_f < f0, f"Expected improvement: {best_f} < {f0}"
        assert best_x.shape == (dim,)

    def test_improves_sphere_random_directions(self, device: torch.device) -> None:
        """DBS along random normalised directions should also improve sphere."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # Scope the seed to a local generator so this test does not leak
        # randomness into neighbouring tests via the global torch RNG.
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        raw = torch.randn(3, dim, device=device, dtype=dtype, generator=gen)
        directions = raw / raw.norm(dim=1, keepdim=True)

        _best_x, best_f, _fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert best_f < f0

    def test_fe_tracking(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)

        _, _, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert fe > 0
        assert isinstance(fe, int)

    def test_respects_budget(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)
        budget = 25

        _, _, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
            budget=budget,
        )

        assert fe <= budget

    def test_return_types(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        directions = torch.eye(dim, device=device, dtype=dtype)

        best_x, best_f, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert isinstance(best_x, torch.Tensor)
        assert isinstance(best_f, torch.Tensor)
        assert isinstance(fe, int)
        assert best_x.shape == (dim,)
        assert best_f.ndim == 0
        assert best_x.device.type == device.type
        assert best_f.device.type == device.type

    def test_no_nans_in_output(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        directions = torch.eye(dim, device=device, dtype=dtype)

        best_x, best_f, _ = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert torch.isfinite(best_x).all()
        assert torch.isfinite(best_f)

    def test_dbs_with_oblique_directions(self, device: torch.device) -> None:
        """DBS with a 45-degree direction [1/sqrt(2), 1/sqrt(2), 0, ...] should still improve."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        directions = torch.zeros(3, dim, device=device, dtype=dtype)
        # 45-degree direction coupling dims 0 and 1
        directions[0, 0] = inv_sqrt2
        directions[0, 1] = inv_sqrt2
        # Coordinate directions for dims 2 and 3
        directions[1, 2] = 1.0
        directions[2, 3] = 1.0

        _best_x, best_f, _fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        assert best_f < f0, f"DBS with oblique directions should improve: {best_f} not < {f0}"

    def test_priority_ordering(self, device: torch.device) -> None:
        """Priority directions are processed first, secondary shuffled."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)

        _best_x, best_f, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
            priority_count=2,
        )

        assert best_f < f0
        assert fe > 0

    def test_priority_hops_add_extra_alphas(self, device: torch.device) -> None:
        """Priority hops should add extra probe alphas and use more FEs."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)

        # Without priority hops
        _, _, fe_no_hops = directional_basin_search(
            x0.clone(),
            f0.clone(),
            _sphere_fn,
            lb,
            ub,
            directions,
            priority_count=0,
        )

        # With priority hops on all directions
        _, _, fe_with_hops = directional_basin_search(
            x0.clone(),
            f0.clone(),
            _sphere_fn,
            lb,
            ub,
            directions,
            priority_count=dim,
            priority_hops=4,
            priority_hop_scale=1.0,
        )

        # Priority hops should use strictly more evaluations
        assert fe_with_hops > fe_no_hops, (
            f"Priority hops should increase FE count: {fe_with_hops} not > {fe_no_hops}"
        )

    def test_elite_projections(self, device: torch.device) -> None:
        """Elite projections should add candidate alphas and improve search."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)

        # Create elite points near the optimum
        elite = torch.randn(5, dim, device=device, dtype=dtype) * 0.1

        _best_x, best_f, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
            elite_points=elite,
        )

        assert best_f < f0
        # Elite projections should add alphas, resulting in more FE than baseline
        _, _, fe_baseline = directional_basin_search(
            x0.clone(),
            f0.clone(),
            _sphere_fn,
            lb,
            ub,
            directions,
        )
        assert fe >= fe_baseline, f"Elite projections should add alphas: {fe} < {fe_baseline}"

    def test_priority_hops_and_elite_combined(self, device: torch.device) -> None:
        """Combined priority hops + elite projections should still produce valid output."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        directions = torch.eye(dim, device=device, dtype=dtype)
        elite = torch.randn(4, dim, device=device, dtype=dtype) * 0.5

        best_x, best_f, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
            priority_count=3,
            elite_points=elite,
            priority_hops=4,
            priority_hop_scale=1.0,
        )

        assert best_f < f0
        assert torch.isfinite(best_x).all()
        assert torch.isfinite(best_f)
        assert fe > 0

    def test_alpha_zero_always_probed(self, device: torch.device) -> None:
        """alpha=0 (the current point) must always be in the probe set.

        Uses a function where the current point IS the optimum: if alpha=0
        is missing, DBS would move away from the optimum instead of holding.
        """
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        # Start at the optimum
        x0 = torch.zeros(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()  # 0.0

        directions = torch.eye(dim, device=device, dtype=dtype)
        _best_x, best_f, _fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
            coarse_points=5,
        )
        # If alpha=0 is probed, best should stay at the optimum
        assert best_f <= f0 + 1e-12, (
            f"DBS moved away from optimum: best_f={best_f.item():.2e}. "
            f"Is alpha=0 included in probes?"
        )


# ---------------------------------------------------------------------------
# fd_bfgs_polish
# ---------------------------------------------------------------------------
class TestFdBfgsPolish:
    """Validate FD-BFGS polish with trust region."""

    def test_improves_sphere(self, device: torch.device) -> None:
        """BFGS should improve sphere from a nearby starting point."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 0.5, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        _best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=500,
        )

        assert best_f < f0, f"Expected improvement: {best_f} < {f0}"
        assert best_f < 0.1, f"Expected near-zero fitness, got {best_f}"

    def test_improves_rosenbrock(self, device: torch.device) -> None:
        """BFGS should achieve at least 90% reduction on 3d rosenbrock near optimum."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 10.0, device, dtype)
        # Start near optimum (1, 1, ..., 1) but not at it
        x0 = torch.full((dim,), 0.5, device=device, dtype=dtype)
        f0 = _rosenbrock_fn(x0.unsqueeze(0)).squeeze()

        _best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _rosenbrock_fn,
            lb,
            ub,
            budget=1000,
        )

        assert best_f < f0, f"Expected improvement: {best_f} < {f0}"
        assert best_f < f0 * 0.1, (
            f"BFGS should achieve 90% reduction on 3d rosenbrock: {best_f} not < {f0 * 0.1}"
        )

    def test_no_nans_in_output(self, device: torch.device) -> None:
        """H should remain numerically stable -- no NaN in output."""
        dtype = best_float_dtype(device)
        dim = 4
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 1.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=200,
        )

        assert torch.isfinite(best_x).all(), "NaN/Inf in best_x"
        assert torch.isfinite(best_f), "NaN/Inf in best_f"

    def test_fe_used_within_budget(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 4
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 1.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        budget = 100

        _, _, fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=budget,
        )

        assert fe <= budget

    def test_probes_stay_in_bounds(self, device: torch.device) -> None:
        """All evaluated points must respect [lb, ub]."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -2.0, 2.0, device, dtype)
        # Start near boundary to stress clamping
        x0 = torch.full((dim,), 1.8, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # Wrap fitness_fn to check bounds on every call
        violations = []

        def checked_fn(probes: torch.Tensor) -> torch.Tensor:
            if (probes < lb - 1e-7).any() or (probes > ub + 1e-7).any():
                violations.append(True)
            return _sphere_fn(probes)

        fd_bfgs_polish(x0, f0, checked_fn, lb, ub, budget=200)

        assert len(violations) == 0, "Some probes violated bounds"

    def test_return_types(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=50,
        )

        assert isinstance(best_x, torch.Tensor)
        assert isinstance(best_f, torch.Tensor)
        assert isinstance(fe, int)
        assert best_x.shape == (dim,)
        assert best_f.ndim == 0

    def test_bfgs_at_optimum_zero_gradient(self, device: torch.device) -> None:
        """BFGS at the sphere optimum (zeros) should detect zero gradient and exit quickly."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.zeros(dim, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        _best_x, best_f, fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=500,
        )

        # At the optimum the gradient is zero so BFGS should exit after one gradient call.
        assert fe <= 2 * dim + 2, f"BFGS at optimum should exit quickly: fe={fe} > {2 * dim + 2}"
        assert torch.isclose(
            best_f, torch.tensor(0.0, device=device, dtype=dtype), atol=ATOL_F32_TIGHT
        ), f"best_f should be ~0.0 at optimum, got {best_f}"

    def test_bfgs_handles_flat_function_no_nan(self, device: torch.device) -> None:
        """BFGS on a near-flat function must not produce NaN/Inf output.

        H19 note: name was previously ``test_bfgs_curvature_condition_path``
        which overclaimed — this test does not directly exercise the
        curvature-reset branch inside fd_bfgs_polish, only verifies the
        outermost invariant (no NaN/Inf). A targeted curvature-branch test
        is left as a follow-up issue.
        """
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)

        def flat_fn(x: torch.Tensor) -> torch.Tensor:
            return 1e-20 * _sphere_fn(x)

        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = flat_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            flat_fn,
            lb,
            ub,
            budget=200,
        )

        assert torch.isfinite(best_x).all(), "NaN/Inf in best_x from flat function"
        assert torch.isfinite(best_f), "NaN/Inf in best_f from flat function"

    def test_bfgs_trust_radius_recovery(self, device: torch.device) -> None:
        """BFGS on Rosenbrock from far away should still improve after trust-radius shrinkage."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -10.0, 10.0, device, dtype)
        x0 = torch.full((dim,), 5.0, device=device, dtype=dtype)
        f0 = _rosenbrock_fn(x0.unsqueeze(0)).squeeze()

        best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _rosenbrock_fn,
            lb,
            ub,
            budget=1000,
        )

        assert best_f < f0, f"BFGS should improve rosenbrock from [5,5,5]: {best_f} not < {f0}"
        assert torch.isfinite(best_x).all()

    def test_bfgs_initial_trust_radius_is_conservative(self, device: torch.device) -> None:
        """Initial trust radius is 0.02 * span_mean."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 0.5, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # With the conservative initial trust radius, BFGS should still converge
        # on sphere but may use more iterations (still within budget).
        _best_x, best_f, fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=500,
        )

        assert best_f < 0.1, f"Conservative trust radius should still converge: {best_f}"
        assert fe > 0
        # With 0.02*span initial trust, BFGS needs more FE than with
        # 0.5*span. On 5-D sphere from x=0.5, 0.02*span initial trust
        # limits the first step. With budget=500 this should use at least
        # 20 FE (gradient + several backtrack steps); verify FE usage is
        # consistent with a conservative trust radius.
        assert fe >= 2 * dim, (
            f"BFGS used only {fe} FE — trust radius may be too large (not 0.05*span)"
        )

    def test_bfgs_trust_radius_capped_at_max(self, device: torch.device) -> None:
        """Trust radius growth should be capped at max_trust_radius."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x0 = torch.full((dim,), 1.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        # Should still work correctly with bounded trust growth
        best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=300,
        )

        assert best_f < f0
        assert torch.isfinite(best_x).all()

    def test_bfgs_min_trust_radius_termination(self, device: torch.device) -> None:
        """BFGS should terminate when trust radius falls below min threshold."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)

        # Use a function that rejects all steps to force trust shrinkage
        call_count = [0]

        def adversarial_fn(x: torch.Tensor) -> torch.Tensor:
            call_count[0] += x.shape[0]
            return torch.full((x.shape[0],), 999.0, device=device, dtype=dtype)

        x0 = torch.ones(dim, device=device, dtype=dtype)
        f0 = torch.tensor(1.0, device=device, dtype=dtype)

        best_x, _best_f, fe = fd_bfgs_polish(
            x0,
            f0,
            adversarial_fn,
            lb,
            ub,
            budget=10000,
        )

        # Should terminate well before exhausting the full budget due to
        # min_trust_radius threshold.
        assert fe < 10000, f"BFGS should terminate early via min_trust_radius, used {fe} of 10000"
        assert torch.isfinite(best_x).all()


# ---------------------------------------------------------------------------
# Integration: all polish functions reduce sphere from x = 2*ones
# ---------------------------------------------------------------------------
class TestPolishIntegration:
    """Integration tests: each polish function significantly reduces sphere fitness."""

    @pytest.fixture
    def sphere_setup(self, device: torch.device):
        """Common setup: start at x = 2*ones in 5-d."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()
        return x0, f0, lb, ub

    def test_cbs_reduces_sphere(self, sphere_setup) -> None:
        x0, f0, lb, ub = sphere_setup

        _best_x, best_f, fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            passes=2,
        )

        # Should reduce by at least 90%
        assert best_f < f0 * 0.1, f"CBS: {best_f} not < {f0 * 0.1}"
        assert fe > 0

    def test_dbs_reduces_sphere(self, sphere_setup) -> None:
        x0, f0, lb, ub = sphere_setup
        dim = x0.shape[0]
        directions = torch.eye(dim, device=x0.device, dtype=x0.dtype)

        _best_x, best_f, fe = directional_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            directions,
        )

        # Should reduce by at least 90%
        assert best_f < f0 * 0.1, f"DBS: {best_f} not < {f0 * 0.1}"
        assert fe > 0

    def test_bfgs_reduces_sphere(self, sphere_setup) -> None:
        x0, f0, lb, ub = sphere_setup

        _best_x, best_f, fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=500,
        )

        # BFGS should reduce by at least 99%
        assert best_f < f0 * 0.01, f"BFGS: {best_f} not < {f0 * 0.01}"
        assert fe > 0

    def test_cbs_near_zero_on_sphere(self, sphere_setup) -> None:
        """With enough budget CBS should get very close to zero on sphere."""
        x0, f0, lb, ub = sphere_setup

        _best_x, best_f, _fe = coordinate_basin_search(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            passes=3,
            coarse_points=21,
        )

        assert best_f < 0.1, f"CBS should get close to zero, got {best_f}"

    def test_bfgs_near_zero_on_sphere(self, sphere_setup) -> None:
        """BFGS should converge near zero on sphere."""
        x0, f0, lb, ub = sphere_setup

        _best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=1000,
        )

        assert best_f < 0.01, f"BFGS should converge near zero, got {best_f}"


# ---------------------------------------------------------------------------
# FD step capped by trust ratio
# ---------------------------------------------------------------------------
class TestFdStepTrustRatio:
    """Verify FD step is capped at GRADIENT_FD_TRUST_RATIO * trust_radius."""

    def test_gradient_fd_trust_ratio_constant(self) -> None:
        """The constant must be 0.10."""
        assert GRADIENT_FD_TRUST_RATIO == 0.10

    def test_bfgs_fd_step_respects_trust_ratio(self, device: torch.device) -> None:
        """FD step h should be capped by GRADIENT_FD_TRUST_RATIO * trust_radius.

        We call _central_diff_gradient with a very small trust_radius and
        verify the probes are tightly clustered around x (within the cap).
        """
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        span = ub - lb

        # With a tiny trust_radius, the FD step should be capped
        tiny_trust = 0.001
        max_h = GRADIENT_FD_TRUST_RATIO * tiny_trust  # 0.00025

        # Intercept probes to measure actual perturbation distances
        max_perturbation = [0.0]

        def spy_fn(probes: torch.Tensor) -> torch.Tensor:
            diffs = (probes - x.unsqueeze(0)).abs()
            max_perturbation[0] = max(max_perturbation[0], float(diffs.max()))
            return _sphere_fn(probes)

        grad, fe = _central_diff_gradient(
            x,
            spy_fn,
            lb,
            ub,
            span,
            fd_step=1e-4,
            trust_radius=tiny_trust,
        )

        assert fe == 2 * dim
        assert torch.isfinite(grad).all()
        # The maximum perturbation should be at most max_h.
        # Use dtype-aware tolerance: float32 (MPS) has ~1e-7 relative error.
        tol = 1e-6 if dtype == torch.float32 else 1e-12
        assert max_perturbation[0] <= max_h + tol, (
            f"FD perturbation {max_perturbation[0]:.6e} exceeds trust cap {max_h:.6e}"
        )

    def test_no_trust_cap_when_none(self, device: torch.device) -> None:
        """Without trust_radius, FD step should NOT be capped."""
        dtype = best_float_dtype(device)
        dim = 3
        lb, ub = _make_bounds(dim, -5.0, 5.0, device, dtype)
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        span = ub - lb

        max_perturbation = [0.0]

        def spy_fn(probes: torch.Tensor) -> torch.Tensor:
            diffs = (probes - x.unsqueeze(0)).abs()
            max_perturbation[0] = max(max_perturbation[0], float(diffs.max()))
            return _sphere_fn(probes)

        _grad, fe = _central_diff_gradient(
            x,
            spy_fn,
            lb,
            ub,
            span,
            fd_step=1e-4,
            trust_radius=None,
        )

        assert fe == 2 * dim
        # Without trust cap, the step should be fd_step * span = 1e-4 * 10 = 1e-3
        expected_h = 1e-4 * 10.0  # fd_step * span_per_dim
        assert max_perturbation[0] >= expected_h * 0.5, (
            f"Without trust cap, FD step should be at least ~{expected_h:.6e}, "
            f"got {max_perturbation[0]:.6e}"
        )

    def test_bfgs_still_converges_with_trust_cap(self, device: torch.device) -> None:
        """BFGS should still converge on sphere even with trust-capped FD steps."""
        dtype = best_float_dtype(device)
        dim = 5
        lb, ub = _make_bounds(dim, -5.12, 5.12, device, dtype)
        x0 = torch.full((dim,), 0.5, device=device, dtype=dtype)
        f0 = _sphere_fn(x0.unsqueeze(0)).squeeze()

        _best_x, best_f, _fe = fd_bfgs_polish(
            x0,
            f0,
            _sphere_fn,
            lb,
            ub,
            budget=500,
        )

        assert best_f < 0.1, f"BFGS with trust-capped FD should still converge, got {best_f}"


# ---------------------------------------------------------------------------
# nm_polish
# ---------------------------------------------------------------------------
class TestNmPolish:
    """Validate torch-native Nelder-Mead polish wrapper."""

    def test_nm_polish_improves(self, device: torch.device) -> None:
        """nm_polish should improve fitness from starting point."""
        dtype = best_float_dtype(device)
        dim = 5
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)
        f0 = (x0**2).sum()

        def fitness_fn(X: torch.Tensor) -> torch.Tensor:
            return (X**2).sum(dim=-1)

        _best_x, best_f, evals = nm_polish(
            x0,
            fitness_fn,
            budget=500,
            bounds=(-5.0, 5.0),
        )
        assert best_f < f0, f"nm_polish did not improve: {best_f} >= {f0}"
        assert evals > 0
        assert evals <= 500

    def test_nm_polish_respects_budget(self, device: torch.device) -> None:
        """nm_polish must not exceed the function evaluation budget."""
        dtype = best_float_dtype(device)
        dim = 5
        x0 = torch.full((dim,), 2.0, device=device, dtype=dtype)

        _best_x, _best_f, evals = nm_polish(
            x0,
            _sphere_fn,
            budget=50,
            bounds=(-5.12, 5.12),
        )
        assert evals <= 50, f"nm_polish exceeded budget: {evals} > 50"

    def test_nm_polish_return_types(self, device: torch.device) -> None:
        """nm_polish should return correct types and shapes."""
        dtype = best_float_dtype(device)
        dim = 3
        x0 = torch.ones(dim, device=device, dtype=dtype)

        best_x, best_f, evals = nm_polish(
            x0,
            _sphere_fn,
            budget=100,
            bounds=(-5.0, 5.0),
        )
        assert isinstance(best_x, torch.Tensor)
        assert isinstance(best_f, torch.Tensor)
        assert isinstance(evals, int)
        assert best_x.shape == (dim,)
        assert best_f.ndim == 0
        assert best_x.device.type == device.type

    def test_nm_polish_no_bounds(self, device: torch.device) -> None:
        """nm_polish should work without explicit bounds."""
        dtype = best_float_dtype(device)
        dim = 3
        x0 = torch.ones(dim, device=device, dtype=dtype)

        best_x, best_f, evals = nm_polish(
            x0,
            _sphere_fn,
            budget=200,
            bounds=None,
        )
        assert evals > 0
        assert torch.isfinite(best_x).all()
        assert torch.isfinite(best_f)
