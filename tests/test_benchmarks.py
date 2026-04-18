"""Tests for torch_dfo.benchmarks.classical benchmark functions."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import ATOL_F64_DEFAULT
from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import (
    BenchmarkProblem,
    BenchmarkSuite,
    ackley,
    griewank,
    levy,
    make_rotated,
    make_shifted,
    random_rotation_matrix,
    random_shift,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    zakharov,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
N, D = 5, 10  # batch size and default dimension for shape tests
ATOL_F64 = 1e-10
ATOL_F32 = 1e-5


def _atol(dtype: torch.dtype) -> float:
    return ATOL_F64 if dtype == torch.float64 else ATOL_F32


# ---------------------------------------------------------------------------
# Parametrised list of (function, optimum_point_factory, expected_value)
# ---------------------------------------------------------------------------
_OPTIMA = [
    ("sphere", sphere, lambda d, **kw: torch.zeros(d, **kw), 0.0),
    ("rosenbrock", rosenbrock, lambda d, **kw: torch.ones(d, **kw), 0.0),
    ("rastrigin", rastrigin, lambda d, **kw: torch.zeros(d, **kw), 0.0),
    ("ackley", ackley, lambda d, **kw: torch.zeros(d, **kw), 0.0),
    ("griewank", griewank, lambda d, **kw: torch.zeros(d, **kw), 0.0),
    (
        "schwefel",
        schwefel,
        lambda d, **kw: torch.full((d,), 420.9687, **kw),
        0.0,
    ),
    ("levy", levy, lambda d, **kw: torch.ones(d, **kw), 0.0),
    ("zakharov", zakharov, lambda d, **kw: torch.zeros(d, **kw), 0.0),
]

_ALL_FUNCTIONS = [sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy, zakharov]

# Functions known to be non-negative everywhere
_NONNEG_FUNCTIONS = [sphere, rastrigin]


# ===================================================================
# Shape tests
# ===================================================================


class TestBatchedShapes:
    """All functions must handle both ``(N, D)`` and ``(D,)`` inputs."""

    @pytest.mark.parametrize("fn", _ALL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_batched_shape(self, fn: object, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (N,), f"{fn.__name__}: expected (N,), got {y.shape}"

    @pytest.mark.parametrize("fn", _ALL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_single_shape(self, fn: object, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(D, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (), f"{fn.__name__}: expected scalar, got {y.shape}"


# ===================================================================
# Known-optimum tests
# ===================================================================


class TestKnownOptima:
    """Each function should return its known optimum at the canonical minimiser."""

    @pytest.mark.parametrize(
        ("name", "fn", "opt_factory", "expected"),
        _OPTIMA,
        ids=[t[0] for t in _OPTIMA],
    )
    def test_optimum_single(
        self,
        name: str,
        fn: object,
        opt_factory: object,
        expected: float,
        device: torch.device,
    ) -> None:
        dtype = best_float_dtype(device)
        x = opt_factory(D, device=device, dtype=dtype)
        val = fn(x)
        # Schwefel's approximation is less precise -- use tighter tolerance
        tol = 0.02 if name == "schwefel" else _atol(dtype)
        assert val.shape == ()
        assert torch.abs(val - expected) < tol, (
            f"{name}: f(optimum) = {val.item()}, expected {expected}"
        )

    @pytest.mark.parametrize(
        ("name", "fn", "opt_factory", "expected"),
        _OPTIMA,
        ids=[t[0] for t in _OPTIMA],
    )
    def test_optimum_batched(
        self,
        name: str,
        fn: object,
        opt_factory: object,
        expected: float,
        device: torch.device,
    ) -> None:
        dtype = best_float_dtype(device)
        x = opt_factory(D, device=device, dtype=dtype).unsqueeze(0).expand(N, -1)
        val = fn(x)
        tol = 0.02 if name == "schwefel" else _atol(dtype)
        assert val.shape == (N,)
        assert torch.all(torch.abs(val - expected) < tol), (
            f"{name}: f(optimum) = {val.tolist()}, expected {expected}"
        )


# ===================================================================
# Non-negativity / sanity checks
# ===================================================================


class TestSanity:
    """Smoke tests ensuring functions return finite values and optima are minima."""

    @pytest.mark.parametrize("fn", _ALL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_finite_output(self, fn: object, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype)
        y = fn(x)
        assert torch.all(torch.isfinite(y)), f"{fn.__name__} produced non-finite values"

    @pytest.mark.parametrize("fn", _NONNEG_FUNCTIONS, ids=lambda f: f.__name__)
    def test_nonneg_output(self, fn: object, device: torch.device) -> None:
        """Sphere and rastrigin are known to be >= 0 everywhere."""
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype)
        y = fn(x)
        assert torch.all(y >= 0), f"{fn.__name__} produced negative values"

    def test_sphere_positive(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype)
        assert torch.all(sphere(x) >= 0)

    def test_rastrigin_positive(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype) * 0.1  # stay near origin
        assert torch.all(rastrigin(x) >= -_atol(dtype))

    def test_zakharov_positive(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, D, device=device, dtype=dtype) * 0.1
        assert torch.all(zakharov(x) >= -_atol(dtype))


# ===================================================================
# Shifted / rotated variant tests
# ===================================================================


class TestShiftedRotated:
    """Tests for make_shifted, make_rotated, and helper generators."""

    def test_make_shifted_optimum(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        shift = torch.ones(D, device=device, dtype=dtype) * 2.0
        fn = make_shifted(sphere, shift)
        # Optimum should now be at x = shift
        x_opt = shift.clone()
        assert torch.abs(fn(x_opt)) < _atol(dtype)

    def test_make_shifted_batched(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        shift = torch.ones(D, device=device, dtype=dtype)
        fn = make_shifted(sphere, shift)
        x = torch.randn(N, D, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (N,)

    def test_make_rotated_optimum_preserved(self, device: torch.device) -> None:
        """Rotation around the origin preserves f(0) for origin-optimal functions."""
        dtype = best_float_dtype(device)
        rot = random_rotation_matrix(D, device=device, dtype=dtype)
        fn = make_rotated(sphere, rot)
        x_opt = torch.zeros(D, device=device, dtype=dtype)
        assert torch.abs(fn(x_opt)) < _atol(dtype)

    def test_make_rotated_batched(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        rot = random_rotation_matrix(D, device=device, dtype=dtype)
        fn = make_rotated(sphere, rot)
        x = torch.randn(N, D, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (N,)

    def test_random_rotation_matrix_orthogonality(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        r = random_rotation_matrix(D, device=device, dtype=dtype)
        eye = torch.eye(D, device=device, dtype=dtype)
        assert r.shape == (D, D)
        assert torch.allclose(r @ r.T, eye, atol=_atol(dtype)), "R @ R^T != I"
        assert torch.allclose(r.T @ r, eye, atol=_atol(dtype)), "R^T @ R != I"

    def test_random_shift_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        lb, ub = -5.0, 5.0
        s = random_shift(D, lb, ub, device=device, dtype=dtype)
        assert s.shape == (D,)
        assert torch.all(s >= lb)
        assert torch.all(s <= ub)

    def test_random_rotation_matrix_deterministic(self) -> None:
        """Same seed produces the same matrix."""
        g1 = torch.Generator(device="cpu")
        g1.manual_seed(123)
        r1 = random_rotation_matrix(D, dtype=torch.float64, generator=g1)

        g2 = torch.Generator(device="cpu")
        g2.manual_seed(123)
        r2 = random_rotation_matrix(D, dtype=torch.float64, generator=g2)

        assert torch.allclose(r1, r2)

    def test_random_shift_deterministic(self) -> None:
        """Same seed produces the same shift vector."""
        g1 = torch.Generator(device="cpu")
        g1.manual_seed(123)
        s1 = random_shift(D, -5, 5, dtype=torch.float64, generator=g1)

        g2 = torch.Generator(device="cpu")
        g2.manual_seed(123)
        s2 = random_shift(D, -5, 5, dtype=torch.float64, generator=g2)

        assert torch.allclose(s1, s2)

    def test_random_rotation_reseeds_from_initial_seed(self) -> None:
        """The source code re-seeds from generator.initial_seed(), meaning
        the generator state is ignored -- only the initial seed matters.
        Calling random_rotation_matrix twice with the SAME generator object
        produces identical results. This is intentional for reproducibility.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(456)
        r1 = random_rotation_matrix(D, dtype=torch.float64, generator=g)
        r2 = random_rotation_matrix(D, dtype=torch.float64, generator=g)
        assert torch.allclose(r1, r2), (
            "Expected identical results from re-seeded generator -- "
            "the function uses generator.initial_seed() to create a fresh CPU gen"
        )

    def test_rotation_matrix_det_abs_is_one(self) -> None:
        """A random orthogonal matrix from QR decomposition has |det(R)| = 1.
        The sign-correction in the source normalises the diagonal of R but
        does not guarantee det = +1 (proper rotation); det = -1 (improper
        rotation / reflection) is equally valid for an orthogonal matrix.
        """
        r = random_rotation_matrix(5, dtype=torch.float64)
        det = torch.linalg.det(r)
        assert torch.abs(torch.abs(det) - 1.0) < 1e-10, (
            f"|det(R)| = {torch.abs(det).item()}, expected 1.0 for an orthogonal matrix"
        )

    def test_make_rotated_changes_values(self) -> None:
        """A rotated rastrigin should produce different values than the
        unrotated version at a non-origin, non-axis-aligned point.
        """
        rot = random_rotation_matrix(3, dtype=torch.float64)
        fn_rotated = make_rotated(rastrigin, rot)

        x = torch.tensor([1.5, 2.3, 0.7], dtype=torch.float64)
        val_original = rastrigin(x)
        val_rotated = fn_rotated(x)

        assert not torch.allclose(val_original, val_rotated), (
            "Rotated rastrigin should differ from unrotated at a non-origin point"
        )


# ===================================================================
# Batch vs single consistency
# ===================================================================


class TestBatchConsistency:
    """Batch evaluation must match individual evaluation."""

    @pytest.mark.parametrize(
        "fn",
        [sphere, rosenbrock, rastrigin, ackley, griewank, levy],
        ids=lambda f: f.__name__,
    )
    def test_batch_vs_single_consistency(self, fn: object) -> None:
        """Evaluate N points as a batch and individually; results must match."""
        torch.manual_seed(42)
        batch = torch.randn(N, D, dtype=torch.float64)

        f_batch = fn(batch)
        f_single = torch.stack([fn(batch[i : i + 1]).squeeze(0) for i in range(N)])

        assert torch.allclose(f_batch, f_single, atol=ATOL_F64_DEFAULT), (
            f"{fn.__name__}: batch vs single mismatch, "
            f"max diff = {(f_batch - f_single).abs().max().item()}"
        )


# ===================================================================
# BenchmarkSuite tests
# ===================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite.classical() and BenchmarkSuite.full()."""

    def test_classical_count(self) -> None:
        problems = BenchmarkSuite.classical()
        # 6 functions x 2 dims = 12
        assert len(problems) == 12

    def test_classical_types(self) -> None:
        problems = BenchmarkSuite.classical()
        for p in problems:
            assert isinstance(p, BenchmarkProblem)
            assert callable(p.fn)
            assert isinstance(p.dim, int) and p.dim > 0
            assert isinstance(p.bounds, tuple) and len(p.bounds) == 2
            assert p.bounds[0] < p.bounds[1]

    def test_classical_dims(self) -> None:
        problems = BenchmarkSuite.classical(dims=(5, 20))
        dims_found = {p.dim for p in problems}
        assert dims_found == {5, 20}

    def test_classical_canonical_bounds(self) -> None:
        """D4: pin BBOB/CEC canonical bounds for each classical problem.

        Sources: these bounds are the literature standard (e.g. sphere and
        rastrigin on [-5.12, 5.12], ackley on [-32.768, 32.768]). This test
        exists so a silent refactor of ``_BOUNDS`` in
        ``src/torch_dfo/benchmarks/classical.py`` does not drift them.
        """
        expected: dict[str, tuple[float, float]] = {
            "sphere": (-5.12, 5.12),
            "rosenbrock": (-2.048, 2.048),
            "rastrigin": (-5.12, 5.12),
            "ackley": (-32.768, 32.768),
            "griewank": (-600.0, 600.0),
            "levy": (-10.0, 10.0),
        }
        problems = BenchmarkSuite.classical(dims=(10,))
        seen: dict[str, tuple[float, float]] = {}
        for p in problems:
            # Name is e.g. "sphere_10d"; strip dim suffix to compare.
            family = p.name.rsplit("_", 1)[0]
            seen[family] = p.bounds
        assert set(seen.keys()) == set(expected.keys()), (
            f"classical() family set drifted: got {set(seen.keys())}, "
            f"expected {set(expected.keys())}"
        )
        for fam, want in expected.items():
            assert seen[fam] == want, (
                f"Canonical bounds for {fam} drifted: got {seen[fam]}, expected {want}"
            )

    def test_full_count(self) -> None:
        problems = BenchmarkSuite.full()
        assert len(problems) == 16

    def test_full_names(self) -> None:
        problems = BenchmarkSuite.full()
        names = {p.name for p in problems}
        assert "shifted_sphere_30d" in names
        assert "shifted_rosenbrock_30d" in names
        assert "rotated_rastrigin_30d" in names
        assert "rotated_ackley_30d" in names

    def test_full_custom_stress_dim(self) -> None:
        problems = BenchmarkSuite.full(dims=(10, 20, 40), stress_dim=40)
        names = {p.name for p in problems}
        assert "shifted_sphere_40d" in names
        assert "shifted_rosenbrock_40d" in names
        assert "rotated_rastrigin_40d" in names
        assert "rotated_ackley_40d" in names

    def test_full_callable(self, device: torch.device) -> None:
        """Every problem in the full suite can be evaluated without error."""
        dtype = best_float_dtype(device)
        problems = BenchmarkSuite.full(device=device, dtype=dtype)
        for p in problems:
            x = torch.randn(N, p.dim, device=device, dtype=dtype)
            y = p.fn(x)
            assert y.shape == (N,), f"{p.name}: expected (N,), got {y.shape}"
            assert torch.all(torch.isfinite(y)), f"{p.name}: non-finite values"

    def test_full_deterministic(self) -> None:
        """Same seed produces identical suites."""
        p1 = BenchmarkSuite.full(seed=99)
        p2 = BenchmarkSuite.full(seed=99)
        x = torch.randn(N, 30, dtype=torch.float64)
        for a, b in zip(p1, p2, strict=True):
            assert a.name == b.name
            if a.dim == 30:
                ya = a.fn(x)
                yb = b.fn(x)
                assert torch.allclose(ya, yb), f"{a.name}: results differ across seeds"


# ===================================================================
# Additional edge-case tests
# ===================================================================


class TestEdgeCases:
    """Edge cases: D=1, D=2, large batches."""

    @pytest.mark.parametrize("fn", _ALL_FUNCTIONS, ids=lambda f: f.__name__)
    def test_dim_2(self, fn: object, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(N, 2, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (N,)

    @pytest.mark.parametrize(
        "fn",
        [sphere, rastrigin, ackley, griewank, schwefel, zakharov],
        ids=lambda f: f.__name__,
    )
    def test_dim_1(self, fn: object, device: torch.device) -> None:
        """Functions that are well-defined for D=1."""
        dtype = best_float_dtype(device)
        x = torch.randn(N, 1, device=device, dtype=dtype)
        y = fn(x)
        assert y.shape == (N,)

    def test_rosenbrock_needs_at_least_2d(self, device: torch.device) -> None:
        """Rosenbrock with D=2 is the classic banana function."""
        dtype = best_float_dtype(device)
        x = torch.ones(2, device=device, dtype=dtype)
        assert torch.abs(rosenbrock(x)) < _atol(dtype)

    def test_levy_needs_at_least_2d(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.ones(2, device=device, dtype=dtype)
        assert torch.abs(levy(x)) < _atol(dtype)

    def test_large_batch(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        x = torch.randn(1000, D, device=device, dtype=dtype)
        y = sphere(x)
        assert y.shape == (1000,)
