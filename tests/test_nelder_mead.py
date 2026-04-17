"""Tests for NelderMead simplex optimizer -- ask/tell interface, convergence, bounds."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import (
    ATOL_F64_DEFAULT,
    CONV_NELDER_MEAD_1D,
    CONV_ROSENBROCK_2D,
    CONV_SPHERE_10D_TIGHT,
)
from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import rosenbrock, sphere
from torch_dfo.nelder_mead import NelderMead


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Validate constructor wiring and pop_size = dim + 1."""

    def test_pop_size_equals_dim_plus_one(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=5, bounds=10.0, device=device, dtype=dtype, seed=42)
        assert opt.pop_size == 6
        assert opt.population.shape == (6, 5)

    def test_pop_size_various_dims(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        for d in (1, 2, 10, 50):
            opt = NelderMead(dim=d, bounds=5.0, device=device, dtype=dtype, seed=0)
            assert opt.pop_size == d + 1

    def test_default_coefficients(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=1.0, device=device, dtype=dtype)
        assert opt.alpha == 1.0
        assert opt.gamma == 2.0
        assert opt.rho == 0.5
        assert opt.shrink_coeff == 0.5

    def test_custom_coefficients(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(
            dim=3,
            bounds=1.0,
            device=device,
            dtype=dtype,
            alpha=1.5,
            gamma=2.5,
            rho=0.4,
            shrink=0.6,
        )
        assert opt.alpha == 1.5
        assert opt.gamma == 2.5
        assert opt.rho == 0.4
        assert opt.shrink_coeff == 0.6

    def test_device_and_dtype(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=4, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt.device == device
        assert opt.dtype == dtype
        assert opt.population.device.type == device.type
        assert opt.population.dtype == dtype
        assert opt.fitness.device.type == device.type
        assert opt.fitness.dtype == dtype

    def test_tuple_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=(-2.0, 8.0), device=device, dtype=dtype, seed=0)
        assert torch.all(opt.lb == -2.0)
        assert torch.all(opt.ub == 8.0)


# ---------------------------------------------------------------------------
# ask() shapes
# ---------------------------------------------------------------------------
class TestAskShapes:
    """Verify tensor shapes returned by ask() in different states."""

    def test_first_ask_returns_simplex(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 5
        opt = NelderMead(dim=dim, bounds=10.0, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert candidates.shape == (dim + 1, dim)
        assert candidates.device.type == device.type
        assert candidates.dtype == dtype

    def test_subsequent_ask_returns_four_candidates(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 4
        opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)

        # First ask/tell cycle (init)
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # Second ask: should return 4 candidates
        c2 = opt.ask()
        assert c2.shape == (4, dim)
        assert c2.device.type == device.type
        assert c2.dtype == dtype

    def test_post_shrink_ask_returns_full_simplex(self, device: torch.device) -> None:
        """After a shrink, the next ask() must return the full simplex for re-eval."""
        dtype = best_float_dtype(device)
        dim = 3

        # Use Chebyshev norm (max|x_i|) -- non-smooth, triggers shrink frequently
        def chebyshev(x: torch.Tensor) -> torch.Tensor:
            return x.abs().max(dim=-1).values

        opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)

        c = opt.ask()
        f = chebyshev(c)
        opt.tell(c, f)

        found_shrink = False
        for _ in range(500):
            c = opt.ask()
            if c.shape[0] == dim + 1:
                # This is a post-shrink re-evaluation ask
                found_shrink = True
                assert c.shape == (dim + 1, dim)
                break
            f = chebyshev(c)
            opt.tell(c, f)

        assert found_shrink, "Shrink was never triggered in 500 iterations"


# ---------------------------------------------------------------------------
# tell() updates
# ---------------------------------------------------------------------------
class TestTellUpdates:
    """Verify that tell() modifies optimizer state correctly."""

    def test_initial_tell_stores_population_and_fitness(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3
        opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        assert torch.allclose(opt.fitness, f)
        assert torch.allclose(opt.population, c)

    def test_generation_increments(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt._generation == 0

        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 1

        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 2

    def test_best_tracked_across_iterations(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)

        c = opt.ask()
        opt.tell(c, sphere(c))
        _, f0 = opt.best()

        # Run a few more iterations -- best must strictly improve on sphere
        for _ in range(10):
            c = opt.ask()
            opt.tell(c, sphere(c))

        _, f_later = opt.best()
        assert f_later.item() < f0.item()

    def test_best_returns_clones(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        opt.tell(c, sphere(c))

        sol, fit = opt.best()
        sol.fill_(999.0)
        fit.fill_(-1.0)
        sol2, fit2 = opt.best()
        assert not torch.all(sol2 == 999.0)
        assert fit2.item() != -1.0


# ---------------------------------------------------------------------------
# Within bounds
# ---------------------------------------------------------------------------
class TestWithinBounds:
    """All candidates returned by ask() must respect [lb, ub]."""

    def test_initial_simplex_within_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=5, bounds=(-3.0, 7.0), device=device, dtype=dtype, seed=42)
        c = opt.ask()
        assert torch.all(c >= opt.lb)
        assert torch.all(c <= opt.ub)

    def test_candidates_within_bounds_over_many_iterations(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=4, bounds=(-2.0, 2.0), device=device, dtype=dtype, seed=42)

        c = opt.ask()
        assert torch.all(c >= opt.lb) and torch.all(c <= opt.ub)
        opt.tell(c, sphere(c))

        for _ in range(50):
            c = opt.ask()
            assert torch.all(c >= opt.lb), f"Candidate below lb: {c.min()}"
            assert torch.all(c <= opt.ub), f"Candidate above ub: {c.max()}"
            opt.tell(c, sphere(c))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """Same seed must produce identical trajectories."""

    def test_same_seed_same_trajectory(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 4

        def run_optimizer(seed: int) -> list[float]:
            opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=seed)
            history: list[float] = []
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            history.append(opt.best_fitness.item())
            for _ in range(20):
                c = opt.ask()
                f = sphere(c)
                opt.tell(c, f)
                history.append(opt.best_fitness.item())
            return history

        h1 = run_optimizer(seed=123)
        h2 = run_optimizer(seed=123)
        assert h1 == h2, "Trajectories differ with same seed"

    def test_different_seeds_differ(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 3

        def get_initial_simplex(seed: int) -> torch.Tensor:
            opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=seed)
            return opt.ask()

        c1 = get_initial_simplex(seed=1)
        c2 = get_initial_simplex(seed=2)
        assert not torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# Convergence: sphere 10d  (CPU float64 only)
# ---------------------------------------------------------------------------
class TestSphereConvergence:
    """Nelder-Mead should converge on 10d sphere to < 1e-8 within 5000 cycles."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_sphere_10d(self, seed: int) -> None:
        dim = 10
        opt = NelderMead(
            dim=dim,
            bounds=(-5.12, 5.12),
            device="cpu",
            dtype=torch.float64,
            seed=seed,
        )

        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        for _ in range(5000):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        _, best_f = opt.best()
        assert best_f.item() < CONV_SPHERE_10D_TIGHT, (
            f"Sphere 10d did not converge: best_f={best_f.item():.2e} (seed={seed})"
        )


# ---------------------------------------------------------------------------
# Convergence: rosenbrock 2d  (CPU float64 only)
# ---------------------------------------------------------------------------
class TestRosenbrockConvergence:
    """Nelder-Mead should converge on 2d Rosenbrock to < 1e-4 within 3000 cycles."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_rosenbrock_2d(self, seed: int) -> None:
        dim = 2
        opt = NelderMead(
            dim=dim,
            bounds=(-5.0, 10.0),
            device="cpu",
            dtype=torch.float64,
            seed=seed,
        )

        c = opt.ask()
        f = rosenbrock(c)
        opt.tell(c, f)

        for _ in range(3000):
            c = opt.ask()
            f = rosenbrock(c)
            opt.tell(c, f)

        _, best_f = opt.best()
        assert best_f.item() < CONV_ROSENBROCK_2D, (
            f"Rosenbrock 2d did not converge: best_f={best_f.item():.2e} (seed={seed})"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and 1-dimensional problems."""

    def test_1d_problem(self, device: torch.device) -> None:
        """Nelder-Mead should work on a 1-dimensional problem."""
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=1, bounds=(-10.0, 10.0), device=device, dtype=dtype, seed=42)
        assert opt.pop_size == 2  # simplex has 2 vertices in 1d

        c = opt.ask()
        assert c.shape == (2, 1)
        f = sphere(c)
        opt.tell(c, f)

        for _ in range(100):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        _, best_f = opt.best()
        assert best_f.item() < CONV_NELDER_MEAD_1D  # 1d sphere converges easily in 100 iters

    def test_scalar_bounds(self, device: torch.device) -> None:
        """Scalar bounds should create symmetric [-b, b] range."""
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert torch.all(opt.lb == -5.0)
        assert torch.all(opt.ub == 5.0)

    def test_fitness_monotonically_improves(self, device: torch.device) -> None:
        """best_fitness should never increase (worsen) across iterations."""
        dtype = best_float_dtype(device)
        opt = NelderMead(dim=3, bounds=5.0, device=device, dtype=dtype, seed=42)

        c = opt.ask()
        opt.tell(c, sphere(c))
        prev_best = opt.best_fitness.item()

        for _ in range(50):
            c = opt.ask()
            opt.tell(c, sphere(c))
            current_best = opt.best_fitness.item()
            assert current_best <= prev_best + 1e-12, (
                f"best_fitness worsened: {prev_best} -> {current_best}"
            )
            prev_best = current_best


# ---------------------------------------------------------------------------
# NM operations
# ---------------------------------------------------------------------------
class TestNMOperations:
    """Verify Nelder-Mead simplex operations at the formula level."""

    def test_reflect_operation(self, device: torch.device) -> None:
        """Reflected point must match: centroid + alpha * (centroid - worst)."""
        dtype = best_float_dtype(device)
        dim = 2
        opt = NelderMead(dim=dim, bounds=10.0, device=device, dtype=dtype, seed=42)

        # Initialize the optimizer so _initialized=True and _needs_full_eval=False
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # Manually set known simplex vertices and fitness
        # Vertices: best=(0,0) f=0, mid=(1,0) f=1, worst=(0,2) f=4
        opt.population[0] = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        opt.population[1] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        opt.population[2] = torch.tensor([0.0, 2.0], device=device, dtype=dtype)
        opt.fitness[0] = torch.tensor(0.0, device=device, dtype=dtype)
        opt.fitness[1] = torch.tensor(1.0, device=device, dtype=dtype)
        opt.fitness[2] = torch.tensor(4.0, device=device, dtype=dtype)

        candidates = opt.ask()

        # ask() sorts by fitness first, so after sorting:
        # best=(0,0) f=0, mid=(1,0) f=1, worst=(0,2) f=4
        # centroid = mean of all except worst = mean([(0,0),(1,0)]) = (0.5, 0)
        # reflected = centroid + alpha*(centroid - worst) = (0.5,0) + 1.0*((0.5,0)-(0,2))
        #           = (0.5,0) + (0.5,-2) = (1.0, -2.0)
        centroid = torch.tensor([0.5, 0.0], device=device, dtype=dtype)
        worst = torch.tensor([0.0, 2.0], device=device, dtype=dtype)
        expected_reflected = centroid + opt.alpha * (centroid - worst)

        # candidates[0] is the reflected point
        assert torch.allclose(candidates[0], expected_reflected, atol=ATOL_F64_DEFAULT), (
            f"Reflected point mismatch: got {candidates[0]}, expected {expected_reflected}"
        )

    def test_shrink_trigger(self, device: torch.device) -> None:
        """Shrink triggers when all NM operations fail to improve on the worst vertex."""
        dtype = best_float_dtype(device)
        dim = 2
        opt = NelderMead(dim=dim, bounds=10.0, device=device, dtype=dtype, seed=42)

        # Initialize
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # Set up simplex where reflected/expanded/contracted will all be bad
        # Place vertices in a tight cluster near a corner so operations overshoot
        opt.population[0] = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        opt.population[1] = torch.tensor([0.1, 0.0], device=device, dtype=dtype)
        opt.population[2] = torch.tensor([0.0, 0.1], device=device, dtype=dtype)
        opt.fitness[0] = torch.tensor(0.0, device=device, dtype=dtype)
        opt.fitness[1] = torch.tensor(0.01, device=device, dtype=dtype)
        opt.fitness[2] = torch.tensor(0.01, device=device, dtype=dtype)

        candidates = opt.ask()
        assert candidates.shape == (4, dim)

        # Feed fitness values that are ALL worse than the worst vertex (f=0.01)
        # This forces inside contraction to fail, triggering shrink
        bad_fitness = torch.tensor([100.0, 200.0, 300.0, 400.0], device=device, dtype=dtype)
        opt.tell(candidates, bad_fitness)

        # After failed contraction, _needs_full_eval should be True (shrink happened)
        assert opt._needs_full_eval is True, (
            "Shrink should have been triggered but _needs_full_eval is False"
        )

    def test_sorted_simplex_ordering(self, device: torch.device) -> None:
        """After iterations, simplex fitness should be sorted non-decreasingly."""
        dtype = best_float_dtype(device)
        dim = 3
        opt = NelderMead(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)

        c = opt.ask()
        opt.tell(c, sphere(c))

        for _ in range(30):
            c = opt.ask()
            opt.tell(c, sphere(c))

        # After the next ask(), the simplex is sorted by fitness (best first)
        # We just need to trigger sorting via ask() to verify ordering
        c = opt.ask()
        if c.shape[0] == 4:
            # Normal iteration: simplex was sorted by ask()
            for i in range(opt.pop_size - 1):
                assert opt.fitness[i] <= opt.fitness[i + 1], (
                    f"Simplex not sorted at index {i}: "
                    f"{opt.fitness[i].item()} > {opt.fitness[i + 1].item()}"
                )

    def test_bounds_clamping_at_boundary(self, device: torch.device) -> None:
        """Candidates near tight bounds must be clamped within [lb, ub]."""
        dtype = best_float_dtype(device)
        dim = 3
        opt = NelderMead(dim=dim, bounds=(-0.5, 0.5), device=device, dtype=dtype, seed=42)

        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        for _ in range(50):
            c = opt.ask()
            assert torch.all(c >= opt.lb), (
                f"Candidate below lb: min={c.min().item()}, lb={opt.lb[0].item()}"
            )
            assert torch.all(c <= opt.ub), (
                f"Candidate above ub: max={c.max().item()}, ub={opt.ub[0].item()}"
            )
            f = sphere(c)
            opt.tell(c, f)
