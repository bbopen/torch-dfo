"""Tests for torch_dfo.base -- BaseOptimizer ask/tell interface and workspace."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import CONV_BASE_WITH_BOUNDS
from tests.conftest import best_float_dtype
from torch_dfo.base import BaseOptimizer
from torch_dfo.utils import clamp_to_bounds


# ---------------------------------------------------------------------------
# Concrete subclass for integration testing
# ---------------------------------------------------------------------------
class RandomSearch(BaseOptimizer):
    """Minimal concrete subclass: uniform random candidates within bounds."""

    def ask(self) -> torch.Tensor:
        candidates = self._rand(self.pop_size, self.dim)
        candidates = candidates * (self.ub - self.lb) + self.lb
        self.population.copy_(candidates)
        self._generation += 1
        return candidates

    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        self.fitness.copy_(fitness)
        self._update_best(candidates, fitness)


# ---------------------------------------------------------------------------
# Constructor and field initialisation
# ---------------------------------------------------------------------------
class TestConstructor:
    """Validate that the constructor wires up all fields correctly."""

    def test_scalar_fields(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=5, bounds=10.0, pop_size=20, device=device, dtype=dtype)
        assert opt.dim == 5
        assert opt.pop_size == 20
        assert opt.device == device
        assert opt.dtype == dtype
        assert opt._generation == 0

    def test_bounds_shape_and_values(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=3, bounds=(-2.0, 4.0), pop_size=10, device=device, dtype=dtype)
        assert opt.lb.shape == (3,)
        assert opt.ub.shape == (3,)
        assert torch.all(opt.lb == -2.0)
        assert torch.all(opt.ub == 4.0)
        assert opt.lb.device.type == device.type
        assert opt.ub.device.type == device.type
        assert opt.lb.dtype == dtype
        assert opt.ub.dtype == dtype

    def test_scalar_bound_symmetric(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=4, bounds=5.0, pop_size=8, device=device, dtype=dtype)
        assert torch.all(opt.lb == -5.0)
        assert torch.all(opt.ub == 5.0)

    def test_generator_created(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=99)
        assert isinstance(opt._gen, torch.Generator)
        # Generator device must be CPU or CUDA (never MPS/XLA)
        assert opt._gen_device.type in ("cpu", "cuda")

    def test_seed_none_accepted(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=None)
        assert isinstance(opt._gen, torch.Generator)


# ---------------------------------------------------------------------------
# Pre-allocated tensor shapes, device, dtype
# ---------------------------------------------------------------------------
class TestPreallocatedTensors:
    """Verify workspace tensors have the correct shape, device, and dtype."""

    def test_population_shape(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=7, bounds=1.0, pop_size=15, device=device, dtype=default_dtype)
        assert opt.population.shape == (15, 7)
        assert opt.population.device.type == device.type
        assert opt.population.dtype == default_dtype

    def test_fitness_shape_and_init(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=3, bounds=1.0, pop_size=10, device=device, dtype=default_dtype)
        assert opt.fitness.shape == (10,)
        assert opt.fitness.device.type == device.type
        assert opt.fitness.dtype == default_dtype
        assert torch.all(opt.fitness == float("inf"))

    def test_best_solution_shape_and_init(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = BaseOptimizer(dim=4, bounds=1.0, pop_size=6, device=device, dtype=default_dtype)
        assert opt.best_solution.shape == (4,)
        assert opt.best_solution.device.type == device.type
        assert opt.best_solution.dtype == default_dtype
        assert torch.all(opt.best_solution == 0.0)

    def test_best_fitness_init(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=default_dtype)
        assert opt.best_fitness.shape == ()
        assert opt.best_fitness.device.type == device.type
        assert opt.best_fitness.dtype == default_dtype
        assert opt.best_fitness.item() == float("inf")


# ---------------------------------------------------------------------------
# best() returns clones
# ---------------------------------------------------------------------------
class TestBestReturnsClones:
    """Modifying values returned by best() must not affect internal state."""

    def test_solution_clone(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=3, bounds=1.0, pop_size=4, device=device, dtype=default_dtype)
        sol, _ = opt.best()
        sol.fill_(999.0)
        assert torch.all(opt.best_solution == 0.0), "best_solution was mutated through clone"

    def test_fitness_clone(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=3, bounds=1.0, pop_size=4, device=device, dtype=default_dtype)
        _, fit = opt.best()
        fit.fill_(-1.0)
        assert opt.best_fitness.item() == float("inf"), "best_fitness was mutated through clone"


# ---------------------------------------------------------------------------
# _update_best
# ---------------------------------------------------------------------------
class TestUpdateBest:
    """Validate global-best tracking logic across successive batches."""

    def test_first_batch_updates(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=2, bounds=5.0, pop_size=3, device=device, dtype=default_dtype)
        candidates = torch.tensor(
            [[1.0, 2.0], [0.5, 0.5], [3.0, 3.0]],
            device=device,
            dtype=default_dtype,
        )
        fitness = torch.tensor([5.0, 0.5, 18.0], device=device, dtype=default_dtype)
        opt._update_best(candidates, fitness)

        assert opt.best_fitness.item() == pytest.approx(0.5)
        assert torch.equal(opt.best_solution, candidates[1])

    def test_second_batch_improves(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(dim=2, bounds=5.0, pop_size=2, device=device, dtype=default_dtype)
        # First batch
        c1 = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=device, dtype=default_dtype)
        f1 = torch.tensor([2.0, 8.0], device=device, dtype=default_dtype)
        opt._update_best(c1, f1)
        assert opt.best_fitness.item() == pytest.approx(2.0)

        # Second batch -- better
        c2 = torch.tensor([[0.1, 0.1], [3.0, 3.0]], device=device, dtype=default_dtype)
        f2 = torch.tensor([0.02, 18.0], device=device, dtype=default_dtype)
        opt._update_best(c2, f2)
        assert opt.best_fitness.item() == pytest.approx(0.02)
        assert torch.equal(opt.best_solution, c2[0])

    def test_no_improvement_keeps_best(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = BaseOptimizer(dim=2, bounds=5.0, pop_size=2, device=device, dtype=default_dtype)
        # First batch establishes the best
        c1 = torch.tensor([[0.1, 0.1], [2.0, 2.0]], device=device, dtype=default_dtype)
        f1 = torch.tensor([0.02, 8.0], device=device, dtype=default_dtype)
        opt._update_best(c1, f1)
        prev_best_x = opt.best_solution.clone()
        prev_best_f = opt.best_fitness.clone()

        # Second batch -- worse
        c2 = torch.tensor([[5.0, 5.0], [4.0, 4.0]], device=device, dtype=default_dtype)
        f2 = torch.tensor([50.0, 32.0], device=device, dtype=default_dtype)
        opt._update_best(c2, f2)
        assert opt.best_fitness.item() == pytest.approx(prev_best_f.item())
        assert torch.equal(opt.best_solution, prev_best_x)

    def test_equal_fitness_no_update(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """When batch best equals current best, the original solution is preserved."""
        opt = BaseOptimizer(dim=2, bounds=5.0, pop_size=1, device=device, dtype=default_dtype)
        c1 = torch.tensor([[1.0, 2.0]], device=device, dtype=default_dtype)
        f1 = torch.tensor([3.0], device=device, dtype=default_dtype)
        opt._update_best(c1, f1)

        # Same fitness, different location -- strict < means no update
        c2 = torch.tensor([[9.0, 9.0]], device=device, dtype=default_dtype)
        f2 = torch.tensor([3.0], device=device, dtype=default_dtype)
        opt._update_best(c2, f2)
        assert torch.equal(opt.best_solution, c1[0])

    def test_update_best_with_nan_fitness(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """NaN fitness must not corrupt best_fitness.

        PyTorch semantics: NaN < inf is False, so _update_best should
        leave best_fitness at inf when all fitness values are NaN.
        """
        opt = RandomSearch(
            dim=2,
            bounds=5.0,
            pop_size=3,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        candidates = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            device=device,
            dtype=default_dtype,
        )
        nan_fitness = torch.tensor(
            [float("nan"), float("nan"), float("nan")],
            device=device,
            dtype=default_dtype,
        )
        opt._update_best(candidates, nan_fitness)
        assert opt.best_fitness.item() == float("inf"), (
            "NaN fitness corrupted best_fitness -- should remain inf"
        )

    def test_update_best_with_all_inf_batch(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """All-inf fitness batch must not change an already-established best."""
        opt = RandomSearch(
            dim=2,
            bounds=5.0,
            pop_size=2,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        # Establish a finite best first
        c1 = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=device, dtype=default_dtype)
        f1 = torch.tensor([3.0, 7.0], device=device, dtype=default_dtype)
        opt._update_best(c1, f1)
        prev_best = opt.best_fitness.item()

        # Now feed all-inf
        c2 = torch.tensor([[9.0, 9.0], [8.0, 8.0]], device=device, dtype=default_dtype)
        f2 = torch.tensor([float("inf"), float("inf")], device=device, dtype=default_dtype)
        opt._update_best(c2, f2)
        assert opt.best_fitness.item() == pytest.approx(prev_best), (
            "All-inf batch should not change an established best"
        )

    def test_update_best_with_negative_fitness(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """Negative fitness values are valid; the most negative should win."""
        opt = RandomSearch(
            dim=2,
            bounds=5.0,
            pop_size=3,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        candidates = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            device=device,
            dtype=default_dtype,
        )
        fitness = torch.tensor([-5.0, -1.0, -10.0], device=device, dtype=default_dtype)
        opt._update_best(candidates, fitness)

        assert opt.best_fitness.item() == pytest.approx(-10.0)
        assert torch.equal(opt.best_solution, candidates[2])


# ---------------------------------------------------------------------------
# ask / tell raise NotImplementedError on base class
# ---------------------------------------------------------------------------
# ``BaseOptimizer`` is intentionally *not* declared with ``ABCMeta``.
# Its ``ask``/``tell`` methods raise ``NotImplementedError`` at runtime
# rather than being abstract-by-metaclass, so subclasses can import and
# instantiate ``BaseOptimizer`` in testing harnesses (see e.g.
# ``test_update_best_with_nan_fitness`` using a concrete subclass) without
# paying the metaclass import cost. The tests below pin that runtime contract.
class TestBaseAbstractMethods:
    """Base class ask/tell must raise NotImplementedError (not ABCMeta)."""

    def test_ask_raises(self) -> None:
        opt = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device="cpu")
        with pytest.raises(NotImplementedError):
            opt.ask()

    def test_tell_raises(self) -> None:
        opt = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device="cpu")
        candidates = torch.zeros(4, 2)
        fitness = torch.zeros(4)
        with pytest.raises(NotImplementedError):
            opt.tell(candidates, fitness)


# ---------------------------------------------------------------------------
# Random helper methods
# ---------------------------------------------------------------------------
class TestRandomHelpers:
    """Validate _rand, _randn, _randperm, _randint produce correct outputs."""

    def test_rand_shape_and_device(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(
            dim=3,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        t = opt._rand(5, 3)
        assert t.shape == (5, 3)
        assert t.device.type == device.type
        assert t.dtype == default_dtype
        # Uniform [0, 1)
        assert torch.all(t >= 0.0)
        assert torch.all(t < 1.0)

    def test_randn_shape_and_device(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = BaseOptimizer(
            dim=3,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        t = opt._randn(10, 3)
        assert t.shape == (10, 3)
        assert t.device.type == device.type
        assert t.dtype == default_dtype

    def test_randperm_shape_and_device(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = BaseOptimizer(
            dim=3,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        t = opt._randperm(10)
        assert t.shape == (10,)
        assert t.device.type == device.type
        # Must contain each integer exactly once
        assert torch.equal(t.sort().values, torch.arange(10, device=device))

    def test_randint_shape_and_device(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = BaseOptimizer(
            dim=3,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        t = opt._randint(0, 5, (8,))
        assert t.shape == (8,)
        assert t.device.type == device.type
        assert torch.all(t >= 0)
        assert torch.all(t < 5)

    def test_rand_cross_device_mps_fallback(self) -> None:
        """D5: When ``_gen_device`` != target device (e.g. MPS), ``_rand`` must
        generate on the generator's device then move to the target device.

        Skip if MPS is unavailable. CUDA is not a useful substitute because its
        Generator lives on-device, so ``_gen_device == device`` there.
        """
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("requires MPS to exercise the _gen_device != device fallback")
        mps = torch.device("mps")
        opt = BaseOptimizer(dim=3, bounds=1.0, pop_size=4, device=mps, dtype=torch.float32, seed=42)
        # Expected: generator lives on CPU (MPS has no native generator) but
        # _rand must return an MPS tensor.
        assert opt._gen_device.type == "cpu", (
            f"Expected CPU-backed generator on MPS device, got {opt._gen_device}"
        )
        t = opt._rand(4, 3)
        assert t.device.type == "mps", (
            f"_rand did not move to target device: got {t.device}"
        )
        # Same check for _randn, _randperm, _randint.
        assert opt._randn(4, 3).device.type == "mps"
        assert opt._randperm(5).device.type == "mps"
        assert opt._randint(0, 5, (4,)).device.type == "mps"

    def test_rand_reproducibility(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=7)
        opt2 = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=7)
        assert torch.equal(opt1._rand(6, 2), opt2._rand(6, 2))

    def test_randn_reproducibility(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=13)
        opt2 = BaseOptimizer(dim=2, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=13)
        assert torch.equal(opt1._randn(6, 2), opt2._randn(6, 2))

    def test_randperm_randint_dtype(self, device: torch.device, default_dtype: torch.dtype) -> None:
        """_randperm and _randint must return int64 tensors."""
        opt = BaseOptimizer(
            dim=3,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        perm = opt._randperm(10)
        assert perm.dtype == torch.int64, f"_randperm returned {perm.dtype}, expected int64"
        rint = opt._randint(0, 10, (5,))
        assert rint.dtype == torch.int64, f"_randint returned {rint.dtype}, expected int64"

    def test_randn_statistical_sanity(self, device: torch.device) -> None:
        """_randn should produce approximately N(0,1) samples."""
        dtype = best_float_dtype(device)
        opt = BaseOptimizer(dim=1, bounds=1.0, pop_size=4, device=device, dtype=dtype, seed=0)
        t = opt._randn(10000)
        mean = t.float().mean().item()
        std = t.float().std().item()
        assert abs(mean) < 0.1, f"_randn mean {mean} too far from 0"
        assert abs(std - 1.0) < 0.15, f"_randn std {std} too far from 1.0"


# ---------------------------------------------------------------------------
# RandomSearch integration: full ask/tell loop on sphere
# ---------------------------------------------------------------------------
class TestRandomSearchIntegration:
    """End-to-end optimization loop with the concrete RandomSearch subclass."""

    def test_sphere_optimization(self, device: torch.device, default_dtype: torch.dtype) -> None:
        """RandomSearch on f(x) = sum(x^2) should find something < 5.0
        with pop=50 over 20 generations on a 5d sphere.
        """
        dim = 5
        opt = RandomSearch(
            dim=dim,
            bounds=5.0,
            pop_size=50,
            device=device,
            dtype=default_dtype,
            seed=42,
        )

        for _ in range(20):
            candidates = opt.ask()
            f = (candidates**2).sum(dim=-1)
            opt.tell(candidates, f)

        sol, fit = opt.best()
        assert fit.item() < CONV_BASE_WITH_BOUNDS, (
            f"Random search with pop=50, 20 gens on 5d sphere should find "
            f"< {CONV_BASE_WITH_BOUNDS}, got {fit.item()}"
        )
        assert fit.item() >= 0.0, "Sphere function cannot be negative"
        assert sol.shape == (dim,)
        assert sol.device.type == device.type
        assert sol.dtype == default_dtype

    def test_generation_counter_increments(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = RandomSearch(
            dim=2,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=0,
        )
        assert opt._generation == 0
        opt.ask()
        assert opt._generation == 1
        opt.ask()
        assert opt._generation == 2

    def test_candidates_within_bounds(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = RandomSearch(
            dim=3,
            bounds=(-2.0, 3.0),
            pop_size=100,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        for _ in range(5):
            candidates = opt.ask()
            assert torch.all(candidates >= opt.lb), "Candidates below lower bound"
            assert torch.all(candidates <= opt.ub), "Candidates above upper bound"

    def test_population_stored(self, device: torch.device, default_dtype: torch.dtype) -> None:
        opt = RandomSearch(
            dim=2,
            bounds=1.0,
            pop_size=8,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        candidates = opt.ask()
        assert torch.equal(opt.population, candidates)

        # Modify candidates in-place and verify population is unchanged
        # (proving they are independent copies)
        saved_pop = opt.population.clone()
        candidates.fill_(999.0)
        assert torch.equal(opt.population, saved_pop), (
            "opt.population changed after modifying candidates -- not independent copies"
        )

    def test_fitness_stored_after_tell(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        opt = RandomSearch(
            dim=2,
            bounds=1.0,
            pop_size=4,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        candidates = opt.ask()
        f = (candidates**2).sum(dim=-1)
        opt.tell(candidates, f)
        assert torch.equal(opt.fitness, f)

    def test_clamp_to_bounds_compatible(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """Verify that candidates can be clamped using the utility function."""
        opt = RandomSearch(
            dim=3,
            bounds=1.0,
            pop_size=10,
            device=device,
            dtype=default_dtype,
            seed=42,
        )
        candidates = opt.ask()
        clamped = clamp_to_bounds(candidates, opt.lb, opt.ub)
        assert clamped.shape == candidates.shape
        assert clamped.device.type == device.type
