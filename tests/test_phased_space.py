"""Tests for PhasedDFO space= and initial_points= parameters."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import CONV_SPACE_SIMPLE_CLOSURE, SMOKE_F_INIT_CMAES
from tests.conftest import best_float_dtype
from torch_dfo.phased import PhasedDFO
from torch_dfo.space import Float, SearchSpace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space(n: int = 3) -> SearchSpace:
    """Return a simple SearchSpace with *n* Float parameters in [0, 1]."""
    return SearchSpace([Float(f"x{i}", 0.0, 1.0) for i in range(n)])


# ---------------------------------------------------------------------------
# TestPhasedDFOWithSpace
# ---------------------------------------------------------------------------


class TestPhasedDFOWithSpace:
    def test_space_sets_dim_and_bounds(self, device: torch.device) -> None:
        """When space is provided, dim == space.dim and bounds are [0, 1]."""
        dtype = best_float_dtype(device)
        space = _make_space(3)
        opt = PhasedDFO(space=space, budget=500, device=device, dtype=dtype)
        assert opt.dim == 3
        assert torch.all(opt.lb == 0.0)
        assert torch.all(opt.ub == 1.0)

    def test_space_conflict_with_dim_raises(self) -> None:
        """Providing both dim=5 and space with dim=3 raises ValueError."""
        space = _make_space(3)
        with pytest.raises(ValueError, match=r"conflicts with space\.dim"):
            PhasedDFO(dim=5, space=space, budget=500)

    def test_no_dim_no_space_raises(self) -> None:
        """Calling PhasedDFO() with neither dim nor space raises ValueError."""
        with pytest.raises(ValueError, match="Either dim or space must be provided"):
            PhasedDFO(budget=500)

    def test_space_property_returns_space(self, device: torch.device) -> None:
        """opt.space is the exact SearchSpace object passed in."""
        dtype = best_float_dtype(device)
        space = _make_space(4)
        opt = PhasedDFO(space=space, budget=500, device=device, dtype=dtype)
        assert opt.space is space

    def test_dim_only_still_works(self, device: torch.device) -> None:
        """Legacy positional dim still works (no regression)."""
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=5, bounds=5.0, budget=500, device=device, dtype=dtype)
        assert opt.dim == 5
        assert opt.space is None

    def test_ask_returns_unit_cube_when_space_set(self) -> None:
        """ask() returns raw [0, 1] tensors even when space is set."""
        space = SearchSpace([Float("x", -5.0, 5.0), Float("y", 0.0, 1.0)])
        opt = PhasedDFO(space=space, budget=100, seed=42, device="cpu", dtype=torch.float64)
        candidates = opt.ask()
        assert candidates.ndim == 2
        assert candidates.shape[1] == 2
        assert candidates.min() >= -1e-6
        assert candidates.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TestPhasedDFOWarmStart
# ---------------------------------------------------------------------------


class TestPhasedDFOWarmStart:
    def test_initial_points_tensor_seeds_population(self, device: torch.device) -> None:
        """Zero initial_points tensor → first rows of first ask() are zeros."""
        dtype = best_float_dtype(device)
        dim = 3
        pop_size = 10
        # Build zero tensor in [0, 1] — valid for the (0.0, 1.0) bounds SHADE uses.
        zeros = torch.zeros(pop_size, dim, dtype=dtype)
        opt = PhasedDFO(
            dim=dim,
            bounds=(0.0, 1.0),
            budget=1000,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            initial_points=zeros,
        )
        candidates = opt.ask()
        # The first pop_size rows should all be zero (warm-start applied).
        assert candidates.shape == (pop_size, dim)
        assert torch.all(candidates[:pop_size] == 0.0)

    def test_initial_points_too_many_truncated(self, device: torch.device) -> None:
        """N > pop_size: only the first pop_size rows are seeded."""
        dtype = best_float_dtype(device)
        dim = 3
        pop_size = 5
        # Provide more points than pop_size.
        pts = torch.zeros(pop_size + 4, dim, dtype=dtype)
        # Should not raise; excess rows are ignored.
        opt = PhasedDFO(
            dim=dim,
            bounds=(0.0, 1.0),
            budget=500,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            initial_points=pts,
        )
        candidates = opt.ask()
        assert candidates.shape[0] == pop_size

    def test_initial_points_too_few_padded(self, device: torch.device) -> None:
        """N < pop_size: remaining rows are NOT all zero (randomly initialised)."""
        dtype = best_float_dtype(device)
        dim = 3
        pop_size = 10
        n_seed = 3
        pts = torch.zeros(n_seed, dim, dtype=dtype)
        opt = PhasedDFO(
            dim=dim,
            bounds=(0.0, 1.0),
            budget=1000,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            initial_points=pts,
            seed=42,
        )
        candidates = opt.ask()
        assert candidates.shape == (pop_size, dim)
        # The seeded rows are zero.
        assert torch.all(candidates[:n_seed] == 0.0)
        # The remaining rows are not all zero (random fill).
        assert not torch.all(candidates[n_seed:] == 0.0)

    def test_initial_points_list_dict_requires_space(self) -> None:
        """Passing list[dict] without space raises ValueError."""
        with pytest.raises(ValueError, match="requires space to be set"):
            PhasedDFO(
                dim=3,
                bounds=(0.0, 1.0),
                budget=500,
                initial_points=[{"x0": 0.5, "x1": 0.5, "x2": 0.5}],
            )

    def test_initial_points_shape_mismatch_raises(self) -> None:
        """Tensor with wrong second dim raises ValueError."""
        dim = 3
        wrong = torch.zeros(5, dim + 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="initial_points must have shape"):
            PhasedDFO(dim=dim, bounds=(0.0, 1.0), budget=500, device="cpu", initial_points=wrong)

    def test_initial_points_list_dict_with_space_encodes(self) -> None:
        """list[dict] initial_points with space= encodes and optimizer runs without error.

        The optimizer works in encoded [0, 1] space internally; when space= is set,
        the fitness function receives decoded list[dict] configs.  We verify that
        construction succeeds (encoding happened) and the optimizer completes.
        """
        space = SearchSpace([Float("x", -5.0, 5.0)])
        initial = [{"x": 0.0}]  # encodes to 0.5 in [0, 1] → decoded x = 0.0
        opt = PhasedDFO(
            space=space,
            budget=50,
            seed=42,
            initial_points=initial,
            device="cpu",
            dtype=torch.float64,
        )
        # Fitness function receives decoded list[dict]; minimize |x| (optimum at x=0.0).
        _, best_f = opt.optimize(  # type: ignore[arg-type]
            lambda cfgs: torch.tensor([abs(float(c["x"])) for c in cfgs], dtype=torch.float64),
        )
        # With x=0.0 seeded the best fitness should be ~0.
        assert best_f.item() < CONV_SPACE_SIMPLE_CLOSURE

    def test_initial_points_empty_list_raises(self) -> None:
        """Empty list[dict] raises ValueError."""
        space = SearchSpace([Float("x", -5.0, 5.0)])
        with pytest.raises(ValueError, match="must not be empty"):
            PhasedDFO(space=space, budget=50, initial_points=[])

    def test_initial_points_out_of_bounds_clamped(self, device: torch.device) -> None:
        """Values > 1.0 passed to space=(0,1) are clamped to 1.0."""
        dtype = best_float_dtype(device)
        space = _make_space(3)
        pop_size = 4
        # All-ones-plus-epsilon: should be clamped to exactly 1.0.
        pts = torch.full((pop_size, 3), 2.0, dtype=dtype)
        opt = PhasedDFO(
            space=space,
            budget=500,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            initial_points=pts,
        )
        candidates = opt.ask()
        assert candidates.shape == (pop_size, 3)
        assert torch.all(candidates <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# TestOptimizeDecoding
# ---------------------------------------------------------------------------


class TestOptimizeDecoding:
    def test_optimize_with_space_decodes_to_configs(self) -> None:
        """optimize() passes list[dict] to fitness_fn when space is set."""
        space = SearchSpace([Float("x", -5.0, 5.0), Float("y", 0.0, 1.0)])
        received_configs: list[dict[str, object]] = []

        def fitness(cfgs: list[dict[str, object]]) -> torch.Tensor:
            received_configs.extend(cfgs)
            return torch.tensor(
                [abs(float(c["x"])) + float(c["y"]) for c in cfgs],
                dtype=torch.float64,
            )

        opt = PhasedDFO(space=space, budget=50, seed=42, device="cpu", dtype=torch.float64)
        opt.optimize(fitness)  # type: ignore[arg-type]
        assert len(received_configs) > 0
        for cfg in received_configs:
            assert "x" in cfg and "y" in cfg
            assert -5.0 <= float(cfg["x"]) <= 5.0
            assert 0.0 <= float(cfg["y"]) <= 1.0

    def test_optimize_without_space_passes_tensor(self) -> None:
        """optimize() passes raw tensors when no space is set."""
        received_tensors: list[torch.Tensor] = []

        def fitness(x: torch.Tensor) -> torch.Tensor:
            received_tensors.append(x)
            return x.sum(dim=-1)

        opt = PhasedDFO(
            dim=2,
            bounds=(-1.0, 1.0),
            budget=50,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        opt.optimize(fitness)
        assert len(received_tensors) > 0
        assert isinstance(received_tensors[0], torch.Tensor)


# ---------------------------------------------------------------------------
# TestDoneAndReset
# ---------------------------------------------------------------------------


def _make_opt(budget: int = 100) -> PhasedDFO:
    return PhasedDFO(
        dim=2,
        bounds=(-1.0, 1.0),
        budget=budget,
        seed=42,
        device="cpu",
        dtype=torch.float64,
    )


def _sum_fitness(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=-1)


class TestDoneAndReset:
    def test_done_false_initially(self) -> None:
        assert not _make_opt().done

    def test_done_true_after_budget_exhausted(self) -> None:
        opt = _make_opt(budget=50)
        opt.optimize(_sum_fitness)
        assert opt.done

    def test_done_true_at_phase_3(self) -> None:
        opt = _make_opt()
        opt._phase = 3  # force phase; 3 >= 2 so done must be True
        assert opt.done

    def test_done_true_at_phase_2(self) -> None:
        """Done returns True at phase 2 — Polish is running, ask/tell has nothing left."""
        opt = PhasedDFO(
            dim=2,
            bounds=(-1.0, 1.0),
            budget=100,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        opt._phase = 2  # polish phase — ask/tell has nothing left
        assert opt.done

    def test_reset_clears_state(self) -> None:
        opt = _make_opt()
        opt.optimize(_sum_fitness)
        assert opt.done
        opt.reset()
        assert not opt.done
        assert opt._fe_count == 0
        assert opt._phase == 0

    def test_re_optimization_works_after_reset(self) -> None:
        opt = _make_opt()
        opt.optimize(_sum_fitness)
        opt.reset()
        best_x, _best_f = opt.optimize(_sum_fitness)
        assert best_x.shape == (2,)

    def test_budget_preserved_across_reset(self) -> None:
        opt = _make_opt()
        opt.reset()
        assert opt._budget == 100


# ---------------------------------------------------------------------------
# TestAskTellLoop
# ---------------------------------------------------------------------------


class TestAskTellLoop:
    """Tests for the public ask/tell interface with done property."""

    def test_loop_terminates_with_done(self) -> None:
        """While not opt.done loop always terminates within budget.

        done returns True at phase >= 2, so the loop exits before Polish phase
        and ask() never returns an empty batch during a well-formed loop.
        """
        opt = PhasedDFO(
            dim=2,
            bounds=(-5.0, 5.0),
            budget=100,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        # Budget=100, default pop_size≈12 for dim=2 → ~9 iters to exhaust DE.
        # Bound = ceil(budget/pop_size) + margin for any internal restart/probe
        # overhead; 30 is ~3x the nominal iteration count — tight enough to
        # catch runaway loops, loose enough to absorb occasional extra asks.
        count = 0
        max_iters = (opt._budget // opt.pop_size) + 20  # ≈30 here
        while not opt.done:
            candidates = opt.ask()
            f_vals = candidates.pow(2).sum(dim=-1)
            opt.tell(candidates, f_vals)
            count += 1
            assert count <= max_iters, (
                f"Loop did not terminate within {max_iters} iters (saw {count})"
            )
        assert opt.done

    def test_shapes_consistent_across_iterations(self) -> None:
        """ask() returns (pop_size, dim) tensor; tell() accepts matching fitness."""
        opt = PhasedDFO(
            dim=3,
            bounds=(-1.0, 1.0),
            budget=80,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        iteration_count = 0
        while not opt.done and iteration_count < 5:
            candidates = opt.ask()
            assert candidates.ndim == 2
            assert candidates.shape[1] == 3
            f_vals = candidates.pow(2).sum(dim=-1)
            assert f_vals.shape == (candidates.shape[0],)
            opt.tell(candidates, f_vals)
            iteration_count += 1

    def test_best_improves_or_stays_over_loop(self) -> None:
        """best_fitness should be non-increasing across tell() calls (minimize).

        done fires at phase >= 2, so Polish is never reached in a plain ask/tell
        loop and all batches returned by ask() are non-empty.
        """
        opt = PhasedDFO(
            dim=2,
            bounds=(-5.0, 5.0),
            budget=200,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        prev_best = float("inf")
        while not opt.done:
            candidates = opt.ask()
            f_vals = candidates.pow(2).sum(dim=-1)
            opt.tell(candidates, f_vals)
            current_best = opt.best_fitness.item()
            assert current_best <= prev_best + 1e-10  # best_fitness non-increasing
            prev_best = current_best
        # At least one tell() call must have occurred and improved from inf.
        assert prev_best < SMOKE_F_INIT_CMAES

    def test_ask_tell_with_space(self) -> None:
        """ask/tell loop works with SearchSpace.

        ask() returns raw [0,1] tensors (not decoded configs); tell() accepts them.
        """
        space = SearchSpace([Float("x", -5.0, 5.0), Float("y", 0.0, 1.0)])
        opt = PhasedDFO(
            space=space,
            budget=80,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        # ask() returns raw [0, 1] tensors (not decoded configs); decode only happens in optimize()
        iteration_count = 0
        while not opt.done and iteration_count < 3:
            candidates = opt.ask()
            assert candidates.ndim == 2
            assert candidates.shape[1] == 2
            assert candidates.min() >= -1e-6  # in [0, 1]
            assert candidates.max() <= 1.0 + 1e-6
            # tell() also takes raw tensors
            f_vals = candidates.pow(2).sum(dim=-1)
            opt.tell(candidates, f_vals)
            iteration_count += 1

    def test_phase_stays_below_2_in_ask_tell_loop(self) -> None:
        """A pure ask/tell loop never calls ask() when phase >= 2.

        tell() may set _phase=2 as a side-effect of consuming the last budget
        slot, but done then returns True and the loop exits before ask() is
        called again.  We therefore assert the phase at the top of the loop
        (i.e. immediately after the done check passes), which is the only point
        where ask() is about to be called.
        """
        opt = PhasedDFO(
            dim=2,
            bounds=(-5.0, 5.0),
            budget=200,
            seed=42,
            device="cpu",
            dtype=torch.float64,
        )
        fitness = lambda x: x.pow(2).sum(dim=-1)  # noqa: E731
        while not opt.done:
            # If done is False, phase must still be < 2 (ask() is safe to call)
            assert opt._phase < 2, f"done was False but phase={opt._phase} at fe={opt._fe_count}"
            candidates = opt.ask()
            opt.tell(candidates, fitness(candidates))
        assert opt.done
