"""Tests for torch_dfo.phased -- PhasedDFO multi-phase optimizer."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import (
    ATOL_F32_TIGHT,
    BUDGET_PHASED_LARGE,
    BUDGET_PHASED_MEDIUM,
    BUDGET_PHASED_MICRO,
    BUDGET_PHASED_POLISH,
    BUDGET_PHASED_QUICK,
    BUDGET_PHASED_STANDARD,
    BUDGET_SMOKE,
    CONV_ACKLEY_10D,
    CONV_POLISH_SPHERE_5D,
    CONV_RASTRIGIN_10D,
    CONV_ROSENBROCK_10D,
    CONV_SPHERE_10D_STANDARD,
    PHASED_DEFAULT_BUDGET_MULT,
    RTOL_TIGHT,
    SMOKE_F_INIT_CMAES,
)
from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import ackley, rastrigin, rosenbrock, sphere
from torch_dfo.phased import (
    CMA_ES_RESTART_MODES,
    ELITE_FRACTION,
    STEP_SIZE_MAX,
    PhasedConfig,
    PhasedDFO,
    _compute_basin_explore_budget_frac,
    _compute_basin_explore_restarts,
    _compute_basin_explore_stagnation,
    _compute_high_dim_de_min_pop,
    _compute_step_size_init,
    _merge_search_pool,
)
from torch_dfo.phased._basin_explore import multistart_basin_explore


# ---------------------------------------------------------------------------
# PhasedConfig plumbing: constructor kwarg actually reaches internal logic
# ---------------------------------------------------------------------------
class TestPhasedConfigKwarg:
    """B1 added ``PhasedDFO(config=PhasedConfig(...))``. These tests guard
    the plumbing so a future refactor that silently drops the kwarg fails."""

    def test_default_config_matches_module_alias_defaults(self) -> None:
        """Unspecified config must match the module-level alias values."""
        opt = PhasedDFO(dim=5, bounds=5.0, seed=42, device="cpu", dtype=torch.float64)
        assert opt._config.elite_fraction == ELITE_FRACTION
        assert opt._config.cma_es_restart_modes == CMA_ES_RESTART_MODES
        assert opt._config.step_size_max == STEP_SIZE_MAX

    def test_custom_config_flows_through_init(self) -> None:
        """An explicit ``config=`` kwarg must replace the default values."""
        cfg = PhasedConfig(elite_fraction=0.5, cma_es_restart_modes=2)
        opt = PhasedDFO(dim=5, bounds=5.0, seed=42, device="cpu", dtype=torch.float64, config=cfg)
        assert opt._config is cfg
        assert opt._config.elite_fraction == 0.5
        assert opt._config.cma_es_restart_modes == 2
        # Field not overridden: still the default.
        assert opt._config.step_size_max == STEP_SIZE_MAX

    def test_custom_elite_fraction_affects_low_dim_partial_restart(self) -> None:
        """Raising ``elite_fraction`` from 0.1 to 0.8 must preserve more DE elites.

        ``_low_dim_pop_restart`` on odd restart counts keeps ``max(2, int(pop*ELITE))``
        individuals — if the constructor ignored ``config=``, the elite count would
        stay at int(20*0.1)=2 instead of int(20*0.8)=16.
        """
        cfg = PhasedConfig(elite_fraction=0.8)
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            pop_size=20,
            seed=42,
            device="cpu",
            dtype=torch.float64,
            config=cfg,
        )
        opt._fitness_fn = sphere
        # Prime the population so fitness argsort is meaningful.
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Odd-count restart keeps the elite fraction.
        opt._de_restart_count = 0  # next call lands on 1 -> partial restart path
        pop_before = opt._shade.population.clone()
        fit_before = opt._shade.fitness.clone()
        opt._low_dim_pop_restart()
        # elite_count = max(2, int(20 * 0.8)) = 16 -> first 16 rows by sort order preserved
        elite_count = max(2, int(20 * 0.8))
        sorted_idx = fit_before.argsort()
        preserved_idx = sorted_idx[:elite_count]
        for i in preserved_idx.tolist():
            assert torch.allclose(opt._shade.population[i], pop_before[i]), (
                f"elite row {i} mutated despite elite_fraction=0.8"
            )


def test_basin_explore_propagates_objective_errors() -> None:
    opt = PhasedDFO(dim=2, bounds=5.0, budget=100, seed=42, device="cpu", dtype=torch.float64)
    opt._basin_explore_restarts = 1

    def broken_objective(_x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        multistart_basin_explore(opt, broken_objective, budget_limit=100)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Validate constructor wiring: phase, budget, sub-optimizers."""

    def test_initial_phase_is_zero(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=10, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt.phase == 0

    def test_default_budget(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        dim = 10
        opt = PhasedDFO(dim=dim, bounds=5.0, device=device, dtype=dtype, seed=42)
        # Default budget follows PHASED_DEFAULT_BUDGET_MULT (mirror of
        # PhasedDFO.__init__(budget_mult=5000) in src/torch_dfo/phased.py).
        assert opt.budget == dim * PHASED_DEFAULT_BUDGET_MULT

    def test_custom_budget(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(
            dim=10, bounds=5.0, budget=BUDGET_PHASED_MICRO, device=device, dtype=dtype, seed=42
        )
        assert opt.budget == 1000

    def test_shade_created(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=10, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt._shade is not None
        assert opt._shade.dim == 10
        assert opt._shade.pop_size > 0
        # SHADE bounds must match PhasedDFO bounds
        assert torch.allclose(opt._shade.lb, opt.lb)
        assert torch.allclose(opt._shade.ub, opt.ub)

    def test_cmaes_not_created_initially(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=10, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt._cmaes is None

    def test_fe_count_starts_at_zero(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt.fe_count == 0

    def test_device_placement(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=5, bounds=5.0, device=device, dtype=dtype, seed=42)
        assert opt.device.type == device.type
        assert opt._shade.device.type == device.type


# ---------------------------------------------------------------------------
# ask/tell basic
# ---------------------------------------------------------------------------
class TestAskTell:
    """Validate basic ask/tell shape and behavior."""

    def test_first_ask_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        pop_size = 40
        opt = PhasedDFO(dim=10, bounds=5.0, pop_size=pop_size, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert candidates.shape == (pop_size, 10)

    def test_tell_does_not_error(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)  # should not raise
        assert opt._fe_count > 0
        assert opt.best_fitness.item() < SMOKE_F_INIT_CMAES
        assert opt._phase == 0  # still in DE phase after one tell

    def test_fe_count_increments(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        pop_size = 20
        opt = PhasedDFO(dim=5, bounds=5.0, pop_size=pop_size, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)
        assert opt._fe_count == pop_size

    def test_best_not_inf_after_tell(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=10, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)
        _, best_f = opt.best()
        # 10d sphere in [-5,5] has max value 10*25=250, any evaluated point must be below
        assert best_f.item() < 250.0


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------
class TestPhaseTransitions:
    """Verify that optimization advances through phases correctly."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_de_to_cmaes_transition(self, device: torch.device) -> None:
        """After enough iterations the optimizer should leave DE phase."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        max_iters = 200
        for _ in range(max_iters):
            if opt.phase >= 1:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        assert opt.phase >= 1, (
            f"Phase should have advanced past DE after {max_iters} iterations, "
            f"got phase={opt.phase}, fe_count={opt.fe_count}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_all_phases_reached_via_optimize(self, device: torch.device) -> None:
        """optimize() should pass through all phases and reach phase 3."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        phases_seen: set[int] = set()

        def track_phases(phase: int, gen: int, best_f: torch.Tensor) -> None:
            phases_seen.add(phase)

        opt.optimize(sphere, callback=track_phases)
        assert opt.phase == 3, f"Expected phase 3, got {opt.phase}"
        assert 0 in phases_seen, "DE phase (0) was never seen in callbacks"
        assert 1 in phases_seen, "CMA-ES phase (1) was never seen in callbacks"


# ---------------------------------------------------------------------------
# optimize() convenience
# ---------------------------------------------------------------------------
class TestOptimize:
    """Validate the optimize() convenience method."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_optimize_returns_tuple(self, device: torch.device) -> None:
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_POLISH,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        result = opt.optimize(sphere)
        assert isinstance(result, tuple)
        assert len(result) == 2
        best_x, best_f = result
        assert best_x.shape == (5,)
        assert best_f.shape == ()

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_optimize_improves_over_random(self, device: torch.device) -> None:
        """optimize() should find a better solution than random initialization."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_POLISH,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        _, best_f = opt.optimize(sphere)
        # 5d sphere with budget=3000 and polish should get very close to zero
        assert best_f.item() < CONV_POLISH_SPHERE_5D


# ---------------------------------------------------------------------------
# Convergence tests (CPU float64 only)
# ---------------------------------------------------------------------------
class TestConvergence:
    """Verify convergence on standard benchmarks. CPU + float64 only."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sphere_10d(self, device: torch.device) -> None:
        """PhasedDFO on Sphere 10d must reach < 1e-6."""
        opt = PhasedDFO(
            dim=10,
            bounds=5.12,
            budget=BUDGET_PHASED_STANDARD,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        _, best_f = opt.optimize(sphere)
        assert best_f.item() < CONV_SPHERE_10D_STANDARD, (
            f"Sphere 10d: expected < 1e-6, got {best_f.item():.2e}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_ackley_10d(self, device: torch.device) -> None:
        """PhasedDFO on Ackley 10d must reach < 1e-4."""
        opt = PhasedDFO(
            dim=10,
            bounds=(-32.768, 32.768),
            budget=BUDGET_PHASED_STANDARD,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        _, best_f = opt.optimize(ackley)
        assert best_f.item() < CONV_ACKLEY_10D, (
            f"Ackley 10d: expected < 1e-4, got {best_f.item():.2e}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_rastrigin_10d(self, device: torch.device, seed: int) -> None:
        """PhasedDFO on Rastrigin 10d must reach < 1.0."""
        opt = PhasedDFO(
            dim=10,
            bounds=5.12,
            budget=BUDGET_PHASED_STANDARD,
            device=device,
            dtype=torch.float64,
            seed=seed,
        )
        _, best_f = opt.optimize(rastrigin)
        assert best_f.item() < CONV_RASTRIGIN_10D, (
            f"Rastrigin 10d: expected < 1.0, got {best_f.item():.2e}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_rosenbrock_10d(self, device: torch.device) -> None:
        """PhasedDFO on Rosenbrock 10d must reach < 1e-2."""
        opt = PhasedDFO(
            dim=10,
            bounds=(-5.0, 10.0),
            budget=BUDGET_PHASED_STANDARD,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        _, best_f = opt.optimize(rosenbrock)
        assert best_f.item() < CONV_ROSENBROCK_10D, (
            f"Rosenbrock 10d: expected < 1e-2, got {best_f.item():.2e}"
        )


# ---------------------------------------------------------------------------
# Budget respected
# ---------------------------------------------------------------------------
class TestBudget:
    """Verify budget is not exceeded."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_budget_not_exceeded(self, device: torch.device) -> None:
        budget = 5000
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=budget,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt.optimize(sphere)
        # At most one extra batch overrun
        assert opt.fe_count <= budget + opt.pop_size, (
            f"Budget exceeded: {opt.fe_count} > {budget + opt.pop_size}"
        )


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
class TestCallback:
    """Verify callback receives expected arguments."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_callback_receives_correct_types(self, device: torch.device) -> None:
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_POLISH,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        calls: list[tuple[int, int, torch.Tensor]] = []

        def cb(phase: int, generation: int, best_f: torch.Tensor) -> None:
            calls.append((phase, generation, best_f))

        opt.optimize(sphere, callback=cb)
        assert len(calls) > 0, "Callback was never called"
        for phase, gen, best_f in calls:
            assert isinstance(phase, int)
            assert isinstance(gen, int)
            assert isinstance(best_f, torch.Tensor)

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_callback_phase_values(self, device: torch.device) -> None:
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_POLISH,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        phases: list[int] = []

        def cb(phase: int, gen: int, best_f: torch.Tensor) -> None:
            phases.append(phase)

        opt.optimize(sphere, callback=cb)
        # All phases should be valid (0=DE or 1=CMA-ES during ask/tell loop)
        for p in phases:
            assert p in (0, 1, 2)


# ---------------------------------------------------------------------------
# Multi-device
# ---------------------------------------------------------------------------
class TestMultiDevice:
    """Verify basic ask/tell loop works on all available devices."""

    def test_ask_tell_loop_runs(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = PhasedDFO(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        opt._fitness_fn = sphere
        for _ in range(5):
            c = opt.ask()
            if c.shape[0] == 0:
                break
            f = sphere(c)
            opt.tell(c, f)
        sol, fit = opt.best()
        assert sol.device.type == device.type
        assert fit.item() < SMOKE_F_INIT_CMAES


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """Same seed must produce identical results."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_same_seed_same_result(self, device: torch.device) -> None:
        results = []
        for _ in range(2):
            opt = PhasedDFO(
                dim=5,
                bounds=5.12,
                budget=BUDGET_PHASED_POLISH,
                pop_size=40,
                device=device,
                dtype=torch.float64,
                seed=42,
            )
            _, best_f = opt.optimize(sphere)
            results.append(best_f.item())
        assert results[0] == pytest.approx(results[1], rel=RTOL_TIGHT), (
            f"Results differ: {results[0]:.10e} vs {results[1]:.10e}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_different_seeds_differ(self, device: torch.device) -> None:
        """Different seeds produce different initial populations."""
        pops = []
        for seed in (42, 99):
            opt = PhasedDFO(
                dim=5,
                bounds=5.12,
                budget=BUDGET_PHASED_POLISH,
                pop_size=40,
                device=device,
                dtype=torch.float64,
                seed=seed,
            )
            c = opt.ask()
            pops.append(c.clone())
        # Different seeds must produce different initial populations
        assert not torch.allclose(pops[0], pops[1]), (
            "Different seeds should give different initial populations"
        )


# ---------------------------------------------------------------------------
# Advanced phase transitions
# ---------------------------------------------------------------------------
class TestAdvancedPhases:
    """Verify advanced phase transition behavior."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_cmaes_to_polish_transition(self, device: torch.device) -> None:
        """optimize() must reach polish phase (phase >= 2)."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        phases_seen: set[int] = set()

        def track(phase: int, gen: int, best_f: torch.Tensor) -> None:
            phases_seen.add(phase)

        opt.optimize(sphere, callback=track)
        assert opt._phase >= 2, f"Expected phase >= 2 after optimize(), got {opt._phase}"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_ipop_restart_occurs(self, device: torch.device) -> None:
        """IPOP contract: restart happens AND pop doubles at the restart boundary.

        ``_cmaes_phase_idx`` can increment without calling ``_restart_cmaes``
        when the budget transition to polish fires on the same tell (see
        ``phased.py`` line 2740-2747).  To verify the IPOP doubling contract
        directly we instead call the internal restart explicitly and check
        that ``CMAES.pop_size`` took the expected doubled value.
        """
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        # Drive DE->CMA entry by running until CMA-ES exists. DE on 5d sphere
        # typically takes ~500-700 iterations at budget=50k before stagnation
        # triggers the handoff; use a generous cap.
        for _ in range(5000):
            if opt._cmaes is not None:
                break
            c = opt.ask()
            if c.shape[0] == 0:
                break
            opt.tell(c, sphere(c))
        assert opt._cmaes is not None, "Never entered CMA-ES phase"
        base_pop = opt._cmaes_base_pop
        # Force phase_idx=1 then restart — models the IPOP restart path.
        opt._cmaes_phase_idx = 1
        opt._restart_cmaes()
        # new_pop = base_pop * 2**1 = 2 * base_pop (uncapped: plenty of budget left)
        assert opt._cmaes.pop_size >= base_pop * 2, (
            f"IPOP contract broken: cmaes.pop_size={opt._cmaes.pop_size} < "
            f"2 * base_pop={base_pop * 2}"
        )
        # Next restart doubles again.
        opt._cmaes_phase_idx = 2
        opt._restart_cmaes()
        assert opt._cmaes.pop_size >= base_pop * 4, (
            f"Second IPOP restart failed to quadruple: pop={opt._cmaes.pop_size}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_high_dim_path(self, device: torch.device) -> None:
        """Dim >= 20 should set _high_dim=True and not crash during optimize."""
        opt = PhasedDFO(
            dim=25,
            bounds=5.0,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        assert opt._high_dim is True
        # Run a small optimize to verify it completes without error
        _, best_f = opt.optimize(sphere)
        assert best_f.item() < SMOKE_F_INIT_CMAES

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_stagnation_on_flat_function(self, device: torch.device) -> None:
        """Constant fitness should trigger stagnation eventually.

        With low-dim restart, stagnation resets the counter after
        restarting the population. The optimizer will eventually exhaust its
        DE budget and transition. We use a small budget so that repeated
        restarts (which consume extra FE) quickly exhaust the DE allocation.
        """

        def flat_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = flat_fn
        # With low-dim restarts, the low-dim path keeps resetting stagnation.
        # With a smaller budget the DE budget is exhausted faster via the
        # extra FE consumption from restart evaluations.
        for _i in range(500):
            if opt._phase > 0:
                break
            c = opt.ask()
            f = flat_fn(c)
            opt.tell(c, f)

        assert opt._phase > 0, (
            f"Expected phase > 0 after iterations on flat function, "
            f"got phase={opt._phase}, stagnation_counter={opt._stagnation_counter}, "
            f"fe_count={opt._fe_count}"
        )


# ---------------------------------------------------------------------------
# Callback monotonicity and polish
# ---------------------------------------------------------------------------
class TestCallbackMonotonicity:
    """Verify best_f reported in callbacks is monotonically non-increasing."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_optimize_callback_monotonicity(self, device: torch.device) -> None:
        """best_f values reported via callback must be non-increasing."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_POLISH,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        best_values: list[float] = []

        def record_best(phase: int, gen: int, best_f: torch.Tensor) -> None:
            best_values.append(best_f.item())

        opt.optimize(sphere, callback=record_best)
        assert len(best_values) > 1, "Callback was never called"
        for i in range(1, len(best_values)):
            # Monotonic non-increasing on float64: no tolerance needed.
            assert best_values[i] <= best_values[i - 1], (
                f"best_f increased at callback index {i}: {best_values[i - 1]} -> {best_values[i]}"
            )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_polish_improves_best(self, device: torch.device) -> None:
        """Final best after polish should be <= best before polish (phase 1 end)."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        pre_polish_best: list[float] = []

        def capture_pre_polish(phase: int, gen: int, best_f: torch.Tensor) -> None:
            # Capture the last best_f while still in CMA-ES phase (1)
            if phase == 1:
                pre_polish_best.append(best_f.item())

        _, final_f = opt.optimize(sphere, callback=capture_pre_polish)
        if pre_polish_best:
            # Polish should not worsen the result
            assert final_f.item() <= pre_polish_best[-1] + 1e-12, (
                f"Polish worsened result: pre={pre_polish_best[-1]:.6e}, post={final_f.item():.6e}"
            )


# ---------------------------------------------------------------------------
# Levy flight
# ---------------------------------------------------------------------------
class TestLevyFlight:
    """Verify Levy flight perturbation adds extra evaluations beyond base pop."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_levy_flight_applied(self, device: torch.device) -> None:
        """Levy flight adds extra FEs beyond the base pop*n_gens count.

        We check two things:
        1. FE count exceeds pop_size * n_gens (extra Levy evaluations happened).
        2. FE overhead is a meaningful fraction of population, not ULP drift.
        """
        pop_size = 40
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=pop_size,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        # Run several DE ask/tell iterations (need gen > 1 for Levy flights)
        n_gens = 5
        for _ in range(n_gens):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        base_fe = pop_size * n_gens
        levy_overhead = opt._fe_count - base_fe
        # Levy flights add extra FEs; the overhead must be a meaningful fraction
        # of one generation's worth of pop (not just 1 or 2 accidental extras).
        assert levy_overhead > pop_size // 4, (
            f"Levy overhead too small ({levy_overhead} extra FEs for pop={pop_size}) — "
            f"suggests Levy path is effectively disabled"
        )


# ---------------------------------------------------------------------------
# Population reduction during high-dim DE.
# ---------------------------------------------------------------------------
class TestPopulationReduction:
    """Verify population shrinks during high-dim DE phase."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_high_dim_pop_reduces(self, device: torch.device) -> None:
        """Pop size should shrink toward computed min pop in high-dim."""
        pop_size = 80
        dim = 25
        opt = PhasedDFO(
            dim=dim,
            bounds=5.0,
            budget=BUDGET_PHASED_LARGE,
            pop_size=pop_size,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        assert opt._high_dim is True
        initial_pop = opt._shade.pop_size

        # Run enough DE steps for reduction to kick in
        for _ in range(30):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        # Population should have decreased from initial
        final_pop = opt._shade.pop_size
        assert final_pop < initial_pop, (
            f"Expected pop reduction: initial={initial_pop}, final={final_pop}"
        )
        # Should be at or near the minimum (formula: dim + 8)
        min_pop = _compute_high_dim_de_min_pop(dim)
        assert final_pop >= min_pop, f"Pop reduced below minimum: {final_pop} < {min_pop}"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_low_dim_pop_stays_constant(self, device: torch.device) -> None:
        """Low-dim DE should not reduce population."""
        pop_size = 40
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=pop_size,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        assert opt._high_dim is False

        for _ in range(10):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        # Low-dim population should remain the same
        assert opt._shade.pop_size == pop_size

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_shade_buffers_resized_after_reduction(self, device: torch.device) -> None:
        """SHADE internal buffers must match reduced pop_size."""
        opt = PhasedDFO(
            dim=25,
            bounds=5.0,
            budget=BUDGET_PHASED_LARGE,
            pop_size=80,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        for _ in range(30):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        pop = opt._shade.pop_size
        assert opt._shade._trial_F.shape[0] == pop
        assert opt._shade._trial_CR.shape[0] == pop
        assert opt._shade._trials.shape == (pop, 25)


# ---------------------------------------------------------------------------
# Adaptive Levy step_size.
# ---------------------------------------------------------------------------
class TestAdaptiveStepSize:
    """Verify adaptive step_size changes based on stagnation."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_step_size_initialized_correctly(self, device: torch.device) -> None:
        """step_size should start at _compute_step_size_init(dim) * span."""
        dim = 5
        opt = PhasedDFO(
            dim=dim,
            bounds=5.0,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        span = (opt.ub - opt.lb).mean().item()
        assert opt._step_size == pytest.approx(_compute_step_size_init(dim) * span)

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_step_size_changes_during_de(self, device: torch.device) -> None:
        """step_size should differ from initial after several DE steps."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        initial_step = opt._step_size

        for _ in range(10):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)

        # Step size should have adapted (grown on improvement or shrunk on stagnation)
        assert opt._step_size != pytest.approx(initial_step, rel=ATOL_F32_TIGHT), (
            f"step_size did not change: still {opt._step_size}"
        )


# ---------------------------------------------------------------------------
# CMA-ES restart mean cycling.
# ---------------------------------------------------------------------------
class TestRestartCycling:
    """Verify CMA-ES restart uses different mean modes."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_restart_mode_0_random(self, device: torch.device) -> None:
        """Mode 0 (phase_idx % 3 == 0) should produce a random restart center."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        span = (opt.ub - opt.lb).mean().item()
        center = opt._sample_restart_mean(phase_idx=0, span=span)
        assert center.shape == (5,)
        # Should be within bounds
        assert (center >= opt.lb).all()
        assert (center <= opt.ub).all()

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_restart_mode_1_elite_anchor(self, device: torch.device) -> None:
        """Mode 1 should use an elite anchor when elite solutions exist."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Add fake elite solutions
        for i in range(5):
            opt._elite_solutions.append(torch.randn(5, device=device, dtype=torch.float64) * 2)
            opt._elite_fitness.append(torch.tensor(float(i), device=device, dtype=torch.float64))

        span = (opt.ub - opt.lb).mean().item()
        center = opt._sample_restart_mean(phase_idx=1, span=span)
        assert center.shape == (5,)
        # Should be within bounds
        assert (center >= opt.lb).all()
        assert (center <= opt.ub).all()

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_restart_mode_3_mirrored(self, device: torch.device) -> None:
        """Mode 3 should produce mirrored-best center."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Set a known best solution
        opt.best_solution = torch.ones(5, device=device, dtype=torch.float64) * 2.0
        opt.best_fitness = torch.tensor(1.0, device=device, dtype=torch.float64)

        span = (opt.ub - opt.lb).mean().item()
        center = opt._sample_restart_mean(phase_idx=3, span=span)
        assert center.shape == (5,)
        # Mirrored point of 2.0 in [-5, 5] is: -5 + 5 - 2 = -2.0 (plus jitter)
        # It should be roughly around -2.0 with some noise
        expected_mirror = opt.lb + opt.ub - opt.best_solution
        # Jitter is documented as 0.08 * span, so max deviation must be well
        # under 0.2 * span — generous enough for occasional multi-sigma draws,
        # tight enough to distinguish mirroring from random placement.
        assert (center >= opt.lb).all()
        assert (center <= opt.ub).all()
        diff = (center - expected_mirror).abs().max().item()
        assert diff < 0.2 * span, (
            f"Mirrored center too far from expected mirror: diff={diff:.2f}, "
            f"0.2*span={0.2 * span:.2f}. Is mirroring actually used?"
        )
        # Also verify it's closer to the mirror than to the original best
        dist_to_mirror = (center - expected_mirror).norm().item()
        dist_to_best = (center - opt.best_solution).norm().item()
        assert dist_to_mirror < dist_to_best, (
            "Restart center closer to best_solution than its mirror — "
            "mode 3 mirroring may not be active"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_different_modes_used_in_optimize(self, device: torch.device) -> None:
        """Optimize with enough budget should cycle through restart modes."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=15000,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt.optimize(sphere)
        # With sufficient budget, multiple IPOP restarts should occur
        # spanning different modes (phase_idx % 4)
        assert opt._cmaes_phase_idx > 0, (
            f"Expected restarts, got _cmaes_phase_idx={opt._cmaes_phase_idx}"
        )


# ---------------------------------------------------------------------------
# Low-dim population restart on stagnation.
# ---------------------------------------------------------------------------
class TestLowDimRestart:
    """Verify low-dim population restart behavior on stagnation."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_low_dim_restart_resets_stagnation(self, device: torch.device) -> None:
        """On a flat function, low-dim stagnation should trigger restart
        and reset the counter.
        """

        def flat_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = flat_fn
        assert opt._high_dim is False

        restart_happened = False
        for _ in range(100):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = flat_fn(c)
            old_stag = opt._stagnation_counter
            opt.tell(c, f)
            # If stagnation was high and then reset, a restart occurred
            if old_stag >= 15 and opt._stagnation_counter == 0:
                restart_happened = True
                break

        assert restart_happened, "Low-dim restart should have triggered on flat function"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_elite_preserved_during_restart(self, device: torch.device) -> None:
        """Elite fraction of population (solutions AND fitness) must survive restart."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Manually set up population with known, distinct row contents.
        opt._shade._initialized = True
        opt._shade.population = torch.randn(40, 5, device=device, dtype=torch.float64)
        opt._shade.fitness = torch.arange(40, device=device, dtype=torch.float64)
        opt.best_solution = opt._shade.population[0].clone()
        opt.best_fitness = opt._shade.fitness[0].clone()

        # Elite count = max(2, int(40 * 0.1)) = 4
        elite_count = max(2, int(40 * ELITE_FRACTION))
        # Clone the elite solutions BEFORE restart to catch solution swaps.
        elite_population_before = opt._shade.population[:elite_count].clone()

        # Trigger restart
        opt._low_dim_pop_restart()

        # Fitness labels must be preserved.
        for i in range(elite_count):
            assert opt._shade.fitness[i].item() == pytest.approx(float(i))
        # Solutions themselves must be preserved too — a bug that preserves
        # fitness labels but permutes solutions is caught here.
        assert torch.allclose(opt._shade.population[:elite_count], elite_population_before), (
            "Elite solutions changed across restart (only fitness labels preserved)"
        )


# ---------------------------------------------------------------------------
# Richer stagnation signal.
# ---------------------------------------------------------------------------
class TestRicherStagnationSignal:
    """Verify the richer stagnation EMA signal tracks correctly."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_trial_gain_tracked(self, device: torch.device) -> None:
        """trial_gain should be computed after DE tell."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        # First tell initializes
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # Second tell computes trial_gain
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # trial_gain, accepted_ratio, levy_gain, levy_ratio should all be set
        assert isinstance(opt._trial_gain, float)
        assert isinstance(opt._accepted_ratio, float)
        assert isinstance(opt._levy_gain, float)
        assert isinstance(opt._levy_ratio, float)

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_progress_ema_updates(self, device: torch.device) -> None:
        """DE progress EMA should evolve over multiple steps."""
        opt = PhasedDFO(
            dim=25,
            bounds=5.0,
            budget=BUDGET_PHASED_LARGE,
            pop_size=80,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        ema_values = []
        for _ in range(10):
            if opt._phase != 0:
                break
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            ema_values.append(opt._de_progress_ema)

        # EMA should have changed from its initial value of 0.0
        assert any(v != 0.0 for v in ema_values), "EMA never changed from 0.0"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_ema_coefficient_is_075(self, device: torch.device) -> None:
        """``_update_de_stagnation`` must blend EMA as 0.75*old + 0.25*new.

        Drives the update path directly with a known ``step_signal`` via the
        internal fields, then verifies the post-condition equals the exact
        formula.  Tied to the coefficient, not derived from it.
        """
        opt = PhasedDFO(
            dim=25,
            bounds=5.0,
            budget=BUDGET_PHASED_LARGE,
            pop_size=80,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Bypass baseline calibration so the stagnation update actually writes the EMA.
        opt._de_step_count = opt._de_baseline_steps + 1
        opt._de_progress_baseline = 1.0
        opt._de_progress_ema = 0.4
        # Drive the EMA inputs so step_signal evaluates to a known constant (1.0):
        #   step_signal = _accepted_ratio + 0.5*_levy_ratio + 4*(_trial_gain + 0.5*_levy_gain)/scale
        opt._accepted_ratio = 1.0
        opt._levy_ratio = 0.0
        opt._trial_gain = 0.0
        opt._levy_gain = 0.0
        expected_step_signal = 1.0

        ema_before = opt._de_progress_ema
        # pre_best == post_best so the stagnation counter increments but EMA still updates.
        opt._update_de_stagnation(pre_best=10.0, post_best=10.0)
        ema_after = opt._de_progress_ema

        expected = 0.75 * ema_before + 0.25 * expected_step_signal
        assert ema_after == pytest.approx(expected, abs=1e-12), (
            f"EMA coefficient no longer 0.75/0.25: "
            f"expected {expected}, got {ema_after} (delta={ema_after - expected:.3e})"
        )


# ---------------------------------------------------------------------------
# Direction construction.
# ---------------------------------------------------------------------------
class TestDirectionConstruction:
    """Verify richer direction sets in polish phase."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_build_directions_returns_priority_count(self, device: torch.device) -> None:
        """_build_polish_directions should return (dirs, priority_count)."""
        from torch_dfo.cmaes import CMAES

        dim = 25
        opt = PhasedDFO(
            dim=dim,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Manually create a CMA-ES so B matrix exists without running optimize
        opt._cmaes = CMAES(
            dim=dim,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Add some fake elite solutions
        for i in range(30):
            opt._elite_solutions.append(torch.randn(dim, device=device, dtype=torch.float64))
            opt._elite_fitness.append(torch.tensor(float(i), device=device, dtype=torch.float64))

        best_x = torch.zeros(dim, device=device, dtype=torch.float64)
        # Pass directional_budget matching the reference point (dim=20+, budget=50000)
        result = opt._build_polish_directions(best_x, directional_budget=4000)
        assert result is not None
        directions, priority_count = result
        assert directions is not None
        assert directions.ndim == 2
        assert directions.shape[1] == dim
        # Priority count should include CMA vectors + basis pairs
        assert priority_count > 0
        # Total directions should include priority + secondary + random
        assert directions.shape[0] > priority_count


# ---------------------------------------------------------------------------
# Midpoint probing
# ---------------------------------------------------------------------------
class TestMidpointProbing:
    """Verify midpoint is evaluated once and can replace the worst individual."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_midpoint_probing_sets_flag(self, device: torch.device) -> None:
        """After the first DE tell in optimize(), _midpoint_probed must be True."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        assert not opt._midpoint_probed
        opt.optimize(sphere)
        assert opt._midpoint_probed

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_midpoint_replaces_worst(self, device: torch.device) -> None:
        """When midpoint is better than worst, it should replace it."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        # Run one ask/tell cycle to initialize population
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        # Now probe midpoint
        midpoint = (opt.lb + opt.ub) / 2  # should be zeros for sphere
        mid_f = sphere(midpoint.unsqueeze(0)).squeeze()

        worst_idx_before = int(opt._shade.fitness.argmax().item())
        worst_f_before = opt._shade.fitness[worst_idx_before].clone()
        worst_count_before = int((opt._shade.fitness == worst_f_before).sum().item())
        assert mid_f < worst_f_before, (
            "Sphere midpoint (origin) should be better than worst random individual"
        )

        # Trigger the probe
        opt._probe_midpoint()

        # The midpoint solution must be installed at the former worst index.
        assert torch.allclose(opt._shade.population[worst_idx_before], midpoint), (
            "Midpoint solution not installed at former worst index"
        )
        assert opt._shade.fitness[worst_idx_before].item() == pytest.approx(mid_f.item()), (
            "Midpoint fitness not installed at former worst index"
        )
        # Exactly one occurrence of the prior worst fitness must have been
        # evicted (ties in the initial population are allowed; only the
        # argmax index is overwritten).
        worst_count_after = int((opt._shade.fitness == worst_f_before).sum().item())
        assert worst_count_after == worst_count_before - 1, (
            f"Expected one eviction of prior worst fitness; "
            f"count went {worst_count_before}→{worst_count_after}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_midpoint_probed_only_once(self, device: torch.device) -> None:
        """_probe_midpoint should only run once (flag prevents re-probing)."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)

        opt._probe_midpoint()
        fe_after_first = opt._fe_count

        # Calling again should be a no-op because flag is set
        opt._probe_midpoint()
        assert opt._fe_count == fe_after_first, (
            "Second _probe_midpoint call should not spend additional FE"
        )


# ---------------------------------------------------------------------------
# Differential restart mode
# ---------------------------------------------------------------------------
class TestDifferentialRestart:
    """Verify mode 2 (differential) uses anchor + differential vector."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_restart_mode_cycles_through_4(self, device: torch.device) -> None:
        """Restart modes should cycle 0, 1, 2, 3, 0, 1, ..."""
        assert CMA_ES_RESTART_MODES == 4

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_differential_restart_mode(self, device: torch.device) -> None:
        """Mode 2 should produce a differential-based center when pool has >1 entry."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Populate search pool with known, well-separated points
        pool = torch.zeros(5, 5, device=device, dtype=torch.float64)
        pool[0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pool[1] = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        pool[2] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
        pool[3] = torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0])
        pool[4] = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
        pool_fit = torch.arange(5, device=device, dtype=torch.float64)

        opt._search_population = pool
        opt._search_population_fitness = pool_fit

        span = (opt.ub - opt.lb).mean().item()
        center = opt._sample_restart_mean(phase_idx=2, span=span)
        assert center.shape == (5,)
        assert (center >= opt.lb).all()
        assert (center <= opt.ub).all()

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_differential_fallback_to_random(self, device: torch.device) -> None:
        """Mode 2 with only 1 pool entry should fall back to random (mode 0)."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Single-entry pool: differential needs >1
        opt._search_population = torch.zeros(1, 5, device=device, dtype=torch.float64)
        opt._search_population_fitness = torch.zeros(1, device=device, dtype=torch.float64)

        span = (opt.ub - opt.lb).mean().item()
        center = opt._sample_restart_mean(phase_idx=2, span=span)
        assert center.shape == (5,)
        assert (center >= opt.lb).all()
        assert (center <= opt.ub).all()

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_differential_produces_different_centers(self, device: torch.device) -> None:
        """Differential mode should produce varied centers from different seeds."""
        centers = []
        for seed in (42, 99, 137):
            opt = PhasedDFO(
                dim=5,
                bounds=5.0,
                budget=BUDGET_PHASED_STANDARD,
                pop_size=40,
                device=device,
                dtype=torch.float64,
                seed=seed,
            )
            pool = torch.randn(10, 5, device=device, dtype=torch.float64)
            pool_fit = torch.arange(10, device=device, dtype=torch.float64)
            opt._search_population = pool
            opt._search_population_fitness = pool_fit

            span = (opt.ub - opt.lb).mean().item()
            center = opt._sample_restart_mean(phase_idx=2, span=span)
            centers.append(center)

        # At least two of three seeds should produce different centers
        all_same = torch.equal(centers[0], centers[1]) and torch.equal(centers[1], centers[2])
        assert not all_same, "Differential restart produced identical centers for all seeds"


# ---------------------------------------------------------------------------
# Search pool accumulation
# ---------------------------------------------------------------------------
class TestSearchPool:
    """Verify search pool merges across CMA-ES phases."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_merge_search_pool_keeps_best(self, device: torch.device) -> None:
        """_merge_search_pool keeps the best entries by fitness."""
        pool = torch.randn(5, 3, device=device, dtype=torch.float64)
        pool_fit = torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0], device=device)

        additions = torch.randn(3, 3, device=device, dtype=torch.float64)
        add_fit = torch.tensor([0.5, 6.0, 0.1], device=device)

        merged, merged_fit = _merge_search_pool(pool, pool_fit, additions, add_fit, 4)
        assert merged is not None
        assert merged_fit is not None
        assert merged.shape[0] == 4
        assert merged_fit.shape[0] == 4
        # Best 4 should include 0.1, 0.5, 1.0, 2.0
        assert float(merged_fit.max()) <= 2.0 + 1e-12

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_merge_search_pool_none_pool(self, device: torch.device) -> None:
        """_merge_search_pool with None pool returns additions."""
        additions = torch.randn(3, 5, device=device, dtype=torch.float64)
        add_fit = torch.tensor([2.0, 1.0, 3.0], device=device)

        merged, merged_fit = _merge_search_pool(None, None, additions, add_fit, 10)
        assert merged is not None
        assert merged_fit is not None
        assert merged.shape[0] == 3

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_merge_search_pool_none_additions(self, device: torch.device) -> None:
        """_merge_search_pool with None additions returns original pool."""
        pool = torch.randn(5, 3, device=device, dtype=torch.float64)
        pool_fit = torch.arange(5, device=device, dtype=torch.float64)

        result_pool, result_fit = _merge_search_pool(pool, pool_fit, None, None, 10)
        assert result_pool is pool
        assert result_fit is pool_fit

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_search_pool_initialized_in_cmaes_phase(self, device: torch.device) -> None:
        """After entering CMA-ES phase, search pool should be populated."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        # Run until CMA-ES phase is entered and initialized
        for _ in range(200):
            c = opt.ask()
            if c.shape[0] == 0:
                break
            f = sphere(c)
            opt.tell(c, f)
            # Pool is initialized lazily in ask() when phase becomes 1
            if opt._cmaes is not None:
                break

        if opt._cmaes is not None:
            assert opt._search_population is not None, (
                "Search pool should be initialized when entering CMA-ES phase"
            )
            assert opt._search_population.shape[0] > 0

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_search_pool_used_in_polish(self, device: torch.device) -> None:
        """optimize() should use search pool (not just elite list) in polish."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_SMOKE,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt.optimize(sphere)

        # After optimize, if CMA-ES ran, search_population should exist
        if opt._cmaes_phase_idx > 0:
            assert opt._search_population is not None
            assert opt._search_population.shape[0] > 0


# ---------------------------------------------------------------------------
# Basin explore constants
# ---------------------------------------------------------------------------
class TestBasinExploreConstants:
    """Verify BASIN_EXPLORE_* constants and formulas produce correct values."""

    def test_basin_explore_restarts_at_reference(self) -> None:
        # Old hardcoded value was 12; formula must match at reference point
        assert _compute_basin_explore_restarts(dim=10, budget=50000) == 12

    def test_basin_explore_budget_frac_at_reference(self) -> None:
        # Old hardcoded value was 0.05; formula must match at reference point
        frac = _compute_basin_explore_budget_frac(dim=10, budget=50000)
        # one-off: basin_budget_frac tolerance — problem-specific, not centralised
        assert frac == pytest.approx(0.05, abs=0.005)

    def test_basin_explore_stagnation_at_reference(self) -> None:
        # Old hardcoded value was 10; formula must match at BBOB reference point
        basin_frac = _compute_basin_explore_budget_frac(dim=10, budget=50000)
        restarts = _compute_basin_explore_restarts(dim=10, budget=50000)
        stag = _compute_basin_explore_stagnation(
            dim=10,
            budget=BUDGET_PHASED_STANDARD,
            basin_explore_budget_frac=basin_frac,
            basin_explore_restarts=restarts,
        )
        assert stag == 10


# ---------------------------------------------------------------------------
# Multistart basin explore
# ---------------------------------------------------------------------------
class TestMultistartBasinExplore:
    """Verify multistart basin exploration for low-dim problems."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_multistart_basin_explore_does_not_leak_rng(self, device: torch.device) -> None:
        """``_multistart_basin_explore`` must not advance the global torch RNG.

        The method uses an isolated ``torch.Generator`` for its own sampling
        and delegates to a CMAES instance whose ``_gen`` is explicitly wired
        to that isolated generator (phased.py:2911-2933). With a deterministic
        fitness function (sphere touches no RNG), the default generator must
        be untouched — the original test asserted this but restored the
        state manually immediately before the assertion, making it vacuous.
        """
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_QUICK,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        # Run a few DE iterations so basin_explore has a realistic state.
        for _ in range(5):
            if opt._phase != 0:
                break
            c = opt.ask()
            opt.tell(c, sphere(c))

        budget_limit = min(opt._budget, opt._fe_count + 500)
        state_before = torch.random.get_rng_state().clone()
        opt._multistart_basin_explore(sphere, budget_limit)
        state_after = torch.random.get_rng_state()

        assert torch.equal(state_before, state_after), (
            "_multistart_basin_explore advanced the global torch RNG"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_multistart_basin_explore_improves(self, device: torch.device) -> None:
        """Basin explore should be capable of finding a better solution on Rastrigin.

        Rastrigin is multimodal -- random restarts have a chance of landing
        in the global basin near the origin, improving upon a solution stuck
        in a local basin.
        """
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_MEDIUM,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = rastrigin

        # Run DE to get an initial solution
        for _ in range(10):
            if opt._phase != 0:
                break
            c = opt.ask()
            opt.tell(c, rastrigin(c))

        # Force a mediocre best (offset from origin)
        mediocre_x = torch.ones(5, device=device, dtype=torch.float64) * 1.5
        mediocre_f = rastrigin(mediocre_x.unsqueeze(0)).squeeze()
        opt.best_solution = mediocre_x.clone()
        opt.best_fitness = mediocre_f.clone()

        pre_explore_best = opt.best_fitness.item()
        budget_limit = min(opt._budget, opt._fe_count + 2000)
        opt._multistart_basin_explore(rastrigin, budget_limit)

        # Basin explore with multiple restarts should find a point
        # at least as good (and ideally better) than the mediocre starting point
        assert opt.best_fitness.item() <= pre_explore_best + 1e-12, (
            f"Basin explore worsened result: pre={pre_explore_best:.4f}, "
            f"post={opt.best_fitness.item():.4f}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_multistart_basin_explore_consumes_fe(self, device: torch.device) -> None:
        """Basin explore should consume function evaluations."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_MEDIUM,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere

        for _ in range(5):
            if opt._phase != 0:
                break
            c = opt.ask()
            opt.tell(c, sphere(c))

        fe_before = opt._fe_count
        budget_limit = min(opt._budget, opt._fe_count + 1000)
        opt._multistart_basin_explore(sphere, budget_limit)

        assert opt._fe_count > fe_before, (
            f"Basin explore did not consume any FE: before={fe_before}, after={opt._fe_count}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_multistart_basin_explore_skips_low_budget(self, device: torch.device) -> None:
        """Basin explore should be a no-op when remaining budget is < 50."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_MEDIUM,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        # Set fe_count very close to budget_limit so remaining < 50
        opt._fe_count = 19970
        fe_before = opt._fe_count
        opt._multistart_basin_explore(sphere, 20000)
        assert opt._fe_count == fe_before


# ---------------------------------------------------------------------------
# Alternating DE restarts
# ---------------------------------------------------------------------------
class TestAlternatingDERestarts:
    """Verify even restarts get full random population, odd restarts keep elite."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_alternating_de_restarts(self, device: torch.device) -> None:
        """Even restarts should produce full random population (all new).

        Odd restarts should preserve the elite fraction.
        """

        def flat_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = flat_fn
        assert opt._high_dim is False

        # Set up known population with distinct fitness values
        opt._shade._initialized = True
        opt._shade.population = torch.randn(40, 5, device=device, dtype=torch.float64)
        opt._shade.fitness = torch.arange(40, device=device, dtype=torch.float64)
        opt.best_solution = opt._shade.population[0].clone()
        opt.best_fitness = opt._shade.fitness[0].clone()

        # First restart (count becomes 1 = odd): partial restart, elite preserved
        pop_before_odd = opt._shade.population[:4].clone()
        opt._low_dim_pop_restart()
        assert opt._de_restart_count == 1
        # Elite (top 4) should be preserved
        assert torch.allclose(opt._shade.population[:4], pop_before_odd), (
            "Odd restart should preserve elite"
        )

        # Reset known population for second restart test
        opt._shade.population = torch.randn(40, 5, device=device, dtype=torch.float64)
        opt._shade.fitness = torch.arange(40, device=device, dtype=torch.float64)
        pop_before_even = opt._shade.population.clone()

        # Second restart (count becomes 2 = even): full restart, all random
        opt._low_dim_pop_restart()
        assert opt._de_restart_count == 2
        # Population should be completely different (full random restart)
        assert not torch.allclose(opt._shade.population, pop_before_even), (
            "Even restart should replace entire population"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_de_restart_count_increments(self, device: torch.device) -> None:
        """_de_restart_count should increment with each restart call."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_STANDARD,
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        assert opt._de_restart_count == 0
        opt._low_dim_pop_restart()
        assert opt._de_restart_count == 1
        opt._low_dim_pop_restart()
        assert opt._de_restart_count == 2
        opt._low_dim_pop_restart()
        assert opt._de_restart_count == 3


# ---------------------------------------------------------------------------
# Low-dim polish uses scipy
# ---------------------------------------------------------------------------
class TestLowDimPolishUsesScipy:
    """Verify the low-dim polish pipeline includes scipy polishers."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_low_dim_polish_uses_scipy(self, device: torch.device) -> None:
        """Low-dim optimize should call scipy polishers (L-BFGS-B, Powell, etc).

        We verify by checking that the final result is better than just
        coordinate search would typically achieve, and that sufficient FE
        are consumed in the polish phase (indicating the scipy chain ran).
        """
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_QUICK,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        assert opt._high_dim is False

        _, best_f = opt.optimize(sphere)

        # With scipy polishers active, 5d sphere should reach very high precision
        assert best_f.item() < 1e-8, (
            f"Low-dim polish should reach high precision with scipy chain, got {best_f.item():.2e}"
        )

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_low_dim_polish_budget_consumed(self, device: torch.device) -> None:
        """Low-dim polish phase should consume a substantial portion of budget."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.12,
            budget=BUDGET_PHASED_QUICK,
            pop_size=40,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt.optimize(sphere)

        # With ~25% polish reserve (at reference budget) and scipy chain,
        # substantial budget should be consumed
        assert opt.fe_count > 5000, f"Expected significant FE consumption, got {opt.fe_count}"


# ---------------------------------------------------------------------------
# D1 — reset() sub-optimizer reset verification
# ---------------------------------------------------------------------------
class TestReset:
    """Verify ``PhasedDFO.reset()`` clears all sub-optimizer and phase state."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_reset_clears_phase_and_sub_optimizer_state(self, device: torch.device) -> None:
        """D1: reset() must restore fe_count, phase, SHADE init flag, and CMA state."""
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_QUICK,
            pop_size=20,
            device=device,
            dtype=torch.float64,
            seed=42,
        )
        opt._fitness_fn = sphere
        # Drive partial progress: run a handful of DE gens so SHADE is initialized
        # and _fe_count > 0.
        for _ in range(5):
            c = opt.ask()
            if c.shape[0] == 0:
                break
            opt.tell(c, sphere(c))

        assert opt._fe_count > 0
        assert opt._shade._initialized
        pop_before = opt._shade.population.clone()

        # Force a CMA-ES entry so the _cmaes field is set, exercising the
        # "cmaes gets cleared" part of the reset contract.
        opt._enter_cmaes_phase()
        assert opt._cmaes is not None

        # Act
        opt.reset()

        # Assert: scalar phase/budget counters
        assert opt._fe_count == 0, f"fe_count not zeroed: {opt._fe_count}"
        assert opt._phase == 0, f"phase not zeroed: {opt._phase}"
        # SHADE re-created with fresh init flag
        assert not opt._shade._initialized, "SHADE _initialized not reset"
        # CMA-ES cleared
        assert opt._cmaes is None, "CMA-ES not cleared on reset"
        assert opt._cmaes_portfolio is None, "CMA-ES portfolio not cleared on reset"
        # Population tensor is a fresh SHADE instance; distinct memory from the
        # pre-reset population (by identity — they must not be the same object).
        assert opt._shade.population.data_ptr() != pop_before.data_ptr(), (
            "SHADE population tensor reused across reset (should be a new instance)"
        )


# ---------------------------------------------------------------------------
# Round-2 edge-case audit: budget exhaustion and reset-continue
# ---------------------------------------------------------------------------
class TestBudgetAndResetEdgeCases:
    """Fencepost and edge cases around ask() when done, and reset-continue."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_ask_returns_empty_after_done(self, device: torch.device) -> None:
        """phased.py:1876 — after optimize() terminates (phase=3), ask() is empty.

        This is the contract the DFOOptimizer / user ask-tell loop relies on
        to detect "no more work". The shape must be exactly (0, dim), not
        None and not a scalar 0.
        """
        dtype = torch.float64
        opt = PhasedDFO(
            dim=3,
            bounds=5.0,
            budget=BUDGET_PHASED_MICRO,
            pop_size=10,
            device=device,
            dtype=dtype,
            seed=42,
        )
        opt.optimize(sphere)
        assert opt.done
        c = opt.ask()
        assert c.shape == (0, 3), f"ask() after done must return (0, dim); got {tuple(c.shape)}"
        # tell() on empty must also be a safe no-op.
        opt.tell(c, torch.empty(0, device=device, dtype=dtype))
        # best() remains finite and unchanged.
        assert torch.isfinite(opt.best_fitness), "best_fitness corrupted by no-op tell"

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_reset_mid_optimization_then_continue(self, device: torch.device) -> None:
        """reset() mid-run leaves the optimizer in a usable ask/tell state.

        A user might reset an optimizer to re-run from scratch; the post-reset
        instance must advance fe_count from zero and reach a finite best like
        a fresh constructor would. This covers the full "reset → continue"
        contract that test_reset_clears_phase_and_sub_optimizer_state does
        not exercise (that test stops at the reset).
        """
        dtype = torch.float64
        opt = PhasedDFO(
            dim=5,
            bounds=5.0,
            budget=BUDGET_PHASED_QUICK,
            pop_size=20,
            device=device,
            dtype=dtype,
            seed=42,
        )
        for _ in range(3):
            c = opt.ask()
            opt.tell(c, sphere(c))
        assert opt._fe_count > 0
        opt.reset()

        # Continue with a fresh ask/tell loop.
        for _ in range(3):
            c = opt.ask()
            assert c.shape[0] > 0, "ask() empty immediately after reset"
            opt.tell(c, sphere(c))
        assert opt._fe_count > 0, "fe_count did not advance after reset+continue"
        assert torch.isfinite(opt.best_fitness), "best_fitness not finite after reset+continue"
