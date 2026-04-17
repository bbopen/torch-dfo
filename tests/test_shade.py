"""Tests for torch_dfo.shade -- SHADE optimizer."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import (
    ATOL_F64_DEFAULT,
    CONV_SPHERE_5D_FAST,
    CONV_SPHERE_10D_STANDARD,
    SMOKE_F_INIT_SHADE,
    SMOKE_F_MULTIMODAL_IMPROVEMENT,
)
from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import sphere
from torch_dfo.shade import SHADE


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Validate constructor wiring: shapes, devices, initial values."""

    def test_population_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=10, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        assert opt.population.shape == (40, 10)

    def test_fitness_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=10, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        assert opt.fitness.shape == (40,)

    def test_device_placement(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        assert opt.population.device.type == device.type
        assert opt.fitness.device.type == device.type
        assert opt.memory_F.device.type == device.type
        assert opt.memory_CR.device.type == device.type
        assert opt.best_solution.device.type == device.type

    def test_initial_memory_values(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=5,
            bounds=5.0,
            pop_size=20,
            memory_size=6,
            device=device,
            dtype=dtype,
            seed=42,
        )
        assert opt.memory_F.shape == (6,)
        assert opt.memory_CR.shape == (6,)
        assert torch.all(opt.memory_F == 0.7)
        assert torch.all(opt.memory_CR == 0.7)

    def test_archive_initially_empty(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=8, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        assert opt._archive.shape == (0, 8)
        assert opt._archive.device.type == device.type

    def test_archive_max_from_ratio(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=5,
            bounds=5.0,
            pop_size=50,
            archive_ratio=1.5,
            device=device,
            dtype=dtype,
            seed=42,
        )
        assert opt._archive_max == 75

    def test_custom_parameters(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=3,
            bounds=(-2.0, 4.0),
            pop_size=60,
            memory_size=10,
            p_min=0.05,
            p_max=0.4,
            archive_ratio=2.0,
            device=device,
            dtype=dtype,
            seed=7,
        )
        assert opt.p_min == 0.05
        assert opt.p_max == 0.4
        assert opt.memory_size == 10
        assert opt._archive_max == 120
        assert opt.dim == 3
        assert opt.pop_size == 60


# ---------------------------------------------------------------------------
# ask() shape and bounds
# ---------------------------------------------------------------------------
class TestAskShape:
    """Validate that ask() returns correctly shaped, bounded tensors."""

    # (Removed test_initial_ask_shape -- duplicate of test_population_shape)

    def test_initial_ask_device(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        assert candidates.device.type == device.type

    def test_initial_ask_within_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=8,
            bounds=(-3.0, 7.0),
            pop_size=50,
            device=device,
            dtype=dtype,
            seed=42,
        )
        candidates = opt.ask()
        assert (candidates >= opt.lb).all(), "Candidates below lower bound"
        assert (candidates <= opt.ub).all(), "Candidates above upper bound"

    def test_subsequent_ask_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=10, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        # First ask (init)
        c1 = opt.ask()
        f1 = sphere(c1)
        opt.tell(c1, f1)
        # Second ask (mutation/crossover)
        c2 = opt.ask()
        assert c2.shape == (40, 10)

    def test_subsequent_ask_within_bounds(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=6,
            bounds=(-5.0, 5.0),
            pop_size=30,
            device=device,
            dtype=dtype,
            seed=42,
        )
        c1 = opt.ask()
        f1 = sphere(c1)
        opt.tell(c1, f1)
        c2 = opt.ask()
        assert (c2 >= opt.lb).all(), "Trial vectors below lower bound"
        assert (c2 <= opt.ub).all(), "Trial vectors above upper bound"

    def test_subsequent_ask_device(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c1 = opt.ask()
        f1 = sphere(c1)
        opt.tell(c1, f1)
        c2 = opt.ask()
        assert c2.device.type == device.type


# ---------------------------------------------------------------------------
# tell() updates
# ---------------------------------------------------------------------------
class TestTellUpdates:
    """Validate that tell() properly updates population and fitness."""

    def test_fitness_stored_after_first_tell(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        fitness = sphere(candidates)
        opt.tell(candidates, fitness)
        assert torch.allclose(opt.fitness, fitness)

    def test_population_changes_after_tell(self, device: torch.device) -> None:
        """Run enough generations to move the population off its init state, robustly across seeds."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        # Init
        c = opt.ask()
        f = sphere(c)
        opt.tell(c, f)
        pop_after_init = opt.population.clone()
        # Run 5 generations
        for _ in range(5):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        # Population should change after multiple generations
        assert not torch.equal(opt.population, pop_after_init)

    def test_fitness_updates_after_tell(self, device: torch.device) -> None:
        """Greedy selection guarantees mean fitness strictly improves after a tell."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        c1 = opt.ask()
        f1 = sphere(c1)
        opt.tell(c1, f1)
        fit_after_init = opt.fitness.clone()
        # Run multiple generations so greedy selection accumulates improvements
        for _ in range(5):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        # Mean fitness must strictly decrease (greedy selection on sphere)
        assert opt.fitness.mean() < fit_after_init.mean()

    def test_generation_counter_increments(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        assert opt._generation == 0
        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 1
        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._generation == 2


# ---------------------------------------------------------------------------
# Best tracking
# ---------------------------------------------------------------------------
class TestBestTracking:
    """Validate that best solution/fitness improve over iterations."""

    def test_best_improves_over_iterations(self, device: torch.device) -> None:
        """SHADE on the sphere must strictly improve the best fitness over iterations."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        # Run a few generations
        for _ in range(20):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        _, best_f_early = opt.best()
        for _ in range(80):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        _, best_f_late = opt.best()
        assert best_f_late < best_f_early

    def test_best_solution_shape(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=7, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        opt.tell(c, sphere(c))
        sol, fit = opt.best()
        assert sol.shape == (7,)
        assert fit.shape == ()
        assert sol.device.type == device.type

    def test_best_not_inf_after_tell(self, device: torch.device) -> None:
        """Best fitness should be within a meaningful bound after one tell, not just finite."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c = opt.ask()
        opt.tell(c, sphere(c))
        _, fit = opt.best()
        # 5d sphere in [-5,5]: max possible = 5*25=125, use 500 for safety
        assert fit.item() < SMOKE_F_INIT_SHADE


# ---------------------------------------------------------------------------
# Convergence on Sphere 10d (CPU + float64 only)
# ---------------------------------------------------------------------------
class TestConvergence:
    """Verify SHADE converges on standard benchmarks."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sphere_10d_convergence(self, device: torch.device, seed: int) -> None:
        """SHADE on Sphere 10d must reach < 1e-6 within 5000 generations."""
        opt = SHADE(
            dim=10,
            bounds=5.12,
            pop_size=40,
            memory_size=6,
            p_min=0.1,
            p_max=0.3,
            device=device,
            dtype=torch.float64,
            seed=seed,
        )
        for _ in range(5000):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            _, best_f = opt.best()
            if best_f.item() < 1e-6:
                break
        _, final_f = opt.best()
        assert final_f.item() < CONV_SPHERE_10D_STANDARD, (
            f"SHADE failed to converge on Sphere 10d: best={final_f.item():.2e}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 7])
    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_sphere_5d_fast_convergence(self, device: torch.device, seed: int) -> None:
        """SHADE on Sphere 5d should converge very quickly."""
        opt = SHADE(
            dim=5,
            bounds=5.12,
            pop_size=30,
            device=device,
            dtype=torch.float64,
            seed=seed,
        )
        for _ in range(2000):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
            _, best_f = opt.best()
            if best_f.item() < 1e-8:
                break
        _, final_f = opt.best()
        assert final_f.item() < CONV_SPHERE_5D_FAST, (
            f"SHADE failed to converge on Sphere 5d: best={final_f.item():.2e}"
        )


# ---------------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------------
class TestMemoryUpdate:
    """Validate SHADE's success-history memory adaptation."""

    def test_memory_changes_after_improvements(self, device: torch.device) -> None:
        """memory_F should update after improvements and stay in the sampling clamp range [0.05, 1.0]."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        initial_F = opt.memory_F.clone()
        initial_CR = opt.memory_CR.clone()
        # Init generation
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Run enough generations that some memory update should occur
        for _ in range(20):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        # Majority of memory slots must have moved meaningfully from 0.7 init
        # (ULP-level drift of a single slot should not satisfy the test).
        n_slots = opt.memory_F.numel()
        majority = (n_slots // 2) + 1  # e.g. 4/6 ≥ 3
        n_f_moved = int((torch.abs(opt.memory_F - 0.7) > 1e-6).sum().item())
        n_cr_moved = int((torch.abs(opt.memory_CR - 0.7) > 1e-6).sum().item())
        assert n_f_moved >= majority, (
            f"Memory F only {n_f_moved}/{n_slots} slots moved >1e-6 from 0.7"
        )
        assert n_cr_moved >= majority, (
            f"Memory CR only {n_cr_moved}/{n_slots} slots moved >1e-6 from 0.7"
        )
        # Verify memory_F stays in sampling clamping range [0.05, 1.0]
        assert torch.all(opt.memory_F >= 0.05), f"Memory F below 0.05: {opt.memory_F}"
        assert torch.all(opt.memory_F <= 1.0), f"Memory F above 1.0: {opt.memory_F}"

    def test_memory_values_in_valid_range(self, device: torch.device) -> None:
        """memory_F should be in [0.05, 1.0] -- the Cauchy clamp range."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        for _ in range(30):
            c = opt.ask()
            opt.tell(c, sphere(c))
        # Memory F should be in [0.05, 1.0] (the F clamping range)
        assert torch.all(opt.memory_F >= 0.05), f"Memory F below 0.05: {opt.memory_F}"
        assert torch.all(opt.memory_F <= 1.0), f"Memory F above 1.0: {opt.memory_F}"
        # Memory CR should be in [0, 1]
        assert (opt.memory_CR >= 0).all(), f"Memory CR below 0: {opt.memory_CR}"
        assert (opt.memory_CR <= 1).all(), f"Memory CR above 1: {opt.memory_CR}"


# ---------------------------------------------------------------------------
# Lehmer mean
# ---------------------------------------------------------------------------
class TestLehmerMean:
    """Validate the Lehmer mean computation in SHADE memory updates."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_lehmer_mean_correctness(self, device: torch.device) -> None:
        """Verify Lehmer mean is computed correctly for known F values."""
        dtype = torch.float64
        pop_size = 6
        dim = 3
        opt = SHADE(
            dim=dim,
            bounds=5.0,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            seed=42,
        )
        # Run the init ask/tell cycle (gen 0)
        c0 = opt.ask()
        f0 = sphere(c0)
        opt.tell(c0, f0)

        # Now gen 1: ask to generate trial vectors
        trials = opt.ask()

        # Manually set known trial_F values for all individuals
        known_F = torch.tensor(
            [0.3, 0.7, 0.5, 0.2, 0.8, 0.6],
            device=device,
            dtype=dtype,
        )
        opt._trial_F.copy_(known_F)

        # Save the current memory position and parent fitness
        mem_pos = opt._memory_pos
        parent_fitness = opt.fitness.clone()

        # Construct fitness so that individuals 0, 2, 4 improve (lower fitness)
        # and individuals 1, 3, 5 do NOT improve
        trial_fitness = parent_fitness.clone()
        # Make individuals 0, 2, 4 strictly better
        trial_fitness[0] = parent_fitness[0] * 0.5
        trial_fitness[2] = parent_fitness[2] * 0.3
        trial_fitness[4] = parent_fitness[4] * 0.7
        # Make individuals 1, 3, 5 worse (no improvement)
        trial_fitness[1] = parent_fitness[1] * 1.5
        trial_fitness[3] = parent_fitness[3] * 2.0
        trial_fitness[5] = parent_fitness[5] * 1.1

        opt.tell(trials, trial_fitness)

        # Compute expected Lehmer mean from the successful F values
        succ_F = known_F[torch.tensor([0, 2, 4])]
        delta_f = torch.stack(
            [
                parent_fitness[0] - trial_fitness[0],
                parent_fitness[2] - trial_fitness[2],
                parent_fitness[4] - trial_fitness[4],
            ],
        )
        weights = delta_f / delta_f.sum()
        expected_lehmer = (weights * succ_F**2).sum() / (weights * succ_F).sum()

        assert torch.allclose(
            opt.memory_F[mem_pos],
            expected_lehmer,
            atol=ATOL_F64_DEFAULT,
        ), (
            f"Lehmer mean mismatch: got {opt.memory_F[mem_pos].item()}, "
            f"expected {expected_lehmer.item()}"
        )


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
class TestArchive:
    """Validate the external JADE-style archive."""

    def test_archive_grows_after_improvements(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        # Init
        c = opt.ask()
        opt.tell(c, sphere(c))
        assert opt._archive.shape[0] == 0, "Archive should be empty after init"
        # Run a few generations -- improvements should add parents to archive
        for _ in range(10):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        assert opt._archive.shape[0] > 0, "Archive should be non-empty after improvements"
        assert opt._archive.shape[1] == 5, "Archive dim mismatch"

    def test_archive_respects_max_size(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(
            dim=5,
            bounds=5.0,
            pop_size=40,
            archive_ratio=0.5,
            device=device,
            dtype=dtype,
            seed=42,
        )
        max_archive = opt._archive_max  # 40 * 0.5 = 20
        for _ in range(50):
            c = opt.ask()
            opt.tell(c, sphere(c))
        assert opt._archive.shape[0] <= max_archive, (
            f"Archive size {opt._archive.shape[0]} exceeds max {max_archive}"
        )

    def test_archive_on_correct_device(self, device: torch.device) -> None:
        """After enough generations the archive is guaranteed non-empty and on the correct device."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        for _ in range(30):
            c = opt.ask()
            opt.tell(c, sphere(c))
        assert opt._archive.shape[0] > 0, "Archive should be non-empty after 30 generations"
        assert opt._archive.device.type == device.type

    def test_archive_stores_replaced_parents(self, device: torch.device) -> None:
        """Archive should contain old parent vectors that greedy selection replaced."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        # Init
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Save parents before the next tell
        parents_before = opt.population.clone()
        # Run one more generation
        c = opt.ask()
        f = sphere(c)
        # Identify which will improve
        improved_mask = f < opt.fitness
        old_parents_to_archive = parents_before[improved_mask]
        opt.tell(c, f)
        # Fail-closed: precondition MUST hold so the containment check is meaningful.
        assert improved_mask.any(), (
            "Precondition violated: expected at least one trial to beat its parent."
        )
        for parent in old_parents_to_archive:
            found = False
            for i in range(opt._archive.shape[0]):
                if torch.allclose(opt._archive[i], parent):
                    found = True
                    break
            assert found, "Archive missing a replaced parent vector"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    """Same seed must produce identical results."""

    def test_same_seed_same_init(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        opt2 = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        c1 = opt1.ask()
        c2 = opt2.ask()
        assert torch.equal(c1, c2), "Same seed should give same initial population"

    def test_same_seed_same_optimization(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        results = []
        for _ in range(2):
            opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=123)
            for _g in range(15):
                c = opt.ask()
                f = sphere(c)
                opt.tell(c, f)
            results.append(opt.best())
        sol1, fit1 = results[0]
        sol2, fit2 = results[1]
        assert torch.equal(sol1, sol2), "Same seed should give same best solution"
        assert torch.equal(fit1, fit2), "Same seed should give same best fitness"

    def test_different_seeds_differ(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt1 = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        opt2 = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=99)
        c1 = opt1.ask()
        c2 = opt2.ask()
        assert not torch.equal(c1, c2), "Different seeds should give different populations"


# ---------------------------------------------------------------------------
# Multi-device (uses the parametrized device fixture from conftest)
# ---------------------------------------------------------------------------
class TestMultiDevice:
    """Verify SHADE works on all available devices."""

    def test_ask_tell_loop_runs(self, device: torch.device) -> None:
        """Ask/tell loop runs cleanly and makes measurable progress after a few generations on the sphere."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        for _ in range(5):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        sol, fit = opt.best()
        assert sol.device.type == device.type
        assert fit.item() < SMOKE_F_MULTIMODAL_IMPROVEMENT

    def test_no_nan_in_population(self, device: torch.device) -> None:
        dtype = best_float_dtype(device)
        opt = SHADE(dim=8, bounds=5.0, pop_size=30, device=device, dtype=dtype, seed=42)
        for _ in range(10):
            c = opt.ask()
            f = sphere(c)
            opt.tell(c, f)
        assert torch.isfinite(opt.population).all(), "Population contains NaN/Inf"
        assert torch.isfinite(opt.fitness).all(), "Fitness contains NaN/Inf"


# ---------------------------------------------------------------------------
# Revert detection: F must be Normal, not Cauchy
# ---------------------------------------------------------------------------
class TestFSamplingIsNormal:
    """F samples should have Normal-like kurtosis, not Cauchy."""

    @pytest.mark.parametrize("device", [torch.device("cpu")])
    def test_f_kurtosis_is_normal_not_cauchy(self, device: torch.device) -> None:
        """Normal excess kurtosis ~0; Cauchy excess kurtosis is huge (>10).

        Collect many trial_F samples and verify kurtosis stays low, proving
        Normal distribution (not Cauchy). Fails if Cauchy is used for F.
        """
        dtype = torch.float64
        opt = SHADE(dim=5, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        # Init generation
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Collect F samples over many generations. 600 * pop_size=40 = 24k raw
        # samples; after boundary clipping, >20k interior samples remain —
        # enough statistical basis for a tight kurtosis bound.
        all_f: list[torch.Tensor] = []
        for _ in range(600):
            opt.ask()
            all_f.append(opt._trial_F.clone())
        samples = torch.cat(all_f)
        # Keep only interior values (not clamped to boundaries)
        interior = samples[(samples > 0.06) & (samples < 0.99)]
        assert interior.numel() > 2000, "Not enough interior F samples"
        mean_f = interior.mean()
        std_f = interior.std()
        # Excess kurtosis: Normal = 0, Cauchy = infinite (observed >> 10).
        # Bound of 2.0 is a safe statistical margin at >20k samples.
        kurtosis = ((interior - mean_f) ** 4).mean() / (std_f**4) - 3.0
        assert kurtosis < 2.0, (
            f"F excess kurtosis={kurtosis:.2f} suggests Cauchy, not Normal (Normal=0, Cauchy→inf)"
        )


# ---------------------------------------------------------------------------
# F/CR clamping boundaries
# ---------------------------------------------------------------------------
class TestFCRClamping:
    """Validate that trial F and CR parameters are properly clamped."""

    def test_f_cr_clamping_boundary(self, device: torch.device) -> None:
        """After a non-init ask, trial F and CR are clamped to their valid ranges."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=40, device=device, dtype=dtype, seed=42)
        # Init generation
        c = opt.ask()
        opt.tell(c, sphere(c))
        # Non-init ask (gen > 0): this sets _trial_F and _trial_CR
        opt.ask()
        assert torch.all(opt._trial_F >= 0.05), (
            f"trial_F below 0.05: min={opt._trial_F.min().item()}"
        )
        assert torch.all(opt._trial_F <= 1.0), f"trial_F above 1.0: max={opt._trial_F.max().item()}"
        assert torch.all(opt._trial_CR >= 0.0), (
            f"trial_CR below 0.0: min={opt._trial_CR.min().item()}"
        )
        assert torch.all(opt._trial_CR <= 1.0), (
            f"trial_CR above 1.0: max={opt._trial_CR.max().item()}"
        )


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------
class TestGreedySelection:
    """Validate that greedy selection correctly replaces improved individuals."""

    def test_greedy_selection_correctness(self, device: torch.device) -> None:
        """Greedy selection replaces only improved individuals; worse trials are rejected."""
        dtype = best_float_dtype(device)
        pop_size = 4
        dim = 2
        opt = SHADE(
            dim=dim,
            bounds=5.0,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            seed=42,
        )
        # Init generation
        c0 = opt.ask()
        f0 = sphere(c0)
        opt.tell(c0, f0)

        # Save parents before the next generation
        parents = opt.population.clone()
        parent_fitness = opt.fitness.clone()

        # Ask for trial vectors
        trials = opt.ask()

        # Construct fitness where individuals 0 and 2 improve, 1 and 3 do not
        trial_fitness = torch.empty(pop_size, device=device, dtype=dtype)
        trial_fitness[0] = parent_fitness[0] * 0.5  # improved
        trial_fitness[1] = parent_fitness[1] * 1.5  # NOT improved
        trial_fitness[2] = parent_fitness[2] * 0.3  # improved
        trial_fitness[3] = parent_fitness[3] * 2.0  # NOT improved

        opt.tell(trials, trial_fitness)

        # Individuals 0 and 2 should now be the trial vectors
        assert torch.equal(opt.population[0], trials[0]), (
            "Improved individual 0 should be the trial vector"
        )
        assert torch.equal(opt.population[2], trials[2]), (
            "Improved individual 2 should be the trial vector"
        )
        # Individuals 1 and 3 should be unchanged (still the parents)
        assert torch.equal(opt.population[1], parents[1]), (
            "Non-improved individual 1 should remain as parent"
        )
        assert torch.equal(opt.population[3], parents[3]), (
            "Non-improved individual 3 should remain as parent"
        )


# ---------------------------------------------------------------------------
# First generation early return
# ---------------------------------------------------------------------------
class TestFirstGeneration:
    """Validate the first generation (gen=0) early return behavior."""

    def test_first_generation_early_return(self, device: torch.device) -> None:
        """After the first ask/tell, gen=1, the population matches what was told, and the archive is empty."""
        dtype = best_float_dtype(device)
        opt = SHADE(dim=5, bounds=5.0, pop_size=20, device=device, dtype=dtype, seed=42)
        candidates = opt.ask()
        fitness = sphere(candidates)
        opt.tell(candidates, fitness)
        # Generation counter should be 1 after first tell
        assert opt._generation == 1
        # Population should match the candidates passed
        assert torch.equal(opt.population, candidates)
        # No archive entries should have been added during gen=0
        assert opt._archive.shape[0] == 0, "Archive should be empty after the first generation"


# ---------------------------------------------------------------------------
# Warm-start via initial_population
# ---------------------------------------------------------------------------
class TestSHADEWarmStart:
    def test_initial_population_seeds_first_rows(self, device, default_dtype):
        # Create initial_population as zeros (5 rows)
        n_seed = 5
        initial = torch.zeros(n_seed, 4, device=device, dtype=default_dtype)
        opt = SHADE(
            dim=4,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=default_dtype,
            initial_population=initial,
        )
        candidates = opt.ask()
        # First n_seed rows should be at origin (zeros are within bounds)
        assert torch.allclose(candidates[:n_seed], initial, atol=ATOL_F64_DEFAULT)

    def test_initial_population_clamped_to_bounds(self, device, default_dtype):
        # Points outside bounds should be clamped
        initial = torch.full((3, 4), 10.0, device=device, dtype=default_dtype)  # outside [-5, 5]
        opt = SHADE(
            dim=4,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=default_dtype,
            initial_population=initial,
        )
        candidates = opt.ask()
        # Clamped to upper bound 5.0
        expected = torch.full((3, 4), 5.0, device=device, dtype=default_dtype)
        assert torch.allclose(candidates[:3], expected, atol=ATOL_F64_DEFAULT)

    def test_initial_population_none_uses_opposition_init(self, device, default_dtype):
        # Without initial_population, population is random (not zeros)
        opt = SHADE(
            dim=4,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=default_dtype,
            seed=42,
            initial_population=None,
        )
        candidates = opt.ask()
        # Candidates should NOT all be zero (opposition init gives varied points)
        assert not torch.allclose(candidates, torch.zeros_like(candidates), atol=ATOL_F64_DEFAULT)

    def test_initial_population_freed_after_first_ask(self, device, default_dtype):
        initial = torch.zeros(5, 4, device=device, dtype=default_dtype)
        opt = SHADE(
            dim=4,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=default_dtype,
            initial_population=initial,
        )
        opt.ask()  # first ask applies seed
        assert opt._initial_population is None  # freed

    def test_initial_population_larger_than_pop_size_truncated(self, device, default_dtype):
        # More seed points than pop_size → only first pop_size used
        n_seed = 30  # > pop_size=20
        initial = torch.zeros(n_seed, 4, device=device, dtype=default_dtype)
        opt = SHADE(
            dim=4,
            bounds=(-5.0, 5.0),
            pop_size=20,
            device=device,
            dtype=default_dtype,
            initial_population=initial,
        )
        candidates = opt.ask()
        assert candidates.shape[0] == 20  # still pop_size rows
        # All 20 rows should be from seed (zeros)
        expected_zeros = torch.zeros(20, 4, device=device, dtype=default_dtype)
        assert torch.allclose(candidates, expected_zeros, atol=ATOL_F64_DEFAULT)
