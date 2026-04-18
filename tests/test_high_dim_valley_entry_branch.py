"""Tests for the dim40+ incumbent valley-entry CMA branch."""

from __future__ import annotations

import math

import pytest
import torch

from tests.conftest import best_float_dtype
from torch_dfo.benchmarks import sphere
from torch_dfo.phased import (
    HIGH_DIM_VALLEY_ENTRY_BRANCH,
    HIGH_DIM_VALLEY_ENTRY_FOCUS_CYCLE,
    HIGH_DIM_VALLEY_ENTRY_LINE_SAMPLES,
    HIGH_DIM_VALLEY_ENTRY_LINE_SCALE,
    HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_CYCLE,
    HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_EVAL_RATIO,
    HIGH_DIM_VALLEY_ENTRY_PATH_MEMORY,
    HIGH_DIM_VALLEY_ENTRY_PATH_SCALE,
    HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS,
    HIGH_DIM_VALLEY_ENTRY_TERMINAL_FOCUS_FRACTION,
    PhasedDFO,
    _compute_valley_focus_generation_bounds,
)


def _seed_high_dim_elite_state(opt: PhasedDFO) -> None:
    """Populate SHADE with a deterministic finite elite cloud."""
    dtype = opt.dtype
    device = opt.device
    base = torch.linspace(-0.6, 0.6, opt.dim, device=device, dtype=dtype)
    scales = torch.linspace(0.2, 1.2, opt._shade.pop_size, device=device, dtype=dtype)
    population = scales.unsqueeze(1) * base.unsqueeze(0)
    population[1::2] = population[1::2].flip(dims=(1,))
    fitness = sphere(population)
    best_idx = fitness.argmin()

    opt._shade.population = population.clone()
    opt._shade.fitness = fitness.clone()
    opt.best_solution = population[best_idx].clone()
    opt.best_fitness = fitness[best_idx].clone()


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_has_incumbent_lm_valley_branch(device: torch.device) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)

    opt._enter_cmaes_phase_portfolio()

    assert opt._cmaes_portfolio is not None
    assert (
        tuple(branch.pop_size for branch in opt._cmaes_portfolio)
        == HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS
    )
    branch = opt._cmaes_portfolio[HIGH_DIM_VALLEY_ENTRY_BRANCH]
    assert torch.allclose(branch.mean, opt.best_solution)
    assert branch.path_memory == HIGH_DIM_VALLEY_ENTRY_PATH_MEMORY
    assert branch.path_scale == pytest.approx(HIGH_DIM_VALLEY_ENTRY_PATH_SCALE)
    assert branch.path_line_samples == HIGH_DIM_VALLEY_ENTRY_LINE_SAMPLES
    assert branch.path_line_scale == pytest.approx(HIGH_DIM_VALLEY_ENTRY_LINE_SCALE)
    assert branch._path_count > 0
    assert branch._path_vectors.device.type == device.type

    for idx, other in enumerate(opt._cmaes_portfolio):
        if idx != HIGH_DIM_VALLEY_ENTRY_BRANCH:
            assert other.path_memory == 0
            assert other.path_line_samples == 0


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_roundtrip_preserves_cma_modes(device: torch.device) -> None:
    dtype = best_float_dtype(device)
    opt1 = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt1)
    opt1._phase = 1

    candidates = opt1.ask()
    opt1.tell(candidates, sphere(candidates))
    state = opt1.state_dict()

    opt2 = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=99,
    )
    opt2.load_state_dict(state)

    assert opt1._cmaes_portfolio is not None
    assert opt2._cmaes_portfolio is not None
    assert opt2._portfolio_generation == opt1._portfolio_generation
    assert opt2._portfolio_active_indices == opt1._portfolio_active_indices

    for branch1, branch2 in zip(opt1._cmaes_portfolio, opt2._cmaes_portfolio, strict=True):
        assert branch2.active is branch1.active
        assert branch2.mirrored is branch1.mirrored
        assert branch2.path_memory == branch1.path_memory
        assert branch2.path_line_samples == branch1.path_line_samples

    assert torch.allclose(opt1.ask(), opt2.ask())


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim20_portfolio_keeps_all_random_cma_branches(device: torch.device) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=20,
        bounds=5.0,
        budget=40000,
        pop_size=40,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)

    opt._enter_cmaes_phase_portfolio()

    assert opt._cmaes_portfolio is not None
    assert tuple(branch.pop_size for branch in opt._cmaes_portfolio) == (24, 12, 12, 12)
    assert all(branch.path_memory == 0 for branch in opt._cmaes_portfolio)
    assert all(branch.path_line_samples == 0 for branch in opt._cmaes_portfolio)


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_ask_tell_uses_dim40_lambda_schedule(device: torch.device) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)
    opt._phase = 1

    candidates = opt.ask()

    assert candidates.shape == (sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS), opt.dim)
    fitness = sphere(candidates)
    opt.tell(candidates, fitness)

    assert opt._fe_count == sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)
    assert opt._cmaes_portfolio is not None
    assert (
        tuple(branch.pop_size for branch in opt._cmaes_portfolio)
        == HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS
    )


def test_dim40_focus_bounds_are_budget_parity_based() -> None:
    min_focus, max_focus = _compute_valley_focus_generation_bounds()
    full_lam = sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)
    valley_lam = HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS[HIGH_DIM_VALLEY_ENTRY_BRANCH]

    assert min_focus == math.ceil(full_lam / valley_lam)
    assert max_focus == math.ceil(
        HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_EVAL_RATIO * full_lam / valley_lam,
    )
    assert max_focus >= HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_CYCLE - 1


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_interleaves_incumbent_only_generations(
    device: torch.device,
) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)
    opt._phase = 1

    full_n = sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)
    valley_n = HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS[HIGH_DIM_VALLEY_ENTRY_BRANCH]
    min_focus, _ = _compute_valley_focus_generation_bounds()

    candidates = opt.ask()
    assert candidates.shape == (full_n, opt.dim)
    opt.tell(candidates, sphere(candidates))

    assert opt._portfolio_generation == 1
    assert opt._portfolio_active_indices == ()

    for generation in range(1, min_focus + 1):
        candidates = opt.ask()
        assert opt._portfolio_active_indices == (HIGH_DIM_VALLEY_ENTRY_BRANCH,)
        assert candidates.shape == (valley_n, opt.dim)
        non_improving = torch.full(
            (valley_n,),
            1e12,
            device=device,
            dtype=dtype,
        )
        opt.tell(candidates, non_improving)
        assert opt._portfolio_generation == generation + 1

    candidates = opt.ask()
    assert candidates.shape == (full_n, opt.dim)
    assert opt._portfolio_active_indices == tuple(
        range(len(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)),
    )


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_extends_focus_burst_while_valley_improves(
    device: torch.device,
) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)
    opt._phase = 1

    full_n = sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)
    valley_n = HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS[HIGH_DIM_VALLEY_ENTRY_BRANCH]
    valley_start = sum(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS[:HIGH_DIM_VALLEY_ENTRY_BRANCH])
    min_focus, max_focus_gens = _compute_valley_focus_generation_bounds()

    candidates = opt.ask()
    assert candidates.shape == (full_n, opt.dim)
    fitness = torch.full((full_n,), 100.0, device=device, dtype=dtype)
    fitness[valley_start : valley_start + valley_n] = 10.0
    opt.tell(candidates, fitness)

    assert opt._valley_focus_remaining == min_focus
    assert opt._valley_focus_streak == 0

    for focus_idx in range(max_focus_gens):
        candidates = opt.ask()
        assert opt._portfolio_active_indices == (HIGH_DIM_VALLEY_ENTRY_BRANCH,)
        assert candidates.shape == (valley_n, opt.dim)
        improving_fitness = torch.full(
            (valley_n,),
            9.0 - focus_idx,
            device=device,
            dtype=dtype,
        )
        opt.tell(candidates, improving_fitness)
        assert opt._valley_focus_streak == focus_idx + 1

    assert opt._valley_focus_remaining == 0
    candidates = opt.ask()
    assert candidates.shape == (full_n, opt.dim)
    assert opt._portfolio_active_indices == tuple(
        range(len(HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS)),
    )


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim40_portfolio_locks_to_incumbent_in_terminal_cma_window(
    device: torch.device,
) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=40,
        bounds=5.0,
        budget=80000,
        pop_size=48,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)
    opt._phase = 1

    initial_candidates = opt.ask()
    opt.tell(initial_candidates, sphere(initial_candidates))
    assert opt._cmaes_portfolio is not None

    opt._valley_focus_remaining = 0
    cma_total = opt._cmaes_budget - opt._cmaes_fe_start
    terminal_budget = math.ceil(HIGH_DIM_VALLEY_ENTRY_TERMINAL_FOCUS_FRACTION * cma_total)
    opt._fe_count = opt._cmaes_budget - terminal_budget

    candidates = opt.ask()

    valley_n = HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS[HIGH_DIM_VALLEY_ENTRY_BRANCH]
    assert opt._portfolio_active_indices == (HIGH_DIM_VALLEY_ENTRY_BRANCH,)
    assert candidates.shape == (valley_n, opt.dim)


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_dim20_portfolio_keeps_full_batches_across_generations(
    device: torch.device,
) -> None:
    dtype = best_float_dtype(device)
    opt = PhasedDFO(
        dim=20,
        bounds=5.0,
        budget=40000,
        pop_size=40,
        device=device,
        dtype=dtype,
        seed=42,
    )
    _seed_high_dim_elite_state(opt)
    opt._phase = 1

    full_n = sum((24, 12, 12, 12))
    for generation in range(HIGH_DIM_VALLEY_ENTRY_FOCUS_CYCLE + 1):
        candidates = opt.ask()
        assert candidates.shape == (full_n, opt.dim)
        assert opt._portfolio_active_indices == (0, 1, 2, 3)
        opt.tell(candidates, sphere(candidates))
        assert opt._portfolio_generation == generation + 1

    opt._fe_count = opt._cmaes_budget - 1
    candidates = opt.ask()
    assert candidates.shape == (full_n, opt.dim)
    assert opt._portfolio_active_indices == (0, 1, 2, 3)
