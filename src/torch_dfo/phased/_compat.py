"""Compatibility properties for PhasedDFO historical private state names."""

from __future__ import annotations

import torch

from torch_dfo.cmaes import CMAES
from torch_dfo.phased._state import (
    PhasedCMAState,
    PhasedDEState,
    PhasedPolishState,
    PhasedValleyState,
)


class PhasedStateCompatMixin:
    """Expose the pre-state-bundle private attribute surface."""

    _polish: PhasedPolishState
    _valley: PhasedValleyState
    _de: PhasedDEState
    _cma: PhasedCMAState

    @property
    def _elite_solutions(self) -> list[torch.Tensor]:
        return self._polish.elite_solutions

    @_elite_solutions.setter
    def _elite_solutions(self, value: list[torch.Tensor]) -> None:
        self._polish.elite_solutions = value

    @property
    def _elite_fitness(self) -> list[torch.Tensor]:
        return self._polish.elite_fitness

    @_elite_fitness.setter
    def _elite_fitness(self, value: list[torch.Tensor]) -> None:
        self._polish.elite_fitness = value

    @property
    def _search_population(self) -> torch.Tensor | None:
        return self._polish.search_population

    @_search_population.setter
    def _search_population(self, value: torch.Tensor | None) -> None:
        self._polish.search_population = value

    @property
    def _search_population_fitness(self) -> torch.Tensor | None:
        return self._polish.search_population_fitness

    @_search_population_fitness.setter
    def _search_population_fitness(self, value: torch.Tensor | None) -> None:
        self._polish.search_population_fitness = value

    @property
    def _search_pool_limit(self) -> int:
        return self._polish.search_pool_limit

    @_search_pool_limit.setter
    def _search_pool_limit(self, value: int) -> None:
        self._polish.search_pool_limit = int(value)

    @property
    def _basin_explore_restarts(self) -> int:
        return self._valley.basin_explore_restarts

    @_basin_explore_restarts.setter
    def _basin_explore_restarts(self, value: int) -> None:
        self._valley.basin_explore_restarts = int(value)

    @property
    def _basin_explore_budget_frac(self) -> float:
        return self._valley.basin_explore_budget_frac

    @_basin_explore_budget_frac.setter
    def _basin_explore_budget_frac(self, value: float) -> None:
        self._valley.basin_explore_budget_frac = float(value)

    @property
    def _basin_explore_stagnation(self) -> int:
        return self._valley.basin_explore_stagnation

    @_basin_explore_stagnation.setter
    def _basin_explore_stagnation(self, value: int) -> None:
        self._valley.basin_explore_stagnation = int(value)

    @property
    def _portfolio_stag(self) -> list[int]:
        return self._valley.portfolio_stag

    @_portfolio_stag.setter
    def _portfolio_stag(self, value: list[int]) -> None:
        self._valley.portfolio_stag = value

    @property
    def _portfolio_best_f(self) -> list[float]:
        return self._valley.portfolio_best_f

    @_portfolio_best_f.setter
    def _portfolio_best_f(self, value: list[float]) -> None:
        self._valley.portfolio_best_f = value

    @property
    def _portfolio_branch_stag_limit(self) -> int:
        return self._valley.portfolio_branch_stag_limit

    @_portfolio_branch_stag_limit.setter
    def _portfolio_branch_stag_limit(self, value: int) -> None:
        self._valley.portfolio_branch_stag_limit = int(value)

    @property
    def _portfolio_sigma0(self) -> list[float]:
        return self._valley.portfolio_sigma0

    @_portfolio_sigma0.setter
    def _portfolio_sigma0(self, value: list[float]) -> None:
        self._valley.portfolio_sigma0 = value

    @property
    def _portfolio_generation(self) -> int:
        return self._valley.portfolio_generation

    @_portfolio_generation.setter
    def _portfolio_generation(self, value: int) -> None:
        self._valley.portfolio_generation = int(value)

    @property
    def _portfolio_active_indices(self) -> tuple[int, ...]:
        return self._valley.portfolio_active_indices

    @_portfolio_active_indices.setter
    def _portfolio_active_indices(self, value: tuple[int, ...]) -> None:
        self._valley.portfolio_active_indices = tuple(value)

    @property
    def _valley_focus_remaining(self) -> int:
        return self._valley.valley_focus_remaining

    @_valley_focus_remaining.setter
    def _valley_focus_remaining(self, value: int) -> None:
        self._valley.valley_focus_remaining = int(value)

    @property
    def _valley_focus_streak(self) -> int:
        return self._valley.valley_focus_streak

    @_valley_focus_streak.setter
    def _valley_focus_streak(self, value: int) -> None:
        self._valley.valley_focus_streak = int(value)

    @property
    def _phase(self) -> int:
        return self._de.phase

    @_phase.setter
    def _phase(self, value: int) -> None:
        self._de.phase = int(value)

    @property
    def _fe_count(self) -> int:
        return self._de.fe_count

    @_fe_count.setter
    def _fe_count(self, value: int) -> None:
        self._de.fe_count = int(value)

    @property
    def _high_dim(self) -> bool:
        return self._de.high_dim

    @_high_dim.setter
    def _high_dim(self, value: bool) -> None:
        self._de.high_dim = bool(value)

    @property
    def _high_dim_de_min_pop(self) -> int:
        return self._de.high_dim_de_min_pop

    @_high_dim_de_min_pop.setter
    def _high_dim_de_min_pop(self, value: int) -> None:
        self._de.high_dim_de_min_pop = int(value)

    @property
    def _restart_stagnation(self) -> int:
        return self._de.restart_stagnation

    @_restart_stagnation.setter
    def _restart_stagnation(self, value: int) -> None:
        self._de.restart_stagnation = int(value)

    @property
    def _de_baseline_steps(self) -> int:
        return self._de.de_baseline_steps

    @_de_baseline_steps.setter
    def _de_baseline_steps(self, value: int) -> None:
        self._de.de_baseline_steps = int(value)

    @property
    def _de_max_stagnation(self) -> int:
        return self._de.de_max_stagnation

    @_de_max_stagnation.setter
    def _de_max_stagnation(self, value: int) -> None:
        self._de.de_max_stagnation = int(value)

    @property
    def _de_target_steps(self) -> int:
        return self._de.de_target_steps

    @_de_target_steps.setter
    def _de_target_steps(self, value: int) -> None:
        self._de.de_target_steps = int(value)

    @property
    def _stagnation_counter(self) -> int:
        return self._de.stagnation_counter

    @_stagnation_counter.setter
    def _stagnation_counter(self, value: int) -> None:
        self._de.stagnation_counter = int(value)

    @property
    def _de_progress_ema(self) -> float:
        return self._de.de_progress_ema

    @_de_progress_ema.setter
    def _de_progress_ema(self, value: float) -> None:
        self._de.de_progress_ema = float(value)

    @property
    def _de_progress_baseline(self) -> float:
        return self._de.de_progress_baseline

    @_de_progress_baseline.setter
    def _de_progress_baseline(self, value: float) -> None:
        self._de.de_progress_baseline = float(value)

    @property
    def _de_step_count(self) -> int:
        return self._de.de_step_count

    @_de_step_count.setter
    def _de_step_count(self, value: int) -> None:
        self._de.de_step_count = int(value)

    @property
    def _de_best_f_prev(self) -> float:
        return self._de.de_best_f_prev

    @_de_best_f_prev.setter
    def _de_best_f_prev(self, value: float) -> None:
        self._de.de_best_f_prev = float(value)

    @property
    def _de_phase_start_f(self) -> float:
        return self._de.de_phase_start_f

    @_de_phase_start_f.setter
    def _de_phase_start_f(self, value: float) -> None:
        self._de.de_phase_start_f = float(value)

    @property
    def _trial_gain(self) -> float:
        return self._de.trial_gain

    @_trial_gain.setter
    def _trial_gain(self, value: float) -> None:
        self._de.trial_gain = float(value)

    @property
    def _levy_gain(self) -> float:
        return self._de.levy_gain

    @_levy_gain.setter
    def _levy_gain(self, value: float) -> None:
        self._de.levy_gain = float(value)

    @property
    def _accepted_ratio(self) -> float:
        return self._de.accepted_ratio

    @_accepted_ratio.setter
    def _accepted_ratio(self, value: float) -> None:
        self._de.accepted_ratio = float(value)

    @property
    def _levy_ratio(self) -> float:
        return self._de.levy_ratio

    @_levy_ratio.setter
    def _levy_ratio(self, value: float) -> None:
        self._de.levy_ratio = float(value)

    @property
    def _step_size(self) -> float:
        return self._de.step_size

    @_step_size.setter
    def _step_size(self, value: float) -> None:
        self._de.step_size = float(value)

    @property
    def _midpoint_probed(self) -> bool:
        return self._de.midpoint_probed

    @_midpoint_probed.setter
    def _midpoint_probed(self, value: bool) -> None:
        self._de.midpoint_probed = bool(value)

    @property
    def _de_restart_count(self) -> int:
        return self._de.de_restart_count

    @_de_restart_count.setter
    def _de_restart_count(self, value: int) -> None:
        self._de.de_restart_count = int(value)

    @property
    def _cmaes(self) -> CMAES | None:
        return self._cma.cmaes

    @_cmaes.setter
    def _cmaes(self, value: CMAES | None) -> None:
        self._cma.cmaes = value

    @property
    def _cmaes_phase_idx(self) -> int:
        return self._cma.cmaes_phase_idx

    @_cmaes_phase_idx.setter
    def _cmaes_phase_idx(self, value: int) -> None:
        self._cma.cmaes_phase_idx = int(value)

    @property
    def _cmaes_portfolio(self) -> list[CMAES] | None:
        return self._cma.cmaes_portfolio

    @_cmaes_portfolio.setter
    def _cmaes_portfolio(self, value: list[CMAES] | None) -> None:
        self._cma.cmaes_portfolio = value

    @property
    def _cmaes_fe_start(self) -> int:
        return self._cma.cmaes_fe_start

    @_cmaes_fe_start.setter
    def _cmaes_fe_start(self, value: int) -> None:
        self._cma.cmaes_fe_start = int(value)

    @property
    def _cmaes_phase_count(self) -> int:
        return self._cma.cmaes_phase_count

    @_cmaes_phase_count.setter
    def _cmaes_phase_count(self, value: int) -> None:
        self._cma.cmaes_phase_count = int(value)

    @property
    def _cmaes_phase_budgets(self) -> list[int]:
        return self._cma.cmaes_phase_budgets

    @_cmaes_phase_budgets.setter
    def _cmaes_phase_budgets(self, value: list[int]) -> None:
        self._cma.cmaes_phase_budgets = value

    @property
    def _cmaes_stagnation_counter(self) -> int:
        return self._cma.cmaes_stagnation_counter

    @_cmaes_stagnation_counter.setter
    def _cmaes_stagnation_counter(self, value: int) -> None:
        self._cma.cmaes_stagnation_counter = int(value)

    @property
    def _cmaes_phase_best_f(self) -> float:
        return self._cma.cmaes_phase_best_f

    @_cmaes_phase_best_f.setter
    def _cmaes_phase_best_f(self, value: float) -> None:
        self._cma.cmaes_phase_best_f = float(value)

    @property
    def _cmaes_overall_start_f(self) -> float:
        return self._cma.cmaes_overall_start_f

    @_cmaes_overall_start_f.setter
    def _cmaes_overall_start_f(self, value: float) -> None:
        self._cma.cmaes_overall_start_f = float(value)

    @property
    def _cmaes_entered(self) -> bool:
        return self._cma.cmaes_entered

    @_cmaes_entered.setter
    def _cmaes_entered(self, value: bool) -> None:
        self._cma.cmaes_entered = bool(value)

    @property
    def _cmaes_base_pop(self) -> int:
        return self._cma.cmaes_base_pop

    @_cmaes_base_pop.setter
    def _cmaes_base_pop(self, value: int) -> None:
        self._cma.cmaes_base_pop = int(value)

    @property
    def _cma_sigma_min(self) -> float:
        return self._cma.cma_sigma_min

    @_cma_sigma_min.setter
    def _cma_sigma_min(self, value: float) -> None:
        self._cma.cma_sigma_min = float(value)

    @property
    def _cma_sigma_max(self) -> float:
        return self._cma.cma_sigma_max

    @_cma_sigma_max.setter
    def _cma_sigma_max(self, value: float) -> None:
        self._cma.cma_sigma_max = float(value)

    @property
    def _cma_restart_sigma_min(self) -> float:
        return self._cma.cma_restart_sigma_min

    @_cma_restart_sigma_min.setter
    def _cma_restart_sigma_min(self, value: float) -> None:
        self._cma.cma_restart_sigma_min = float(value)

    @property
    def _cma_restart_sigma_max(self) -> float:
        return self._cma.cma_restart_sigma_max

    @_cma_restart_sigma_max.setter
    def _cma_restart_sigma_max(self, value: float) -> None:
        self._cma.cma_restart_sigma_max = float(value)

    @property
    def _cma_stagnation(self) -> int:
        return self._cma.cma_stagnation

    @_cma_stagnation.setter
    def _cma_stagnation(self, value: int) -> None:
        self._cma.cma_stagnation = int(value)

    @property
    def _high_dim_cma_stagnation(self) -> int:
        return self._cma.high_dim_cma_stagnation

    @_high_dim_cma_stagnation.setter
    def _high_dim_cma_stagnation(self, value: int) -> None:
        self._cma.high_dim_cma_stagnation = int(value)

    @property
    def _high_dim_cma_warm_stagnation(self) -> int:
        return self._cma.high_dim_cma_warm_stagnation

    @_high_dim_cma_warm_stagnation.setter
    def _high_dim_cma_warm_stagnation(self, value: int) -> None:
        self._cma.high_dim_cma_warm_stagnation = int(value)
