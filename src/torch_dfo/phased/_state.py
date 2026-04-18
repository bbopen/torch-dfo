"""State bundles for :class:`torch_dfo.phased.orchestrator.PhasedDFO`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch_dfo._state_utils import (
    clone_optional_tensor,
    clone_tensor_list,
    restore_optional_tensor,
    restore_tensor_list,
)
from torch_dfo.cmaes import CMAES


@dataclass
class PhasedPolishState:
    """Mutable elite and search-pool state used by the polish phase."""

    elite_solutions: list[torch.Tensor]
    elite_fitness: list[torch.Tensor]
    search_population: torch.Tensor | None
    search_population_fitness: torch.Tensor | None
    search_pool_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "_elite_solutions": clone_tensor_list(self.elite_solutions),
            "_elite_fitness": clone_tensor_list(self.elite_fitness),
            "_search_population": clone_optional_tensor(self.search_population),
            "_search_population_fitness": clone_optional_tensor(self.search_population_fitness),
            "_search_pool_limit": self.search_pool_limit,
        }

    @classmethod
    def from_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PhasedPolishState:
        return cls(
            elite_solutions=restore_tensor_list(
                state["_elite_solutions"],
                device=device,
                dtype=dtype,
            ),
            elite_fitness=restore_tensor_list(
                state["_elite_fitness"],
                device=device,
                dtype=dtype,
            ),
            search_population=restore_optional_tensor(
                state["_search_population"],
                device=device,
                dtype=dtype,
            ),
            search_population_fitness=restore_optional_tensor(
                state["_search_population_fitness"],
                device=device,
                dtype=dtype,
            ),
            search_pool_limit=int(state["_search_pool_limit"]),
        )


@dataclass
class PhasedValleyState:
    """Mutable basin-explore, portfolio, and valley-focus counters."""

    basin_explore_restarts: int
    basin_explore_budget_frac: float
    basin_explore_stagnation: int
    portfolio_stag: list[int]
    portfolio_best_f: list[float]
    portfolio_branch_stag_limit: int
    portfolio_sigma0: list[float]
    portfolio_generation: int
    portfolio_active_indices: tuple[int, ...]
    valley_focus_remaining: int
    valley_focus_streak: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "_basin_explore_restarts": self.basin_explore_restarts,
            "_basin_explore_budget_frac": self.basin_explore_budget_frac,
            "_basin_explore_stagnation": self.basin_explore_stagnation,
            "_portfolio_stag": list(self.portfolio_stag),
            "_portfolio_best_f": list(self.portfolio_best_f),
            "_portfolio_sigma0": list(self.portfolio_sigma0),
            "_portfolio_generation": self.portfolio_generation,
            "_portfolio_active_indices": tuple(self.portfolio_active_indices),
            "_portfolio_branch_stag_limit": self.portfolio_branch_stag_limit,
            "_valley_focus_remaining": self.valley_focus_remaining,
            "_valley_focus_streak": self.valley_focus_streak,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> PhasedValleyState:
        return cls(
            basin_explore_restarts=int(state["_basin_explore_restarts"]),
            basin_explore_budget_frac=float(state["_basin_explore_budget_frac"]),
            basin_explore_stagnation=int(state["_basin_explore_stagnation"]),
            portfolio_stag=[int(v) for v in state["_portfolio_stag"]],
            portfolio_best_f=[float(v) for v in state["_portfolio_best_f"]],
            portfolio_branch_stag_limit=int(state["_portfolio_branch_stag_limit"]),
            portfolio_sigma0=[float(v) for v in state["_portfolio_sigma0"]],
            portfolio_generation=int(state["_portfolio_generation"]),
            portfolio_active_indices=tuple(int(v) for v in state["_portfolio_active_indices"]),
            valley_focus_remaining=int(state["_valley_focus_remaining"]),
            valley_focus_streak=int(state["_valley_focus_streak"]),
        )


@dataclass
class PhasedDEState:
    """Mutable DE phase progress, stagnation, and probe state."""

    phase: int
    fe_count: int
    high_dim: bool
    high_dim_de_min_pop: int
    restart_stagnation: int
    de_baseline_steps: int
    de_max_stagnation: int
    de_target_steps: int
    stagnation_counter: int
    de_progress_ema: float
    de_progress_baseline: float
    de_step_count: int
    de_best_f_prev: float
    de_phase_start_f: float
    trial_gain: float
    levy_gain: float
    accepted_ratio: float
    levy_ratio: float
    step_size: float
    midpoint_probed: bool
    de_restart_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "_phase": self.phase,
            "_fe_count": self.fe_count,
            "_high_dim": self.high_dim,
            "_high_dim_de_min_pop": self.high_dim_de_min_pop,
            "_restart_stagnation": self.restart_stagnation,
            "_de_baseline_steps": self.de_baseline_steps,
            "_de_max_stagnation": self.de_max_stagnation,
            "_de_target_steps": self.de_target_steps,
            "_stagnation_counter": self.stagnation_counter,
            "_de_progress_ema": self.de_progress_ema,
            "_de_progress_baseline": self.de_progress_baseline,
            "_de_step_count": self.de_step_count,
            "_de_best_f_prev": self.de_best_f_prev,
            "_de_phase_start_f": self.de_phase_start_f,
            "_trial_gain": self.trial_gain,
            "_levy_gain": self.levy_gain,
            "_accepted_ratio": self.accepted_ratio,
            "_levy_ratio": self.levy_ratio,
            "_step_size": self.step_size,
            "_midpoint_probed": self.midpoint_probed,
            "_de_restart_count": self.de_restart_count,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> PhasedDEState:
        return cls(
            phase=int(state["_phase"]),
            fe_count=int(state["_fe_count"]),
            high_dim=bool(state["_high_dim"]),
            high_dim_de_min_pop=int(state["_high_dim_de_min_pop"]),
            restart_stagnation=int(state["_restart_stagnation"]),
            de_baseline_steps=int(state["_de_baseline_steps"]),
            de_max_stagnation=int(state["_de_max_stagnation"]),
            de_target_steps=int(state["_de_target_steps"]),
            stagnation_counter=int(state["_stagnation_counter"]),
            de_progress_ema=float(state["_de_progress_ema"]),
            de_progress_baseline=float(state["_de_progress_baseline"]),
            de_step_count=int(state["_de_step_count"]),
            de_best_f_prev=float(state["_de_best_f_prev"]),
            de_phase_start_f=float(state["_de_phase_start_f"]),
            trial_gain=float(state["_trial_gain"]),
            levy_gain=float(state["_levy_gain"]),
            accepted_ratio=float(state["_accepted_ratio"]),
            levy_ratio=float(state["_levy_ratio"]),
            step_size=float(state["_step_size"]),
            midpoint_probed=bool(state["_midpoint_probed"]),
            de_restart_count=int(state["_de_restart_count"]),
        )


@dataclass
class PhasedCMAState:
    """Mutable CMA-ES phase and nested optimizer state."""

    cmaes: CMAES | None
    cmaes_phase_idx: int
    cmaes_portfolio: list[CMAES] | None
    cmaes_fe_start: int
    cmaes_phase_count: int
    cmaes_phase_budgets: list[int]
    cmaes_stagnation_counter: int
    cmaes_phase_best_f: float
    cmaes_overall_start_f: float
    cmaes_entered: bool
    cmaes_base_pop: int
    cma_sigma_min: float
    cma_sigma_max: float
    cma_restart_sigma_min: float
    cma_restart_sigma_max: float
    cma_stagnation: int
    high_dim_cma_stagnation: int
    high_dim_cma_warm_stagnation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "_cmaes_phase_idx": self.cmaes_phase_idx,
            "_cmaes_fe_start": self.cmaes_fe_start,
            "_cmaes_phase_count": self.cmaes_phase_count,
            "_cmaes_phase_budgets": list(self.cmaes_phase_budgets),
            "_cmaes_stagnation_counter": self.cmaes_stagnation_counter,
            "_cmaes_phase_best_f": self.cmaes_phase_best_f,
            "_cmaes_overall_start_f": self.cmaes_overall_start_f,
            "_cmaes_entered": self.cmaes_entered,
            "_cmaes_base_pop": self.cmaes_base_pop,
            "_cma_sigma_min": self.cma_sigma_min,
            "_cma_sigma_max": self.cma_sigma_max,
            "_cma_restart_sigma_min": self.cma_restart_sigma_min,
            "_cma_restart_sigma_max": self.cma_restart_sigma_max,
            "_cma_stagnation": self.cma_stagnation,
            "_high_dim_cma_stagnation": self.high_dim_cma_stagnation,
            "_high_dim_cma_warm_stagnation": self.high_dim_cma_warm_stagnation,
            "_cmaes_state": self.cmaes.state_dict() if self.cmaes is not None else None,
            "_cmaes_portfolio_states": (
                [c.state_dict() for c in self.cmaes_portfolio]
                if self.cmaes_portfolio is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        state: dict[str, Any],
        *,
        dim: int,
        bounds: float | tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> PhasedCMAState:
        cmaes = _restore_cmaes(state.get("_cmaes_state"), dim, bounds, device, dtype)
        portfolio_states = state.get("_cmaes_portfolio_states")
        cmaes_portfolio = (
            [
                restored
                for restored in (
                    _restore_cmaes(branch_state, dim, bounds, device, dtype)
                    for branch_state in portfolio_states
                )
                if restored is not None
            ]
            if portfolio_states is not None
            else None
        )
        return cls(
            cmaes=cmaes,
            cmaes_phase_idx=int(state["_cmaes_phase_idx"]),
            cmaes_portfolio=cmaes_portfolio,
            cmaes_fe_start=int(state["_cmaes_fe_start"]),
            cmaes_phase_count=int(state["_cmaes_phase_count"]),
            cmaes_phase_budgets=[int(v) for v in state["_cmaes_phase_budgets"]],
            cmaes_stagnation_counter=int(state["_cmaes_stagnation_counter"]),
            cmaes_phase_best_f=float(state["_cmaes_phase_best_f"]),
            cmaes_overall_start_f=float(state["_cmaes_overall_start_f"]),
            cmaes_entered=bool(state["_cmaes_entered"]),
            cmaes_base_pop=int(state["_cmaes_base_pop"]),
            cma_sigma_min=float(state["_cma_sigma_min"]),
            cma_sigma_max=float(state["_cma_sigma_max"]),
            cma_restart_sigma_min=float(state["_cma_restart_sigma_min"]),
            cma_restart_sigma_max=float(state["_cma_restart_sigma_max"]),
            cma_stagnation=int(state["_cma_stagnation"]),
            high_dim_cma_stagnation=int(state["_high_dim_cma_stagnation"]),
            high_dim_cma_warm_stagnation=int(state["_high_dim_cma_warm_stagnation"]),
        )


def _restore_cmaes(
    state: dict[str, Any] | None,
    dim: int,
    bounds: float | tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> CMAES | None:
    if state is None:
        return None
    pop_size = int(state["population"].shape[0])
    opt = CMAES(
        dim=dim,
        bounds=bounds,
        pop_size=pop_size,
        device=device,
        dtype=dtype,
        active=bool(state.get("active", False)),
        mirrored=bool(state.get("mirrored", False)),
    )
    opt.load_state_dict(state)
    return opt
