"""PhasedDFO: Multi-phase derivative-free optimizer.

Three-phase budget-managed pipeline:
  1. SHADE-DE warmup  -- quick exploration with Levy flight perturbation
  2. IPOP-CMA-ES      -- exploitation with population-doubling restarts
  3. Polish            -- directional + coordinate + FD-BFGS refinement

References
----------
Tanabe & Fukunaga, "Success-History Based Parameter Adaptation for
    Differential Evolution" (2013).
Hansen, "The CMA Evolution Strategy: A Tutorial" (2016).

"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, overload

import torch

from torch_dfo._operators import levy_flight_perturbation
from torch_dfo.base import BaseOptimizer
from torch_dfo.cmaes import CMAES
from torch_dfo.phased import _basin_explore as _be
from torch_dfo.phased import _cma_phase as _cma
from torch_dfo.phased import _de_phase as _de
from torch_dfo.phased import _polish_phase as _pp
from torch_dfo.phased._common import _merge_search_pool
from torch_dfo.phased._compat import PhasedStateCompatMixin
from torch_dfo.phased._config import PhasedConfig
from torch_dfo.phased._state import (
    PhasedCMAState,
    PhasedDEState,
    PhasedPolishState,
    PhasedValleyState,
)
from torch_dfo.shade import SHADE
from torch_dfo.space import SearchSpace
from torch_dfo.utils import clamp_to_bounds

# ---------------------------------------------------------------------------
# Hyperparameter constants
# ---------------------------------------------------------------------------
# Population sizing, budget allocation, and polish sub-allocation are all
# computed as functions of (dim, budget) via the _compute_* helpers below.
# User-tunable knobs now live on :class:`PhasedConfig` in ``_config.py``;
# the module-level names below are aliases derived from the dataclass field
# defaults and are kept for backward-compatible imports.  The rationale for
# why each knob is "universal" (does not need to scale with dim/budget) is
# preserved alongside the field declarations in ``_config.py``.
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = PhasedConfig()

# DE phase — progress tracking (relative measures, not absolute counts)
HIGH_DIM_DE_PROGRESS_RATIO = _CONFIG_DEFAULTS.high_dim_de_progress_ratio
HIGH_DIM_DE_PROGRESS_FLOOR = _CONFIG_DEFAULTS.high_dim_de_progress_floor

# Adaptive Levy step size (fractions of span)
STEP_SIZE_MIN = _CONFIG_DEFAULTS.step_size_min
STEP_SIZE_MAX = _CONFIG_DEFAULTS.step_size_max

# Low-dim population restart
ELITE_FRACTION = _CONFIG_DEFAULTS.elite_fraction

# IPOP doubling factor (Hansen, IPOP-CMA-ES).  Standard algorithm constant.
CMA_ES_POP_GROWTH = _CONFIG_DEFAULTS.cma_es_pop_growth

# Restart covariance blend: 80% inherited covariance + 20% identity.
CMA_ES_RESTART_COV_BLEND = _CONFIG_DEFAULTS.cma_es_restart_cov_blend

# CMA-ES restart modes: 4 center modes (random, elite anchor, differential,
# mirrored-best).
CMA_ES_RESTART_MODES = _CONFIG_DEFAULTS.cma_es_restart_modes

# K=4 parallel portfolio for high-dim CMA-ES phase.
_K_PORTFOLIO_LAMBDAS: tuple[int, ...] = _CONFIG_DEFAULTS.k_portfolio_lambdas
_K_PORTFOLIO_SIGMA_FRACS: tuple[float, ...] = _CONFIG_DEFAULTS.k_portfolio_sigma_fracs

# Dim>=40 valley-entry branch.
HIGH_DIM_VALLEY_ENTRY_DIM = _CONFIG_DEFAULTS.high_dim_valley_entry_dim
HIGH_DIM_VALLEY_ENTRY_BRANCH = _CONFIG_DEFAULTS.high_dim_valley_entry_branch
HIGH_DIM_VALLEY_ENTRY_PATH_MEMORY = _CONFIG_DEFAULTS.high_dim_valley_entry_path_memory
HIGH_DIM_VALLEY_ENTRY_PATH_SCALE = _CONFIG_DEFAULTS.high_dim_valley_entry_path_scale
HIGH_DIM_VALLEY_ENTRY_LINE_SAMPLES = _CONFIG_DEFAULTS.high_dim_valley_entry_line_samples
HIGH_DIM_VALLEY_ENTRY_LINE_SCALE = _CONFIG_DEFAULTS.high_dim_valley_entry_line_scale
HIGH_DIM_VALLEY_ENTRY_RESTART_JITTER = _CONFIG_DEFAULTS.high_dim_valley_entry_restart_jitter
# Dim>=40 CMA portfolio population schedule.
HIGH_DIM_VALLEY_ENTRY_PORTFOLIO_LAMBDAS: tuple[int, ...] = (
    _CONFIG_DEFAULTS.high_dim_valley_entry_portfolio_lambdas
)
# Dim>=40 focus-burst scheduling.
HIGH_DIM_VALLEY_ENTRY_FOCUS_CYCLE = _CONFIG_DEFAULTS.high_dim_valley_entry_focus_cycle
HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_CYCLE = _CONFIG_DEFAULTS.high_dim_valley_entry_max_focus_cycle
HIGH_DIM_VALLEY_ENTRY_FOCUS_EVAL_RATIO = _CONFIG_DEFAULTS.high_dim_valley_entry_focus_eval_ratio
HIGH_DIM_VALLEY_ENTRY_MAX_FOCUS_EVAL_RATIO = (
    _CONFIG_DEFAULTS.high_dim_valley_entry_max_focus_eval_ratio
)
# Dim>=40 terminal focus window.
HIGH_DIM_VALLEY_ENTRY_TERMINAL_FOCUS_FRACTION = (
    _CONFIG_DEFAULTS.high_dim_valley_entry_terminal_focus_fraction
)

# Jitter scales as fractions of span.  These control exploration radius
# around each restart center mode.
CMA_ES_ELITE_RESTART_JITTER = _CONFIG_DEFAULTS.cma_es_elite_restart_jitter
CMA_ES_DIFFERENTIAL_RESTART_SCALE = _CONFIG_DEFAULTS.cma_es_differential_restart_scale
CMA_ES_DIFFERENTIAL_RESTART_JITTER = _CONFIG_DEFAULTS.cma_es_differential_restart_jitter
CMA_ES_MIRROR_RESTART_JITTER = _CONFIG_DEFAULTS.cma_es_mirror_restart_jitter

# Search pool expansion factor.
CMA_ES_SEARCH_POOL_FACTOR = _CONFIG_DEFAULTS.cma_es_search_pool_factor

# Directional line-search grid resolution.
DIRECTIONAL_REFINEMENT_STAGES = _CONFIG_DEFAULTS.directional_refinement_stages
DIRECTIONAL_REFINEMENT_POINTS = _CONFIG_DEFAULTS.directional_refinement_points
DIRECTIONAL_WINDOW_SHRINK = _CONFIG_DEFAULTS.directional_window_shrink

# Priority hop scale.
DIRECTIONAL_PRIORITY_HOP_SCALE = _CONFIG_DEFAULTS.directional_priority_hop_scale

# Coordinate line-search grid resolution.
COORDINATE_REFINEMENT_STAGES = _CONFIG_DEFAULTS.coordinate_refinement_stages
COORDINATE_REFINEMENT_POINTS = _CONFIG_DEFAULTS.coordinate_refinement_points
COORDINATE_WINDOW_SHRINK = _CONFIG_DEFAULTS.coordinate_window_shrink


# ---------------------------------------------------------------------------
# Adaptive population-sizing formulas
# ---------------------------------------------------------------------------
# Each function takes (dim, budget) and returns a value that approximates the
# old hardcoded constant at the BBOB reference point (dim=10, budget=50000)
# while scaling sensibly elsewhere.


def _compute_pop_size(dim: int, budget: int) -> int:
    """Compute initial DE population size from (dim, budget).

    Old formula: max(80, min(256, budget // 300))
    - At dim=10, budget=50000: gave pop=166
    - At dim=5, budget=500:   gave pop=80 (wastes 80% of budget!)

    New formula:  clamp(budget // (15*dim + 150), floor=budget_aware_floor, cap=min(256, budget//3))
    The divisor ``15*dim + 150`` ensures the ratio budget/pop yields enough
    generations to converge (~300 at the reference point), while the floor
    is budget-aware: ``min(4*dim + 10, max(4, budget // min_gens))`` where
    ``min_gens = max(4, 8 - dim // 10)``.

    The minimum-generations value scales inversely with dimension because each
    generation is more expensive (in exploration coverage) at higher dim, so
    fewer larger generations outperform more smaller generations.  Specifically:

    - dim <  10: min_gens = 8  (8 tight generations minimum)
    - dim = 10:  min_gens = 7
    - dim = 20:  min_gens = 6
    - dim = 30:  min_gens = 5
    - dim = 40+: min_gens = 4  (4 generations minimum at high dim)

    At generous budgets (budget // min_gens >= 4*dim + 10) the floor reverts to
    the dimension-only formula so behaviour is unchanged.  At tight budgets
    relative to dim (e.g. dim=30, budget=500) the old floor of ``4*dim+10=130``
    would leave only ~3 generations; the new floor caps it at
    ``budget // 5 = 100``, giving ~5 generations.  Crucially, at dim=41,
    budget=1000 the cap is ``1000 // 4 = 250 > floor=174``, so the floor stays
    at 174 and behaviour is identical to pre-Exp9 (fixes regression).

    Reference-point verification:
    - dim=10, budget=50000: min_gens=7, cap=7142 > floor=50 -> floor=50; raw=166 -> 166. Unchanged.
    - dim=5,  budget=500:   min_gens=8, cap=62 > floor=30   -> floor=30; raw=10 -> 30. 16+ gens.
    - dim=30, budget=500:   min_gens=5, cap=100 < floor=130 -> floor=100; raw=1 -> 100. 5 gens.
    - dim=41, budget=500:   min_gens=4, cap=125 < floor=174 -> floor=125; raw=1 -> 125. 4 gens.
    - dim=41, budget=1000:  min_gens=4, cap=250 > floor=174 -> floor=174; raw=1 -> 174. Unchanged.
    - dim=50, budget=1000000: raw=1111, capped to 256. Unchanged.
    """
    divisor = 15 * dim + 150
    raw = budget // divisor if divisor > 0 else budget
    min_gens = max(4, 8 - dim // 10)
    floor = min(4 * dim + 10, max(4, budget // min_gens))
    ceiling = min(256, budget // 3) if budget >= 3 else 1
    return max(floor, min(ceiling, raw))


def _compute_high_dim_de_min_pop(dim: int) -> int:
    """Minimum DE population after high-dim linear reduction.

    Old value: 28 (constant).

    New formula: dim + 8.
    Scales linearly with dimension because DE's differential recombination
    operator needs O(dim) distinct individuals to span the search space.
    At dim=20 (the high-dim threshold): 20 + 8 = 28.  Exact match.

    Reference-point verification:
    - dim=20: 28 (matches old value at high-dim threshold)
    - dim=50: 58
    - dim=100: 108
    """
    return dim + 8


def _compute_basin_explore_restarts(dim: int, budget: int) -> int:
    """Number of short CMA-ES restarts for low-dim multistart basin exploration.

    Old value: 12 (constant).

    New formula: clamp(budget // (400 * dim), floor=3, ceiling=20).
    At high budget the optimizer can afford many restarts for thorough
    multi-basin coverage. At low budget only a few restarts are sensible
    so each gets enough evaluations to converge.

    Reference-point verification:
    - dim=10, budget=50000: 50000 // 4000 = 12.  Exact match.
    - dim=5,  budget=500:   500 // 2000 = 0 -> clamped to 3.
    - dim=5,  budget=50000: 50000 // 2000 = 25 -> clamped to 20.
    - dim=2,  budget=100:   100 // 800 = 0 -> clamped to 3.
    """
    divisor = 400 * dim
    raw = budget // divisor if divisor > 0 else budget
    return max(3, min(20, raw))


def _compute_search_pool_max(dim: int, pop_size: int) -> int:
    """Maximum size of the elite search pool across CMA-ES phases.

    Old value: 64 (constant).

    New formula: clamp(2 * pop_size, floor=32, ceiling=256).
    The search pool stores elite candidates used for restart seeding and
    polish direction computation.  It should grow with population size
    (which already encodes dim and budget information) but not without
    bound, since diminishing returns set in quickly.

    Reference-point verification:
    - dim=10, budget=50000 (pop=166): 2*166=332 -> clamped to 256.
      Old effective pool at dim=10: max(20, min(64, 40)) = 40.
      The 64 cap meant at most 64; new ceiling of 256 is more generous
      but the _enter_cmaes_phase logic (which uses CMA_ES_SEARCH_POOL_FACTOR
      and elite_snapshot_size) still provides a tighter per-instance bound.
    - dim=5,  budget=500 (pop=30):  2*30=60 -> 60.
    - dim=50, budget=1000000 (pop=256): 2*256=512 -> clamped to 256.
    """
    return max(32, min(256, 2 * pop_size))


# ---------------------------------------------------------------------------
# Budget-allocation formulas (Experiment A2)
# ---------------------------------------------------------------------------
# These replace the hardcoded budget-fraction constants with functions of
# (dim, budget, pop_size).  The key insight: the "generosity ratio"
# R = budget / (100 * dim) characterises how much budget is available
# relative to the problem size.  R >> 10 means generous (can afford lots
# of polish), R < 1 means tight (minimise overhead, maximise exploration).


def _compute_polish_reserve(dim: int, budget: int, high_dim: bool) -> int:
    """Compute evaluation budget reserved for the polish phase.

    Old values:
    - Low-dim:  max(10*dim, budget * 0.25)  -> reserve = that value
    - High-dim: reserve = budget * 0.08

    New formula uses the generosity ratio R = budget / (100 * dim).
    - Low-dim:  frac = clamp(0.15 + 0.10 / (1 + e^(-(R-3))), 0.15, 0.30)
      A sigmoid that smoothly transitions from 15% at tight budgets to
      ~25% at generous budgets.  At reference (R=50): 0.25.
      At budget=500 dim=10 (R=0.5): ~0.15 -- saves budget for DE+CMA.
    - High-dim: frac = clamp(0.05 + 0.15 / (1 + e^(-(R-3))), 0.05, 0.22)
      A sigmoid centered at R=3 that transitions from 5% at tight budgets
      to ~20% at generous budgets.  At reference (R=50): ~0.20.

      Rationale for the change from 0.08 to 0.20: the high-dim polish pipeline
      runs smoothed-envelope search, directional basin search, FD-BFGS, and then
      the full scipy chain (L-BFGS-B → Powell → FD-BFGS → Nelder-Mead).  For
      20-dim ill-conditioned functions (f10-f14: rotated ellipsoids, discus,
      bent cigar, sharp ridge, sum-of-different-powers), the scipy L-BFGS-B call
      alone needs O(dim × iterations × gradient-evals) evaluations to drive
      precision from ~1e-4 to 1e-8.  At dim=20 this requires ~3000-8000 evals
      for L-BFGS-B and ~2000-4000 for Powell.  With only 8% (8000 evals) reserved,
      the envelope+directional searches consume most of that budget before scipy
      even starts, leaving <2000 evals for the gradient-based polishers -- far
      too few to reach 1e-8 precision.  20% (20000 evals) gives each polisher
      adequate budget.

      The tight-budget floor of 5% (vs 4% before) ensures the minimum floor
      max(10*dim, ...) still kicks in for very small budgets.

    The floor max(10*dim, ...) is preserved -- polish always needs at
    least a few evaluations per dimension.

    Reference-point verification:
    - Low-dim  dim=10, budget=50000 (R=50): sigmoid~1.0, frac=0.25, reserve=12500.
    - Low-dim  dim=10, budget=500   (R=0.5): sigmoid~0.08, frac=0.158, reserve=100(floor).
    - Low-dim  dim=5,  budget=2500  (R=5):  sigmoid~0.88, frac=0.238, reserve=595.
    - High-dim dim=20, budget=100000(R=50): sigmoid~1.0, frac=0.20, reserve=20000.
    - High-dim dim=20, budget=10000 (R=5):  sigmoid~0.88, frac=0.183, reserve=1830.
    - High-dim dim=20, budget=2000  (R=1):  sigmoid~0.12, frac=0.068, reserve=200(floor).
    """
    R = budget / max(100 * dim, 1)
    if high_dim:
        sigmoid = 1.0 / (1.0 + math.exp(-(R - 3)))
        frac = 0.05 + 0.15 * sigmoid
        frac = max(0.05, min(0.22, frac))
    else:
        sigmoid = 1.0 / (1.0 + math.exp(-(R - 3)))
        frac = 0.15 + 0.10 * sigmoid
        frac = max(0.15, min(0.30, frac))
    # Cap the 10*dim floor at budget//4 for tight budgets (R < 0.4).
    # When budget < 40*dim, the 10*dim floor consumes >25% of the budget,
    # starving DE and CMA-ES phases.  At exactly budget=40*dim (R=0.4),
    # 10*dim == budget//4, so the formula is continuous at the boundary.
    dim_floor = 10 * dim if budget >= 40 * dim else budget // 4
    return max(dim_floor, int(budget * frac))


def _compute_de_headroom(dim: int, budget: int, high_dim: bool) -> int:
    """Compute DE headroom (reserved_polish used for DE budget cap in high-dim).

    Old value: max(10*dim, budget * 0.06) for high-dim.

    New formula: max(10*dim, budget * clamp(0.03 + 0.03*sigmoid(R-5), 0.03, 0.10))
    At generous budgets, reserves ~6% as headroom between DE cap and CMA budget.
    At tight budgets, drops to ~3% to leave more for DE.

    Only used in high-dim path; low-dim uses _compute_polish_reserve directly.

    Reference-point verification:
    - dim=10, budget=50000 (R=50): frac=0.06, headroom=3000.  Matches old.
    - dim=20, budget=100000 (R=50): frac=0.06, headroom=6000.
    - dim=20, budget=2000 (R=1): frac=0.031, headroom=200(floor 10*dim=200).
    """
    R = budget / max(100 * dim, 1)
    sigmoid = 1.0 / (1.0 + math.exp(-(R - 5)))
    frac = 0.03 + 0.03 * sigmoid
    frac = max(0.03, min(0.10, frac))
    # Same tight-budget cap as _compute_polish_reserve: limit 10*dim floor to
    # budget//4 when R < 0.4 so DE gets a meaningful allocation.
    dim_floor = 10 * dim if budget >= 40 * dim else budget // 4
    return max(dim_floor, int(budget * frac))


def _compute_de_budget_cap(
    dim: int,
    budget: int,
    pop_size: int,
    high_dim: bool,
    reserved_polish: int,
) -> int:
    """Compute maximum evaluation budget for the DE phase.

    Old values:
    - Low-dim:  min(budget - reserved_polish, budget * 0.40)
    - High-dim: min(budget - reserved_polish, pop_size * 1.5 * 200)

    New formula uses the generosity ratio R = budget / (100 * dim).
    - Low-dim: frac = clamp(0.30 + 0.10*sigmoid(R-3), 0.30, 0.45)
      At reference (R=50): 0.40 (matches old).  At tight (R=0.5): ~0.31.
      Tight budgets give DE a smaller share -- with few evals, CMA-ES
      convergence is more reliable than DE exploration.
    - High-dim: cap = min(pop_size * clamp(200 + 100*sigmoid(R-5), 200, 400),
                          budget * frac_cap)
      where frac_cap = clamp(0.45 + 0.20*sigmoid(-(R-1)), 0.45, 0.65).

      At reference (R=50, pop=166): frac_cap~0.45, min(166*300, 50000*0.45)=22500.
      At tight (R=0.24, dim=41, budget=1000, pop=174): frac_cap~0.586,
        min(174*200, 1000*0.586)=min(34800,586)=586, further capped by budget-headroom=590.
      At tight (R=1, pop=small): pop*~200 -- fewer generations but enough
      for DE to establish a covariance signal for CMA warm-start.

      Rationale for the budget-fraction cap: the step-based formula ``pop*gpm``
      can allocate a disproportionate share of the budget to DE when pop is
      large (e.g. pop=222 at dim=20, budget=100000 gives 222*300=66600 = 66.6%).
      This starves CMA-ES (which converges ill-conditioned problems) and the
      polish phase.  At generous budgets (large R) the cap is ~45%, mirroring
      the low-dim policy.  At tight budgets (small R) the cap rises to ~65%
      because pop_size is floored at 4*dim+10 (large relative to budget), so DE
      needs proportionally more budget just to complete a few generations.
      200 generations is enough for DE to provide a meaningful covariance
      warm-start for CMA-ES when budget is not the constraint.

    The cap is always min(cap, budget - reserved_polish) to guarantee
    CMA-ES + polish can run.

    Reference-point verification:
    - Low-dim  dim=10, budget=50000 (R=50): frac=0.40, cap=20000.  Matches old.
    - Low-dim  dim=10, budget=500   (R=0.5): frac=0.31, cap=155.
    - Low-dim  dim=5,  budget=2500  (R=5):  frac=0.39, cap=975.
    - High-dim dim=41, budget=100000 (R=24.4, pop=174): frac_cap~0.450,
        min(174*300, 45000)=45000 (45%).  Matches Exp8 behavior.
    - High-dim dim=41, budget=1000  (R=0.24, pop=174): frac_cap~0.586,
        min(34948, 586) then min(586, 590)=586 (58.6%).  Restores pre-Exp8 ~590.
    - High-dim dim=30, budget=2000  (R=0.67, pop=130): frac_cap~0.576,
        min(26000, 1152) then budget-headroom limits.
    - High-dim dim=50, budget=250000 (R=50, pop=256): frac_cap~0.450,
        min(76800, 112500)=76800.  Unchanged.
    """
    R = budget / max(100 * dim, 1)
    if high_dim:
        sigmoid = 1.0 / (1.0 + math.exp(-(R - 5)))
        gens_per_member = 200 + 100 * sigmoid
        gens_per_member = max(200, min(400, gens_per_member))
        step_cap = int(pop_size * gens_per_member)
        # At tight budgets (small R) the pop floor (4*dim+10) means each DE
        # generation is expensive; allow a higher fraction cap so DE can complete
        # enough generations to warm-start CMA-ES.  Smoothly relaxes back to 0.45
        # as budgets grow (large R).
        tight_sigmoid = 1.0 / (1.0 + math.exp(R - 1))
        frac_cap_val = 0.45 + 0.20 * tight_sigmoid
        frac_cap_val = max(0.45, min(0.65, frac_cap_val))
        frac_cap = int(budget * frac_cap_val)
        return min(budget - reserved_polish, step_cap, frac_cap)
    sigmoid = 1.0 / (1.0 + math.exp(-(R - 3)))
    frac = 0.30 + 0.10 * sigmoid
    frac = max(0.30, min(0.45, frac))
    return min(budget - reserved_polish, int(budget * frac))


def _compute_basin_explore_budget_frac(dim: int, budget: int) -> float:
    """Compute budget fraction for low-dim multistart basin exploration.

    Old value: 0.05 (constant).

    New formula: clamp(0.02 + 0.03*sigmoid(R-3), 0.02, 0.08)
    where R = budget / (100 * dim).

    At generous budgets (R>>10), the optimizer can afford 5-8% for basin
    exploration.  At tight budgets (R<1), basin exploration drops to 2%
    so each restart gets enough evaluations to be meaningful.  (The number
    of restarts is already scaled via _compute_basin_explore_restarts.)

    At budget=500 dim=10 (R=0.5): frac=0.021, explore_budget=10 -- combined
    with 3 restarts this gives ~3 evals per restart, but the >50 guard in
    _multistart_basin_explore will skip it entirely (which is correct:
    3 evals per restart is useless).

    At reference dim=10 budget=50000 (R=50): frac=0.05, explore_budget=2500.
    Matches old value.

    Reference-point verification:
    - dim=10, budget=50000 (R=50): sigmoid~1.0, frac=0.05.  Exact match.
    - dim=10, budget=500   (R=0.5): sigmoid~0.08, frac=0.022.
    - dim=5,  budget=2500  (R=5):  sigmoid~0.88, frac=0.046.
    - dim=5,  budget=50000 (R=100): sigmoid~1.0, frac=0.05.
    """
    R = budget / max(100 * dim, 1)
    sigmoid = 1.0 / (1.0 + math.exp(-(R - 3)))
    frac = 0.02 + 0.03 * sigmoid
    return max(0.02, min(0.08, frac))


# ---------------------------------------------------------------------------
# Stagnation-threshold formulas (Experiment A3, Part 1)
# ---------------------------------------------------------------------------
# Stagnation thresholds determine how many consecutive non-improving
# generations a phase tolerates before triggering a restart or phase
# transition.  Rather than fixed constants, each threshold is now a
# fraction of the *available generations* in its phase:
#
#   phase_generations = phase_budget / pop_size
#   stagnation = clamp(frac * phase_generations, floor, ceiling)
#
# This ensures stagnation detection adapts naturally to the problem size:
# with a tight budget the optimizer exits stale phases quickly, while a
# generous budget allows more patience before declaring stagnation.
#
# The fractions were calibrated to reproduce the original hardcoded values
# at the BBOB reference point (dim=10, budget=50000, pop=166).


def _compute_de_stagnation_thresholds(
    de_budget: int,
    pop_size: int,
) -> tuple[int, int, int, int]:
    """Compute all four DE stagnation thresholds from phase budget and pop size.

    Returns (restart_stag, baseline_steps, max_stag, target_steps).

    Old values:
    - RESTART_STAGNATION = 20
    - HIGH_DIM_DE_BASELINE_STEPS = 8
    - HIGH_DIM_DE_MAX_STAGNATION = 32
    - HIGH_DIM_DE_TARGET_STEPS = 20

    Formula basis: each threshold is a fixed fraction of the number of DE
    generations the phase can afford (de_budget / pop_size).

    - restart_stag:   17% of DE gens.  The primary stagnation counter —
      if no improvement for this many generations, restart (low-dim) or
      transition to CMA-ES (high-dim default).
    - baseline_steps:  7% of DE gens.  Warmup period for the progress-EMA
      baseline calibration before adaptive stagnation kicks in.
    - max_stag:       26.5% of DE gens.  Extended stagnation limit used
      when the EMA signal shows the DE is still making progress.
    - target_steps:   17% of DE gens.  Controls the population-reduction
      schedule in high-dim (reduction_progress = step / target_steps).

    Reference-point verification (dim=10, budget=50000, pop=166):
    - de_budget=20000, de_gens=120.5
    - restart_stag:  round(0.17 * 120.5) = 20.  Exact match.
    - baseline_steps: round(0.07 * 120.5) = 8.  Exact match.
    - max_stag:      round(0.265 * 120.5) = 32.  Exact match.
    - target_steps:  round(0.17 * 120.5) = 20.  Exact match.
    """
    de_gens = de_budget / max(pop_size, 1)
    restart_stag = max(3, round(0.17 * de_gens))
    baseline_steps = max(3, round(0.07 * de_gens))
    max_stag = max(5, round(0.265 * de_gens))
    target_steps = max(5, round(0.17 * de_gens))
    return restart_stag, baseline_steps, max_stag, target_steps


def _compute_cma_stagnation_thresholds(
    cmaes_budget: int,
    de_budget: int,
    dim: int,
    pop_size: int,
    high_dim: bool,
) -> tuple[int, int, int]:
    """Compute CMA-ES stagnation thresholds from phase budget.

    Returns (base_stag, high_dim_stag, high_dim_warm_stag).

    Old values:
    - CMA_ES_STAGNATION_GENERATIONS = 48    (low-dim)
    - HIGH_DIM_CMA_ES_STAGNATION = 72       (high-dim, phases > 0)
    - HIGH_DIM_CMA_ES_WARM_STAGNATION = 108 (high-dim, phase 0 = warm-started)

    Formula basis: the base threshold is 23% of available CMA-ES generations.
    High-dim thresholds apply the original multipliers (1.5x, 2.25x) to
    account for slower convergence in high-dimensional spaces and extra
    patience warranted by warm-started covariance.

    Ceilings (200/300/450) prevent pathologically long stagnation waits
    when the budget is extremely generous.

    The CMA-ES base population is estimated as max(4+floor(3*ln(dim)), pop//2)
    (matching _enter_cmaes_phase) since the actual CMA-ES instance does not
    exist yet at __init__ time.

    Reference-point verification (dim=10, budget=50000, pop=166):
    - cma_remaining=17500, cma_base_pop=83, cma_gens=210.8
    - base_stag:     round(0.23 * 210.8) = 48.  Exact match.
    - high_dim_stag: round(1.5 * 48)     = 72.  Exact match.
    - warm_stag:     round(1.5 * 72)     = 108. Exact match.
    """
    cma_base_pop = max(4 + math.floor(3 * math.log(max(dim, 2))), pop_size // 2)
    if high_dim:
        cma_base_pop = max(cma_base_pop, dim + 8)
    cma_remaining = max(1, cmaes_budget - de_budget)
    cma_phase_gens = cma_remaining / max(cma_base_pop, 1)

    base_stag = max(10, min(200, round(0.23 * cma_phase_gens)))
    high_dim_stag = max(10, min(300, round(1.5 * base_stag)))
    high_dim_warm_stag = max(10, min(450, round(1.5 * high_dim_stag)))
    return base_stag, high_dim_stag, high_dim_warm_stag


def _compute_basin_explore_stagnation(
    dim: int,
    budget: int,
    basin_explore_budget_frac: float,
    basin_explore_restarts: int,
) -> int:
    """Compute stagnation limit for each basin exploration CMA-ES restart.

    Old value: 10 (constant).

    Formula basis: each mini-restart can run at most
    ``restart_budget / basin_pop`` generations.  The stagnation threshold is
    48% of that capacity — generous enough to let each restart converge on
    its basin while bailing out if truly stuck.

    A ceiling of 50 prevents excessive patience at very generous budgets.

    Reference-point verification (dim=10, budget=50000):
    - basin_budget=2500, n_restarts=12, restart_budget=208, basin_pop=10
    - max_gens=20.8, stag=round(0.48*20.8) = 10.  Exact match.
    """
    basin_budget = int(budget * basin_explore_budget_frac)
    n_restarts = max(basin_explore_restarts, 1)
    restart_budget = basin_budget // n_restarts
    basin_pop = min(
        max(8, 4 + int(3 * math.log(max(dim, 2)))),
        max(1, restart_budget),
    )
    basin_pop = basin_pop + (basin_pop % 2)
    basin_pop = max(basin_pop, 4)
    basin_gens = restart_budget / max(basin_pop, 1)
    return max(3, min(50, round(0.48 * basin_gens)))


# ---------------------------------------------------------------------------
# CMA-ES parameter formulas (Experiment A3, Part 2)
# ---------------------------------------------------------------------------
# These replace hardcoded CMA-ES sigma bounds, phase counts, and Levy step
# size with functions of (dim, budget, pop_size).  The key insight for sigma
# bounds: higher dimensions need wider initial sigma (as fraction of span) to
# maintain effective exploration -- the "useful volume" of a search hypersphere
# shrinks exponentially with dim, so the per-axis step must grow to compensate.
# The log(dim) scaling provides a gentle expansion that avoids over-exploration.
#
# For the number of CMA-ES restart phases, the constraint is that each phase
# must have enough budget for at least a few productive generations.  The
# formula derives the phase count from the available CMA budget and the
# base population size, ensuring every phase gets meaningful work.


def _compute_cma_sigma_bounds(dim: int) -> tuple[float, float, float, float]:
    """Compute CMA-ES sigma bounds (as fractions of span) from dimension.

    Returns (sigma_min, sigma_max, restart_sigma_min, restart_sigma_max).

    Old values (constant):
    - CMA_ES_SIGMA_MIN = 0.01
    - CMA_ES_SIGMA_MAX = 0.10
    - CMA_ES_RESTART_SIGMA_MIN = 0.08
    - CMA_ES_RESTART_SIGMA_MAX = 0.30

    Formula basis: sigma bounds scale as ``base * (1 + alpha * ln(dim/10))``
    where alpha controls how quickly the bounds widen with dimensionality.

    The log(dim/10) term is zero at dim=10 (the BBOB reference), so the
    formula exactly reproduces the old constants at the reference point.
    For dim>10, the bounds widen modestly; for dim<10 they tighten slightly.

    The rationale: in high dimensions, each axis of the CMA-ES distribution
    covers less of the search space, so the overall sigma (which scales all
    axes uniformly) must be larger to maintain comparable per-axis coverage.
    The logarithmic dependence prevents excessive widening.

    Reference-point verification:
    - dim=10: log(10/10)=0. sigma_min=0.01, sigma_max=0.10,
      restart_min=0.08, restart_max=0.30.  Exact match.
    - dim=2:  log(2/10)=-1.61. sigma_min=max(0.005, 0.01*(1-0.32))=0.0068,
      sigma_max=max(0.05, 0.10*(1-0.24))=0.076.
    - dim=20: log(20/10)=0.69. sigma_min=0.01*(1+0.14)=0.0114,
      sigma_max=0.10*(1+0.10)=0.110.
    - dim=50: log(50/10)=1.61. sigma_min=0.01*(1+0.32)=0.0132,
      sigma_max=0.10*(1+0.24)=0.124.
    - dim=100: log(100/10)=2.30. sigma_min=0.01*(1+0.46)=0.0146,
      sigma_max=0.10*(1+0.35)=0.135.
    """
    log_ratio = math.log(max(dim, 2) / 10.0)

    # Initial phase sigma bounds (conservative: narrow band for warm-started CMA)
    sigma_min = 0.01 * (1.0 + 0.20 * log_ratio)
    sigma_min = max(0.005, min(0.03, sigma_min))
    sigma_max = 0.10 * (1.0 + 0.15 * log_ratio)
    sigma_max = max(0.05, min(0.20, sigma_max))

    # Restart phase sigma bounds (wider: restart phases need more exploration)
    restart_sigma_min = 0.08 * (1.0 + 0.15 * log_ratio)
    restart_sigma_min = max(0.04, min(0.15, restart_sigma_min))
    restart_sigma_max = 0.30 * (1.0 + 0.12 * log_ratio)
    restart_sigma_max = max(0.15, min(0.50, restart_sigma_max))

    return sigma_min, sigma_max, restart_sigma_min, restart_sigma_max


def _compute_cma_phases(
    cma_budget: int,
    dim: int,
    pop_size: int,
) -> int:
    """Compute number of CMA-ES IPOP restart phases from available budget.

    Old value: 6 (constant, high-dim only).

    Formula basis: each CMA-ES phase needs at least ``min_gens_per_phase``
    generations (where generation cost = base_pop evaluations) to make
    meaningful progress.  The number of phases is the budget divided by
    (base_pop * min_gens_per_phase), clamped to [1, 8].

    ``min_gens_per_phase`` is set to 15, which gives each restart enough
    iterations to update the covariance and converge on a local basin,
    while the IPOP population doubling ensures later phases cover larger
    basin radii.

    Reference-point verification (dim=10, budget=50000, pop=166):
    - cma_budget = 50000 - polish_reserve(12500) = 37500
    - base_pop = max(4+floor(3*ln(10)), 166//2) = max(10, 83) = 83
    - available_gens = 37500 / 83 = 451.8
    - phases = floor(451.8 / 15) = 30, clamped to 8.
      At the reference point the budget is very generous so we max out at 8.
      Old value was 6 -- slightly more phases here, but each still gets
      451/8 = 56 generations, which is more than enough.

    - dim=20, budget=2000, pop=18:
      cma_budget ~ 2000 - 82 = 1918, base_pop = max(13, 9) = 13
      gens = 1918 / 13 = 147.5, phases = floor(147.5/15) = 9 -> clamped to 8.

    - dim=10, budget=500, pop=18:
      cma_budget ~ 500 - 75 = 425, base_pop = max(10, 9) = 10
      gens = 425 / 10 = 42.5, phases = floor(42.5/15) = 2.
      With old CMA_ES_PHASES_HIGH_DIM=6: each phase = 70 evals = 7 gens.
      Now 2 phases: 212 evals each = 21 gens.  Much more productive.

    - dim=50, budget=1000000, pop=256:
      cma_budget ~ 1000000 - 80000 = 920000, base_pop = max(15, 128) = 128
      gens = 920000 / 128 = 7187.5, phases = floor(7187.5/15) = 479 -> clamped 8.
    """
    # Estimate base CMA-ES population (mirrors _enter_cmaes_phase logic)
    base_pop = max(
        4 + math.floor(3 * math.log(max(dim, 2))),
        pop_size // 2,
    )
    base_pop = max(base_pop, dim + 8)  # high-dim floor

    min_gens_per_phase = 15
    available_gens = cma_budget / max(base_pop, 1)
    raw_phases = int(available_gens / min_gens_per_phase)
    return max(1, min(8, raw_phases))


def _compute_step_size_init(dim: int) -> float:
    """Compute initial Levy flight step size (as fraction of span) from dimension.

    Old value: 0.1 (constant).

    Formula basis: ``0.10 * (1 + 0.15 * ln(dim/10))``, clamped to [0.05, 0.25].

    Higher dimensions benefit from a larger initial perturbation step because
    Levy flights perturb each dimension independently and the probability of
    improving along at least one axis grows with dim.  A slightly wider initial
    step also helps the adaptive mechanism (multiplicative 1.05/0.95) converge
    faster to the correct scale.

    The log(dim/10) term is zero at the reference point (dim=10), giving
    an exact match of 0.10.

    Reference-point verification:
    - dim=10: log(10/10)=0.  init=0.10.  Exact match.
    - dim=2:  log(2/10)=-1.61. init=0.10*(1-0.24)=0.076.
    - dim=5:  log(5/10)=-0.69. init=0.10*(1-0.10)=0.090.
    - dim=20: log(20/10)=0.69. init=0.10*(1+0.10)=0.110.
    - dim=50: log(50/10)=1.61. init=0.10*(1+0.24)=0.124.
    - dim=100: log(100/10)=2.30. init=0.10*(1+0.35)=0.135.
    """
    log_ratio = math.log(max(dim, 2) / 10.0)
    raw = 0.10 * (1.0 + 0.15 * log_ratio)
    return max(0.05, min(0.25, raw))


# ---------------------------------------------------------------------------
# Search-grid and direction-count formulas (Experiment A5)
# ---------------------------------------------------------------------------
# These replace the remaining hardcoded constants in the polish phase.
# Constants fall into two categories:
#
# 1. **Needs formula** -- the value is an absolute count or a threshold that
#    should scale with (dim, budget) to behave correctly at non-reference
#    settings.  Replaced by a function below.
#
# 2. **Universal** -- the value is a grid resolution, shrink factor, or
#    method-inherent fraction that does not depend on problem scale.
#    Kept as a module-level constant (documented above).


def _compute_high_dim_threshold(dim: int, budget: int) -> int:
    """Compute the dimension threshold that separates "low-dim" from "high-dim".

    Old value: 20 (constant).

    New formula: clamp(20 + 5 * (1 - sigmoid(R - 5)), 15, 30)
    where R = budget / (100 * dim).

    Rationale: the high-dim code path includes directional basin search and
    smoothed envelope search, both of which are evaluation-hungry.  At
    generous budgets (R >> 5) they pay for themselves, so the threshold stays
    at 20.  At tight budgets (R < 2) even dim=20 problems cannot afford
    these methods, so the threshold rises to ~24-25 -- effectively skipping
    directional search for moderate-dim problems that lack the budget to
    benefit.

    The ceiling of 30 ensures that truly high-dim problems (dim >= 30)
    always get directional search regardless of budget.

    Reference-point verification:
    - dim=10, budget=50000 (R=50):  sigmoid~1.0, threshold=20. Exact match.
    - dim=20, budget=100000 (R=50): sigmoid~1.0, threshold=20.
    - dim=20, budget=2000 (R=1):    sigmoid~0.02, threshold=24.9 -> 25.
    - dim=10, budget=500 (R=0.5):   sigmoid~0.01, threshold=24.9 -> 25.
    - dim=30, budget=3000 (R=1):    threshold=25, dim=30 >= 25 -> high-dim.
    """
    R = budget / max(100 * dim, 1)
    sigmoid = 1.0 / (1.0 + math.exp(-(R - 5)))
    threshold = 20 + 5 * (1.0 - sigmoid)
    return max(15, min(30, round(threshold)))


def _compute_coordinate_coarse_points(dim: int) -> int:
    """Compute coarse grid points for coordinate basin search.

    Old values: COORDINATE_COARSE_HIGH_DIM = 11, COORDINATE_COARSE_LOW_DIM = 17.
    Used via branching: if high_dim then 11 else 17.

    New formula: clamp(21 - dim // 2, 11, 21).
    At dim=2: 21-1=20 points (low-dim can afford dense per-axis probing).
    At dim=8: 21-4=17 (matches old COORDINATE_COARSE_LOW_DIM at typical low-dim).
    At dim=16: 21-8=13.
    At dim=20+: 21-10=11 (matches old COORDINATE_COARSE_HIGH_DIM).

    This unifies the two constants into a single smooth formula.  The
    per-axis cost of coordinate search is O(coarse_points * dim), so
    reducing coarse_points at higher dim keeps the total polish cost
    manageable.

    Reference-point verification:
    - dim=2:  21 - 1 = 20. More generous for very low dim (was 17).
    - dim=5:  21 - 2 = 19.
    - dim=8:  21 - 4 = 17. Exact match of old COORDINATE_COARSE_LOW_DIM.
    - dim=10: 21 - 5 = 16.
    - dim=20: 21 - 10 = 11. Exact match of old COORDINATE_COARSE_HIGH_DIM.
    - dim=40: 21 - 20 = 1 -> clamped to 11. Still 11 for very high dim.
    """
    return max(11, min(21, 21 - dim // 2))


def _compute_directional_counts(dim: int) -> tuple[int, int, int, int]:
    """Compute direction counts for directional basin search.

    Returns (pca_dirs, random_dirs, elite_dirs, basis_pair_limit).

    Old values (constant):
    - DIRECTIONAL_PCA_DIRECTIONS = 6
    - DIRECTIONAL_RANDOM_DIRECTIONS = 8
    - DIRECTIONAL_ELITE_DIRECTIONS = 6
    - DIRECTIONAL_BASIS_PAIR_LIMIT = 6

    Formula basis: at the reference point (dim=20, the low end of high-dim),
    all values match their old constants.  At higher dimensions, each count
    grows logarithmically because (a) PCA captures more meaningful variance
    directions in higher-dim spaces, (b) random directions need to span more
    of the space, and (c) elite-to-point directions become more informative
    with more axes of variation.

    The log(dim/20) scaling is zero at dim=20 (high-dim threshold) so all
    formulas reproduce exact old values there.  Growth is capped to prevent
    explosion at very high dim.

    Reference-point verification:
    - dim=20: log(20/20)=0. pca=6, random=8, elite=6, pair_limit=6. Exact match.
    - dim=40: log(40/20)=0.69. pca=round(6+2*0.69)=7, random=round(8+3*0.69)=10,
      elite=round(6+2*0.69)=7, pair_limit=round(6+2*0.69)=7.
    - dim=100: log(100/20)=1.61. pca=round(6+3.22)=9, random=round(8+4.83)=13,
      elite=round(6+3.22)=9, pair_limit=round(6+3.22)=9.
    """
    log_ratio = math.log(max(dim, 20) / 20.0)
    pca_dirs = max(6, min(12, round(6 + 2 * log_ratio)))
    random_dirs = max(8, min(16, round(8 + 3 * log_ratio)))
    elite_dirs = max(6, min(12, round(6 + 2 * log_ratio)))
    basis_pair_limit = max(6, min(12, round(6 + 2 * log_ratio)))
    return pca_dirs, random_dirs, elite_dirs, basis_pair_limit


def _compute_envelope_budgets(budget: int) -> tuple[int, int]:
    """Compute smoothed-envelope budget thresholds from total budget.

    Returns (min_remaining, proposal_budget_cap).

    Old values (constant):
    - SMOOTHED_ENVELOPE_MIN_REMAINING = 1600
    - SMOOTHED_ENVELOPE_PROPOSAL_BUDGET = 1800

    These were absolute eval counts calibrated for the reference setting
    (budget=50000).  At that setting, min_remaining=1600 means "only run
    envelope search if at least 1600 evals remain" (3.2% of budget), and
    proposal_budget=1800 means "spend at most 1800 evals polishing each
    blend proposal" (3.6% of budget).

    Formula basis: both scale as fixed fractions of budget, clamped to
    reasonable floors and ceilings.

    - min_remaining = clamp(budget * 0.032, 200, 5000)
      At reference: 50000 * 0.032 = 1600.  Exact match.
      The floor of 200 ensures envelope search never activates when
      there are so few evals left that the probe pairs alone would
      exhaust the budget.

    - proposal_budget_cap = clamp(budget * 0.036, 200, 6000)
      At reference: 50000 * 0.036 = 1800.  Exact match.
      The floor ensures each blend proposal gets enough evals for
      at least a few BFGS steps (2*dim + backtracking).

    Reference-point verification:
    - budget=50000:  min_remaining=1600, proposal_cap=1800.  Exact match.
    - budget=100000: min_remaining=3200, proposal_cap=3600.
    - budget=10000:  min_remaining=320,  proposal_cap=360.
    - budget=2000:   min_remaining=200(floor), proposal_cap=200(floor).
    - budget=500000: min_remaining=5000(cap), proposal_cap=6000(cap).
    """
    min_remaining = max(200, min(5000, round(budget * 0.032)))
    proposal_cap = max(200, min(6000, round(budget * 0.036)))
    return min_remaining, proposal_cap


def _compute_directional_coarse_points(directional_budget: int) -> int:
    """Compute coarse grid points per line-search direction from directional budget.

    Old value: DIRECTIONAL_COARSE_POINTS = 11 (constant).

    At the high-dim reference (dim=20, budget=50000) the directional budget
    is ~4000 evals.  With ~32 directions and 21 evals per direction
    (11 coarse + 2*5 refinement), the total cost is ~720 evals -- about
    18% of the polish budget, well within tolerance.

    At low budgets (e.g. budget=500, dim=20) the directional budget shrinks
    to ~150-200 evals.  Keeping 11 coarse points would consume the entire
    budget on just a few directions.  Scaling coarse points down to 5
    (the minimum for meaningful line-search resolution) preserves the
    coarse-to-fine structure while fitting within tight budgets.

    Formula: clamp(round(3 + 10*db/(db+800)), 5, 11).
    A saturating curve that approaches 13 (capped to 11) for large budgets
    and drops to 5 for small budgets.  The half-point (db=800) is chosen
    so that db=4000 yields 11.

    Reference-point verification:
    - db=4000:  3 + 10*4000/4800 = 11.33 -> 11.  Exact match.
    - db=3000:  3 + 10*3000/3800 = 10.89 -> 11.
    - db=2000:  3 + 10*2000/2800 = 10.14 -> 10.
    - db=1000:  3 + 10*1000/1800 =  8.56 ->  9.
    - db=500:   3 + 10*500/1300  =  6.85 ->  7.
    - db=200:   3 + 10*200/1000  =  5.00 ->  5.
    - db=100:   3 + 10*100/900   =  4.11 ->  5 (floor).
    """
    if directional_budget <= 0:
        return 5
    raw = 3.0 + 10.0 * directional_budget / (directional_budget + 800)
    return max(5, min(11, round(raw)))


def _compute_directional_priority_hops(directional_budget: int) -> int:
    """Compute priority hop probes per priority direction from directional budget.

    Old value: DIRECTIONAL_PRIORITY_HOPS = 4 (constant).

    Priority hops insert fixed-step probes along CMA eigenvector and
    basis-pair directions to explore near-integer steps (important for
    Rastrigin-like multimodal landscapes).  At the high-dim reference
    (db=4000) with ~12 priority directions, 4 hops costs only 48 evals.

    At tight budgets the hop cost becomes significant relative to the
    total.  Scaling from 4 down to 1 preserves at least one hop per
    priority direction (enough to detect nearby basins) while saving
    budget for additional search directions.

    Formula: clamp(round(5*db/(db+1200)), 1, 4).
    A saturating curve that reaches 4 at db=3000+ and drops to 1
    for db < 750.  The half-point (db=1200) gives 2.5 at moderate
    budgets.

    Reference-point verification:
    - db=4000:  5*4000/5200 = 3.85 -> 4.  Exact match.
    - db=3000:  5*3000/4200 = 3.57 -> 4.
    - db=2000:  5*2000/3200 = 3.12 -> 3.
    - db=1000:  5*1000/2200 = 2.27 -> 2.
    - db=500:   5*500/1700  = 1.47 -> 1.
    - db=200:   5*200/1400  = 0.71 -> 1 (floor).
    """
    if directional_budget <= 0:
        return 1
    raw = 5.0 * directional_budget / (directional_budget + 1200)
    return max(1, min(4, round(raw)))


def _compute_directional_basis_pair_basis_count(directional_budget: int) -> int:
    """Compute how many top CMA eigenvectors to combine pairwise.

    Old value: DIRECTIONAL_BASIS_PAIR_BASIS_COUNT = 4 (constant).

    With k basis vectors, C(k,2) unique pairs yield up to 2*C(k,2)
    sum/difference directions (capped by basis_pair_limit):
      k=2 -> 1 pair  -> up to 2 directions
      k=3 -> 3 pairs -> up to 6 directions
      k=4 -> 6 pairs -> up to 12 directions

    At the high-dim reference (db=4000), 4 vectors is ideal: enough
    cross-axis exploration without overwhelming the direction budget.
    At tight budgets, 2 vectors still provides one pair of sum/diff
    directions while saving the evaluation budget for other search
    components (coordinate search, scipy polishers).

    Formula: clamp(round(5*db/(db+1500)), 2, 4).
    A saturating curve that reaches 4 at db=4000 and floors at 2
    for small budgets.  The half-point (db=1500) is slightly higher
    than for priority hops because each additional basis vector
    generates O(k) more directions.

    Reference-point verification:
    - db=4000:  5*4000/5500 = 3.64 -> 4.  Exact match.
    - db=3000:  5*3000/4500 = 3.33 -> 3.
    - db=2000:  5*2000/3500 = 2.86 -> 3.
    - db=1000:  5*1000/2500 = 2.00 -> 2.
    - db=500:   5*500/2000  = 1.25 -> 2 (floor).
    - db=200:   5*200/1700  = 0.59 -> 2 (floor).
    """
    if directional_budget <= 0:
        return 2
    raw = 5.0 * directional_budget / (directional_budget + 1500)
    return max(2, min(4, round(raw)))


# ---------------------------------------------------------------------------
# Polish sub-allocation formula (Experiment 7)
# ---------------------------------------------------------------------------


def _compute_polish_fractions(remaining_after_coord: int) -> tuple[float, float, float]:
    """Compute polish budget fractions for L-BFGS-B, Powell, and FD-BFGS.

    Returns (lbfgsb_frac, powell_frac, fdbfgs_frac).

    Old values (constant):
    - POLISH_LBFGSB_FRAC = 0.40  (L-BFGS-B gets 40% of remaining)
    - POLISH_POWELL_FRAC = 0.50  (Powell gets 50% of post-L-BFGS-B remaining)
    - POLISH_FDBFGS_FRAC = 0.50  (FD-BFGS gets 50% of post-Powell remaining)

    Effective allocation at the reference point:
      L-BFGS-B ~40%, Powell ~30%, FD-BFGS ~15%, Nelder-Mead ~15%.

    Problem: at low polish budgets (remaining < 200), the sequential split
    leaves later polishers (Powell, FD-BFGS, Nelder-Mead) with almost no
    evaluations.  L-BFGS-B is the strongest gradient-based polisher and
    benefits most from concentrated budget.

    Formula: a sigmoid transition centered at remaining=200 smoothly shifts
    from concentrated allocation (80% L-BFGS-B) at tight budgets to the
    original balanced split at generous budgets.

    - lbfgsb_frac = 0.80 - 0.40 * sigmoid
    - powell_frac = 0.20 + 0.30 * sigmoid
    - fdbfgs_frac = 0.20 + 0.30 * sigmoid

    where sigmoid = 1 / (1 + exp(-(remaining - 200) / 50)).

    At remaining >> 300, sigmoid -> 1.0:
      lbfgsb=0.40, powell=0.50, fdbfgs=0.50.  Matches old constants.
    At remaining ~ 50, sigmoid ~ 0.05:
      lbfgsb=0.78, powell=0.22, fdbfgs=0.22.  Concentrates on L-BFGS-B.

    The transition center (200) and scale (50) were chosen so that:
    - At remaining=500+ the fractions are within 0.01 of old values
    - At remaining=100 the L-BFGS-B fraction rises to ~0.75

    Reference-point verification (dim=10, budget=50000):
    - remaining_after_coord is typically 5000-10000 at this setting
    - sigmoid = 1/(1+exp(-96)) = 1.0
    - lbfgsb=0.40, powell=0.50, fdbfgs=0.50.  Exact match.

    Low-budget verification (dim=10, budget=500):
    - remaining_after_coord ~ 50-100
    - At remaining=100: sigmoid=0.119, lbfgsb=0.752, powell=0.236, fdbfgs=0.236
    - At remaining=50:  sigmoid=0.047, lbfgsb=0.781, powell=0.214, fdbfgs=0.214
    """
    s = 1.0 / (1.0 + math.exp(-(remaining_after_coord - 200) / 50))
    lbfgsb_frac = max(0.40, min(0.80, 0.80 - 0.40 * s))
    powell_frac = max(0.20, min(0.50, 0.20 + 0.30 * s))
    fdbfgs_frac = max(0.20, min(0.50, 0.20 + 0.30 * s))
    return lbfgsb_frac, powell_frac, fdbfgs_frac


class PhasedDFO(PhasedStateCompatMixin, BaseOptimizer):
    """Multi-phase derivative-free optimizer: SHADE-DE -> IPOP-CMA-ES -> Polish.

    Allocates a total evaluation budget (default: dim * 5000) across three phases:
    1. SHADE-DE warmup: exits via stagnation detection
    2. IPOP-CMA-ES: multiple restart phases with doubling population
    3. Polish: directional + coordinate + FD-BFGS refinement

    Parameters
    ----------
    dim : int | None
        Dimensionality of the search space. Required if *space* is not given.
    bounds : float | tuple[float, float]
        Search bounds (symmetric scalar or (lo, hi) tuple). Ignored when
        *space* is provided (bounds default to ``(0.0, 1.0)`` in that case).
    budget : int | None
        Total evaluation budget. None uses dim * 5000.
    pop_size : int
        Initial population size for the DE phase.
    device : str | torch.device | None
        Torch device.
    dtype : torch.dtype
        Floating-point dtype.
    seed : int | None
        Reproducibility seed for the internal RNG.
    space : SearchSpace | None
        Optional typed search space.  When given, *dim* is inferred from
        ``space.dim`` and bounds default to ``(0.0, 1.0)``.
    initial_points : torch.Tensor | list[dict[str, object]] | None
        Optional warm-start for the DE population.  Either a
        ``(n, dim)`` float tensor (already encoded to ``[0, 1]``) or a
        ``list[dict]`` that will be encoded via *space*.  If *n* < pop_size
        only the first *n* rows are seeded; if *n* > pop_size the excess
        rows are discarded.

    Examples
    --------
    >>> import torch
    >>> import torch_dfo
    >>> opt = torch_dfo.PhasedDFO(dim=10, bounds=5.0, budget=20_000, seed=0)
    >>> while not opt.done:
    ...     x = opt.ask()
    ...     opt.tell(x, (x ** 2).sum(-1))
    >>> best_x, best_f = opt.best()

    """

    def __init__(
        self,
        dim: int | None = None,
        bounds: float | tuple[float, float] = 5.0,
        budget: int | None = None,
        budget_mult: int = 5000,
        pop_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
        *,
        space: SearchSpace | None = None,
        initial_points: torch.Tensor | list[dict[str, object]] | None = None,
        config: PhasedConfig | None = None,
    ) -> None:
        self._config: PhasedConfig = config if config is not None else PhasedConfig()
        # Resolve dim and bounds from space if provided
        if space is not None:
            if dim is not None and dim != space.dim:
                raise ValueError(
                    f"dim={dim} conflicts with space.dim={space.dim}; "
                    "omit dim when space is provided",
                )
            actual_dim: int = space.dim
            actual_bounds: float | tuple[float, float] = (0.0, 1.0)
        else:
            if dim is None:
                raise ValueError("Either dim or space must be provided")
            actual_dim = dim
            actual_bounds = bounds
        self._space: SearchSpace | None = space

        # Resolve initial_points to a Tensor
        _resolved_initial: torch.Tensor | None = None
        if initial_points is not None:
            if isinstance(initial_points, list):
                if self._space is None:
                    raise ValueError("initial_points as list[dict] requires space to be set")
                if len(initial_points) == 0:
                    raise ValueError("initial_points must not be empty")
                _resolved_device = torch.device(device) if isinstance(device, str) else device
                _resolved_initial = self._space.encode(
                    initial_points,
                    device=_resolved_device,
                    dtype=dtype,
                )
            else:
                _resolved_initial = initial_points
            if _resolved_initial.ndim != 2 or _resolved_initial.shape[1] != actual_dim:
                raise ValueError(
                    f"initial_points must have shape (n, {actual_dim}), "
                    f"got {tuple(_resolved_initial.shape)}",
                )

        # Alias so the rest of __init__ continues to use dim and bounds unchanged
        dim = actual_dim
        bounds = actual_bounds

        actual_budget = budget if budget is not None else dim * budget_mult
        # Adaptive population sizing from (dim, budget).
        # Old formula: max(80, min(256, budget // 300)).
        if pop_size is None:
            pop_size = _compute_pop_size(dim, actual_budget)

        super().__init__(dim, bounds, pop_size, device, dtype, seed)

        # Store raw bounds so reset() can re-create SHADE without re-running __init__
        self._raw_bounds: float | tuple[float, float] = bounds
        self._polish = PhasedPolishState(
            elite_solutions=[],
            elite_fitness=[],
            search_population=None,
            search_population_fitness=None,
            search_pool_limit=0,
        )
        self._valley = PhasedValleyState(
            basin_explore_restarts=0,
            basin_explore_budget_frac=0.0,
            basin_explore_stagnation=0,
            portfolio_stag=[],
            portfolio_best_f=[],
            portfolio_branch_stag_limit=40,
            portfolio_sigma0=[],
            portfolio_generation=0,
            portfolio_active_indices=(),
            valley_focus_remaining=0,
            valley_focus_streak=0,
        )
        self._de = PhasedDEState(
            phase=0,
            fe_count=0,
            high_dim=False,
            high_dim_de_min_pop=0,
            restart_stagnation=0,
            de_baseline_steps=0,
            de_max_stagnation=0,
            de_target_steps=0,
            stagnation_counter=0,
            de_progress_ema=0.0,
            de_progress_baseline=0.0,
            de_step_count=0,
            de_best_f_prev=float("inf"),
            de_phase_start_f=float("inf"),
            trial_gain=0.0,
            levy_gain=0.0,
            accepted_ratio=0.0,
            levy_ratio=0.0,
            step_size=0.0,
            midpoint_probed=False,
            de_restart_count=0,
        )
        self._cma = PhasedCMAState(
            cmaes=None,
            cmaes_phase_idx=0,
            cmaes_portfolio=None,
            cmaes_fe_start=0,
            cmaes_phase_count=1,
            cmaes_phase_budgets=[],
            cmaes_stagnation_counter=0,
            cmaes_phase_best_f=float("inf"),
            cmaes_overall_start_f=float("inf"),
            cmaes_entered=False,
            cmaes_base_pop=0,
            cma_sigma_min=0.0,
            cma_sigma_max=0.0,
            cma_restart_sigma_min=0.0,
            cma_restart_sigma_max=0.0,
            cma_stagnation=0,
            high_dim_cma_stagnation=0,
            high_dim_cma_warm_stagnation=0,
        )

        self._budget = actual_budget
        self._fe_count = 0
        self._phase = 0  # 0=DE, 1=CMA-ES, 2=Polish, 3=Done

        # Dimension threshold: adaptive from (dim, budget) via
        # _compute_high_dim_threshold.  At generous budgets this is 20
        # (matching the old constant); at tight budgets it rises to ~25
        # so moderate-dim problems skip expensive directional search.
        self._high_dim_threshold = _compute_high_dim_threshold(dim, actual_budget)
        self._high_dim = dim >= self._high_dim_threshold

        # Adaptive constants computed from (dim, budget, pop_size)
        self._high_dim_de_min_pop = _compute_high_dim_de_min_pop(dim)
        self._basin_explore_restarts = _compute_basin_explore_restarts(dim, actual_budget)
        self._search_pool_max = _compute_search_pool_max(dim, pop_size)

        # Polish direction counts: adaptive from dim.
        # Old values: PCA=6, random=8, elite=6, basis_pair_limit=6.
        # Grows logarithmically for dim > 20 to capture more variance.
        (
            self._pca_directions,
            self._random_directions,
            self._elite_directions,
            self._basis_pair_limit,
        ) = _compute_directional_counts(dim)

        # Coordinate coarse points: adaptive from dim.
        # Unifies old COORDINATE_COARSE_HIGH_DIM=11 and LOW_DIM=17 into
        # a smooth formula: 21 - dim//2, clamped to [11, 21].
        self._coordinate_coarse_points = _compute_coordinate_coarse_points(dim)

        # Smoothed envelope budget thresholds: adaptive from budget.
        # Old values: MIN_REMAINING=1600, PROPOSAL_BUDGET=1800 (absolute counts).
        # Now scaled as fixed fractions of total budget.
        (
            self._envelope_min_remaining,
            self._envelope_proposal_cap,
        ) = _compute_envelope_budgets(actual_budget)

        # Budget allocation — all fractions computed from (dim, budget, pop_size)
        # via formulas that adapt to the generosity ratio R = budget / (100 * dim).
        self._polish_reserve = _compute_polish_reserve(
            dim,
            actual_budget,
            self._high_dim,
        )
        if self._high_dim:
            reserved_polish = _compute_de_headroom(dim, actual_budget, self._high_dim)
        else:
            reserved_polish = self._polish_reserve
        self._de_budget = _compute_de_budget_cap(
            dim,
            actual_budget,
            pop_size,
            self._high_dim,
            reserved_polish,
        )
        self._cmaes_budget = self._budget - self._polish_reserve
        self._basin_explore_budget_frac = _compute_basin_explore_budget_frac(
            dim,
            actual_budget,
        )

        # Stagnation thresholds — all computed from phase budgets and pop sizes.
        # See _compute_de_stagnation_thresholds, _compute_cma_stagnation_thresholds,
        # and _compute_basin_explore_stagnation for derivation and verification.
        (
            self._restart_stagnation,
            self._de_baseline_steps,
            self._de_max_stagnation,
            self._de_target_steps,
        ) = _compute_de_stagnation_thresholds(self._de_budget, pop_size)
        (
            self._cma_stagnation,
            self._high_dim_cma_stagnation,
            self._high_dim_cma_warm_stagnation,
        ) = _compute_cma_stagnation_thresholds(
            self._cmaes_budget,
            self._de_budget,
            dim,
            pop_size,
            self._high_dim,
        )
        self._basin_explore_stagnation = _compute_basin_explore_stagnation(
            dim,
            actual_budget,
            self._basin_explore_budget_frac,
            self._basin_explore_restarts,
        )

        # Initialize SHADE sub-optimizer
        self._shade = SHADE(
            dim,
            bounds,
            pop_size,
            device=device,
            dtype=dtype,
            seed=seed,
            initial_population=_resolved_initial,
        )

        # Store initial population size for population reduction.
        self._initial_pop_size = pop_size

        # CMA-ES (created when entering Phase 1)
        self._cmaes: CMAES | None = None
        self._cmaes_phase_idx = 0
        # K=4 parallel portfolio (high-dim only; None for low-dim)
        self._cmaes_portfolio: list[CMAES] | None = None
        self._portfolio_stag: list[int] = []
        self._portfolio_best_f: list[float] = []
        self._portfolio_branch_stag_limit: int = 40  # overwritten in _enter_cmaes_phase_portfolio
        self._portfolio_sigma0: list[float] = []  # initial sigma per branch for stop criterion
        self._portfolio_generation: int = 0
        self._portfolio_active_indices: tuple[int, ...] = ()
        self._valley_focus_remaining: int = 0
        self._valley_focus_streak: int = 0
        # fe_count at the moment CMA phase is entered; set by _enter_cmaes_phase.
        # Used to compute fe_in_cmaes relative to the actual CMA start, not the
        # theoretical (budget - cmaes_budget) offset which may differ when DE
        # exits early via stagnation.
        self._cmaes_fe_start = 0
        # CMA-ES phase count: adaptive from (cma_budget, dim, pop_size).
        # Old value: CMA_ES_PHASES_HIGH_DIM=6 for high-dim, 1 for low-dim.
        if self._high_dim:
            self._cmaes_phase_count = _compute_cma_phases(
                self._cmaes_budget,
                dim,
                pop_size,
            )
        else:
            self._cmaes_phase_count = 1
        self._cmaes_stagnation_counter = 0
        self._cmaes_phase_best_f = float("inf")

        # Stagnation detection for DE
        self._stagnation_counter = 0
        self._de_progress_ema = 0.0
        self._de_progress_baseline = 0.0
        self._de_step_count = 0
        self._de_best_f_prev = float("inf")

        # Trajectory signal tracking — start values for improvement deltas
        self._de_phase_start_f: float = float("inf")
        self._cmaes_overall_start_f: float = float("inf")
        self._cmaes_entered: bool = False

        # Richer stagnation signal tracking.
        self._trial_gain = 0.0
        self._levy_gain = 0.0
        self._accepted_ratio = 0.0
        self._levy_ratio = 0.0

        # Adaptive Levy step_size — initial scale from _compute_step_size_init().
        span = (self.ub - self.lb).mean().item()
        self._step_size = _compute_step_size_init(dim) * span
        self._search_span = span

        # CMA-ES sigma bounds — computed from dim via _compute_cma_sigma_bounds()
        (
            self._cma_sigma_min,
            self._cma_sigma_max,
            self._cma_restart_sigma_min,
            self._cma_restart_sigma_max,
        ) = _compute_cma_sigma_bounds(dim)

        # Collection of elite solutions across all phases
        self._elite_solutions: list[torch.Tensor] = []
        self._elite_fitness: list[torch.Tensor] = []

        # Midpoint probing flag (probe once after first DE tell)
        self._midpoint_probed = False

        # Alternating DE restart counter for low-dim
        self._de_restart_count = 0

        # Accumulated search pool across CMA-ES phases
        self._search_population: torch.Tensor | None = None
        self._search_population_fitness: torch.Tensor | None = None
        self._search_pool_limit = self._search_pool_max

        # Fitness function reference (set externally for polish)
        self._fitness_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return full PhasedDFO state as a serializable dict.

        Extends :meth:`BaseOptimizer.state_dict` with all phase-machine,
        nested sub-optimizer, and budget-tracking fields. Nested SHADE and
        CMAES states are stored recursively so that :meth:`load_state_dict`
        can reconstruct bit-exact continuation on the same device.

        Notes
        -----
        Cross-device loads (e.g. CPU checkpoint → CUDA optimizer) fall back
        to seed-based RNG re-initialisation and are non-bit-exact.
        """
        state = super().state_dict()
        state.update(
            {
                "de": self._de.to_dict(),
                "valley": self._valley.to_dict(),
                "polish": self._polish.to_dict(),
                "cma": self._cma.to_dict(),
                "_shade_state": self._shade.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore PhasedDFO from a dict produced by :meth:`state_dict`.

        Reconstructs nested SHADE / CMAES sub-optimizers and restores
        all phase-machine state. The optimizer object must have been
        created with the same ``dim``, ``bounds``, and ``budget`` as the
        one that produced the state dict.
        """
        super().load_state_dict(state)
        self._de = PhasedDEState.from_dict(state["de"])
        self._valley = PhasedValleyState.from_dict(state["valley"])
        self._polish = PhasedPolishState.from_dict(
            state["polish"],
            device=self.device,
            dtype=self.dtype,
        )
        self._cma = PhasedCMAState.from_dict(
            state["cma"],
            dim=self.dim,
            bounds=self._raw_bounds,
            device=self.device,
            dtype=self.dtype,
        )

        # Nested SHADE
        self._shade.load_state_dict(state["_shade_state"])

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    @property
    def phase(self) -> int:
        """Current phase: 0=DE, 1=CMA-ES, 2=Polish, 3=Done."""
        return self._phase

    @property
    def fe_count(self) -> int:
        """Total function evaluations used so far."""
        return self._fe_count

    @property
    def done(self) -> bool:
        """True when the optimizer has exhausted its budget or entered Polish phase.

        Phase >= 2 means Polish is running — this phase is only accessible via
        ``optimize()``, not ask/tell.  A pure ask/tell loop will never reach
        phase 2 because ``done`` returns True first, preventing an infinite spin.

        Use as the loop condition for ask/tell iteration::

            while not opt.done:
                candidates = opt.ask()
                opt.tell(fitness_fn(candidates))
        """
        return self._phase >= 2 or self._fe_count >= self._budget

    @property
    def budget(self) -> int:
        """Total evaluation budget."""
        return self._budget

    @property
    def space(self) -> SearchSpace | None:
        """The SearchSpace used for encoding, or None if not set."""
        return self._space

    @property
    def pop_spread(self) -> float:
        """Std of SHADE population (conditioning proxy). NaN before first ask."""
        if not self._shade._initialized:
            return float("nan")
        return float(self._shade.population.std().item())

    @property
    def step_size(self) -> float:
        """Mean adaptive F (DE phase) or CMA-ES sigma (CMA-ES phase). NaN before first ask."""
        if not self._shade._initialized:
            return float("nan")
        if self._phase == 1 and self._cmaes is not None:
            return float(self._cmaes.sigma)
        return float(self._shade.memory_F.mean().item())

    @property
    def de_improvement(self) -> float:
        """Total improvement accumulated during the DE phase. Zero before first tell."""
        if math.isinf(self._de_phase_start_f):
            return 0.0
        return max(0.0, self._de_phase_start_f - float(self.best_fitness))

    @property
    def cmaes_improvement(self) -> float:
        """Total improvement accumulated since CMA-ES was first entered. Zero if not yet entered."""
        if not self._cmaes_entered:
            return 0.0
        return max(0.0, self._cmaes_overall_start_f - float(self.best_fitness))

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------

    def ask(self) -> torch.Tensor:
        """Generate candidate solutions from the active phase's sub-optimizer.

        Returns
        -------
        torch.Tensor
            (batch_size, dim) candidates. Batch size varies by phase.

        """
        if self._phase == 0:
            return self._shade.ask()
        if self._phase == 1:
            if self._high_dim:
                if self._cmaes_portfolio is None:
                    self._enter_cmaes_phase_portfolio()
                assert self._cmaes_portfolio is not None
                active_indices = self._portfolio_active_indices_for_next_ask()
                self._portfolio_active_indices = active_indices
                return torch.cat(
                    [self._cmaes_portfolio[i].ask() for i in active_indices],
                    dim=0,
                )
            if self._cmaes is None:
                self._enter_cmaes_phase()
            assert self._cmaes is not None
            return self._cmaes.ask()
        # Phase 2 (Polish) or 3 (Done): return empty tensor
        return torch.empty(0, self.dim, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------

    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Report fitness for candidates and update internal state.

        Parameters
        ----------
        candidates : torch.Tensor
            (batch_size, dim) solutions from the last ask().
        fitness : torch.Tensor
            (batch_size,) objective values (lower is better).

        """
        if self._phase == 0:
            self._tell_de(candidates, fitness)
        elif self._phase == 1:
            self._tell_cmaes(candidates, fitness)
        # Phase 2/3: no-op

    # ------------------------------------------------------------------
    # DE phase (Phase 0)
    # ------------------------------------------------------------------

    def _tell_de(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Process DE phase tell: update SHADE, apply Levy flights, check stagnation."""
        n = candidates.shape[0]

        # Record pre-tell best for stagnation tracking
        pre_best = self._shade.best_fitness.item()
        if math.isinf(pre_best):
            pre_best = fitness.min().item()

        # Store pre-tell fitness for trial_gain computation.
        parent_fitness = self._shade.fitness.clone()

        # Delegate to SHADE
        self._shade.tell(candidates, fitness)
        self._fe_count += n

        # Update our global best from SHADE
        self._update_best(self._shade.population, self._shade.fitness)

        # Capture best fitness at DE phase start (set-once on first tell)
        if self._de_step_count == 0:
            self._de_phase_start_f = float(self.best_fitness)

        # High-dim CR override: force CR=1.0 for full crossover exploration
        if self._high_dim and self._shade._generation > 1:
            self._shade.memory_CR.fill_(1.0)

        # Compute trial_gain and accepted_ratio from DE step.
        pop_size = self._shade.pop_size
        if self._shade._generation > 1:
            accepted = self._shade.fitness < parent_fitness[:pop_size]
            if accepted.any():
                trial_delta = (
                    parent_fitness[:pop_size][accepted] - self._shade.fitness[accepted]
                ).clamp_min(0.0)
                self._trial_gain = trial_delta.sum().item() / max(pop_size, 1)
                self._accepted_ratio = accepted.float().mean().item()
            else:
                self._trial_gain = 0.0
                self._accepted_ratio = 0.0
        else:
            self._trial_gain = 0.0
            self._accepted_ratio = 0.5

        # Apply Levy flight perturbation to worst half of population
        # Also computes levy_gain and levy_ratio.
        self._levy_gain = 0.0
        self._levy_ratio = 0.0
        if self._shade._generation > 1 and self._fitness_fn is not None:
            self._apply_levy_flights()

        # Population reduction for high-dim.
        current_pop = self._shade.population.shape[0]
        if self._high_dim and current_pop > self._high_dim_de_min_pop:
            reduction_progress = min(
                1.0,
                (self._de_step_count + 1) / max(self._de_target_steps, 1),
            )
            target_pop = round(
                self._initial_pop_size
                - (self._initial_pop_size - self._high_dim_de_min_pop) * reduction_progress,
            )
            target_pop = max(self._high_dim_de_min_pop, target_pop)
            if target_pop < current_pop:
                keep = self._shade.fitness.argsort()[:target_pop]
                self._shade.population = self._shade.population[keep].clone()
                self._shade.fitness = self._shade.fitness[keep].clone()
                self._shade.pop_size = target_pop
                # Resize SHADE internal buffers to match new pop_size
                self._shade._trial_F = torch.empty(
                    target_pop,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._shade._trial_CR = torch.empty(
                    target_pop,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._shade._trials = torch.empty(
                    target_pop,
                    self.dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                # Trim archive if it exceeds new pop_size
                if self._shade._archive is not None and self._shade._archive.shape[0] > target_pop:
                    perm = self._randperm(self._shade._archive.shape[0])
                    self._shade._archive = self._shade._archive[perm[:target_pop]].clone()

        # Track elite solutions from DE
        top_k = min(10, self._shade.pop_size)
        top_idx = self._shade.fitness.argsort()[:top_k]
        for idx in top_idx:
            self._elite_solutions.append(self._shade.population[idx].clone())
            self._elite_fitness.append(self._shade.fitness[idx].clone())

        # Stagnation detection
        post_best = self.best_fitness.item()
        self._de_step_count += 1
        self._update_de_stagnation(pre_best, post_best)

        # Check phase transition
        stag_threshold = self._get_de_stagnation_threshold()
        if self._stagnation_counter >= stag_threshold:
            if self._high_dim:
                # High-dim: break out of DE to CMA-ES
                self._phase = 1
                self._generation += 1
                return
            # Low-dim population restart on stagnation.
            self._low_dim_pop_restart()
            # After restart, continue DE (do not transition)

        if self._fe_count >= self._de_budget:
            self._phase = 1
            self._generation += 1
            return

        # Adaptive Levy step_size.
        if self._stagnation_counter == 0:
            self._step_size = min(
                self._config.step_size_max * self._search_span, self._step_size * 1.05
            )
        else:
            self._step_size = max(
                self._config.step_size_min * self._search_span, self._step_size * 0.95
            )

        self._generation += 1

    def _apply_levy_flights(self) -> None:
        """Perturb worst half of SHADE population with Levy flights."""
        pop = self._shade.population
        fit = self._shade.fitness
        n = pop.shape[0]
        half = n // 2

        # Sort by fitness (ascending = best first)
        sorted_idx = fit.argsort()
        worst_idx = sorted_idx[half:]

        # Compute search progress
        progress = min(1.0, self._fe_count / max(1, self._de_budget))

        # Use adaptive step_size instead of fixed 0.1.
        worst = pop[worst_idx]
        perturbed = levy_flight_perturbation(
            worst,
            alpha=1.5,
            step_scale=self._step_size,
            progress=progress,
            generator=self._gen,
        )
        perturbed = clamp_to_bounds(perturbed, self.lb, self.ub)

        # Evaluate perturbations
        if self._fitness_fn is not None:
            # Store pre-perturbation fitness for levy_gain.
            worst_fitness_before = fit[worst_idx].clone()

            perturbed_fit = self._fitness_fn(perturbed)
            self._fe_count += perturbed.shape[0]

            # Keep improvements via greedy selection
            improved = perturbed_fit < fit[worst_idx]
            if improved.any():
                # Compute levy_gain and levy_ratio.
                levy_delta = (worst_fitness_before[improved] - perturbed_fit[improved]).clamp_min(
                    0.0,
                )
                self._levy_gain = levy_delta.sum().item() / max(worst_idx.numel(), 1)
                self._levy_ratio = improved.float().mean().item()

                imp_local = improved.nonzero(as_tuple=True)[0]
                global_idx = worst_idx[imp_local]
                self._shade.population[global_idx] = perturbed[imp_local]
                self._shade.fitness[global_idx] = perturbed_fit[imp_local]
                self._update_best(perturbed[imp_local], perturbed_fit[imp_local])
                # Direct stagnation reset on Levy improvement.
                self._stagnation_counter = 0

    def _low_dim_pop_restart(self) -> None:
        return _de.low_dim_pop_restart(self)

    def _probe_midpoint(self) -> None:
        return _de.probe_midpoint(self)

    def _update_de_stagnation(self, pre_best: float, post_best: float) -> None:
        return _de.update_de_stagnation(self, pre_best, post_best)

    def _get_de_stagnation_threshold(self) -> int:
        return _de.get_de_stagnation_threshold(self)

    # ------------------------------------------------------------------
    # CMA-ES phase (Phase 1)
    # ------------------------------------------------------------------

    def _enter_cmaes_phase(self) -> None:
        """Create CMA-ES with warm-started covariance from DE elite."""
        return _cma.enter_cmaes_phase(self)

    def _is_valley_entry_branch(self, idx: int) -> bool:
        """Return True for the dim40+ incumbent LM-CMA portfolio branch."""
        return _cma.is_valley_entry_branch(self, idx)

    def _portfolio_lambdas_for_dim(self) -> tuple[int, ...]:
        """Return the CMA portfolio population schedule for this dimension."""
        return _cma.portfolio_lambdas_for_dim(self)

    def _portfolio_active_indices_for_next_ask(self) -> tuple[int, ...]:
        """Return portfolio branches to sample on the next CMA generation."""
        return _cma.portfolio_active_indices_for_next_ask(self)

    def _in_valley_terminal_focus_window(self) -> bool:
        """Return True when final dim40 CMA budget should stay local."""
        return _cma.in_valley_terminal_focus_window(self)

    def _update_valley_focus_schedule(
        self,
        active_indices: tuple[int, ...],
        valley_branch_improved: bool,
    ) -> None:
        """Adapt the dim40 incumbent-only burst after one portfolio generation."""
        return _cma.update_valley_focus_schedule(self, active_indices, valley_branch_improved)

    def _seed_path_memory_from_elites(self, cma: CMAES, center: torch.Tensor) -> None:
        """Seed a CMA path-memory branch from the current DE elite cloud."""
        return _cma.seed_path_memory_from_elites(self, cma, center)

    def _enter_cmaes_phase_portfolio(self) -> None:
        """Initialize K=4 parallel CMA-ES portfolio for high-dim phase."""
        return _cma.enter_cmaes_phase_portfolio(self)

    def _restart_portfolio_branch(self, idx: int) -> None:
        """Restart one CMA-ES portfolio branch from a new random x0."""
        return _cma.restart_portfolio_branch(self, idx)

    def _get_bounds_tuple(self) -> tuple[float, float]:
        """Return scalar bounds as a tuple for sub-optimizer construction."""
        return _cma.get_bounds_tuple(self)

    def _compute_cmaes_phase_budgets(self, total: int) -> list[int]:
        """Split remaining CMA-ES budget across IPOP phases."""
        return _cma.compute_cmaes_phase_budgets(self, total)

    def _tell_cmaes(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Process CMA-ES phase tell: delegate, check stagnation, handle restarts."""
        # K-portfolio path (high-dim only)
        if self._high_dim:
            if self._cmaes_portfolio is None:
                self._enter_cmaes_phase_portfolio()
            assert self._cmaes_portfolio is not None
            active_indices = self._portfolio_active_indices
            if not active_indices:
                active_indices = tuple(range(len(self._cmaes_portfolio)))
            portfolio_lambdas = _cma.portfolio_lambdas_for_dim(self)
            expected_n = sum(portfolio_lambdas[i] for i in active_indices)
            if candidates.shape[0] != expected_n:
                full_indices = tuple(range(len(self._cmaes_portfolio)))
                full_n = sum(portfolio_lambdas)
                if candidates.shape[0] == full_n:
                    active_indices = full_indices
                else:
                    msg = (
                        "CMA portfolio tell received a batch with incompatible "
                        f"size {candidates.shape[0]} for active branches {active_indices}"
                    )
                    raise ValueError(msg)
            # Split candidates back to per-branch chunks and tell each branch
            offset = 0
            valley_branch_improved = False
            for i in active_indices:
                cma = self._cmaes_portfolio[i]
                lam = portfolio_lambdas[i]
                branch_candidates = candidates[offset : offset + lam]
                branch_fitness = fitness[offset : offset + lam]
                offset += lam
                if branch_candidates.shape[0] > 0:
                    cma.tell(branch_candidates, branch_fitness)
                    if self._is_valley_entry_branch(i):
                        self._update_best(branch_candidates, branch_fitness)
                    # Track per-branch stagnation
                    branch_best = branch_fitness.min().item()
                    previous_branch_best = self._portfolio_best_f[i]
                    branch_improved = branch_best < previous_branch_best
                    if branch_improved:
                        self._portfolio_best_f[i] = branch_best
                        self._portfolio_stag[i] = 0
                    else:
                        self._portfolio_stag[i] += 1
                    if (
                        self._is_valley_entry_branch(i)
                        and branch_improved
                        and math.isfinite(previous_branch_best)
                    ):
                        valley_branch_improved = True
                    # Restart on stagnation OR sigma collapse.  Sigma collapse
                    # is critical for small-sigma branches (e.g. branch 3,
                    # sigma0=0.020) that slowly descend into a local basin
                    # without triggering the stagnation counter — their sigma
                    # shrinks to near-zero while still improving fractionally.
                    sigma_stop = 1e-3 * self._portfolio_sigma0[i]
                    if (
                        self._portfolio_stag[i] >= self._portfolio_branch_stag_limit
                        or cma.sigma < sigma_stop
                    ):
                        self._restart_portfolio_branch(i)
            # Sync global best
            self._update_best(candidates, fitness)
            self._fe_count += candidates.shape[0]
            self._update_valley_focus_schedule(active_indices, valley_branch_improved)
            if self._fe_count >= self._cmaes_budget:
                self._phase = 2
            self._portfolio_generation += 1
            self._portfolio_active_indices = ()
            self._generation += 1
            return

        if self._cmaes is None:
            self._enter_cmaes_phase()
        assert self._cmaes is not None

        n = candidates.shape[0]

        # Delegate to CMA-ES
        self._cmaes.tell(candidates, fitness)
        self._fe_count += n

        # Update our global best from CMA-ES
        self._update_best(candidates, fitness)

        # Track elite solutions from CMA-ES
        top_k = min(5, n)
        top_idx = fitness.argsort()[:top_k]
        for idx in top_idx:
            self._elite_solutions.append(candidates[idx].clone())
            self._elite_fitness.append(fitness[idx].clone())

        # Check stagnation
        current_best = self.best_fitness.item()
        if abs(current_best - self._cmaes_phase_best_f) < 1e-12:
            self._cmaes_stagnation_counter += 1
        else:
            self._cmaes_phase_best_f = min(self._cmaes_phase_best_f, current_best)
            self._cmaes_stagnation_counter = 0

        # Check sigma collapse
        sigma_collapsed = self._cmaes.sigma < 1e-15

        # Stagnation threshold depends on phase
        if self._cmaes_phase_idx == 0 and self._high_dim:
            stag_limit = self._high_dim_cma_warm_stagnation
        elif self._high_dim:
            stag_limit = self._high_dim_cma_stagnation
        else:
            stag_limit = self._cma_stagnation

        # Check per-phase budget.
        # Use the recorded CMA entry point (_cmaes_fe_start) as the offset so
        # that fe_in_cmaes is accurate even when DE exited early (before its
        # theoretical budget was exhausted).  Using the theoretical offset
        # (budget - cmaes_budget) would give a false phase_budget_exhausted
        # signal on the very first CMA iteration whenever DE terminates via
        # stagnation before consuming its full budget cap.
        phase_budget_idx = min(self._cmaes_phase_idx, len(self._cmaes_phase_budgets) - 1)
        phase_fe_limit = sum(self._cmaes_phase_budgets[: phase_budget_idx + 1])
        fe_in_cmaes = self._fe_count - self._cmaes_fe_start
        phase_budget_exhausted = fe_in_cmaes >= phase_fe_limit

        need_restart = (
            self._cmaes_stagnation_counter >= stag_limit
            or sigma_collapsed
            or phase_budget_exhausted
        )

        if need_restart:
            # Merge current phase population into search pool
            if self._cmaes is not None:
                phase_pop = self._cmaes.population
                phase_fit = self._cmaes.fitness
                if phase_pop is not None and phase_fit is not None:
                    self._search_population, self._search_population_fitness = _merge_search_pool(
                        self._search_population,
                        self._search_population_fitness,
                        phase_pop,
                        phase_fit,
                        self._search_pool_limit,
                    )

            self._cmaes_phase_idx += 1
            if (
                self._cmaes_phase_idx >= self._cmaes_phase_count
                or self._fe_count >= self._cmaes_budget
            ):
                # Transition to polish
                self._phase = 2
            else:
                self._restart_cmaes()

        # Also transition if overall CMA-ES budget is exhausted
        if self._fe_count >= self._cmaes_budget:
            self._phase = 2

        self._generation += 1

    def _restart_cmaes(self) -> None:
        """Perform an IPOP restart of CMA-ES with doubled population."""
        return _cma.restart_cmaes(self)

    def _sample_restart_mean(self, phase_idx: int, span: float) -> torch.Tensor:
        """Sample restart center using 4-mode cycling."""
        return _cma.sample_restart_mean(self, phase_idx, span)

    # ------------------------------------------------------------------
    # Multistart basin exploration (low-dim only)
    # ------------------------------------------------------------------

    def _multistart_basin_explore(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        budget_limit: int,
    ) -> None:
        """Run short CMA-ES restarts from random starting points to escape local basins."""
        return _be.multistart_basin_explore(self, fitness_fn, budget_limit)

    # ------------------------------------------------------------------
    # Polish phase (Phase 2)
    # ------------------------------------------------------------------

    def _run_polish(self, fitness_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Execute polish phase: directional + coordinate + scipy precision chain.

        Thin delegating wrapper around :func:`torch_dfo.phased._polish_phase.run_polish`.
        See that function for the full low-dim / high-dim pipeline documentation.
        """
        _pp.run_polish(self, fitness_fn)

    def _build_polish_directions(
        self,
        best_x: torch.Tensor,
        directional_budget: int,
    ) -> tuple[torch.Tensor | None, int]:
        """Build search directions for directional basin search.

        Thin delegating wrapper around
        :func:`torch_dfo.phased._polish_phase.build_polish_directions`.
        """
        return _pp.build_polish_directions(self, best_x, directional_budget)

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the optimizer to its initial state.

        Clears all phase/budget/stagnation state and re-creates the internal
        SHADE population.  Does **not** re-apply ``initial_points`` — create a
        new ``PhasedDFO`` instance if you need to re-seed the population.

        After ``reset()``, the optimizer can be re-run with ``optimize()`` or
        a fresh ask/tell loop.
        """
        # Phase and budget counters
        self._phase = 0
        self._fe_count = 0
        self._generation = 0

        # Re-create SHADE sub-optimizer (no initial_population on reset)
        self._shade = SHADE(
            self.dim,
            self._raw_bounds,
            self._initial_pop_size,
            device=self.device,
            dtype=self.dtype,
        )

        # CMA-ES state
        self._cmaes = None
        self._cmaes_portfolio = None
        self._portfolio_stag = []
        self._portfolio_best_f = []
        self._portfolio_sigma0 = []
        self._portfolio_generation = 0
        self._portfolio_active_indices = ()
        self._valley_focus_remaining = 0
        self._valley_focus_streak = 0
        self._cmaes_phase_idx = 0
        self._cmaes_fe_start = 0
        self._cmaes_stagnation_counter = 0
        self._cmaes_phase_best_f = float("inf")

        # Best tracking (mirrors BaseOptimizer.__init__)
        self.best_solution = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        self.best_fitness = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)

        # Elite archive
        self._elite_solutions = []
        self._elite_fitness = []

        # DE stagnation counters
        self._stagnation_counter = 0
        self._de_progress_ema = 0.0
        self._de_progress_baseline = 0.0
        self._de_step_count = 0
        self._de_best_f_prev = float("inf")

        self._de_phase_start_f = float("inf")
        self._cmaes_overall_start_f = float("inf")
        self._cmaes_entered = False

        # Richer stagnation signal tracking
        self._trial_gain = 0.0
        self._levy_gain = 0.0
        self._accepted_ratio = 0.0
        self._levy_ratio = 0.0

        # Adaptive step size
        span = (self.ub - self.lb).mean().item()
        self._step_size = _compute_step_size_init(self.dim) * span

        # Flags and counters
        self._midpoint_probed = False
        self._de_restart_count = 0

        # Accumulated CMA-ES search pool
        self._search_population = None
        self._search_population_fitness = None
        self._search_pool_limit = self._search_pool_max

        # Fitness function reference (cleared; will be re-set by optimize())
        self._fitness_fn = None

    # ------------------------------------------------------------------
    # optimize() convenience method
    # ------------------------------------------------------------------

    @overload
    def optimize(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Callable[[int, int, torch.Tensor], None] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def optimize(
        self,
        fitness_fn: Callable[[list[dict[str, Any]]], torch.Tensor],
        callback: Callable[[int, int, torch.Tensor], None] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def optimize(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor]
        | Callable[[list[dict[str, Any]]], torch.Tensor],
        callback: Callable[[int, int, torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the complete optimization pipeline.

        When a :class:`~torch_dfo.space.SearchSpace` was provided at
        construction time, *fitness_fn* receives a ``list[dict]`` of decoded
        configs.  Otherwise it receives a raw ``(N, dim)`` tensor.

        Args:
            fitness_fn: batched objective — either ``(N, dim) -> (N,)`` tensor
                callable or ``list[dict] -> (N,)`` tensor callable when
                ``space`` was set.
            callback: optional callable(phase, generation, best_f) per
                generation.

        Returns:
            (best_solution, best_fitness)

        """
        # Decode bridge: when a SearchSpace is attached, wrap fitness_fn so
        # that the optimizer always receives a (N, dim) tensor internally but
        # the user's function receives a list[dict] of decoded configs.
        if self._space is not None:
            _space = self._space
            _raw_fn = fitness_fn

            def actual_fn(candidates: torch.Tensor) -> torch.Tensor:
                return _raw_fn(_space.decode(candidates))  # type: ignore[arg-type]

        else:
            actual_fn = fitness_fn  # type: ignore[assignment]

        self._fitness_fn = actual_fn

        while self._phase < 2 and self._fe_count < self._budget:
            candidates = self.ask()
            if candidates.shape[0] == 0:
                break
            fitness = actual_fn(candidates)
            self._fe_count_before_tell = self._fe_count
            self.tell(candidates, fitness)

            # Probe midpoint once after the first DE tell
            if not self._midpoint_probed and self._phase == 0 and self._fe_count > 0:
                self._probe_midpoint()
            if callback:
                callback(self._phase, self._generation, self.best_fitness)

        # Low-dim multistart basin exploration before polish.
        # Uses a separate torch.Generator; global RNG state is saved/restored.
        if self._phase >= 2 and self._fe_count < self._budget and not self._high_dim:
            explore_budget = int(self._budget * self._basin_explore_budget_frac)
            if explore_budget > 50:
                torch_rng_state = torch.random.get_rng_state()
                explore_end = min(self._budget, self._fe_count + explore_budget)
                self._multistart_basin_explore(actual_fn, explore_end)
                torch.random.set_rng_state(torch_rng_state)

        # Phase 2: Polish (self-contained, calls actual_fn directly)
        if self._phase >= 2 and self._fe_count < self._budget:
            self._run_polish(actual_fn)

        self._phase = 3
        return self.best()
