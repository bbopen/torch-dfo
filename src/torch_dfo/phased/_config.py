"""User-tunable configuration knobs for :class:`PhasedDFO`.

All fields default to the v0.9 algorithm constants so that
``PhasedConfig()`` reproduces v0.9 behavior bit-exactly.  Users who want to
override individual knobs construct a new frozen instance and pass it to
``PhasedDFO(config=...)``.  The dataclass is the single source of truth for
these values; the module-level aliases exposed from ``torch_dfo.phased``
(``ELITE_FRACTION``, ``CMA_ES_RESTART_MODES``, ...) are derived from the
dataclass field defaults and kept for backward-compatible imports.

Fields that encode algorithm invariants rather than tunable knobs
(``_EPS``, debug-mechanism tag sets) remain as module-level constants in
``orchestrator.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhasedConfig:
    """User-tunable configuration for :class:`PhasedDFO`.

    All defaults reproduce v0.9 behavior bit-exactly.  Instances are
    immutable; override by constructing a new instance with the desired
    field values.
    """

    # ------------------------------------------------------------------
    # DE phase — progress tracking
    # ------------------------------------------------------------------
    high_dim_de_progress_ratio: float = 0.35
    high_dim_de_progress_floor: float = 0.10

    # ------------------------------------------------------------------
    # Adaptive Levy step size (fractions of span)
    # ------------------------------------------------------------------
    step_size_min: float = 1e-5
    step_size_max: float = 1.0

    # ------------------------------------------------------------------
    # Low-dim population restart
    # ------------------------------------------------------------------
    elite_fraction: float = 0.1

    # ------------------------------------------------------------------
    # CMA-ES portfolio
    # ------------------------------------------------------------------
    cma_es_pop_growth: int = 2
    cma_es_restart_cov_blend: float = 0.8
    cma_es_restart_modes: int = 4
    k_portfolio_lambdas: tuple[int, ...] = (24, 12, 12, 12)
    k_portfolio_sigma_fracs: tuple[float, ...] = (0.200, 0.043, 0.0093, 0.002)

    # ------------------------------------------------------------------
    # Dim>=40 valley-entry branch
    # ------------------------------------------------------------------
    high_dim_valley_entry_dim: int = 40
    high_dim_valley_entry_branch: int = 1
    high_dim_valley_entry_path_memory: int = 8
    high_dim_valley_entry_path_scale: float = 0.65
    high_dim_valley_entry_line_samples: int = 2
    high_dim_valley_entry_line_scale: float = 1.0
    high_dim_valley_entry_restart_jitter: float = 0.25
    high_dim_valley_entry_portfolio_lambdas: tuple[int, ...] = (18, 12, 8, 6)
    high_dim_valley_entry_focus_cycle: int = 3
    high_dim_valley_entry_max_focus_cycle: int = 5
    high_dim_valley_entry_focus_eval_ratio: float = 1.0
    high_dim_valley_entry_max_focus_eval_ratio: float = 1.75
    high_dim_valley_entry_terminal_focus_fraction: float = 0.25

    # ------------------------------------------------------------------
    # CMA-ES restart jitter scales (fractions of span)
    # ------------------------------------------------------------------
    cma_es_elite_restart_jitter: float = 0.05
    cma_es_differential_restart_scale: float = 1.0
    cma_es_differential_restart_jitter: float = 0.03
    cma_es_mirror_restart_jitter: float = 0.08

    # ------------------------------------------------------------------
    # Search pool expansion factor
    # ------------------------------------------------------------------
    cma_es_search_pool_factor: int = 2

    # ------------------------------------------------------------------
    # Directional line-search grid resolution
    # ------------------------------------------------------------------
    directional_refinement_stages: int = 2
    directional_refinement_points: int = 5
    directional_window_shrink: float = 0.4
    directional_priority_hop_scale: float = 1.0

    # ------------------------------------------------------------------
    # Coordinate line-search grid resolution
    # ------------------------------------------------------------------
    coordinate_refinement_stages: int = 2
    coordinate_refinement_points: int = 5
    coordinate_window_shrink: float = 0.35
