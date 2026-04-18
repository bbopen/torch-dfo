"""DE-phase helpers for :class:`PhasedDFO`.

This module contains the DE (Differential Evolution) phase specific logic
extracted from :class:`torch_dfo.phased.orchestrator.PhasedDFO`. The functions
here take the orchestrator instance as the first argument (``opt``) rather
than being methods on the class; this keeps the module a leaf in the import
graph and avoids circular imports with ``orchestrator.py``.

The public entry points are:

* :func:`probe_midpoint`
* :func:`low_dim_pop_restart`
* :func:`update_de_stagnation`
* :func:`get_de_stagnation_threshold`

Each is a pure move of the corresponding ``_probe_midpoint`` /
``_low_dim_pop_restart`` / ``_update_de_stagnation`` /
``_get_de_stagnation_threshold`` method from ``PhasedDFO``, with ``self``
renamed to ``opt``. No algorithmic behavior is changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_dfo.phased.orchestrator import PhasedDFO


_EPS = 1e-12


def probe_midpoint(opt: PhasedDFO) -> None:
    """Evaluate the search-space midpoint and inject if better than worst.

    Called once after the first DE tell to seed the population with the
    box center, which is often a reasonable starting point.
    """
    if opt._midpoint_probed:
        return
    if opt._fitness_fn is None:
        return
    midpoint = (opt.lb + opt.ub) / 2
    mid_f = opt._fitness_fn(midpoint.unsqueeze(0)).squeeze()
    opt._fe_count += 1
    if torch.isfinite(mid_f):
        worst_idx = int(opt._shade.fitness.argmax().item())
        if mid_f < opt._shade.fitness[worst_idx]:
            opt._shade.population[worst_idx] = midpoint
            opt._shade.fitness[worst_idx] = mid_f
            opt._update_best(midpoint.unsqueeze(0), mid_f.unsqueeze(0))
    opt._midpoint_probed = True


def low_dim_pop_restart(opt: PhasedDFO) -> None:
    """Restart low-dim population on stagnation with alternating modes.

    Even restarts (count % 2 == 0): full restart -- completely random
    population with all fitness set to inf (critical for multi-modal
    functions like f24 where diversity is needed to escape basins).

    Odd restarts (count % 2 == 1): partial restart -- keep the elite
    fraction, randomize the rest.
    """
    opt._de_restart_count += 1
    pop_size = opt._shade.pop_size
    span = opt.ub - opt.lb

    if opt._de_restart_count % 2 == 0:
        # Full restart: completely random population for basin diversity
        new_positions = opt._rand(pop_size, opt.dim) * span + opt.lb
        opt._shade.population = new_positions
        opt._shade.fitness = torch.full(
            (pop_size,),
            float("inf"),
            device=opt.device,
            dtype=opt.dtype,
        )

        # Re-evaluate if fitness_fn is available
        if opt._fitness_fn is not None:
            new_fit = opt._fitness_fn(new_positions)
            opt._fe_count += new_positions.shape[0]
            opt._shade.fitness = new_fit
            opt._update_best(new_positions, new_fit)
    else:
        # Partial restart: keep elite, randomize the rest
        elite_count = max(2, int(pop_size * opt._config.elite_fraction))
        sorted_idx = opt._shade.fitness.argsort()
        restart_idx = sorted_idx[elite_count:]

        if restart_idx.numel() == 0:
            opt._stagnation_counter = 0
            return

        new_positions = opt._rand(restart_idx.numel(), opt.dim) * span + opt.lb
        opt._shade.population[restart_idx] = new_positions
        opt._shade.fitness[restart_idx] = float("inf")

        # Re-evaluate new positions if fitness_fn is available
        if opt._fitness_fn is not None:
            new_fit = opt._fitness_fn(new_positions)
            opt._fe_count += new_positions.shape[0]
            opt._shade.fitness[restart_idx] = new_fit
            opt._update_best(new_positions, new_fit)

    opt._stagnation_counter = 0


def update_de_stagnation(opt: PhasedDFO, pre_best: float, post_best: float) -> None:
    """Update DE stagnation counter.

    Uses simple binary logic for the counter (improve → 0,
    no-improve → +1) while maintaining the EMA progress signal for the
    adaptive threshold computation.
    """
    # Simple binary counter: improve -> reset, no-improve -> increment.
    if post_best < pre_best:
        opt._stagnation_counter = 0
    else:
        opt._stagnation_counter += 1

    # Maintain EMA signal for adaptive threshold (high-dim only).
    if opt._high_dim:
        progress_scale = max(abs(pre_best), 1.0)
        step_signal = (
            opt._accepted_ratio
            + 0.5 * opt._levy_ratio
            + 4.0 * ((opt._trial_gain + 0.5 * opt._levy_gain) / progress_scale)
        )
        opt._de_progress_ema = 0.75 * opt._de_progress_ema + 0.25 * step_signal
        # Baseline calibration
        if opt._de_step_count <= opt._de_baseline_steps or opt._de_progress_baseline <= _EPS:
            opt._de_progress_baseline = max(
                opt._de_progress_baseline,
                step_signal,
                opt._de_progress_ema,
            )


def get_de_stagnation_threshold(opt: PhasedDFO) -> int:
    """Return stagnation threshold for DE phase exit.

    Adaptive threshold with progress floor check.
    """
    if opt._high_dim:
        # Adaptive stagnation threshold.
        if opt._de_step_count >= opt._de_baseline_steps and opt._de_progress_baseline > _EPS:
            progress_ratio = opt._de_progress_ema / max(opt._de_progress_baseline, _EPS)
            if (
                opt._de_progress_ema >= opt._config.high_dim_de_progress_floor
                and progress_ratio >= opt._config.high_dim_de_progress_ratio
            ):
                return opt._de_max_stagnation
        return opt._restart_stagnation
    return opt._restart_stagnation
