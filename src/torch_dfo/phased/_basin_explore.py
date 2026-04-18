"""Basin-exploration helpers for :class:`PhasedDFO`.

This module contains the low-dim multistart basin-exploration logic
extracted from :class:`torch_dfo.phased.orchestrator.PhasedDFO`.  The
public entry point takes the orchestrator instance as the first argument
(``opt``) rather than being a method on the class; this keeps the module
a leaf in the import graph and avoids circular imports with
``orchestrator.py``.

The public entry point mirrors the private ``PhasedDFO`` method it
replaces:

* :func:`multistart_basin_explore`

It is a logic-preserving move of the corresponding
``_multistart_basin_explore`` method from ``PhasedDFO``, with ``self``
renamed to ``opt``.  No algorithmic behavior is changed.

Basin exploration is invoked from ``PhasedDFO.optimize`` on low-dim
problems after the CMA-ES phase and before the polish chain; it does
not touch module-level orchestrator helpers (``_debug_is_disabled``,
``_compute_valley_focus_generation_bounds``, ``_normalize_covariance``,
``_merge_search_pool``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from torch_dfo.cmaes import CMAES
from torch_dfo.phased._common import _EPS

if TYPE_CHECKING:
    from torch_dfo.phased.orchestrator import PhasedDFO


def multistart_basin_explore(
    opt: PhasedDFO,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    budget_limit: int,
) -> None:
    """Run short CMA-ES restarts from random starting points to escape local basins.

    Uses a **separate torch.Generator** so the main optimizer's RNG state
    remains untouched.  This is critical: downstream polishers rely on a
    deterministic torch RNG sequence for reproducible precision.

    Designed for low-dim multi-modal landscapes (e.g. Lunacek bi-Rastrigin f24)
    where a single CMA run converges to whichever basin it starts nearest to.
    Multiple random restarts give high probability of finding the global basin.

    The caller must save and restore the global torch RNG state around this
    method (the separate generator protects the CMAES-internal RNG, but
    fitness_fn evaluations may touch global state).
    """
    remaining = budget_limit - opt._fe_count
    if remaining < 50:
        return

    dim = opt.dim
    device = opt.device
    dtype = opt.dtype
    search_span = float((opt.ub[0] - opt.lb[0]).item())

    n_restarts = opt._basin_explore_restarts
    pop_size = min(
        max(8, 4 + int(3 * math.log(max(dim, 2)))),
        remaining // n_restarts,
    )
    if pop_size < 4:
        return
    # Ensure even pop size for mirrored sampling
    pop_size = pop_size + (pop_size % 2)

    sigma_init = 0.25 * search_span
    sigma_min = opt._cma_sigma_min * search_span
    sigma_max = opt._cma_restart_sigma_max * search_span

    # Create a separate RNG for basin exploration (deterministic but independent)
    basin_seed = int(opt._gen.initial_seed() ^ 999_983) & 0x7FFF_FFFF
    basin_gen = torch.Generator(device=opt._gen_device).manual_seed(basin_seed)

    best_fitness_val = float(opt.best_fitness.item())

    # Build a CMAES instance with the isolated RNG.  We pass sigma0 as a
    # fraction of span (CMAES multiplies sigma0 * span internally).
    bounds = (float(opt.lb[0].item()), float(opt.ub[0].item()))
    cma = CMAES(
        dim=dim,
        bounds=bounds,
        pop_size=pop_size,
        device=device,
        dtype=dtype,
        seed=basin_seed,
        sigma0=0.25,
        mirrored=True,
    )
    cma.sigma_min = sigma_min
    cma.sigma_max = sigma_max
    cma._normalize_on_decomp = False
    cma._gen = basin_gen
    cma._gen_device = opt._gen_device

    for _restart_idx in range(n_restarts):
        restart_remaining = budget_limit - opt._fe_count
        if restart_remaining < pop_size * 3:
            break

        restarts_left = n_restarts - _restart_idx
        restart_budget = min(budget_limit, opt._fe_count + restart_remaining // restarts_left)

        # Random starting point via the isolated generator
        mean_t = (
            torch.rand(dim, device=opt._gen_device, dtype=dtype, generator=basin_gen) * search_span
            + opt.lb[0].item()
        )
        if opt._gen_device != device:
            mean_t = mean_t.to(device)

        cma.restart(new_pop_size=pop_size, mean=mean_t, sigma=sigma_init)
        cma.sigma_min = sigma_min
        cma.sigma_max = sigma_max

        generation = 0
        restart_best = float("inf")
        restart_best_gen = 0

        while opt._fe_count + pop_size <= restart_budget:
            candidates = cma.ask()

            # Evaluate each candidate individually (respecting budget). Objective
            # errors propagate; callers should return inf/nan for invalid points.
            fit = torch.full((pop_size,), float("inf"), device=device, dtype=dtype)
            for i in range(pop_size):
                if opt._fe_count >= restart_budget:
                    break
                val_t = fitness_fn(candidates[i].unsqueeze(0)).squeeze()
                fit[i] = val_t
                opt._fe_count += 1

            if not torch.isfinite(fit).any():
                cma.sigma = max(sigma_min, cma.sigma * 0.5)
                continue

            cma.tell(candidates, fit)
            generation += 1

            best_candidate = float(fit.min().item())
            if best_candidate + _EPS < restart_best:
                restart_best = best_candidate
                restart_best_gen = generation
            if best_candidate < best_fitness_val:
                best_fitness_val = best_candidate
                best_idx = fit.argmin()
                opt.best_fitness = fit[best_idx].clone()
                opt.best_solution = candidates[best_idx].clone()

            if generation - restart_best_gen >= opt._basin_explore_stagnation:
                break
