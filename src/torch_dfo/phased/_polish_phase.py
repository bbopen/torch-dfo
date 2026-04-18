"""Polish-phase helpers for :class:`PhasedDFO`.

This module contains the PhasedDFO-side polish-phase orchestration
extracted from :class:`torch_dfo.phased.orchestrator.PhasedDFO`. The
functions here take the orchestrator instance as the first argument
(``opt``) rather than being methods on the class; this keeps the module
a leaf in the import graph and avoids circular imports with
``orchestrator.py``.

The public entry points are:

* :func:`run_polish` — polish-phase orchestrator (directional + coordinate
  + scipy precision chain).  Wraps the actual polish algorithms from
  :mod:`torch_dfo._polish` with PhasedDFO-side bookkeeping (elite data
  construction, budget accounting, best-tracking updates).
* :func:`build_polish_directions` — builds the search direction set for
  the directional basin search (CMA eigenvectors, basis pairs, elite PCA,
  elite-to-point, and random directions).

Each is a logic-preserving move of the corresponding ``_run_polish`` /
``_build_polish_directions`` method from ``PhasedDFO``, with ``self``
renamed to ``opt``.  No algorithmic behavior is changed.

The actual polish algorithms (``coordinate_basin_search``,
``directional_basin_search``, ``fd_bfgs_polish``, ``nm_polish``,
``smoothed_envelope_search``) live in :mod:`torch_dfo._polish`; this
module only hosts the PhasedDFO wrapper logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from torch_dfo._polish import (
    coordinate_basin_search,
    directional_basin_search,
    fd_bfgs_polish,
    nm_polish,
    smoothed_envelope_search,
)

if TYPE_CHECKING:
    from torch_dfo.phased.orchestrator import PhasedDFO


def run_polish(
    opt: PhasedDFO,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Execute polish phase: directional + coordinate + scipy precision chain.

    Low-dim pipeline (dim < 20):
      1. Coordinate basin search
      2. scipy L-BFGS-B  (adaptive fraction of remaining)
      3. scipy Powell    (adaptive fraction of remaining-after-lbfgsb)
      4. FD-BFGS         (adaptive fraction of remaining-after-powell)
      5. scipy Nelder-Mead (all remaining)

    High-dim pipeline (dim >= 20):
      0. Smoothed envelope search
      1. Directional basin search
      2. Coordinate basin search
      3. FD-BFGS polish
      4. scipy L-BFGS-B  (adaptive fraction of remaining)
      5. scipy Powell    (adaptive fraction of remaining-after-lbfgsb)
      6. FD-BFGS         (adaptive fraction of remaining-after-powell)
      7. scipy Nelder-Mead (all remaining)

    The polisher fractions adapt to the remaining budget via
    _compute_polish_fractions().  At generous budgets (remaining > 500),
    fractions match the old constants (0.40/0.50/0.50).  At tight
    budgets, L-BFGS-B gets up to 80% to concentrate effort on the
    strongest polisher.
    """
    # Local import to avoid circular dependency at module load time.
    from torch_dfo.phased.orchestrator import (
        _compute_directional_coarse_points,
        _compute_directional_priority_hops,
        _compute_polish_fractions,
    )

    remaining = opt._budget - opt._fe_count
    if remaining <= 0:
        opt._phase = 3
        return

    best_x, best_f = opt.best()

    # Build elite data for polish.
    # Prefer accumulated search pool from CMA-ES phases.
    elite_centroid = None
    elite_median = None
    elite_points = None

    if (
        opt._search_population is not None
        and opt._search_population.numel() > 0
        and opt._search_population_fitness is not None
    ):
        sp = opt._search_population
        sp_fit = opt._search_population_fitness
        top_k = min(20, sp.shape[0])
        top_idx = sp_fit.argsort()[:top_k]
        top_elite = sp[top_idx]
        elite_centroid = top_elite.mean(dim=0)
        elite_median = top_elite.median(dim=0).values
        elite_points = top_elite
    elif len(opt._elite_solutions) > 0:
        elite_stack = torch.stack(opt._elite_solutions[-100:])
        elite_f_stack = torch.stack(opt._elite_fitness[-100:])
        top_k = min(20, elite_stack.shape[0])
        top_idx = elite_f_stack.argsort()[:top_k]
        top_elite = elite_stack[top_idx]
        elite_centroid = top_elite.mean(dim=0)
        elite_median = top_elite.median(dim=0).values
        elite_points = top_elite

    # 0. Smoothed envelope search (high-dim multimodal escape)
    if opt._high_dim:
        envelope_budget = min(opt._budget - opt._fe_count, remaining // 4)
        if envelope_budget > 100:
            best_x, best_f, fe = smoothed_envelope_search(
                best_x,
                best_f,
                fitness_fn,
                opt.lb,
                opt.ub,
                budget=envelope_budget,
                min_remaining=opt._envelope_min_remaining,
                proposal_budget_cap=opt._envelope_proposal_cap,
                min_dim=opt._high_dim_threshold,
            )
            opt._fe_count += fe
            if best_f < opt.best_fitness:
                opt.best_solution = best_x.clone()
                opt.best_fitness = best_f.clone()

    # 1. Directional basin search (high-dim only)
    if opt._high_dim:
        directional_budget = opt._budget - opt._fe_count
        if directional_budget > 0:
            directions, priority_count = build_polish_directions(
                opt,
                best_x,
                directional_budget,
            )
            if directions is not None and directions.shape[0] > 0:
                coarse_pts = _compute_directional_coarse_points(
                    directional_budget,
                )
                priority_hp = _compute_directional_priority_hops(
                    directional_budget,
                )
                best_x, best_f, fe = directional_basin_search(
                    best_x,
                    best_f,
                    fitness_fn,
                    opt.lb,
                    opt.ub,
                    directions=directions,
                    coarse_points=coarse_pts,
                    refinement_stages=opt._config.directional_refinement_stages,
                    refinement_points=opt._config.directional_refinement_points,
                    window_shrink=opt._config.directional_window_shrink,
                    budget=directional_budget,
                    priority_count=priority_count,
                    elite_points=elite_points,
                    priority_hops=priority_hp,
                    priority_hop_scale=opt._config.directional_priority_hop_scale,
                )
                opt._fe_count += fe
                if best_f < opt.best_fitness:
                    opt.best_solution = best_x.clone()
                    opt.best_fitness = best_f.clone()

    # 2. Coordinate basin search
    remaining_coord = opt._budget - opt._fe_count
    if remaining_coord > 0:
        best_x, best_f, fe = coordinate_basin_search(
            best_x,
            best_f,
            fitness_fn,
            opt.lb,
            opt.ub,
            elite_centroid=elite_centroid,
            elite_median=elite_median,
            passes=2,
            coarse_points=opt._coordinate_coarse_points,
            refinement_stages=opt._config.coordinate_refinement_stages,
            refinement_points=opt._config.coordinate_refinement_points,
            window_shrink=opt._config.coordinate_window_shrink,
            budget=remaining_coord,
        )
        opt._fe_count += fe
        if best_f < opt.best_fitness:
            opt.best_solution = best_x.clone()
            opt.best_fitness = best_f.clone()

    # 3. Scipy precision polisher chain (both low-dim and high-dim)
    # For high-dim, the first FD-BFGS runs before the scipy chain.
    if opt._high_dim:
        remaining_bfgs = opt._budget - opt._fe_count
        if remaining_bfgs > 2 * opt.dim:
            best_x, best_f, fe = fd_bfgs_polish(
                best_x,
                best_f,
                fitness_fn,
                opt.lb,
                opt.ub,
                budget=remaining_bfgs,
            )
            opt._fe_count += fe
            if best_f < opt.best_fitness:
                opt.best_solution = best_x.clone()
                opt.best_fitness = best_f.clone()

    # 4-6. Scipy polisher chain with adaptive budget fractions.
    # At generous budgets (remaining >> 300), fractions match the old
    # constants (0.40/0.50/0.50).  At tight budgets, L-BFGS-B gets a
    # larger share since it's the strongest gradient-based polisher.
    remaining_after_coord = opt._budget - opt._fe_count
    lbfgsb_frac, powell_frac, fdbfgs_frac = _compute_polish_fractions(
        remaining_after_coord,
    )

    # 4. FD-BFGS (adaptive fraction of remaining, replaces scipy L-BFGS-B)
    if remaining_after_coord > 2 * opt.dim:
        lbfgsb_budget = int(remaining_after_coord * lbfgsb_frac)
        best_x, best_f, fe = fd_bfgs_polish(
            best_x,
            best_f,
            fitness_fn,
            opt.lb,
            opt.ub,
            budget=lbfgsb_budget,
        )
        opt._fe_count += fe
        if best_f < opt.best_fitness:
            opt.best_solution = best_x.clone()
            opt.best_fitness = best_f.clone()

    # 5. Coordinate basin search (adaptive fraction, replaces scipy Powell)
    remaining_after_lbfgsb = opt._budget - opt._fe_count
    if remaining_after_lbfgsb > 20:
        powell_budget = int(remaining_after_lbfgsb * powell_frac)
        best_x, best_f, fe = coordinate_basin_search(
            best_x,
            best_f,
            fitness_fn,
            opt.lb,
            opt.ub,
            budget=powell_budget,
        )
        opt._fe_count += fe
        if best_f < opt.best_fitness:
            opt.best_solution = best_x.clone()
            opt.best_fitness = best_f.clone()

    # 6. FD-BFGS (adaptive fraction of remaining-after-powell)
    remaining_after_powell = opt._budget - opt._fe_count
    if remaining_after_powell > 2 * opt.dim:
        bfgs_budget = int(remaining_after_powell * fdbfgs_frac)
        best_x, best_f, fe = fd_bfgs_polish(
            best_x,
            best_f,
            fitness_fn,
            opt.lb,
            opt.ub,
            budget=bfgs_budget,
        )
        opt._fe_count += fe
        if best_f < opt.best_fitness:
            opt.best_solution = best_x.clone()
            opt.best_fitness = best_f.clone()

    # 7. Nelder-Mead polish (all remaining, replaces scipy Nelder-Mead)
    remaining_final = opt._budget - opt._fe_count
    if remaining_final > 20:
        best_x, best_f, fe = nm_polish(
            best_x,
            fitness_fn,
            budget=remaining_final,
            bounds=(float(opt.lb[0]), float(opt.ub[0])),
        )
        opt._fe_count += fe
        if best_f < opt.best_fitness:
            opt.best_solution = best_x.clone()
            opt.best_fitness = best_f.clone()

    opt._phase = 3


def build_polish_directions(
    opt: PhasedDFO,
    best_x: torch.Tensor,
    directional_budget: int,
) -> tuple[torch.Tensor | None, int]:
    """Build search directions for directional basin search.

    Adds basis pair combinations and elite-to-point directions.
    Returns (directions_tensor, priority_count) where priority directions
    are CMA eigenvectors + basis pairs.
    """
    # Local import to avoid circular dependency at module load time.
    from torch_dfo.phased.orchestrator import (
        _compute_directional_basis_pair_basis_count,
    )

    priority_directions: list[torch.Tensor] = []
    secondary_directions: list[torch.Tensor] = []

    # CMA-ES basis vectors (eigenvectors of covariance)
    cma_basis_subset: list[torch.Tensor] = []
    if opt._cmaes is not None:
        B = opt._cmaes.B
        # Take top eigenvectors (those with largest eigenvalues)
        n_cma = min(opt._pca_directions, opt.dim)
        for i in range(n_cma):
            col_idx = opt.dim - 1 - i  # descending eigenvalue order
            d = B[:, col_idx].clone()
            norm = d.norm()
            if norm > 1e-30:
                d_normalized = d / norm
                priority_directions.append(d_normalized)
                cma_basis_subset.append(d_normalized)

        # Basis pair combinations (sum/diff of top CMA vectors).
        combo_limit = opt._basis_pair_limit
        basis_pair_bc = _compute_directional_basis_pair_basis_count(
            directional_budget,
        )
        combo_basis_count = min(basis_pair_bc, len(cma_basis_subset))
        combo_count = 0
        for i in range(combo_basis_count):
            for j in range(i + 1, combo_basis_count):
                d_sum = cma_basis_subset[i] + cma_basis_subset[j]
                norm_s = d_sum.norm()
                if norm_s > 1e-30:
                    priority_directions.append(d_sum / norm_s)
                combo_count += 1
                if combo_count >= combo_limit:
                    break
                d_diff = cma_basis_subset[i] - cma_basis_subset[j]
                norm_d = d_diff.norm()
                if norm_d > 1e-30:
                    priority_directions.append(d_diff / norm_d)
                combo_count += 1
                if combo_count >= combo_limit:
                    break
            if combo_count >= combo_limit:
                break

    priority_count = len(priority_directions)

    # PCA of elite solutions (secondary)
    if len(opt._elite_solutions) > opt.dim:
        elite_stack = torch.stack(opt._elite_solutions[-100:])
        elite_f_stack = torch.stack(opt._elite_fitness[-100:])
        top_k = min(50, elite_stack.shape[0])
        top_idx = elite_f_stack.argsort()[:top_k]
        top_elite = elite_stack[top_idx]

        centered = top_elite - top_elite.mean(dim=0, keepdim=True)
        if centered.shape[0] > 1:
            cov = (centered.T @ centered) / (centered.shape[0] - 1)
            cov = (cov + cov.T) / 2  # symmetry
            try:
                if opt.device.type not in ("cpu", "cuda"):
                    _eigvals, eigvecs = torch.linalg.eigh(cov.to("cpu"))
                    eigvecs = eigvecs.to(opt.device)
                else:
                    _eigvals, eigvecs = torch.linalg.eigh(cov)
                n_pca = min(opt._pca_directions, opt.dim)
                for i in range(n_pca):
                    col_idx = opt.dim - 1 - i
                    if col_idx >= 0:
                        d = eigvecs[:, col_idx].clone()
                        norm = d.norm()
                        if norm > 1e-30:
                            secondary_directions.append(d / norm)
            except RuntimeError:
                pass

        # Elite-to-point directions.
        elite_dir_count = min(opt._elite_directions, top_elite.shape[0])
        for member in top_elite[:elite_dir_count]:
            d = member - best_x
            norm = d.norm()
            if norm > 1e-30:
                secondary_directions.append(d / norm)

    # Random directions
    n_random = opt._random_directions
    for _ in range(n_random):
        d = opt._randn(opt.dim)
        norm = d.norm()
        if norm > 1e-30:
            secondary_directions.append(d / norm)

    all_directions = priority_directions + secondary_directions
    if not all_directions:
        return None, 0

    return torch.stack(all_directions), priority_count
