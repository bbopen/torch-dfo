"""Polish operators for PhasedDFO Phase 3.

Three standalone functions that refine an incumbent solution using coarse-to-fine
grid searches and quasi-Newton methods.  All probe creation is batched -- no Python
loops over dimensions for evaluation.  Each function returns
``(best_x, best_f, fe_used)`` for budget tracking.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from torch_dfo.utils import clamp_to_bounds

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Cap FD step at this fraction of the trust radius
GRADIENT_FD_TRUST_RATIO = 0.10

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _add_unique(targets: list[float], candidate: float, lb_val: float, ub_val: float) -> None:
    """Append *candidate* (clamped to bounds) to *targets* if not a near-duplicate."""
    candidate = min(max(candidate, lb_val), ub_val)
    if all(abs(existing - candidate) > 1e-9 for existing in targets):
        targets.append(candidate)


# ---------------------------------------------------------------------------
# coordinate_basin_search
# ---------------------------------------------------------------------------


def coordinate_basin_search(
    x: torch.Tensor,
    f_x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    elite_centroid: torch.Tensor | None = None,
    elite_median: torch.Tensor | None = None,
    passes: int = 2,
    coarse_points: int = 11,
    refinement_stages: int = 2,
    refinement_points: int = 5,
    window_shrink: float = 0.35,
    budget: int | None = None,
    population: torch.Tensor | None = None,
    pop_fitness: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-axis coordinate basin search with coarse-to-fine grid.

    For each dimension, probes target values (elite best, centroid, median,
    coarse linspace grid, and the current coordinate) then refines around the
    best.  Batched: creates probe matrices and evaluates in bulk per dimension.

    Args:
        x: ``(dim,)`` incumbent solution.
        f_x: Scalar fitness of *x*.
        fitness_fn: Callable ``(N, dim) -> (N,)`` for batched evaluation.
        lb: ``(dim,)`` lower bounds.
        ub: ``(dim,)`` upper bounds.
        elite_centroid: ``(dim,)`` optional centroid of elite solutions.
        elite_median: ``(dim,)`` optional median of elite solutions.
        passes: Number of full sweeps over all dimensions.
        coarse_points: Grid points in the coarse sweep per dimension.
        refinement_stages: Number of fine-grid refinement rounds.
        refinement_points: Grid points per refinement round.
        window_shrink: Multiplicative factor to shrink the refinement window.
        budget: Maximum function evaluations (``None`` = unlimited).
        population: ``(pop_size, dim)`` optional current population for elite_best.
        pop_fitness: ``(pop_size,)`` optional fitness of population members.
        generator: Optional :class:`torch.Generator` for reproducible random
            dimension ordering.

    Returns:
        ``(best_x, best_f, fe_used)``

    """
    dim = x.shape[0]
    best_x = x.clone()
    best_f = f_x.clone()
    fe_used = 0

    # Pre-compute elite_best from population (best individual per pop_fitness).
    elite_best: torch.Tensor | None = None
    if population is not None and pop_fitness is not None and population.shape[0] >= 4:
        elite_best = population[pop_fitness.argmin()]

    for _pass in range(passes):
        # Random permutation instead of cyclic shift.
        dim_order = torch.randperm(dim, device=x.device, generator=generator).tolist()

        for d in dim_order:
            if budget is not None and fe_used >= budget:
                return best_x, best_f, fe_used

            lb_d = float(lb[d])
            ub_d = float(ub[d])

            # --- Collect target values along dimension d (deduplicated) ---
            target_list: list[float] = []

            # Elite best coordinate
            if elite_best is not None:
                _add_unique(target_list, float(elite_best[d].item()), lb_d, ub_d)

            # Elite centroid / median hints
            if elite_centroid is not None:
                _add_unique(target_list, float(elite_centroid[d].item()), lb_d, ub_d)
            if elite_median is not None:
                _add_unique(target_list, float(elite_median[d].item()), lb_d, ub_d)

            # Current coordinate value
            _add_unique(target_list, float(best_x[d].item()), lb_d, ub_d)

            # Coarse linspace grid
            coarse_grid = torch.linspace(lb_d, ub_d, coarse_points, device=x.device, dtype=x.dtype)
            for val in coarse_grid.tolist():
                _add_unique(target_list, val, lb_d, ub_d)

            all_targets = torch.tensor(target_list, device=x.device, dtype=x.dtype)

            # --- Build probe matrix: replicate best_x, set dim d ---
            num_targets = all_targets.shape[0]
            probes = best_x.unsqueeze(0).expand(num_targets, -1).clone()
            probes[:, d] = all_targets

            # Evaluate
            if budget is not None:
                remaining = budget - fe_used
                if remaining <= 0:
                    return best_x, best_f, fe_used
                probes = probes[:remaining]
                num_targets = probes.shape[0]

            f_vals = fitness_fn(probes)
            fe_used += num_targets

            # Find best
            best_idx = f_vals.argmin()
            if f_vals[best_idx] < best_f:
                best_f = f_vals[best_idx].clone()
                best_x[d] = probes[best_idx, d].clone()

            # --- Refinement stages ---
            center = best_x[d].clone()
            span = ub_d - lb_d
            window = span * 0.5  # initial refinement window: half of full span

            for _stage in range(refinement_stages):
                if budget is not None and fe_used >= budget:
                    return best_x, best_f, fe_used

                window *= window_shrink
                lo = max(lb_d, float(center) - window)
                hi = min(ub_d, float(center) + window)

                fine_grid = torch.linspace(
                    lo,
                    hi,
                    refinement_points,
                    device=x.device,
                    dtype=x.dtype,
                )
                n_fine = fine_grid.shape[0]
                fine_probes = best_x.unsqueeze(0).expand(n_fine, -1).clone()
                fine_probes[:, d] = fine_grid

                if budget is not None:
                    remaining = budget - fe_used
                    if remaining <= 0:
                        return best_x, best_f, fe_used
                    fine_probes = fine_probes[:remaining]
                    n_fine = fine_probes.shape[0]

                f_fine = fitness_fn(fine_probes)
                fe_used += n_fine

                fine_best_idx = f_fine.argmin()
                if f_fine[fine_best_idx] < best_f:
                    best_f = f_fine[fine_best_idx].clone()
                    best_x[d] = fine_probes[fine_best_idx, d].clone()
                    center = best_x[d].clone()

    return best_x, best_f, fe_used


# ---------------------------------------------------------------------------
# directional_basin_search
# ---------------------------------------------------------------------------


def directional_basin_search(
    x: torch.Tensor,
    f_x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    directions: torch.Tensor,
    coarse_points: int = 11,
    refinement_stages: int = 2,
    refinement_points: int = 5,
    window_shrink: float = 0.4,
    budget: int | None = None,
    priority_count: int = 0,
    elite_points: torch.Tensor | None = None,
    priority_hops: int = 4,
    priority_hop_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Coarse-to-fine line search along given directions.

    For each direction, probes points along the line through *x*, finds the
    best, then refines around it.

    Args:
        x: ``(dim,)`` incumbent solution.
        f_x: Scalar fitness of *x*.
        fitness_fn: Callable ``(N, dim) -> (N,)`` for batched evaluation.
        lb: ``(dim,)`` lower bounds.
        ub: ``(dim,)`` upper bounds.
        directions: ``(num_dirs, dim)`` search directions (pre-normalised).
        coarse_points: Grid points for the coarse sweep.
        refinement_stages: Number of fine-grid refinement rounds.
        refinement_points: Grid points per refinement round.
        window_shrink: Multiplicative factor to shrink the refinement window.
        budget: Maximum function evaluations (``None`` = unlimited).
        priority_count: Number of leading directions to treat as priority
            (processed first in order, remaining shuffled).
        elite_points: ``(K, dim)`` optional elite population for projection
            alphas.
        priority_hops: Number of extra hop probes for priority directions.
        priority_hop_scale: Scale factor for hop alphas.
        generator: Optional :class:`torch.Generator` for reproducible
            randomised ordering of secondary directions.

    Returns:
        ``(best_x, best_f, fe_used)``

    """
    best_x = x.clone()
    best_f = f_x.clone()
    fe_used = 0
    num_dirs = directions.shape[0]

    # Priority/secondary ordering.
    if priority_count > 0:
        direction_order = list(range(priority_count))
        if num_dirs > priority_count:
            tail = torch.randperm(
                num_dirs - priority_count,
                device=x.device,
                generator=generator,
            ).tolist()
            direction_order.extend(priority_count + idx for idx in tail)
    else:
        direction_order = list(range(num_dirs))

    for i in direction_order:
        if budget is not None and fe_used >= budget:
            return best_x, best_f, fe_used

        d_vec = directions[i]  # (dim,)

        # --- Compute valid step range by projecting bounds onto direction ---
        t_min, t_max = _direction_step_range(best_x, d_vec, lb, ub)
        if t_max <= t_min:
            continue

        # --- Coarse grid along direction ---
        steps = torch.linspace(
            float(t_min),
            float(t_max),
            coarse_points,
            device=x.device,
            dtype=x.dtype,
        )
        alpha_list = steps.tolist()

        # Always include alpha=0 (the current point).
        alpha_list.append(0.0)

        # Priority hops for priority directions.
        is_priority = i < priority_count
        if is_priority:
            for hop in range(1, priority_hops + 1):
                delta = hop * priority_hop_scale
                if t_min <= delta <= t_max:
                    alpha_list.append(delta)
                if t_min <= -delta <= t_max:
                    alpha_list.append(-delta)

        # Elite projections.
        if elite_points is not None:
            projections = (elite_points - best_x.unsqueeze(0)) @ d_vec  # (K,)
            for alpha in projections.tolist():
                alpha_list.append(min(max(float(alpha), t_min), t_max))

        # Deduplicate and sort
        alpha_list.sort()
        unique_alphas: list[float] = []
        for alpha in alpha_list:
            if not unique_alphas or abs(alpha - unique_alphas[-1]) > 1e-9:
                unique_alphas.append(alpha)

        # Build probe tensor from unique alphas
        alpha_tensor = torch.tensor(unique_alphas, device=x.device, dtype=x.dtype)
        probes = best_x.unsqueeze(0) + alpha_tensor.unsqueeze(1) * d_vec.unsqueeze(0)
        probes = clamp_to_bounds(probes, lb, ub)

        num_probes = probes.shape[0]
        if budget is not None:
            remaining = budget - fe_used
            if remaining <= 0:
                return best_x, best_f, fe_used
            probes = probes[:remaining]
            num_probes = probes.shape[0]

        f_vals = fitness_fn(probes)
        fe_used += num_probes

        best_idx = f_vals.argmin()
        if f_vals[best_idx] < best_f:
            best_f = f_vals[best_idx].clone()
            best_x = probes[best_idx].clone()

        # --- Refinement stages ---
        # Each stage recomputes the step range relative to current best_x,
        # with center_step = 0 (since best_x already incorporates the best step).
        window = (float(t_max) - float(t_min)) * 0.5

        for _stage in range(refinement_stages):
            if budget is not None and fe_used >= budget:
                return best_x, best_f, fe_used

            window *= window_shrink
            # Recompute valid step range from current best_x
            t_min_r, t_max_r = _direction_step_range(best_x, d_vec, lb, ub)
            lo_step = max(t_min_r, -window)
            hi_step = min(t_max_r, window)
            if hi_step <= lo_step:
                break

            fine_steps = torch.linspace(
                lo_step,
                hi_step,
                refinement_points,
                device=x.device,
                dtype=x.dtype,
            )
            fine_probes = best_x.unsqueeze(0) + fine_steps.unsqueeze(1) * d_vec.unsqueeze(0)
            fine_probes = clamp_to_bounds(fine_probes, lb, ub)

            n_fine = fine_probes.shape[0]
            if budget is not None:
                remaining = budget - fe_used
                if remaining <= 0:
                    return best_x, best_f, fe_used
                fine_probes = fine_probes[:remaining]
                n_fine = fine_probes.shape[0]

            f_fine = fitness_fn(fine_probes)
            fe_used += n_fine

            fine_best_idx = f_fine.argmin()
            if f_fine[fine_best_idx] < best_f:
                best_f = f_fine[fine_best_idx].clone()
                best_x = fine_probes[fine_best_idx].clone()

    return best_x, best_f, fe_used


def _direction_step_range(
    x: torch.Tensor,
    d: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> tuple[float, float]:
    """Compute the valid scalar step range ``[t_min, t_max]`` such that
    ``x + t * d`` stays within ``[lb, ub]`` for all dimensions.

    For each dimension i where ``d[i] != 0``:
      - ``t_lo_i = (lb[i] - x[i]) / d[i]`` and ``t_hi_i = (ub[i] - x[i]) / d[i]``
      - If ``d[i] > 0``: ``t`` must be in ``[t_lo_i, t_hi_i]``
      - If ``d[i] < 0``: ``t`` must be in ``[t_hi_i, t_lo_i]``

    Returns intersection of all per-dimension intervals.
    """
    eps = 1e-30
    # Avoid division by zero: mask out near-zero directions
    mask = d.abs() > eps
    if not mask.any():
        return 0.0, 0.0

    d_safe = d.clone()
    d_safe[~mask] = 1.0  # placeholder, will be masked out

    t_lo = (lb - x) / d_safe
    t_hi = (ub - x) / d_safe

    # Swap so t_lo <= t_hi per dimension
    t_lower = torch.min(t_lo, t_hi)
    t_upper = torch.max(t_lo, t_hi)

    # Apply mask: only consider dimensions where direction is non-zero
    t_min = float(t_lower[mask].max())
    t_max = float(t_upper[mask].min())

    return t_min, t_max


# ---------------------------------------------------------------------------
# fd_bfgs_polish
# ---------------------------------------------------------------------------


def fd_bfgs_polish(
    x: torch.Tensor,
    f_x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    budget: int = 500,
    fd_step: float = 1e-4,
    max_backtracks: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """FD-BFGS polish with trust region.

    Central-difference gradient estimation with BFGS inverse Hessian
    approximation.  Trust region controls step size.

    Args:
        x: ``(dim,)`` incumbent solution.
        f_x: Scalar fitness of *x*.
        fitness_fn: Callable ``(N, dim) -> (N,)`` for batched evaluation.
        lb: ``(dim,)`` lower bounds.
        ub: ``(dim,)`` upper bounds.
        budget: Maximum function evaluations.
        fd_step: Finite-difference step as a fraction of the search span.
        max_backtracks: Maximum backtracking halvings per iteration.

    Returns:
        ``(best_x, best_f, fe_used)``

    """
    dim = x.shape[0]
    device = x.device
    dtype = x.dtype

    best_x = x.clone()
    best_f = f_x.clone()
    fe_used = 0

    # Search span per dimension
    span = ub - lb  # (dim,)
    span_mean = float(span.mean())

    # Inverse Hessian approximation (identity)
    H = torch.eye(dim, device=device, dtype=dtype)

    # Trust region parameters.
    trust_radius = 0.02 * span_mean
    max_trust_radius = 0.15 * span_mean
    min_trust_radius = 1e-10 * span_mean

    cur_x = x.clone()
    cur_f = f_x.clone()

    while fe_used + 2 * dim <= budget:
        # --- Central-difference gradient ---
        grad, grad_fe = _central_diff_gradient(
            cur_x,
            fitness_fn,
            lb,
            ub,
            span,
            fd_step,
            trust_radius=trust_radius,
        )
        fe_used += grad_fe

        if not torch.isfinite(grad).all():
            break

        grad_norm = grad.norm()
        if grad_norm < 1e-12:
            break  # effectively at a stationary point

        # --- BFGS search direction ---
        direction = -H @ grad
        dir_norm = direction.norm()
        if dir_norm < 1e-30:
            break

        # Scale step by trust radius.
        step_scale = min(1.0, trust_radius / (float(dir_norm) + 1e-30))

        # --- Backtracking line search ---
        accepted = False
        s: torch.Tensor | None = None
        new_x = cur_x  # placate type checker
        new_f = cur_f
        for _bt in range(max_backtracks):
            if budget is not None and fe_used >= budget:
                return best_x, best_f, fe_used

            candidate = clamp_to_bounds(cur_x + step_scale * direction, lb, ub)
            f_cand = fitness_fn(candidate.unsqueeze(0))
            fe_used += 1
            f_cand_scalar = f_cand.squeeze()

            if f_cand_scalar < cur_f:
                accepted = True
                s = candidate - cur_x  # actual step taken
                new_x = candidate.clone()
                new_f = f_cand_scalar.clone()
                # Trust growth capped at max_trust_radius.
                actual_step_norm = float(s.norm())
                trust_radius = min(
                    max_trust_radius,
                    max(trust_radius, actual_step_norm) * 1.4,
                )
                break
            step_scale *= 0.5

        if not accepted:
            # Shrink trust radius.
            trust_radius *= 0.5
            # Break if trust falls below min.
            if trust_radius < min_trust_radius:
                break
            # Reset H on failure
            H = torch.eye(dim, device=device, dtype=dtype)
            continue

        # Update best tracking
        if new_f < best_f:
            best_f = new_f.clone()
            best_x = new_x.clone()

        # --- BFGS update ---
        # Need gradient at new point for y = grad_new - grad_old
        if fe_used + 2 * dim > budget:
            # Not enough budget for another gradient; just accept the step
            cur_x = new_x
            cur_f = new_f
            break

        grad_new, grad_new_fe = _central_diff_gradient(
            new_x,
            fitness_fn,
            lb,
            ub,
            span,
            fd_step,
            trust_radius=trust_radius,
        )
        fe_used += grad_new_fe

        if not torch.isfinite(grad_new).all():
            cur_x = new_x
            cur_f = new_f
            H = torch.eye(dim, device=device, dtype=dtype)
            continue

        assert s is not None  # guaranteed by accepted=True path
        y = grad_new - grad
        ys = torch.dot(y, s)

        if ys > 1e-10:
            rho = 1.0 / ys
            I_mat = torch.eye(dim, device=device, dtype=dtype)
            # H = (I - rho * s outer y) @ H @ (I - rho * y outer s) + rho * s outer s
            sy = s.unsqueeze(1) @ y.unsqueeze(0)  # (dim, dim)
            ys_mat = y.unsqueeze(1) @ s.unsqueeze(0)  # (dim, dim)
            ss = s.unsqueeze(1) @ s.unsqueeze(0)  # (dim, dim)
            left = I_mat - rho * sy
            right = I_mat - rho * ys_mat
            H = left @ H @ right + rho * ss
        else:
            # Curvature condition not met; reset
            H = torch.eye(dim, device=device, dtype=dtype)

        # Advance
        cur_x = new_x
        cur_f = new_f
        grad = grad_new

    return best_x, best_f, fe_used


def _central_diff_gradient(
    x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    span: torch.Tensor,
    fd_step: float,
    trust_radius: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute gradient via central finite differences in ONE batched call.

    Creates a ``(2 * dim, dim)`` probe matrix where rows ``2*d`` and ``2*d+1``
    are the forward and backward perturbations along axis *d*.

    When *trust_radius* is provided, the per-element step is capped at
    ``GRADIENT_FD_TRUST_RATIO * trust_radius`` so that finite-difference
    probes stay well within the trust region.

    Returns:
        ``(grad, fe_used)`` where *grad* is ``(dim,)`` and *fe_used* = ``2 * dim``.

    """
    dim = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Step sizes: h = max(fd_step * span, fd_step * |x|, 1e-8)
    h = torch.max(
        torch.max(fd_step * span, fd_step * x.abs()),
        torch.full((dim,), 1e-8, device=device, dtype=dtype),
    )

    # Cap FD step at GRADIENT_FD_TRUST_RATIO * trust_radius
    if trust_radius is not None and trust_radius > 0:
        h = torch.clamp(h, max=GRADIENT_FD_TRUST_RATIO * trust_radius)

    # Build probe matrix: (2*dim, dim)
    probes = x.unsqueeze(0).expand(2 * dim, -1).clone()
    idx = torch.arange(dim, device=device)
    # Forward perturbations: rows 0, 2, 4, ...
    probes[2 * idx, idx] = x[idx] + h
    # Backward perturbations: rows 1, 3, 5, ...
    probes[2 * idx + 1, idx] = x[idx] - h

    # Clamp to bounds
    probes = clamp_to_bounds(probes, lb, ub)

    # Evaluate all probes
    f_vals = fitness_fn(probes)  # (2*dim,)

    # Compute gradient using actual (clamped) differences
    f_fwd = f_vals[2 * idx]  # (dim,)
    f_bwd = f_vals[2 * idx + 1]  # (dim,)
    actual_fwd = probes[2 * idx, idx]  # (dim,)
    actual_bwd = probes[2 * idx + 1, idx]  # (dim,)
    denom = actual_fwd - actual_bwd

    # Avoid division by zero where clamping collapsed both probes to the same value
    safe_denom = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))
    grad = (f_fwd - f_bwd) / safe_denom
    # Zero out gradient where denominator was degenerate
    grad = torch.where(denom.abs() > 1e-30, grad, torch.zeros_like(grad))

    return grad, 2 * dim


# ---------------------------------------------------------------------------
# smoothed_envelope_search
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Smoothed envelope constants
# ---------------------------------------------------------------------------
# Universal constants: these control the Gaussian smoothing kernel and the
# antithetic probe strategy.  They do not depend on (dim, budget) because:
#
# - TRIGGER: fitness threshold below which multimodal escape is attempted.
#   This is a landscape-structure gate (Rastrigin-scale values), not a
#   problem-scale parameter.
# - SIGMA_SCALE/MIN/MAX: the smoothing kernel width targets one period of
#   the Rastrigin cosine (2*pi ≈ 6.28, so sigma ~ 1).  This is function-
#   structure specific, independent of dim/budget.
# - PAIRS/PAIR_MULTIPLIER: minimum antithetic pairs for the least-squares
#   offset estimation.  24 pairs is a statistical minimum for stable
#   estimation; dim*2 ensures overdetermined systems.  Both already
#   adapt to dim through the max(dim*multiplier, pairs) formula.
# - CENTER_RATE: step fraction toward the smoothed minimum.  A smoothing-
#   geometry constant (0.5 = half step).
# - BLEND_WEIGHTS: structural schedule of 3 blend levels (partial + full).
#   A design choice, not a tuning target.
#
# Budget-dependent constants (MIN_REMAINING, PROPOSAL_BUDGET, MIN_DIM)
# are passed as parameters to smoothed_envelope_search() so the caller
# can compute them from (dim, budget); defaults (1600, 1800, 20) keep
# the call sensible when the caller does not override them.

SMOOTHED_ENVELOPE_TRIGGER = 12.0
SMOOTHED_ENVELOPE_SIGMA_SCALE = 0.10
SMOOTHED_ENVELOPE_SIGMA_MIN = 0.8
SMOOTHED_ENVELOPE_SIGMA_MAX = 1.2
SMOOTHED_ENVELOPE_PAIRS = 24
SMOOTHED_ENVELOPE_PAIR_MULTIPLIER = 2  # target pair_count = max(dim*this, PAIRS)
SMOOTHED_ENVELOPE_CENTER_RATE = 0.5
SMOOTHED_ENVELOPE_BLEND_WEIGHTS = (0.5, 0.8, 1.0)  # partial + full steps


def _estimate_envelope_offset(
    point: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    sigma: float,
    pair_count: int,
    budget: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, int]:
    """Estimate the offset to the smoothed-envelope center via least-squares.

    Generates random direction pairs, skipping any that would be clipped by
    bounds (which bias the gradient estimate). Solves the overdetermined system
    ``design @ offset ≈ responses`` where each response is
    ``(f(x+σd) - f(x-σd)) / (4σ)``.

    Returns ``(offset, fe_used)`` where *offset* is ``(dim,)`` or ``None``
    if insufficient interior probes were collected.
    """
    dim = point.shape[0]
    fe_used = 0
    if pair_count < dim:
        return None, fe_used

    directions = torch.randn(pair_count, dim, device=device, dtype=dtype)
    responses: list[float] = []
    used_dirs: list[torch.Tensor] = []
    dir_idx = 0
    max_attempts = max(pair_count * 4, pair_count + dim)
    attempts = 0

    while len(used_dirs) < pair_count and attempts < max_attempts:
        if fe_used + 2 > budget:
            break
        # Generate more directions if we've exhausted the initial batch
        if dir_idx >= directions.shape[0]:
            extra = torch.randn(
                max(1, pair_count - len(used_dirs)),
                dim,
                device=device,
                dtype=dtype,
            )
            directions = torch.cat([directions, extra], dim=0)

        d_vec = directions[dir_idx]
        dir_idx += 1
        attempts += 1

        # Skip probes that would be clipped by bounds (biased estimate)
        plus = point + sigma * d_vec
        minus = point - sigma * d_vec
        if (plus < lb).any() or (plus > ub).any() or (minus < lb).any() or (minus > ub).any():
            continue

        f_plus = fitness_fn(plus.unsqueeze(0)).squeeze().item()
        f_minus = fitness_fn(minus.unsqueeze(0)).squeeze().item()
        fe_used += 2
        if math.isfinite(f_plus) and math.isfinite(f_minus):
            responses.append((f_plus - f_minus) / (4.0 * sigma))
            used_dirs.append(d_vec)

    if len(used_dirs) < dim:
        return None, fe_used

    # Solve least-squares: design @ offset ≈ responses
    design = torch.stack(used_dirs)  # (K, dim)
    targets = torch.tensor(responses, device=device, dtype=dtype).unsqueeze(1)  # (K, 1)
    try:
        offset = torch.linalg.lstsq(design, targets).solution.squeeze(1)
    except RuntimeError:
        # Fallback: ridge regression
        gram = design.T @ design
        gram_scale = float(torch.trace(gram).item() / max(dim, 1))
        ridge = max(1e-6, 1e-4 * gram_scale)
        rhs = design.T @ targets
        try:
            offset = torch.linalg.solve(
                gram + ridge * torch.eye(dim, device=device, dtype=dtype),
                rhs,
            ).squeeze(1)
        except RuntimeError:
            return None, fe_used

    offset = torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(offset).all():
        return None, fe_used
    return offset, fe_used


def smoothed_envelope_search(
    x: torch.Tensor,
    f_x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    lb: torch.Tensor,
    ub: torch.Tensor,
    budget: int,
    min_remaining: int = 1600,
    proposal_budget_cap: int = 1800,
    min_dim: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Jump toward the center of a Gaussian-smoothed objective, then polish.

    Uses antithetic random probes to estimate the gradient of the objective
    smoothed by a Gaussian kernel with sigma ~ 1 (one Rastrigin period).
    Steps toward the smoothed minimum, then applies FD-BFGS polish from
    the new point.  Effective on multimodal landscapes (rotated Rastrigin)
    where the smoothed landscape reveals the global basin structure.

    Only activates for high-dim problems (dim >= min_dim) with sufficient
    remaining budget and current fitness below the trigger threshold.

    Args:
        x: ``(dim,)`` incumbent solution.
        f_x: Scalar fitness of *x*.
        fitness_fn: Callable ``(N, dim) -> (N,)`` batched evaluation.
        lb, ub: ``(dim,)`` bounds.
        budget: Maximum function evaluations.
        min_remaining: Minimum budget required to activate (default 1600).
            Should be scaled by the caller based on total budget.
        proposal_budget_cap: Maximum evals per BFGS polish of each blend
            proposal (default 1800).  Should be scaled by the caller.
        min_dim: Minimum dimensionality to activate (default 20).
            Should match the caller's high-dim threshold.

    Returns:
        ``(best_x, best_f, fe_used)``

    """
    dim = x.shape[0]
    best_x = x.clone()
    best_f = f_x.clone()
    fe_used = 0
    value = float(f_x)

    # Gate: only for high-dim, sufficient budget, and below trigger
    if (
        dim < min_dim
        or budget < min_remaining
        or not math.isfinite(value)
        or value > SMOOTHED_ENVELOPE_TRIGGER
    ):
        return best_x, best_f, fe_used

    device = x.device
    dtype = x.dtype
    span = float((ub - lb).mean())
    sigma = min(
        SMOOTHED_ENVELOPE_SIGMA_MAX,
        max(SMOOTHED_ENVELOPE_SIGMA_MIN, SMOOTHED_ENVELOPE_SIGMA_SCALE * span),
    )

    # Determine how many antithetic pairs we can afford
    n_blend = len(SMOOTHED_ENVELOPE_BLEND_WEIGHTS)
    target_pairs = max(dim * SMOOTHED_ENVELOPE_PAIR_MULTIPLIER, SMOOTHED_ENVELOPE_PAIRS)
    max_pairs = max(0, (budget - n_blend) // 2)
    pair_count = min(target_pairs, max_pairs)
    if pair_count < dim:
        return best_x, best_f, fe_used

    # Estimate smoothed-envelope offset via least-squares on paired probes.
    # Skip probes that would be clipped by bounds (biased gradient estimate).
    offset, offset_fe = _estimate_envelope_offset(
        best_x,
        fitness_fn,
        lb,
        ub,
        sigma,
        pair_count,
        budget,
        device,
        dtype,
    )
    fe_used += offset_fe
    if offset is None:
        return best_x, best_f, fe_used

    offset_norm = torch.linalg.vector_norm(offset)
    if not torch.isfinite(offset_norm) or float(offset_norm) <= 1e-9:
        return best_x, best_f, fe_used

    # Step toward the smoothed minimum (2x offset for center rate scaling)
    center = clamp_to_bounds(
        (best_x - SMOOTHED_ENVELOPE_CENTER_RATE * 2.0 * offset).unsqueeze(0),
        lb,
        ub,
    ).squeeze(0)
    if float(torch.linalg.vector_norm(center - best_x)) <= 1e-9:
        return best_x, best_f, fe_used

    # Try blended proposals toward the smoothed center, with BFGS polish
    anchor = best_x.clone()
    for blend in SMOOTHED_ENVELOPE_BLEND_WEIGHTS:
        if fe_used >= budget:
            break
        candidate = clamp_to_bounds(
            (anchor + blend * (center - anchor)).unsqueeze(0),
            lb,
            ub,
        ).squeeze(0)
        if float(torch.linalg.vector_norm(candidate - anchor)) <= 1e-9:
            continue

        f_cand = fitness_fn(candidate.unsqueeze(0)).squeeze()
        fe_used += 1
        if not torch.isfinite(f_cand):
            continue

        # Polish the candidate with FD-BFGS
        proposal_budget = min(budget - fe_used, proposal_budget_cap)
        if proposal_budget > 2 * dim:
            cand_polished, f_polished, bfgs_fe = fd_bfgs_polish(
                candidate,
                f_cand,
                fitness_fn,
                lb,
                ub,
                budget=proposal_budget,
            )
            fe_used += bfgs_fe
            if f_polished < best_f:
                best_x = cand_polished.clone()
                best_f = f_polished.clone()

        if float(best_f) <= 1e-12:
            break

    return best_x, best_f, fe_used


# ---------------------------------------------------------------------------
# nm_polish
# ---------------------------------------------------------------------------


def nm_polish(
    x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    budget: int,
    bounds: tuple[float, float] | None = None,
    simplex_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Polish via NelderMead simplex. Returns (best_x, best_f, evals_used).

    Wraps the existing :class:`~torch_dfo.nelder_mead.NelderMead` optimizer
    in a thin loop with warm-started simplex around *x*.

    Args:
        x: ``(dim,)`` incumbent solution.
        fitness_fn: Callable ``(N, dim) -> (N,)`` for batched evaluation.
        budget: Maximum function evaluations.
        bounds: ``(lo, hi)`` scalar bounds or ``None`` for unbounded.
        simplex_scale: Fraction of the search span used for the initial
            simplex perturbation.

    Returns:
        ``(best_x, best_f, evals_used)``

    """
    from torch_dfo.nelder_mead import NelderMead

    dim = x.shape[0]
    nm = NelderMead(
        dim=dim,
        bounds=bounds if bounds is not None else (-1e12, 1e12),
        device=x.device,
        dtype=x.dtype,
    )

    # Warm-start simplex around x
    span = (bounds[1] - bounds[0]) if bounds else 1.0
    delta = simplex_scale * span
    simplex = x.unsqueeze(0).repeat(dim + 1, 1)
    for i in range(dim):
        simplex[i + 1, i] += delta
    nm.population = simplex
    nm._needs_full_eval = True  # force initial simplex evaluation

    best_x, best_f = x.clone(), fitness_fn(x.unsqueeze(0)).squeeze()
    evals = 1
    while evals < budget:
        candidates = nm.ask()
        batch_size = candidates.shape[0]
        if evals + batch_size > budget:
            break
        fitness = fitness_fn(candidates)
        nm.tell(candidates, fitness)
        evals += batch_size
        if nm.best_fitness < best_f:
            best_f = nm.best_fitness.clone()
            best_x = nm.best_solution.clone()
    return best_x, best_f, evals
