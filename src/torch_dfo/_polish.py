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

GRADIENT_FD_TRUST_RATIO = 0.10


def _add_unique(targets: list[float], candidate: float, lb_val: float, ub_val: float) -> None:
    """Append *candidate* (clamped to bounds) to *targets* if not a near-duplicate."""
    candidate = min(max(candidate, lb_val), ub_val)
    if all(abs(existing - candidate) > 1e-9 for existing in targets):
        targets.append(candidate)


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

    elite_best: torch.Tensor | None = None
    if population is not None and pop_fitness is not None and population.shape[0] >= 4:
        elite_best = population[pop_fitness.argmin()]

    gen_device = generator.device if generator is not None else x.device
    for _pass in range(passes):
        dim_order = torch.randperm(dim, device=gen_device, generator=generator).tolist()

        for d in dim_order:
            if budget is not None and fe_used >= budget:
                return best_x, best_f, fe_used

            lb_d = float(lb[d])
            ub_d = float(ub[d])

            target_list: list[float] = []

            if elite_best is not None:
                _add_unique(target_list, float(elite_best[d].item()), lb_d, ub_d)

            if elite_centroid is not None:
                _add_unique(target_list, float(elite_centroid[d].item()), lb_d, ub_d)
            if elite_median is not None:
                _add_unique(target_list, float(elite_median[d].item()), lb_d, ub_d)

            _add_unique(target_list, float(best_x[d].item()), lb_d, ub_d)

            coarse_grid = torch.linspace(lb_d, ub_d, coarse_points, device=x.device, dtype=x.dtype)
            for val in coarse_grid.tolist():
                _add_unique(target_list, val, lb_d, ub_d)

            all_targets = torch.tensor(target_list, device=x.device, dtype=x.dtype)

            num_targets = all_targets.shape[0]
            probes = best_x.unsqueeze(0).expand(num_targets, -1).clone()
            probes[:, d] = all_targets

            if budget is not None:
                remaining = budget - fe_used
                if remaining <= 0:
                    return best_x, best_f, fe_used
                probes = probes[:remaining]
                num_targets = probes.shape[0]

            f_vals = fitness_fn(probes)
            fe_used += num_targets

            best_idx = f_vals.argmin()
            if f_vals[best_idx] < best_f:
                best_f = f_vals[best_idx].clone()
                best_x[d] = probes[best_idx, d].clone()

            center = best_x[d].clone()
            span = ub_d - lb_d
            window = span * 0.5

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

    gen_device = generator.device if generator is not None else x.device
    if priority_count > 0:
        direction_order = list(range(priority_count))
        if num_dirs > priority_count:
            tail = torch.randperm(
                num_dirs - priority_count,
                device=gen_device,
                generator=generator,
            ).tolist()
            direction_order.extend(priority_count + idx for idx in tail)
    else:
        direction_order = list(range(num_dirs))

    for i in direction_order:
        if budget is not None and fe_used >= budget:
            return best_x, best_f, fe_used

        d_vec = directions[i]

        t_min, t_max = _direction_step_range(best_x, d_vec, lb, ub)
        if t_max <= t_min:
            continue

        steps = torch.linspace(
            float(t_min),
            float(t_max),
            coarse_points,
            device=x.device,
            dtype=x.dtype,
        )
        alpha_list = steps.tolist()

        alpha_list.append(0.0)

        is_priority = i < priority_count
        if is_priority:
            for hop in range(1, priority_hops + 1):
                delta = hop * priority_hop_scale
                if t_min <= delta <= t_max:
                    alpha_list.append(delta)
                if t_min <= -delta <= t_max:
                    alpha_list.append(-delta)

        if elite_points is not None:
            projections = (elite_points - best_x.unsqueeze(0)) @ d_vec
            for alpha in projections.tolist():
                alpha_list.append(min(max(float(alpha), t_min), t_max))

        alpha_list.sort()
        unique_alphas: list[float] = []
        for alpha in alpha_list:
            if not unique_alphas or abs(alpha - unique_alphas[-1]) > 1e-9:
                unique_alphas.append(alpha)

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

        window = (float(t_max) - float(t_min)) * 0.5

        for _stage in range(refinement_stages):
            if budget is not None and fe_used >= budget:
                return best_x, best_f, fe_used

            window *= window_shrink
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
    mask = d.abs() > eps
    if not mask.any():
        return 0.0, 0.0

    d_safe = d.clone()
    d_safe[~mask] = 1.0

    t_lo = (lb - x) / d_safe
    t_hi = (ub - x) / d_safe

    t_lower = torch.min(t_lo, t_hi)
    t_upper = torch.max(t_lo, t_hi)

    t_min = float(t_lower[mask].max())
    t_max = float(t_upper[mask].min())

    return t_min, t_max


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

    span = ub - lb
    span_mean = float(span.mean())

    H = torch.eye(dim, device=device, dtype=dtype)

    trust_radius = 0.02 * span_mean
    max_trust_radius = 0.15 * span_mean
    min_trust_radius = 1e-10 * span_mean

    cur_x = x.clone()
    cur_f = f_x.clone()
    grad: torch.Tensor | None = None

    while fe_used < budget:
        if grad is None:
            if fe_used + 2 * dim > budget:
                break
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
            break

        direction = -H @ grad
        dir_norm = direction.norm()
        if dir_norm < 1e-30:
            break

        step_scale = min(1.0, trust_radius / (float(dir_norm) + 1e-30))

        accepted = False
        s: torch.Tensor | None = None
        new_x = cur_x
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
                s = candidate - cur_x
                new_x = candidate.clone()
                new_f = f_cand_scalar.clone()
                actual_step_norm = float(s.norm())
                trust_radius = min(
                    max_trust_radius,
                    max(trust_radius, actual_step_norm) * 1.4,
                )
                break
            step_scale *= 0.5

        if not accepted:
            trust_radius *= 0.5
            if trust_radius < min_trust_radius:
                break
            H = torch.eye(dim, device=device, dtype=dtype)
            grad = None
            continue

        if new_f < best_f:
            best_f = new_f.clone()
            best_x = new_x.clone()

        if fe_used + 2 * dim > budget:
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
            break

        assert s is not None
        y = grad_new - grad
        ys = torch.dot(y, s)

        if ys > 1e-10:
            rho = 1.0 / ys
            I_mat = torch.eye(dim, device=device, dtype=dtype)
            sy = s.unsqueeze(1) @ y.unsqueeze(0)
            ys_mat = y.unsqueeze(1) @ s.unsqueeze(0)
            ss = s.unsqueeze(1) @ s.unsqueeze(0)
            left = I_mat - rho * sy
            right = I_mat - rho * ys_mat
            H = left @ H @ right + rho * ss
        else:
            H = torch.eye(dim, device=device, dtype=dtype)

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

    h = torch.max(
        torch.max(fd_step * span, fd_step * x.abs()),
        torch.full((dim,), 1e-8, device=device, dtype=dtype),
    )

    if trust_radius is not None and trust_radius > 0:
        h = torch.clamp(h, max=GRADIENT_FD_TRUST_RATIO * trust_radius)

    probes = x.unsqueeze(0).expand(2 * dim, -1).clone()
    idx = torch.arange(dim, device=device)
    probes[2 * idx, idx] = x[idx] + h
    probes[2 * idx + 1, idx] = x[idx] - h

    probes = clamp_to_bounds(probes, lb, ub)

    f_vals = fitness_fn(probes)

    f_fwd = f_vals[2 * idx]
    f_bwd = f_vals[2 * idx + 1]
    actual_fwd = probes[2 * idx, idx]
    actual_bwd = probes[2 * idx + 1, idx]
    denom = actual_fwd - actual_bwd

    safe_denom = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))
    grad = (f_fwd - f_bwd) / safe_denom
    grad = torch.where(denom.abs() > 1e-30, grad, torch.zeros_like(grad))

    return grad, 2 * dim


SMOOTHED_ENVELOPE_TRIGGER = 12.0
SMOOTHED_ENVELOPE_SIGMA_SCALE = 0.10
SMOOTHED_ENVELOPE_SIGMA_MIN = 0.8
SMOOTHED_ENVELOPE_SIGMA_MAX = 1.2
SMOOTHED_ENVELOPE_PAIRS = 24
SMOOTHED_ENVELOPE_PAIR_MULTIPLIER = 2
SMOOTHED_ENVELOPE_CENTER_RATE = 0.5
SMOOTHED_ENVELOPE_BLEND_WEIGHTS = (0.5, 0.8, 1.0)


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
    generator: torch.Generator | None = None,
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

    gen_device = generator.device if generator is not None else device

    def _draw_dirs(count: int) -> torch.Tensor:
        d = torch.randn(count, dim, device=gen_device, dtype=dtype, generator=generator)
        return d.to(device) if gen_device != device else d

    directions = _draw_dirs(pair_count)
    responses: list[float] = []
    used_dirs: list[torch.Tensor] = []
    dir_idx = 0
    max_attempts = max(pair_count * 4, pair_count + dim)
    attempts = 0

    while len(used_dirs) < pair_count and attempts < max_attempts:
        max_affordable_pairs = (budget - fe_used) // 2
        if max_affordable_pairs <= 0:
            break
        need = min(pair_count - len(used_dirs), max_affordable_pairs)

        batch_dirs: list[torch.Tensor] = []
        batch_probes: list[torch.Tensor] = []
        while len(batch_dirs) < need and attempts < max_attempts:
            if dir_idx >= directions.shape[0]:
                extra = _draw_dirs(max(1, pair_count - len(used_dirs) - len(batch_dirs)))
                directions = torch.cat([directions, extra], dim=0)

            d_vec = directions[dir_idx]
            dir_idx += 1
            attempts += 1

            plus = point + sigma * d_vec
            minus = point - sigma * d_vec
            if (plus < lb).any() or (plus > ub).any() or (minus < lb).any() or (minus > ub).any():
                continue

            batch_dirs.append(d_vec)
            batch_probes.append(plus)
            batch_probes.append(minus)

        if not batch_dirs:
            break

        f_batch = fitness_fn(torch.stack(batch_probes))
        fe_used += 2 * len(batch_dirs)
        for j, d_vec in enumerate(batch_dirs):
            f_plus = float(f_batch[2 * j].item())
            f_minus = float(f_batch[2 * j + 1].item())
            if math.isfinite(f_plus) and math.isfinite(f_minus):
                responses.append((f_plus - f_minus) / (4.0 * sigma))
                used_dirs.append(d_vec)

    if len(used_dirs) < dim:
        return None, fe_used

    design = torch.stack(used_dirs)
    targets = torch.tensor(responses, device=device, dtype=dtype).unsqueeze(1)
    try:
        offset = torch.linalg.lstsq(design, targets).solution.squeeze(1)
    except RuntimeError:
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
    generator: torch.Generator | None = None,
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

    n_blend = len(SMOOTHED_ENVELOPE_BLEND_WEIGHTS)
    target_pairs = max(dim * SMOOTHED_ENVELOPE_PAIR_MULTIPLIER, SMOOTHED_ENVELOPE_PAIRS)
    max_pairs = max(0, (budget - n_blend) // 2)
    pair_count = min(target_pairs, max_pairs)
    if pair_count < dim:
        return best_x, best_f, fe_used

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
        generator=generator,
    )
    fe_used += offset_fe
    if offset is None:
        return best_x, best_f, fe_used

    offset_norm = torch.linalg.vector_norm(offset)
    if not torch.isfinite(offset_norm) or float(offset_norm) <= 1e-9:
        return best_x, best_f, fe_used

    center = clamp_to_bounds(
        (best_x - SMOOTHED_ENVELOPE_CENTER_RATE * 2.0 * offset).unsqueeze(0),
        lb,
        ub,
    ).squeeze(0)
    if float(torch.linalg.vector_norm(center - best_x)) <= 1e-9:
        return best_x, best_f, fe_used

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
        if f_cand < best_f:
            best_x = candidate.clone()
            best_f = f_cand.clone()

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


def nm_polish(
    x: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    budget: int,
    bounds: tuple[float, float] | tuple[torch.Tensor, torch.Tensor] | None = None,
    simplex_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Polish via NelderMead simplex. Returns (best_x, best_f, evals_used).

    Wraps the existing :class:`~torch_dfo.nelder_mead.NelderMead` optimizer
    in a thin loop with warm-started simplex around *x*.

    Args:
        x: ``(dim,)`` incumbent solution.
        fitness_fn: Callable ``(N, dim) -> (N,)`` for batched evaluation.
        budget: Maximum function evaluations.
        bounds: ``(lo, hi)`` scalar bounds, ``(lb, ub)`` per-dim tensors,
            or ``None`` for unbounded.
        simplex_scale: Fraction of the search span used for the initial
            simplex perturbation.

    Returns:
        ``(best_x, best_f, evals_used)``

    """
    from torch_dfo.nelder_mead import NelderMead

    if budget <= 0:
        return x.clone(), torch.tensor(float("inf"), device=x.device, dtype=x.dtype), 0

    dim = x.shape[0]
    if bounds is not None:
        nm_bounds = bounds
    else:
        finfo = torch.finfo(x.dtype)
        max_abs = float(x.abs().max().item()) if x.numel() > 0 else 0.0
        half_width = min(max(1e12, max_abs * 10.0 + 1e6), finfo.max * 0.25)
        nm_bounds = (x - half_width, x + half_width)
    nm = NelderMead(
        dim=dim,
        bounds=nm_bounds,
        device=x.device,
        dtype=x.dtype,
    )

    x_clamped = (
        clamp_to_bounds(x.unsqueeze(0), nm.lb, nm.ub).squeeze(0)
        if bounds is not None
        else x.clone()
    )
    span = (
        (nm.ub - nm.lb) if bounds is not None else torch.ones(dim, device=x.device, dtype=x.dtype)
    )
    delta = simplex_scale * span
    simplex = x_clamped.unsqueeze(0).repeat(dim + 1, 1)
    for i in range(dim):
        step = delta[i]
        cand = x_clamped[i] + step
        if bounds is not None:
            if cand > nm.ub[i]:
                cand = x_clamped[i] - step
            cand = torch.clamp(cand, min=nm.lb[i], max=nm.ub[i])
        simplex[i + 1, i] = cand
    nm.population.copy_(simplex)
    nm._initialized = True
    nm._needs_full_eval = True

    raw_f0 = fitness_fn(x_clamped.unsqueeze(0)).squeeze()
    best_x = x_clamped.clone()
    best_f = (
        torch.where(
            torch.isnan(raw_f0),
            torch.tensor(float("inf"), device=x.device, dtype=x.dtype),
            raw_f0,
        )
        if raw_f0.is_floating_point()
        else raw_f0.to(dtype=x.dtype)
    )
    evals = 1
    while evals < budget:
        batch_size = nm.next_batch_size()
        if evals + batch_size > budget:
            break
        candidates = nm.ask()
        fitness = fitness_fn(candidates)
        nm.tell(candidates, fitness)
        evals += candidates.shape[0]
        if nm.best_fitness < best_f:
            best_f = nm.best_fitness.clone()
            best_x = nm.best_solution.clone()
    return best_x, best_f, evals
