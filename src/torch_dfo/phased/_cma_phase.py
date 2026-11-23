"""CMA-ES-phase helpers for :class:`PhasedDFO`.

This module contains the CMA-ES (Phase 1) specific logic extracted from
:class:`torch_dfo.phased.orchestrator.PhasedDFO`.  Functions here take the
orchestrator instance as the first argument (``opt``) rather than being
methods on the class; this keeps the module a leaf in the import graph and
avoids circular imports with ``orchestrator.py``.

The public entry points mirror the private ``PhasedDFO`` methods they
replace:

* :func:`enter_cmaes_phase`
* :func:`enter_cmaes_phase_portfolio`
* :func:`restart_cmaes`
* :func:`restart_portfolio_branch`
* :func:`sample_restart_mean`
* :func:`compute_cmaes_phase_budgets`
* :func:`is_valley_entry_branch`
* :func:`portfolio_lambdas_for_dim`
* :func:`portfolio_active_indices_for_next_ask`
* :func:`in_valley_terminal_focus_window`
* :func:`update_valley_focus_schedule`
* :func:`seed_path_memory_from_elites`
* :func:`get_bounds_tuple`

Each is a logic-preserving move of the corresponding ``_enter_cmaes_phase``
/ ``_restart_cmaes`` / ... method from ``PhasedDFO``, with ``self`` renamed
to ``opt``.  No algorithmic behavior is changed.

Module-level helpers ``_compute_valley_focus_generation_bounds``,
``_normalize_covariance`` and ``_merge_search_pool`` live in the leaf
:mod:`torch_dfo.phased._common` module and are imported directly at
module load time.  ``_debug_is_disabled`` lives in the private
:mod:`torch_dfo.phased._debug` leaf module and is also imported directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from torch_dfo.cmaes import CMAES
from torch_dfo.phased._common import (
    _compute_valley_focus_generation_bounds,
    _normalize_covariance,
)
from torch_dfo.phased._debug import (
    TAG_DIM40_ADAPTIVE_BURST,
    TAG_DIM40_BUDGET_TRANSFER,
    TAG_DIM40_FOCUS_CYCLE,
    TAG_DIM40_LINE_SAMPLING,
    TAG_DIM40_PARITY_BOUNDS,
    TAG_DIM40_TERMINAL_FOCUS,
    TAG_DIM40_VALLEY_BRANCH,
    _debug_is_disabled,
)
from torch_dfo.utils import clamp_to_bounds

if TYPE_CHECKING:
    from torch_dfo.phased.orchestrator import PhasedDFO


def get_bounds_tuple(opt: PhasedDFO) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-dimension tensor bounds for sub-optimizer construction."""
    return (opt.lb, opt.ub)


def enter_cmaes_phase(opt: PhasedDFO) -> None:
    """Create CMA-ES with warm-started covariance from DE elite."""
    if not opt._cmaes_entered:
        opt._cmaes_overall_start_f = float(opt.best_fitness)
        opt._cmaes_entered = True

    elite_count = min(opt._shade.pop_size // 2, opt._shade.pop_size)
    sorted_idx = opt._shade.fitness.argsort()
    elite = opt._shade.population[sorted_idx[:elite_count]]
    elite_mean = elite.mean(dim=0)

    span = (opt.ub - opt.lb).mean().item()

    centered = elite - elite_mean.unsqueeze(0)
    elite_cov = (centered.T @ centered) / max(1, elite_count - 1)

    sigma_raw = max(
        elite.std(dim=0).mean().item(),
        opt._cma_sigma_min * span,
    )
    sigma = min(sigma_raw, opt._cma_sigma_max * span)

    eye = torch.eye(opt.dim, device=opt.device, dtype=opt.dtype)
    if sigma > 1e-30:  # noqa: SIM108
        normalized_cov = elite_cov / (sigma * sigma)
    else:
        normalized_cov = eye
    C_init = 0.7 * normalized_cov + 0.3 * eye

    C_init = (C_init + C_init.T) / 2
    if opt.device.type not in ("cpu", "cuda"):
        eigvals = torch.linalg.eigvalsh(C_init.to("cpu"))
    else:
        eigvals = torch.linalg.eigvalsh(C_init)
    if eigvals.min().item() < 1e-10:
        C_init = C_init + (1e-8 - eigvals.min().item()) * eye

    base_pop = max(
        4 + math.floor(3 * math.log(opt.dim)),
        opt._shade.pop_size // 2,
    )
    if opt._high_dim:
        base_pop = max(base_pop, opt._high_dim_de_min_pop)

    opt._cmaes = CMAES(
        dim=opt.dim,
        bounds=get_bounds_tuple(opt),
        pop_size=base_pop,
        device=opt.device,
        dtype=opt.dtype,
        seed=None,
        sigma0=sigma / span if span > 0 else 0.3,
        mirrored=opt._high_dim,
        active=opt._high_dim,
    )

    opt._cmaes.mean = elite_mean.clone()
    opt._cmaes.sigma = sigma
    opt._cmaes._normalize_on_decomp = False
    opt._cmaes.C = _normalize_covariance(C_init, opt.device, opt.dtype)
    opt._cmaes._update_eigensystem()

    opt._cmaes._gen = opt._gen
    opt._cmaes._gen_device = opt._gen_device

    opt._cmaes_phase_idx = 0
    opt._cmaes_stagnation_counter = 0
    opt._cmaes_phase_best_f = opt.best_fitness.item()
    opt._cmaes_base_pop = base_pop

    elite_fitness = opt._shade.fitness[sorted_idx[:elite_count]]
    opt._search_population = elite.clone()
    opt._search_population_fitness = elite_fitness.clone()
    elite_snapshot_size = max(8, min(2 * opt.dim, 32))
    opt._search_pool_limit = max(
        elite_snapshot_size,
        min(
            opt._search_pool_max,
            elite_snapshot_size * opt._config.cma_es_search_pool_factor,
        ),
    )

    opt._cmaes_fe_start = opt._fe_count
    remaining = opt._cmaes_budget - opt._fe_count
    opt._cmaes_phase_budgets = compute_cmaes_phase_budgets(opt, remaining)


def is_valley_entry_branch(opt: PhasedDFO, idx: int) -> bool:
    """Return True for the dim40+ incumbent LM-CMA portfolio branch."""
    if _debug_is_disabled(TAG_DIM40_VALLEY_BRANCH):
        return False
    return (
        opt._high_dim
        and opt.dim >= opt._config.high_dim_valley_entry_dim
        and idx == opt._config.high_dim_valley_entry_branch
    )


def portfolio_lambdas_for_dim(opt: PhasedDFO) -> tuple[int, ...]:
    """Return the CMA portfolio population schedule for this dimension."""
    if _debug_is_disabled(TAG_DIM40_BUDGET_TRANSFER):
        return opt._config.k_portfolio_lambdas
    if opt._high_dim and opt.dim >= opt._config.high_dim_valley_entry_dim:
        return opt._config.high_dim_valley_entry_portfolio_lambdas
    return opt._config.k_portfolio_lambdas


def portfolio_active_indices_for_next_ask(opt: PhasedDFO) -> tuple[int, ...]:
    """Return portfolio branches to sample on the next CMA generation."""
    if opt._cmaes_portfolio is None:
        return ()

    all_indices = tuple(range(len(opt._cmaes_portfolio)))
    if not (opt._high_dim and opt.dim >= opt._config.high_dim_valley_entry_dim):
        return all_indices
    if opt._config.high_dim_valley_entry_focus_cycle <= 1:
        return all_indices
    if _debug_is_disabled(TAG_DIM40_FOCUS_CYCLE):
        return all_indices
    if in_valley_terminal_focus_window(opt):
        return (opt._config.high_dim_valley_entry_branch,)
    if _debug_is_disabled(TAG_DIM40_ADAPTIVE_BURST):
        if opt._portfolio_generation % opt._config.high_dim_valley_entry_focus_cycle == 0:
            return all_indices
        return (opt._config.high_dim_valley_entry_branch,)
    if opt._valley_focus_remaining > 0:
        return (opt._config.high_dim_valley_entry_branch,)
    return all_indices


def in_valley_terminal_focus_window(opt: PhasedDFO) -> bool:
    """Return True when final dim40 CMA budget should stay local."""
    if _debug_is_disabled(TAG_DIM40_TERMINAL_FOCUS):
        return False
    if not (
        opt._high_dim
        and opt.dim >= opt._config.high_dim_valley_entry_dim
        and opt._config.high_dim_valley_entry_terminal_focus_fraction > 0.0
    ):
        return False
    cma_total = max(1, opt._cmaes_budget - opt._cmaes_fe_start)
    cma_remaining = max(0, opt._cmaes_budget - opt._fe_count)
    terminal_budget = math.ceil(
        opt._config.high_dim_valley_entry_terminal_focus_fraction * cma_total,
    )
    return cma_remaining <= terminal_budget


def update_valley_focus_schedule(
    opt: PhasedDFO,
    active_indices: tuple[int, ...],
    valley_branch_improved: bool,
) -> None:
    """Adapt the dim40 incumbent-only burst after one portfolio generation."""
    if not (
        opt._high_dim
        and opt.dim >= opt._config.high_dim_valley_entry_dim
        and opt._config.high_dim_valley_entry_focus_cycle > 1
    ):
        opt._valley_focus_remaining = 0
        opt._valley_focus_streak = 0
        return

    if _debug_is_disabled(TAG_DIM40_FOCUS_CYCLE) or _debug_is_disabled(
        TAG_DIM40_ADAPTIVE_BURST,
    ):
        opt._valley_focus_remaining = 0
        opt._valley_focus_streak = 0
        return

    if _debug_is_disabled(TAG_DIM40_PARITY_BOUNDS):
        min_focus = max(0, opt._config.high_dim_valley_entry_focus_cycle - 1)
        max_focus = max(min_focus, opt._config.high_dim_valley_entry_max_focus_cycle - 1)
    else:
        min_focus, max_focus = _compute_valley_focus_generation_bounds(
            portfolio_lambdas_for_dim(opt),
        )
    valley_only = active_indices == (opt._config.high_dim_valley_entry_branch,)
    if not valley_only:
        opt._valley_focus_remaining = min_focus
        opt._valley_focus_streak = 0
        return

    opt._valley_focus_remaining = max(0, opt._valley_focus_remaining - 1)
    opt._valley_focus_streak += 1
    if valley_branch_improved and opt._valley_focus_streak < max_focus:
        opt._valley_focus_remaining = max(opt._valley_focus_remaining, 1)


def seed_path_memory_from_elites(opt: PhasedDFO, cma: CMAES, center: torch.Tensor) -> None:
    """Seed a CMA path-memory branch from the current DE elite cloud."""
    if cma.path_memory <= 0 or opt._shade.fitness.numel() == 0:
        return

    top_k = min(opt._shade.fitness.shape[0], max(2 * cma.path_memory, cma.path_memory + 1))
    if top_k <= 1:
        return

    elite_idx = opt._shade.fitness.argsort()[:top_k]
    offsets = opt._shade.population[elite_idx] - center.unsqueeze(0)
    norms = torch.linalg.vector_norm(offsets, dim=1)
    valid = torch.isfinite(norms) & (norms > 1e-12)
    if not valid.any():
        return

    directions = offsets[valid] / norms[valid].unsqueeze(1)
    use = min(cma.path_memory, directions.shape[0])
    cma._path_vectors.zero_()
    cma._path_vectors[:use] = directions[:use]
    cma._path_count = use
    cma._path_pos = use % cma.path_memory


def enter_cmaes_phase_portfolio(opt: PhasedDFO) -> None:
    """Initialize K=4 parallel CMA-ES portfolio for high-dim phase.

    Each branch has a distinct (lambda, sigma) and a random x0 drawn
    uniformly from the search space.  Lambda values (24,12,12,12) give
    ~917 generations in a 55k CMA budget — 3x more than the old
    (96,48,24,12) config (~300 gens), critical for both covariance
    adaptation (ill-conditioned functions) and restart diversity (f24).
    """
    if not opt._cmaes_entered:
        opt._cmaes_overall_start_f = float(opt.best_fitness)
        opt._cmaes_entered = True

    span = (opt.ub - opt.lb).mean().item()
    portfolio_lambdas = portfolio_lambdas_for_dim(opt)
    K = len(portfolio_lambdas)
    opt._cmaes_portfolio = []
    opt._portfolio_stag = [0] * K
    opt._portfolio_best_f = [float("inf")] * K
    opt._portfolio_sigma0 = []
    opt._portfolio_generation = 0
    opt._portfolio_active_indices = ()
    opt._valley_focus_remaining = 0
    opt._valley_focus_streak = 0

    total_lam = sum(portfolio_lambdas)
    cma_remaining = max(1, opt._cmaes_budget - opt._fe_count)
    total_portfolio_gens = cma_remaining // max(total_lam, 1)
    opt._portfolio_branch_stag_limit = max(40, total_portfolio_gens // 15)

    for idx, (lam, sigma_frac) in enumerate(
        zip(portfolio_lambdas, opt._config.k_portfolio_sigma_fracs, strict=True)
    ):
        sigma_abs = sigma_frac * span
        is_valley_branch = is_valley_entry_branch(opt, idx)
        cma = CMAES(
            dim=opt.dim,
            bounds=get_bounds_tuple(opt),
            pop_size=lam,
            device=opt.device,
            dtype=opt.dtype,
            seed=None,
            sigma0=sigma_frac,
            mirrored=True,
            active=True,
            path_memory=(opt._config.high_dim_valley_entry_path_memory if is_valley_branch else 0),
            path_scale=(opt._config.high_dim_valley_entry_path_scale if is_valley_branch else 0.0),
            path_line_samples=(
                opt._config.high_dim_valley_entry_line_samples
                if is_valley_branch and not _debug_is_disabled(TAG_DIM40_LINE_SAMPLING)
                else 0
            ),
            path_line_scale=(
                opt._config.high_dim_valley_entry_line_scale
                if is_valley_branch and not _debug_is_disabled(TAG_DIM40_LINE_SAMPLING)
                else 1.0
            ),
        )
        if is_valley_branch and torch.isfinite(opt.best_fitness):
            x0 = opt.best_solution.clone()
        else:
            x0 = opt.lb + opt._rand(opt.dim) * (opt.ub - opt.lb)
        cma.mean = x0.clone()
        cma.sigma = sigma_abs
        cma._gen = opt._gen
        cma._gen_device = opt._gen_device
        if is_valley_branch:
            seed_path_memory_from_elites(opt, cma, x0)
        opt._cmaes_portfolio.append(cma)
        opt._portfolio_sigma0.append(sigma_abs)

    opt._cmaes_fe_start = opt._fe_count


def restart_portfolio_branch(opt: PhasedDFO, idx: int) -> None:
    """Restart one CMA-ES portfolio branch from a new random x0."""
    assert opt._cmaes_portfolio is not None
    cma = opt._cmaes_portfolio[idx]
    span = (opt.ub - opt.lb).mean().item()
    sigma_abs = opt._config.k_portfolio_sigma_fracs[idx] * span
    if is_valley_entry_branch(opt, idx) and torch.isfinite(opt.best_fitness):
        jitter = opt._randn(opt.dim) * (
            opt._config.high_dim_valley_entry_restart_jitter * sigma_abs
        )
        x0 = opt.best_solution + jitter
        x0 = clamp_to_bounds(x0.unsqueeze(0), opt.lb, opt.ub).squeeze(0)
    else:
        x0 = opt.lb + opt._rand(opt.dim) * (opt.ub - opt.lb)
    eye = torch.eye(opt.dim, device=opt.device, dtype=opt.dtype)
    cma.restart(mean=x0, sigma=sigma_abs, C_init=eye)
    if is_valley_entry_branch(opt, idx):
        seed_path_memory_from_elites(opt, cma, x0)
    opt._portfolio_stag[idx] = 0
    opt._portfolio_best_f[idx] = float("inf")


def compute_cmaes_phase_budgets(opt: PhasedDFO, total: int) -> list[int]:
    """Split remaining CMA-ES budget across IPOP phases.

    For high-dim, the warm-started phase 0 gets a larger initial share
    (40% of total) so it has enough budget to descend through narrow
    ill-conditioned basins.  The remaining phases share the rest equally.

    For low-dim (single phase), the full budget goes to phase 0.

    Rationale for the high-dim front-loading: phase 0 starts from the
    DE warm-start (best-known position) with a tight sigma that gradually
    learns the function's conditioning.  Ill-conditioned unimodal functions
    (f10-f14) require many CMA iterations to traverse the elongated basin.
    With equal splits at n_phases=8, phase 0 only gets total/8 evaluations
    -- insufficient for convergence.  Giving phase 0 40% allows full
    convergence of the warm-started run; the remaining 60% spread over
    7 random restarts is adequate diversity for multimodal functions.

    Reference-point preservation (dim=10 low-dim, single phase):
    phase 0 gets the full remaining budget.  Unchanged.
    """
    n_phases = opt._cmaes_phase_count
    phase_budgets: list[int] = []
    remaining = total

    if n_phases <= 1:
        phase_budgets.append(max(remaining, 200))
        return phase_budgets

    if opt._high_dim:
        phase0_share = max(200, int(total * 0.40))
        phase0_share = min(phase0_share, remaining - (n_phases - 1) * 200)
        phase_budgets.append(phase0_share)
        remaining -= phase0_share
        for i in range(1, n_phases):
            phases_left = n_phases - i
            share = max(remaining // phases_left, 200)
            share = min(share, remaining)
            phase_budgets.append(share)
            remaining -= share
    else:
        for i in range(n_phases):
            phases_left = n_phases - i
            share = max(remaining // phases_left, 200)
            share = min(share, remaining)
            phase_budgets.append(share)
            remaining -= share

    return phase_budgets


def restart_cmaes(opt: PhasedDFO) -> None:
    """Perform an IPOP restart of CMA-ES with doubled population.

    Restart mean cycles through 4 modes based on phase_idx % 4:
    0 = random, 1 = pool anchor, 2 = differential, 3 = mirrored best.
    """
    assert opt._cmaes is not None

    span = (opt.ub - opt.lb).mean().item()
    eye = torch.eye(opt.dim, device=opt.device, dtype=opt.dtype)

    if opt._high_dim and opt._cmaes_phase_idx % 4 == 1:
        new_pop = opt._cmaes_base_pop
        sigma = 0.002 * span
        restart_center = opt.lb + opt._rand(opt.dim) * (opt.ub - opt.lb)
        C_init = eye.clone()
    else:
        new_pop = opt._cmaes_base_pop * (opt._config.cma_es_pop_growth**opt._cmaes_phase_idx)

        remaining = opt._cmaes_budget - opt._fe_count
        min_gens = 10
        max_pop = max(remaining // min_gens, opt._cmaes_base_pop)
        new_pop = min(new_pop, max_pop)

        restart_center = sample_restart_mean(opt, opt._cmaes_phase_idx, span)

        sigma = (
            opt._cma_restart_sigma_min
            + opt._rand(1).item() * (opt._cma_restart_sigma_max - opt._cma_restart_sigma_min)
        ) * span

        old_C = opt._cmaes.C.clone()
        blend = opt._config.cma_es_restart_cov_blend
        C_init = (1 - blend) * eye + blend * old_C

        C_init = (C_init + C_init.T) / 2

        C_init = _normalize_covariance(C_init, opt.device, opt.dtype)

    opt._cmaes.restart(
        new_pop_size=new_pop,
        mean=restart_center,
        sigma=sigma,
        C_init=C_init,
    )

    opt._cmaes_stagnation_counter = 0
    opt._cmaes_phase_best_f = opt.best_fitness.item()


def sample_restart_mean(opt: PhasedDFO, phase_idx: int, span: float) -> torch.Tensor:
    """Sample restart center using 4-mode cycling.

    Mode 0: random position in search space.
    Mode 1: random anchor from search pool + jitter.
    Mode 2: differential restart (anchor + scale*(anchor - partner) + jitter).
    Mode 3: mirrored best_solution (lb + ub - best) + jitter.
    """
    restart_pool: torch.Tensor | None = None
    if opt._search_population is not None and opt._search_population.numel() > 0:
        restart_pool = opt._search_population
    elif len(opt._elite_solutions) > 0:
        restart_pool = torch.stack(opt._elite_solutions[-50:])

    restart_mode = phase_idx % opt._config.cma_es_restart_modes

    if restart_mode == 1 and restart_pool is not None and restart_pool.shape[0] > 0:
        anchor_pos = int(opt._randint(0, restart_pool.shape[0], (1,)).item())
        anchor = restart_pool[anchor_pos]
        jitter = opt._randn(opt.dim) * (opt._config.cma_es_elite_restart_jitter * span)
        restart_center = anchor + jitter
        return clamp_to_bounds(restart_center.unsqueeze(0), opt.lb, opt.ub).squeeze(0)

    if restart_mode == 2 and restart_pool is not None and restart_pool.shape[0] > 1:
        pair = opt._randperm(restart_pool.shape[0])[:2]
        anchor = restart_pool[pair[0]]
        partner = restart_pool[pair[1]]
        differential = anchor - partner
        diff_norm = torch.linalg.vector_norm(differential)
        if torch.isfinite(diff_norm) and float(diff_norm) > 1e-9:
            jitter = opt._randn(opt.dim) * (opt._config.cma_es_differential_restart_jitter * span)
            candidate = (
                anchor + opt._config.cma_es_differential_restart_scale * differential + jitter
            )
            return clamp_to_bounds(candidate.unsqueeze(0), opt.lb, opt.ub).squeeze(0)

    if restart_mode == 3:
        best_x, _ = opt.best()
        mirrored = opt.lb + opt.ub - best_x
        jitter = opt._randn(opt.dim) * (opt._config.cma_es_mirror_restart_jitter * span)
        restart_center = mirrored + jitter
        return clamp_to_bounds(restart_center.unsqueeze(0), opt.lb, opt.ub).squeeze(0)

    return opt._rand(opt.dim) * (opt.ub - opt.lb) + opt.lb
