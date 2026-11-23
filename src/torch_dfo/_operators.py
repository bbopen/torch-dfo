"""Vectorized differential evolution operators for torch-dfo.

All functions operate on batched tensors. Zero Python loops over population.
"""

from __future__ import annotations

import math

import torch


def de_current_to_pbest_mutation(
    population: torch.Tensor,
    fitness: torch.Tensor,
    F: torch.Tensor,
    p_fraction: float,
    archive: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DE/current-to-pbest/1 mutation, fully vectorized.

    v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)

    Where x_pbest is randomly selected from top p fraction,
    x_r1 from population, x_r2 from population + archive.

    Returns: (pop_size, dim) donor vectors
    """
    pop_size, _dim = population.shape
    device = population.device
    gen_device = generator.device if generator is not None else device

    p_count = max(2, math.ceil(pop_size * p_fraction))
    sorted_indices = fitness.argsort()
    pbest_pool = sorted_indices[:p_count]

    pbest_choices = torch.randint(0, p_count, (pop_size,), device=gen_device, generator=generator)
    if gen_device != device:
        pbest_choices = pbest_choices.to(device)
    x_pbest = population[pbest_pool[pbest_choices]]

    r1 = torch.randint(0, pop_size - 1, (pop_size,), device=gen_device, generator=generator)
    if gen_device != device:
        r1 = r1.to(device)
    i_idx = torch.arange(pop_size, device=device)
    r1 = torch.where(r1 >= i_idx, r1 + 1, r1)
    x_r1 = population[r1]

    if archive is not None and archive.shape[0] > 0:
        combined = torch.cat([population, archive], dim=0)
    else:
        combined = population
    total = combined.shape[0]

    r2 = torch.randint(0, total - 2, (pop_size,), device=gen_device, generator=generator)
    if gen_device != device:
        r2 = r2.to(device)
    r2 = torch.where(r2 >= i_idx, r2 + 1, r2)
    r2 = torch.where(r2 >= r1, r2 + 1, r2)
    r2 = torch.where(r2 == i_idx, (r2 + 1) % total, r2)
    r2 = r2.clamp(max=total - 1)
    x_r2 = combined[r2]

    F_col = F.unsqueeze(1)
    return population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)


def de_binomial_crossover(
    donor: torch.Tensor,
    target: torch.Tensor,
    CR: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Binomial crossover with forced j_rand, fully vectorized.

    u_ij = donor_ij  if rand() < CR_i or j == j_rand
           target_ij otherwise

    Returns: (pop_size, dim) trial vectors
    """
    pop_size, dim = donor.shape
    device = donor.device
    gen_device = generator.device if generator is not None else device

    rand_mat = torch.rand(pop_size, dim, device=gen_device, generator=generator)
    if gen_device != device:
        rand_mat = rand_mat.to(device)
    mask = rand_mat < CR.unsqueeze(1)

    j_rand = torch.randint(0, dim, (pop_size,), device=gen_device, generator=generator)
    if gen_device != device:
        j_rand = j_rand.to(device)
    mask.scatter_(1, j_rand.unsqueeze(1), True)

    return torch.where(mask, donor, target)


def levy_flight_perturbation(
    x: torch.Tensor,
    alpha: float = 1.5,
    step_scale: float = 0.1,
    progress: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mantegna algorithm for Levy flights, fully batched.

    Generates heavy-tailed step sizes and applies them to input solutions.
    Step scale anneals: effective_scale = step_scale * (1 - 0.7 * progress)

    Returns: (N, dim) perturbed solutions
    """
    N, dim = x.shape
    device = x.device
    dtype = x.dtype
    gen_device = generator.device if generator is not None else device

    num = math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2)
    den = math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2)
    sigma_u = (num / den) ** (1 / alpha)

    u = torch.randn(N, dim, device=gen_device, dtype=dtype, generator=generator) * sigma_u
    v = torch.randn(N, dim, device=gen_device, dtype=dtype, generator=generator)
    if gen_device != device:
        u = u.to(device)
        v = v.to(device)

    step = u / (v.abs() + 1e-30).pow(1.0 / alpha)

    step = 10.0 * torch.tanh(step / 10.0)

    effective_scale = step_scale * (1.0 - 0.7 * progress)

    return x + effective_scale * step


def opposition_init(
    pop_size: int,
    dim: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Opposition-based population initialization.

    Generate pop_size/2 random points, mirror them as x' = lb + ub - x,
    combine and truncate to pop_size.

    Returns: (pop_size, dim) initial population
    """
    device = lb.device
    dtype = lb.dtype
    gen_device = generator.device if generator is not None else device

    half = (pop_size + 1) // 2

    rand_pts = torch.rand(half, dim, device=gen_device, dtype=dtype, generator=generator)
    if gen_device != device:
        rand_pts = rand_pts.to(device)
    rand_pts = lb + rand_pts * (ub - lb)

    opposition = lb + ub - rand_pts

    combined = torch.cat([rand_pts, opposition], dim=0)
    return combined[:pop_size]
