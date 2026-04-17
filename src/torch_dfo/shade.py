"""Success-History Based Adaptive Differential Evolution (SHADE).

Tanabe & Fukunaga (2014). Self-adaptive F and CR via success-history memory.
Uses current-to-pbest/1 mutation with JADE-style external archive.

References
----------
R. Tanabe and A. Fukunaga, "Success-History Based Parameter Adaptation
for Differential Evolution," 2013 IEEE Congress on Evolutionary Computation,
pp. 71-78, 2013.

"""

from __future__ import annotations

from typing import Any

import torch

from torch_dfo._operators import (
    de_binomial_crossover,
    de_current_to_pbest_mutation,
    opposition_init,
)
from torch_dfo.base import BaseOptimizer


class SHADE(BaseOptimizer):
    """Success-History Based Adaptive Differential Evolution.

    Tanabe & Fukunaga (2014). Self-adaptive F and CR via success-history memory.
    Uses current-to-pbest/1 mutation with JADE-style archive.

    Parameters
    ----------
    dim : int
        Dimensionality of the search space.
    bounds : float | tuple[float, float]
        Search bounds (symmetric scalar or (lo, hi) tuple).
    pop_size : int
        Population size.
    memory_size : int
        Length of the circular success-history buffer for F and CR.
    p_min : float
        Minimum p-best fraction (reached at the end of the search).
    p_max : float
        Maximum p-best fraction (used at the start of the search).
    archive_ratio : float
        Maximum archive size as a fraction of pop_size.
    device : str | torch.device | None
        Torch device.
    dtype : torch.dtype
        Floating-point dtype.
    seed : int | None
        Reproducibility seed for the internal RNG.
    initial_population : torch.Tensor | None, optional
        Warm-start seed for the first ask(). The first ``n`` rows of the
        population are replaced with the first ``n`` rows of this tensor
        (clamped to bounds), where ``n = min(len(initial_population), pop_size)``.
        The tensor is freed after the first ask(). Defaults to ``None``.

    Examples
    --------
    >>> import torch
    >>> import torch_dfo
    >>> opt = torch_dfo.SHADE(dim=20, bounds=5.0, pop_size=60, seed=0)
    >>> for _ in range(100):
    ...     x = opt.ask()
    ...     opt.tell(x, (x ** 2).sum(-1))
    >>> best_x, best_f = opt.best()

    """

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float],
        pop_size: int = 80,
        memory_size: int = 6,
        p_min: float = 0.1,
        p_max: float = 0.3,
        archive_ratio: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
        *,
        initial_population: torch.Tensor | None = None,
    ):
        super().__init__(dim, bounds, pop_size, device, dtype, seed)

        # SHADE memory (circular buffer for F and CR)
        self.memory_size = memory_size
        self.memory_F = torch.full((memory_size,), 0.7, device=self.device, dtype=self.dtype)
        self.memory_CR = torch.full((memory_size,), 0.7, device=self.device, dtype=self.dtype)
        self._memory_pos = 0

        # p-best parameters (adaptive from p_max to p_min over generations)
        self.p_min = p_min
        self.p_max = p_max

        # Archive (JADE-style: stores replaced parents)
        self.archive_ratio = archive_ratio
        self._archive_max = int(pop_size * archive_ratio)
        self._archive = torch.empty(0, dim, device=self.device, dtype=self.dtype)

        # Per-offspring trial parameters (pre-allocated)
        self._trial_F = torch.empty(pop_size, device=self.device, dtype=self.dtype)
        self._trial_CR = torch.empty(pop_size, device=self.device, dtype=self.dtype)

        # Trial vectors storage
        self._trials = torch.empty(pop_size, dim, device=self.device, dtype=self.dtype)

        # Initialization flag
        self._initialized = False

        # Warm-start: optional initial population to seed first n rows
        self._initial_population: torch.Tensor | None = (
            initial_population.to(device=self.device, dtype=self.dtype).clone()
            if initial_population is not None
            else None
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return SHADE state as a serializable dict."""
        state = super().state_dict()
        state.update(
            {
                "memory_F": self.memory_F.clone(),
                "memory_CR": self.memory_CR.clone(),
                "_memory_pos": self._memory_pos,
                "_archive": self._archive.clone(),
                "_trial_F": self._trial_F.clone(),
                "_trial_CR": self._trial_CR.clone(),
                "_trials": self._trials.clone(),
                "_initialized": self._initialized,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore SHADE state from a dict produced by :meth:`state_dict`."""
        super().load_state_dict(state)
        self.memory_F.copy_(state["memory_F"])
        self.memory_CR.copy_(state["memory_CR"])
        self._memory_pos = state["_memory_pos"]
        self._archive = state["_archive"].clone()
        self._trial_F.copy_(state["_trial_F"])
        self._trial_CR.copy_(state["_trial_CR"])
        self._trials.copy_(state["_trials"])
        self._initialized = state["_initialized"]

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------
    def ask(self) -> torch.Tensor:
        """Generate candidate solutions.

        On the first call, initialises the population via opposition-based
        learning. If ``initial_population`` was provided at construction, the
        first ``n`` rows are replaced with those seeded points (clamped to bounds).
        On subsequent calls, produces trial vectors through SHADE's adaptive
        mutation and crossover.

        Returns
        -------
        torch.Tensor
            (pop_size, dim) candidate solutions.

        """
        if not self._initialized:
            pop = opposition_init(self.pop_size, self.dim, self.lb, self.ub, generator=self._gen)
            self.population.copy_(pop)
            if self._initial_population is not None:
                n = min(self._initial_population.shape[0], self.pop_size)
                self.population[:n].copy_(self._initial_population[:n].clamp(self.lb, self.ub))
                self._initial_population = None  # free memory after use
            self._initialized = True
            return self.population.clone()

        # --- Sample F and CR from memory ---
        # Pick random memory index for each individual
        mem_idx = self._randint(0, self.memory_size, (self.pop_size,))

        # F_i ~ Normal(memory_F[k], 0.1), clamped to [0.05, 1.0]
        loc_F = self.memory_F[mem_idx]
        trial_F = loc_F + 0.1 * self._randn(self.pop_size)
        trial_F = trial_F.clamp(0.05, 1.0)

        # CR_i ~ Normal(memory_CR[k], 0.1), clamped to [0.0, 1.0]
        loc_CR = self.memory_CR[mem_idx]
        trial_CR = loc_CR + 0.1 * self._randn(self.pop_size)
        trial_CR = trial_CR.clamp(0.0, 1.0)

        # Store for tell() memory update
        self._trial_F.copy_(trial_F)
        self._trial_CR.copy_(trial_CR)

        # --- Adaptive p_fraction: linear decay from p_max to p_min ---
        # Use a logistic-like schedule; for simplicity, linear over 5000 gens
        max_gen = 5000.0
        progress = min(self._generation / max_gen, 1.0)
        p_fraction = self.p_max - (self.p_max - self.p_min) * progress
        # Ensure at least 1 individual in pbest pool
        p_fraction = max(p_fraction, 1.0 / self.pop_size)

        # --- Mutation: current-to-pbest/1 ---
        archive = self._archive if self._archive.shape[0] > 0 else None
        donor = de_current_to_pbest_mutation(
            self.population,
            self.fitness,
            trial_F,
            p_fraction,
            archive=archive,
            generator=self._gen,
        )

        # --- Crossover: binomial ---
        trials = de_binomial_crossover(
            donor,
            self.population,
            trial_CR,
            generator=self._gen,
        )

        # --- Clamp to bounds ---
        trials = torch.clamp(trials, self.lb, self.ub)

        self._trials.copy_(trials)
        return trials

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------
    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Report fitness for the last ask() candidates and update state.

        Performs greedy selection, archives replaced parents, updates the
        SHADE success-history memory, and advances the generation counter.

        Parameters
        ----------
        candidates : torch.Tensor
            (pop_size, dim) solutions from the last ``ask()`` call.
        fitness : torch.Tensor
            (pop_size,) objective values (lower is better).

        """
        # On the very first tell (initialization), just store fitness and update best
        if self._generation == 0:
            self.fitness.copy_(fitness)
            self.population.copy_(candidates)
            self._update_best(candidates, fitness)
            self._generation += 1
            return

        # --- Greedy selection: vectorized ---
        improved = fitness < self.fitness
        new_pop = torch.where(improved.unsqueeze(1), candidates, self.population)
        new_fit = torch.where(improved, fitness, self.fitness)

        # --- Archive: add replaced parents for successful trials ---
        if improved.any():
            replaced_parents = self.population[improved]
            self._archive = torch.cat([self._archive, replaced_parents], dim=0)
            # Truncate archive if over capacity by keeping random subset
            if self._archive.shape[0] > self._archive_max:
                perm = self._randperm(self._archive.shape[0])
                self._archive = self._archive[perm[: self._archive_max]]

        # --- SHADE memory update for successful offspring ---
        if improved.any():
            succ_F = self._trial_F[improved]
            succ_CR = self._trial_CR[improved]

            # Improvement weights: delta_f_j = f_parent_j - f_offspring_j
            delta_f = self.fitness[improved] - fitness[improved]
            weights = delta_f / delta_f.sum()

            # Lehmer mean for F: sum(w * F^2) / sum(w * F)
            wF = weights * succ_F
            wF2 = weights * succ_F * succ_F
            denom = wF.sum()
            lehmer_F = wF2.sum() / denom if denom > 0 else self.memory_F[self._memory_pos]

            # Weighted arithmetic mean for CR
            mean_CR = (weights * succ_CR).sum()

            self.memory_F[self._memory_pos] = lehmer_F
            self.memory_CR[self._memory_pos] = mean_CR
            self._memory_pos = (self._memory_pos + 1) % self.memory_size

        # --- Update population and fitness ---
        self.population.copy_(new_pop)
        self.fitness.copy_(new_fit)

        # --- Update global best ---
        self._update_best(self.population, self.fitness)

        self._generation += 1
