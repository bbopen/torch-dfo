"""Base optimizer class with ask/tell interface and pre-allocated workspace."""

from __future__ import annotations

from typing import Any

import torch

from torch_dfo.utils import make_generator, normalize_bounds, resolve_device


class BaseOptimizer:
    """Abstract base for all derivative-free optimizers.

    Subclasses must implement ask() and tell().
    Pre-allocates population, fitness, and best-tracking tensors.
    """

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float],
        pop_size: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.device = resolve_device(device)
        self.dtype = dtype
        self.lb, self.ub = normalize_bounds(bounds, dim, self.device, self.dtype)
        self._rng_seed = seed  # stored for cross-device serialization
        self._gen = make_generator(seed, self.device)
        self._gen_device = self._gen.device  # Cache for random tensor generation
        self._generation = 0

        # Pre-allocated workspace
        self.population = torch.empty(pop_size, dim, device=self.device, dtype=self.dtype)
        self.fitness = torch.full((pop_size,), float("inf"), device=self.device, dtype=self.dtype)
        self.best_solution = torch.zeros(dim, device=self.device, dtype=self.dtype)
        self.best_fitness = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)

    def ask(self) -> torch.Tensor:
        """Generate candidate solutions. Returns (pop_size, dim) tensor."""
        raise NotImplementedError

    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Report fitness values for candidates from the last ask().

        Args:
            candidates: (pop_size, dim) tensor of solutions
            fitness: (pop_size,) tensor of objective values (lower is better)

        """
        raise NotImplementedError

    def best(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (best_solution, best_fitness) found so far."""
        return self.best_solution.clone(), self.best_fitness.clone()

    def _update_best(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Update best_solution/best_fitness from new candidates.

        Uses torch.where for torch.compile compatibility -- no Python if/else on tensors.
        """
        # Find best in this batch
        batch_best_idx = fitness.argmin()
        batch_best_f = fitness[batch_best_idx]
        batch_best_x = candidates[batch_best_idx]

        # Update global best using torch.where (compile-safe)
        improved = batch_best_f < self.best_fitness
        self.best_fitness = torch.where(improved, batch_best_f, self.best_fitness)
        # For the solution vector, expand improved to match dim
        self.best_solution = torch.where(improved, batch_best_x, self.best_solution)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state as a serializable dict.

        All tensors are cloned so mutating the returned dict does not
        affect the optimizer.

        Notes
        -----
        The RNG state (``_gen_state``) is binary and device-specific.
        Same-device round-trips are bit-exact. Cross-device loads
        (e.g. saving on CPU, restoring on CUDA) fall back to re-seeding
        from ``_rng_seed`` — the optimizer will continue from the same
        seed but the sequence will diverge from the original.
        """
        return {
            "population": self.population.clone(),
            "fitness": self.fitness.clone(),
            "best_solution": self.best_solution.clone(),
            "best_fitness": self.best_fitness.clone(),
            "_generation": self._generation,
            "_gen_state": self._gen.get_state(),
            "_rng_seed": self._rng_seed,
            "_saved_device_type": self._gen.device.type,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from a dict produced by :meth:`state_dict`.

        Notes
        -----
        When the saved device type differs from the current generator device
        (e.g. loading a CPU checkpoint into a CUDA optimizer), the RNG is
        re-seeded from ``_rng_seed`` rather than restoring the binary state.
        Continuation is non-bit-exact in that case.
        """
        self.population.copy_(state["population"])
        self.fitness.copy_(state["fitness"])
        self.best_solution.copy_(state["best_solution"])
        self.best_fitness.copy_(state["best_fitness"])
        self._generation = state["_generation"]

        saved_device_type = state.get("_saved_device_type", self._gen.device.type)
        if saved_device_type == self._gen.device.type:
            self._gen.set_state(state["_gen_state"])
        else:
            # Cross-device: binary RNG state is not portable; re-seed instead.
            seed = state.get("_rng_seed")
            if seed is not None:
                self._gen.manual_seed(seed)
            # If no seed was stored we leave the generator in its current state.

    def _rand(self, *shape: int) -> torch.Tensor:
        """Generate uniform random tensor on the correct device.

        Handles MPS/XLA by generating on CPU then moving.
        """
        t = torch.rand(*shape, device=self._gen_device, dtype=self.dtype, generator=self._gen)
        if self._gen_device != self.device:
            t = t.to(self.device)
        return t

    def _randn(self, *shape: int) -> torch.Tensor:
        """Generate normal random tensor on the correct device."""
        t = torch.randn(*shape, device=self._gen_device, dtype=self.dtype, generator=self._gen)
        if self._gen_device != self.device:
            t = t.to(self.device)
        return t

    def _randperm(self, n: int) -> torch.Tensor:
        """Generate random permutation on the correct device."""
        t = torch.randperm(n, device=self._gen_device, generator=self._gen)
        if self._gen_device != self.device:
            t = t.to(self.device)
        return t

    def _randint(self, low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate random integers on the correct device."""
        t = torch.randint(low, high, shape, device=self._gen_device, generator=self._gen)
        if self._gen_device != self.device:
            t = t.to(self.device)
        return t
