"""Nelder-Mead simplex method for derivative-free local optimization.

Implements the standard reflect/expand/contract/shrink operations adapted
for a batched ask/tell interface.  Each ``ask()`` after initialization
returns 4 candidate points (reflected, expanded, outside-contracted,
inside-contracted) so that ``tell()`` can decide which operation to accept
using ``torch.where`` chains rather than Python ``if/else`` branching,
keeping the hot path friendly to ``torch.compile``.
"""

from __future__ import annotations

from typing import Any

import torch

from torch_dfo.base import BaseOptimizer


class NelderMead(BaseOptimizer):
    """Nelder-Mead simplex method for derivative-free local optimization.

    The simplex always has ``dim + 1`` vertices.  The optimizer accepts
    the standard Nelder-Mead hyper-parameters (reflection, expansion,
    contraction, and shrink coefficients).

    Ask/tell protocol
    -----------------
    * First ``ask()`` returns the initial simplex -- ``(dim+1, dim)`` tensor.
    * After the initial ``tell()``, every ``ask()`` returns 4 candidate
      points ``(4, dim)`` corresponding to the reflected, expanded,
      outside-contracted, and inside-contracted points.
    * After a **shrink** step, the next ``ask()`` returns the full simplex
      again for re-evaluation -- ``(dim+1, dim)`` tensor.
    * ``tell()`` selects the best operation via ``torch.where`` and updates
      the simplex accordingly.

    Parameters
    ----------
    dim : int
        Dimensionality of the search space.
    bounds : float | tuple[float, float]
        Search bounds (see ``normalize_bounds``).
    device, dtype, seed :
        Forwarded to :class:`BaseOptimizer`.
    alpha : float
        Reflection coefficient (default 1.0).
    gamma : float
        Expansion coefficient (default 2.0).
    rho : float
        Contraction coefficient (default 0.5).
    shrink : float
        Shrink coefficient (default 0.5).

    Examples
    --------
    >>> import torch
    >>> import torch_dfo
    >>> opt = torch_dfo.NelderMead(dim=5, bounds=5.0, seed=0)
    >>> for _ in range(200):
    ...     x = opt.ask()
    ...     opt.tell(x, (x ** 2).sum(-1))
    >>> best_x, best_f = opt.best()

    """

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float],
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        shrink: float = 0.5,
    ):
        pop_size = dim + 1
        super().__init__(dim, bounds, pop_size, device, dtype, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.shrink_coeff = shrink

        # State tracking
        self._initialized = False
        self._needs_full_eval = True  # True => next ask() returns full simplex

        # Pre-allocated candidate storage (4 trial points per iteration)
        self._candidates = torch.zeros(4, dim, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return Nelder-Mead state as a serializable dict."""
        state = super().state_dict()
        state.update(
            {
                "_initialized": self._initialized,
                "_needs_full_eval": self._needs_full_eval,
                "_candidates": self._candidates.clone(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore Nelder-Mead state from a dict."""
        super().load_state_dict(state)
        self._initialized = state["_initialized"]
        self._needs_full_eval = state["_needs_full_eval"]
        self._candidates.copy_(state["_candidates"])

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------
    def ask(self) -> torch.Tensor:
        """Return candidate points for evaluation.

        Returns
        -------
        torch.Tensor
            ``(dim+1, dim)`` on the first call or after a shrink step;
            ``(4, dim)`` on every other call.

        """
        if not self._initialized:
            # Random initial simplex within bounds
            self.population[:] = self._rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb
            self._initialized = True
            self._needs_full_eval = True
            return self.population.clone()

        if self._needs_full_eval:
            # After a shrink: re-evaluate the full simplex
            return self.population.clone()

        # Sort simplex by fitness (best first)
        sorted_idx = self.fitness.argsort()
        self.population[:] = self.population[sorted_idx]
        self.fitness[:] = self.fitness[sorted_idx]

        worst = self.population[-1]
        centroid = self.population[:-1].mean(dim=0)

        # Reflected point
        reflected = centroid + self.alpha * (centroid - worst)
        # Expanded point
        expanded = centroid + self.gamma * (reflected - centroid)
        # Outside contraction
        contracted_out = centroid + self.rho * (reflected - centroid)
        # Inside contraction
        contracted_in = centroid - self.rho * (centroid - worst)

        candidates = torch.stack([reflected, expanded, contracted_out, contracted_in])
        candidates = torch.clamp(candidates, self.lb, self.ub)

        self._candidates[:] = candidates
        return candidates

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------
    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Report fitness values and update the simplex.

        Parameters
        ----------
        candidates : torch.Tensor
            The exact tensor returned by the most recent ``ask()``.
        fitness : torch.Tensor
            Objective values for each candidate (lower is better).

        """
        if self._needs_full_eval:
            # Store initial (or post-shrink) simplex evaluation
            self.population[:] = candidates[: self.pop_size]
            self.fitness[:] = fitness[: self.pop_size]
            self._needs_full_eval = False
            self._update_best(candidates[: self.pop_size], fitness[: self.pop_size])
            self._generation += 1
            return

        # Unpack the 4 candidate fitness values
        f_r = fitness[0]
        f_e = fitness[1]
        f_co = fitness[2]
        f_ci = fitness[3]

        x_r = candidates[0]
        x_e = candidates[1]
        x_co = candidates[2]
        x_ci = candidates[3]

        f_best = self.fitness[0]
        f_second_worst = self.fitness[-2]
        f_worst = self.fitness[-1]

        # ----------------------------------------------------------
        # Determine which NM operation applies
        # ----------------------------------------------------------
        use_expansion = f_r < f_best
        accept_reflect = (f_best <= f_r) & (f_r < f_second_worst)
        try_outside = ~accept_reflect & ~use_expansion & (f_r < f_worst)
        try_inside = ~accept_reflect & ~use_expansion & ~try_outside

        # Expansion: pick the better of expanded vs reflected
        expansion_choice = torch.where(f_e < f_r, x_e, x_r)
        expansion_f = torch.where(f_e < f_r, f_e, f_r)

        # Contraction acceptance checks
        outside_ok = f_co <= f_r
        inside_ok = f_ci < f_worst

        # Shrink required when the chosen contraction failed
        need_shrink = (try_outside & ~outside_ok) | (try_inside & ~inside_ok)

        if need_shrink.item():
            best_vertex = self.population[0].clone()
            self.population[1:] = best_vertex + self.shrink_coeff * (
                self.population[1:] - best_vertex
            )
            self.population[:] = torch.clamp(self.population, self.lb, self.ub)
            self._needs_full_eval = True
        else:
            # Select replacement for the worst vertex
            new_x = torch.where(
                use_expansion,
                expansion_choice,
                torch.where(
                    accept_reflect,
                    x_r,
                    torch.where(try_outside, x_co, x_ci),
                ),
            )
            new_f = torch.where(
                use_expansion,
                expansion_f,
                torch.where(
                    accept_reflect,
                    f_r,
                    torch.where(try_outside, f_co, f_ci),
                ),
            )

            self.population[-1] = new_x
            self.fitness[-1] = new_f

        self._update_best(self.population, self.fitness)
        self._generation += 1
