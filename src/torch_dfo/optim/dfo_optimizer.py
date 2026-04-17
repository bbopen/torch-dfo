"""torch.optim.Optimizer wrapper for derivative-free optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class DFOOptimizer(torch.optim.Optimizer):
    """Derivative-free optimizer with torch.optim.Optimizer interface.

    Unlike gradient-based optimizers, each step() call evaluates the closure
    pop_size times (once per candidate solution). This is the fundamental
    cost model of derivative-free optimization.

    Supports two closure modes:

    - **Sequential**: ``closure()`` takes no args, returns scalar loss.
      Called pop_size times per step. Simple but slow.
    - **Batched**: ``closure_batched(candidates)`` takes ``(pop_size, dim)``
      tensor, returns ``(pop_size,)`` losses. Fast for GPU workloads.

    Parameters
    ----------
    params :
        Model parameters to optimize (iterable of :class:`torch.Tensor`
        or list of param-group dicts).
    algorithm : str
        ``'shade'`` (default), ``'cmaes'``, or ``'nelder_mead'``.
        ``'phased'`` is not supported: it drives its own ``optimize()``
        loop and does not fit the ``torch.optim.Optimizer.step()`` model.
    budget : int | None
        Total function evaluations.  Default: ``dim * 5000``.
    bounds : float | tuple[float, float]
        Search bounds.  **Required** -- DFO needs explicit bounds.
    **kwargs :
        Forwarded to the underlying optimizer constructor
        (e.g. ``pop_size``, ``sigma0``, ``seed``).
    """

    def __init__(
        self,
        params: Any,
        algorithm: str = "shade",
        budget: int | None = None,
        bounds: float | tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        if bounds is None:
            raise ValueError(
                "bounds is required -- DFO needs explicit search bounds. "
                "Use bounds=(-1, 1) or similar based on your parameter scale."
            )

        # Collect all parameter tensors ---------------------------------
        param_list: list[torch.Tensor] = list(params)
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            flat_params: list[torch.Tensor] = []
            for group in param_list:
                flat_params.extend(group["params"])
            param_list = flat_params

        if len(param_list) == 0:
            raise ValueError("DFOOptimizer received an empty parameter list.")

        self._param_shapes = [p.shape for p in param_list]
        self._param_sizes = [p.numel() for p in param_list]
        self._dim = sum(self._param_sizes)
        self._params = param_list

        device = param_list[0].device
        dtype = param_list[0].dtype
        # Validate all params share the same device and dtype
        for i, p in enumerate(param_list[1:], 1):
            if p.device != device:
                raise ValueError(
                    f"All parameters must be on the same device. "
                    f"param[0] is on {device}, param[{i}] is on {p.device}"
                )
            if p.dtype != dtype:
                raise ValueError(
                    f"All parameters must have the same dtype. "
                    f"param[0] has {dtype}, param[{i}] has {p.dtype}"
                )

        if budget is None:
            budget = self._dim * 5000
        self._budget = budget
        self._evals = 0

        # Build the underlying ask/tell optimizer -----------------------
        from torch_dfo import CMAES, SHADE, NelderMead

        algo_map = {
            "shade": SHADE,
            "cmaes": CMAES,
            "nelder_mead": NelderMead,
        }
        if algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(algo_map)}")

        algo_cls = algo_map[algorithm]
        self._inner = algo_cls(
            dim=self._dim,
            bounds=bounds,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        defaults: dict[str, Any] = {
            "algorithm": algorithm,
            "bounds": bounds,
            "budget": budget,
        }
        super().__init__(param_list, defaults)

    # ------------------------------------------------------------------
    # Parameter <-> flat vector helpers
    # ------------------------------------------------------------------

    def _flatten_params(self) -> torch.Tensor:
        """Flatten model parameters into a 1-D vector."""
        return torch.cat([p.data.reshape(-1) for p in self._params])

    def _set_params(self, flat: torch.Tensor) -> None:
        """Set model parameters from a 1-D vector."""
        offset = 0
        for p, shape, size in zip(self._params, self._param_shapes, self._param_sizes, strict=True):
            p.data.copy_(flat[offset : offset + size].reshape(shape))
            offset += size

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        closure_batched: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one optimization step (ask/tell cycle).

        Exactly one of *closure* or *closure_batched* must be provided.

        Parameters
        ----------
        closure : callable, optional
            Sequential closure -- takes no arguments, returns a scalar loss.
            Model parameters are set to each candidate in turn.
        closure_batched : callable, optional
            Batched closure -- takes a ``(pop_size, dim)`` tensor of candidate
            solutions and returns a ``(pop_size,)`` tensor of losses.

        Returns
        -------
        torch.Tensor
            Scalar loss of the best candidate in this generation.
        """
        if closure is None and closure_batched is None:
            raise ValueError("Provide closure or closure_batched to step().")
        if self.is_exhausted:
            raise RuntimeError(
                f"DFOOptimizer budget exhausted ({self._evals}/{self._budget} evals). "
                "Guard calls to step() with `if not opt.is_exhausted:` or raise the budget."
            )

        candidates = self._inner.ask()
        pop_size = candidates.shape[0]

        if closure_batched is not None:
            fitness = closure_batched(candidates)
        else:
            assert closure is not None
            losses: list[torch.Tensor] = []
            original_params = self._flatten_params().clone()
            for i in range(pop_size):
                self._set_params(candidates[i])
                loss = closure()
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=candidates.device, dtype=candidates.dtype)
                losses.append(loss.detach())
            fitness = torch.stack(losses).to(device=candidates.device, dtype=candidates.dtype)
            # Restore before tell so _set_params to best below is clean
            self._set_params(original_params)

        self._inner.tell(candidates, fitness)
        self._evals += pop_size

        # Set model parameters to the best solution found so far
        self._set_params(self._inner.best_solution)
        return fitness.min()

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    @property
    def budget_remaining(self) -> int:
        """Number of function evaluations remaining."""
        return max(0, self._budget - self._evals)

    @property
    def is_exhausted(self) -> bool:
        """True when the optimization budget is fully consumed."""
        return self._evals >= self._budget

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Return optimizer state including the wrapped inner DFO optimizer.

        Extends :class:`torch.optim.Optimizer.state_dict` with the keys
        required to restore a ``DFOOptimizer``: the inner optimizer's own
        ``state_dict`` and the current evaluation counter.
        """
        state = super().state_dict()
        state["_inner"] = self._inner.state_dict()
        state["_evals"] = self._evals
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        """Restore optimizer state produced by :meth:`state_dict`.

        The ``_inner`` and ``_evals`` extensions are consumed here; the
        remaining keys are forwarded to the base ``torch.optim.Optimizer``.
        """
        inner_state = state_dict.pop("_inner", None)
        evals = state_dict.pop("_evals", None)
        super().load_state_dict(state_dict)
        if inner_state is not None:
            self._inner.load_state_dict(inner_state)
        if evals is not None:
            self._evals = int(evals)
