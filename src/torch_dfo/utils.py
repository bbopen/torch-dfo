"""Shared utilities for torch-dfo: device resolution, RNG, bounds handling."""

from __future__ import annotations

import threading
import warnings

import torch

_nan_warn_lock = threading.Lock()
_NAN_WARNED = False


def sanitize_fitness(fitness: torch.Tensor) -> torch.Tensor:
    """Sanitize fitness values: replace NaN, handle Inf.

    Contract:
    - NaN  -> replaced with worst_seen + 1.0 (warning on first occurrence)
    - +Inf -> kept as valid but dominated
    - -Inf -> raises ValueError

    Thread-safe: the "warn once" flag uses a lock.
    """
    global _NAN_WARNED
    if (fitness == float("-inf")).any():
        raise ValueError("Fitness contains negative infinity — likely a bug in fitness function.")
    nan_mask = torch.isnan(fitness)
    if nan_mask.any():
        with _nan_warn_lock:
            if not _NAN_WARNED:
                warnings.warn(
                    "NaN fitness values detected; replacing with worst_seen + 1.0",
                    stacklevel=2,
                )
                _NAN_WARNED = True
        finite = fitness[~nan_mask]
        worst = finite.max() if finite.numel() > 0 else torch.tensor(0.0, device=fitness.device)
        fitness = fitness.clone()
        fitness[nan_mask] = worst + 1.0
    return fitness


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return the best available device if *device* is ``None``.

    Priority: user-specified > CUDA > MPS > CPU.
    Accepts any valid ``torch.device`` string or object.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_generator(seed: int | None, device: torch.device) -> torch.Generator:
    """Create a seeded ``torch.Generator`` on *device*.

    ``torch.Generator`` only supports CPU and CUDA natively.  For other
    backends (MPS, XLA, ...) we fall back to a CPU generator so callers can
    still obtain reproducible random tensors (generate on CPU, then ``.to()``).
    """
    gen_device = device if device.type in ("cpu", "cuda") else torch.device("cpu")
    gen = torch.Generator(device=gen_device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def clamp_to_bounds(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Broadcast-safe clamping to bounds.

    *lb* / *ub* are ``(dim,)`` tensors from :func:`normalize_bounds`.
    """
    return torch.clamp(x, min=lb, max=ub)


def normalize_bounds(
    bounds: float | tuple[float, float] | tuple[torch.Tensor, torch.Tensor] | list[float],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert scalar / tuple bounds to a pair of ``(dim,)`` tensors.

    Args:
        bounds: One of:
            - a scalar ``b``  -> ``lb = -|b|, ub = |b|``
            - a 2-tuple of scalars ``(lo, hi)``
            - a 2-tuple of tensors already shaped ``(dim,)``
        dim: Problem dimensionality.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(lb, ub)`` each of shape ``(dim,)`` on *device* with *dtype*.

    Raises:
        ValueError: If *bounds* has an unsupported type, or if any dimension
            has a non-positive span (``ub <= lb``).

    """
    if isinstance(bounds, (int, float)):
        if bounds == 0:
            raise ValueError(
                "bounds must have positive span on every dimension; "
                "got scalar bounds=0 which yields lb == ub == 0"
            )
        lb = torch.full((dim,), -abs(bounds), device=device, dtype=dtype)
        ub = torch.full((dim,), abs(bounds), device=device, dtype=dtype)
    elif isinstance(bounds, (tuple, list)):
        lb_val, ub_val = bounds
        if isinstance(lb_val, torch.Tensor) and isinstance(ub_val, torch.Tensor):
            lb = lb_val.to(device=device, dtype=dtype)
            ub = ub_val.to(device=device, dtype=dtype)
        else:
            lb = torch.full((dim,), float(lb_val), device=device, dtype=dtype)
            ub = torch.full((dim,), float(ub_val), device=device, dtype=dtype)
    else:
        msg = f"Unsupported bounds type: {type(bounds)}"
        raise ValueError(msg)
    if not torch.all(ub > lb):
        raise ValueError(
            "bounds must have positive span on every dimension; "
            "got at least one index where ub <= lb"
        )
    return lb, ub
