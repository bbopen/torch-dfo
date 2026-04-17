"""Typed mixed hyperparameter search space with encode/decode support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Float:
    """A continuous floating-point hyperparameter."""

    name: str
    lower: float
    upper: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError(
                f"Float '{self.name}': upper ({self.upper}) must be > lower ({self.lower})",
            )
        if self.log and self.lower <= 0:
            raise ValueError(f"Float '{self.name}': log=True requires lower > 0, got {self.lower}")


@dataclass
class Int:
    """An integer hyperparameter (encoded as float, decoded by rounding)."""

    name: str
    lower: int
    upper: int
    log: bool = False

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(
                f"Int '{self.name}': upper ({self.upper}) must be >= lower ({self.lower})",
            )
        if self.log and self.lower <= 0:
            raise ValueError(f"Int '{self.name}': log=True requires lower > 0, got {self.lower}")


@dataclass
class Categorical:
    """A categorical hyperparameter with a fixed list of choices."""

    name: str
    choices: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"Categorical '{self.name}': choices must not be empty")


# Type alias for any parameter type
Param = Float | Int | Categorical


class SearchSpace:
    """Mixed hyperparameter search space with encode/decode support.

    Each parameter occupies exactly one dimension in the encoded tensor,
    always normalised to ``[0, 1]``.
    """

    def __init__(self, params: list[Float | Int | Categorical]) -> None:
        if not params:
            raise ValueError("SearchSpace requires at least one parameter")
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate parameter names in SearchSpace")
        self._params: list[Float | Int | Categorical] = list(params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of encoded dimensions (one per parameter)."""
        return len(self._params)

    @property
    def params(self) -> list[Float | Int | Categorical]:
        """Copy of the parameter list (mutating the copy is safe)."""
        return list(self._params)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        configs: list[dict[str, Any]],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Encode a list of config dicts into a ``(N, dim)`` float tensor.

        All output values are clipped to ``[0, 1]``.  Raises ``ValueError``
        for log-scale parameters whose lower bound is ``<= 0``, or for
        unknown categorical values.
        """
        rows: list[list[float]] = []
        for i, cfg in enumerate(configs):
            row: list[float] = []
            for p in self._params:
                if p.name not in cfg:
                    raise KeyError(f"Config at index {i} is missing required parameter '{p.name}'")
                v = cfg[p.name]
                row.append(self._encode_one(p, v))
            rows.append(row)

        t = torch.tensor(rows, dtype=dtype, device=device)
        return t.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, x: torch.Tensor) -> list[dict[str, Any]]:
        """Decode a ``(N, dim)`` tensor into a list of config dicts.

        A ``(dim,)`` 1-D tensor is promoted to ``(1, dim)`` first.
        All encoded values are clipped to ``[0, 1]`` before decoding.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.clamp(0.0, 1.0)
        results: list[dict[str, Any]] = []
        for i in range(x.shape[0]):
            cfg: dict[str, Any] = {}
            for j, p in enumerate(self._params):
                u = float(x[i, j].item())
                cfg[p.name] = self._decode_one(p, u)
            results.append(cfg)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_one(p: Float | Int | Categorical, v: Any) -> float:
        """Encode a single value for parameter *p* to ``[0, 1]``."""
        if isinstance(p, (Float, Int)):
            lo = float(p.lower)
            hi = float(p.upper)
            val = float(v)
            if p.log:
                if lo <= 0.0:
                    raise ValueError(
                        f"Parameter '{p.name}': log=True requires lower > 0, got lower={lo}",
                    )
                return (math.log(val) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return (val - lo) / (hi - lo)
        # Categorical
        choices = p.choices
        # .index() raises ValueError for unknown values — desired behaviour
        idx = choices.index(v)
        # Categorical encodes to [0, (n-1)/n], never reaching 1.0.
        # This is intentional: u=1.0 is decoded to the last choice via clamp,
        # so there is no gap, but encoding avoids emitting 1.0 to prevent
        # the decode floor() from indexing out of bounds.
        return idx / len(choices)

    @staticmethod
    def _decode_one(p: Float | Int | Categorical, u: float) -> Any:
        """Decode a single ``[0, 1]`` value for parameter *p*."""
        if isinstance(p, (Float, Int)):
            lo = float(p.lower)
            hi = float(p.upper)
            if p.log:
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            else:
                val = lo + u * (hi - lo)
            if isinstance(p, Int):
                return round(val)
            return val
        # Categorical
        choices = p.choices
        idx = min(math.floor(u * len(choices)), len(choices) - 1)
        return choices[idx]
