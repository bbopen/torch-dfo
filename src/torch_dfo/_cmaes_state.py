"""State containers for CMA-ES internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch_dfo._state_utils import clone_tensor, restore_tensor


@dataclass
class CMAPathState:
    """Mutable evolution-path and path-sampling state for CMA-ES."""

    p_sigma: torch.Tensor
    p_c: torch.Tensor
    path_memory: int
    path_scale: float
    path_line_samples: int
    path_line_scale: float
    _path_vectors: torch.Tensor
    _path_count: int
    _path_pos: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copy state dictionary for this path block."""
        return {
            "p_sigma": clone_tensor(self.p_sigma),
            "p_c": clone_tensor(self.p_c),
            "path_memory": self.path_memory,
            "path_scale": self.path_scale,
            "path_line_samples": self.path_line_samples,
            "path_line_scale": self.path_line_scale,
            "_path_vectors": clone_tensor(self._path_vectors),
            "_path_count": self._path_count,
            "_path_pos": self._path_pos,
        }

    @classmethod
    def from_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> CMAPathState:
        """Restore path state from :meth:`to_dict` output."""
        return cls(
            p_sigma=restore_tensor(state["p_sigma"], device=device, dtype=dtype),
            p_c=restore_tensor(state["p_c"], device=device, dtype=dtype),
            path_memory=int(state["path_memory"]),
            path_scale=float(state["path_scale"]),
            path_line_samples=int(state["path_line_samples"]),
            path_line_scale=float(state["path_line_scale"]),
            _path_vectors=restore_tensor(state["_path_vectors"], device=device, dtype=dtype),
            _path_count=int(state["_path_count"]),
            _path_pos=int(state["_path_pos"]),
        )


@dataclass(frozen=True)
class CMAAdaptationRates:
    """Hansen adaptation rates computed from population size and dimension."""

    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    mu_eff: float
    weights: torch.Tensor
