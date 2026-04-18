"""State containers for SHADE internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torch_dfo._state_utils import clone_tensor, restore_tensor


@dataclass
class SHADEMemory:
    """Mutable success-history and trial buffers for SHADE."""

    memory_F: torch.Tensor
    memory_CR: torch.Tensor
    _memory_pos: int
    _archive: torch.Tensor
    _trial_F: torch.Tensor
    _trial_CR: torch.Tensor
    _trials: torch.Tensor
    _initialized: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copy state dictionary for this memory block."""
        return {
            "memory_F": clone_tensor(self.memory_F),
            "memory_CR": clone_tensor(self.memory_CR),
            "_memory_pos": self._memory_pos,
            "_archive": clone_tensor(self._archive),
            "_trial_F": clone_tensor(self._trial_F),
            "_trial_CR": clone_tensor(self._trial_CR),
            "_trials": clone_tensor(self._trials),
            "_initialized": self._initialized,
        }

    @classmethod
    def from_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> SHADEMemory:
        """Restore memory from :meth:`to_dict` output."""
        return cls(
            memory_F=restore_tensor(state["memory_F"], device=device, dtype=dtype),
            memory_CR=restore_tensor(state["memory_CR"], device=device, dtype=dtype),
            _memory_pos=int(state["_memory_pos"]),
            _archive=restore_tensor(state["_archive"], device=device, dtype=dtype),
            _trial_F=restore_tensor(state["_trial_F"], device=device, dtype=dtype),
            _trial_CR=restore_tensor(state["_trial_CR"], device=device, dtype=dtype),
            _trials=restore_tensor(state["_trials"], device=device, dtype=dtype),
            _initialized=bool(state["_initialized"]),
        )
