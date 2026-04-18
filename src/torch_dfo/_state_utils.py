"""Small helpers for optimizer state clone/restore paths."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()


def clone_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return tensor.clone() if tensor is not None else None


def clone_tensor_list(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in tensors]


def restore_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype).clone()


def restore_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor | None:
    return restore_tensor(tensor, device=device, dtype=dtype) if tensor is not None else None


def restore_tensor_list(
    tensors: Sequence[torch.Tensor],
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> list[torch.Tensor]:
    return [restore_tensor(tensor, device=device, dtype=dtype) for tensor in tensors]
