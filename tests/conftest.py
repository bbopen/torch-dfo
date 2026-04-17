"""Shared pytest fixtures and configuration for torch-dfo tests."""

from __future__ import annotations

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Force a specific torch device for all tests (e.g. cpu, cuda, mps)",
    )


def get_available_devices() -> list[str]:
    """Return a list of torch device strings available on the current machine."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=get_available_devices())
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrised fixture that yields every available device.

    When ``--device`` is passed on the CLI the fixture returns only that
    device, regardless of what is available.
    """
    forced = request.config.getoption("--device")
    if forced is not None:
        return torch.device(forced)
    return torch.device(request.param)


@pytest.fixture
def seed() -> int:
    """Default reproducibility seed."""
    return 42


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
_FLOAT64_UNSUPPORTED_BACKENDS: set[str] = {"mps"}


def best_float_dtype(device: torch.device) -> torch.dtype:
    """Return the highest-precision float dtype supported by *device*.

    MPS does not support float64, so we fall back to float32.  CUDA, CPU, XLA,
    and other backends get float64.
    """
    if device.type in _FLOAT64_UNSUPPORTED_BACKENDS:
        return torch.float32
    return torch.float64


@pytest.fixture
def default_dtype(device: torch.device) -> torch.dtype:
    """Per-device best available float dtype (float64 where supported)."""
    return best_float_dtype(device)


# ---------------------------------------------------------------------------
# Shared tolerance constants (kept as aliases for any external consumer;
# authoritative definitions live in tests/_thresholds.py).
# ---------------------------------------------------------------------------
from tests._thresholds import ATOL_F32_DEFAULT as ATOL_FLOAT32  # noqa: E402, F401
from tests._thresholds import ATOL_F64_DEFAULT as ATOL_FLOAT64  # noqa: E402, F401
from tests._thresholds import RTOL_DEFAULT  # noqa: E402, F401
