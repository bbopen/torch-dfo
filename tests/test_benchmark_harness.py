"""Tests for the benchmark harness (benchmarks/run_benchmarks.py) utilities."""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helper: detect optional dependencies
# ---------------------------------------------------------------------------


def _yahpo_available() -> bool:
    return importlib.util.find_spec("yahpo_gym") is not None


def _configspace_available() -> bool:
    return importlib.util.find_spec("ConfigSpace") is not None


# ---------------------------------------------------------------------------
# Sentinel tests — always run, no optional deps needed
# ---------------------------------------------------------------------------


def test_integer_hp_rounding_sentinel_defined() -> None:
    """The integer HP type sentinel is defined as a tuple (may be empty on non-YAHPO systems)."""
    from benchmarks.run_benchmarks import _CS_INT_HP_TYPES

    assert isinstance(_CS_INT_HP_TYPES, tuple), (
        f"_CS_INT_HP_TYPES should be a tuple, got {type(_CS_INT_HP_TYPES)}"
    )


def test_integer_hp_rounding_sentinel_nonempty_when_configspace_available() -> None:
    """When ConfigSpace is importable the sentinel must contain at least one type."""
    if not _configspace_available():
        pytest.skip("ConfigSpace not installed")

    from benchmarks.run_benchmarks import _CS_INT_HP_TYPES

    assert len(_CS_INT_HP_TYPES) > 0, (
        "ConfigSpace is available but _CS_INT_HP_TYPES is empty — "
        "the lazy import block failed silently"
    )
    for tp in _CS_INT_HP_TYPES:
        assert isinstance(tp, type), f"Every entry must be a type, got {tp!r}"


# ---------------------------------------------------------------------------
# Integration test: round-trip produces int for integer HPs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_yahpo_available() and _configspace_available()),
    reason="yahpo_gym + ConfigSpace not available",
)
def test_tensor_to_yahpo_config_rounds_integers() -> None:
    """Integer HPs decoded via _tensor_to_yahpo_config must be Python ints."""
    import torch
    from ConfigSpace.hyperparameters import (
        UniformIntegerHyperparameter,
    )

    from benchmarks.run_benchmarks import _tensor_to_yahpo_config

    # Build a minimal ConfigSpace with one integer HP and one float HP
    try:
        import ConfigSpace as CS
    except ImportError:
        pytest.skip("ConfigSpace not importable")

    cs = CS.ConfigurationSpace(seed=0)
    cs.add_hyperparameter(
        UniformIntegerHyperparameter("n_layers", lower=1, upper=10, default_value=5)
    )
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=1e-5, upper=1e-1))

    # Use a tensor that would produce a non-integer decoded value (e.g. 0.37)
    x = torch.tensor([0.37, 0.5], dtype=torch.float64)
    config = _tensor_to_yahpo_config(x, cs)

    n_layers_val = config["n_layers"]
    lr_val = config["lr"]

    assert isinstance(n_layers_val, int), (
        f"n_layers (UniformIntegerHyperparameter) should decode to int, "
        f"got {type(n_layers_val).__name__} = {n_layers_val!r}"
    )
    # Float HP should remain float
    assert isinstance(lr_val, float), (
        f"lr (UniformFloatHyperparameter) should decode to float, "
        f"got {type(lr_val).__name__} = {lr_val!r}"
    )

    # Value must be in the valid integer range
    assert 1 <= n_layers_val <= 10, f"n_layers out of range: {n_layers_val}"
