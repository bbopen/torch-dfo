"""Tests that every public API symbol in ``torch_dfo.__all__`` is importable."""

from __future__ import annotations

import pytest

import torch_dfo


def test_all_is_non_empty() -> None:
    assert len(torch_dfo.__all__) > 0, "__all__ is empty; no public API declared"


@pytest.mark.parametrize("name", torch_dfo.__all__)
def test_public_symbol_exists(name: str) -> None:
    """Every name in __all__ resolves to a real attribute on the package."""
    assert hasattr(torch_dfo, name), (
        f"torch_dfo.__all__ advertises {name!r} but the attribute is missing"
    )


def test_all_covers_advertised_api() -> None:
    """Spot-check that the canonical entry points are all in __all__.

    If one of these ever disappears from __all__ we want to know, even
    though the parametrized test above would also catch most mistakes.
    """
    expected_minimum = {
        "CMAES",
        "SHADE",
        "NelderMead",
        "PhasedDFO",
        "DLRPortfolio",
        "DFOOptimizer",
        "SearchSpace",
        "Float",
        "Int",
        "Categorical",
        "sphere",
        "rosenbrock",
        "BenchmarkSuite",
    }
    missing = expected_minimum - set(torch_dfo.__all__)
    assert not missing, f"__all__ is missing canonical exports: {missing}"
