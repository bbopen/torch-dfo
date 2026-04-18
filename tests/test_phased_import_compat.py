"""Compatibility guard for the phased package layout.

The phased DFO orchestrator was converted from a single module
(``src/torch_dfo/phased.py``) into a package (``src/torch_dfo/phased/``).
These tests pin the re-export invariants so a future refactor that breaks
an import path fails loudly.
"""

from __future__ import annotations

import pytest

import torch_dfo
from torch_dfo import phased as phased_pkg
from torch_dfo.phased import PhasedDFO
from torch_dfo.phased.orchestrator import PhasedDFO as PhasedDFOOrchestrator


def test_phased_package_reexports_are_identity_preserving() -> None:
    assert PhasedDFO is PhasedDFOOrchestrator
    assert torch_dfo.PhasedDFO is PhasedDFO


@pytest.mark.parametrize("name", phased_pkg.__all__)
def test_every_public_name_resolves(name: str) -> None:
    """Every entry in ``torch_dfo.phased.__all__`` must resolve to a real object.

    Catches the failure mode where a refactor drops a re-export but leaves
    the ``__all__`` list untouched — ``from torch_dfo.phased import X`` would
    start raising ImportError for existing callers.
    """
    assert hasattr(phased_pkg, name), (
        f"'{name}' is in torch_dfo.phased.__all__ but missing from the module"
    )
    obj = getattr(phased_pkg, name)
    assert obj is not None, f"torch_dfo.phased.{name} resolved to None"
