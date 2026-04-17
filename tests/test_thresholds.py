"""Invariants on the shared thresholds module.

The module itself is trivial (constants), but we pin a few invariants that
would regress silently if someone reordered or mistyped a value.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests import _thresholds as th


def test_f32_tolerances_are_looser_than_f64() -> None:
    assert th.ATOL_F32_DEFAULT > th.ATOL_F64_DEFAULT
    assert th.ATOL_F32_TIGHT > th.ATOL_F64_TIGHT
    assert th.TOL_CMAES_C_SYMMETRY_F32 > th.TOL_CMAES_C_SYMMETRY_F64
    assert th.TOL_CMAES_INVSQRT_IDENTITY_F32 > th.TOL_CMAES_INVSQRT_IDENTITY_F64
    assert th.TOL_CMAES_BEST_F32 > th.TOL_CMAES_BEST_F64


def test_atol_for_routes_by_dtype() -> None:
    assert th.atol_for(torch.float32) == th.ATOL_F32_DEFAULT
    assert th.atol_for(torch.float64) == th.ATOL_F64_DEFAULT


def test_budgets_are_ordered() -> None:
    assert th.BUDGET_PHASED_MICRO < th.BUDGET_PHASED_POLISH
    assert th.BUDGET_PHASED_POLISH < th.BUDGET_PHASED_QUICK
    assert th.BUDGET_PHASED_QUICK < th.BUDGET_PHASED_STANDARD
    assert th.BUDGET_PHASED_STANDARD < th.BUDGET_PHASED_LARGE


def test_convergence_targets_are_positive_and_finite() -> None:
    convs = [getattr(th, n) for n in dir(th) if n.startswith("CONV_")]
    assert len(convs) > 0
    for v in convs:
        assert v > 0 and math.isfinite(v)


def test_smoke_ceilings_are_looser_than_convergence_targets() -> None:
    # A smoke ceiling should never be tighter than the tightest convergence
    # target for the same family — otherwise it's mislabeled.
    assert th.SMOKE_F_INIT_CMAES > th.CONV_SPHERE_10D_TIGHT
    assert th.BEST_F_MPS_SMOKE_CEIL > th.CONV_SPHERE_10D_STANDARD


@pytest.mark.parametrize(
    "name",
    ["ATOL_F64_DEFAULT", "CONV_SPHERE_10D_TIGHT", "BUDGET_PHASED_STANDARD"],
)
def test_exports_present(name: str) -> None:
    assert hasattr(th, name)
