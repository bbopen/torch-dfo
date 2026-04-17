"""Shared threshold constants for the torch-dfo test suite.

Numeric tolerances, convergence targets, smoke ceilings, evaluation
budgets, and population sizes used across the tests. Centralising them
keeps tolerance scale visible in one place and lets future tightening
land in a single diff.

Categories:

- Tolerances: dtype-aware ``atol`` / ``rtol`` for ``allclose`` and
  ``pytest.approx``.
- Convergence targets: per-problem ``best_f`` ceilings (``CONV_*``).
- Smoke ceilings: loose bounds for did-it-run tests (``SMOKE_*``).
- Budgets: standard FE budgets (``BUDGET_*``).
- Populations: standard ``pop_size`` values (``POP_*``).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Dtype-aware tolerances
# ---------------------------------------------------------------------------
# The test suite has accumulated two float64 tolerance values (1e-10 and
# 1e-12) and two float32 values (1e-5 and 1e-6). These correspond to
# two different numerical tightness regimes used by different test families.

ATOL_F64_DEFAULT: float = 1e-10
ATOL_F64_TIGHT: float = 1e-12
ATOL_F32_DEFAULT: float = 1e-5
ATOL_F32_TIGHT: float = 1e-6
RTOL_DEFAULT: float = 1e-5
RTOL_TIGHT: float = 1e-10

# Dynamic tolerances already used inline in test_cmaes.py (lines 291-292,
# 305-306, 925-926): float32 gets looser, float64 gets tighter.
TOL_CMAES_C_SYMMETRY_F64: float = 1e-12
TOL_CMAES_C_SYMMETRY_F32: float = 1e-6
TOL_CMAES_INVSQRT_IDENTITY_F64: float = 1e-10
TOL_CMAES_INVSQRT_IDENTITY_F32: float = 1e-4
TOL_CMAES_BEST_F64: float = 1e-10
TOL_CMAES_BEST_F32: float = 1e-4


def atol_for(dtype: torch.dtype) -> float:
    """Return the default ``allclose`` atol for *dtype*.

    Mirrors the existing ``tests/conftest.py::best_float_dtype`` pairing:
    float32 gets ``ATOL_F32_DEFAULT`` (1e-5), float64 gets
    ``ATOL_F64_DEFAULT`` (1e-10).
    """
    if dtype == torch.float32:
        return ATOL_F32_DEFAULT
    return ATOL_F64_DEFAULT


# ---------------------------------------------------------------------------
# Convergence targets — per-problem ceilings used in CONVERGENCE tests.
# These are the meaningful "algorithm actually found the minimum" asserts.
# ---------------------------------------------------------------------------

# Sphere family — separable quadratic, converges fast on every algorithm.
CONV_SPHERE_10D_TIGHT: float = 1e-8  # CMAES active on 200 evals, SHADE on 2000 evals
CONV_SPHERE_10D_STANDARD: float = 1e-6  # CMAES/SHADE standard tests
CONV_SPHERE_5D_FAST: float = 1e-8  # SHADE 5d
CONV_SPHERE_1D: float = 1e-6  # CMAES dim=1 edge case
CONV_SPHERE_HIGH_DIM: float = 1e-10  # CMAES high-dim

# Ackley family — basin-of-attraction multimodal, easier than rastrigin.
CONV_ACKLEY_10D: float = 1e-4

# Rastrigin family — many local minima, algorithms struggle.
CONV_RASTRIGIN_10D: float = 1.0  # PhasedDFO at budget=50k
CONV_CMAES_RASTRIGIN_10D: float = 2.5  # CMAES IPOP over 5 restarts

# Rosenbrock family — curved valley, hardest of the standard set.
CONV_ROSENBROCK_2D: float = 1e-4  # NelderMead
CONV_ROSENBROCK_10D: float = 1e-2  # PhasedDFO

# Polish sub-phase — short targeted optimization after CMAES.
CONV_POLISH_SPHERE_5D: float = 0.1

# DFOOptimizer wrapper — convergence via torch.optim interface.
CONV_DFO_XOR_LOSS: float = 0.4  # MLP on XOR via SHADE
CONV_DFO_QUADRATIC_NORM: float = 1.0  # params norm after opt
CONV_DFO_CMAES_4D: float = 5.0  # loss ceiling
CONV_DFO_NELDER_MEAD_3D: float = 5.0  # loss ceiling
CONV_SPACE_SIMPLE_CLOSURE: float = 0.5  # SearchSpace-based opt

# Nelder-Mead 1D edge case.
CONV_NELDER_MEAD_1D: float = 0.01

# BaseOptimizer smoke on sphere.
CONV_BASE_WITH_BOUNDS: float = 5.0


# ---------------------------------------------------------------------------
# Smoke ceilings — bounds used in "did it run without exploding" tests, NOT
# convergence proofs. Tightened to force tests to prove real improvement
# from the initial random population rather than merely verifying no-explode.
# Rationale: 5d sphere in [-5,5] has theoretical max sum(25)*5 = 125; any
# optimizer making forward progress should beat 50 within a few generations.
# ---------------------------------------------------------------------------

SMOKE_F_INIT_CMAES: float = 50.0  # best_f after init, pre-opt
SMOKE_F_INIT_SHADE: float = 50.0  # SHADE init sanity (well below 5d-sphere max of 125)
SMOKE_F_MULTIMODAL_IMPROVEMENT: float = 100.0  # SHADE multimodal improvement
BEST_F_MPS_SMOKE_CEIL: float = 200.0  # MPS smoke — only verifies no-crash


# ---------------------------------------------------------------------------
# Budgets — FE budgets used across test configurations.
# ---------------------------------------------------------------------------

BUDGET_SMOKE: int = 5_000  # MPS smoke tests
BUDGET_STANDARD: int = 10_000  # XOR and small problems
BUDGET_DFO_BATCHED: int = 5_000
BUDGET_DFO_CMAES: int = 2_000
BUDGET_PHASED_STANDARD: int = 50_000  # PhasedDFO convergence tests
BUDGET_PHASED_LARGE: int = 100_000  # PhasedDFO large-scale
BUDGET_PHASED_MEDIUM: int = 20_000
BUDGET_PHASED_QUICK: int = 10_000
BUDGET_PHASED_POLISH: int = 3_000
BUDGET_PHASED_MICRO: int = 1_000


# ---------------------------------------------------------------------------
# Population sizes — standard pop_size values.
# ---------------------------------------------------------------------------

POP_DFO_XOR: int = 30
POP_DFO_QUADRATIC: int = 20
POP_DFO_DEFAULT: int = 10
POP_SHADE_STANDARD: int = 20
POP_CMAES_STANDARD: int = 12


# ---------------------------------------------------------------------------
# PhasedDFO default-budget multiplier (mirror of
# ``src/torch_dfo/phased.py::PhasedDFO.__init__(budget_mult=5000)``).
# Kept here so test_phased.py::test_default_budget imports the constant
# rather than duplicating the literal.
# ---------------------------------------------------------------------------

PHASED_DEFAULT_BUDGET_MULT: int = 5000
