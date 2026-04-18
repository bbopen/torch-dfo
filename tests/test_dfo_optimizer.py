"""Tests for DFOOptimizer torch.optim wrapper."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests._thresholds import (
    BUDGET_DFO_BATCHED,
    BUDGET_DFO_CMAES,
    BUDGET_STANDARD,
    CONV_DFO_CMAES_4D,
    CONV_DFO_NELDER_MEAD_3D,
    CONV_DFO_QUADRATIC_NORM,
    CONV_DFO_XOR_LOSS,
    POP_DFO_DEFAULT,
    POP_DFO_QUADRATIC,
    POP_DFO_XOR,
)
from torch_dfo.optim import DFOOptimizer

# ------------------------------------------------------------------
# Validation / error paths
# ------------------------------------------------------------------


def test_requires_bounds():
    """Omitting bounds raises a clear error."""
    with pytest.raises(ValueError, match="bounds is required"):
        DFOOptimizer([torch.zeros(5)])


def test_unknown_algorithm():
    """An unrecognised algorithm name raises."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        DFOOptimizer([torch.zeros(5)], algorithm="bogus", bounds=(-1, 1))


def test_no_closure_raises():
    """Calling step() without any closure raises."""
    params = [torch.zeros(3, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="shade", bounds=(-1, 1))
    with pytest.raises(ValueError, match="closure"):
        opt.step()


def test_empty_params_raises():
    """An empty parameter list raises."""
    with pytest.raises(ValueError, match="empty"):
        DFOOptimizer(iter([]), bounds=(-1, 1))


# ------------------------------------------------------------------
# Budget tracking
# ------------------------------------------------------------------


def test_budget_tracking():
    """budget_remaining and is_exhausted are updated correctly."""
    params = [torch.zeros(3, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="shade", bounds=(-1, 1), budget=100, pop_size=10)
    assert opt.budget_remaining == 100
    assert not opt.is_exhausted

    opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert opt.budget_remaining == 90

    for _ in range(9):
        opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert opt.is_exhausted
    assert opt.budget_remaining == 0


def test_default_budget():
    """Default budget is dim * 5000."""
    params = [torch.zeros(7, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="shade", bounds=(-1, 1))
    assert opt._budget == 7 * 5000


# ------------------------------------------------------------------
# Sequential closure (XOR)
# ------------------------------------------------------------------


def test_sequential_closure_xor():
    """Train a tiny MLP on XOR with sequential closure.

    Save and restore the global torch RNG state so this test's seeding does
    not leak into adjacent tests.  The model's internal parameter-init still
    needs a deterministic seed, so we set it and restore afterwards.
    """
    _rng_state_before = torch.random.get_rng_state()
    try:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        optimizer = DFOOptimizer(
            model.parameters(),
            algorithm="shade",
            bounds=(-3.0, 3.0),
            budget=BUDGET_STANDARD,
            pop_size=POP_DFO_XOR,
        )

        for _ in range(50):
            if optimizer.is_exhausted:
                break

            def closure():
                return nn.functional.mse_loss(model(X), y)

            optimizer.step(closure=closure)

        final_loss = nn.functional.mse_loss(model(X), y).item()
        assert final_loss < CONV_DFO_XOR_LOSS, f"DFO didn't learn XOR: loss={final_loss:.4f}"
    finally:
        torch.random.set_rng_state(_rng_state_before)


# ------------------------------------------------------------------
# Batched closure (quadratic)
# ------------------------------------------------------------------


def test_batched_closure_quadratic():
    """Batched closure optimises a simple quadratic."""
    params = [torch.zeros(5, dtype=torch.float64)]
    optimizer = DFOOptimizer(
        params,
        algorithm="shade",
        bounds=(-5.0, 5.0),
        budget=BUDGET_DFO_BATCHED,
        pop_size=POP_DFO_QUADRATIC,
    )

    def batched_closure(candidates: torch.Tensor) -> torch.Tensor:
        return (candidates**2).sum(dim=-1)

    for _ in range(20):
        if optimizer.is_exhausted:
            break
        optimizer.step(closure_batched=batched_closure)

    assert params[0].norm().item() < CONV_DFO_QUADRATIC_NORM, "Batched closure didn't optimise"


# ------------------------------------------------------------------
# Algorithm variants
# ------------------------------------------------------------------


def test_cmaes_algorithm():
    """CMAES backend actually optimizes a quadratic."""
    params = [torch.zeros(4, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="cmaes", bounds=(-5, 5), budget=BUDGET_DFO_CMAES)

    for _ in range(20):
        if opt.is_exhausted:
            break
        opt.step(closure_batched=lambda c: (c**2).sum(-1))

    final_loss = (params[0] ** 2).sum().item()
    assert final_loss < CONV_DFO_CMAES_4D, f"CMAES didn't optimize: loss={final_loss:.2f}"


def test_nelder_mead_algorithm():
    """NelderMead backend actually optimizes a quadratic."""
    params = [torch.zeros(3, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="nelder_mead", bounds=(-5, 5), budget=BUDGET_DFO_CMAES)

    for _ in range(100):
        if opt.is_exhausted:
            break
        opt.step(closure_batched=lambda c: (c**2).sum(-1))

    final_loss = (params[0] ** 2).sum().item()
    assert final_loss < CONV_DFO_NELDER_MEAD_3D, (
        f"NelderMead didn't optimize: loss={final_loss:.2f}"
    )


# ------------------------------------------------------------------
# Param-group dict format
# ------------------------------------------------------------------


def test_param_groups_dict_format():
    """Accepts the dict-based param_groups format."""
    p1 = torch.zeros(3, dtype=torch.float64)
    p2 = torch.zeros(4, dtype=torch.float64)
    opt = DFOOptimizer(
        [{"params": [p1]}, {"params": [p2]}],
        algorithm="shade",
        bounds=(-1, 1),
        budget=200,  # one-off: dict-format smoke only needs tiny budget
        pop_size=POP_DFO_DEFAULT,
    )
    assert opt._dim == 7
    opt.step(closure_batched=lambda c: (c**2).sum(-1))
    # Both params should be updated
    assert not (p1 == 0).all(), "First param group not updated"
    assert not (p2 == 0).all(), "Second param group not updated"


# ------------------------------------------------------------------
# step() return value
# ------------------------------------------------------------------


def test_step_returns_best_loss():
    """step() returns the minimum loss in the generation."""
    params = [torch.zeros(3, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="shade", bounds=(-5, 5), pop_size=POP_DFO_DEFAULT)
    loss = opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0.0


# ------------------------------------------------------------------
# D2 — state_dict / load_state_dict roundtrip
# ------------------------------------------------------------------


def test_dfo_optimizer_state_dict_roundtrip_via_tempfile(tmp_path):
    """D2: state_dict → save → load into fresh DFOOptimizer → verify consistency.

    Current implementation detail: ``DFOOptimizer`` inherits
    ``state_dict/load_state_dict`` from ``torch.optim.Optimizer`` but does not
    override them to capture the inner optimizer's state. The roundtrip here
    therefore verifies the ``torch.optim.Optimizer`` base contract (state and
    param_groups survive save/load); a full inner-state roundtrip is a known
    gap documented in the test below.
    """
    # Build on CPU, float64 for determinism.
    p1 = torch.zeros(3, dtype=torch.float64)
    opt1 = DFOOptimizer(
        [p1],
        algorithm="shade",
        bounds=(-2.0, 2.0),
        budget=1000,
        pop_size=10,
        seed=42,
    )
    # Run one step so there is something to snapshot.
    loss1 = opt1.step(closure_batched=lambda c: (c**2).sum(-1))
    assert isinstance(loss1, torch.Tensor)

    # Save via torch.save (exercises the full pickling path).
    save_path = tmp_path / "opt_state.pt"
    torch.save(opt1.state_dict(), save_path)
    state = torch.load(save_path, weights_only=False)

    # Load into a *fresh* instance with identical construction.
    p2 = torch.zeros(3, dtype=torch.float64)
    opt2 = DFOOptimizer(
        [p2],
        algorithm="shade",
        bounds=(-2.0, 2.0),
        budget=1000,
        pop_size=10,
        seed=42,
    )
    opt2.load_state_dict(state)

    # The torch.optim.Optimizer base contract: state and param_groups survive.
    assert set(opt1.state_dict().keys()) == set(opt2.state_dict().keys())
    # Next step on both should succeed and produce finite losses.
    loss1b = opt1.step(closure_batched=lambda c: (c**2).sum(-1))
    loss2b = opt2.step(closure_batched=lambda c: (c**2).sum(-1))
    assert torch.isfinite(loss1b)
    assert torch.isfinite(loss2b)


def test_dfo_optimizer_state_dict_roundtrip_preserves_inner_and_evals():
    """``DFOOptimizer.state_dict`` snapshots the inner DFO optimizer and eval counter.

    The torch.optim-level ``state`` dict remains unused (DFO has no per-parameter
    state), but the extension keys ``_inner`` and ``_evals`` carry the
    optimization progress so a save/load round-trip produces identical next
    candidates.
    """
    p1 = torch.zeros(3, dtype=torch.float64)
    opt1 = DFOOptimizer([p1], algorithm="shade", bounds=(-1.0, 1.0), pop_size=10, seed=42)
    opt1.step(closure_batched=lambda c: (c**2).sum(-1))
    sd = opt1.state_dict()

    assert "_inner" in sd, "state_dict must carry the inner DFO optimizer state"
    assert sd["_evals"] == opt1._evals

    p2 = torch.zeros(3, dtype=torch.float64)
    opt2 = DFOOptimizer([p2], algorithm="shade", bounds=(-1.0, 1.0), pop_size=10, seed=99)
    opt2.load_state_dict(sd)
    assert opt2._evals == opt1._evals

    # Next ask must match between source and restored inner optimizers.
    a1, a2 = opt1._inner.ask(), opt2._inner.ask()
    assert torch.allclose(a1, a2), "next ask diverged after DFOOptimizer roundtrip"


# ------------------------------------------------------------------
# Round-2 edge-case audit: construction validators
# ------------------------------------------------------------------


def test_mixed_dtype_params_raise():
    """dfo_optimizer.py:83 — params with mismatched dtypes are rejected.

    This raise path previously had no coverage; silently accepting mixed
    dtypes would corrupt the flat parameter vector inside _set_params.
    """
    p1 = torch.zeros(3, dtype=torch.float32)
    p2 = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"same dtype"):
        DFOOptimizer([p1, p2], bounds=(-1, 1))


def test_step_raises_when_budget_exhausted():
    """``DFOOptimizer.step()`` raises ``RuntimeError`` once the budget is consumed.

    Callers that ignore ``is_exhausted`` get a loud failure instead of silently
    burning 2-10x their budget. The eval counter must not advance past the
    stated budget.
    """
    params = [torch.zeros(3, dtype=torch.float64)]
    opt = DFOOptimizer(params, algorithm="shade", bounds=(-1, 1), budget=20, pop_size=10)
    opt.step(closure_batched=lambda c: (c**2).sum(-1))
    opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert opt.is_exhausted
    evals_at_limit = opt._evals
    with pytest.raises(RuntimeError, match="budget exhausted"):
        opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert opt._evals == evals_at_limit, "step() advanced eval counter past exhaustion"
