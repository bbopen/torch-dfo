"""Tests for sanitize_fitness utility."""

from __future__ import annotations

import pytest
import torch

import torch_dfo.utils as _utils
from torch_dfo.utils import sanitize_fitness


@pytest.fixture(autouse=True)
def _reset_nan_warning():
    """Reset the global NaN warning flag before each test."""
    _utils._NAN_WARNED = False


def test_nan_replaced():
    f = torch.tensor([1.0, float("nan"), 3.0])
    result = sanitize_fitness(f)
    assert not torch.isnan(result).any()
    assert result[1].item() == pytest.approx(4.0)  # worst (3.0) + 1.0


def test_pos_inf_kept():
    f = torch.tensor([1.0, float("inf"), 3.0])
    result = sanitize_fitness(f)
    assert result[1] == float("inf")


def test_neg_inf_raises():
    f = torch.tensor([1.0, float("-inf"), 3.0])
    with pytest.raises(ValueError, match="negative infinity"):
        sanitize_fitness(f)


def test_all_nan():
    f = torch.tensor([float("nan"), float("nan")])
    result = sanitize_fitness(f)
    assert not torch.isnan(result).any()
    assert (result == 1.0).all()  # worst=0.0 (empty) + 1.0


def test_no_mutation_of_input():
    f = torch.tensor([1.0, float("nan"), 3.0])
    f_before = f.clone()
    sanitize_fitness(f)
    # Original NaN must survive — sanitize_fitness must not mutate its input.
    assert torch.isnan(f[1])
    assert torch.equal(torch.isnan(f), torch.isnan(f_before))

# ---------------------------------------------------------------------------
# D3 — sanitize_fitness end-to-end integration with CMAES / SHADE.
# ---------------------------------------------------------------------------


def _nan_interleaved_sphere(x: torch.Tensor) -> torch.Tensor:
    """Sphere fitness with NaN injected at every other row."""
    f = (x ** 2).sum(dim=-1)
    mask = torch.arange(f.shape[0], device=f.device) % 2 == 0
    f = f.clone()
    f[mask] = float("nan")
    return f


def test_sanitize_fitness_integration_cmaes() -> None:
    """D3: CMAES sees NaN fitness → sanitize → tell → internal state clean."""
    from torch_dfo.cmaes import CMAES
    opt = CMAES(
        dim=5, bounds=5.0, pop_size=10, device="cpu", dtype=torch.float64, seed=42,
    )
    for _ in range(3):
        c = opt.ask()
        raw = _nan_interleaved_sphere(c)
        clean = sanitize_fitness(raw)
        opt.tell(c, clean)
    assert not torch.isnan(opt.fitness).any(), "CMAES.fitness contains NaN after sanitized tells"
    assert not torch.isnan(opt.best_fitness), "CMAES.best_fitness is NaN after sanitized tells"
    assert torch.isfinite(opt.mean).all(), "CMAES.mean drifted to NaN/Inf under NaN fitness"
    assert opt.sigma > 0, f"CMAES.sigma went non-positive: {opt.sigma}"


def test_sanitize_fitness_integration_shade() -> None:
    """D3: SHADE sees NaN fitness → sanitize → tell → internal state clean."""
    from torch_dfo.shade import SHADE
    opt = SHADE(
        dim=5, bounds=5.0, pop_size=10, device="cpu", dtype=torch.float64, seed=42,
    )
    for _ in range(3):
        c = opt.ask()
        raw = _nan_interleaved_sphere(c)
        clean = sanitize_fitness(raw)
        opt.tell(c, clean)
    assert not torch.isnan(opt.fitness).any(), "SHADE.fitness contains NaN after sanitized tells"
    assert not torch.isnan(opt.best_fitness), "SHADE.best_fitness is NaN after sanitized tells"
    assert not torch.isnan(opt.memory_F).any(), "SHADE.memory_F contains NaN"
    assert not torch.isnan(opt.memory_CR).any(), "SHADE.memory_CR contains NaN"



def test_sanitize_fitness_integration_phased() -> None:
    """D3 (round-2): PhasedDFO sees NaN fitness → sanitize → tell → state clean.

    PhasedDFO was not covered alongside CMAES/SHADE in the original sanitize
    integration tests despite being the headline optimizer for 0.9.0. A NaN
    leaking into SHADE.memory_F via PhasedDFO's tell path would silently
    corrupt subsequent generations.
    """
    from torch_dfo import PhasedDFO
    opt = PhasedDFO(
        dim=5,
        bounds=5.0,
        budget=3000,
        pop_size=20,
        device="cpu",
        dtype=torch.float64,
        seed=42,
    )
    for _ in range(5):
        c = opt.ask()
        if c.shape[0] == 0:
            break
        raw = _nan_interleaved_sphere(c)
        clean = sanitize_fitness(raw)
        opt.tell(c, clean)
    assert torch.isfinite(opt.best_fitness), (
        "PhasedDFO.best_fitness drifted to NaN/Inf under sanitized NaN fitness"
    )
    assert torch.isfinite(opt._shade.population).all(), (
        "PhasedDFO SHADE population corrupted by NaN fitness"
    )
    assert not torch.isnan(opt._shade.memory_F).any(), (
        "PhasedDFO SHADE memory_F contains NaN"
    )
    assert not torch.isnan(opt._shade.memory_CR).any(), (
        "PhasedDFO SHADE memory_CR contains NaN"
    )
