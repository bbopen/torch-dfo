"""MPS device smoke test — verifies PhasedDFO runs entirely on GPU."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import (
    BEST_F_MPS_SMOKE_CEIL,
    BUDGET_SMOKE,
    POP_DFO_DEFAULT,
    POP_SHADE_STANDARD,
)

MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_phased_on_mps():
    """PhasedDFO.optimize() runs start-to-finish on MPS."""
    from torch_dfo import PhasedDFO
    from torch_dfo.benchmarks import rosenbrock

    device = torch.device("mps")

    def fitness_fn(X):
        return rosenbrock(X.to(dtype=torch.float32))

    opt = PhasedDFO(
        dim=10,
        bounds=5.0,
        budget=BUDGET_SMOKE,
        seed=42,
        device=device,
        dtype=torch.float32,
    )
    best_x, best_f = opt.optimize(fitness_fn)
    assert best_x.device.type == "mps"
    assert best_f.device.type == "mps"
    assert best_f.item() < BEST_F_MPS_SMOKE_CEIL


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_shade_on_mps():
    """SHADE ask/tell works on MPS."""
    from torch_dfo import SHADE

    device = torch.device("mps")
    opt = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=42,
        device=device,
        dtype=torch.float32,
    )
    candidates = opt.ask()
    assert candidates.device.type == "mps"
    fitness = (candidates**2).sum(dim=-1)
    opt.tell(candidates, fitness)
    assert opt.best_solution.device.type == "mps"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_dlr_on_mps():
    """DLRPortfolio ask/tell works on MPS."""
    from torch_dfo.dlr_cma import DLRPortfolio

    device = torch.device("mps")
    lb = torch.full((5,), -5.0, dtype=torch.float32, device=device)
    ub = torch.full((5,), 5.0, dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(42)
    dlr = DLRPortfolio(
        dim=5,
        lb=lb,
        ub=ub,
        lambdas=(12,),
        sigma_fracs=(0.3,),
        device=device,
        dtype=torch.float32,
        rng=gen,
    )
    candidates = dlr.ask()
    assert candidates.device.type == "mps"
    fitness = (candidates**2).sum(dim=-1)
    dlr.tell(candidates, fitness)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_cmaes_on_mps() -> None:
    """CMAES ask/tell runs on MPS without device or dtype errors."""
    from torch_dfo import CMAES

    device = torch.device("mps")
    opt = CMAES(
        dim=5,
        bounds=5.0,
        pop_size=POP_DFO_DEFAULT,
        seed=42,
        device=device,
        dtype=torch.float32,
    )
    candidates = opt.ask()
    assert candidates.device.type == "mps"
    assert candidates.shape == (POP_DFO_DEFAULT, 5)
    fitness = (candidates**2).sum(dim=-1)
    opt.tell(candidates, fitness)
    sol, fit = opt.best()
    assert sol.device.type == "mps"
    assert fit.device.type == "mps"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_nelder_mead_on_mps() -> None:
    """NelderMead ask/tell runs on MPS without device or dtype errors."""
    from torch_dfo import NelderMead

    device = torch.device("mps")
    opt = NelderMead(
        dim=3,
        bounds=5.0,
        seed=42,
        device=device,
        dtype=torch.float32,
    )
    candidates = opt.ask()
    assert candidates.device.type == "mps"
    fitness = (candidates**2).sum(dim=-1)
    opt.tell(candidates, fitness)
    sol, fit = opt.best()
    assert sol.device.type == "mps"
    assert fit.device.type == "mps"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_dfo_optimizer_on_mps() -> None:
    """DFOOptimizer wrapper runs one step on MPS without errors."""
    from torch_dfo import DFOOptimizer

    device = torch.device("mps")
    params = [torch.zeros(3, dtype=torch.float32, device=device)]
    opt = DFOOptimizer(
        params,
        algorithm="shade",
        bounds=(-5.0, 5.0),
        budget=500,
        pop_size=POP_DFO_DEFAULT,
    )
    loss = opt.step(closure_batched=lambda c: (c**2).sum(-1))
    assert isinstance(loss, torch.Tensor)
    assert loss.device.type == "mps"
