"""Tests for state_dict / load_state_dict round-trip on all optimizers."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import (
    ATOL_F64_DEFAULT,
    BUDGET_PHASED_QUICK,
    POP_CMAES_STANDARD,
    POP_SHADE_STANDARD,
)
from torch_dfo import CMAES, SHADE, NelderMead, PhasedDFO
from torch_dfo.dlr_cma import DLRPortfolio


def test_shade_roundtrip():
    opt1 = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=42, device="cpu")
    c1 = opt1.ask()
    opt1.tell(c1, torch.arange(POP_SHADE_STANDARD, dtype=torch.float64))
    state = opt1.state_dict()

    opt2 = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=99, device="cpu")
    opt2.load_state_dict(state)

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2), "SHADE roundtrip failed"


def test_cmaes_roundtrip():
    opt1 = CMAES(dim=5, bounds=5.0, pop_size=POP_CMAES_STANDARD, seed=42, device="cpu")
    c1 = opt1.ask()
    opt1.tell(c1, torch.arange(POP_CMAES_STANDARD, dtype=torch.float64))
    state = opt1.state_dict()

    opt2 = CMAES(dim=5, bounds=5.0, pop_size=POP_CMAES_STANDARD, seed=99, device="cpu")
    opt2.load_state_dict(state)

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2), "CMAES roundtrip failed"


def test_nelder_mead_roundtrip():
    opt1 = NelderMead(dim=3, bounds=5.0, seed=42, device="cpu")
    c1 = opt1.ask()
    opt1.tell(c1, torch.arange(c1.shape[0], dtype=torch.float64))
    state = opt1.state_dict()

    opt2 = NelderMead(dim=3, bounds=5.0, seed=99, device="cpu")
    opt2.load_state_dict(state)

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2), "NelderMead roundtrip failed"


def test_dlr_roundtrip():
    lb = torch.full((10,), -5.0, dtype=torch.float64)
    ub = torch.full((10,), 5.0, dtype=torch.float64)
    gen1 = torch.Generator(device="cpu").manual_seed(42)
    dlr1 = DLRPortfolio(
        dim=10,
        lb=lb,
        ub=ub,
        lambdas=(12, 12),
        sigma_fracs=(0.3, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64,
        rng=gen1,
    )
    c1 = dlr1.ask()
    dlr1.tell(c1, torch.arange(c1.shape[0], dtype=torch.float64))
    state = dlr1.state_dict()

    gen2 = torch.Generator(device="cpu").manual_seed(99)
    dlr2 = DLRPortfolio(
        dim=10,
        lb=lb,
        ub=ub,
        lambdas=(12, 12),
        sigma_fracs=(0.3, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64,
        rng=gen2,
    )
    dlr2.load_state_dict(state)

    a1, a2 = dlr1.ask(), dlr2.ask()
    assert torch.allclose(a1, a2), "DLR roundtrip failed"


def test_shade_multi_generation_roundtrip():
    """After 5 generations, every persisted SHADE tensor must round-trip bit-exactly."""
    opt1 = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=42, device="cpu")
    for _ in range(5):
        c = opt1.ask()
        opt1.tell(c, torch.randn(POP_SHADE_STANDARD, dtype=torch.float64))

    state = opt1.state_dict()
    opt2 = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=99, device="cpu")
    opt2.load_state_dict(state)

    # Adaptive-state bit-exact equality. A C6-class regression in any of these
    # tensors would otherwise slip through a "next ask matches" check.
    for attr in ("population", "fitness", "best_solution", "best_fitness"):
        a, b = getattr(opt1, attr), getattr(opt2, attr)
        assert torch.equal(a, b), f"SHADE state tensor '{attr}' diverged across roundtrip"
    for attr in ("memory_F", "memory_CR", "_archive", "_trial_F", "_trial_CR", "_trials"):
        a, b = getattr(opt1._memory, attr), getattr(opt2._memory, attr)
        assert torch.equal(a, b), f"SHADE memory tensor '{attr}' diverged across roundtrip"
    assert opt1._memory._memory_pos == opt2._memory._memory_pos
    assert opt1._memory._initialized == opt2._memory._initialized
    assert opt1._generation == opt2._generation

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2), "SHADE next ask diverged after roundtrip"


def test_cmaes_multi_generation_roundtrip():
    """After 5 generations, every persisted CMAES tensor must round-trip bit-exactly."""
    opt1 = CMAES(dim=5, bounds=5.0, pop_size=POP_CMAES_STANDARD, seed=42, device="cpu")
    for _ in range(5):
        c = opt1.ask()
        opt1.tell(c, torch.randn(POP_CMAES_STANDARD, dtype=torch.float64))

    state = opt1.state_dict()
    opt2 = CMAES(dim=5, bounds=5.0, pop_size=POP_CMAES_STANDARD, seed=99, device="cpu")
    opt2.load_state_dict(state)

    for attr in (
        "population",
        "fitness",
        "best_solution",
        "best_fitness",
        "C",
        "B",
        "D_diag",
        "C_invsqrt",
        "mean",
    ):
        a, b = getattr(opt1, attr), getattr(opt2, attr)
        assert torch.equal(a, b), f"CMAES state tensor '{attr}' diverged across roundtrip"
    for attr in ("p_sigma", "p_c", "_path_vectors"):
        a, b = getattr(opt1._path, attr), getattr(opt2._path, attr)
        assert torch.equal(a, b), f"CMAES path tensor '{attr}' diverged across roundtrip"
    for attr in ("sigma", "_decomp_gen", "_generation"):
        assert getattr(opt1, attr) == getattr(opt2, attr), (
            f"CMAES scalar '{attr}' diverged across roundtrip"
        )
    for attr in ("_path_count", "_path_pos"):
        assert getattr(opt1._path, attr) == getattr(opt2._path, attr), (
            f"CMAES path scalar '{attr}' diverged across roundtrip"
        )

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2), "CMAES next ask diverged after roundtrip"


def test_state_dict_is_deep_copy():
    """Mutating state_dict output must not affect the original optimizer."""
    opt = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=42, device="cpu")
    c = opt.ask()
    opt.tell(c, torch.arange(POP_SHADE_STANDARD, dtype=torch.float64))

    state = opt.state_dict()
    state["population"].zero_()

    assert not torch.allclose(opt.population, state["population"]), (
        "state_dict should return cloned tensors"
    )


def test_phased_roundtrip() -> None:
    """PhasedDFO state_dict round-trip preserves the next ask()."""
    opt1 = PhasedDFO(
        dim=5,
        bounds=5.0,
        budget=BUDGET_PHASED_QUICK,
        seed=42,
        device="cpu",
        dtype=torch.float64,
    )
    c1 = opt1.ask()
    opt1.tell(c1, (c1**2).sum(dim=-1))
    state = opt1.state_dict()

    opt2 = PhasedDFO(
        dim=5,
        bounds=5.0,
        budget=BUDGET_PHASED_QUICK,
        seed=99,
        device="cpu",
        dtype=torch.float64,
    )
    opt2.load_state_dict(state)

    a1, a2 = opt1.ask(), opt2.ask()
    assert torch.allclose(a1, a2, atol=ATOL_F64_DEFAULT), "PhasedDFO roundtrip failed"


def test_phased_roundtrip_across_phases() -> None:
    """Round-trip AFTER a DE→CMA phase transition preserves next ask().

    Uses BUDGET_PHASED_QUICK (10k) and loops until phase >= 1, failing the
    test if that never happens — otherwise the roundtrip check would have
    nothing to verify and could silently pass.
    """
    opt1 = PhasedDFO(
        dim=5,
        bounds=5.0,
        budget=BUDGET_PHASED_QUICK,
        seed=42,
        device="cpu",
        dtype=torch.float64,
    )
    # Drive DE all the way into CMA so the roundtrip actually spans phases.
    # BUDGET_PHASED_QUICK (10k) is ample for dim=5 to handoff to CMA.
    for _ in range(5000):
        if opt1.phase >= 1:
            break
        if opt1.done:
            break
        c = opt1.ask()
        opt1.tell(c, (c**2).sum(dim=-1))

    assert opt1.phase >= 1, (
        f"Precondition failed: DE never handed off to CMA-ES "
        f"(phase={opt1.phase}, fe={opt1.fe_count}). Roundtrip across phases not exercised."
    )

    state = opt1.state_dict()

    opt2 = PhasedDFO(
        dim=5,
        bounds=5.0,
        budget=BUDGET_PHASED_QUICK,
        seed=99,
        device="cpu",
        dtype=torch.float64,
    )
    opt2.load_state_dict(state)

    assert opt2.phase == opt1.phase, "phase not preserved"
    assert opt2.fe_count == opt1.fe_count, "fe_count not preserved"

    # No ``if not opt1.done`` guard — phase >= 1 implies CMA-ES is mid-run and
    # ``ask()`` will produce a valid candidate batch.
    a1, a2 = opt1.ask(), opt2.ask()
    assert a1.shape[0] > 0, "opt1.ask() returned empty batch after phase=1 handoff"
    assert torch.allclose(a1, a2, atol=ATOL_F64_DEFAULT), (
        f"ask() diverged after phase={opt1.phase} roundtrip"
    )


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="requires MPS (see test docstring for why CUDA is excluded)",
)
def test_shade_roundtrip_cpu_to_mps() -> None:
    """State saved on CPU loads cleanly into an MPS-device optimizer.

    MPS uses the CPU-backed torch Generator under the hood, so the RNG
    state saved from a CPU optimizer has the right byte layout for a subsequent
    MPS load. CUDA is deliberately excluded here because its Generator state
    uses a different, larger binary representation; the relevant error in that
    case is ``RuntimeError: RNG state is wrong size`` from
    ``src/torch_dfo/base.py::BaseOptimizer.load_state_dict``.
    """
    from torch_dfo import SHADE

    dtype = torch.float32  # MPS does not support float64
    src = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=42,
        device="cpu",
        dtype=dtype,
    )
    c = src.ask()
    src.tell(c, torch.arange(POP_SHADE_STANDARD, dtype=dtype))
    state = src.state_dict()

    dst = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=99,
        device="mps",
        dtype=dtype,
    )
    dst.load_state_dict(state)

    next_candidate = dst.ask()
    assert next_candidate.device.type == "mps"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_shade_roundtrip_cpu_to_cuda() -> None:
    """CPU→CUDA load must re-seed the CUDA generator from the saved ``_rng_seed``.

    Binary RNG state is not portable across device kinds; the fallback path
    in ``BaseOptimizer.load_state_dict`` re-seeds instead. This test pins that
    behaviour by a direct ``initial_seed()`` check and by showing the
    restored optimizer's next ask matches a freshly-seeded CUDA optimizer
    that was loaded from the same state — proving the fallback activated
    rather than silently leaving the destination generator at birth-seed 99.
    """
    src = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=42,
        device="cpu",
        dtype=torch.float64,
    )
    c = src.ask()
    src.tell(c, torch.arange(POP_SHADE_STANDARD, dtype=torch.float64))
    state = src.state_dict()

    dst = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=99,  # deliberately different from src seed
        device="cuda",
        dtype=torch.float64,
    )
    dst.load_state_dict(state)

    assert dst.population.device.type == "cuda"
    assert torch.allclose(dst.population.cpu(), src.population, atol=ATOL_F64_DEFAULT), (
        "population not preserved across CPU→CUDA load"
    )
    # The re-seed fallback must have run: dst's generator should now carry
    # the source's seed (42), not its birth seed (99).
    assert dst._gen.initial_seed() == 42, (
        f"cross-device load did not re-seed: expected 42, got {dst._gen.initial_seed()}"
    )

    # And the restored optimizer's next ask matches a peer CUDA optimizer that
    # started life with the source seed and loaded the same state — proof that
    # the re-seed path produced the expected downstream RNG trajectory.
    peer = SHADE(
        dim=5,
        bounds=5.0,
        pop_size=POP_SHADE_STANDARD,
        seed=42,
        device="cuda",
        dtype=torch.float64,
    )
    peer.load_state_dict(state)
    assert torch.allclose(dst.ask(), peer.ask(), atol=ATOL_F64_DEFAULT)


def test_load_state_dict_shape_mismatch_raises() -> None:
    """load_state_dict on a differently-sized optimizer raises predictably."""
    from torch_dfo import SHADE

    src = SHADE(dim=5, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=42, device="cpu")
    src.tell(src.ask(), torch.arange(POP_SHADE_STANDARD, dtype=torch.float64))
    state = src.state_dict()

    # Different dim — should refuse to load.
    dst = SHADE(dim=8, bounds=5.0, pop_size=POP_SHADE_STANDARD, seed=42, device="cpu")
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        dst.load_state_dict(state)
