"""Tests for the standalone quantized thermal-control reference."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _reference():
    name = "test_engineering_control_reference"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "examples" / "06_engineering_control.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_simulator_matches_independent_unrolled_recurrence() -> None:
    reference = _reference()
    scenarios = reference.fixed_scenarios("train")
    requested = torch.tensor(
        [[-1.0, -0.75, -0.5, -0.25] * 4, [0.0, 0.25, 0.5, 0.75] * 4],
        dtype=torch.float64,
    )
    trace, commands = reference.simulate(requested, scenarios)
    outdoor, internal, initial = scenarios.tensor(device=torch.device("cpu"), dtype=torch.float64)
    expected = reference.analytic_linear_reference(commands, outdoor, internal, initial)
    assert torch.allclose(trace, expected, rtol=0.0, atol=1e-12)


def test_actuator_saturates_then_quantizes() -> None:
    reference = _reference()
    commands = reference.effective_actuation(
        torch.tensor([[-1.4, -0.37, -0.12, 0.13, 0.37, 1.4]], dtype=torch.float64)
    )
    assert torch.equal(
        commands,
        torch.tensor([[-1.0, -0.25, 0.0, 0.25, 0.25, 1.0]], dtype=torch.float64),
    )


def test_evaluator_identity_includes_plant_parameters() -> None:
    reference = _reference()
    default = reference.FrozenEvaluator("train")
    changed = reference.FrozenEvaluator(
        "train", reference.ThermalPlant(actuator_capacity_w=1_000.0)
    )
    plan = torch.ones(1, reference.HORIZON, dtype=torch.float64)
    assert default.evaluator_hash != changed.evaluator_hash
    assert not torch.equal(default(plan), changed(plan))


def test_fixed_splits_are_disjoint_and_stable() -> None:
    reference = _reference()
    train = reference.fixed_scenarios("train")
    heldout = reference.fixed_scenarios("heldout")
    assert train.count == 8
    assert heldout.count == 5
    assert reference.scenario_hash(train) != reference.scenario_hash(heldout)
    assert reference.task_hash() == reference.task_hash()


def test_phase_zero_probes_discriminate_without_reading_heldout() -> None:
    reference = _reference()
    probes = reference.run_evaluator_probes()
    assert not probes["degenerate_zero"]["passes"]
    assert probes["ceiling_domain_control"]["passes"]
    assert probes["cheap_exploit_constant_cooling"]["passes"]
    assert not probes["runaway_all_heat"]["passes"]
    null = probes["null_uniform_random"]
    assert null["seed"] == 24_681
    assert null["sample_count"] == 64
    assert null["min_score"] <= null["median_score"] <= null["max_score"]
    assert null["mean_score"] > probes["ceiling_domain_control"]["score"]


def test_runner_uses_counted_public_search_run() -> None:
    reference = _reference()
    benchmark = _benchmark()
    selected, charged, score, elapsed = benchmark._run_torch_dfo(
        "random",
        reference.FrozenEvaluator("train"),
        budget=12,
        seed=17,
        device=torch.device("cpu"),
    )
    assert selected.shape == (1, reference.HORIZON)
    assert charged == 12
    assert score >= 0.0
    assert elapsed >= 0.0


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "engineering_control.py"
    spec = importlib.util.spec_from_file_location("test_engineering_control_benchmark", path)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    return benchmark


def test_optional_evotorch_adapter_uses_physical_bounds(device) -> None:
    pytest.importorskip("evotorch")
    if device.type not in ("cpu", "cuda"):
        pytest.skip("reference benchmark supports CPU and CUDA")
    reference = _reference()
    evaluator = reference.FrozenEvaluator("train")
    selected, charged, score, _ = _benchmark()._run_evotorch_cma(
        evaluator, 12, 17, 0.30, None, device
    )
    assert charged == 12
    assert bool((selected.abs() <= 1.0).all())
    assert float(evaluator(selected).item()) == pytest.approx(score)
