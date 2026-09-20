"""Focused checks for the fixed CMA comparison runner."""

from __future__ import annotations

import importlib.util
import math
from importlib import metadata

import pytest
import torch

from benchmarks import cma_comparison as comparison


def test_fixture_is_repeatable_and_orthogonal() -> None:
    first = comparison.fixture()
    second = comparison.fixture()
    assert first == second
    assert len(first["sha256_float64_le_shift_then_rotation"]) == 64
    rotation = torch.tensor(first["rotation"], dtype=torch.float64)
    assert torch.allclose(
        rotation.T @ rotation, torch.eye(comparison.DIM, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_presearch_objective_probes_and_known_optima() -> None:
    rows = comparison.objective_probes(comparison.fixture())
    cpu = [row for row in rows if row["device"] == "cpu"]
    assert len(cpu) == 15
    assert {row["probe"] for row in cpu} == {
        "known_optimum",
        "shifted_origin",
        "partial_candidate",
        "fixed_random_batch",
    }
    assert all(math.isfinite(score) for row in cpu for score in row["observed"])


def test_original_thermal_admission_probes() -> None:
    probes = comparison.thermal_evaluator_probes()["cpu"]
    assert not probes["degenerate_zero"]["passes"]
    assert probes["ceiling_domain_control"]["passes"]
    assert not probes["runaway_all_heat"]["passes"]


def test_counted_objective_rejects_bad_values_and_keeps_attempts() -> None:
    task = comparison.make_task("shifted_sphere", torch.device("cpu"), comparison.fixture())
    observed = comparison.CountedObjective(task)
    point = torch.zeros(comparison.POP_SIZE, comparison.DIM, dtype=torch.float64)
    scores = observed(point)
    assert scores.shape == (comparison.POP_SIZE,)
    assert observed.attempted == observed.completed == comparison.POP_SIZE
    assert observed.best_x is not None

    invalid = point.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite coordinates"):
        observed(invalid)
    assert observed.attempted == comparison.POP_SIZE * 2
    assert observed.completed == comparison.POP_SIZE

    with pytest.raises(ValueError, match="wrong count"):
        observed(point[:1])
    assert observed.attempted == comparison.POP_SIZE * 2 + 1

    bad_task = comparison.Task(
        task.name,
        lambda x: torch.full(
            (x.shape[0],),
            float("nan"),
            dtype=x.dtype,
            device=x.device,
        ),
        task.target,
        task.device,
        task.oracle,
    )
    failed = comparison.CountedObjective(bad_task)
    with pytest.raises(ValueError, match="non-finite scores"):
        failed(point)
    assert failed.attempted == comparison.POP_SIZE
    assert failed.completed == 0


@pytest.mark.parametrize("method", ["torch_dfo_cpu", "random_cpu"])
def test_native_one_generation_uses_exact_budget(method: str) -> None:
    result = comparison.run_one(
        method, "shifted_sphere", 11, comparison.POP_SIZE, comparison.fixture()
    )
    assert result["status"] == "ok"
    assert result["attempted"] == result["completed"] == comparison.POP_SIZE
    assert result["batch_calls"] == len(result["trace"]) == 1
    assert result["search_run"]["charged"] == comparison.POP_SIZE
    assert result["cpu_oracle_value"] == pytest.approx(result["best_value"], rel=1e-12)


@pytest.mark.skipif(importlib.util.find_spec("cma") is None, reason="pycma is unavailable")
def test_pycma_one_generation_uses_exact_budget() -> None:
    result = comparison.run_one(
        "pycma_cpu", "shifted_sphere", 11, comparison.POP_SIZE, comparison.fixture()
    )
    assert result["status"] == "ok"
    assert result["attempted"] == result["completed"] == comparison.POP_SIZE
    assert result["batch_calls"] == len(result["trace"]) == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("evox") is None,
    reason="CUDA or EvoX is unavailable",
)
def test_evox_cuda_one_generation_uses_exact_budget() -> None:
    try:
        result = comparison.run_one(
            "evox_cuda", "shifted_sphere", 11, comparison.POP_SIZE, comparison.fixture()
        )
    except comparison.PartialRunError as error:
        message = str(error)
        if (
            torch.__version__.startswith("2.13.")
            and metadata.version("evox") == "1.4.0"
            and "Encountered aliasing" in message
            and "torch.cond" in message
            and error.partial["attempted"] == comparison.POP_SIZE
            and error.partial["completed"] == comparison.POP_SIZE
            and error.partial["batch_calls"] == 1
        ):
            pytest.xfail("stock EvoX 1.4.0 torch.cond aliasing on Torch 2.13")
        raise
    assert result["status"] == "ok"
    assert result["attempted"] == result["completed"] == comparison.POP_SIZE
    assert result["batch_calls"] == len(result["trace"]) == 1
    assert result["evox_covariance_probe"]["reference_status"] == (
        "unvalidated_mathematical_CMAES_reference"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_one_generation_uses_exact_budget() -> None:
    result = comparison.run_one(
        "torch_dfo_cuda", "shifted_sphere", 11, comparison.POP_SIZE, comparison.fixture()
    )
    assert result["attempted"] == result["completed"] == comparison.POP_SIZE
    assert result["cuda_allocated_bytes"]["peak_delta"] >= 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cma") is None,
    reason="CUDA or pycma is unavailable",
)
def test_pycma_cuda_objective_one_generation_uses_exact_budget() -> None:
    result = comparison.run_one(
        "pycma_cuda_objective", "shifted_sphere", 11, comparison.POP_SIZE, comparison.fixture()
    )
    assert result["attempted"] == result["completed"] == comparison.POP_SIZE
    assert result["pycma_mode"] == "CPU_strategy_CUDA_objective"
