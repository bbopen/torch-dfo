"""Tests for the benchmark harness (benchmarks/run_benchmarks.py) utilities."""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helper: detect optional dependencies
# ---------------------------------------------------------------------------


def _yahpo_available() -> bool:
    return importlib.util.find_spec("yahpo_gym") is not None


def _configspace_available() -> bool:
    return importlib.util.find_spec("ConfigSpace") is not None


# ---------------------------------------------------------------------------
# Sentinel tests — always run, no optional deps needed
# ---------------------------------------------------------------------------


def test_integer_hp_rounding_sentinel_defined() -> None:
    """The integer HP type sentinel is defined as a tuple (may be empty on non-YAHPO systems)."""
    from benchmarks.run_benchmarks import _CS_INT_HP_TYPES

    assert isinstance(_CS_INT_HP_TYPES, tuple), (
        f"_CS_INT_HP_TYPES should be a tuple, got {type(_CS_INT_HP_TYPES)}"
    )


def test_integer_hp_rounding_sentinel_nonempty_when_configspace_available() -> None:
    """When ConfigSpace is importable the sentinel must contain at least one type."""
    if not _configspace_available():
        pytest.skip("ConfigSpace not installed")

    from benchmarks.run_benchmarks import _CS_INT_HP_TYPES

    assert len(_CS_INT_HP_TYPES) > 0, (
        "ConfigSpace is available but _CS_INT_HP_TYPES is empty — "
        "the lazy import block failed silently"
    )
    for tp in _CS_INT_HP_TYPES:
        assert isinstance(tp, type), f"Every entry must be a type, got {tp!r}"


# ---------------------------------------------------------------------------
# Integration test: round-trip produces int for integer HPs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_yahpo_available() and _configspace_available()),
    reason="yahpo_gym + ConfigSpace not available",
)
def test_tensor_to_yahpo_config_rounds_integers() -> None:
    """Integer HPs decoded via _tensor_to_yahpo_config must be Python ints."""
    import torch
    from ConfigSpace.hyperparameters import (
        UniformIntegerHyperparameter,
    )

    from benchmarks.run_benchmarks import _tensor_to_yahpo_config

    # Build a minimal ConfigSpace with one integer HP and one float HP
    try:
        import ConfigSpace as CS
    except ImportError:
        pytest.skip("ConfigSpace not importable")

    cs = CS.ConfigurationSpace(seed=0)
    cs.add_hyperparameter(
        UniformIntegerHyperparameter("n_layers", lower=1, upper=10, default_value=5)
    )
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=1e-5, upper=1e-1))

    # Use a tensor that would produce a non-integer decoded value (e.g. 0.37)
    x = torch.tensor([0.37, 0.5], dtype=torch.float64)
    config = _tensor_to_yahpo_config(x, cs)

    n_layers_val = config["n_layers"]
    lr_val = config["lr"]

    assert isinstance(n_layers_val, int), (
        f"n_layers (UniformIntegerHyperparameter) should decode to int, "
        f"got {type(n_layers_val).__name__} = {n_layers_val!r}"
    )
    # Float HP should remain float
    assert isinstance(lr_val, float), (
        f"lr (UniformFloatHyperparameter) should decode to float, "
        f"got {type(lr_val).__name__} = {lr_val!r}"
    )

    # Value must be in the valid integer range
    assert 1 <= n_layers_val <= 10, f"n_layers out of range: {n_layers_val}"


@pytest.mark.parametrize("target_at", [None, 1, 5])
def test_pycma_counts_partial_generation_and_uses_official_target(monkeypatch, target_at):
    """A full generation must not overrun a residual budget or invent a target."""
    import sys
    from types import SimpleNamespace

    from benchmarks.run_benchmarks import _run_pycma_on_coco

    class Problem:
        dimension = 2
        lower_bounds = (-5.0, -5.0)
        upper_bounds = (5.0, 5.0)
        initial_solution = (0.0, 0.0)
        evaluations = 0
        final_target_hit = False

        def __call__(self, x):
            self.evaluations += 1
            if self.evaluations == target_at:
                self.final_target_hit = True
            return -float(self.evaluations)

    class Strategy:
        popsize = 4

        def __init__(self, *args):
            pass

        def stop(self):
            return False

        def ask(self):
            return [[0.0, 0.0]] * self.popsize

        def tell(self, solutions, fitnesses):
            assert len(solutions) == len(fitnesses) == self.popsize

    monkeypatch.setitem(sys.modules, "cma", SimpleNamespace(CMAEvolutionStrategy=Strategy))
    problem = Problem()
    result = _run_pycma_on_coco(problem, 5)
    expected_calls = target_at or 5
    assert result.fe_used == problem.evaluations == expected_calls
    assert result.best_fitness == -float(expected_calls)
    assert result.solved is (target_at is not None)


def test_yahpo_comparison_aggregates_repeats_without_row_order_dependence():
    from benchmarks.run_benchmarks import BenchmarkResult, SuiteReport

    def row(value, optimizer, suite="yahpo", dim=2):
        return BenchmarkResult(suite, "same-name", dim, optimizer, value, 0.0, 1, 0.0, False)

    rows = [row(2.0, "torch-dfo"), row(1.0, "random"), row(5.0, "random")]
    # Other suites and dimensions must not enter this comparison.
    rows += [row(-100.0, "random", suite="bbob"), row(-100.0, "random", dim=3)]
    report = SuiteReport("comparison", rows)
    assert report.yahpo_wins_vs_random() == (1, 1)
    report.results.reverse()
    assert report.yahpo_wins_vs_random() == (1, 1)


def test_coco_uses_literal_instance_ids_and_official_target():
    pytest.importorskip("cocoex")
    pytest.importorskip("cma")
    from benchmarks.run_benchmarks import run_bbob

    report = run_bbob(dims=[2], instances=[6], functions=[3], budget_mult=20)
    assert len(report.results) == 2
    for result in report.results:
        assert result.problem_name == "bbob_f003_i06_d02"
        assert result.fe_used <= 40
        assert not result.solved
        # No locally estimated optimum can be used to declare success.
        import math

        assert math.isnan(result.precision)


@pytest.mark.parametrize("failure", [float("inf"), -float("inf"), float("nan")])
def test_yahpo_nonfinite_failures_cannot_win(failure):
    from benchmarks.run_benchmarks import BenchmarkResult, SuiteReport

    report = SuiteReport(
        "failures",
        [
            BenchmarkResult("yahpo", "a", 2, "torch-dfo", failure, 0.0, 1, 0.0, False),
            BenchmarkResult("yahpo", "a", 2, "random", 1.0, 0.0, 1, 0.0, False),
        ],
    )
    assert report.yahpo_wins_vs_random() == (0, 1)


def test_yahpo_retains_each_repeat_and_its_evaluation_cost(monkeypatch):
    import math
    import sys
    from types import SimpleNamespace

    import torch

    import benchmarks.run_benchmarks as harness
    import torch_dfo

    calls = []

    class Benchmark:
        instances = ("example",)
        config_space = SimpleNamespace(get_hyperparameters=lambda: [])

        def __init__(self, name):
            pass

        def set_instance(self, instance):
            pass

        def objective_function(self, config):
            calls.append(config)
            return [{"val_accuracy": 1.0}]

    class Search:
        def __init__(self, **kwargs):
            self._fe_count = kwargs["budget"]

        def optimize(self, fitness):
            x = torch.zeros(self._fe_count, 0)
            return x[0], fitness(x).min()

    monkeypatch.setitem(sys.modules, "yahpo_gym", SimpleNamespace(BenchmarkSet=Benchmark))
    monkeypatch.setattr(torch_dfo, "PhasedDFO", Search)
    monkeypatch.setattr(harness, "_tensor_to_yahpo_config", lambda row, space: {})
    report = harness.run_yahpo(scenarios=["lcbench"], budgets=[2], random_repeats=3)
    assert len(report.results) == 4
    assert sum(r.fe_used for r in report.results) == len(calls) == 8
    assert all(math.isnan(r.precision) for r in report.results)
