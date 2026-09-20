"""Fixture checks for NASBench201 search accounting and data isolation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from benchmarks import nasbench201 as nas

ARCH_A = (0, 0, 0, 0, 0, 0)
ARCH_B = (1, 0, 0, 0, 0, 0)


def _record(index: int, ops: tuple[int, ...], accuracies: tuple[float, float]) -> dict:
    validation = {
        dataset: [
            {"seed": seed, "valid-accuracy": accuracy} for seed, accuracy in enumerate(accuracies)
        ]
        for dataset in nas.DATASETS
    }
    return {
        "index": index,
        "ops": list(ops),
        "phenotype": f"fixture-{index}",
        "validation": validation,
    }


def _table() -> nas.ValidationTable:
    return nas.ValidationTable(
        [_record(0, ARCH_A, (90.0, 80.0)), _record(1, ARCH_B, (90.0, 90.0))],
        require_full=False,
    )


def _logits(ops: tuple[int, ...]) -> torch.Tensor:
    rows = torch.full((nas.DIM,), -0.5, dtype=torch.float64)
    for edge, category in enumerate(ops):
        rows[edge * nas.CATEGORIES + category] = 0.5
    return rows


def test_decoder_priority_and_bounds() -> None:
    priority = nas.tie_priority(7)
    assert priority == nas.tie_priority(7)
    assert all(sorted(order) == list(range(5)) for order in priority)
    decoded, ties = nas.decode_logits([0.0] * nas.DIM, priority)
    assert decoded == tuple(order[0] for order in priority)
    assert ties == nas.EDGES
    assert {nas.tie_priority(seed)[0][0] for seed in range(30)} == set(range(5))
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        nas.decode_logits([2.0] + [0.0] * (nas.DIM - 1), priority)


def test_duplicates_use_shared_private_visit_trials_and_budget() -> None:
    table = _table()
    first = nas.ValidationOracle(table, "cifar10", 4, 3, nas.tie_priority(4))
    second = nas.ValidationOracle(table, "cifar10", 4, 3, nas.tie_priority(4))
    rows = torch.stack((_logits(ARCH_A), _logits(ARCH_A), _logits(ARCH_B)))
    scores = first.score_tensor(rows)
    assert first.facts()["attempted"] == first.facts()["completed"] == 3
    assert first.facts()["unique_architectures"] == 2
    assert first.facts()["repeated_architectures"] == 1
    assert [item["visit_index"] for item in first.trace] == [0, 1, 0]
    assert scores.tolist() == [item["validation_error"] for item in first.trace]
    second.observe_ops(ARCH_B)
    assert [second.observe_ops(ARCH_A), second.observe_ops(ARCH_A)] == scores[:2].tolist()
    assert [item["trial_index"] for item in first.trace[:2]] == [
        nas.trial_index("cifar10", 4, ARCH_A, visit, 2) for visit in range(2)
    ]
    with pytest.raises(ValueError, match="over-budget"):
        first.score_tensor(rows[:1])
    assert first.completed == 3


def test_earliest_equal_error_wins() -> None:
    oracle = nas.ValidationOracle(_table(), "cifar100", 1, 2, nas.tie_priority(1))
    oracle.observe_ops(ARCH_B)
    oracle.observe_ops(ARCH_B)
    assert oracle.best_step == 1
    assert oracle.best_ops == ARCH_B
    assert [item["incumbent_step"] for item in oracle.trace] == [1, 1]


def test_partial_batch_failure_retains_consumed_observations() -> None:
    oracle = nas.ValidationOracle(_table(), "cifar10", 0, 2, nas.tie_priority(0))
    bad = _logits(ARCH_A)
    bad[0] = 2.0
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        oracle.score_tensor(torch.stack((_logits(ARCH_A), bad)))
    assert oracle.attempted == 2
    assert oracle.completed == 1
    assert oracle.trace[0]["step"] == 1


def test_aging_mutates_exactly_one_edge_to_different_operation() -> None:
    rng = random.Random(12)
    for _ in range(200):
        child = nas._mutate_one_edge(ARCH_A, rng)
        changed = [edge for edge in range(nas.EDGES) if child[edge] != ARCH_A[edge]]
        assert len(changed) == 1
        assert 0 <= child[changed[0]] < nas.CATEGORIES


def test_aging_starts_with_100_observations_then_mutates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_record(0, ARCH_A, (90.0, 90.0))]
    index = 1
    for edge in range(nas.EDGES):
        for category in range(1, nas.CATEGORIES):
            child = list(ARCH_A)
            child[edge] = category
            records.append(_record(index, tuple(child), (90.0, 90.0)))
            index += 1
    table = nas.ValidationTable(records, require_full=False)
    monkeypatch.setattr(nas, "_random_ops", lambda _rng: ARCH_A)
    result = nas.run_one(table, "cifar10", "aging_cpu", 2, 101)
    assert result["status"] == "ok"
    assert result["completed"] == 101
    assert all(item["ops"] == list(ARCH_A) for item in result["trace"][:100])
    assert sum(a != b for a, b in zip(result["trace"][100]["ops"], ARCH_A, strict=True)) == 1


def test_final_test_file_opens_only_after_search_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_file = tmp_path / "validation.json"
    validation_file.write_text(
        json.dumps({"source": {"fidelity": 200}, "records": [_record(0, ARCH_A, (90, 80))]})
    )
    test_file = tmp_path / "test.json"
    test_file.write_text(
        json.dumps(
            {
                "source": {"fidelity": 200},
                "records": [
                    {
                        "index": 0,
                        "test": {
                            dataset: [
                                {"seed": 0, "test-accuracy": 70},
                                {"seed": 1, "test-accuracy": 90},
                            ]
                            for dataset in nas.DATASETS
                        },
                    }
                ],
            }
        )
    )
    output = tmp_path / "runs.jsonl"
    monkeypatch.setattr(nas, "_random_ops", lambda _rng: ARCH_A)
    original = nas.load_final_test

    def checked_final_load(path: Path, table: nas.ValidationTable) -> tuple[dict, dict]:
        events = [json.loads(line) for line in output.read_text().splitlines()]
        assert [event["type"] for event in events] == ["header", "run", "run"]
        assert all(event["result"]["completed"] == 2 for event in events[1:])
        return original(path, table)

    monkeypatch.setattr(nas, "load_final_test", checked_final_load)
    summary = nas.run_panel(
        validation_file=validation_file,
        test_file=test_file,
        output=output,
        seeds=[0, 1],
        methods=["random_cpu"],
        datasets=["cifar10"],
        budget=2,
        require_full_data=False,
    )
    assert summary["status"] == "complete"
    assert len(summary["selected_test"]) == 2
    assert all(item["selected_test_error_mean"] == 20.0 for item in summary["selected_test"])
    assert output.with_suffix(".summary.json").is_file()
