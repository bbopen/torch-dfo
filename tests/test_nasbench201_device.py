"""Focused checks for the device-resident NASBench201 trial scorer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.nasbench201 import decode_logits, tie_priority, trial_index
from benchmarks.nasbench201_device import (
    POWERS,
    DeviceResidentScorer,
    ValidationTable,
    make_trial_schedule,
    rank_by_category,
)


def _single_architecture_table(seed: int) -> ValidationTable:
    ops = tuple(order[0] for order in tie_priority(seed))
    code = sum(op * weight for op, weight in zip(ops, POWERS, strict=True))
    mapping = np.full(5**6, -1, dtype=np.int64)
    mapping[code] = 0
    return ValidationTable(
        dataset="cifar10",
        ops=(ops,),
        values=np.array([[11.0, 12.0, 13.0]], dtype=np.float64),
        counts=np.array([3], dtype=np.int64),
        id_by_code=mapping,
    )


def test_tie_priority_is_a_permutation_for_each_edge() -> None:
    ranks = rank_by_category(tie_priority(7))
    assert ranks.shape == (6, 5)
    assert all(sorted(row.tolist()) == list(range(5)) for row in ranks)


def test_duplicate_candidates_get_distinct_private_trial_visits() -> None:
    seed = 7
    table = _single_architecture_table(seed)
    schedule = make_trial_schedule(table, seed, budget=40)
    scorer = DeviceResidentScorer(table, schedule, seed, "cpu", budget=40)
    logits = torch.zeros((20, 30), dtype=torch.float64)
    expected_ops, tied_edges = decode_logits(logits[0].tolist(), tie_priority(seed))
    assert expected_ops == table.ops[0]
    assert tied_edges == 6

    first = scorer(logits).tolist()
    second = scorer(logits).tolist()
    trace = scorer.trace()
    expected_trials = [trial_index("cifar10", seed, table.ops[0], i, 3) for i in range(40)]
    expected_errors = [table.values[0, trial] for trial in expected_trials]
    assert first + second == expected_errors
    assert trace["architecture_ids"] == [0] * 40
    assert trace["visit_indices"] == list(range(40))
    assert trace["trial_indices"] == expected_trials
    assert trace["tied_edges"] == [6] * 40


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_decoder_agrees_with_shared_decoder(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seed = 13
    table = _single_architecture_table(seed)
    schedule = make_trial_schedule(table, seed, budget=20)
    scorer = DeviceResidentScorer(table, schedule, seed, device, budget=20)
    row = torch.zeros((20, 6, 5), dtype=torch.float64, device=device)
    ops = table.ops[0]
    for edge, operation in enumerate(ops):
        row[:, edge, operation] = 1.0
    ids, ties = scorer.decode_ids(row.reshape(20, 30))
    assert ids.cpu().tolist() == [0] * 20
    assert ties.cpu().tolist() == [0] * 20
    expected_ops, tied_edges = decode_logits(row[0].cpu().reshape(-1).tolist(), tie_priority(seed))
    assert expected_ops == ops
    assert tied_edges == 0
