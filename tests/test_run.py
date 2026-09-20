"""Contract tests for the synchronous SearchRun API."""

from __future__ import annotations

import pytest
import torch

from torch_dfo import (
    CMAES,
    SHADE,
    CandidateBatch,
    EvaluationResult,
    RandomSearch,
    SearchRun,
    minimize,
)


def _cma(seed: int = 4) -> CMAES:
    return CMAES(dim=3, bounds=2.0, pop_size=4, device="cpu", seed=seed)


def _sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=1)


def test_ask_reserves_one_batch_and_rejects_a_second_ask() -> None:
    run = SearchRun(_cma(), max_evals=8)

    batch = run.ask()

    assert isinstance(batch, CandidateBatch)
    assert len(batch.ids) == 4
    assert batch.x.shape == (4, 3)
    assert run.result.reserved_evals == 4
    with pytest.raises(RuntimeError, match="pending"):
        run.ask()


def test_cma_does_not_draw_when_its_full_generation_cannot_fit() -> None:
    optimizer = _cma()
    peer = _cma()
    run = SearchRun(optimizer, max_evals=3)

    assert run.ask() is None
    assert run.done
    assert run.result.charged_evals == 0
    assert torch.equal(optimizer.ask(), peer.ask())


def test_random_search_consumes_the_final_residual() -> None:
    optimizer = RandomSearch(dim=2, bounds=1.0, pop_size=5, device="cpu", seed=9)
    run = SearchRun(optimizer, max_evals=7)

    first = run.ask()
    assert first is not None
    run.tell(first, _sphere(first.x))
    second = run.ask()

    assert second is not None
    assert second.x.shape == (2, 2)
    run.tell(second, _sphere(second.x))
    assert run.done
    assert run.result.charged_evals == 7


def test_step_records_fixed_repeat_values_and_aggregate() -> None:
    run = SearchRun(_cma(), max_evals=8, repeats=2)
    calls = 0

    def objective(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return _sphere(x) + calls

    run.step(objective)
    result = run.result

    assert calls == 2
    assert result.charged_evals == 8
    assert result.completed_evals == 8
    assert result.best_x is not None
    assert result.best_value is not None
    assert result.raw_results[0].values.shape == (4, 2)
    assert torch.equal(result.raw_results[0].values[:, 1], result.raw_results[0].values[:, 0] + 1)


def test_step_stops_before_call_when_public_candidates_change() -> None:
    run = SearchRun(_cma(), max_evals=8)
    batch = run.ask()
    assert batch is not None
    batch.x.add_(1)
    calls = 0

    def objective(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return _sphere(x)

    run.step(objective)
    result = run.result

    assert calls == 0
    assert result.charged_evals == 0
    assert result.best_x is None
    assert "changed before evaluation" in result.stop_reason


def test_step_charges_known_calls_when_objective_mutates_its_input() -> None:
    run = SearchRun(_cma(), max_evals=16, repeats=2)

    def objective(x: torch.Tensor) -> torch.Tensor:
        values = _sphere(x)
        x.add_(1)
        return values

    run.step(objective)
    result = run.result

    assert result.charged_evals == 4
    assert result.attempted_evals == 4
    assert result.completed_evals == 4
    assert result.best_x is None
    assert "changed its candidate input" in result.stop_reason


def test_reordered_external_results_follow_the_original_candidate_order() -> None:
    left = SearchRun(_cma(), max_evals=8)
    right = SearchRun(_cma(), max_evals=8)
    left_batch = left.ask()
    right_batch = right.ask()
    assert left_batch is not None and right_batch is not None
    values = torch.tensor([[4.0], [1.0], [3.0], [2.0]], dtype=torch.float64)

    right.tell(EvaluationResult(ids=right_batch.ids, values=values))
    order = torch.tensor([2, 0, 3, 1])
    left.tell(
        EvaluationResult(
            ids=tuple(left_batch.ids[index] for index in order.tolist()),
            values=values.index_select(0, order),
        ),
    )

    left_next = left.ask()
    right_next = right.ask()
    assert left_next is not None and right_next is not None
    assert torch.equal(left_next.x, right_next.x)


def test_invalid_result_ids_charge_the_reservation_without_an_optimizer_update() -> None:
    optimizer = _cma()
    run = SearchRun(optimizer, max_evals=8)
    batch = run.ask()
    assert batch is not None
    original_mean = optimizer.mean.clone()

    with pytest.raises(ValueError, match="invalid evaluation result"):
        run.tell(EvaluationResult(ids=("wrong",) * 4, values=torch.zeros(4, 1)))

    assert run.done
    assert run.result.charged_evals == 4
    assert torch.equal(optimizer.mean, original_mean)


def test_nonfinite_values_stop_without_an_incumbent() -> None:
    run = SearchRun(_cma(), max_evals=8)
    batch = run.ask()
    assert batch is not None

    run.tell(batch, torch.full((4, 1), float("nan")))
    result = run.result

    assert result.charged_evals == 4
    assert result.best_x is None
    assert "non-finite" in result.stop_reason


def test_checkpoint_forks_run_identity_and_preserves_next_batch() -> None:
    run = SearchRun(_cma(), max_evals=12)
    first = run.ask()
    assert first is not None
    run.tell(first, _sphere(first.x).unsqueeze(1))
    state = run.state_dict()
    restored = SearchRun.from_checkpoint(state)

    original_next = run.ask()
    restored_next = restored.ask()

    assert original_next is not None and restored_next is not None
    assert restored.run_id != run.run_id
    assert run.run_id in restored.result.lineage
    assert original_next.ids == restored_next.ids
    assert torch.equal(original_next.x, restored_next.x)


def test_pending_checkpoint_is_rejected() -> None:
    run = SearchRun(_cma(), max_evals=8)
    assert run.ask() is not None

    with pytest.raises(RuntimeError, match="pending"):
        run.state_dict()


def test_minimize_uses_the_same_run_contract() -> None:
    result = minimize(_sphere, RandomSearch(2, 1.0, pop_size=3, seed=2), max_evals=7)

    assert result.charged_evals == 7
    assert result.best_x is not None


@pytest.mark.parametrize("algorithm", ["cma", "shade", "random"])
def test_run_steps_each_supported_algorithm_on_available_devices(
    algorithm: str,
    device: torch.device,
    default_dtype: torch.dtype,
) -> None:
    if algorithm == "cma":
        optimizer = CMAES(2, 1.0, pop_size=4, device=device, dtype=default_dtype, seed=3)
    elif algorithm == "shade":
        optimizer = SHADE(2, 1.0, pop_size=4, device=device, dtype=default_dtype, seed=3)
    else:
        optimizer = RandomSearch(2, 1.0, pop_size=4, device=device, dtype=default_dtype, seed=3)

    run = SearchRun(optimizer, max_evals=4)
    run.step(_sphere)

    assert run.result.charged_evals == 4
    assert run.result.best_x is not None
