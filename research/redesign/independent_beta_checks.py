"""Independent SearchRun contract probes for the torch-dfo beta."""

from __future__ import annotations

import io

import torch

from torch_dfo import CMAES, SHADE, EvaluationResult, RandomSearch, SearchRun


def sphere(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(dim=1)


def cma(seed: int = 17) -> CMAES:
    return CMAES(dim=3, bounds=(-1.0, 1.0), pop_size=4, device="cpu", seed=seed)


def test_caps_and_repeats() -> None:
    too_small = SearchRun(cma(), max_evals=3)
    peer = cma()
    assert too_small.ask() is None
    assert too_small.result.charged_evals == 0
    assert torch.equal(too_small.optimizer.ask(), peer.ask())

    exact = SearchRun(cma(), max_evals=4)
    exact.step(sphere)
    result = exact.result
    assert exact.done and result.charged_evals == result.attempted_evals == 4
    assert result.reserved_evals == 0

    calls = 0

    def repeated(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return sphere(x) + calls

    repeat_run = SearchRun(cma(), max_evals=9, repeats=2)
    repeat_run.step(repeated)
    result = repeat_run.result
    assert calls == 2
    assert result.charged_evals == result.completed_evals == 8
    assert result.reserved_evals == 0 and repeat_run.ask() is None
    assert result.raw_results[0].values.shape == (4, 2)
    assert torch.equal(result.raw_results[0].values[:, 1], result.raw_results[0].values[:, 0] + 1)

    random_run = SearchRun(RandomSearch(3, (-1.0, 1.0), pop_size=5, seed=17), max_evals=7)
    while not random_run.done:
        random_run.step(sphere)
    assert random_run.result.charged_evals == 7
    assert [len(row.ids) for row in random_run.result.raw_results] == [5, 2]


def test_failure_cost_and_terminal_state() -> None:
    run = SearchRun(cma(), max_evals=8)

    def fails(_x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("simulator failed")

    run.step(fails)
    result = run.result
    assert run.done and result.charged_evals == 4
    assert result.attempted_evals == 0 and result.accounting_uncertain
    assert result.best_x is None and result.best_value is None
    assert "simulator failed" in result.stop_reason

    class MutatingOptimizer(RandomSearch):
        def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
            self.population[0, 0] = 99.0
            raise RuntimeError("update failed after mutation")

    broken = SearchRun(MutatingOptimizer(3, (-1.0, 1.0), pop_size=4, seed=17), max_evals=8)
    broken.step(sphere)
    result = broken.result
    assert broken.done and result.charged_evals == result.attempted_evals == 4
    assert float(broken.optimizer.population[0, 0]) == 99.0
    assert "update failed after mutation" in result.stop_reason
    try:
        broken.state_dict()
    except RuntimeError:
        pass
    else:
        raise AssertionError("poisoned optimizer produced a resumable checkpoint")


def test_ids_and_nonfinite() -> None:
    for ids in (("old:0", "old:1", "old:2", "old:3"), ("duplicate",) * 4):
        run = SearchRun(cma(), max_evals=8)
        run.ask()
        try:
            run.tell(EvaluationResult(ids=ids, values=torch.ones(4, 1)))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid IDs were accepted")
        assert run.done and run.result.charged_evals == 4
        assert run.result.best_x is None

    run = SearchRun(cma(), max_evals=4)
    batch = run.ask()
    assert batch is not None
    forged = tuple(f"forged:{index}" for index in range(4))
    try:
        batch.ids = forged
    except (AttributeError, TypeError):
        pass
    else:
        try:
            run.tell(batch, torch.arange(4, dtype=torch.float64))
        except ValueError:
            pass
        else:
            raise AssertionError("mutated candidate IDs were accepted")

    nonfinite = SearchRun(cma(), max_evals=4)
    batch = nonfinite.ask()
    assert batch is not None
    nonfinite.tell(batch, torch.full((4,), float("nan")))
    result = nonfinite.result
    assert nonfinite.done and result.charged_evals == 4
    assert result.best_x is None and "non-finite" in result.stop_reason


def test_global_rng_and_checkpoint() -> None:
    torch.manual_seed(1203)
    global_state = torch.get_rng_state().clone()
    run = SearchRun(cma(), max_evals=12)
    run.step(sphere)
    assert torch.equal(torch.get_rng_state(), global_state)

    makers = (
        lambda: CMAES(3, (-1.0, 1.0), pop_size=4, seed=17),
        lambda: SHADE(3, (-1.0, 1.0), pop_size=8, seed=17),
        lambda: RandomSearch(3, (-1.0, 1.0), pop_size=5, seed=17),
    )
    for make in makers:
        original = SearchRun(make(), max_evals=make().pop_size * 3)
        original.step(sphere)
        checkpoint = original.state_dict()
        fresh = SearchRun(make(), max_evals=make().pop_size * 3)
        fresh_id = fresh.run_id
        fresh.load_state_dict(checkpoint)
        assert fresh.run_id != fresh_id
        assert original.run_id in fresh.result.lineage
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        fork = SearchRun.from_checkpoint(torch.load(buffer, weights_only=False))
        assert original.run_id != fork.run_id
        assert original.run_id in fork.result.lineage
        left = original.ask()
        right = fork.ask()
        assert left is not None and right is not None
        assert left.ids == right.ids and torch.equal(left.x, right.x)
        fresh_next = fresh.ask()
        assert fresh_next is not None
        assert fresh_next.ids == left.ids and torch.equal(fresh_next.x, left.x)
        original.tell(left, sphere(left.x))
        fork.tell(right, sphere(right.x))
        left_next = original.ask()
        right_next = fork.ask()
        assert left_next is not None and right_next is not None
        assert left_next.ids == right_next.ids and torch.equal(left_next.x, right_next.x)
        right_next.x[0, 0] = 99.0
        assert float(left_next.x[0, 0]) != 99.0

        self_source = SearchRun(make(), max_evals=make().pop_size * 3)
        self_source.step(sphere)
        previous_id = self_source.run_id
        self_source.load_state_dict(self_source.state_dict())
        assert self_source.run_id != previous_id
        assert previous_id in self_source.result.lineage


if __name__ == "__main__":
    for label, check in (
        ("caps and repeats", test_caps_and_repeats),
        ("failure cost and terminal state", test_failure_cost_and_terminal_state),
        ("IDs and nonfinite values", test_ids_and_nonfinite),
        ("RNG and checkpoint", test_global_rng_and_checkpoint),
    ):
        check()
        print(f"PASS: {label}")
