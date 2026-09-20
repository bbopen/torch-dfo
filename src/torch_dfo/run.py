"""Synchronous, budgeted runs for scalar torch-dfo optimizers."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import torch

from torch_dfo.base import BaseOptimizer

Objective = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CandidateBatch:
    """Candidates reserved for one fixed-repeat evaluation batch."""

    ids: tuple[str, ...]
    x: torch.Tensor
    repeats: int
    evaluator_id: str


@dataclass
class EvaluationResult:
    """Raw scalar objective values for one candidate batch."""

    ids: tuple[str, ...]
    values: torch.Tensor
    attempted: torch.Tensor | None = None
    completed: torch.Tensor | None = None
    error: str | None = None
    accounting_uncertain: bool = False


@dataclass(frozen=True)
class SearchResult:
    """Detached result data from a search run."""

    best_x: torch.Tensor | None
    best_value: torch.Tensor | None
    best_id: str | None
    raw_results: tuple[EvaluationResult, ...]
    scheduled_evals: int
    charged_evals: int
    attempted_evals: int
    completed_evals: int
    reserved_evals: int
    max_evals: int
    stop_reason: str | None
    failures: tuple[str, ...]
    accounting_uncertain: bool
    run_id: str
    lineage: tuple[str, ...]


@dataclass
class _PendingBatch:
    public: CandidateBatch
    private_x: torch.Tensor
    reservation: int


class RandomSearch(BaseOptimizer):
    """Uniform random search that permits a final partial batch."""

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float],
        pop_size: int = 32,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ):
        if pop_size < 1:
            raise ValueError("RandomSearch requires pop_size >= 1")
        super().__init__(dim, bounds, pop_size, device, dtype, seed)

    def ask(self, count: int | None = None) -> torch.Tensor:
        """Sample ``count`` candidates, or one full population."""
        count = self.pop_size if count is None else count
        if count < 1 or count > self.pop_size:
            raise ValueError(f"count must be in [1, {self.pop_size}]")
        candidates = self._rand(count, self.dim)
        candidates = candidates * (self.ub - self.lb) + self.lb
        self.population[:count].copy_(candidates)
        self._generation += 1
        return candidates.clone()

    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Record scalar fitness and update the best observed candidate."""
        if candidates.ndim != 2 or candidates.shape[1] != self.dim:
            raise ValueError(f"candidates must have shape (n, {self.dim})")
        if fitness.shape != (candidates.shape[0],):
            raise ValueError("fitness must have one value per candidate")
        count = candidates.shape[0]
        self.population[:count].copy_(candidates)
        self.fitness[:count].copy_(fitness)
        self._update_best(candidates, fitness)


class SearchRun:
    """Drive scalar, synchronous minimization with one pending batch.

    This class supports continuous box-bounded scalar objectives. It does not
    support constraints, async execution, or adaptive repeats.
    """

    _SCHEMA = 1

    def __init__(
        self,
        optimizer: BaseOptimizer,
        max_evals: int,
        evaluator_id: str = "tensor",
        repeats: int = 1,
    ):
        if not isinstance(optimizer, BaseOptimizer):
            raise TypeError("optimizer must be a torch_dfo BaseOptimizer")
        if isinstance(max_evals, bool) or not isinstance(max_evals, int) or max_evals < 0:
            raise ValueError("max_evals must be a non-negative integer")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise ValueError("evaluator_id must be a non-empty string")

        self.optimizer = optimizer
        self.max_evals = max_evals
        self.evaluator_id = evaluator_id
        self.repeats = repeats
        self._run_id = uuid4().hex
        self._candidate_stream_id = self._run_id
        self._lineage: tuple[str, ...] = ()
        self._next_candidate = 0
        self._pending: _PendingBatch | None = None
        self._state = "ready"
        self._stop_reason: str | None = None
        self._scheduled_evals = 0
        self._charged_evals = 0
        self._attempted_evals = 0
        self._completed_evals = 0
        self._reserved_evals = 0
        self._history: list[EvaluationResult] = []
        self._failures: list[str] = []
        self._accounting_uncertain = False
        self._best_x: torch.Tensor | None = None
        self._best_value: torch.Tensor | None = None
        self._best_id: str | None = None

    @property
    def done(self) -> bool:
        """Return true after the budget or a failure stops the run."""
        return self._state != "ready"

    @property
    def run_id(self) -> str:
        """Return this run's identity."""
        return self._run_id

    @property
    def pending(self) -> CandidateBatch | None:
        """Return the public pending batch, if one exists."""
        return None if self._pending is None else self._pending.public

    @property
    def remaining_evals(self) -> int:
        """Return logical evaluation budget not charged or reserved."""
        return self.max_evals - self._charged_evals - self._reserved_evals

    @property
    def result(self) -> SearchResult:
        """Return a detached snapshot of the run state."""
        return SearchResult(
            best_x=_clone_tensor(self._best_x),
            best_value=_clone_tensor(self._best_value),
            best_id=self._best_id,
            raw_results=tuple(_clone_result(item) for item in self._history),
            scheduled_evals=self._scheduled_evals,
            charged_evals=self._charged_evals,
            attempted_evals=self._attempted_evals,
            completed_evals=self._completed_evals,
            reserved_evals=self._reserved_evals,
            max_evals=self.max_evals,
            stop_reason=self._stop_reason,
            failures=tuple(self._failures),
            accounting_uncertain=self._accounting_uncertain,
            run_id=self._run_id,
            lineage=self._lineage,
        )

    def ask(self) -> CandidateBatch | None:
        """Reserve and return the next batch, or ``None`` when stopped."""
        if self._pending is not None:
            raise RuntimeError("tell the pending batch before asking again")
        if self.done:
            return None

        count = self._next_batch_size()
        if count == 0:
            self._stop("budget exhausted")
            return None
        try:
            proposed = (
                self.optimizer.ask(count)
                if isinstance(self.optimizer, RandomSearch)
                else self.optimizer.ask()
            )
        except Exception as error:
            self._fail_without_pending(f"optimizer ask failed: {error}")
            return None
        if proposed.shape != (count, self.optimizer.dim):
            self._fail_without_pending("optimizer ask returned the wrong candidate shape")
            return None
        same_device = proposed.device.type == self.optimizer.device.type
        if not same_device or proposed.dtype != self.optimizer.dtype:
            self._fail_without_pending("optimizer ask returned the wrong device or dtype")
            return None

        private_x = proposed.detach().clone()
        ids = tuple(
            f"{self._candidate_stream_id}:{self._next_candidate + index}" for index in range(count)
        )
        public = CandidateBatch(ids, private_x.clone(), self.repeats, self.evaluator_id)
        reservation = count * self.repeats
        self._pending = _PendingBatch(public, private_x, reservation)
        self._scheduled_evals += reservation
        self._reserved_evals = reservation
        self._next_candidate += count
        return public

    def step(self, objective: Objective) -> None:
        """Evaluate and tell one batch through a deterministic batched objective."""
        if not callable(objective):
            raise TypeError("objective must be callable")
        if self.pending is None:
            self.ask()
        if self._pending is None:
            return
        pending = self._pending
        if not torch.equal(pending.public.x, pending.private_x):
            self._failure(self._blank(pending, "candidate values changed before evaluation"), 0)
            return

        count = pending.private_x.shape[0]
        values = torch.full(
            (count, self.repeats),
            float("nan"),
            device=self.optimizer.device,
            dtype=self.optimizer.dtype,
        )
        attempted = torch.zeros_like(values, dtype=torch.bool)
        completed = torch.zeros_like(values, dtype=torch.bool)
        for repeat in range(self.repeats):
            execution_x = pending.private_x.detach().clone()
            try:
                output = objective(execution_x)
            except Exception as error:
                self._failure(
                    self._make_result(
                        pending,
                        values,
                        attempted,
                        completed,
                        f"objective raised: {error}",
                        True,
                    ),
                    pending.reservation,
                    True,
                )
                return
            attempted[:, repeat] = True
            completed[:, repeat] = True
            try:
                observed = self._objective_values(output, count)
            except (TypeError, ValueError) as error:
                self._failure(
                    self._make_result(
                        pending,
                        values,
                        attempted,
                        completed,
                        f"objective returned invalid values: {error}",
                    ),
                    _count(attempted),
                )
                return
            values[:, repeat] = observed
            if not torch.equal(execution_x, pending.private_x):
                self._failure(
                    self._make_result(
                        pending,
                        values,
                        attempted,
                        completed,
                        "objective changed its candidate input",
                    ),
                    _count(attempted),
                )
                return
            if not torch.isfinite(observed).all():
                self._failure(
                    self._make_result(
                        pending,
                        values,
                        attempted,
                        completed,
                        "objective returned non-finite values",
                    ),
                    _count(attempted),
                )
                return
        self.tell(self._make_result(pending, values, attempted, completed))

    def tell(
        self,
        result: EvaluationResult | CandidateBatch,
        values: torch.Tensor | None = None,
    ) -> None:
        """Record an external result, or values for a returned batch.

        ``tell(batch, values)`` attests that each scheduled call completed.
        For one repeat, values may have shape ``(candidate,)``.
        """
        pending = self._need_pending()
        if not torch.equal(pending.public.x, pending.private_x):
            self._failure(
                self._blank(pending, "candidate values changed before tell"),
                pending.reservation,
                True,
            )
            raise ValueError("candidate values changed before tell")
        if isinstance(result, CandidateBatch):
            if values is None:
                raise TypeError("values are required when telling a CandidateBatch")
            if self.repeats == 1 and values.shape == (len(result.ids),):
                values = values.unsqueeze(1)
            result = EvaluationResult(result.ids, values)
        elif values is not None:
            raise TypeError("values may only accompany a CandidateBatch")
        elif not isinstance(result, EvaluationResult):
            raise TypeError("tell requires an EvaluationResult or CandidateBatch")

        try:
            observed = self._canonical(result, pending)
        except (TypeError, ValueError) as error:
            self._failure(
                self._blank(pending, f"invalid evaluation result: {error}"),
                pending.reservation,
                True,
            )
            raise ValueError(f"invalid evaluation result: {error}") from error

        assert observed.attempted is not None
        assert observed.completed is not None
        charge = (
            pending.reservation if observed.accounting_uncertain else _count(observed.attempted)
        )
        if observed.accounting_uncertain or observed.error is not None:
            self._failure(observed, charge, observed.accounting_uncertain)
            return
        if not bool(observed.completed.all()):
            self._failure(observed, charge)
            return
        if not torch.isfinite(observed.values).all():
            observed.error = "evaluation returned non-finite values"
            self._failure(observed, charge)
            return

        self._settle(observed, charge)
        aggregate = observed.values.mean(dim=1)
        self._update_best(pending.private_x, pending.public.ids, aggregate)
        try:
            self.optimizer.tell(pending.private_x.clone(), aggregate)
        except Exception as error:
            self._failures.append(f"optimizer tell failed: {error}")
            self._stop(self._failures[-1], failed=True)
            return
        if self._next_batch_size() == 0:
            self._stop("budget exhausted")

    def state_dict(self) -> dict[str, Any]:
        """Return a quiescent checkpoint for trusted ``torch.save`` use."""
        if self._pending is not None:
            raise RuntimeError("checkpointing a pending batch is not supported")
        if self._state == "failed":
            raise RuntimeError("checkpointing a failed run is not supported")
        return {
            "schema": self._SCHEMA,
            "header": _header(self.optimizer),
            "optimizer": copy.deepcopy(self.optimizer),
            "max_evals": self.max_evals,
            "evaluator_id": self.evaluator_id,
            "repeats": self.repeats,
            "run_id": self._run_id,
            "candidate_stream_id": self._candidate_stream_id,
            "lineage": self._lineage,
            "next_candidate": self._next_candidate,
            "state": self._state,
            "stop_reason": self._stop_reason,
            "counters": (
                self._scheduled_evals,
                self._charged_evals,
                self._attempted_evals,
                self._completed_evals,
            ),
            "history": tuple(_clone_result(item) for item in self._history),
            "failures": tuple(self._failures),
            "accounting_uncertain": self._accounting_uncertain,
            "best_x": _clone_tensor(self._best_x),
            "best_value": _clone_tensor(self._best_value),
            "best_id": self._best_id,
            "pending": None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load a compatible checkpoint as a new run fork."""
        if self._pending is not None:
            raise RuntimeError("cannot load a checkpoint while a batch is pending")
        if state.get("schema") != self._SCHEMA or state.get("pending") is not None:
            raise ValueError("checkpoint is not quiescent or has an unsupported schema")
        saved = state.get("optimizer")
        if not isinstance(saved, BaseOptimizer):
            raise ValueError("checkpoint has no compatible optimizer")
        if not _same_optimizer(saved, self.optimizer) or not _valid_header(
            state.get("header"), saved
        ):
            raise ValueError("checkpoint optimizer configuration does not match this run")
        if (
            state.get("max_evals") != self.max_evals
            or state.get("evaluator_id") != self.evaluator_id
            or state.get("repeats") != self.repeats
        ):
            raise ValueError("checkpoint run settings do not match this run")
        if state.get("state") not in ("ready", "stopped"):
            raise ValueError("checkpoint is not resumable")

        run_id = state.get("run_id")
        stream_id = state.get("candidate_stream_id")
        lineage = state.get("lineage")
        counters = state.get("counters")
        history = state.get("history")
        failures = state.get("failures")
        if not isinstance(run_id, str) or not isinstance(stream_id, str):
            raise ValueError("checkpoint has invalid identities")
        if not isinstance(lineage, tuple) or not all(isinstance(item, str) for item in lineage):
            raise ValueError("checkpoint has invalid lineage")
        if not _is_int(state.get("next_candidate")) or not _valid_counters(counters):
            raise ValueError("checkpoint has invalid counters")
        if not isinstance(history, tuple) or not all(
            isinstance(item, EvaluationResult) for item in history
        ):
            raise ValueError("checkpoint has invalid result history")
        if not isinstance(failures, tuple) or not all(isinstance(item, str) for item in failures):
            raise ValueError("checkpoint has invalid failures")
        best_x = state.get("best_x")
        best_value = state.get("best_value")
        best_id = state.get("best_id")
        if (best_x is None) != (best_value is None):
            raise ValueError("checkpoint best candidate and value disagree")
        if best_x is not None and (
            not isinstance(best_x, torch.Tensor) or best_x.shape != (saved.dim,)
        ):
            raise ValueError("checkpoint has an invalid best candidate")
        if best_value is not None and (
            not isinstance(best_value, torch.Tensor) or best_value.ndim != 0
        ):
            raise ValueError("checkpoint has an invalid best value")
        if best_id is not None and not isinstance(best_id, str):
            raise ValueError("checkpoint has an invalid best candidate ID")
        if not isinstance(state.get("accounting_uncertain"), bool):
            raise ValueError("checkpoint has invalid accounting state")

        scheduled, charged, attempted, completed = cast(tuple[int, int, int, int], counters)
        if charged > self.max_evals or attempted > charged or completed > attempted:
            raise ValueError("checkpoint has inconsistent evaluation counters")
        self.optimizer = copy.deepcopy(saved)
        self._run_id = uuid4().hex
        self._candidate_stream_id = stream_id
        self._lineage = (*lineage, run_id)
        self._next_candidate = state["next_candidate"]
        self._state = state["state"]
        self._stop_reason = state.get("stop_reason")
        self._scheduled_evals = scheduled
        self._charged_evals = charged
        self._attempted_evals = attempted
        self._completed_evals = completed
        self._reserved_evals = 0
        self._history = [_clone_result(item) for item in history]
        self._failures = list(failures)
        self._accounting_uncertain = state["accounting_uncertain"]
        self._best_x = _clone_tensor(best_x)
        self._best_value = _clone_tensor(best_value)
        self._best_id = best_id

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> SearchRun:
        """Create a new run fork from a dictionary returned by ``state_dict``."""
        optimizer = state.get("optimizer")
        if not isinstance(optimizer, BaseOptimizer):
            raise ValueError("checkpoint has no compatible optimizer")
        max_evals = state.get("max_evals")
        evaluator_id = state.get("evaluator_id")
        repeats = state.get("repeats")
        if (
            not _is_int(max_evals)
            or not isinstance(evaluator_id, str)
            or not isinstance(repeats, int)
            or isinstance(repeats, bool)
        ):
            raise ValueError("checkpoint has invalid run settings")
        max_evals = cast(int, max_evals)
        run = cls(
            copy.deepcopy(optimizer),
            max_evals,
            evaluator_id,
            repeats,
        )
        run.load_state_dict(state)
        return run

    def _next_batch_size(self) -> int:
        if isinstance(self.optimizer, RandomSearch):
            return min(self.optimizer.pop_size, self.remaining_evals // self.repeats)
        required = self.optimizer.pop_size * self.repeats
        return self.optimizer.pop_size if required <= self.remaining_evals else 0

    def _objective_values(self, output: torch.Tensor, count: int) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("objective must return a torch.Tensor")
        if output.shape != (count,):
            raise ValueError(f"expected shape {(count,)}, got {tuple(output.shape)}")
        if output.is_complex():
            raise ValueError("objective values must be real")
        return output.detach().to(self.optimizer.device, self.optimizer.dtype).clone()

    def _canonical(self, result: EvaluationResult, pending: _PendingBatch) -> EvaluationResult:
        ids = tuple(result.ids)
        expected = pending.public.ids
        expected_shape: tuple[int, int] = (len(expected), self.repeats)
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("result IDs do not match the pending batch")
        if not isinstance(result.values, torch.Tensor) or result.values.shape != expected_shape:
            raise ValueError(f"result values must have shape {expected_shape}")
        if result.values.is_complex():
            raise ValueError("result values must be real")
        mask_shape = cast(tuple[int, int], expected_shape)
        attempted = _mask(result.attempted, mask_shape, self.optimizer.device, "attempted")
        completed = _mask(result.completed, mask_shape, self.optimizer.device, "completed")
        if bool((completed & ~attempted).any()):
            raise ValueError("completed calls must also be attempted")
        if result.error is not None and not isinstance(result.error, str):
            raise TypeError("result error must be a string or None")
        if not isinstance(result.accounting_uncertain, bool):
            raise TypeError("accounting_uncertain must be a bool")
        index = {candidate_id: position for position, candidate_id in enumerate(ids)}
        order = torch.tensor(
            [index[candidate_id] for candidate_id in expected],
            dtype=torch.long,
            device=result.values.device,
        )
        return EvaluationResult(
            expected,
            result.values.detach()
            .index_select(0, order)
            .to(
                self.optimizer.device,
                self.optimizer.dtype,
            )
            .clone(),
            attempted.index_select(0, order).to(self.optimizer.device),
            completed.index_select(0, order).to(self.optimizer.device),
            result.error,
            result.accounting_uncertain,
        )

    def _make_result(
        self,
        pending: _PendingBatch,
        values: torch.Tensor,
        attempted: torch.Tensor,
        completed: torch.Tensor,
        error: str | None = None,
        accounting_uncertain: bool = False,
    ) -> EvaluationResult:
        return EvaluationResult(
            pending.public.ids,
            values.detach().clone(),
            attempted.detach().clone(),
            completed.detach().clone(),
            error,
            accounting_uncertain,
        )

    def _blank(self, pending: _PendingBatch, error: str) -> EvaluationResult:
        shape = (pending.private_x.shape[0], self.repeats)
        values = torch.full(
            shape,
            float("nan"),
            device=self.optimizer.device,
            dtype=self.optimizer.dtype,
        )
        mask = torch.zeros(shape, device=self.optimizer.device, dtype=torch.bool)
        return self._make_result(pending, values, mask, mask, error)

    def _settle(self, result: EvaluationResult, charge: int) -> None:
        pending = self._need_pending()
        if charge < 0 or charge > pending.reservation:
            raise RuntimeError("invalid evaluation charge")
        self._charged_evals += charge
        self._attempted_evals += _count(result.attempted)
        self._completed_evals += _count(result.completed)
        self._history.append(_clone_result(result))
        self._pending = None
        self._reserved_evals = 0

    def _failure(
        self,
        result: EvaluationResult,
        charge: int,
        accounting_uncertain: bool = False,
    ) -> None:
        self._settle(result, charge)
        self._failures.append(result.error or "evaluation failed")
        self._accounting_uncertain = self._accounting_uncertain or accounting_uncertain
        self._stop(self._failures[-1], failed=True)

    def _fail_without_pending(self, reason: str) -> None:
        self._failures.append(reason)
        self._stop(reason, failed=True)

    def _stop(self, reason: str, *, failed: bool = False) -> None:
        self._stop_reason = reason
        self._state = "failed" if failed else "stopped"

    def _update_best(
        self,
        candidates: torch.Tensor,
        ids: tuple[str, ...],
        aggregate: torch.Tensor,
    ) -> None:
        index = int(aggregate.argmin().item())
        value = aggregate[index]
        if self._best_value is None or bool((value < self._best_value).item()):
            self._best_x = candidates[index].detach().clone()
            self._best_value = value.detach().clone()
            self._best_id = ids[index]

    def _need_pending(self) -> _PendingBatch:
        if self._pending is None:
            raise RuntimeError("there is no pending batch")
        if self.done:
            raise RuntimeError("the run has stopped")
        return self._pending


def minimize(
    objective: Objective,
    optimizer: BaseOptimizer,
    max_evals: int,
    evaluator_id: str = "tensor",
    repeats: int = 1,
) -> SearchResult:
    """Minimize a deterministic batched scalar objective through ``SearchRun``."""
    run = SearchRun(optimizer, max_evals, evaluator_id, repeats)
    while not run.done:
        run.step(objective)
    return run.result


def _clone_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _clone_result(result: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(
        tuple(result.ids),
        result.values.detach().clone(),
        _clone_tensor(result.attempted),
        _clone_tensor(result.completed),
        result.error,
        result.accounting_uncertain,
    )


def _count(mask: torch.Tensor | None) -> int:
    return 0 if mask is None else int(mask.sum().item())


def _mask(
    value: torch.Tensor | None,
    shape: tuple[int, int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != shape:
        raise ValueError(f"{name} must be a bool tensor with shape {shape}")
    return value.detach().clone()


def _header(optimizer: BaseOptimizer) -> dict[str, Any]:
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "dim": optimizer.dim,
        "pop_size": optimizer.pop_size,
        "device": str(optimizer.device),
        "dtype": str(optimizer.dtype),
        "lb": optimizer.lb.detach().clone(),
        "ub": optimizer.ub.detach().clone(),
    }


def _same_optimizer(left: BaseOptimizer, right: BaseOptimizer) -> bool:
    return (
        type(left) is type(right)
        and left.dim == right.dim
        and left.pop_size == right.pop_size
        and left.device == right.device
        and left.dtype == right.dtype
        and torch.equal(left.lb, right.lb)
        and torch.equal(left.ub, right.ub)
    )


def _valid_header(value: object, optimizer: BaseOptimizer) -> bool:
    if not isinstance(value, dict):
        return False
    expected = _header(optimizer)
    return all(
        key in value
        and (
            torch.equal(value[key], expected[key])
            if isinstance(expected[key], torch.Tensor) and isinstance(value[key], torch.Tensor)
            else value[key] == expected[key]
        )
        for key in expected
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_counters(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 4 and all(_is_int(item) for item in value)
