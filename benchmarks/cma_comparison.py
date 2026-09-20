#!/usr/bin/env python3
"""Run the fixed 16-D CMA comparison with counted objective calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch_dfo import CMAES, RandomSearch, SearchRun  # noqa: E402
from torch_dfo.benchmarks.classical import rastrigin, sphere  # noqa: E402

DIM = 16
POP_SIZE = 32
DEFAULT_BUDGET = 3200
DEFAULT_SEEDS = (11, 12, 13, 14, 15)
SIGMA = 0.6
GUARD = 1_000_000.0
FIXTURE_SEED = 20260920
EVOX_SHA = "1c242cc9533fbd7e2e0c0342456f4e0f52449e5d"
PYCMA_SHA = "48a821b166eceef544847a0b9b66113ac26f1530"
EVOX_FILE_SHA256 = "3c68f5f274af8fe9f85f7257a368a84b71800a4611c94ccae3b589bfca69cb0a"
PYCMA_FILE_SHA256 = "77076e97a1df36a9c744b4820adebc6acbf8288599263c119804827f89070700"
METHODS = (
    "torch_dfo_cpu",
    "torch_dfo_cuda",
    "evox_cuda",
    "pycma_cpu",
    "pycma_cuda_objective",
    "random_cpu",
)
TASKS = ("shifted_sphere", "rotated_ellipsoid", "rotated_rastrigin", "thermal_train")
TARGETS = {
    "shifted_sphere": 1e-8,
    "rotated_ellipsoid": 1e-4,
    "rotated_rastrigin": 1e-4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _source(path: Path, relative_file: str, expected_hash: str, expected_head: str) -> dict:
    path = path.resolve()
    file = path / relative_file
    if not file.is_file():
        raise FileNotFoundError(f"required upstream source is missing: {file}")
    actual_hash = _sha256(file)
    if actual_hash != expected_hash:
        raise RuntimeError(f"upstream source hash differs: {file}: {actual_hash}")
    head = _git_head(path)
    if head is not None and head != expected_head:
        raise RuntimeError(f"upstream Git HEAD differs: {path}: {head}")
    return {"path": str(path), "file": relative_file, "sha256": actual_hash, "git_head": head}


def _upstream_source(
    source: Path | None,
    package: str,
    relative_file: str,
    expected_hash: str,
    expected_head: str,
    version: str,
) -> dict:
    """Check pinned source bytes without forcing imports from a checkout."""
    if source is not None:
        return _source(source, relative_file, expected_hash, expected_head)
    installed = metadata.distribution(package)
    if installed.version != version:
        raise RuntimeError(f"{package} version differs: {installed.version}")
    file = Path(installed.locate_file(relative_file.removeprefix("src/"))).resolve()
    if not file.is_file() or _sha256(file) != expected_hash:
        raise RuntimeError(f"installed {package} source bytes differ: {file}")
    return {
        "distribution": package,
        "version": installed.version,
        "file": str(file),
        "sha256": expected_hash,
        "git_head": None,
    }


def _imported_source(symbol: object, expected_hash: str) -> dict:
    source_file = inspect.getsourcefile(symbol)
    if source_file is None:
        raise RuntimeError(f"cannot identify imported source for {symbol}")
    path = Path(source_file).resolve()
    actual = _sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"imported source differs from pinned bytes: {path}: {actual}")
    return {"file": str(path), "sha256": actual}


def _load_reference():
    path = ROOT / "examples" / "06_engineering_control.py"
    name = "torch_dfo_cma_comparison_thermal"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load thermal reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture() -> dict:
    """Build one CPU float64 fixture and record its exact bytes and values."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(FIXTURE_SEED)
    raw = torch.randn(DIM, DIM, dtype=torch.float64, generator=gen)
    rotation, upper = torch.linalg.qr(raw)
    signs = torch.where(torch.diagonal(upper) < 0, -1.0, 1.0)
    rotation = (rotation * signs).contiguous()
    shift = torch.linspace(0.5, 1.5, DIM, dtype=torch.float64).contiguous()
    digest = hashlib.sha256(shift.numpy().tobytes() + rotation.numpy().tobytes()).hexdigest()
    return {
        "seed": FIXTURE_SEED,
        "shift": shift.tolist(),
        "rotation": rotation.tolist(),
        "sha256_float64_le_shift_then_rotation": digest,
    }


@dataclass(frozen=True)
class Task:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    target: float
    device: torch.device
    oracle: Callable[[np.ndarray], float]


def make_task(name: str, device: torch.device, fixed: dict) -> Task:
    shift = torch.tensor(fixed["shift"], device=device, dtype=torch.float64)
    rotation = torch.tensor(fixed["rotation"], device=device, dtype=torch.float64)
    shift_np = np.asarray(fixed["shift"], dtype=np.float64)
    rotation_np = np.asarray(fixed["rotation"], dtype=np.float64)
    if name == "shifted_sphere":
        return Task(
            name,
            lambda x: sphere(x - shift),
            TARGETS[name],
            device,
            lambda x: float(np.sum((x - shift_np) ** 2)),
        )
    if name == "rotated_ellipsoid":
        weights = torch.logspace(0, 6, DIM, device=device, dtype=torch.float64)
        weights_np = np.logspace(0, 6, DIM, dtype=np.float64)
        return Task(
            name,
            lambda x: (((x - shift) @ rotation) ** 2 * weights).sum(-1),
            TARGETS[name],
            device,
            lambda x: float(np.sum(((x - shift_np) @ rotation_np) ** 2 * weights_np)),
        )
    if name == "rotated_rastrigin":
        return Task(
            name,
            lambda x: rastrigin((x - shift) @ rotation),
            TARGETS[name],
            device,
            lambda x: float(
                10 * DIM
                + np.sum(
                    ((x - shift_np) @ rotation_np) ** 2
                    - 10 * np.cos(2 * np.pi * ((x - shift_np) @ rotation_np))
                )
            ),
        )
    if name == "thermal_train":
        ref = _load_reference()
        train = ref.FrozenEvaluator("train")
        prepared = train.prepare(device, torch.float64, compile=False)
        return Task(
            name,
            prepared,
            float(ref.PASS_THRESHOLD),
            device,
            lambda x: float(train(torch.from_numpy(x.copy()).unsqueeze(0))[0].item()),
        )
    raise ValueError(f"unknown task: {name}")


class CountedObjective:
    """Reject invalid batches and track every observed candidate."""

    def __init__(self, task: Task, *, guard: float | None = None):
        self.task = task
        self.guard = guard
        self.batch_calls = 0
        self.attempted = 0
        self.completed = 0
        self.best_value = math.inf
        self.best_x: list[float] | None = None
        self.evals_to_target: int | None = None
        self.target_hit_elapsed_s: float | None = None
        self.guard_hits = 0
        self.start_time: float | None = None
        self.trace: list[dict] = []

    def partial(self) -> dict:
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "batch_calls": self.batch_calls,
            "best_value": None if self.best_x is None else self.best_value,
            "best_x": self.best_x,
            "evals_to_target": self.evals_to_target,
            "target_hit_elapsed_s": self.target_hit_elapsed_s,
            "guard_hits": self.guard_hits,
            "trace": self.trace,
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_calls += 1
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != DIM:
            raise ValueError(f"candidate batch must have shape (N, {DIM})")
        count = x.shape[0]
        self.attempted += count
        if count != POP_SIZE or x.dtype != torch.float64 or x.device != self.task.device:
            raise ValueError("candidate batch has wrong count, dtype, or device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("candidate batch has non-finite coordinates")
        if self.guard is not None:
            self.guard_hits += int((x.abs() >= self.guard).any(dim=1).sum().item())
            if self.guard_hits:
                raise ValueError("native candidate reached the wide bound guard")
        values = self.task.fn(x)
        if not isinstance(values, torch.Tensor) or values.shape != (count,):
            raise ValueError("objective must return one score per candidate")
        if (
            values.dtype != torch.float64
            or values.device != x.device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("objective returned wrong dtype, device, or non-finite scores")
        scores = values.detach().cpu().tolist()
        points = x.detach().cpu().tolist()
        for score, point in zip(scores, points, strict=True):
            self.completed += 1
            if score < self.best_value:
                self.best_value = score
                self.best_x = point
            if self.evals_to_target is None and self.best_value <= self.task.target:
                self.evals_to_target = self.completed
                if self.start_time is not None:
                    self.target_hit_elapsed_s = time.perf_counter() - self.start_time
        return values


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _row(objective: CountedObjective, start: float) -> dict:
    return {
        "generation": objective.batch_calls,
        "attempted": objective.attempted,
        "completed": objective.completed,
        "best_value": None if not math.isfinite(objective.best_value) else objective.best_value,
        "elapsed_s": time.perf_counter() - start,
    }


def _finish(
    task: Task, objective: CountedObjective, budget: int, timings: dict, trace: list[dict]
) -> dict:
    if (
        objective.attempted != budget
        or objective.completed != budget
        or objective.batch_calls != budget // POP_SIZE
    ):
        raise RuntimeError(
            f"incomplete objective accounting: {objective.attempted}/{objective.completed}/{budget}"
        )
    if objective.best_x is None:
        raise RuntimeError("no observed incumbent")
    oracle_value = task.oracle(np.asarray(objective.best_x, dtype=np.float64))
    relative_tolerance = 0.0 if task.name == "thermal_train" else 1e-12
    absolute_tolerance = 1e-10
    if not math.isfinite(oracle_value) or not math.isclose(
        objective.best_value,
        oracle_value,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise RuntimeError(f"CPU oracle mismatch: {objective.best_value} versus {oracle_value}")
    result = {
        "status": "ok",
        "attempted": objective.attempted,
        "completed": objective.completed,
        "batch_calls": objective.batch_calls,
        "best_value": objective.best_value,
        "best_x": objective.best_x,
        "target": task.target,
        "evals_to_target": objective.evals_to_target,
        "target_hit_elapsed_s": objective.target_hit_elapsed_s,
        "guard_hits": objective.guard_hits,
        "cpu_oracle_value": oracle_value,
        "oracle_relative_tolerance": relative_tolerance,
        "oracle_absolute_tolerance": absolute_tolerance,
        "timings_s": timings,
        "trace": trace,
    }
    if task.name == "thermal_train":
        ref = _load_reference()
        selected = torch.tensor(objective.best_x, dtype=torch.float64).unsqueeze(0)
        result["effective_actuation"] = ref.effective_actuation(selected)[0].tolist()
        heldout = ref.FrozenEvaluator("heldout")
        result["heldout_diagnostic"] = float(heldout(selected)[0].item())
        result["heldout_candidate_evaluations"] = 1
    return result


def _native(method: str, task: Task, seed: int, budget: int, objective: CountedObjective) -> dict:
    trace = objective.trace
    _sync(task.device)
    start = time.perf_counter()
    objective.start_time = start
    if method == "random_cpu":
        optimizer = RandomSearch(DIM, (-2.0, 2.0), POP_SIZE, task.device, torch.float64, seed)
    else:
        optimizer = CMAES(
            DIM,
            (-GUARD, GUARD),
            POP_SIZE,
            task.device,
            torch.float64,
            seed,
            SIGMA / (2 * GUARD),
            mirrored=False,
            active=False,
        )
    run = SearchRun(optimizer, max_evals=budget, evaluator_id=task.name, repeats=1)
    _sync(task.device)
    initialized = time.perf_counter()
    first_done = initialized
    while not run.done:
        run.step(objective)
        _sync(task.device)
        trace.append(_row(objective, start))
        if len(trace) == 1:
            first_done = time.perf_counter()
    total_end = time.perf_counter()
    result = run.result
    if result.failures or result.accounting_uncertain or result.stop_reason != "budget exhausted":
        raise RuntimeError(f"SearchRun failed: {result.stop_reason}; {result.failures}")
    if (
        any(
            value != budget
            for value in (
                result.scheduled_evals,
                result.charged_evals,
                result.attempted_evals,
                result.completed_evals,
            )
        )
        or result.reserved_evals
    ):
        raise RuntimeError("SearchRun counters differ from full budget")
    if result.best_x is None or result.best_value is None:
        raise RuntimeError("SearchRun returned no incumbent")
    if not math.isclose(float(result.best_value), objective.best_value, rel_tol=1e-12, abs_tol=0):
        raise RuntimeError("SearchRun incumbent differs from counted incumbent")
    timings = {
        "initialization": initialized - start,
        "generation_1": first_done - initialized,
        "remaining_generations": total_end - first_done,
        "total": total_end - start,
    }
    output = _finish(task, objective, budget, timings, trace)
    output["search_run"] = {
        "scheduled": result.scheduled_evals,
        "charged": result.charged_evals,
        "attempted": result.attempted_evals,
        "completed": result.completed_evals,
        "stop_reason": result.stop_reason,
        "failures": list(result.failures),
    }
    return output


def _evox(
    task: Task, seed: int, budget: int, source: Path | None, objective: CountedObjective
) -> dict:
    if task.device.type != "cuda":
        raise ValueError("EvoX comparison requires CUDA")
    _upstream_source(
        source,
        "evox",
        "src/evox/algorithms/so/es_variants/cma_es.py",
        EVOX_FILE_SHA256,
        EVOX_SHA,
        "1.4.0",
    )
    if source is not None:
        sys.path.insert(0, str(source.resolve() / "src"))
    from evox.algorithms.so.es_variants import CMAES as EvoXCMAES
    from evox.core import Problem
    from evox.workflows import EvalMonitor, StdWorkflow

    imported_source = _imported_source(EvoXCMAES, EVOX_FILE_SHA256)

    class CountedProblem(Problem):
        def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
            return objective(pop)

    trace = objective.trace
    old_dtype = torch.get_default_dtype()
    cuda_index = task.device.index if task.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index]):
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(seed)
            _sync(task.device)
            start = time.perf_counter()
            objective.start_time = start
            mean = torch.zeros(DIM, dtype=torch.float64, device=task.device)
            algorithm = EvoXCMAES(
                mean_init=mean, sigma=SIGMA, pop_size=POP_SIZE, device=task.device
            )
            monitor = EvalMonitor(
                full_fit_history=False,
                full_sol_history=False,
                history_device=task.device,
                device=task.device,
            )
            workflow = StdWorkflow(algorithm, CountedProblem(), monitor=monitor, device=task.device)
            _sync(task.device)
            initialized = time.perf_counter()
            first_done = initialized
            for generation in range(budget // POP_SIZE):
                if generation == 0:
                    workflow.init_step()
                else:
                    workflow.step()
                _sync(task.device)
                trace.append(_row(objective, start))
                if generation == 0:
                    first_done = time.perf_counter()
            total_end = time.perf_counter()
            if not math.isclose(
                float(monitor.get_best_fitness()), objective.best_value, rel_tol=1e-12, abs_tol=0
            ):
                raise RuntimeError("EvoX monitor incumbent differs from counted incumbent")
            timings = {
                "initialization": initialized - start,
                "generation_1": first_done - initialized,
                "remaining_generations": total_end - first_done,
                "total": total_end - start,
            }
        finally:
            torch.set_default_dtype(old_dtype)
    output = _finish(task, objective, budget, timings, trace)
    output["evox_mode"] = "unmodified_eager_StdWorkflow"
    output["imported_source"] = imported_source
    probe_path = ROOT / "research" / "redesign" / "results" / "evox-covariance-probe.json"
    output["evox_covariance_probe"] = {
        "path": str(probe_path.relative_to(ROOT)),
        "sha256": _sha256(probe_path),
        "finding": "stock_v1.4.0_rank_one_covariance_update_differs_on_CPU_and_CUDA",
        "reference_status": "unvalidated_mathematical_CMAES_reference",
    }
    return output


def _pycma(
    task: Task, seed: int, budget: int, source: Path | None, objective: CountedObjective
) -> dict:
    _upstream_source(
        source, "cma", "cma/evolution_strategy.py", PYCMA_FILE_SHA256, PYCMA_SHA, "4.5.0"
    )
    if source is not None:
        sys.path.insert(0, str(source.resolve()))
    import cma

    imported_source = _imported_source(cma.CMAEvolutionStrategy, PYCMA_FILE_SHA256)

    rng = np.random.RandomState(seed)
    trace = objective.trace
    _sync(task.device)
    start = time.perf_counter()
    objective.start_time = start
    options = {
        "popsize": POP_SIZE,
        "CMA_active": False,
        "CMA_mirrors": 0,
        "verbose": -9,
        "verb_log": 0,
        "seed": np.nan,
        "randn": rng.randn,
    }
    strategy = cma.CMAEvolutionStrategy(np.zeros(DIM, dtype=np.float64), SIGMA, options)
    _sync(task.device)
    initialized = time.perf_counter()
    first_done = initialized
    for generation in range(budget // POP_SIZE):
        candidates = strategy.ask()
        points = torch.from_numpy(np.asarray(candidates, dtype=np.float64)).to(task.device)
        scores = objective(points)
        strategy.tell(candidates, scores.detach().cpu().tolist())
        _sync(task.device)
        trace.append(_row(objective, start))
        if generation == 0:
            first_done = time.perf_counter()
    total_end = time.perf_counter()
    timings = {
        "initialization": initialized - start,
        "generation_1": first_done - initialized,
        "remaining_generations": total_end - first_done,
        "total": total_end - start,
    }
    output = _finish(task, objective, budget, timings, trace)
    output["imported_source"] = imported_source
    output["pycma_mode"] = (
        "CPU_strategy_CUDA_objective"
        if task.device.type == "cuda"
        else "CPU_strategy_CPU_objective"
    )
    output["pycma_stop_conditions_ignored_until_fixed_budget"] = True
    return output


def run_one(
    method: str,
    task_name: str,
    seed: int,
    budget: int,
    fixed: dict,
    evox_source: Path | None = None,
    pycma_source: Path | None = None,
) -> dict:
    if method not in METHODS or task_name not in TASKS:
        raise ValueError("unknown method or task")
    if budget < POP_SIZE or budget % POP_SIZE:
        raise ValueError(f"budget must be a positive multiple of {POP_SIZE}")
    device = torch.device(
        "cuda:0" if method in ("torch_dfo_cuda", "evox_cuda", "pycma_cuda_objective") else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA method requested but CUDA is unavailable")
    _sync(device)
    memory_start = None
    if device.type == "cuda":
        memory_start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    task = make_task(task_name, device, fixed)
    _sync(device)
    task_setup_s = time.perf_counter() - setup_start
    objective = CountedObjective(task, guard=GUARD if method.startswith("torch_dfo") else None)
    try:
        if method in ("torch_dfo_cpu", "torch_dfo_cuda", "random_cpu"):
            output = _native(method, task, seed, budget, objective)
        elif method == "evox_cuda":
            output = _evox(task, seed, budget, evox_source, objective)
        else:
            output = _pycma(task, seed, budget, pycma_source, objective)
    except Exception as error:
        raise PartialRunError(error, objective.partial()) from error
    output["timings_s"]["task_setup"] = task_setup_s
    output["timings_s"]["complete_total"] = task_setup_s + output["timings_s"]["total"]
    output["target_hit_complete_elapsed_s"] = (
        None
        if objective.target_hit_elapsed_s is None
        else task_setup_s + objective.target_hit_elapsed_s
    )
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        output["cuda_allocated_bytes"] = {
            "baseline": memory_start,
            "peak": peak,
            "end": torch.cuda.memory_allocated(device),
            "peak_delta": peak - memory_start,
        }
    return output


class PartialRunError(RuntimeError):
    def __init__(self, cause: Exception, partial: dict):
        super().__init__(str(cause))
        self.partial = partial


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def objective_probes(fixed: dict) -> list[dict]:
    """Check task values on both devices before a search starts."""
    rows = []
    shift = np.asarray(fixed["shift"], dtype=np.float64)
    partial = shift.copy()
    partial[DIM // 2 :] = 0.0
    random_gen = torch.Generator(device="cpu")
    random_gen.manual_seed(FIXTURE_SEED + 1)
    random_batch = torch.randn(4, DIM, dtype=torch.float64, generator=random_gen).numpy()
    points = {
        "known_optimum": shift[None, :],
        "shifted_origin": np.zeros((1, DIM), dtype=np.float64),
        "partial_candidate": partial[None, :],
        "fixed_random_batch": random_batch,
    }
    for name in TASKS:
        cpu_scores: dict[str, list[float]] = {}
        for device in (torch.device("cpu"), torch.device("cuda:0")):
            if device.type == "cuda" and not torch.cuda.is_available():
                continue
            task = make_task(name, device, fixed)
            for label, batch in points.items():
                if name == "thermal_train" and label == "known_optimum":
                    continue
                tensor = torch.from_numpy(batch.copy()).to(device)
                observed = task.fn(tensor).detach().cpu().tolist()
                oracle = [task.oracle(point) for point in batch]
                relative = 0.0 if name == "thermal_train" else 1e-12
                for got, expected in zip(observed, oracle, strict=True):
                    if not math.isfinite(got) or not math.isclose(
                        got,
                        expected,
                        rel_tol=relative,
                        abs_tol=1e-10,
                    ):
                        raise RuntimeError(
                            f"pre-search objective probe failed: {name}/{label} on {device}"
                        )
                if label == "known_optimum" and not math.isclose(
                    observed[0],
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(f"known optimum probe failed: {name} on {device}")
                row = {
                    "task": name,
                    "device": str(device),
                    "probe": label,
                    "points": batch.tolist(),
                    "observed": observed,
                    "cpu_oracle": oracle,
                }
                if device.type == "cpu":
                    cpu_scores[label] = observed
                else:
                    peer = cpu_scores[label]
                    differences = [abs(a - b) for a, b in zip(observed, peer, strict=True)]
                    row["cpu_cuda_max_abs_difference"] = max(differences)
                    for cuda_score, cpu_score in zip(observed, peer, strict=True):
                        if not math.isclose(
                            cuda_score,
                            cpu_score,
                            rel_tol=relative,
                            abs_tol=1e-10,
                        ):
                            raise RuntimeError(f"CPU/CUDA parity failed: {name}/{label}")
                rows.append(row)
    return rows


def thermal_evaluator_probes() -> dict:
    """Retain the frozen evaluator's original admission probes."""
    ref = _load_reference()
    results = {}
    for device in (torch.device("cpu"), torch.device("cuda:0")):
        if device.type == "cuda" and not torch.cuda.is_available():
            continue
        probes = ref.run_evaluator_probes(device=device, dtype=torch.float64)
        if probes["degenerate_zero"]["passes"]:
            raise RuntimeError(f"thermal zero probe unexpectedly passed on {device}")
        if not probes["ceiling_domain_control"]["passes"]:
            raise RuntimeError(f"thermal control probe failed on {device}")
        if probes["runaway_all_heat"]["passes"]:
            raise RuntimeError(f"thermal runaway probe unexpectedly passed on {device}")
        results[str(device)] = probes
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--evox-source", type=Path, default=None)
    parser.add_argument("--pycma-source", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.budget < POP_SIZE or args.budget % POP_SIZE or len(set(args.seeds)) != len(args.seeds):
        parser.error("budget must be a positive multiple of 32 and seeds must be unique")
    import torch_dfo

    if not Path(torch_dfo.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("torch_dfo must load from this checkout")
    torch.set_num_threads(1)
    fixed = fixture()
    ref = _load_reference()
    data: dict = {
        "protocol": "cma-comparison-v1",
        "status": "running",
        "config": {
            "dim": DIM,
            "population": POP_SIZE,
            "budget": args.budget,
            "seeds": args.seeds,
            "methods": METHODS,
            "tasks": TASKS,
            "sigma": SIGMA,
            "native_guard_abs": GUARD,
            "native_sigma0_fraction": SIGMA / (2 * GUARD),
            "float_dtype": "torch.float64",
            "cpu_threads": 1,
            "thermal_note": (
                "All optimizers search raw requests. The objective saturates and "
                "quantizes actuators. This differs from the older bounded study."
            ),
            "timing_note": (
                "Task setup, method initialization, and all generations are included. "
                "CUDA synchronizes at timing boundaries. Objectives run eager. "
                "EvoX torch.cond may compile internally."
            ),
            "target_time_note": (
                "target_hit_elapsed_s starts at method initialization. "
                "target_hit_complete_elapsed_s also includes task setup."
            ),
            "algorithm_modes": {
                "torch_dfo": (
                    "Passive, unmirrored CMA-ES. Each eigendecomposition clamps "
                    "eigenvalues and normalizes them by their mean."
                ),
                "evox": (
                    "Stock v1.4.0 passive CMA-ES. It decomposes periodically "
                    "and clamps eigenvalues. Its rank-one covariance update has "
                    "the confirmed scalar-dot-product error."
                ),
                "pycma": (
                    "Stock 4.5.0 CMA-ES with active update and mirroring disabled. "
                    "Its own adaptation and decomposition remain unchanged."
                ),
            },
            "evox_reference_status": (
                "stock_upstream_covariance_bug_confirmed; quality_results_quarantined"
            ),
        },
        "fixture": fixed,
        "sources": {
            "torch_dfo_head": _git_head(ROOT),
            "torch_dfo_cma_sha256": _sha256(ROOT / "src" / "torch_dfo" / "cmaes.py"),
            "comparison_sha256": _sha256(Path(__file__)),
            "thermal_sha256": _sha256(ROOT / "examples" / "06_engineering_control.py"),
            "thermal_task_hash": ref.task_hash(),
            "evox_covariance_probe_sha256": _sha256(
                ROOT / "research" / "redesign" / "results" / "evox-covariance-probe.json"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runs": [],
    }
    _atomic_json(args.output, data)
    try:
        data["sources"]["evox"] = _upstream_source(
            args.evox_source,
            "evox",
            "src/evox/algorithms/so/es_variants/cma_es.py",
            EVOX_FILE_SHA256,
            EVOX_SHA,
            "1.4.0",
        )
        data["sources"]["pycma"] = _upstream_source(
            args.pycma_source,
            "cma",
            "cma/evolution_strategy.py",
            PYCMA_FILE_SHA256,
            PYCMA_SHA,
            "4.5.0",
        )
        data["objective_probes"] = objective_probes(fixed)
        data["thermal_evaluator_probes"] = thermal_evaluator_probes()
        _atomic_json(args.output, data)
    except Exception as error:
        data["status"] = "failed_preflight"
        data["preflight_error"] = repr(error)
        data["preflight_traceback"] = traceback.format_exc()
        _atomic_json(args.output, data)
        return 1
    for seed_index, seed in enumerate(args.seeds):
        rotated = METHODS[seed_index % len(METHODS) :] + METHODS[: seed_index % len(METHODS)]
        for task_name in TASKS:
            for method in rotated:
                record = {"seed": seed, "task": task_name, "method": method, "budget": args.budget}
                try:
                    record.update(
                        run_one(
                            method,
                            task_name,
                            seed,
                            args.budget,
                            fixed,
                            args.evox_source,
                            args.pycma_source,
                        )
                    )
                except Exception as error:
                    record.update(
                        {
                            "status": "failed",
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if isinstance(error, PartialRunError):
                        record.update(error.partial)
                data["runs"].append(record)
                _atomic_json(args.output, data)
    expected = len(args.seeds) * len(TASKS) * len(METHODS)
    failed = [run for run in data["runs"] if run["status"] != "ok"]
    data["status"] = "complete" if len(data["runs"]) == expected and not failed else "failed"
    data["expected_runs"] = expected
    data["failed_runs"] = len(failed)
    _atomic_json(args.output, data)
    return 0 if data["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
