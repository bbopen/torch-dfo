#!/usr/bin/env python3
"""Compare search methods on recorded NASBench201 validation trials."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch_dfo import CMAES, SHADE, SearchRun  # noqa: E402

DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
OPERATIONS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
METHODS = (
    "torch_dfo_cmaes_cpu",
    "torch_dfo_cmaes_cuda",
    "torch_dfo_shade_cpu",
    "torch_dfo_shade_cuda",
    "evox_de_cpu",
    "evox_de_cuda",
    "evox_cmaes_cpu",
    "evox_cmaes_cuda",
    "random_cpu",
    "aging_cpu",
)
DIM = 30
EDGES = 6
CATEGORIES = 5
POPULATION = 20
DEFAULT_BUDGET = 1_000
DEFAULT_SEEDS = tuple(range(30))
EVOX_COMMIT = "1c242cc9533fbd7e2e0c0342456f4e0f52449e5d"
EVOX_FILE_HASHES = {
    "de": "ea1c29db651e96ab18edc8f5ea8a21a9c73b155f71620c47e441bf36c84b98f0",
    "cmaes": "3c68f5f274af8fe9f85f7257a368a84b71800a4611c94ccae3b589bfca69cb0a",
    "workflow": "83046bffb416bb2845cc937a12b8bdfb8cc7c5cadcba4cb46684cbda6f27ac92",
}
VALIDATION_EXPORT_SHA256 = "afe4bce58d24c83144ad12f8ff718425c59027203de945d12e1f8ca8a5b04c20"
TEST_EXPORT_SHA256 = "bb120e83d82c386fc986a72b33322d90b4f966f14568e83c4d09360f72973a41"
TRIAL_HASH_VERSION = "nasbench201-trial-v1"
TIE_HASH_VERSION = "nasbench201-tie-v1"

Ops = tuple[int, int, int, int, int, int]
Priority = tuple[tuple[int, ...], ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tie_priority(run_seed: int) -> Priority:
    """Choose one score-blind category order for each edge and run seed."""
    orders = []
    for edge in range(EDGES):
        payload = f"{TIE_HASH_VERSION}|{run_seed}|{edge}".encode("ascii")
        private_seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        order = list(range(CATEGORIES))
        random.Random(private_seed).shuffle(order)
        orders.append(tuple(order))
    return tuple(orders)


def decode_logits(row: Sequence[float], priority: Priority) -> tuple[Ops, int]:
    """Decode six groups of five keys and count edges with exact maxima ties."""
    if len(row) != DIM or len(priority) != EDGES:
        raise ValueError("decoder requires 30 keys and six priority orders")
    ops = []
    tied_edges = 0
    for edge in range(EDGES):
        keys = row[edge * CATEGORIES : (edge + 1) * CATEGORIES]
        if any(
            not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0 for value in keys
        ):
            raise ValueError("continuous keys must be finite and in [-1, 1]")
        maximum = max(keys)
        winners = {category for category, value in enumerate(keys) if value == maximum}
        tied_edges += len(winners) > 1
        order = priority[edge]
        if sorted(order) != list(range(CATEGORIES)):
            raise ValueError("decoder priority is not a category permutation")
        ops.append(next(category for category in order if category in winners))
    return tuple(ops), tied_edges  # type: ignore[return-value]


def trial_index(
    dataset: str, run_seed: int, ops: Sequence[int], visit_index: int, trial_count: int
) -> int:
    """Draw one trial from a private, method-independent hash stream."""
    if (
        dataset not in DATASETS
        or len(ops) != EDGES
        or any(op not in range(CATEGORIES) for op in ops)
    ):
        raise ValueError("invalid dataset or architecture")
    if visit_index < 0 or trial_count < 1:
        raise ValueError("visit index and trial count must be valid")
    payload = (
        TRIAL_HASH_VERSION.encode("ascii")
        + b"\0"
        + dataset.encode("ascii")
        + b"\0"
        + struct.pack("<q", run_seed)
        + bytes(ops)
        + struct.pack("<Q", visit_index)
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % trial_count


def _checked_ops(value: Sequence[int]) -> Ops:
    if len(value) != EDGES or any(
        type(op) is not int or op not in range(CATEGORIES) for op in value
    ):
        raise ValueError("architecture must contain six operation indices from 0 to 4")
    return tuple(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class ValidationRecord:
    index: int
    ops: Ops
    phenotype: str
    trials: dict[str, tuple[tuple[int, float], ...]]


class ValidationTable:
    """Hold validation scores only. This class never reads final test data."""

    def __init__(self, records: Sequence[dict[str, Any]], *, require_full: bool = True):
        if require_full and len(records) != CATEGORIES**EDGES:
            raise ValueError("the official export must have 15,625 architectures")
        self.by_ops: dict[Ops, ValidationRecord] = {}
        self.by_index: dict[int, ValidationRecord] = {}
        for raw in records:
            ops = _checked_ops(raw["ops"])
            index = raw["index"]
            if type(index) is not int or index < 0 or ops in self.by_ops or index in self.by_index:
                raise ValueError("duplicate or invalid architecture index")
            if "test" in raw or "test-accuracy" in str(raw.get("validation", {})):
                raise ValueError("validation export contains test data")
            trials: dict[str, tuple[tuple[int, float], ...]] = {}
            for dataset in DATASETS:
                values = raw["validation"][dataset]
                if len(values) not in (2, 3):
                    raise ValueError("each architecture needs two or three validation trials")
                parsed = []
                seen_seeds = set()
                for trial in values:
                    seed, accuracy = trial["seed"], float(trial["valid-accuracy"])
                    if type(seed) is not int or seed in seen_seeds or not 0 <= accuracy <= 100:
                        raise ValueError("invalid validation trial")
                    seen_seeds.add(seed)
                    parsed.append((seed, accuracy))
                trials[dataset] = tuple(parsed)
            record = ValidationRecord(index, ops, str(raw["phenotype"]), trials)
            self.by_ops[ops] = record
            self.by_index[index] = record

    @classmethod
    def from_path(cls, path: Path, *, require_full: bool = True) -> tuple[ValidationTable, dict]:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or "source" not in document or "records" not in document:
            raise ValueError("invalid validation export")
        source = document["source"]
        if source.get("fidelity") != 200:
            raise ValueError("validation export must use fidelity 200")
        return cls(document["records"], require_full=require_full), source


class ValidationOracle:
    """Charge each proposed architecture and retain every observed score."""

    def __init__(
        self, table: ValidationTable, dataset: str, run_seed: int, budget: int, priority: Priority
    ):
        if dataset not in DATASETS or budget < 1:
            raise ValueError("invalid dataset or budget")
        self.table = table
        self.dataset = dataset
        self.run_seed = run_seed
        self.budget = budget
        self.priority = priority
        self.visits: Counter[Ops] = Counter()
        self.trace: list[dict[str, Any]] = []
        self.attempted = 0
        self.batch_calls = 0
        self.tie_count = 0
        self.best_step: int | None = None
        self.best_error = math.inf
        self.best_ops: Ops | None = None

    @property
    def completed(self) -> int:
        return len(self.trace)

    def observe_ops(self, ops_value: Sequence[int], *, tied_edges: int = 0) -> float:
        self.attempted += 1
        return self._score_ops(ops_value, tied_edges=tied_edges)

    def _score_ops(self, ops_value: Sequence[int], *, tied_edges: int = 0) -> float:
        if self.completed >= self.budget:
            raise RuntimeError("validation observation budget exceeded")
        ops = _checked_ops(ops_value)
        record = self.table.by_ops[ops]
        visit = self.visits[ops]
        trials = record.trials[self.dataset]
        selected = trial_index(self.dataset, self.run_seed, ops, visit, len(trials))
        trial_seed, accuracy = trials[selected]
        error = 100.0 - accuracy
        step = self.completed + 1
        self.visits[ops] += 1
        self.tie_count += tied_edges
        if error < self.best_error:
            self.best_error = error
            self.best_ops = ops
            self.best_step = step
        self.trace.append(
            {
                "step": step,
                "architecture_index": record.index,
                "ops": list(ops),
                "visit_index": visit,
                "trial_index": selected,
                "trial_seed": trial_seed,
                "validation_error": error,
                "incumbent_step": self.best_step,
                "tied_edges": tied_edges,
            }
        )
        return error

    def observe_logits(self, row: Sequence[float]) -> float:
        ops, tied_edges = decode_logits(row, self.priority)
        return self.observe_ops(ops, tied_edges=tied_edges)

    def score_tensor(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2 or candidates.shape[1] != DIM:
            raise ValueError("continuous candidate batch must have shape (N, 30)")
        if candidates.dtype != torch.float64 or self.completed + candidates.shape[0] > self.budget:
            raise ValueError("wrong dtype or over-budget candidate batch")
        self.batch_calls += 1
        self.attempted += candidates.shape[0]
        rows = candidates.detach().to("cpu").tolist()
        scores = []
        for row in rows:
            ops, tied_edges = decode_logits(row, self.priority)
            scores.append(self._score_ops(ops, tied_edges=tied_edges))
        return torch.tensor(scores, dtype=torch.float64, device=candidates.device)

    def facts(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "batch_calls": self.batch_calls,
            "unique_architectures": len(self.visits),
            "repeated_architectures": self.completed - len(self.visits),
            "decoder_tied_edges": self.tie_count,
            "best_step": self.best_step,
            "best_validation_error": None if self.best_ops is None else self.best_error,
            "best_ops": None if self.best_ops is None else list(self.best_ops),
            "best_architecture_index": (
                None if self.best_ops is None else self.table.by_ops[self.best_ops].index
            ),
            "trace": self.trace,
        }


def make_native_optimizer(method: str, device: torch.device, seed: int) -> CMAES | SHADE:
    """Use the stock native settings shared by both performance studies."""
    common = {
        "dim": DIM,
        "bounds": (-1.0, 1.0),
        "pop_size": POPULATION,
        "device": device,
        "dtype": torch.float64,
        "seed": seed,
    }
    if method.startswith("torch_dfo_cmaes_"):
        return CMAES(**common, sigma0=0.3, mirrored=False, active=False)
    if method.startswith("torch_dfo_shade_"):
        return SHADE(**common)
    raise ValueError(f"unknown native method: {method}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_native(
    method: str, device: torch.device, seed: int, oracle: ValidationOracle, timing: dict
) -> dict:
    optimizer = make_native_optimizer(method, device, seed)
    run = SearchRun(optimizer, max_evals=oracle.budget, evaluator_id="nasbench201", repeats=1)
    _sync(device)
    timing["construction_s"] = time.perf_counter() - timing["start"]
    while not run.done:
        run.step(oracle.score_tensor)
    result = run.result
    accounting = {
        "scheduled": result.scheduled_evals,
        "charged": result.charged_evals,
        "attempted": result.attempted_evals,
        "completed": result.completed_evals,
        "stop_reason": result.stop_reason,
        "failures": list(result.failures),
        "accounting_uncertain": result.accounting_uncertain,
    }
    timing["search_run"] = accounting
    if (
        result.failures
        or result.accounting_uncertain
        or result.stop_reason != "budget exhausted"
        or any(
            accounting[key] != oracle.budget
            for key in ("scheduled", "charged", "attempted", "completed")
        )
        or oracle.completed != oracle.budget
    ):
        raise RuntimeError(f"native run did not finish its exact budget: {accounting}")
    if result.best_value is None or not math.isclose(
        float(result.best_value), oracle.best_error, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("SearchRun incumbent differs from the observed incumbent")
    return {"native_accounting": accounting}


def _evox_imports(method: str, source: Path | None) -> tuple[type, type, type, dict]:
    """Verify the imported EvoX class before any timed search."""
    if source is not None:
        source_path = str(source.resolve() / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
    from evox.core import Problem
    from evox.workflows import StdWorkflow

    if method.startswith("evox_de_"):
        from evox.algorithms.so.de_variants import DE as AlgorithmClass

        expected_hash = EVOX_FILE_HASHES["de"]
    else:
        from evox.algorithms.so.es_variants import CMAES as AlgorithmClass

        expected_hash = EVOX_FILE_HASHES["cmaes"]
    imported_file = inspect.getsourcefile(AlgorithmClass)
    if imported_file is None or sha256(Path(imported_file)) != expected_hash:
        raise RuntimeError("imported EvoX algorithm source differs from the pinned version")
    workflow_file = inspect.getsourcefile(StdWorkflow)
    if workflow_file is None or sha256(Path(workflow_file)) != EVOX_FILE_HASHES["workflow"]:
        raise RuntimeError("imported EvoX workflow source differs from the pinned version")
    if source is None and metadata.version("evox") != "1.4.0":
        raise RuntimeError("EvoX package version differs from 1.4.0")
    return (
        AlgorithmClass,
        Problem,
        StdWorkflow,
        {
            "file": str(Path(imported_file).resolve()),
            "sha256": expected_hash,
        },
    )


def _run_evox(
    method: str,
    device: torch.device,
    seed: int,
    oracle: ValidationOracle,
    timing: dict,
    source: Path | None,
) -> dict:
    AlgorithmClass, Problem, StdWorkflow, imported = _evox_imports(method, source)

    class RecordedProblem(Problem):
        def evaluate(self, population: torch.Tensor) -> torch.Tensor:
            return oracle.score_tensor(population)

    old_dtype = torch.get_default_dtype()
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(seed)
            if method.startswith("evox_de_"):
                lower = torch.full((DIM,), -1.0, dtype=torch.float64, device=device)
                upper = torch.full((DIM,), 1.0, dtype=torch.float64, device=device)
                algorithm = AlgorithmClass(pop_size=POPULATION, lb=lower, ub=upper, device=device)
                transform = None
            else:
                mean = torch.zeros(DIM, dtype=torch.float64, device=device)
                algorithm = AlgorithmClass(
                    mean_init=mean, sigma=0.6, pop_size=POPULATION, device=device
                )
                # The stock latent CMA update remains unbounded. Only evaluated keys are clamped.
                transform = torch.nn.Hardtanh(-1.0, 1.0)
            workflow = StdWorkflow(
                algorithm, RecordedProblem(), solution_transform=transform, device=device
            )
            _sync(device)
            timing["construction_s"] = time.perf_counter() - timing["start"]
            for generation in range(oracle.budget // POPULATION):
                if generation == 0:
                    workflow.init_step()
                else:
                    workflow.step()
        finally:
            torch.set_default_dtype(old_dtype)
    if oracle.completed != oracle.budget or oracle.batch_calls != oracle.budget // POPULATION:
        raise RuntimeError("EvoX did not evaluate exactly one population per generation")
    return {"imported_evox_source": imported}


def _random_ops(rng: random.Random) -> Ops:
    return tuple(rng.randrange(CATEGORIES) for _ in range(EDGES))  # type: ignore[return-value]


def _mutate_one_edge(parent: Ops, rng: random.Random) -> Ops:
    edge = rng.randrange(EDGES)
    replacement = rng.randrange(CATEGORIES - 1)
    if replacement >= parent[edge]:
        replacement += 1
    child = list(parent)
    child[edge] = replacement
    return tuple(child)  # type: ignore[return-value]


def _run_random(seed: int, oracle: ValidationOracle, timing: dict) -> dict:
    rng = random.Random(seed)
    timing["construction_s"] = time.perf_counter() - timing["start"]
    for _ in range(oracle.budget):
        oracle.observe_ops(_random_ops(rng))
    return {}


def _run_aging(seed: int, oracle: ValidationOracle, timing: dict) -> dict:
    if oracle.budget < 100:
        raise ValueError("aging evolution needs at least 100 observations")
    rng = random.Random(seed)
    population: deque[tuple[float, int, Ops]] = deque()
    timing["construction_s"] = time.perf_counter() - timing["start"]
    for _ in range(100):
        ops = _random_ops(rng)
        error = oracle.observe_ops(ops)
        population.append((error, oracle.completed, ops))
    while oracle.completed < oracle.budget:
        sampled = [population[rng.randrange(len(population))] for _ in range(25)]
        parent = min(sampled, key=lambda item: (item[0], item[1]))[2]
        child = _mutate_one_edge(parent, rng)
        error = oracle.observe_ops(child)
        population.append((error, oracle.completed, child))
        population.popleft()
    return {"aging_population": 100, "aging_tournament": 25}


def run_one(
    table: ValidationTable,
    dataset: str,
    method: str,
    seed: int,
    budget: int,
    *,
    evox_source: Path | None = None,
) -> dict:
    """Run one method and return its full trace, including failure facts."""
    if method not in METHODS or dataset not in DATASETS:
        raise ValueError("unknown method or dataset")
    if budget < 1 or (method not in ("random_cpu", "aging_cpu") and budget % POPULATION):
        raise ValueError("continuous methods require a positive multiple of population 20")
    device = torch.device("cuda:0" if method.endswith("_cuda") else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA method requested without CUDA")
    oracle = ValidationOracle(table, dataset, seed, budget, tie_priority(seed))
    result: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "budget": budget,
        "device": str(device),
        "tie_priority": [list(order) for order in oracle.priority],
    }
    timing: dict[str, Any] = {}
    try:
        _sync(device)
        timing["start"] = time.perf_counter()
        if method.startswith("torch_dfo_"):
            extra = _run_native(method, device, seed, oracle, timing)
        elif method.startswith("evox_"):
            extra = _run_evox(method, device, seed, oracle, timing, evox_source)
        elif method == "random_cpu":
            extra = _run_random(seed, oracle, timing)
        else:
            extra = _run_aging(seed, oracle, timing)
        _sync(device)
        timing["total_s"] = time.perf_counter() - timing["start"]
        if oracle.completed != budget or oracle.attempted != budget:
            raise RuntimeError("method did not consume the exact validation budget")
        result.update(extra)
        result["status"] = "ok"
    except Exception as error:
        failure = traceback.format_exc()
        try:
            _sync(device)
        except Exception as sync_error:
            failure += f"\nCUDA synchronization also failed: {sync_error!r}"
        if "start" in timing:
            timing["total_s"] = time.perf_counter() - timing["start"]
        result.update({"status": "failed", "error": repr(error), "traceback": failure})
    result.update(oracle.facts())
    result["timing_s"] = {
        key: value for key, value in timing.items() if key != "start" and key != "search_run"
    }
    if "search_run" in timing:
        result["native_accounting"] = timing["search_run"]
    return result


def check_evox_source(source: Path | None) -> dict:
    """Check both stock algorithm files before the search panel starts."""
    relative = {
        "de": Path("src/evox/algorithms/so/de_variants/de.py"),
        "cmaes": Path("src/evox/algorithms/so/es_variants/cma_es.py"),
        "workflow": Path("src/evox/workflows/std_workflow.py"),
    }
    if source is not None:
        root = source.resolve()
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if head != EVOX_COMMIT:
            raise RuntimeError(f"EvoX Git commit differs: {head}")
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("EvoX source checkout has local changes")
        files = {name: root / file for name, file in relative.items()}
    else:
        distribution = metadata.distribution("evox")
        if distribution.version != "1.4.0":
            raise RuntimeError(f"EvoX version differs: {distribution.version}")
        files = {
            name: Path(distribution.locate_file(str(file).removeprefix("src/"))).resolve()
            for name, file in relative.items()
        }
        head = None
    for name, file in files.items():
        if not file.is_file() or sha256(file) != EVOX_FILE_HASHES[name]:
            raise RuntimeError(f"EvoX {name} source bytes differ: {file}")
    return {
        "version": "1.4.0",
        "git_head": head,
        "files": {
            name: {"path": str(file), "sha256": EVOX_FILE_HASHES[name]}
            for name, file in files.items()
        },
    }


def load_final_test(path: Path, table: ValidationTable) -> tuple[dict[int, dict[str, float]], dict]:
    """Read test scores after all search methods have finished."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or "source" not in document or "records" not in document:
        raise ValueError("invalid final test export")
    if document["source"].get("fidelity") != 200:
        raise ValueError("final test export must use fidelity 200")
    if len(document["records"]) != len(table.by_index):
        raise ValueError("validation and test architecture counts differ")
    means: dict[int, dict[str, float]] = {}
    for raw in document["records"]:
        index = raw["index"]
        if index not in table.by_index or index in means:
            raise ValueError("test export has a missing or duplicate architecture index")
        means[index] = {}
        for dataset in DATASETS:
            trials = raw["test"][dataset]
            if len(trials) not in (2, 3):
                raise ValueError("final test record must contain two or three trials")
            accuracies = [float(trial["test-accuracy"]) for trial in trials]
            if any(not math.isfinite(value) or not 0 <= value <= 100 for value in accuracies):
                raise ValueError("invalid final test accuracy")
            means[index][dataset] = sum(100.0 - value for value in accuracies) / len(accuracies)
    return means, document["source"]


def _append_event(path: Path, event: dict) -> None:
    """Append one durable JSONL record without rewriting earlier runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_summary(path: Path, summary: dict) -> None:
    target = path.with_suffix(".summary.json")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, target)


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_panel(
    *,
    validation_file: Path,
    test_file: Path,
    output: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    methods: Sequence[str] = METHODS,
    datasets: Sequence[str] = DATASETS,
    budget: int = DEFAULT_BUDGET,
    evox_source: Path | None = None,
    protocol_file: Path | None = None,
    require_full_data: bool = True,
) -> dict:
    """Save each run before opening the separate final test export."""
    if output.exists() and output.stat().st_size:
        raise FileExistsError(f"output already has data: {output}")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or not methods
        or len(set(methods)) != len(methods)
        or not datasets
        or len(set(datasets)) != len(datasets)
    ):
        raise ValueError("seeds, methods, and datasets must be nonempty and unique")
    if any(method not in METHODS for method in methods) or any(
        dataset not in DATASETS for dataset in datasets
    ):
        raise ValueError("unknown method or dataset")
    if budget < 1 or any(
        method not in ("random_cpu", "aging_cpu") and budget % POPULATION for method in methods
    ):
        raise ValueError("continuous method budgets must be multiples of 20")
    if "aging_cpu" in methods and budget < 100:
        raise ValueError("aging evolution needs at least 100 observations")

    torch.set_num_threads(1)
    protocol = protocol_file or ROOT / "research" / "redesign" / "nasbench201-protocol.md"
    if not protocol.is_file():
        raise FileNotFoundError(f"frozen protocol is missing: {protocol}")
    table, validation_source = ValidationTable.from_path(
        validation_file, require_full=require_full_data
    )
    validation_hash = sha256(validation_file)
    if require_full_data and validation_hash != VALIDATION_EXPORT_SHA256:
        raise ValueError("validation export differs from the frozen official export")
    sources: dict[str, Any] = {
        "repository_head": _git_head(ROOT),
        "runner_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(protocol),
        "validation_export_sha256": validation_hash,
        "torch_dfo_cmaes_sha256": sha256(ROOT / "src/torch_dfo/cmaes.py"),
        "torch_dfo_shade_sha256": sha256(ROOT / "src/torch_dfo/shade.py"),
        "torch_dfo_search_run_sha256": sha256(ROOT / "src/torch_dfo/run.py"),
        "nasbench201_exporter_sha256": sha256(ROOT / "benchmarks/nasbench201_data.py"),
        "validation_source": validation_source,
    }
    if any(method.startswith("evox_") for method in methods):
        sources["evox"] = check_evox_source(evox_source)
    header = {
        "type": "header",
        "format": "nasbench201-quality-jsonl-v1",
        "config": {
            "datasets": list(datasets),
            "methods": list(methods),
            "seeds": list(seeds),
            "budget": budget,
            "population": POPULATION,
            "continuous_dim": DIM,
            "continuous_bounds": [-1.0, 1.0],
            "native_cmaes_sigma0": 0.3,
            "evox_cmaes_sigma": 0.6,
            "aging_population": 100,
            "aging_tournament": 25,
            "tie_hash_version": TIE_HASH_VERSION,
            "trial_hash_version": TRIAL_HASH_VERSION,
            "test_file_read_phase": "after_all_search_runs",
            "runtime_scope": (
                "wall_including_jsonl_s includes construction, search, and durable trace write"
            ),
        },
        "sources": sources,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    _append_event(output, header)
    compact_runs = []
    for seed in seeds:
        rotated_methods = list(methods[seed % len(methods) :]) + list(
            methods[: seed % len(methods)]
        )
        for dataset in datasets:
            for method in rotated_methods:
                wall_start = time.perf_counter()
                try:
                    record = run_one(table, dataset, method, seed, budget, evox_source=evox_source)
                except Exception as error:
                    record = {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "budget": budget,
                        "status": "failed",
                        "attempted": 0,
                        "completed": 0,
                        "batch_calls": 0,
                        "unique_architectures": 0,
                        "repeated_architectures": 0,
                        "decoder_tied_edges": 0,
                        "best_step": None,
                        "best_validation_error": None,
                        "best_ops": None,
                        "best_architecture_index": None,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "trace": [],
                    }
                _append_event(output, {"type": "run", "result": record})
                compact = {
                    key: value for key, value in record.items() if key not in ("trace", "traceback")
                }
                compact["wall_including_jsonl_s"] = time.perf_counter() - wall_start
                compact_runs.append(compact)

    # This is the first access to test_file. All searches and durable run writes are complete.
    final_test, test_source = load_final_test(test_file, table)
    test_hash = sha256(test_file)
    if require_full_data and test_hash != TEST_EXPORT_SHA256:
        raise ValueError("final test export differs from the frozen official export")
    selected = [
        {
            "dataset": record["dataset"],
            "method": record["method"],
            "seed": record["seed"],
            "selected_architecture_index": record["best_architecture_index"],
            "selected_ops": record["best_ops"],
            "selected_validation_error": record["best_validation_error"],
            "selected_test_error_mean": final_test[record["best_architecture_index"]][
                record["dataset"]
            ],
        }
        for record in compact_runs
        if record["status"] == "ok"
    ]
    final = {
        "type": "final_test",
        "test_export_sha256": test_hash,
        "test_source": test_source,
        "selected": selected,
    }
    _append_event(output, final)
    failures = sum(record["status"] != "ok" for record in compact_runs)
    summary = {
        "status": "complete" if failures == 0 else "partial",
        "expected_runs": len(seeds) * len(datasets) * len(methods),
        "recorded_runs": len(compact_runs),
        "failed_runs": failures,
        "sources": {
            **sources,
            "test_export_sha256": final["test_export_sha256"],
            "test_source": test_source,
        },
        "config": header["config"],
        "environment": header["environment"],
        "runs": compact_runs,
        "selected_test": selected,
        "events_path": str(output),
    }
    _append_event(
        output, {"type": "complete", "status": summary["status"], "failed_runs": failures}
    )
    _write_summary(output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL result path")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--evox-source", type=Path)
    args = parser.parse_args(argv)
    summary = run_panel(
        validation_file=args.validation_file,
        test_file=args.test_file,
        output=args.output,
        seeds=args.seeds,
        methods=args.methods,
        datasets=args.datasets,
        budget=args.budget,
        evox_source=args.evox_source,
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
