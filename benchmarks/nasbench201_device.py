#!/usr/bin/env python3
"""Time CPU and CUDA optimizers against the same NASBench201 validation table.

This is an adapted, device-resident EvoXBench scorer. It is not the stock
Django/NumPy evaluator. The validation export and its source hashes are inputs.
No held-out test file is opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.nasbench201 import TRIAL_HASH_VERSION, tie_priority, trial_index  # noqa: E402
from torch_dfo import SearchRun  # noqa: E402

DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
METHODS = ("torch_dfo_cmaes", "torch_dfo_shade", "evox_de", "evox_cmaes")
POPULATION = 20
BUDGET = 1000
DIM = 30
POWERS = (1, 5, 25, 125, 625, 3125)
VALIDATION_SHA256 = "afe4bce58d24c83144ad12f8ff718425c59027203de945d12e1f8ca8a5b04c20"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class ValidationTable:
    dataset: str
    ops: tuple[tuple[int, ...], ...]
    values: np.ndarray
    counts: np.ndarray
    id_by_code: np.ndarray


def load_validation_table(path: Path, dataset: str) -> ValidationTable:
    """Build one raw error table from the validated search-only export."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    document = json.loads(path.read_text())
    source = document["source"]
    if int(source["fidelity"]) != 200:
        raise ValueError("validation export must use fidelity 200")
    records = document["records"]
    if len(records) != 5**6:
        raise ValueError("validation export must contain all 15,625 architectures")
    values = np.full((len(records), 3), np.nan, dtype=np.float64)
    counts = np.empty(len(records), dtype=np.int64)
    id_by_code = np.full(5**6, -1, dtype=np.int64)
    ops_rows: list[tuple[int, ...]] = []
    for index, row in enumerate(records):
        if row["index"] != index:
            raise ValueError("records must be in exporter index order")
        ops = tuple(int(op) for op in row["ops"])
        if len(ops) != 6 or any(op < 0 or op > 4 for op in ops):
            raise ValueError(f"invalid six-operation row {index}")
        code = sum(op * weight for op, weight in zip(ops, POWERS, strict=True))
        if id_by_code[code] >= 0:
            raise ValueError(f"duplicate architecture code {code}")
        id_by_code[code] = index
        ops_rows.append(ops)
        trials = row["validation"][dataset]
        if len(trials) not in (2, 3):
            raise ValueError(f"expected two or three validation trials for row {index}")
        counts[index] = len(trials)
        for trial_number, trial in enumerate(trials):
            error = 100.0 - float(trial["valid-accuracy"])
            if not np.isfinite(error) or not 0 <= error <= 100:
                raise ValueError(f"invalid validation error for row {index}")
            values[index, trial_number] = error
    if np.any(id_by_code < 0):
        raise ValueError("validation export does not cover the full search space")
    return ValidationTable(dataset, tuple(ops_rows), values, counts, id_by_code)


def make_trial_schedule(table: ValidationTable, run_seed: int, budget: int = BUDGET) -> np.ndarray:
    """Precompute the quality runner's exact private trial draw for every visit."""
    if budget < 1:
        raise ValueError("budget must be positive")
    schedule = np.empty((len(table.ops), budget), dtype=np.uint8)
    prefix = (
        TRIAL_HASH_VERSION.encode("ascii")
        + b"\0"
        + table.dataset.encode("ascii")
        + b"\0"
        + struct.pack("<q", run_seed)
    )
    visit_suffixes = tuple(struct.pack("<Q", visit) for visit in range(budget))
    for arch_id, ops in enumerate(table.ops):
        count = int(table.counts[arch_id])
        base = hashlib.sha256(prefix + bytes(ops))
        for visit, suffix in enumerate(visit_suffixes):
            digest = base.copy()
            digest.update(suffix)
            schedule[arch_id, visit] = int.from_bytes(digest.digest(), "big") % count
        for visit in {0, budget // 2, budget - 1}:
            expected = trial_index(table.dataset, run_seed, ops, visit, count)
            if schedule[arch_id, visit] != expected:
                raise RuntimeError("precomputed trial schedule differs from quality runner")
    return schedule


def rank_by_category(priority: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Convert six ordered category permutations to argmin tie ranks."""
    if len(priority) != 6:
        raise ValueError("tie priority must contain six edges")
    ranks = np.empty((6, 5), dtype=np.int64)
    for edge, order in enumerate(priority):
        if sorted(order) != list(range(5)):
            raise ValueError("each tie priority must permute the five operations")
        for rank, category in enumerate(order):
            ranks[edge, category] = rank
    return ranks


class DeviceResidentScorer:
    """Decode logits and return sampled validation errors without host reads."""

    def __init__(
        self,
        table: ValidationTable,
        schedule: np.ndarray,
        run_seed: int,
        device: torch.device | str,
        *,
        budget: int = BUDGET,
        record_trace: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.budget = budget
        if schedule.shape != (len(table.ops), budget):
            raise ValueError("trial schedule shape does not match the table and budget")
        self.values = torch.as_tensor(table.values.copy(), device=self.device)
        self.device = self.values.device
        self.id_by_code = torch.as_tensor(table.id_by_code.copy(), device=self.device)
        self.schedule = torch.as_tensor(schedule.copy(), device=self.device)
        ranks = rank_by_category(tie_priority(run_seed))
        self.ranks = torch.as_tensor(ranks, device=self.device)
        self.powers = torch.tensor(POWERS, dtype=torch.int64, device=self.device)
        self.lower = torch.tril(
            torch.ones((POPULATION, POPULATION), dtype=torch.bool, device=self.device),
            diagonal=-1,
        )
        self.visits = torch.zeros(len(table.ops), dtype=torch.int64, device=self.device)
        self.charged = 0
        self.record_trace = record_trace
        if record_trace:
            self.ids_trace = torch.empty(budget, dtype=torch.int64, device=self.device)
            self.visit_trace = torch.empty(budget, dtype=torch.int64, device=self.device)
            self.trial_trace = torch.empty(budget, dtype=torch.int64, device=self.device)
            self.error_trace = torch.empty(budget, dtype=torch.float64, device=self.device)
            self.tie_trace = torch.empty(budget, dtype=torch.int64, device=self.device)

    def decode_ids(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent logits to exporter IDs with the shared tie priority."""
        if logits.ndim != 2 or logits.shape[1] != DIM:
            raise ValueError("candidate tensor must have shape (batch, 30)")
        count = logits.shape[0]
        if logits.device != self.device or logits.dtype != torch.float64:
            raise ValueError("candidate device or dtype differs from the frozen workload")
        groups = logits.reshape(count, 6, 5)
        maxima = groups.max(dim=2, keepdim=True).values
        is_max = groups == maxima
        ranks = self.ranks.unsqueeze(0).expand(count, -1, -1)
        ops = torch.where(is_max, ranks, 5).argmin(dim=2)
        codes = (ops * self.powers).sum(dim=1)
        ids = self.id_by_code[codes]
        tied_edges = (is_max.sum(dim=2) > 1).sum(dim=1)
        return ids, tied_edges

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        count = logits.shape[0]
        if count != POPULATION or self.charged + count > self.budget:
            raise ValueError("candidate batch or budget differs from the frozen workload")
        ids, tied_edges = self.decode_ids(logits)
        # Each duplicate in the same batch receives the next visit number.
        equals = ids[:, None] == ids[None, :]
        in_batch = (equals & self.lower).sum(dim=1)
        visits = self.visits[ids] + in_batch
        chosen = self.schedule[ids, visits].to(torch.int64)
        errors = self.values[ids, chosen]
        self.visits.index_add_(0, ids, torch.ones_like(ids))
        if self.record_trace:
            start = self.charged
            stop = start + count
            self.ids_trace[start:stop] = ids
            self.visit_trace[start:stop] = visits
            self.trial_trace[start:stop] = chosen
            self.error_trace[start:stop] = errors
            self.tie_trace[start:stop] = tied_edges
        self.charged += count
        return errors

    def trace(self) -> dict:
        """Move the complete search trace to the host after timing ends."""
        if not self.record_trace:
            raise ValueError("this scorer does not record a trace")
        n = self.charged
        return {
            "architecture_ids": self.ids_trace[:n].cpu().tolist(),
            "visit_indices": self.visit_trace[:n].cpu().tolist(),
            "trial_indices": self.trial_trace[:n].cpu().tolist(),
            "validation_errors": self.error_trace[:n].cpu().tolist(),
            "tied_edges": self.tie_trace[:n].cpu().tolist(),
        }


class CompiledDeviceResidentScorer(DeviceResidentScorer):
    """Use tensor trace positions so EvoX can attempt a compiled workflow."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cursor = torch.zeros((), dtype=torch.int64, device=self.device)
        self.row_indices = torch.arange(POPULATION, dtype=torch.int64, device=self.device)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        ids, tied_edges = self.decode_ids(logits)
        equals = ids[:, None] == ids[None, :]
        in_batch = (equals & self.lower).sum(dim=1)
        visits = self.visits[ids] + in_batch
        chosen = self.schedule[ids, visits].to(torch.int64)
        errors = self.values[ids, chosen]
        self.visits.index_add_(0, ids, torch.ones_like(ids))
        positions = self.cursor + self.row_indices
        self.ids_trace.index_copy_(0, positions, ids)
        self.visit_trace.index_copy_(0, positions, visits)
        self.trial_trace.index_copy_(0, positions, chosen)
        self.error_trace.index_copy_(0, positions, errors)
        self.tie_trace.index_copy_(0, positions, tied_edges)
        self.cursor.add_(POPULATION)
        return errors


def check_table_parity(table: ValidationTable, device: torch.device) -> None:
    """Check all exported errors and architecture IDs after device transfer."""
    values = torch.as_tensor(table.values.copy(), device=device).cpu().numpy()
    counts = torch.as_tensor(table.counts.copy(), device=device).cpu().numpy()
    codes = torch.as_tensor(table.id_by_code.copy(), device=device).cpu().numpy()
    if not np.array_equal(values, table.values, equal_nan=True):
        raise RuntimeError("device validation table differs from export")
    if not np.array_equal(counts, table.counts) or not np.array_equal(codes, table.id_by_code):
        raise RuntimeError("device architecture mapping differs from export")
    # Decode every architecture once. This also tests the six-edge order.
    dummy_schedule = np.zeros((len(table.ops), 20), dtype=np.uint8)
    scorer = DeviceResidentScorer(table, dummy_schedule, 0, device, budget=20, record_trace=False)
    for start in range(0, len(table.ops), 1024):
        group = table.ops[start : start + 1024]
        operations = torch.as_tensor(group, dtype=torch.int64, device=device)
        logits = torch.zeros((len(group), 6, 5), dtype=torch.float64, device=device)
        logits.scatter_(2, operations.unsqueeze(-1), 1.0)
        ids, tied = scorer.decode_ids(logits.reshape(-1, DIM))
        expected = torch.arange(start, start + len(group), device=device)
        if not torch.equal(ids, expected) or not torch.equal(tied, torch.zeros_like(tied)):
            raise RuntimeError("device logit decoder differs from exporter architecture IDs")


def fixed_batches(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(419)
    return (2 * torch.rand((50, POPULATION, DIM), dtype=torch.float64, generator=generator) - 1).to(
        device
    )


def objective_only(
    table: ValidationTable, schedule: np.ndarray, seed: int, device: torch.device
) -> dict:
    """Time 1,000 fixed lookups, including logits decoding and trial choice."""
    batches = fixed_batches(device)
    warm = DeviceResidentScorer(table, schedule, seed, device, record_trace=False)
    for batch in batches[:5]:
        warm(batch)
    sync(device)
    samples = []
    for _ in range(5):
        scorer = DeviceResidentScorer(table, schedule, seed, device, record_trace=False)
        sync(device)
        start = time.perf_counter()
        for batch in batches:
            scorer(batch)
        sync(device)
        samples.append(time.perf_counter() - start)
        if scorer.charged != BUDGET:
            raise RuntimeError("objective-only budget mismatch")
    return {"seconds": samples, "median_seconds": statistics.median(samples)}


def audit_trace(table: ValidationTable, schedule: np.ndarray, trace: dict) -> None:
    """Check every charged result and visit against the frozen table."""
    seen: dict[int, int] = {}
    for arch_id, visit, trial, error in zip(
        trace["architecture_ids"],
        trace["visit_indices"],
        trace["trial_indices"],
        trace["validation_errors"],
        strict=True,
    ):
        if not 0 <= arch_id < len(table.ops) or visit != seen.get(arch_id, 0):
            raise RuntimeError("recorded architecture visits are not contiguous")
        if trial != int(schedule[arch_id, visit]):
            raise RuntimeError("recorded trial differs from private schedule")
        if error != float(table.values[arch_id, trial]):
            raise RuntimeError("recorded error differs from validation export")
        seen[arch_id] = visit + 1


def check_scorer_parity(
    table: ValidationTable, schedule: np.ndarray, seed: int, devices: list[str]
) -> None:
    """Compare identical proposals and repeat visits on every requested device."""
    expected = None
    for device_name in devices:
        device = torch.device(device_name)
        scorer = DeviceResidentScorer(table, schedule, seed, device)
        proposals = fixed_batches(device)
        for batch in proposals:
            scorer(batch)
        trace = scorer.trace()
        audit_trace(table, schedule, trace)
        if expected is not None:
            for key in trace:
                if trace[key] != expected[key]:
                    raise RuntimeError(f"CPU/CUDA scorer parity failed for {key}")
        else:
            expected = trace
        # The all-tied candidate revisits one architecture 20 times.
        repeated = DeviceResidentScorer(table, schedule[:, :20], seed, device, budget=20)
        repeated(torch.zeros((POPULATION, DIM), dtype=torch.float64, device=device))
        repeat_trace = repeated.trace()
        if repeat_trace["visit_indices"] != list(range(POPULATION)):
            raise RuntimeError("repeated architecture visit indices differ")
        audit_trace(table, schedule, repeat_trace)


def make_workflow(method: str, device: torch.device, scorer):
    """Build one stock EvoX workflow with the shared scorer."""
    from evox.algorithms.so.de_variants import DE
    from evox.algorithms.so.es_variants import CMAES as EvoXCMAES
    from evox.core import Problem
    from evox.workflows import StdWorkflow

    class TableProblem(Problem):
        def evaluate(self, population: torch.Tensor) -> torch.Tensor:
            return scorer(population)

    if method == "evox_de":
        bounds = torch.full((DIM,), -1.0, dtype=torch.float64, device=device)
        algorithm = DE(POPULATION, bounds, -bounds, device=device)
        transform = None
    elif method == "evox_cmaes":
        algorithm = EvoXCMAES(
            torch.zeros(DIM, dtype=torch.float64, device=device),
            sigma=0.6,
            pop_size=POPULATION,
            device=device,
        )
        # EvoX updates its unbounded latent distribution. Match the quality
        # runner by clamping only the keys sent to the evaluator.
        transform = torch.nn.Hardtanh(-1.0, 1.0)
    else:
        raise ValueError(f"unknown EvoX method: {method}")
    return StdWorkflow(algorithm, TableProblem(), solution_transform=transform, device=device)


def run_search(
    method: str,
    table: ValidationTable,
    schedule: np.ndarray,
    seed: int,
    device: torch.device,
    *,
    compile_evox: bool = False,
) -> dict:
    """Time one complete search and save its observed validation trace."""
    from benchmarks.nasbench201 import make_native_optimizer

    cache_reset_seconds = None
    if compile_evox:
        sync(device)
        reset_start = time.perf_counter()
        torch.compiler.reset()
        cache_reset_seconds = time.perf_counter() - reset_start
    sync(device)
    scorer_start = time.perf_counter()
    scorer_class = CompiledDeviceResidentScorer if compile_evox else DeviceResidentScorer
    scorer = scorer_class(table, schedule, seed, device)
    sync(device)
    scorer_staging_seconds = time.perf_counter() - scorer_start
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        sync(device)
        construction_start = time.perf_counter()
        if method.startswith("torch_dfo_"):
            optimizer = make_native_optimizer(f"{method}_{device.type}", device, seed)
            runner = SearchRun(optimizer, BUDGET, evaluator_id="nasbench201", repeats=1)

            def step(first: bool) -> None:
                runner.step(scorer)

        else:
            workflow = make_workflow(method, device, scorer)
            evox_step = workflow.step
            if compile_evox:
                if method != "evox_de":
                    raise ValueError("optional compilation is limited to EvoX DE")
                from evox.core import compile as evox_compile
                from torch._dynamo.utils import counters as dynamo_counters

                evox_step = evox_compile(workflow.step, fullgraph=True)

            def step(first: bool) -> None:
                if first:
                    workflow.init_step()
                else:
                    evox_step()

        sync(device)
        construction_seconds = time.perf_counter() - construction_start
        start = time.perf_counter()
        step(True)
        sync(device)
        first_seconds = time.perf_counter() - start
        first_compiled_seconds = None
        first_compiled_graphs = None
        total_compiled_graphs = None
        if compile_evox:
            graphs_before = int(dynamo_counters["stats"]["unique_graphs"])
            compiled_start = time.perf_counter()
            step(False)
            sync(device)
            first_compiled_seconds = time.perf_counter() - compiled_start
            first_compiled_graphs = (
                int(dynamo_counters["stats"]["unique_graphs"]) - graphs_before
            )
            remaining_steps = BUDGET // POPULATION - 2
        else:
            remaining_steps = BUDGET // POPULATION - 1
        steady_start = time.perf_counter()
        for _ in range(remaining_steps):
            step(False)
        sync(device)
        steady_seconds = time.perf_counter() - steady_start
        search_seconds = time.perf_counter() - start
        if compile_evox:
            total_compiled_graphs = (
                int(dynamo_counters["stats"]["unique_graphs"]) - graphs_before
            )
            if first_compiled_graphs < 1 or total_compiled_graphs != first_compiled_graphs:
                raise RuntimeError(
                    "compiled EvoX graph count differs: "
                    f"first={first_compiled_graphs}, total={total_compiled_graphs}"
                )
            scorer.charged = int(scorer.cursor.item())
        if scorer.charged != BUDGET:
            raise RuntimeError(f"search charged {scorer.charged} rather than {BUDGET}")
        native_accounting = None
        if method.startswith("torch_dfo_"):
            result = runner.result
            native_accounting = {
                "scheduled": result.scheduled_evals,
                "charged": result.charged_evals,
                "attempted": result.attempted_evals,
                "completed": result.completed_evals,
                "stop_reason": result.stop_reason,
                "failures": list(result.failures),
                "accounting_uncertain": result.accounting_uncertain,
            }
            if (
                any(
                    native_accounting[key] != BUDGET
                    for key in ("scheduled", "charged", "attempted", "completed")
                )
                or result.stop_reason != "budget exhausted"
                or result.failures
                or result.accounting_uncertain
            ):
                raise RuntimeError(f"native managed-loop accounting differs: {native_accounting}")
        trace = scorer.trace()
        audit_trace(table, schedule, trace)
        best_pos = min(range(BUDGET), key=lambda i: trace["validation_errors"][i])
        return {
            "status": "complete",
            "scorer_staging_seconds": scorer_staging_seconds,
            "compiler_cache_reset_seconds": cache_reset_seconds,
            "construction_seconds": construction_seconds,
            "first_query_seconds": first_seconds,
            "first_compiled_generation_seconds": first_compiled_seconds,
            "first_compiled_unique_graphs": first_compiled_graphs,
            "total_compiled_unique_graphs": total_compiled_graphs,
            "steady_generations_seconds": steady_seconds,
            "full_search_seconds": search_seconds,
            "remaining_search_seconds": search_seconds - first_seconds,
            "charged": scorer.charged,
            "native_accounting": native_accounting,
            "unique_architectures": len(set(trace["architecture_ids"])),
            "tied_edges": sum(trace["tied_edges"]),
            "selected_architecture_id": trace["architecture_ids"][best_pos],
            "selected_validation_error": trace["validation_errors"][best_pos],
            "trace": trace,
        }
    except Exception as error:
        error.charged_evals = int(scorer.cursor.item()) if compile_evox else scorer.charged
        raise
    finally:
        torch.set_default_dtype(old_dtype)


def optimizer_loop_floor(method: str, seed: int, device: torch.device) -> dict:
    """Time the managed optimizer loop with a cheap device-resident objective."""
    from benchmarks.nasbench201 import make_native_optimizer

    class CheapObjective:
        def __init__(self) -> None:
            self.charged = 0

        def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
            self.charged += candidates.shape[0]
            return candidates.square().sum(dim=1)

    objective = CheapObjective()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        sync(device)
        construction_start = time.perf_counter()
        if method.startswith("torch_dfo_"):
            optimizer = make_native_optimizer(f"{method}_{device.type}", device, seed)
            runner = SearchRun(optimizer, BUDGET, evaluator_id="nasbench201-floor", repeats=1)

            def step(first: bool) -> None:
                runner.step(objective)

        else:
            workflow = make_workflow(method, device, objective)

            def step(first: bool) -> None:
                if first:
                    workflow.init_step()
                else:
                    workflow.step()

        sync(device)
        construction_seconds = time.perf_counter() - construction_start
        sync(device)
        start = time.perf_counter()
        step(True)
        sync(device)
        first_seconds = time.perf_counter() - start
        for _ in range(BUDGET // POPULATION - 1):
            step(False)
        sync(device)
        elapsed = time.perf_counter() - start
        if objective.charged != BUDGET:
            raise RuntimeError("optimizer floor did not charge the exact budget")
        return {
            "construction_seconds": construction_seconds,
            "first_query_seconds": first_seconds,
            "managed_loop_seconds": elapsed,
            "charged": objective.charged,
        }
    except Exception as error:
        error.charged_evals = objective.charged
        raise
    finally:
        torch.set_default_dtype(old_dtype)


def main() -> None:
    from benchmarks.nasbench201 import check_evox_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evox-source", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--compile-evox", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if "cuda" in args.devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.compile_evox and "evox_de" not in args.methods:
        raise ValueError("--compile-evox requires evox_de")
    if not args.validation.is_file():
        raise FileNotFoundError(args.validation)
    validation_hash = sha256(args.validation)
    if validation_hash != VALIDATION_SHA256:
        raise RuntimeError(f"validation export hash differs: {validation_hash}")
    if any(method.startswith("evox_") for method in args.methods):
        evox_source = check_evox_source(args.evox_source)
        if args.evox_source is not None:
            sys.path.insert(0, str(args.evox_source.resolve() / "src"))
    else:
        evox_source = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "scope": "Adapted device-resident NASBench201 hp200 validation scorer; no test queries",
        "validation_path": str(args.validation.resolve()),
        "validation_sha256": validation_hash,
        "evox_source": evox_source,
        "source_sha256": {
            "benchmarks/nasbench201_device.py": sha256(Path(__file__)),
            "benchmarks/nasbench201.py": sha256(ROOT / "benchmarks/nasbench201.py"),
            "research/redesign/nasbench201-protocol.md": sha256(
                ROOT / "research/redesign/nasbench201-protocol.md"
            ),
            "src/torch_dfo/cmaes.py": sha256(ROOT / "src/torch_dfo/cmaes.py"),
            "src/torch_dfo/shade.py": sha256(ROOT / "src/torch_dfo/shade.py"),
            "src/torch_dfo/run.py": sha256(ROOT / "src/torch_dfo/run.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "config": {
            "datasets": args.datasets,
            "methods": args.methods,
            "devices": args.devices,
            "seeds": args.seeds,
            "dtype": "float64",
            "population": POPULATION,
            "budget": BUDGET,
            "compiled_evox_de_attempt": args.compile_evox,
            "compiled_evox_fullgraph": args.compile_evox,
            "compiled_cache_reset_per_search": args.compile_evox,
        },
        "table_loads": [],
        "warmups": [],
        "objective_records": [],
        "records": [],
    }

    def save() -> None:
        temporary = args.output.with_name(args.output.name + ".tmp")
        with temporary.open("w") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)

    save()
    for dataset in args.datasets:
        table_start = time.perf_counter()
        table = load_validation_table(args.validation, dataset)
        report["table_loads"].append(
            {"dataset": dataset, "seconds": time.perf_counter() - table_start}
        )
        for device_name in args.devices:
            check_table_parity(table, torch.device(device_name))
        warm_schedule_start = time.perf_counter()
        warm_schedule = make_trial_schedule(table, 419)
        warm_schedule_seconds = time.perf_counter() - warm_schedule_start
        for method in args.methods:
            for device_name in args.devices:
                warmup = {
                    "dataset": dataset,
                    "seed": 419,
                    "method": method,
                    "device": device_name,
                    "schedule_build_seconds": warm_schedule_seconds,
                }
                try:
                    result = run_search(
                        method, table, warm_schedule, 419, torch.device(device_name)
                    )
                    warmup.update(
                        {
                            "status": "complete",
                            "charged": result["charged"],
                            "construction_seconds": result["construction_seconds"],
                            "first_query_seconds": result["first_query_seconds"],
                        }
                    )
                except Exception as error:
                    warmup.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "charged": getattr(error, "charged_evals", None),
                        }
                    )
                if method == "evox_de" and args.compile_evox:
                    try:
                        compiled = run_search(
                            method,
                            table,
                            warm_schedule,
                            419,
                            torch.device(device_name),
                            compile_evox=True,
                        )
                        warmup["compiled"] = {
                            "status": "complete",
                            "charged": compiled["charged"],
                            "compiler_cache_reset_seconds": compiled[
                                "compiler_cache_reset_seconds"
                            ],
                            "construction_seconds": compiled["construction_seconds"],
                            "first_query_seconds": compiled["first_query_seconds"],
                            "first_compiled_generation_seconds": compiled[
                                "first_compiled_generation_seconds"
                            ],
                            "first_compiled_unique_graphs": compiled[
                                "first_compiled_unique_graphs"
                            ],
                            "total_compiled_unique_graphs": compiled[
                                "total_compiled_unique_graphs"
                            ],
                            "first_compiled_generation_includes_compile": True,
                        }
                    except Exception as error:
                        warmup["compiled"] = {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "charged": getattr(error, "charged_evals", None),
                        }
                report["warmups"].append(warmup)
                save()
        warmup_status = {
            (item["method"], item["device"]): item["status"]
            for item in report["warmups"]
            if item["dataset"] == dataset
        }
        for seed in args.seeds:
            setup_start = time.perf_counter()
            schedule = make_trial_schedule(table, seed)
            schedule_seconds = time.perf_counter() - setup_start
            check_scorer_parity(table, schedule, seed, args.devices)
            objective_status = {}
            for device_name in args.devices:
                device = torch.device(device_name)
                objective_record = {
                    "dataset": dataset,
                    "seed": seed,
                    "device": device_name,
                    "trial_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                }
                try:
                    objective_record.update(objective_only(table, schedule, seed, device))
                    objective_record["status"] = "complete"
                except Exception as error:
                    objective_record.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                report["objective_records"].append(objective_record)
                objective_status[device_name] = objective_record["status"]
                save()
            order = [(method, device) for method in args.methods for device in args.devices]
            order = order[seed % len(order) :] + order[: seed % len(order)]
            for method, device_name in order:
                device = torch.device(device_name)
                record = {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "device": device_name,
                    "schedule_build_seconds": schedule_seconds,
                    "trial_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                    "warmup_status": warmup_status[(method, device_name)],
                    "objective_status": objective_status[device_name],
                }
                try:
                    record["optimizer_floor"] = optimizer_loop_floor(method, seed, device)
                except Exception as error:
                    record["optimizer_floor"] = {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "charged": getattr(error, "charged_evals", None),
                        "traceback": traceback.format_exc(),
                    }
                try:
                    record["search"] = run_search(method, table, schedule, seed, device)
                    record["status"] = (
                        "complete"
                        if record["optimizer_floor"].get("status") != "failed"
                        and record["warmup_status"] == "complete"
                        and record["objective_status"] == "complete"
                        else "partial"
                    )
                except Exception as error:
                    record["search"] = {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "charged": getattr(error, "charged_evals", None),
                        "traceback": traceback.format_exc(),
                    }
                    record["status"] = "failed"
                if method == "evox_de" and args.compile_evox:
                    compiled_warmup = next(
                        item["compiled"]["status"]
                        for item in report["warmups"]
                        if item["dataset"] == dataset
                        and item["method"] == method
                        and item["device"] == device_name
                    )
                    try:
                        compiled = run_search(
                            method, table, schedule, seed, device, compile_evox=True
                        )
                        compiled["compile_first_use_included_in_search_seconds"] = True
                        compiled["warmup_status"] = compiled_warmup
                        if compiled_warmup != "complete":
                            compiled["status"] = "partial"
                        record["compiled_search"] = compiled
                    except Exception as error:
                        record["compiled_search"] = {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "charged": getattr(error, "charged_evals", None),
                            "traceback": traceback.format_exc(),
                        }
                report["records"].append(record)
                save()
    failure_counts = {
        "warmups": sum(item["status"] == "failed" for item in report["warmups"]),
        "compiled_warmups": sum(
            item.get("compiled", {}).get("status") == "failed" for item in report["warmups"]
        ),
        "objective_only": sum(item["status"] == "failed" for item in report["objective_records"]),
        "optimizer_floor": sum(
            item["optimizer_floor"].get("status") == "failed" for item in report["records"]
        ),
        "search": sum(item["search"].get("status") == "failed" for item in report["records"]),
        "compiled_search": sum(
            item.get("compiled_search", {}).get("status") == "failed" for item in report["records"]
        ),
        "partial_lanes": sum(item["status"] == "partial" for item in report["records"]),
        "partial_compiled_lanes": sum(
            item.get("compiled_search", {}).get("status") == "partial" for item in report["records"]
        ),
    }
    report["failure_counts"] = failure_counts
    report["status"] = "complete" if not any(failure_counts.values()) else "partial"
    save()


if __name__ == "__main__":
    main()
