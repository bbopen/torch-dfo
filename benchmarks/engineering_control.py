#!/usr/bin/env python
"""Run the frozen quantized thermal-control reference study.

The study uses train data to select every candidate.  It evaluates each
selected candidate once on the held-out split.  It reports every evaluation
count, including that final validation call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "examples" / "06_engineering_control.py"


def _reference() -> Any:
    """Load the standalone example without making the examples directory a package."""
    name = "torch_dfo_engineering_control_reference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference example: {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record_selected(
    *,
    name: str,
    seed: int | None,
    selected: torch.Tensor,
    train: Any,
    heldout: Any,
    train_score: float,
    train_candidates: int,
    search_wall_time_s: float,
    device: torch.device,
) -> dict[str, Any]:
    """Validate one train-selected plan once on the held-out split."""
    _synchronize(device)
    started = time.perf_counter()
    heldout_evaluation = heldout.evaluate(selected)
    _synchronize(device)
    return {
        "algorithm": name,
        "seed": seed,
        "train_score": train_score,
        "heldout_score": float(heldout_evaluation.score.item()),
        "heldout_gate_pass": bool(heldout_evaluation.score.item() <= _reference().PASS_THRESHOLD),
        "train_candidate_evaluations": train_candidates,
        "validation_candidate_evaluations": 1,
        "train_scenario_rollouts": train_candidates * train.scenarios.count,
        "validation_scenario_rollouts": heldout.scenarios.count,
        "search_wall_time_s": search_wall_time_s,
        "validation_wall_time_s": time.perf_counter() - started,
    }


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA before timing. CPU calls return immediately."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_optimizer(name: str, seed: int, sigma0: float | None, device: torch.device) -> Any:
    """Create one current torch-dfo optimizer with the fixed population size."""
    import torch_dfo

    ref = _reference()
    if name == "random":
        return torch_dfo.RandomSearch(
            dim=ref.HORIZON,
            bounds=(-1.0, 1.0),
            pop_size=12,
            seed=seed,
            device=device,
            dtype=torch.float64,
        )
    if name == "cmaes":
        return torch_dfo.CMAES(
            dim=ref.HORIZON,
            bounds=(-1.0, 1.0),
            pop_size=12,
            sigma0=0.30 if sigma0 is None else sigma0,
            seed=seed,
            device=device,
            dtype=torch.float64,
        )
    if name == "shade":
        return torch_dfo.SHADE(
            dim=ref.HORIZON,
            bounds=(-1.0, 1.0),
            pop_size=12,
            seed=seed,
            device=device,
            dtype=torch.float64,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _run_torch_dfo(
    name: str,
    train: Any,
    budget: int,
    seed: int,
    device: torch.device,
    sigma0: float | None = None,
) -> tuple[torch.Tensor, int, float, float]:
    """Run one public SearchRun and return its counted incumbent."""
    from torch_dfo import SearchRun

    optimizer = _build_optimizer(name, seed, sigma0, device)
    run = SearchRun(optimizer, max_evals=budget, evaluator_id="thermal-control-v2", repeats=1)
    _synchronize(device)
    started = time.perf_counter()
    while not run.done:
        run.step(train)
    _synchronize(device)
    result = run.result
    if result.failures:
        raise RuntimeError(
            f"search failed after {result.charged_evals} evaluations: {result.failures}"
        )
    if result.best_x is None or result.best_value is None:
        raise RuntimeError("optimizer did not produce a candidate")
    return (
        result.best_x.detach().clone().unsqueeze(0),
        result.charged_evals,
        float(result.best_value.item()),
        time.perf_counter() - started,
    )


def run_development_trials(budget: int, device: torch.device) -> list[dict[str, Any]]:
    """Run the predeclared train-only CMA-ES sigma trials.

    A lower final train score wins. Ties select the lower sigma in the caller.
    This function never creates or reads a held-out evaluator.
    """
    ref = _reference()
    train = ref.FrozenEvaluator("train")
    rows: list[dict[str, Any]] = []
    for sigma0 in (0.15, 0.30, 0.55):
        _, used, best_value, elapsed = _run_torch_dfo(
            "cmaes", train, budget, seed=10_001, device=device, sigma0=sigma0
        )
        rows.append(
            {
                "sigma0": sigma0,
                "train_score": best_value,
                "train_candidate_evaluations": used,
                "train_scenario_rollouts": used * train.scenarios.count,
                "search_wall_time_s": elapsed,
                "decision": "pending",
            }
        )
    winner = min(rows, key=lambda row: (row["train_score"], row["sigma0"]))
    for row in rows:
        row["decision"] = "keep" if row is winner else "reject"
    return rows


def _grid_constant(train: Any, device: torch.device) -> tuple[torch.Tensor, int, float, float]:
    """Select the best constant command from the quantization grid on train."""
    ref = _reference()
    commands = torch.arange(-1.0, 1.01, 0.25, device=device, dtype=torch.float64)
    candidates = commands.unsqueeze(1).expand(-1, ref.HORIZON).clone()
    _synchronize(device)
    started = time.perf_counter()
    scores = train(candidates)
    _synchronize(device)
    index = int(scores.argmin().item())
    return (
        candidates[index].unsqueeze(0),
        int(candidates.shape[0]),
        float(scores[index].item()),
        time.perf_counter() - started,
    )


def _run_evotorch_cma(
    train: Any,
    budget: int,
    seed: int,
    sigma0: float,
    evotorch_src: Path | None,
    device: torch.device,
) -> tuple[torch.Tensor, int, float, float]:
    """Run an optional upstream EvoTorch CMA-ES baseline on full populations.

    EvoTorch owns its ask, evaluate, and update sequence. This adapter counts
    each complete 12-member population. EvoTorch CMA-ES requires an unbounded
    latent problem. The physical simulator clamps the selected command plan.
    """
    if evotorch_src is not None:
        source = str(evotorch_src.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        from evotorch import Problem
        from evotorch.algorithms import CMAES
    except ImportError as exc:
        raise RuntimeError(
            "EvoTorch is unavailable. Pass --evotorch-src /path/to/evotorch/src "
            "or install its documented dependencies."
        ) from exc

    ref = _reference()

    objective_calls = 0

    def objective(candidates: torch.Tensor) -> torch.Tensor:
        nonlocal objective_calls
        objective_calls += int(candidates.shape[0])
        return train(candidates.clamp(-1.0, 1.0))

    problem = Problem(
        "min",
        objective,
        initial_bounds=(-1.0, 1.0),
        solution_length=ref.HORIZON,
        dtype=torch.float64,
        eval_dtype=torch.float64,
        device=device,
        seed=seed,
        vectorized=True,
    )
    optimizer = CMAES(
        problem,
        center_init=torch.zeros(ref.HORIZON, dtype=torch.float64, device=device),
        stdev_init=sigma0 * 2.0,
        popsize=12,
        active=False,
    )
    best_x: torch.Tensor | None = None
    best_value = float("inf")
    _synchronize(device)
    started = time.perf_counter()
    while objective_calls + 12 <= budget:
        optimizer.step()
        candidates = optimizer.population.values
        values = optimizer.population.access_evals(0)
        index = int(values.argmin().item())
        value = float(values[index].item())
        if value < best_value:
            best_value = value
            best_x = candidates[index].detach().clone().clamp(-1.0, 1.0).unsqueeze(0)
    _synchronize(device)
    if best_x is None:
        raise RuntimeError("EvoTorch did not produce a candidate")
    return best_x, objective_calls, best_value, time.perf_counter() - started


def run_engineering_benchmark(
    *,
    budget: int = 96,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105),
    include_evotorch: bool = False,
    evotorch_src: Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the frozen study and return a JSON-serializable full report."""
    if budget < 12:
        raise ValueError("budget must allow at least one 12-member population")
    ref = _reference()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train = ref.FrozenEvaluator("train")
    probes = ref.run_evaluator_probes(device=device)
    if probes["degenerate_zero"]["passes"] or not probes["ceiling_domain_control"]["passes"]:
        raise RuntimeError("the frozen evaluator failed its required phase-zero probes")
    if probes["runaway_all_heat"]["passes"]:
        raise RuntimeError("the frozen evaluator accepted the runaway heating probe")
    trials = run_development_trials(budget, device)
    selected_sigma = min(trials, key=lambda row: (row["train_score"], row["sigma0"]))["sigma0"]
    heldout = ref.FrozenEvaluator("heldout")
    audit_candidates = 68
    audit_rollouts = audit_candidates * train.scenarios.count

    records: list[dict[str, Any]] = []
    pi = ref.mean_scenario_pi_control(train.scenarios, device=device, dtype=torch.float64)
    _synchronize(device)
    started = time.perf_counter()
    pi_score = float(train(pi).item())
    _synchronize(device)
    pi_elapsed = time.perf_counter() - started
    records.append(
        _record_selected(
            name="mean-scenario-pi",
            seed=None,
            selected=pi,
            train=train,
            heldout=heldout,
            train_score=pi_score,
            train_candidates=1,
            search_wall_time_s=pi_elapsed,
            device=device,
        )
    )
    grid, grid_count, grid_score, grid_elapsed = _grid_constant(train, device)
    records.append(
        _record_selected(
            name="constant-grid",
            seed=None,
            selected=grid,
            train=train,
            heldout=heldout,
            train_score=grid_score,
            train_candidates=grid_count,
            search_wall_time_s=grid_elapsed,
            device=device,
        )
    )
    for seed in seeds:
        for name, sigma0 in (("random", None), ("cmaes", selected_sigma), ("shade", None)):
            selected, train_count, train_score, elapsed = _run_torch_dfo(
                name, train, budget, seed, device, sigma0
            )
            records.append(
                _record_selected(
                    name=name,
                    seed=seed,
                    selected=selected,
                    train=train,
                    heldout=heldout,
                    train_score=train_score,
                    train_candidates=train_count,
                    search_wall_time_s=elapsed,
                    device=device,
                )
            )
        if include_evotorch:
            selected, train_count, train_score, elapsed = _run_evotorch_cma(
                train, budget, seed, selected_sigma, evotorch_src, device
            )
            records.append(
                _record_selected(
                    name="evotorch-cmaes",
                    seed=seed,
                    selected=selected,
                    train=train,
                    heldout=heldout,
                    train_score=train_score,
                    train_candidates=train_count,
                    search_wall_time_s=elapsed,
                    device=device,
                )
            )

    total_train = sum(row["train_candidate_evaluations"] for row in records) + sum(
        row["train_candidate_evaluations"] for row in trials
    )
    total_validation = sum(row["validation_candidate_evaluations"] for row in records)
    return {
        "task_version": ref.TASK_VERSION,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                REFERENCE_PATH,
                Path(__file__),
                *sorted((ROOT / "src/torch_dfo").rglob("*.py")),
            )
        },
        "task_hash": ref.task_hash(),
        "evaluator_hashes": {"train": train.evaluator_hash, "heldout": heldout.evaluator_hash},
        "scenario_hashes": {"train": train.scenario_hash, "heldout": heldout.scenario_hash},
        "budget_requested_per_search": budget,
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "seeds": list(seeds),
        "probes_train_only": probes,
        "development_trials_train_only": trials,
        "selected_cma_sigma0": selected_sigma,
        "evotorch": {
            "enabled": include_evotorch,
            "source": None if evotorch_src is None else str(evotorch_src),
            "configuration": {
                "population_size": 12,
                "center": "zeros",
                "stdev_init": selected_sigma * 2.0,
                "active_covariance": False,
                "latent_problem_bounds": None,
                "evaluated_plan_bounds": [-1.0, 1.0],
                "note": (
                    "EvoTorch CMA-ES adapts unbounded latent proposals. "
                    "The simulator clamps them before evaluation."
                ),
            },
        },
        "results": records,
        "cost_accounting": {
            "total_outer_search_candidate_evaluations": total_train,
            "total_validation_candidate_evaluations": total_validation,
            "total_candidate_evaluations": total_train + total_validation,
            "evaluator_audit_candidate_evaluations": audit_candidates,
            "evaluator_audit_scenario_rollouts": audit_rollouts,
            "total_including_audit_candidate_evaluations": (
                total_train + total_validation + audit_candidates
            ),
            "total_train_scenario_rollouts": sum(row["train_scenario_rollouts"] for row in records)
            + sum(row["train_scenario_rollouts"] for row in trials),
            "total_validation_scenario_rollouts": sum(
                row["validation_scenario_rollouts"] for row in records
            ),
        },
    }


def main() -> None:
    """Run the reference benchmark and print the complete JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evotorch", action="store_true")
    parser.add_argument("--evotorch-src", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run_engineering_benchmark(
        budget=args.budget,
        include_evotorch=args.evotorch,
        evotorch_src=args.evotorch_src,
        device=args.device,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
