#!/usr/bin/env python
"""Run a small CPU calibration comparison on calisim's Lotka-Volterra example.

The study fits the first 16 annual lynx observations and scores the selected
training incumbent once on the final five observations. It is a bounded
temporal split of a public example, not a general calibration claim.

Adapted task source: https://github.com/Plant-Food-Research-Open/calisim/blob/
fa6539b2d1e4a39210fef3d9dabee97b893ec69c/examples/evolutionary/evotorch_example.py
See benchmarks/licenses/calisim-LICENSE.txt for the Apache-2.0 source notice.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CALISIM_REVISION = "fa6539b2d1e4a39210fef3d9dabee97b893ec69c"
CALISIM_MODEL = Path("calisim/example_models/lotka_volterra.py")
CALISIM_EXAMPLE = Path("examples/evolutionary/evotorch_example.py")
TASK_VERSION = "calisim-lotka-temporal-v1"
POPULATION_SIZE = 20
EVALUATION_BUDGET = 400
SEEDS = (0, 1, 2, 3, 4)
UNIT_LOWER = np.zeros(2, dtype=np.float64)
UNIT_UPPER = np.ones(2, dtype=np.float64)
PHYSICAL_LOWER = np.array([0.45, 0.02], dtype=np.float64)
PHYSICAL_UPPER = np.array([0.55, 0.03], dtype=np.float64)
YEARS = np.arange(1900.0, 1921.0, 1.0)
OBSERVED_LYNX = np.array(
    [
        4.0,
        6.1,
        9.8,
        35.2,
        59.4,
        41.7,
        19.0,
        13.0,
        8.3,
        9.1,
        7.4,
        8.0,
        12.3,
        19.5,
        45.7,
        51.1,
        29.7,
        15.8,
        9.7,
        10.1,
        8.6,
    ],
    dtype=np.float64,
)
TRAIN_YEARS = YEARS[:16]
HELDOUT_YEARS = YEARS[16:]
TRAIN_OBSERVATIONS = OBSERVED_LYNX[:16]
HELDOUT_OBSERVATIONS = OBSERVED_LYNX[16:]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file_hash(value: Any) -> dict[str, str]:
    """Return the resolved implementation file and digest for one runtime class."""
    source = inspect.getsourcefile(value)
    if source is None:
        raise RuntimeError(f"cannot identify source file for {value}")
    path = Path(source).resolve()
    if not path.is_file():
        raise RuntimeError(f"runtime source file is missing: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _json_hash(value: Any) -> str:
    """Return a stable digest for JSON-compatible protocol data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_unit_matrix(candidates: Any) -> np.ndarray:
    """Validate and return a finite two-parameter unit-space matrix."""
    if isinstance(candidates, torch.Tensor):
        values = candidates.detach().cpu().numpy()
    else:
        values = np.asarray(candidates, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 1:
        raise ValueError("unit candidates must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ValueError("unit candidates must be finite")
    if np.any(values < UNIT_LOWER) or np.any(values > UNIT_UPPER):
        raise ValueError("unit candidates must be within [0, 1]^2")
    return values.astype(np.float64, copy=True)


def physical_from_unit(candidates: Any) -> np.ndarray:
    """Map validated unit candidates to alpha and beta in source units."""
    unit = _as_unit_matrix(candidates)
    return PHYSICAL_LOWER + unit * (PHYSICAL_UPPER - PHYSICAL_LOWER)


def unit_from_physical(parameters: Any) -> np.ndarray:
    """Map finite source-unit parameters back to the common unit box."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 1:
        raise ValueError("physical parameters must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ValueError("physical parameters must be finite")
    unit = (values - PHYSICAL_LOWER) / (PHYSICAL_UPPER - PHYSICAL_LOWER)
    if np.any(unit < UNIT_LOWER) or np.any(unit > UNIT_UPPER):
        raise ValueError("physical parameters must be within the source bounds")
    return unit


def _require_years(years: Any) -> np.ndarray:
    """Validate requested annual prediction times."""
    values = np.asarray(years, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("years must be a non-empty finite vector")
    if np.any(np.diff(values) <= 0) or values[0] < YEARS[0] or values[-1] > YEARS[-1]:
        raise ValueError("years must be increasing and within 1900 through 1920")
    return values


def simulate_lynx(parameters: Any, years: Any) -> np.ndarray:
    """Simulate the public source model for valid alpha, beta, and years."""
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the calibration benchmark") from exc

    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("parameters must be one finite alpha, beta pair")
    if np.any(values < PHYSICAL_LOWER) or np.any(values > PHYSICAL_UPPER):
        raise ValueError("parameters must be within the public source bounds")
    requested_years = _require_years(years)
    alpha, beta = (float(value) for value in values)

    def derivative(_: float, state: np.ndarray) -> np.ndarray:
        hare, lynx = state
        return np.array(
            [
                alpha * hare - beta * hare * lynx,
                -0.84 * lynx + 0.026 * hare * lynx,
            ],
            dtype=np.float64,
        )

    solution = solve_ivp(
        fun=derivative,
        y0=[34.0, 5.9],
        t_span=(YEARS[0], float(requested_years[-1])),
        t_eval=requested_years,
    )
    if not solution.success or solution.y.shape != (2, requested_years.size):
        raise RuntimeError(f"Lotka-Volterra integration failed: {solution.message}")
    lynx = solution.y[1, :]
    if not np.isfinite(lynx).all():
        raise RuntimeError("Lotka-Volterra integration returned non-finite lynx values")
    return lynx


def mean_squared_error(observed: Any, predicted: Any) -> float:
    """Return finite MSE for two finite, equally shaped vectors."""
    expected = np.asarray(observed, dtype=np.float64)
    actual = np.asarray(predicted, dtype=np.float64)
    if expected.ndim != 1 or actual.shape != expected.shape or expected.size < 1:
        raise ValueError("observed and predicted must be non-empty matching vectors")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("observed and predicted values must be finite")
    value = float(np.mean((expected - actual) ** 2))
    if not np.isfinite(value):
        raise RuntimeError("MSE is non-finite")
    return value


@dataclass
class CalibrationObjective:
    """Instrumented objective for one immutable time split."""

    split: Literal["train", "heldout"]
    observations: np.ndarray | None = None
    candidate_calls: int = 0
    objective_calls: int = 0

    def __post_init__(self) -> None:
        if self.split == "train":
            self.years = TRAIN_YEARS.copy()
            self.observed = TRAIN_OBSERVATIONS.copy()
        elif self.split == "heldout":
            self.years = HELDOUT_YEARS.copy()
            self.observed = HELDOUT_OBSERVATIONS.copy()
        else:
            raise ValueError("split must be train or heldout")
        if self.observations is not None:
            replacement = np.asarray(self.observations, dtype=np.float64)
            if replacement.shape != self.observed.shape or not np.isfinite(replacement).all():
                raise ValueError("replacement observations must match the selected split")
            self.observed = replacement.copy()

    def score_unit(self, candidates: Any) -> np.ndarray:
        """Evaluate one or more unit candidates and count actual candidates."""
        physical = physical_from_unit(candidates)
        self.objective_calls += 1
        scores: list[float] = []
        for row in physical:
            self.candidate_calls += 1
            prediction = simulate_lynx(row, self.years)
            scores.append(mean_squared_error(self.observed, prediction))
        scores_array = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(scores_array).all():
            raise RuntimeError("calibration objective returned non-finite scores")
        return scores_array

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        """Return one score per Torch candidate without changing its values."""
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("candidates must be a torch tensor")
        scores = self.score_unit(candidates)
        return torch.as_tensor(scores, device=candidates.device, dtype=candidates.dtype)


def run_evaluator_probes() -> dict[str, Any]:
    """Run four checks that attack the frozen train-only evaluator."""
    reference = np.array([0.52, 0.024], dtype=np.float64)
    trajectory = simulate_lynx(reference, TRAIN_YEARS)
    synthetic_objective = CalibrationObjective("train", observations=trajectory.copy())
    synthetic_mse = float(synthetic_objective.score_unit(unit_from_physical(reference))[0])
    wrong_parameter_mse = float(
        synthetic_objective.score_unit(unit_from_physical(np.array([0.45, 0.02])))[0]
    )

    degenerate_mse = mean_squared_error(TRAIN_OBSERVATIONS, np.zeros_like(TRAIN_OBSERVATIONS))
    mean_prediction_mse = mean_squared_error(
        TRAIN_OBSERVATIONS,
        np.full_like(TRAIN_OBSERVATIONS, TRAIN_OBSERVATIONS.mean()),
    )
    objective = CalibrationObjective("train")
    cached_fitness = 0.0
    physical_recheck = float(objective.score_unit(unit_from_physical(reference))[0])

    malformed_inputs = (
        np.array([[np.nan, 0.5]], dtype=np.float64),
        np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
        np.array([[1.1, 0.5]], dtype=np.float64),
    )
    rejected = 0

    def reject(candidate: np.ndarray) -> None:
        nonlocal rejected
        try:
            physical_from_unit(candidate)
        except ValueError:
            rejected += 1

    for malformed in malformed_inputs:
        reject(malformed)

    results = {
        "known_synthetic_trajectory": {
            "mse": synthetic_mse,
            "candidate_calls": synthetic_objective.candidate_calls - 1,
            "objective_calls": synthetic_objective.objective_calls - 1,
            "passes": bool(synthetic_mse <= 1e-24),
        },
        "degenerate_prediction": {
            "zero_prediction_mse": degenerate_mse,
            "mean_prediction_mse": mean_prediction_mse,
            "candidate_calls": 0,
            "objective_calls": 0,
            "passes": bool(degenerate_mse > 0.0 and mean_prediction_mse > 0.0),
        },
        "cached_fitness_recheck": {
            "cached_fitness": cached_fitness,
            "physical_mse": physical_recheck,
            "candidate_calls": objective.candidate_calls,
            "objective_calls": objective.objective_calls,
            "passes": bool(physical_recheck != cached_fitness and objective.candidate_calls == 1),
        },
        "wrong_parameter_and_null_controls": {
            "wrong_parameter_mse": wrong_parameter_mse,
            "candidate_calls": 1,
            "objective_calls": 1,
            "rejected_malformed_inputs": rejected,
            "passes": bool(
                np.isfinite(wrong_parameter_mse)
                and wrong_parameter_mse > synthetic_mse
                and rejected == 3
            ),
        },
    }
    if not all(bool(row["passes"]) for row in results.values()):
        raise RuntimeError("the train evaluator did not pass its required probes")
    return {
        "probes": results,
        "cost_accounting": {
            "candidate_calls": sum(int(row["candidate_calls"]) for row in results.values()),
            "objective_calls": sum(int(row["objective_calls"]) for row in results.values()),
            "synthetic_fixture_simulator_calls": 1,
        },
    }


def _build_optimizer(name: str, seed: int) -> Any:
    """Build one native optimizer with the frozen population and unit box."""
    import torch_dfo

    common = {
        "dim": 2,
        "bounds": (0.0, 1.0),
        "pop_size": POPULATION_SIZE,
        "seed": seed,
        "device": "cpu",
        "dtype": torch.float64,
    }
    if name == "random":
        return torch_dfo.RandomSearch(**common)
    if name == "cmaes":
        return torch_dfo.CMAES(**common, sigma0=0.1)
    if name == "shade":
        return torch_dfo.SHADE(**common)
    raise ValueError(f"unknown native algorithm: {name}")


def run_native_search(
    name: str, objective: Any, *, budget: int, seed: int
) -> tuple[np.ndarray, float, dict[str, int | float | None]]:
    """Run one native SearchRun and verify callback and budget accounting."""
    from torch_dfo import SearchRun

    started = time.perf_counter()
    optimizer = _build_optimizer(name, seed)
    run = SearchRun(optimizer, max_evals=budget, evaluator_id="calisim-lotka-train", repeats=1)
    while not run.done:
        run.step(objective)
    elapsed = time.perf_counter() - started
    result = run.result
    if result.failures:
        raise RuntimeError(
            f"{name} failed after {result.charged_evals} evaluations: {result.failures}"
        )
    if result.best_x is None or result.best_value is None:
        raise RuntimeError(f"{name} produced no train incumbent")
    candidate_calls = int(objective.candidate_calls)
    objective_calls = int(objective.objective_calls)
    if result.charged_evals != budget or candidate_calls != result.charged_evals:
        raise RuntimeError(
            f"{name} accounting mismatch: SearchRun={result.charged_evals}, "
            f"callback={candidate_calls}"
        )
    return (
        _as_unit_matrix(result.best_x),
        float(result.best_value.item()),
        {
            "candidate_evaluations": candidate_calls,
            "objective_callback_calls": objective_calls,
            "searchrun_charged_evaluations": result.charged_evals,
            "initialization_and_search_seconds": elapsed,
        },
    )


def _run_evotorch_cmaes(
    objective: CalibrationObjective, *, budget: int, seed: int, evotorch_src: Path | None
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run optional EvoTorch CMA-ES with external best-seen tracking.

    EvoTorch uses unbounded latent CMA proposals. This adapter clamps each
    proposal into the same unit box before it reaches the physical objective.
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
            "EvoTorch was requested but is unavailable. Pass --evotorch-src or install EvoTorch."
        ) from exc

    best_unit: np.ndarray | None = None
    best_value = float("inf")

    def evaluator(candidates: torch.Tensor) -> torch.Tensor:
        nonlocal best_unit, best_value
        bounded = candidates.clamp(0.0, 1.0)
        scores = objective(bounded)
        index = int(torch.argmin(scores).item())
        value = float(scores[index].item())
        if value < best_value:
            best_value = value
            best_unit = _as_unit_matrix(bounded[index]).reshape(2)
        return scores

    started = time.perf_counter()
    problem = Problem(
        "min",
        evaluator,
        initial_bounds=(0.0, 1.0),
        solution_length=2,
        dtype=torch.float64,
        eval_dtype=torch.float64,
        device="cpu",
        seed=seed,
        vectorized=True,
    )
    optimizer = CMAES(
        problem,
        center_init=torch.full((2,), 0.5, dtype=torch.float64),
        stdev_init=0.1,
        popsize=POPULATION_SIZE,
        active=False,
    )
    while objective.candidate_calls + POPULATION_SIZE <= budget:
        optimizer.step()
    elapsed = time.perf_counter() - started
    if best_unit is None:
        raise RuntimeError("EvoTorch CMA-ES produced no train incumbent")
    if objective.candidate_calls != budget:
        raise RuntimeError(
            f"EvoTorch accounting mismatch: callback={objective.candidate_calls}, budget={budget}"
        )
    return (
        best_unit.reshape(1, 2),
        best_value,
        {
            "candidate_evaluations": objective.candidate_calls,
            "objective_callback_calls": objective.objective_calls,
            "searchrun_charged_evaluations": None,
            "initialization_and_search_seconds": elapsed,
            "evotorch_cmaes_source": _source_file_hash(CMAES),
        },
    )


def _run_scipy_differential_evolution(
    objective: CalibrationObjective, *, seed: int
) -> tuple[np.ndarray, float, dict[str, int | float | bool | None]]:
    """Run the fixed SciPy DE baseline and retain the actual best callback value."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the differential-evolution baseline") from exc

    best_unit: np.ndarray | None = None
    best_value = float("inf")

    def evaluator(candidate: np.ndarray) -> float:
        nonlocal best_unit, best_value
        score = float(objective.score_unit(np.asarray(candidate, dtype=np.float64))[0])
        if score < best_value:
            best_value = score
            best_unit = _as_unit_matrix(candidate).reshape(2)
        return score

    started = time.perf_counter()
    differential_evolution(
        evaluator,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        popsize=10,
        maxiter=19,
        polish=False,
        tol=0.0,
        atol=0.0,
        seed=seed,
        workers=1,
        updating="deferred",
    )
    elapsed = time.perf_counter() - started
    if best_unit is None:
        raise RuntimeError("SciPy differential evolution produced no train incumbent")
    return (
        best_unit.reshape(1, 2),
        best_value,
        {
            "candidate_evaluations": objective.candidate_calls,
            "objective_callback_calls": objective.objective_calls,
            "searchrun_charged_evaluations": None,
            "initialization_and_search_seconds": elapsed,
            "candidate_budget_expected": EVALUATION_BUDGET,
            "candidate_budget_complete": objective.candidate_calls == EVALUATION_BUDGET,
        },
    )


def _source_metadata(calisim_root: Path) -> dict[str, Any]:
    """Verify and identify the requested public source checkout."""
    import torch_dfo

    active_package = Path(torch_dfo.__file__).resolve()
    expected_package = (ROOT / "src" / "torch_dfo" / "__init__.py").resolve()
    if active_package != expected_package:
        raise RuntimeError(
            f"benchmark must import this checkout's torch_dfo; loaded {active_package}. "
            "Run with PYTHONPATH=src."
        )
    root = calisim_root.resolve()
    model = root / CALISIM_MODEL
    example = root / CALISIM_EXAMPLE
    if not model.is_file():
        raise RuntimeError(f"calisim model source is missing: {model}")
    if not example.is_file():
        raise RuntimeError(f"calisim example source is missing: {example}")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != CALISIM_REVISION:
        raise RuntimeError(f"calisim revision must be {CALISIM_REVISION}, found {revision}")
    torch_dfo_sources = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted((ROOT / "src" / "torch_dfo").rglob("*.py"))
    }
    return {
        "torch_dfo_version": torch_dfo.__version__,
        "torch_dfo_active_package": str(active_package),
        "calisim_revision": revision,
        "calisim_model_sha256": _sha256(model),
        "calisim_example_sha256": _sha256(example),
        "benchmark_sha256": _sha256(Path(__file__)),
        "torch_dfo_source_sha256": torch_dfo_sources,
    }


def _select_and_audit(
    name: str,
    seed: int,
    selected_unit: np.ndarray,
    train_value: float,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    """Score one training-selected incumbent once on held-out observations."""
    heldout = CalibrationObjective("heldout")
    heldout_value = float(heldout.score_unit(selected_unit)[0])
    if heldout.candidate_calls != 1 or heldout.objective_calls != 1:
        raise RuntimeError("held-out audit must evaluate exactly one selected candidate")
    return {
        "algorithm": name,
        "seed": seed,
        "selected_unit_parameters": selected_unit.reshape(2).tolist(),
        "selected_physical_parameters": physical_from_unit(selected_unit).reshape(2).tolist(),
        "train_mse": train_value,
        "heldout_mse": heldout_value,
        **accounting,
        "final_audit_candidate_calls": heldout.candidate_calls,
        "final_audit_objective_calls": heldout.objective_calls,
    }


class BenchmarkRunFailure(RuntimeError):
    """Carry completed records and current accounting after a run failure."""

    def __init__(self, error: Exception, completed: list[dict[str, Any]], state: dict[str, Any]):
        super().__init__(str(error))
        self.completed = completed.copy()
        self.state = state


def _medians(records: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Return one median train, held-out, and elapsed value per algorithm."""
    names = sorted({str(row["algorithm"]) for row in records})
    rows: list[dict[str, float | str]] = []
    for name in names:
        selected = [row for row in records if row["algorithm"] == name]
        rows.append(
            {
                "algorithm": name,
                "median_train_mse": float(np.median([row["train_mse"] for row in selected])),
                "median_heldout_mse": float(np.median([row["heldout_mse"] for row in selected])),
                "median_initialization_and_search_seconds": float(
                    np.median([row["initialization_and_search_seconds"] for row in selected])
                ),
            }
        )
    return rows


def run_calibration_benchmark(
    *,
    calisim_root: Path,
    include_evotorch: bool = False,
    evotorch_src: Path | None = None,
    budget: int = EVALUATION_BUDGET,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Run the frozen five-seed CPU study and return the complete report."""
    if budget != EVALUATION_BUDGET:
        raise ValueError(f"the frozen protocol requires budget={EVALUATION_BUDGET}")
    if budget % POPULATION_SIZE:
        raise ValueError("budget must be divisible by the fixed population size")
    if tuple(seeds) != SEEDS:
        raise ValueError(f"the frozen protocol requires seeds={SEEDS}")
    torch.set_num_threads(1)
    source = _source_metadata(calisim_root)
    probe_report = run_evaluator_probes()
    try:
        import scipy
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the calibration benchmark") from exc
    records: list[dict[str, Any]] = []

    def append_run(name: str, seed: int, runner: Any) -> None:
        train = CalibrationObjective("train")
        try:
            selected, train_value, accounting = runner(train)
            records.append(_select_and_audit(name, seed, selected, train_value, accounting))
        except Exception as exc:
            raise BenchmarkRunFailure(
                exc,
                records,
                {
                    "algorithm": name,
                    "seed": seed,
                    "candidate_calls": train.candidate_calls,
                    "objective_calls": train.objective_calls,
                },
            ) from exc

    for seed in seeds:
        for name in ("random", "cmaes", "shade"):
            append_run(
                name,
                seed,
                lambda train, name=name, seed=seed: run_native_search(
                    name, train, budget=budget, seed=seed
                ),
            )
        append_run(
            "scipy-differential-evolution",
            seed,
            lambda train, seed=seed: _run_scipy_differential_evolution(train, seed=seed),
        )
        if include_evotorch:
            append_run(
                "evotorch-cmaes",
                seed,
                lambda train, seed=seed: _run_evotorch_cmaes(
                    train, budget=budget, seed=seed, evotorch_src=evotorch_src
                ),
            )
    protocol = {
        "train_years": TRAIN_YEARS.astype(int).tolist(),
        "heldout_years": HELDOUT_YEARS.astype(int).tolist(),
        "observed_lynx": OBSERVED_LYNX.tolist(),
        "train_observation_count": int(TRAIN_OBSERVATIONS.size),
        "heldout_observation_count": int(HELDOUT_OBSERVATIONS.size),
        "budget_per_run": budget,
        "population_size": POPULATION_SIZE,
        "seeds": list(seeds),
        "unit_bounds": [0.0, 1.0],
        "physical_bounds": [PHYSICAL_LOWER.tolist(), PHYSICAL_UPPER.tolist()],
        "native_methods": ["random", "cmaes", "shade"],
        "cmaes_initial_mean": [0.5, 0.5],
        "native_cmaes_sigma0": 0.1,
        "scipy_differential_evolution": {
            "bounds": [0.0, 1.0],
            "popsize": 10,
            "population_members": 20,
            "maxiter": 19,
            "polish": False,
            "tol": 0.0,
            "atol": 0.0,
            "workers": 1,
            "updating": "deferred",
            "expected_candidate_calls_without_early_stop": 400,
        },
        "evotorch_stdev_init": 0.1,
        "evotorch_proposal_handling": (
            "unbounded latent proposals are clamped to [0, 1]^2 before physical evaluation"
        ),
    }
    return {
        "task_version": TASK_VERSION,
        "protocol_hash": _json_hash({"protocol": protocol, "source": source}),
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "protocol": protocol,
        "probes_train_only": probe_report["probes"],
        "probe_cost_accounting": probe_report["cost_accounting"],
        "evotorch": {
            "enabled": include_evotorch,
            "source": None if evotorch_src is None else str(evotorch_src),
        },
        "raw_per_seed_results": records,
        "medians": _medians(records),
        "claim_limit": (
            "This report has no superiority flag. Five held-out years from one public series "
            "do not show broad generalization."
        ),
    }


def main() -> None:
    """Run the calibration comparison and write a JSON report when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calisim-root", type=Path, default=ROOT.parent / "downstream-calisim")
    parser.add_argument("--evotorch", action="store_true")
    parser.add_argument("--evotorch-src", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run_calibration_benchmark(
            calisim_root=args.calisim_root,
            include_evotorch=args.evotorch,
            evotorch_src=args.evotorch_src,
        )
    except BenchmarkRunFailure as exc:
        failure = {
            "status": "failed",
            "error": str(exc),
            "completed_raw_per_seed_results": exc.completed,
            "current_failure": exc.state,
        }
        text = json.dumps(failure, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        raise
    except Exception as exc:
        failure = {"status": "failed", "error": str(exc)}
        text = json.dumps(failure, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        raise
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
