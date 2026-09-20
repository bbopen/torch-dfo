#!/usr/bin/env python
"""Comprehensive benchmark harness for torch-dfo.

Runs PhasedDFO against multiple benchmark sources and baselines,
producing a structured report.

Usage:
    python benchmarks/run_benchmarks.py                    # full suite
    python benchmarks/run_benchmarks.py --suite bbob       # COCO/BBOB only
    python benchmarks/run_benchmarks.py --suite bbob --dims 10
    python benchmarks/run_benchmarks.py --suite yahpo      # YAHPO HPO only
    python benchmarks/run_benchmarks.py --suite quick       # smaller suite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional: ConfigSpace integer HP types (used in _tensor_to_yahpo_config)
# ---------------------------------------------------------------------------
try:
    from ConfigSpace.hyperparameters import (
        NormalIntegerHyperparameter as _NormalIntHP,
    )
    from ConfigSpace.hyperparameters import (
        UniformIntegerHyperparameter as _UniformIntHP,
    )

    _CS_INT_HP_TYPES: tuple[type, ...] = (_UniformIntHP, _NormalIntHP)
except ImportError:
    _CS_INT_HP_TYPES = ()

# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of a single benchmark problem evaluation."""

    suite: str
    problem_name: str
    dim: int
    optimizer: str
    best_fitness: float
    precision: float  # best_fitness - f_opt (if known)
    fe_used: int
    wall_time_s: float
    solved: bool  # Official suite target hit; False when no target is defined.


@dataclass
class SuiteReport:
    """Aggregated results for a benchmark suite."""

    suite_name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def solved_count(self) -> int:
        return sum(1 for r in self.results if r.solved)

    def solved_by_optimizer(self, name: str) -> int:
        return sum(1 for r in self.results if r.optimizer == name and r.solved)

    def yahpo_wins_vs_random(self) -> tuple[int, int]:
        """Count how many YAHPO problems torch-dfo beats random baseline."""
        problems = sorted(
            {
                (r.problem_name, r.dim)
                for r in self.results
                if r.suite == "yahpo" and r.optimizer == "torch-dfo"
            }
        )
        wins, total = 0, 0
        for name, dim in problems:
            rows = [
                r
                for r in self.results
                if r.suite == "yahpo" and r.problem_name == name and r.dim == dim
            ]
            dfo = [r.best_fitness for r in rows if r.optimizer == "torch-dfo"]
            rand = [r.best_fitness for r in rows if r.optimizer == "random"]
            if dfo and rand:
                total += 1
                # Retain nonfinite failures in the means. They cannot create a win.
                dfo_mean = np.mean(dfo) if all(map(math.isfinite, dfo)) else float("inf")
                rand_mean = np.mean(rand) if all(map(math.isfinite, rand)) else float("inf")
                if dfo_mean < rand_mean:
                    wins += 1
        return wins, total

    def summary(self) -> str:
        optimizers = sorted({r.optimizer for r in self.results})
        problems = len({r.problem_name for r in self.results})
        lines = [f"\n{'=' * 70}", f"Suite: {self.suite_name} ({problems} problems)", "=" * 70]
        for opt in optimizers:
            opt_results = [r for r in self.results if r.optimizer == opt]
            solved = sum(1 for r in opt_results if r.solved)
            precisions = [r.precision for r in opt_results if math.isfinite(r.precision)]
            mean_prec = np.mean(precisions) if precisions else float("nan")
            mean_time = np.mean([r.wall_time_s for r in opt_results])
            lines.append(
                f"  {opt:<20} solved={solved}/{len(opt_results)}  "
                f"mean_prec={mean_prec:.2e}  mean_time={mean_time:.2f}s"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# COCO/BBOB benchmark runner
# ---------------------------------------------------------------------------


def run_bbob(
    dims: list[int] | None = None,
    instances: list[int] | None = None,
    functions: list[int] | None = None,
    budget_mult: int = 5000,
    run_pycma: bool = True,
) -> SuiteReport:
    """Run BBOB benchmark suite."""
    import cocoex

    if dims is None:
        dims = [2, 5, 10, 20]
    if instances is None:
        instances = [1, 2, 3]
    if functions is None:
        functions = list(range(1, 25))

    report = SuiteReport(suite_name="COCO/BBOB")

    for dim in dims:
        for fid in functions:
            for iid in instances:
                budget = dim * budget_mult

                # --- torch-dfo ---
                suite = cocoex.Suite(
                    "bbob",
                    f"instances: {iid}",
                    f"function_indices: {fid} dimensions: {dim}",
                )
                for p in suite:
                    result = _run_torch_dfo_on_coco(p, budget)
                    result.suite = "bbob"
                    result.problem_name = p.id
                    report.results.append(result)
                    p.free()

                # --- pycma baseline ---
                if run_pycma:
                    suite = cocoex.Suite(
                        "bbob",
                        f"instances: {iid}",
                        f"function_indices: {fid} dimensions: {dim}",
                    )
                    for p in suite:
                        result = _run_pycma_on_coco(p, budget)
                        result.suite = "bbob"
                        result.problem_name = p.id
                        report.results.append(result)
                        p.free()

    return report


def _run_torch_dfo_on_coco(problem, budget: int) -> BenchmarkResult:
    """Run PhasedDFO on a COCO problem."""
    from torch_dfo import PhasedDFO

    dim = problem.dimension
    lb = float(problem.lower_bounds[0])
    ub = float(problem.upper_bounds[0])

    def fitness_fn(x_batch: torch.Tensor) -> torch.Tensor:
        x_np = x_batch.detach().cpu().numpy()
        return torch.tensor([problem(row) for row in x_np], dtype=torch.float64)

    t0 = time.perf_counter()
    opt = PhasedDFO(
        dim=dim,
        bounds=(lb, ub),
        budget=budget,
        device="cpu",
        dtype=torch.float64,
        seed=42,
    )
    _, best_f = opt.optimize(fitness_fn)
    elapsed = time.perf_counter() - t0

    return BenchmarkResult(
        suite="",
        problem_name="",
        dim=dim,
        optimizer="torch-dfo",
        best_fitness=best_f.item(),
        precision=float("nan"),  # COCO does not expose an exact optimum here.
        fe_used=int(problem.evaluations),
        wall_time_s=elapsed,
        solved=bool(problem.final_target_hit),
    )


def _run_pycma_on_coco(problem, budget: int) -> BenchmarkResult:
    """Run population-doubling CMA-ES restarts with a strict evaluation cap."""
    import cma

    dim = problem.dimension
    lb = problem.lower_bounds[0]
    ub = problem.upper_bounds[0]
    sigma0 = (ub - lb) / 4.0
    t0 = time.perf_counter()
    best_f = float("inf")
    fe_used = 0
    base_pop = None

    for restart in range(20):
        remaining = budget - fe_used
        if remaining <= 0 or problem.final_target_hit:
            break
        opts = {
            "seed": 42 + restart,
            "maxfevals": remaining,
            "bounds": [lb, ub],
            "verbose": -9,
            "tolfun": 1e-12,
            "tolx": 1e-12,
        }
        if restart > 0:
            opts["popsize"] = min(int(base_pop * 2**restart), 512)
            x0 = lb + np.random.RandomState(42 + restart).rand(dim) * (ub - lb)
        else:
            x0 = problem.initial_solution
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        if base_pop is None:
            base_pop = es.popsize
        while not es.stop() and fe_used < budget and not problem.final_target_hit:
            solutions = es.ask()
            # A final partial generation contributes observations but cannot update CMA-ES.
            evaluated = solutions[: budget - fe_used]
            fitnesses = []
            for x in evaluated:
                value = float(problem(x))
                fe_used += 1
                best_f = min(best_f, value)
                fitnesses.append(value)
                if problem.final_target_hit:
                    break
            if len(fitnesses) != len(solutions) or problem.final_target_hit:
                break
            es.tell(solutions, fitnesses)

    return BenchmarkResult(
        suite="",
        problem_name="",
        dim=dim,
        optimizer="pycma",
        best_fitness=best_f,
        precision=float("nan"),
        fe_used=int(problem.evaluations),
        wall_time_s=time.perf_counter() - t0,
        solved=bool(problem.final_target_hit),
    )


# ---------------------------------------------------------------------------
# BBOB-noisy benchmark runner
# ---------------------------------------------------------------------------


def run_bbob_noisy(
    dims: list[int] | None = None,
    instances: list[int] | None = None,
    budget_mult: int = 5000,
) -> SuiteReport:
    """Run BBOB noisy benchmark suite (torch-dfo only, no pycma baseline)."""
    import cocoex

    if dims is None:
        dims = [10, 20]
    if instances is None:
        instances = [1]

    report = SuiteReport(suite_name="COCO/BBOB-noisy")

    for dim in dims:
        inst_str = ",".join(map(str, instances))
        suite = cocoex.Suite("bbob-noisy", f"instances: {inst_str}", f"dimensions: {dim}")
        budget = dim * budget_mult

        for p in suite:
            result = _run_torch_dfo_on_coco(p, budget)
            result.suite = "bbob-noisy"
            result.problem_name = p.id
            report.results.append(result)
            p.free()

    return report


# ---------------------------------------------------------------------------
# YAHPO HPO benchmark runner
# ---------------------------------------------------------------------------

# Primary metric per YAHPO scenario (extract by name, not position)
YAHPO_METRICS: dict[str, str] = {
    "lcbench": "val_accuracy",
    "nb301": "val_accuracy",
    "rbv2_xgboost": "acc",
    "rbv2_svm": "acc",
    "rbv2_ranger": "acc",
    "rbv2_rpart": "acc",
    "rbv2_glmnet": "acc",
    "rbv2_aknn": "acc",
    "rbv2_super": "acc",
    "iaml_glmnet": "auc",
    "iaml_ranger": "auc",
    "iaml_rpart": "auc",
    "iaml_super": "auc",
    "iaml_xgboost": "auc",
}


def _extract_yahpo_metric(result: object, scenario: str) -> float:
    """Extract primary metric from YAHPO result by name."""
    metric_name = YAHPO_METRICS.get(scenario)
    if metric_name is None:
        raise ValueError(f"Unknown YAHPO scenario: {scenario}")
    row = result[0] if isinstance(result, list) else result
    if metric_name not in row:
        raise KeyError(f"Metric '{metric_name}' not in result keys: {list(row.keys())}")
    return float(row[metric_name])


def run_yahpo(
    scenarios: list[str] | None = None,
    n_instances: int = 3,
    budgets: list[int] | None = None,
    run_random_baseline: bool = True,
    random_repeats: int = 10,
) -> SuiteReport:
    """Run YAHPO surrogate HPO benchmarks with random baseline comparison."""
    try:
        from yahpo_gym import BenchmarkSet
    except ImportError:
        print("YAHPO Gym not available, skipping", file=sys.stderr)
        return SuiteReport(suite_name="YAHPO (skipped)")

    if scenarios is None:
        scenarios = ["lcbench", "rbv2_xgboost", "rbv2_svm"]
    if budgets is None:
        budgets = [500, 1000, 2000]

    report = SuiteReport(suite_name="YAHPO HPO")

    for scenario_name in scenarios:
        try:
            bs = BenchmarkSet(scenario_name)
        except Exception as e:
            print(f"  YAHPO {scenario_name}: {e}", file=sys.stderr)
            continue

        instances = bs.instances[:n_instances]
        hp_names = [hp.name for hp in bs.config_space.get_hyperparameters()]
        dim = len(hp_names)

        for inst in instances:
            bs.set_instance(inst)

            def make_yahpo_fitness(benchmark_set, config_space, scen_name):
                def fitness_fn(x_batch: torch.Tensor) -> torch.Tensor:
                    results = []
                    for row in x_batch:
                        config = _tensor_to_yahpo_config(row, config_space)
                        try:
                            result = benchmark_set.objective_function(config)
                            val = -_extract_yahpo_metric(result, scen_name)
                        except Exception:
                            val = float("inf")
                        results.append(val)
                    return torch.tensor(results, dtype=torch.float64)

                return fitness_fn

            fitness = make_yahpo_fitness(bs, bs.config_space, scenario_name)

            for budget in budgets:
                from torch_dfo import PhasedDFO

                prob_name = f"{scenario_name}/{inst}/b{budget}"

                # --- torch-dfo ---
                t0 = time.perf_counter()
                opt = PhasedDFO(
                    dim=dim,
                    bounds=(0.0, 1.0),
                    budget=budget,
                    device="cpu",
                    dtype=torch.float64,
                    seed=42,
                )
                _, best_f = opt.optimize(fitness)
                elapsed = time.perf_counter() - t0

                report.results.append(
                    BenchmarkResult(
                        suite="yahpo",
                        problem_name=prob_name,
                        dim=dim,
                        optimizer="torch-dfo",
                        best_fitness=best_f.item(),
                        precision=float("nan"),
                        fe_used=opt._fe_count,
                        wall_time_s=elapsed,
                        solved=False,
                    )
                )

                # --- random baseline ---
                if run_random_baseline:
                    for rep in range(random_repeats):
                        t0 = time.perf_counter()
                        rng = torch.Generator().manual_seed(rep)
                        best_rand = float("inf")
                        for _ in range(budget):
                            x = torch.rand(1, dim, dtype=torch.float64, generator=rng)
                            f_val = fitness(x).item()
                            if f_val < best_rand:
                                best_rand = f_val
                        report.results.append(
                            BenchmarkResult(
                                suite="yahpo",
                                problem_name=prob_name,
                                dim=dim,
                                optimizer="random",
                                best_fitness=best_rand,
                                precision=float("nan"),
                                fe_used=budget,
                                wall_time_s=time.perf_counter() - t0,
                                solved=False,
                            )
                        )

    return report


def _tensor_to_yahpo_config(x: torch.Tensor, config_space):
    """Convert a [0,1] tensor to a YAHPO config space configuration.

    Handles conditional hyperparameters by only setting active parameters.
    Uses a seeded ConfigSpace sample as the base, then overrides active
    parameters from the tensor values.
    """
    # Use tensor values as a seed for reproducible base config
    seed = int(x.sum().item() * 1e6) & 0x7FFF_FFFF
    config_space.seed(seed)
    config = config_space.sample_configuration()

    hps = config_space.get_hyperparameters()
    for i, hp in enumerate(hps):
        if i >= x.shape[0]:
            break
        # Skip inactive (conditional) hyperparameters
        try:
            val = float(x[i].item())
            if hasattr(hp, "lower") and hasattr(hp, "upper"):
                if hasattr(hp, "log") and hp.log:
                    log_lower = math.log(max(hp.lower, 1e-30))
                    log_upper = math.log(max(hp.upper, 1e-30))
                    decoded = math.exp(log_lower + val * (log_upper - log_lower))
                else:
                    decoded = hp.lower + val * (hp.upper - hp.lower)
                if _CS_INT_HP_TYPES and isinstance(hp, _CS_INT_HP_TYPES):
                    config[hp.name] = round(decoded)
                else:
                    config[hp.name] = decoded
            elif hasattr(hp, "choices"):
                idx = min(int(val * len(hp.choices)), len(hp.choices) - 1)
                config[hp.name] = hp.choices[idx]
        except (ValueError, KeyError):
            pass  # inactive conditional — skip
    return config


# ---------------------------------------------------------------------------
# Gymnasium RL benchmark runner
# ---------------------------------------------------------------------------


def run_gymnasium(
    envs: list[str] | None = None,
    budget: int = 500,
    episodes_per_eval: int = 5,
) -> SuiteReport:
    """Run RL policy tuning benchmarks via Gymnasium."""
    try:
        import gymnasium as gym
    except ImportError:
        print("Gymnasium not available, skipping", file=sys.stderr)
        return SuiteReport(suite_name="Gymnasium (skipped)")

    if envs is None:
        envs = ["CartPole-v1", "MountainCarContinuous-v0", "Pendulum-v1"]

    report = SuiteReport(suite_name="Gymnasium RL")

    for env_name in envs:
        try:
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            if isinstance(env.action_space, gym.spaces.Discrete):
                act_dim = env.action_space.n
                continuous = False
            else:
                act_dim = env.action_space.shape[0]
                continuous = True
            env.close()

            # Linear policy: weights matrix (obs_dim × act_dim) + bias
            policy_dim = obs_dim * act_dim + act_dim

            def make_rl_fitness(env_name_inner, obs_d, act_d, cont, n_episodes):
                def fitness_fn(x_batch: torch.Tensor) -> torch.Tensor:
                    results = []
                    for row in x_batch:
                        params = row.detach().cpu().numpy()
                        W = params[: obs_d * act_d].reshape(obs_d, act_d)
                        b = params[obs_d * act_d :]

                        total_reward = 0.0
                        for _ in range(n_episodes):
                            env_inner = gym.make(env_name_inner)
                            obs, _ = env_inner.reset()
                            ep_reward = 0.0
                            for _ in range(1000):
                                action_raw = obs @ W + b
                                if cont:
                                    action = np.clip(action_raw, -1, 1)
                                else:
                                    action = int(np.argmax(action_raw))
                                obs, reward, terminated, truncated, _ = env_inner.step(action)
                                ep_reward += reward
                                if terminated or truncated:
                                    break
                            env_inner.close()
                            total_reward += ep_reward
                        results.append(-total_reward / n_episodes)  # minimize negative reward
                    return torch.tensor(results, dtype=torch.float64)

                return fitness_fn

            fitness = make_rl_fitness(env_name, obs_dim, act_dim, continuous, episodes_per_eval)

            from torch_dfo import PhasedDFO

            t0 = time.perf_counter()
            opt = PhasedDFO(
                dim=policy_dim,
                bounds=(-2.0, 2.0),
                budget=budget,
                device="cpu",
                dtype=torch.float64,
                seed=42,
            )
            _, best_f = opt.optimize(fitness)
            elapsed = time.perf_counter() - t0

            report.results.append(
                BenchmarkResult(
                    suite="gymnasium",
                    problem_name=env_name,
                    dim=policy_dim,
                    optimizer="torch-dfo",
                    best_fitness=-best_f.item(),  # report as positive reward
                    precision=float("nan"),
                    fe_used=opt._fe_count,
                    wall_time_s=elapsed,
                    solved=False,  # no absolute target for RL
                )
            )

        except Exception as e:  # noqa: PERF203
            print(f"  Gymnasium {env_name}: {e}", file=sys.stderr)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="torch-dfo comprehensive benchmark harness")
    parser.add_argument(
        "--suite",
        choices=["all", "bbob", "bbob-noisy", "yahpo", "gymnasium", "quick"],
        default="quick",
        help="Which suite(s) to run",
    )
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated dims (bbob)")
    parser.add_argument("--no-pycma", action="store_true", help="Skip pycma baseline")
    parser.add_argument("--no-random", action="store_true", help="Skip YAHPO random baseline")
    parser.add_argument(
        "--yahpo-budgets",
        type=str,
        default=None,
        help="Comma-separated budgets for YAHPO (default: 500,1000,2000)",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(",")] if args.dims else None
    reports: list[SuiteReport] = []

    start = time.time()

    if args.suite in ("all", "bbob", "quick"):
        quick_dims = [10] if args.suite == "quick" else dims
        quick_instances = [1] if args.suite == "quick" else None
        print("Running COCO/BBOB...", file=sys.stderr)
        reports.append(
            run_bbob(
                dims=quick_dims,
                instances=quick_instances,
                run_pycma=not args.no_pycma,
            )
        )

    if args.suite in ("all", "bbob-noisy"):
        print("Running COCO/BBOB-noisy...", file=sys.stderr)
        reports.append(run_bbob_noisy())

    if args.suite in ("all", "yahpo"):
        yahpo_budgets = (
            [int(b) for b in args.yahpo_budgets.split(",")] if args.yahpo_budgets else None
        )
        print("Running YAHPO HPO...", file=sys.stderr)
        reports.append(
            run_yahpo(
                budgets=yahpo_budgets,
                run_random_baseline=not args.no_random,
            )
        )

    if args.suite in ("all", "gymnasium"):
        print("Running Gymnasium RL...", file=sys.stderr)
        reports.append(run_gymnasium())

    # Print reports
    for report in reports:
        print(report.summary())

    total_time = time.time() - start
    print(f"\nTotal wall time: {total_time:.1f}s")

    # Save JSON if requested
    if args.output:
        data = []
        for report in reports:
            data.extend(
                {
                    "suite": r.suite,
                    "problem": r.problem_name,
                    "dim": r.dim,
                    "optimizer": r.optimizer,
                    "best_fitness": r.best_fitness,
                    "precision": r.precision if math.isfinite(r.precision) else None,
                    "fe_used": r.fe_used,
                    "wall_time_s": r.wall_time_s,
                    "solved": r.solved,
                }
                for r in report.results
            )
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
