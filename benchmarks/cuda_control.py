"""Measure prepared CUDA objectives through the unchanged public search API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

import torch_dfo
from torch_dfo import CMAES, SearchRun


def reference_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "06_engineering_control.py"
    spec = importlib.util.spec_from_file_location("cuda_control_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def profile_call(call):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        call()
        torch.cuda.synchronize()
    events = profile.key_averages()
    kernels = [
        event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    return {
        "cuda_event_count": len(kernels),
        "host_scalar_reads": sum(e.count for e in events if e.key == "aten::_local_scalar_dense"),
        "top_cpu_events": [
            {"name": e.key, "count": e.count, "self_cpu_us": e.self_cpu_time_total}
            for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:10]
        ],
    }


def search(evaluator, reference, population, dtype, seed):
    optimizer = CMAES(16, 1.0, pop_size=population, device="cuda", dtype=dtype, seed=seed)
    run = SearchRun(optimizer, population * 20)

    def execute():
        while not run.done:
            run.step(evaluator)
        return run.result

    result, seconds = timed(execute)
    assert (
        result.charged_evals == result.completed_evals == result.attempted_evals == population * 20
    )
    assert result.stop_reason == "budget exhausted"
    assert all(record.error is None for record in result.raw_results)
    assert not result.failures and not result.accounting_uncertain
    assert result.best_x is not None and result.best_value is not None
    score = float(result.best_value.item())
    selected = result.best_x
    audit_score = float(reference(selected.reshape(1, -1)).item())
    return {
        "seconds": seconds,
        "charged": result.charged_evals,
        "attempted": result.attempted_evals,
        "completed": result.completed_evals,
        "stop_reason": result.stop_reason,
        "failures": result.failures,
        "accounting_uncertain": result.accounting_uncertain,
        "best": score,
        "reference_best": audit_score,
        "selected": selected.cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    root = Path(__file__).resolve().parents[1]
    assert Path(torch_dfo.__file__).resolve().is_relative_to(root / "src")
    reference = reference_module()
    frozen = reference.FrozenEvaluator("train")
    report = {
        "status": "running",
        "accepted": False,
        "scope": "Warm CUDA evaluator and managed CMA-ES execution; construction excluded.",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch_dfo": torch_dfo.__file__,
        },
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), root / "examples/06_engineering_control.py"]
        },
        "evaluator_hash": frozen.evaluator_hash,
        "probes": reference.run_evaluator_probes(device="cuda"),
        "configurations": [],
    }

    baseline_path = root / "research/redesign/results/engineering-cuda.json"
    baseline = json.loads(baseline_path.read_text())
    assert frozen.evaluator_hash == baseline["evaluator_hashes"]["train"]
    for name, values in report["probes"].items():
        for key, value in values.items():
            previous = baseline["probes_train_only"][name][key]
            assert abs(float(value) - float(previous)) <= 1e-10, (name, key, value, previous)
    report["historical_probe_parity"] = True

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for dtype in (torch.float32, torch.float64):
        for population in (12, 4096):
            tolerance = 1e-3 if dtype == torch.float32 else 1e-10
            x = (
                2
                * torch.rand(
                    population,
                    16,
                    device="cuda",
                    dtype=dtype,
                    generator=torch.Generator(device="cuda").manual_seed(419),
                )
                - 1
            )
            evaluators = {"reference": frozen}
            preparation = {"reference": 0.0}
            for name, compile_enabled in (("prepared", False), ("compiled", True)):
                evaluators[name], preparation[name] = timed(
                    lambda enabled=compile_enabled, dtype=dtype: frozen.prepare(
                        device="cuda", dtype=dtype, compile=enabled
                    )
                )
            first_calls = {}
            expected = None
            for name, evaluator in evaluators.items():
                scores, first_calls[name] = timed(lambda ev=evaluator, x=x: ev(x))
                if expected is None:
                    expected = scores.clone()
                else:
                    torch.testing.assert_close(scores, expected, atol=tolerance, rtol=0)
                for _ in range(5):
                    evaluator(x)
                torch.cuda.synchronize()
            timings = {name: [] for name in evaluators}
            names = list(evaluators)
            for trial in range(15):
                for name in names[trial % 3 :] + names[: trial % 3]:
                    _, seconds = timed(lambda ev=evaluators[name], x=x: ev(x))
                    timings[name].append(seconds)
            profiles = {}
            for name, evaluator in evaluators.items():
                optimizer = CMAES(16, 1.0, pop_size=population, device="cuda", dtype=dtype, seed=0)
                run = SearchRun(optimizer, population * 2)
                run.step(evaluator)
                profiles[name] = {
                    "evaluator": profile_call(lambda ev=evaluator, x=x: ev(x)),
                    "managed_generation": profile_call(lambda ev=evaluator, run=run: run.step(ev)),
                }
            pairs = []
            for seed in range(11, 16):
                order = names[seed % 3 :] + names[: seed % 3]
                runs = {
                    name: search(evaluators[name], frozen, population, dtype, seed)
                    for name in order
                }
                reference_score = runs["reference"]["reference_best"]
                for run in runs.values():
                    run["reference_error"] = abs(run["best"] - run["reference_best"])
                    run["paired_score_error"] = abs(run["reference_best"] - reference_score)
                pairs.append({"seed": seed, "runs": runs})
            ratios = [
                p["runs"]["reference"]["seconds"] / p["runs"]["compiled"]["seconds"] for p in pairs
            ]
            medians = {
                name: statistics.median(p["runs"][name]["seconds"] for p in pairs) for name in names
            }
            saving = medians["reference"] - medians["compiled"]
            setup_cost = preparation["compiled"] + first_calls["compiled"]
            correct = all(
                run["reference_error"] <= tolerance and run["paired_score_error"] <= tolerance
                for pair in pairs
                for run in pair["runs"].values()
            )
            row = {
                "dtype": str(dtype),
                "population": population,
                "budget": population * 20,
                "tolerance": tolerance,
                "preparation_seconds": preparation,
                "first_call_seconds": first_calls,
                "evaluator_samples_seconds": timings,
                "evaluator_median_seconds": {
                    name: statistics.median(values) for name, values in timings.items()
                },
                "profiles": profiles,
                "pairs": pairs,
                "search_median_seconds": medians,
                "reference_over_compiled_ratios": ratios,
                "median_paired_speedup": statistics.median(ratios),
                "ratio_of_search_medians": medians["reference"] / medians["compiled"],
                "conservative_searches_to_repay_setup": math.ceil(setup_cost / saving)
                if saving > 0
                else None,
                "correct": correct,
                "accepted": correct
                and medians["reference"] / medians["compiled"] >= 1.10
                and min(ratios) > 1.0,
            }
            report["configurations"].append(row)
            save()
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "dtype",
                            "population",
                            "median_paired_speedup",
                            "correct",
                            "accepted",
                        )
                    }
                ),
                flush=True,
            )
    report["status"] = "complete"
    report["accepted"] = all(row["correct"] for row in report["configurations"]) and any(
        row["accepted"] for row in report["configurations"]
    )
    save()
    if not all(row["correct"] for row in report["configurations"]):
        raise RuntimeError("CUDA objective or paired search failed numerical parity")


if __name__ == "__main__":
    main()
