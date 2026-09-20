"""Summarize saved outcomes without changing the frozen comparison."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quality", type=Path, required=True)
parser.add_argument("--device", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
quality = json.loads(args.quality.read_text())
groups = defaultdict(list)
observations = {(r["dataset"], r["method"], r["seed"]): r for r in quality["runs"]}
for dataset, method, _seed in observations:
    groups[dataset, method] = []
by_seed = {}
for row in quality["selected_test"]:
    key = row["dataset"], row["method"]
    groups[key].append(row)
    full_key = (*key, row["seed"])
    by_seed[full_key] = row["selected_test_error_mean"]


def distribution(values):
    if not values:
        return None
    quartiles = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else [values[0]] * 3
    )
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "min": min(values),
        "max": max(values),
    }


rows = []
for (dataset, method), values in sorted(groups.items()):
    scores = [r["selected_test_error_mean"] for r in values]
    runs = [observations[dataset, method, r["seed"]] for r in values]
    attempts = [r for r in quality["runs"] if r["dataset"] == dataset and r["method"] == method]
    row = {
        "dataset": dataset,
        "method": method,
        "attempted_runs": len(attempts),
        "successful_runs": len(runs),
        "failed_runs": sum(r["status"] != "ok" for r in attempts),
        "attempted_observations": sum(r["attempted"] for r in attempts),
        "test_error": distribution(scores),
        "unique_architectures": distribution([r["unique_architectures"] for r in runs]),
        "decoder_tied_edges": distribution([r["decoder_tied_edges"] for r in runs]),
        "comparisons": {},
    }
    for baseline in ("random_cpu", "aging_cpu"):
        if method == baseline or not values:
            continue
        differences = [
            r["selected_test_error_mean"] - by_seed[dataset, baseline, r["seed"]] for r in values
        ]
        rng = random.Random(419)
        bootstrap = sorted(
            statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000)
        )
        row["comparisons"][baseline] = {
            "mean_difference": statistics.mean(differences),
            "bootstrap_95_pointwise": [bootstrap[249], bootstrap[9749]],
            "lower_error_seeds": sum(v < 0 for v in differences),
            "equal_error_seeds": sum(v == 0 for v in differences),
            "higher_error_seeds": sum(v > 0 for v in differences),
        }
    rows.append(row)
result = {
    "quality": rows,
    "quality_status": quality["status"],
    "quality_recorded_runs": quality["recorded_runs"],
    "quality_failed_runs": quality["failed_runs"],
    "uncertainty": "Paired-seed bootstrap of mean test-error differences, 10000 draws, seed419. "
    "Pointwise intervals, without multiple-comparison correction; descriptive only.",
}
path = args.device
if path is not None:
    device = json.loads(path.read_text())
    result["device_status"] = device["status"]
    buckets = defaultdict(list)
    for record in device["records"]:
        buckets[record["dataset"], record["method"], record["device"]].append(record)
    speed = []
    for (dataset, method, dev), values in sorted(buckets.items()):
        successful = [r for r in values if r["status"] == "complete"]
        item = {
            "dataset": dataset,
            "method": method,
            "device": dev,
            "successful": len(successful),
            "failed": len(values) - len(successful),
        }
        if successful:
            item["full_search_seconds"] = distribution(
                [r["search"]["full_search_seconds"] for r in successful]
            )
            item["optimizer_floor_seconds"] = distribution(
                [r["optimizer_floor"]["managed_loop_seconds"] for r in successful]
            )
        compiled = [
            r["compiled_search"]
            for r in values
            if r.get("compiled_search", {}).get("status") == "complete"
        ]
        if compiled:
            item["compiled_steady_48_generations_seconds"] = distribution(
                [r["steady_generations_seconds"] for r in compiled]
            )
            item["compiled_first_generation_seconds"] = distribution(
                [r["first_compiled_generation_seconds"] for r in compiled]
            )
        speed.append(item)
    result["device"] = speed
    result["objective"] = device["objective_records"]
args.output.write_text(json.dumps(result, indent=2) + "\n")
print("Saved analysis.json for", len(rows), "quality groups")
