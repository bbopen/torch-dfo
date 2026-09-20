"""Audit NASBench201 validation feedback before running optimizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def probe(path: Path) -> dict:
    """Use validation records only. No result from this audit seeds an optimizer."""
    raw = path.read_bytes()
    records = json.loads(raw)["records"]
    by_ops = {tuple(record["ops"]): record for record in records}
    result = {
        "validation_sha256": hashlib.sha256(raw).hexdigest(),
        "test_data_opened": False,
        "datasets": {},
        "scope": "Metric audit only; exhaustive validation information is not search feedback.",
    }
    for dataset in ("cifar10", "cifar100", "ImageNet16-120"):
        errors = {
            ops: [100.0 - trial["valid-accuracy"] for trial in record["validation"][dataset]]
            for ops, record in by_ops.items()
        }
        means = {ops: statistics.mean(values) for ops, values in errors.items()}
        best_ops = min(means, key=means.__getitem__)
        floor = means[(0,) * 6]
        excellent = means[(3,) * 6]
        all_skip = means[(1,) * 6]
        repeated = errors[(3,) * 6]
        rng = random.Random(419)
        all_ops = list(errors)
        random_means = [means[rng.choice(all_ops)] for _ in range(1000)]
        passed = floor > excellent and floor > means[best_ops]
        result["datasets"][dataset] = {
            "degenerate_all_none_mean_error": floor,
            "ceiling_all_conv3x3_mean_error": excellent,
            "exhaustive_validation_minimum_mean_error": means[best_ops],
            "exhaustive_validation_minimum_ops": list(best_ops),
            "cheap_shortcut_all_skip_mean_error": all_skip,
            "repeat_exploit": {
                "architecture": [3] * 6,
                "hypothetical_observation_budget": 1000,
                "unique_architectures": 1,
                "mean_validation_error": excellent,
                "minimum_recorded_trial_error": min(repeated),
                "potential_optimistic_gap": excellent - min(repeated),
                "interpretation": (
                    "Repeated draws can lower the observed minimum without improving the network. "
                    "Charge every draw and report final held-out performance."
                ),
            },
            "null_uniform_1000": {
                "mean_error_median": statistics.median(random_means),
                "best_mean_error": min(random_means),
                "interpretation": "Reference audit of mean validation scores, not a search run.",
            },
            "floor_above_known_good_and_ceiling": passed,
        }
    result["passed"] = all(
        row["floor_above_known_good_and_ceiling"] for row in result["datasets"].values()
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = probe(args.validation_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": results["passed"], "datasets": list(results["datasets"])}))
    raise SystemExit(0 if results["passed"] else 1)
