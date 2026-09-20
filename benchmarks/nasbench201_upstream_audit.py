"""Audit EvoXBench NASBench201 exports against its unnormalized evaluator."""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

root = os.path.abspath(sys.argv[1])
os.environ["EVOXBENCH_MODEL"] = os.path.join(root, "data20240229", "data")

from evoxbench.database.init import init  # noqa: E402

init(os.path.join(root, "database"))

from evoxbench.benchmarks.nb201 import NASBench201Evaluator  # noqa: E402

with open(os.path.join(root, "search-validation-hp200.json"), encoding="utf-8") as source_file:
    validation = json.load(source_file)
with open(os.path.join(root, "final-test-hp200.json"), encoding="utf-8") as source_file:
    final_test = json.load(source_file)

validation_records = validation["records"]
test_records = final_test["records"]
assert len(validation_records) == len(test_records) == 15625
assert (
    [entry["index"] for entry in validation_records]
    == [entry["index"] for entry in test_records]
    == list(range(15625))
)

original_choice = np.random.choice
results = {}
batch_size = 256
tolerance = 1e-12

try:
    for dataset in ("cifar10", "cifar100", "ImageNet16-120"):
        evaluator = NASBench201Evaluator(fidelity=200, objs="err", dataset=dataset)
        dataset_result = {"validation": {}, "test_mean": {}}
        for selected_trial in range(3):
            # The source evaluator calls choice once per architecture. This
            # replacement selects a specified source trial without changing it.
            np.random.choice = lambda values, trial=selected_trial: values[
                min(trial, len(values) - 1)
            ]
            compared = 0
            mismatches = 0
            max_absolute_error = 0.0
            for start in range(0, len(validation_records), batch_size):
                batch = validation_records[start : start + batch_size]
                source_stats = evaluator.evaluate(
                    [entry["phenotype"] for entry in batch], true_eval=False
                )
                assert len(source_stats) == len(batch)
                for entry, stats in zip(batch, source_stats, strict=True):
                    source_trial = entry["validation"][dataset][
                        min(selected_trial, len(entry["validation"][dataset]) - 1)
                    ]
                    expected_error = 100 - source_trial["valid-accuracy"]
                    difference = float(abs(stats["err"] - expected_error))
                    max_absolute_error = max(max_absolute_error, difference)
                    mismatches += int(difference > tolerance)
                    compared += 1
            dataset_result["validation"][f"trial_{selected_trial}"] = {
                "compared": compared,
                "mismatches": mismatches,
                "max_absolute_error": max_absolute_error,
            }

        np.random.choice = original_choice
        compared = 0
        mismatches = 0
        max_absolute_error = 0.0
        for start in range(0, len(validation_records), batch_size):
            batch = validation_records[start : start + batch_size]
            source_stats = evaluator.evaluate(
                [entry["phenotype"] for entry in batch], true_eval=True
            )
            assert len(source_stats) == len(batch)
            for entry, test_entry, stats in zip(
                batch, test_records[start : start + batch_size], source_stats, strict=True
            ):
                assert entry["index"] == test_entry["index"]
                trial_values = [trial["test-accuracy"] for trial in test_entry["test"][dataset]]
                expected_error = 100 - sum(trial_values) / len(trial_values)
                difference = float(abs(stats["err"] - expected_error))
                max_absolute_error = max(max_absolute_error, difference)
                mismatches += int(difference > tolerance)
                compared += 1
        dataset_result["test_mean"] = {
            "compared": compared,
            "mismatches": mismatches,
            "max_absolute_error": max_absolute_error,
        }

        # These probes verify the source's mapping for repeated and missing
        # phenotypes. They do not select an architecture by performance.
        first = validation_records[0]
        np.random.choice = lambda values: values[0]
        duplicate = evaluator.evaluate([first["phenotype"], first["phenotype"]])
        dataset_result["duplicate_phenotype_equal"] = duplicate[0] == duplicate[1]
        try:
            evaluator.evaluate(["|invalid~0|"])
        except KeyError:
            dataset_result["invalid_phenotype_rejected"] = True
        else:
            dataset_result["invalid_phenotype_rejected"] = False
        np.random.choice = original_choice
        results[dataset] = dataset_result
finally:
    np.random.choice = original_choice

audit = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": validation["source"]["commit"],
    "sqlite_sha256": validation["source"]["sqlite_sha256"],
    "validation_export": "search-validation-hp200.json",
    "final_test_export": "final-test-hp200.json",
    "evaluator": "NASBench201Evaluator(fidelity=200, objs='err')",
    "normalization": False,
    "batch_size": batch_size,
    "absolute_error_tolerance": tolerance,
    "results": results,
}
path = os.path.join(root, "upstream-evaluator-parity-audit.json")
with open(path + ".tmp", "w", encoding="utf-8") as output:
    json.dump(audit, output, indent=2, sort_keys=True)
os.replace(path + ".tmp", path)
print(json.dumps(audit, indent=2, sort_keys=True))
