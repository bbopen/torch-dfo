"""Independently audit every quality-panel event using only Python stdlib."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import struct
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_SHA256 = "afe4bce58d24c83144ad12f8ff718425c59027203de945d12e1f8ca8a5b04c20"
TEST_SHA256 = "bb120e83d82c386fc986a72b33322d90b4f966f14568e83c4d09360f72973a41"
TRIAL_VERSION = b"nasbench201-trial-v1"
TOLERANCE = 1e-12


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def trial_index(dataset: str, seed: int, ops: tuple[int, ...], visit: int, count: int) -> int:
    payload = (
        TRIAL_VERSION
        + b"\0"
        + dataset.encode("ascii")
        + b"\0"
        + struct.pack("<q", seed)
        + bytes(ops)
        + struct.pack("<Q", visit)
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % count


def audit(args: argparse.Namespace) -> dict:
    issue_counts: collections.Counter[str] = collections.Counter()
    first_issues: list[dict] = []

    def check(condition: bool, category: str, location: dict) -> None:
        if not condition:
            issue_counts[category] += 1
            if len(first_issues) < 30:
                first_issues.append({"category": category, **location})

    validation_hash = digest(args.validation)
    test_hash = digest(args.final_test)
    check(validation_hash == VALIDATION_SHA256, "validation_hash", {})
    check(test_hash == TEST_SHA256, "test_hash", {})
    validation_document = json.loads(args.validation.read_text())
    test_document = json.loads(args.final_test.read_text())
    validation_by_index = {row["index"]: row for row in validation_document["records"]}
    test_by_index = {row["index"]: row for row in test_document["records"]}
    check(len(validation_by_index) == 15625, "validation_cardinality", {})
    check(len(test_by_index) == 15625, "test_cardinality", {})

    status_by_method: collections.Counter[tuple[str, str]] = collections.Counter()
    status_by_dataset: collections.Counter[tuple[str, str]] = collections.Counter()
    failure_patterns: collections.Counter[tuple[str, str, int, int]] = collections.Counter()
    observed_grid: collections.Counter[tuple[str, str, int]] = collections.Counter()
    run_facts: dict[tuple[str, str, int], dict] = {}
    trace_rows = 0
    successful = 0
    failures = 0
    max_validation_error_difference = 0.0
    max_test_mean_difference = 0.0
    line_count = 0
    header = None
    final_event = None
    completion_event = None
    expected_run_order = None
    run_position = 0
    phase = "header"

    with args.events.open(encoding="utf-8") as stream:
        for line_count, raw in enumerate(stream, 1):
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "header":
                check(phase == "header" and line_count == 1, "header_order", {"line": line_count})
                phase = "runs"
                header = event
                config = event["config"]
                methods = config["methods"]
                datasets = config["datasets"]
                seeds = config["seeds"]
                check(len(methods) == 10 and len(datasets) == 3 and seeds == list(range(30)), "grid_config", {})
                check(config["budget"] == 1000, "budget_config", {})
                check(event["sources"]["validation_export_sha256"] == VALIDATION_SHA256, "header_validation_hash", {})
                expected_run_order = [
                    (dataset, method, seed)
                    for seed in seeds
                    for dataset in datasets
                    for method in methods[seed % len(methods) :] + methods[: seed % len(methods)]
                ]
                continue

            if kind == "run":
                check(phase == "runs", "run_event_order", {"line": line_count})
                run = event["result"]
                key = (run["dataset"], run["method"], run["seed"])
                location = {"dataset": key[0], "method": key[1], "seed": key[2]}
                check(
                    expected_run_order is not None
                    and run_position < len(expected_run_order)
                    and key == expected_run_order[run_position],
                    "run_order",
                    location,
                )
                run_position += 1
                observed_grid[key] += 1
                status = run["status"]
                status_by_method[(key[1], status)] += 1
                status_by_dataset[(key[0], status)] += 1
                if status == "ok":
                    successful += 1
                else:
                    failures += 1
                    error_type = str(run.get("error", "unknown")).split("(", 1)[0]
                    failure_patterns[(key[1], error_type, run.get("attempted", -1), run.get("completed", -1))] += 1

                trace = run.get("trace", [])
                trace_rows += len(trace)
                visits: collections.Counter[tuple[int, ...]] = collections.Counter()
                best_step = None
                best_error = math.inf
                best_ops = None
                best_architecture_index = None
                tie_sum = 0
                for position, item in enumerate(trace, 1):
                    step_location = {**location, "step": position}
                    ops = tuple(item["ops"])
                    index = item["architecture_index"]
                    record = validation_by_index.get(index)
                    check(item["step"] == position, "trace_step_order", step_location)
                    check(record is not None and tuple(record["ops"]) == ops, "architecture_mapping", step_location)
                    visit = visits[ops]
                    check(item["visit_index"] == visit, "visit_order", step_location)
                    visits[ops] += 1
                    if record is not None and key[0] in record["validation"]:
                        trials = record["validation"][key[0]]
                        chosen = trial_index(key[0], key[2], ops, visit, len(trials))
                        check(item["trial_index"] == chosen, "trial_choice", step_location)
                        source_trial = trials[chosen]
                        check(item["trial_seed"] == source_trial["seed"], "trial_seed", step_location)
                        expected_error = 100.0 - source_trial["valid-accuracy"]
                        difference = abs(item["validation_error"] - expected_error)
                        max_validation_error_difference = max(max_validation_error_difference, difference)
                        check(difference <= TOLERANCE, "validation_value", step_location)
                    error = item["validation_error"]
                    if error < best_error:
                        best_error = error
                        best_step = position
                        best_ops = ops
                        best_architecture_index = index
                    check(item["incumbent_step"] == best_step, "incumbent_order", step_location)
                    ties = item["tied_edges"]
                    check(type(ties) is int and 0 <= ties <= 6, "tie_count", step_location)
                    tie_sum += ties

                check(run["completed"] == len(trace), "completed_count", location)
                check(run["attempted"] >= run["completed"], "attempted_count", location)
                check(run["budget"] == 1000, "run_budget", location)
                check(run["unique_architectures"] == len(visits), "unique_count", location)
                check(run["repeated_architectures"] == len(trace) - len(visits), "repeat_count", location)
                check(run["decoder_tied_edges"] == tie_sum, "trace_tie_total", location)
                check(run["best_step"] == best_step, "best_step", location)
                check(run["best_ops"] == (list(best_ops) if best_ops else None), "best_ops", location)
                check(run["best_architecture_index"] == best_architecture_index, "best_architecture", location)
                if best_step is None:
                    check(run["best_validation_error"] is None, "best_error_empty", location)
                else:
                    check(abs(run["best_validation_error"] - best_error) <= TOLERANCE, "best_validation_error", location)
                if status == "ok":
                    check(run["attempted"] == 1000 and len(trace) == 1000, "successful_budget", location)
                    if key[1].startswith("torch_dfo_"):
                        accounting = run["native_accounting"]
                        check(
                            all(accounting[field] == 1000 for field in ("scheduled", "charged", "attempted", "completed"))
                            and accounting["stop_reason"] == "budget exhausted"
                            and not accounting["failures"]
                            and not accounting["accounting_uncertain"],
                            "native_accounting",
                            location,
                        )
                run_facts[key] = {
                    "status": status,
                    "best_step": best_step,
                    "best_ops": best_ops,
                    "best_architecture_index": best_architecture_index,
                    "best_validation_error": best_error if best_step is not None else None,
                }
                continue

            if kind == "final_test":
                check(phase == "runs" and run_position == 900, "final_event_order", {"line": line_count})
                phase = "final"
                final_event = event
                check(event["test_export_sha256"] == TEST_SHA256, "final_test_hash", {})
                continue

            if kind == "complete":
                check(phase == "final", "completion_order", {"line": line_count})
                phase = "complete"
                completion_event = event
                continue

            check(False, "unknown_event", {"line": line_count})

    check(line_count == 903 and phase == "complete", "event_count", {})
    check(expected_run_order is not None and run_position == len(expected_run_order), "run_count", {})
    if expected_run_order is not None:
        check(set(observed_grid) == set(expected_run_order), "grid_coverage", {})
    check(all(count == 1 for count in observed_grid.values()), "grid_duplicates", {})

    selected_count = 0
    if final_event is not None:
        selected = final_event["selected"]
        expected_selected = [key for key in expected_run_order if run_facts[key]["status"] == "ok"]
        check(len(selected) == len(expected_selected) == successful, "selected_count", {})
        for position, item in enumerate(selected):
            key = (item["dataset"], item["method"], item["seed"])
            location = {"dataset": key[0], "method": key[1], "seed": key[2]}
            check(position < len(expected_selected) and key == expected_selected[position], "selected_order", location)
            facts = run_facts.get(key)
            if facts is None:
                check(False, "selected_unknown_run", location)
                continue
            check(facts["status"] == "ok", "selected_failed_run", location)
            check(item["selected_architecture_index"] == facts["best_architecture_index"], "selected_architecture", location)
            check(item["selected_ops"] == list(facts["best_ops"]), "selected_ops", location)
            check(abs(item["selected_validation_error"] - facts["best_validation_error"]) <= TOLERANCE, "selected_validation", location)
            test_row = test_by_index.get(facts["best_architecture_index"])
            if test_row is None:
                check(False, "selected_test_missing", location)
                continue
            values = [trial["test-accuracy"] for trial in test_row["test"][key[0]]]
            expected_test_error = sum(100.0 - value for value in values) / len(values)
            difference = abs(item["selected_test_error_mean"] - expected_test_error)
            max_test_mean_difference = max(max_test_mean_difference, difference)
            check(difference <= TOLERANCE, "selected_test_mean", location)
            selected_count += 1

    summary_document = json.loads(args.summary.read_text())
    check(summary_document["expected_runs"] == 900, "summary_expected", {})
    check(summary_document["recorded_runs"] == run_position, "summary_recorded", {})
    check(summary_document["failed_runs"] == failures, "summary_failures", {})
    check(len(summary_document["selected_test"]) == selected_count, "summary_selected", {})
    check(
        summary_document["status"] == ("complete" if failures == 0 else "partial"),
        "summary_status",
        {},
    )
    if completion_event is not None:
        check(completion_event["failed_runs"] == failures, "completion_failures", {})

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent stdlib audit of raw quality JSONL; no optimization or Torch import",
        "sources": {
            "events_sha256": digest(args.events),
            "validation_sha256": validation_hash,
            "final_test_sha256": test_hash,
            "summary_sha256": digest(args.summary),
        },
        "denominators": {
            "jsonl_lines": line_count,
            "grid_cells": len(expected_run_order) if expected_run_order else 0,
            "run_events": run_position,
            "successful_runs": successful,
            "failed_runs": failures,
            "trace_rows": trace_rows,
            "selected_test_rows": selected_count,
        },
        "status_by_method": {f"{method}/{status}": count for (method, status), count in sorted(status_by_method.items())},
        "status_by_dataset": {f"{dataset}/{status}": count for (dataset, status), count in sorted(status_by_dataset.items())},
        "failure_patterns": [
            {"method": method, "error_type": error_type, "attempted": attempted, "completed": completed, "runs": count}
            for (method, error_type, attempted, completed), count in sorted(failure_patterns.items())
        ],
        "max_absolute_differences": {
            "validation": max_validation_error_difference,
            "selected_test_mean": max_test_mean_difference,
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "first_issues": first_issues,
        "pass": not issue_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--final-test", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = audit(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("pass", "denominators", "failure_patterns", "issue_counts")}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
