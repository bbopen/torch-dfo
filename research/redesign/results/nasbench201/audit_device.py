"""Check the corrected device panel against official data and frozen sources."""

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
EXPECTED_SOURCE_SHA256 = {
    "benchmarks/nasbench201_device.py": "dda6b49243ef29e1314e0291a0eb5939c0cc312d53233e20e60ba5e0f0d6c7d9",
    "benchmarks/nasbench201.py": "df3b91df1eb4a3512c5069e3ab82e9959209f7113d4fcc93c7a8bb7b82c75131",
    "research/redesign/nasbench201-protocol.md": "c98a97a963c8fce398bdd9ee785697a3464c61e49008a72c3a99174e278068fa",
    "src/torch_dfo/cmaes.py": "ecd49f62513f23dcb3211a5dbe28dc265b00309c395c717f3707c3ef2f55e955",
    "src/torch_dfo/shade.py": "27952a9ede7b8d3b25b9484f3f388ce139d690a1926d48b731eb8a3d1a315ede",
    "src/torch_dfo/run.py": "0a0e9ad4b68e1258a7b386189409a7495be513ece38b2795e42fb0f5cd34ca0a",
}
EXPECTED_EVOX_SOURCE = {
    "de": "ea1c29db651e96ab18edc8f5ea8a21a9c73b155f71620c47e441bf36c84b98f0",
    "cmaes": "3c68f5f274af8fe9f85f7257a368a84b71800a4611c94ccae3b589bfca69cb0a",
    "workflow": "83046bffb416bb2845cc937a12b8bdfb8cc7c5cadcba4cb46684cbda6f27ac92",
}
EXPECTED_EVOX_COMMIT = "1c242cc9533fbd7e2e0c0342456f4e0f52449e5d"
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
    issues: collections.Counter[str] = collections.Counter()
    first_issues: list[dict] = []

    def check(condition: bool, kind: str, loc: dict) -> None:
        if not condition:
            issues[kind] += 1
            if len(first_issues) < 30:
                first_issues.append({"kind": kind, **loc})

    validation_hash = digest(args.validation)
    check(validation_hash == VALIDATION_SHA256, "validation_hash", {})
    validation = json.loads(args.validation.read_text())
    by_id = {row["index"]: row for row in validation["records"]}
    check(len(by_id) == 15625, "architecture_count", {})
    report = json.loads(args.report.read_text())
    config = report["config"]
    datasets = config["datasets"]
    methods = config["methods"]
    devices = config["devices"]
    seeds = config["seeds"]
    expected_records = {
        (dataset, seed, method, device)
        for dataset in datasets
        for seed in seeds
        for method in methods
        for device in devices
    }
    actual_records: collections.Counter[tuple[str, int, str, str]] = collections.Counter()
    status_by_lane: collections.Counter[tuple[str, str, str]] = collections.Counter()
    failure_patterns: collections.Counter[tuple[str, str, str, str]] = collections.Counter()
    audited_trace_rows = 0
    max_validation_error_difference = 0.0
    complete_searches = 0
    compiled_searches = 0

    check(report["validation_sha256"] == VALIDATION_SHA256, "report_validation_hash", {})
    check(
        datasets == ["cifar10", "cifar100", "ImageNet16-120"],
        "dataset_panel",
        {},
    )
    check(
        methods == ["torch_dfo_cmaes", "torch_dfo_shade", "evox_de", "evox_cmaes"],
        "method_panel",
        {},
    )
    check(devices == ["cpu", "cuda"] and seeds == list(range(5)), "device_seed_panel", {})
    check(report.get("source_sha256") == EXPECTED_SOURCE_SHA256, "frozen_source_hashes", {})
    evox_source = report.get("evox_source") or {}
    check(evox_source.get("version") == "1.4.0", "evox_version", {})
    check(evox_source.get("git_head") == EXPECTED_EVOX_COMMIT, "evox_commit", {})
    check(
        {name: item.get("sha256") for name, item in evox_source.get("files", {}).items()}
        == EXPECTED_EVOX_SOURCE,
        "evox_file_hashes",
        {},
    )
    check(
        all(config.get(name) is True for name in (
            "compiled_evox_de_attempt", "compiled_evox_fullgraph", "compiled_cache_reset_per_search"
        )),
        "compiled_protocol_flags",
        {},
    )
    check(config["budget"] == 1000 and config["population"] == 20, "panel_budget", {})
    check(config["dtype"] == "float64", "panel_dtype", {})
    check(len(report["table_loads"]) == len(datasets), "table_load_count", {})
    check(len(report["warmups"]) == len(datasets) * len(methods) * len(devices), "warmup_count", {})
    check(len(report["objective_records"]) == len(datasets) * len(seeds) * len(devices), "objective_count", {})
    check(len(report["records"]) == len(expected_records), "record_count", {})
    warmup_grid = collections.Counter(
        (row["dataset"], row["method"], row["device"]) for row in report["warmups"]
    )
    check(
        set(warmup_grid)
        == {(dataset, method, device) for dataset in datasets for method in methods for device in devices},
        "warmup_grid",
        {},
    )
    check(all(count == 1 for count in warmup_grid.values()), "warmup_duplicates", {})
    expected_compiled_warmups = len(datasets) * len(devices)
    actual_compiled_warmups = 0
    for warmup in report["warmups"]:
        loc = {key: warmup[key] for key in ("dataset", "method", "device")}
        compiled = warmup.get("compiled")
        if warmup["method"] != "evox_de":
            check(compiled is None, "unexpected_compiled_warmup", loc)
            continue
        actual_compiled_warmups += 1
        check(compiled is not None, "missing_compiled_warmup", loc)
        if compiled is None:
            continue
        check(compiled.get("status") == "complete", "compiled_warmup_status", loc)
        check(compiled.get("charged") == 1000, "compiled_warmup_charge", loc)
        check(
            compiled.get("first_compiled_unique_graphs") == 1
            and compiled.get("total_compiled_unique_graphs") == 1,
            "compiled_warmup_graphs",
            loc,
        )
        check(
            compiled.get("first_compiled_generation_includes_compile") is True,
            "compiled_warmup_first_use_label",
            loc,
        )
        check(
            isinstance(compiled.get("compiler_cache_reset_seconds"), (int, float))
            and compiled["compiler_cache_reset_seconds"] >= 0,
            "compiled_warmup_reset_time",
            loc,
        )
    check(actual_compiled_warmups == expected_compiled_warmups, "compiled_warmup_count", {})

    def inspect_trace(search: dict, loc: dict) -> None:
        nonlocal audited_trace_rows, max_validation_error_difference
        trace = search["trace"]
        keys = ("architecture_ids", "visit_indices", "trial_indices", "validation_errors", "tied_edges")
        check(all(len(trace[key]) == 1000 for key in keys), "trace_length", loc)
        seen: collections.Counter[int] = collections.Counter()
        best_position = None
        best_error = math.inf
        tie_total = 0
        for position, (arch_id, visit, trial, error, ties) in enumerate(
            zip(*(trace[key] for key in keys), strict=True)
        ):
            row_loc = {**loc, "position": position}
            source = by_id.get(arch_id)
            check(source is not None, "architecture_id", row_loc)
            if source is None:
                continue
            ops = tuple(source["ops"])
            check(visit == seen[arch_id], "visit_order", row_loc)
            seen[arch_id] += 1
            trials = source["validation"][loc["dataset"]]
            expected_trial = trial_index(loc["dataset"], loc["seed"], ops, visit, len(trials))
            check(trial == expected_trial, "trial_choice", row_loc)
            expected_error = 100.0 - trials[expected_trial]["valid-accuracy"]
            difference = abs(error - expected_error)
            max_validation_error_difference = max(max_validation_error_difference, difference)
            check(difference <= TOLERANCE, "validation_value", row_loc)
            if error < best_error:
                best_error = error
                best_position = position
            check(type(ties) is int and 0 <= ties <= 6, "tie_count", row_loc)
            tie_total += ties
            audited_trace_rows += 1
        check(search["charged"] == 1000, "search_charge", loc)
        check(search["unique_architectures"] == len(seen), "unique_count", loc)
        check(search["tied_edges"] == tie_total, "tie_total", loc)
        if best_position is not None:
            check(search["selected_architecture_id"] == trace["architecture_ids"][best_position], "selection_architecture", loc)
            check(abs(search["selected_validation_error"] - best_error) <= TOLERANCE, "selection_validation", loc)
        check(
            search["full_search_seconds"] >= search["first_query_seconds"] >= 0
            and search["steady_generations_seconds"] >= 0,
            "timing_parts",
            loc,
        )
        accounting = search.get("native_accounting")
        if loc["method"].startswith("torch_dfo_"):
            check(
                accounting is not None
                and all(accounting[name] == 1000 for name in ("scheduled", "charged", "attempted", "completed"))
                and accounting["stop_reason"] == "budget exhausted"
                and not accounting["failures"]
                and not accounting["accounting_uncertain"],
                "native_accounting",
                loc,
            )

    for row in report["objective_records"]:
        loc = {key: row[key] for key in ("dataset", "seed", "device")}
        if row["status"] == "complete":
            check(len(row["seconds"]) == 5 and all(value > 0 for value in row["seconds"]), "objective_samples", loc)
        else:
            failure_patterns[("objective_only", row["device"], row["status"], row.get("error_type", ""))] += 1

    for row in report["records"]:
        key = tuple(row[name] for name in ("dataset", "seed", "method", "device"))
        loc = dict(zip(("dataset", "seed", "method", "device"), key, strict=True))
        actual_records[key] += 1
        status_by_lane[(row["method"], row["device"], row["status"])] += 1
        search = row["search"]
        if search["status"] == "complete":
            complete_searches += 1
            inspect_trace(search, loc)
        else:
            failure_patterns[("search", row["device"], search["status"], search.get("error_type", ""))] += 1
        floor = row["optimizer_floor"]
        if floor.get("status") == "failed":
            failure_patterns[("optimizer_floor", row["device"], "failed", floor.get("error_type", ""))] += 1
        else:
            check(floor["charged"] == 1000 and floor["managed_loop_seconds"] >= 0, "optimizer_floor", loc)
        expected_lane_status = (
            "failed"
            if search["status"] == "failed"
            else "partial"
            if floor.get("status") == "failed"
            or row["warmup_status"] != "complete"
            or row["objective_status"] != "complete"
            else "complete"
        )
        check(row["status"] == expected_lane_status, "lane_status", loc)
        if row["method"] == "evox_de":
            compiled = row.get("compiled_search")
            check(compiled is not None, "missing_compiled_search", loc)
            if compiled is None:
                continue
            check(compiled.get("status") == "complete", "compiled_search_status", loc)
            check(
                compiled.get("first_compiled_unique_graphs") == 1
                and compiled.get("total_compiled_unique_graphs") == 1,
                "compiled_search_graphs",
                loc,
            )
            check(
                compiled.get("compile_first_use_included_in_search_seconds") is True,
                "compiled_search_first_use_label",
                loc,
            )
            check(
                isinstance(compiled.get("first_compiled_generation_seconds"), (int, float))
                and compiled["first_compiled_generation_seconds"] > 0,
                "compiled_search_first_use_time",
                loc,
            )
            check(
                isinstance(compiled.get("compiler_cache_reset_seconds"), (int, float))
                and compiled["compiler_cache_reset_seconds"] >= 0,
                "compiled_search_reset_time",
                loc,
            )
            if compiled.get("status") in ("complete", "partial"):
                compiled_searches += 1
                inspect_trace(compiled, {**loc, "method": "evox_de_compiled"})
            else:
                failure_patterns[("compiled_search", row["device"], compiled.get("status", "missing"), compiled.get("error_type", ""))] += 1
        else:
            check("compiled_search" not in row, "unexpected_compiled_search", loc)

    check(set(actual_records) == expected_records, "grid_coverage", {})
    check(all(count == 1 for count in actual_records.values()), "grid_duplicates", {})
    check(
        compiled_searches == len(datasets) * len(seeds) * len(devices),
        "compiled_search_count",
        {},
    )
    check(not any("test" in key.lower() for key in report), "test_key_present", {})
    independent_failures = {
        "warmups": sum(item["status"] == "failed" for item in report["warmups"]),
        "compiled_warmups": sum(item.get("compiled", {}).get("status") == "failed" for item in report["warmups"]),
        "objective_only": sum(item["status"] == "failed" for item in report["objective_records"]),
        "optimizer_floor": sum(item["optimizer_floor"].get("status") == "failed" for item in report["records"]),
        "search": sum(item["search"].get("status") == "failed" for item in report["records"]),
        "compiled_search": sum(item.get("compiled_search", {}).get("status") == "failed" for item in report["records"]),
        "partial_lanes": sum(item["status"] == "partial" for item in report["records"]),
        "partial_compiled_lanes": sum(item.get("compiled_search", {}).get("status") == "partial" for item in report["records"]),
    }
    check(report["failure_counts"] == independent_failures, "failure_counts", {})
    expected_status = "complete" if not any(independent_failures.values()) else "partial"
    check(report["status"] == expected_status, "top_status", {})

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent stdlib audit of corrected device report; no Torch import or GPU jobs",
        "sources": {
            "report_sha256": digest(args.report),
            "validation_sha256": validation_hash,
            "reported_source_sha256": report.get("source_sha256"),
            "reported_evox_git_head": evox_source.get("git_head"),
        },
        "denominators": {
            "expected_lanes": len(expected_records),
            "recorded_lanes": len(report["records"]),
            "complete_searches": complete_searches,
            "compiled_searches_audited": compiled_searches,
            "compiled_warmups_checked": actual_compiled_warmups,
            "trace_rows_audited": audited_trace_rows,
        },
        "status_by_lane": {
            f"{method}/{device}/{status}": count
            for (method, device, status), count in sorted(status_by_lane.items())
        },
        "failure_patterns": [
            {"workload": workload, "device": device, "status": status, "error_type": error_type, "runs": count}
            for (workload, device, status, error_type), count in sorted(failure_patterns.items())
        ],
        "max_validation_absolute_difference": max_validation_error_difference,
        "issue_counts": dict(sorted(issues.items())),
        "first_issues": first_issues,
        "pass": not issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = audit(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("pass", "denominators", "failure_patterns", "issue_counts")}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
