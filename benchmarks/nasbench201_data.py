"""Export separate validation and test tables from the official NASBench201 database."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
SQLITE_SHA256 = "d2dbd2eff43fdbc59dd3ec1873a92c98a1e1c88ecbeb4f0b62cff818e82921a5"
SOURCE = {
    "repository": "https://github.com/EMI-Group/evoxbench",
    "commit": "2f1ae28720a09fdf3e487bcd9de91433512fc106",
    "database_archive_url": (
        "https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing"
    ),
    "database_archive_sha256": "eb53451b37365517078bd1442ceba428e5649160394e6359e4c178eb07c8bf71",
    "sqlite_sha256": SQLITE_SHA256,
    "table": "nasbench201_nasbench201result",
    "fidelity": 200,
    "datasets": DATASETS,
    "ops_encoding": OPS,
    "trial_order": "Unchanged from each source more_info hp200 list",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export(database: Path, output: Path) -> dict:
    """Check the pinned database and retain each trial in its original order."""
    if digest(database) != SQLITE_SHA256:
        raise ValueError("The database differs from the frozen official archive")
    validation_records, test_records = [], []
    seen_indices, seen_ops = set(), set()
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            'SELECT "index", phenotype, more_info, cost200 '
            'FROM nasbench201_nasbench201result ORDER BY "index"'
        )
        for index, phenotype, more_raw, cost_raw in rows:
            arch_ops = re.findall(
                r"(none|skip_connect|nor_conv_1x1|nor_conv_3x3|avg_pool_3x3)~[012]",
                phenotype,
            )
            if len(arch_ops) != 6:
                raise ValueError(f"Malformed phenotype at index {index}")
            encoded = [OPS.index(op) for op in arch_ops]
            rebuilt, position = [], 0
            for node in range(1, 4):
                edges = "|".join(f"{OPS[encoded[position + edge]]}~{edge}" for edge in range(node))
                rebuilt.append(f"|{edges}|")
                position += node
            if phenotype != "+".join(rebuilt):
                raise ValueError(f"Noncanonical phenotype at index {index}")
            more, cost = json.loads(more_raw), json.loads(cost_raw)
            validation, final_test = {}, {}
            for dataset in DATASETS:
                key = "cifar10-valid" if dataset == "cifar10" else dataset
                validation[dataset] = [
                    {"seed": trial["seed"], "valid-accuracy": trial["valid-accuracy"]}
                    for trial in more[key]["hp200"]
                ]
                final_test[dataset] = [
                    {"seed": trial["seed"], "test-accuracy": trial["test-accuracy"]}
                    for trial in more[dataset]["hp200"]
                ]
                if not validation[dataset] or not final_test[dataset]:
                    raise ValueError(f"Empty hp200 trials at index {index}, {dataset}")
            validation_records.append(
                {
                    "index": index,
                    "phenotype": phenotype,
                    "ops": encoded,
                    "validation": validation,
                    "cost200": {
                        dataset: {key: cost[dataset][key] for key in ("params", "flops")}
                        for dataset in DATASETS
                    },
                }
            )
            test_records.append({"index": index, "test": final_test})
            seen_indices.add(index)
            seen_ops.add(tuple(encoded))
    finally:
        connection.close()
    if seen_indices != set(range(15625)) or len(seen_ops) != 15625:
        raise ValueError("NASBench201 search space is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for filename, records in (
        ("search-validation-hp200.json", validation_records),
        ("final-test-hp200.json", test_records),
    ):
        path = output / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"source": SOURCE, "records": records}, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
        result[filename] = {"records": len(records), "sha256": digest(path)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.database, args.output), indent=2))
