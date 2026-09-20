#!/usr/bin/env python3
"""Reproduce the stock EvoX 1.4.0 CMA-ES workflow failure on Torch 2.13."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import traceback
from importlib import metadata
from pathlib import Path

import torch
from evox.algorithms.so.es_variants import CMAES
from evox.core import Problem
from evox.workflows import StdWorkflow

PINNED_CMA_SHA256 = "3c68f5f274af8fe9f85f7257a368a84b71800a4611c94ccae3b589bfca69cb0a"


class Sphere(Problem):
    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        return population.square().sum(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path(inspect.getsourcefile(CMAES)).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata.version("evox") != "1.4.0" or source_hash != PINNED_CMA_SHA256:
        raise RuntimeError(f"wrong EvoX source: {source} {source_hash}")

    previous_dtype = torch.get_default_dtype()
    result = {
        "evox": "1.4.0",
        "torch": torch.__version__,
        "device": args.device,
        "source": str(source),
        "source_sha256": source_hash,
        "call": "unmodified StdWorkflow(CMAES, Sphere).init_step()",
    }
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(11)
        device = torch.device(args.device)
        algorithm = CMAES(torch.zeros(16, device=device), sigma=0.6, pop_size=32, device=device)
        workflow = StdWorkflow(algorithm, Sphere(), device=device)
        workflow.init_step()
        result["outcome"] = "completed"
    except Exception as error:
        result["outcome"] = "raised"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        torch.set_default_dtype(previous_dtype)

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if result["outcome"] == "raised" and "Encountered aliasing" in result["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
