"""Compare the same search through direct and managed execution."""

import argparse
import json
import statistics
import time

import torch

from torch_dfo import CMAES, SearchRun

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
torch.set_num_threads(1)


def sync():
    if args.device == "cuda":
        torch.cuda.synchronize()


def execute(managed, seed, expensive):
    optimizer = CMAES(16, 1.0, pop_size=12, seed=seed, device=args.device)
    calls = 0
    matrix = torch.arange(16 * 256, device=args.device, dtype=torch.float64).reshape(16, 256).sin()

    def objective(x):
        nonlocal calls
        calls += len(x)
        if expensive:
            y = x @ matrix
            for _ in range(20):
                y = y.sin() + y * 0.25
            return y.square().mean(-1)
        return x.square().sum(-1)

    sync()
    start = time.perf_counter()
    if managed:
        run = SearchRun(optimizer, 240)
        while not run.done:
            run.step(objective)
        result = run.result
        best = result.best_value.item()
        charged = result.charged_evals
    else:
        for _ in range(20):
            x = optimizer.ask()
            optimizer.tell(x, objective(x))
        best = optimizer.best()[1].item()
        charged = calls
    sync()
    return {
        "seconds": time.perf_counter() - start,
        "calls": calls,
        "charged": charged,
        "best": best,
    }


rows = []
for expensive in [False, True]:
    execute(False, 0, expensive)
    execute(True, 0, expensive)
    pairs = []
    for seed in [11, 12, 13, 14, 15]:
        order = [False, True] if seed % 2 else [True, False]
        runs = {str(managed): execute(managed, seed, expensive) for managed in order}
        direct, managed = runs["False"], runs["True"]
        assert direct["calls"] == managed["calls"] == managed["charged"] == 240
        assert direct["best"] == managed["best"]
        pairs.append(
            {
                "seed": seed,
                "direct": direct,
                "managed": managed,
                "ratio": managed["seconds"] / direct["seconds"],
            }
        )
    rows.append(
        {
            "objective": "synthetic_compute" if expensive else "sphere",
            "pairs": pairs,
            "median_ratio": statistics.median(x["ratio"] for x in pairs),
        }
    )
print(
    json.dumps(
        {
            "device": args.device,
            "torch": torch.__version__,
            "measurements": rows,
            "scope": "Five within-process pairs; synthetic workloads, no general speed claim.",
        },
        indent=2,
    )
)
