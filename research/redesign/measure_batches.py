"""Time the fixed evaluator across batch sizes, with matching CPU/CUDA inputs."""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[2] / "examples" / "06_engineering_control.py"
spec = importlib.util.spec_from_file_location("reference", path)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
evaluator = ref.FrozenEvaluator("train")
rows = []
for dtype in [torch.float32, torch.float64]:
    for n in [12, 4096]:
        x = (
            2
            * torch.rand(n, ref.HORIZON, generator=torch.Generator().manual_seed(419), dtype=dtype)
            - 1
        )
        timings = {}
        reference = None
        for device, threads in [("cpu", 1), ("cpu", 8), ("cuda", 1)]:
            torch.set_num_threads(threads)
            values = x.to(device)
            for _ in range(3):
                evaluator(values)
            if device == "cuda":
                torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                scores = evaluator(values)
                if device == "cuda":
                    torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            if reference is None:
                reference = scores.cpu()
            error = (reference - scores.cpu()).abs().max().item()
            tolerance = 1e-3 if dtype == torch.float32 else 1e-10
            assert error <= tolerance, (error, tolerance)
            timings[f"{device}-{threads}threads"] = {
                "median_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "max_abs_score_difference": error,
            }
        fastest_cpu = min(
            timings["cpu-1threads"]["median_seconds"], timings["cpu-8threads"]["median_seconds"]
        )
        rows.append(
            {
                "dtype": str(dtype),
                "population": n,
                "timings": timings,
                "fastest_tested_cpu_over_cuda": fastest_cpu
                / timings["cuda-1threads"]["median_seconds"],
            }
        )
print(
    json.dumps(
        {
            "evaluator_hash": evaluator.evaluator_hash,
            "scope": (
                "Evaluator throughput only; five timed repetitions after three warmups. "
                "Compare best of tested CPU thread counts. Not end-to-end optimization speed."
            ),
            "results": rows,
        },
        indent=2,
    )
)
