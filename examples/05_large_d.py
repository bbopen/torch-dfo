"""
High-dimensional optimization
==============================

torch-dfo targets problems where CPU-based libraries stall.  This script
benchmarks wall-clock time per generation at increasing dimensions so you
can profile your own hardware.

``DLRPortfolio`` uses diagonal-plus-low-rank covariance with no
eigendecomposition, making it the best choice for d ≥ 200 on GPU.
``SHADE`` also scales well because the DE mutation kernel is O(d · pop).

Practical GPU ceilings on an NVIDIA RTX A4500 (20 GB VRAM) will be added
here after the scaling sweep completes (see GitHub Actions workflow
``scaling-probe.yml``).
"""

import time

import torch

import torch_dfo
from torch_dfo.dlr_cma import DLRPortfolio


def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1)


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # float64 is ~8× slower on consumer GPUs

print(f"Device: {device}  dtype: {dtype}\n")
print(f"{'Optimizer':<16} {'dim':>6} {'pop':>6} {'sec/gen':>10}")
print("-" * 44)

configs = [
    ("SHADE", 40, 60),
    ("SHADE", 200, 80),
    ("SHADE", 500, 100),
    ("DLRPortfolio", 40, 60),
    ("DLRPortfolio", 200, 60),
    ("DLRPortfolio", 500, 60),
]

WARMUP = 2
TIMED = 5

for name, dim, pop in configs:
    try:
        if name == "SHADE":
            opt = torch_dfo.SHADE(
                dim=dim,
                bounds=5.0,
                pop_size=pop,
                device=device,
                dtype=dtype,
                seed=42,
            )
        else:
            lb = torch.full((dim,), -5.0, dtype=dtype, device=device)
            ub = torch.full((dim,), 5.0, dtype=dtype, device=device)
            rng = torch.Generator(device="cpu").manual_seed(42)
            opt = DLRPortfolio(
                dim=dim,
                lb=lb,
                ub=ub,
                lambdas=(pop // 4,) * 4,
                sigma_fracs=(0.200, 0.043, 0.0093, 0.002),
                device=torch.device(device),
                dtype=dtype,
                rng=rng,
            )

        # Warmup
        for _ in range(WARMUP):
            x = opt.ask()
            opt.tell(x, sphere(x))

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(TIMED):
            x = opt.ask()
            opt.tell(x, sphere(x))

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - t0) / TIMED
        print(f"{name:<16} {dim:>6} {pop:>6} {elapsed:>10.4f}s")

    except torch.cuda.OutOfMemoryError:  # noqa: PERF203
        print(f"{name:<16} {dim:>6} {pop:>6}  {'OOM':>10}")
