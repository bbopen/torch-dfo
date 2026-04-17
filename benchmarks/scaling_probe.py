#!/usr/bin/env python
"""Single-point scaling probe for torch-dfo on one (optimizer, dim, function) triple.

Runs a small number of warmup + timed ask/tell cycles on the requested
classical function, then prints a single JSON line to stdout describing:

    {
      "optimizer": str,
      "dim": int,
      "function": str,
      "known_optimum": float,
      "device": str,
      "dtype": str,
      "pop_size": int | null,
      "status": "ok" | "oom" | "construction_oom" | "construction_fail"
                | "too_slow" | "runtime_error",
      "peak_vram_mb": float | null,
      "median_sec_per_gen": float | null,
      "mean_sec_per_gen": float | null,
      "first_gen_sec": float | null,
      "final_loss": float | null,
      "fevals": int,
      "error": str | null
    }

Status meanings:
    ok                 : warmup + all timed gens completed.
    construction_oom   : OOM raised during optimizer construction.
    construction_fail  : non-OOM exception during construction.
    oom                : OOM during ask/tell.
    too_slow           : a single generation exceeded --max-sec-per-gen.
    runtime_error      : any other exception during ask/tell.

The caller (run_scaling_sweep.py) runs this as a subprocess so an OOM
kill on the worker does not take down the sweep driver.

Usage:
    python benchmarks/scaling_probe.py --optimizer CMAES --dim 640 \
        --function sphere --device cuda --warmup 2 --timed 5 --max-sec-per-gen 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from statistics import median

import torch

from torch_dfo import CMAES, SHADE, DLRPortfolio, NelderMead, PhasedDFO
from torch_dfo.benchmarks.classical import _BOUNDS, _FUNCTIONS

_OPTIMIZERS = ("CMAES", "SHADE", "NelderMead", "PhasedDFO", "DLRPortfolio")
_FUNCTION_NAMES = tuple(_FUNCTIONS)


def _build(
    name: str,
    dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    bounds: tuple[float, float],
):
    """Return an object exposing ask()/tell(candidates, fitness)."""
    if name == "CMAES":
        return CMAES(dim=dim, bounds=bounds, device=device, dtype=dtype, seed=seed)
    if name == "SHADE":
        return SHADE(dim=dim, bounds=bounds, device=device, dtype=dtype, seed=seed)
    if name == "NelderMead":
        return NelderMead(dim=dim, bounds=bounds, device=device, dtype=dtype, seed=seed)
    if name == "PhasedDFO":
        return PhasedDFO(dim=dim, bounds=bounds, device=device, dtype=dtype, seed=seed)
    if name == "DLRPortfolio":
        dev = torch.device(device)
        lb = torch.full((dim,), bounds[0], device=dev, dtype=dtype)
        ub = torch.full((dim,), bounds[1], device=dev, dtype=dtype)
        gen = torch.Generator(device=dev if dev.type == "cpu" else "cpu").manual_seed(seed)
        return DLRPortfolio(
            dim=dim,
            lb=lb,
            ub=ub,
            lambdas=(24, 12, 12, 12),
            sigma_fracs=(0.200, 0.043, 0.0093, 0.002),
            device=dev,
            dtype=dtype,
            rng=gen,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _pop_size(opt) -> int | None:
    if hasattr(opt, "pop_size"):
        return int(opt.pop_size)
    if hasattr(opt, "lambdas"):
        return int(sum(opt.lambdas))
    return None


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device)) / (1024**2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True, choices=_OPTIMIZERS)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument(
        "--function",
        default="sphere",
        choices=_FUNCTION_NAMES,
        help="Classical benchmark function to optimize.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-sec-per-gen",
        type=float,
        default=300.0,
        help="Abort if any single generation exceeds this wall-clock.",
    )
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    fn, known_optimum = _FUNCTIONS[args.function]
    bounds = _BOUNDS[args.function]

    result: dict = {
        "optimizer": args.optimizer,
        "dim": args.dim,
        "function": args.function,
        "known_optimum": known_optimum,
        "device": str(device),
        "dtype": args.dtype,
        "pop_size": None,
        "status": "ok",
        "peak_vram_mb": None,
        "median_sec_per_gen": None,
        "mean_sec_per_gen": None,
        "first_gen_sec": None,
        "final_loss": None,
        "fevals": 0,
        "error": None,
    }

    _reset_peak(device)

    # ---- construction -------------------------------------------------
    try:
        opt = _build(args.optimizer, args.dim, args.device, dtype, args.seed, bounds)
        _sync(device)
    except BaseException as exc:
        result["status"] = "construction_oom" if _is_oom(exc) else "construction_fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["peak_vram_mb"] = _peak_mb(device)
        print(json.dumps(result))
        return 0

    result["pop_size"] = _pop_size(opt)

    # ---- generations --------------------------------------------------
    gen_times: list[float] = []
    best_f: float | None = None
    fevals = 0
    total = args.warmup + args.timed
    try:
        for g in range(total):
            _sync(device)
            t0 = time.perf_counter()
            cand = opt.ask()
            fit = fn(cand)
            opt.tell(cand, fit)
            _sync(device)
            dt = time.perf_counter() - t0

            fevals += int(cand.shape[0])
            f_min = float(fit.min().item())
            best_f = f_min if best_f is None else min(best_f, f_min)

            if g == 0:
                result["first_gen_sec"] = dt
            if g >= args.warmup:
                gen_times.append(dt)

            if dt > args.max_sec_per_gen:
                result["status"] = "too_slow"
                result["error"] = (
                    f"gen {g} took {dt:.2f}s > --max-sec-per-gen={args.max_sec_per_gen}s"
                )
                break
    except BaseException as exc:
        result["status"] = "oom" if _is_oom(exc) else "runtime_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["status"] == "runtime_error":
            result["error"] += "\n" + traceback.format_exc(limit=3)

    if gen_times:
        result["median_sec_per_gen"] = float(median(gen_times))
        result["mean_sec_per_gen"] = float(sum(gen_times) / len(gen_times))
    result["fevals"] = fevals
    result["final_loss"] = best_f
    result["peak_vram_mb"] = _peak_mb(device)

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
