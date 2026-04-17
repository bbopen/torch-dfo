#!/usr/bin/env python
"""Driver for the torch-dfo scaling sweep.

Runs benchmarks/scaling_probe.py as a subprocess for each
(function, optimizer, dim) triple.  Running each point in its own
process isolates OOM kills, frees GPU memory between points, and
prevents one failing optimizer from polluting the measurement of
another.

For each (function, optimizer) pair we walk a dim ladder and stop that
pair's sweep as soon as we see a terminal status (oom / construction_oom
/ too_slow / construction_fail / runtime_error) — we still record that
point and then move on to the next pair.  All measurements are written
to --out.

The safety cap is enforced by scaling_probe.py via --max-sec-per-gen.
The driver adds a wall-clock subprocess timeout ~2x that to catch hangs
where synchronize() alone would not return.

Usage:
    python benchmarks/run_scaling_sweep.py \\
        --out scaling_results.json \\
        --device cuda \\
        --dtype float64 \\
        --max-sec-per-gen 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch_dfo.benchmarks.classical import _FUNCTIONS

_OPTIMIZERS = ("CMAES", "SHADE", "NelderMead", "PhasedDFO", "DLRPortfolio")
_DEFAULT_DIMS = (40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480)
_DEFAULT_FUNCTIONS = tuple(_FUNCTIONS)
_TERMINAL_STATUSES = {
    "oom",
    "construction_oom",
    "construction_fail",
    "too_slow",
    "runtime_error",
}


def _run_point(
    probe: Path,
    optimizer: str,
    dim: int,
    function: str,
    device: str,
    dtype: str,
    warmup: int,
    timed: int,
    max_sec_per_gen: float,
    timeout_sec: float,
) -> dict:
    """Run one scaling_probe.py invocation and return its parsed JSON."""
    cmd = [
        sys.executable,
        str(probe),
        "--optimizer",
        optimizer,
        "--dim",
        str(dim),
        "--function",
        function,
        "--device",
        device,
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--timed",
        str(timed),
        "--max-sec-per-gen",
        str(max_sec_per_gen),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "optimizer": optimizer,
            "dim": dim,
            "function": function,
            "device": device,
            "dtype": dtype,
            "status": "subprocess_timeout",
            "error": f"subprocess exceeded {timeout_sec}s",
            "subprocess_elapsed_sec": time.perf_counter() - t0,
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
        }
    elapsed = time.perf_counter() - t0

    # scaling_probe prints exactly one JSON line on stdout on success.
    # On segfault / OOM-kill / import error there may be no JSON.
    last_line = ""
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            last_line = s
            break

    if last_line:
        try:
            payload = json.loads(last_line)
            payload["subprocess_elapsed_sec"] = elapsed
            payload["subprocess_returncode"] = proc.returncode
            if proc.returncode != 0 and payload.get("status") == "ok":
                payload["status"] = "runtime_error"
                payload["error"] = (
                    f"non-zero returncode {proc.returncode}; stderr: {proc.stderr[-500:]}"
                )
            return payload
        except json.JSONDecodeError:
            pass

    # Probe was killed / crashed before printing JSON.
    # Treat a non-zero return with empty stdout as probable OOM kill.
    probable = "oom" if proc.returncode in (-9, 137) else "runtime_error"
    return {
        "optimizer": optimizer,
        "dim": dim,
        "function": function,
        "device": device,
        "dtype": dtype,
        "status": probable,
        "error": f"no JSON on stdout; rc={proc.returncode}",
        "subprocess_elapsed_sec": elapsed,
        "subprocess_returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=list(_DEFAULT_DIMS),
        help=f"Dim ladder. Default: {_DEFAULT_DIMS}",
    )
    ap.add_argument(
        "--optimizers",
        nargs="+",
        default=list(_OPTIMIZERS),
        choices=_OPTIMIZERS,
    )
    ap.add_argument(
        "--functions",
        nargs="+",
        default=list(_DEFAULT_FUNCTIONS),
        choices=_DEFAULT_FUNCTIONS,
        help=f"Classical benchmark functions. Default: {_DEFAULT_FUNCTIONS}",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--max-sec-per-gen", type=float, default=300.0)
    args = ap.parse_args()

    probe = Path(__file__).parent / "scaling_probe.py"
    if not probe.exists():
        print(f"ERROR: {probe} not found", file=sys.stderr)
        return 2

    # Subprocess wall-clock budget: cover (warmup+timed) gens at the cap,
    # plus construction + import overhead, with a 2x safety margin.
    gens = args.warmup + args.timed
    timeout_sec = max(120.0, 2.0 * gens * args.max_sec_per_gen + 60.0)

    meta = {
        "device": args.device,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "timed": args.timed,
        "max_sec_per_gen": args.max_sec_per_gen,
        "subprocess_timeout_sec": timeout_sec,
        "dims": args.dims,
        "optimizers": args.optimizers,
        "functions": args.functions,
        "env": {
            k: os.environ[k]
            for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")
            if k in os.environ
        },
    }

    results: list[dict] = []
    for function in args.functions:
        print(f"\n########## function={function} ##########", flush=True)
        for optimizer in args.optimizers:
            print(f"\n=== {function} / {optimizer} ===", flush=True)
            for dim in args.dims:
                print(f"  d={dim:>6d} ... ", end="", flush=True)
                r = _run_point(
                    probe=probe,
                    optimizer=optimizer,
                    dim=dim,
                    function=function,
                    device=args.device,
                    dtype=args.dtype,
                    warmup=args.warmup,
                    timed=args.timed,
                    max_sec_per_gen=args.max_sec_per_gen,
                    timeout_sec=timeout_sec,
                )
                results.append(r)

                status = r.get("status", "?")
                sec = r.get("median_sec_per_gen")
                vram = r.get("peak_vram_mb")
                sec_str = f"{sec:.3f}s/gen" if isinstance(sec, (int, float)) else "—"
                vram_str = f"{vram:.0f} MB" if isinstance(vram, (int, float)) else "—"
                print(f"{status:<18s} {sec_str:>14s}  vram={vram_str}", flush=True)

                # Persist after each point so a crash still leaves data.
                Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=2))

                if status in _TERMINAL_STATUSES or status == "subprocess_timeout":
                    print(
                        f"  -> stopping {function}/{optimizer} sweep at d={dim} ({status})",
                        flush=True,
                    )
                    break

    print(f"\nWrote {len(results)} points to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
