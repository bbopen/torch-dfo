# Contributing to torch-dfo

torch-dfo is GPU-first, PyTorch-native, and MIT-licensed. Contributions
that keep those three properties intact are welcome.

## Quick start

```
git clone https://github.com/bbopen/torch-dfo.git
cd torch-dfo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

CI runs on Linux with CPU-only PyTorch across Python 3.10–3.12, plus a
self-hosted CUDA runner (RTX A4500) that exercises the same suite on GPU.
The CUDA lane is non-blocking — if the pod is offline, the CPU lane remains
the authoritative merge gate. For local Apple Silicon coverage,
`tests/test_mps_smoke.py` runs every public optimizer on MPS
(`PhasedDFO`, `SHADE`, `CMAES`, `NelderMead`, `DLRPortfolio`, `DFOOptimizer`).

## Expectations

- **PyTorch-native.** No NumPy round-trips in the hot path. If a feature
  needs NumPy, that should live in an optional extra, not in the core
  optimizers.
- **Ask/tell loop.** Optimizers expose `ask() -> Tensor`, `tell(candidates,
  fitness)`, and `optimize(fn)`. New optimizers should follow this shape.
- **Tests before merge.** Every behavior change needs a test. Run
  `pytest tests/ -q` and `ruff check src/ tests/` before opening a PR.
- **No regressions on the classical suite.** The 22-problem suite defined
  in `src/torch_dfo/benchmarks/classical.py` is the baseline — if your
  change might touch it, run `benchmarks/run_benchmarks.py` and report.
- **Gated mechanisms** specific to particular dim ranges (e.g. the
  `dim >= HIGH_DIM_VALLEY_ENTRY_DIM` branches in `phased.py`) should stay
  gated. Do not expand their scope without evidence.

## Test thresholds

Every numeric threshold used in a test assertion lives in
[`tests/_thresholds.py`](tests/_thresholds.py), organized by category
(tolerances, per-problem convergence ceilings, smoke ceilings, budgets,
population sizes). If you need a new threshold:

1. Check `tests/_thresholds.py` first — there's a good chance a constant
   already fits your case.
2. If you need a new one, add a descriptively named constant under the
   matching section. Prefer `CATEGORY_FAMILY_PROBLEM` form (e.g.
   `CONV_SPHERE_10D_TIGHT`).
3. Never inline a numeric threshold in a new test.

## Known limitations

- `PhasedDFO.state_dict()` inherits from `BaseOptimizer` and captures only the common state (population, fitness, best, RNG, generation counter). It does not serialize the nested SHADE and CMA-ES sub-optimizers, phase-transition state, budget trackers, or the CMA covariance path. The failing case is pinned by `tests/test_serialization.py::test_phased_roundtrip`.
- `load_state_dict` across device classes (CPU → CUDA) raises `RuntimeError: RNG state is wrong size` because CUDA's Generator state uses a different binary layout. CPU → MPS works (MPS Generator is CPU-backed). Pinned by `tests/test_serialization.py::test_shade_roundtrip_cpu_to_cuda_xfail`.

## Opening a PR

1. Branch off `master`. Include a one-paragraph description of what
   changed and why.
2. Keep commits focused. One mechanism per commit where possible.
3. CI must pass — lint, tests on Python 3.10–3.12, and the build step.
4. If your change moves numbers on benchmarks, include before/after in
   the PR body.

## Reporting issues

Please include:
- torch-dfo version (`python -c "import torch_dfo; print(torch_dfo.__version__)"`)
- PyTorch version and device (CPU / CUDA / MPS)
- Minimal reproducer

