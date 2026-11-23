# Contributing to torch-dfo

`torch-dfo` is a PyTorch-native, MIT-licensed derivative-free optimization library.

## Local setup

```bash
git clone https://github.com/bbopen/torch-dfo.git
cd torch-dfo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

## Required checks

Run these before opening a PR:

```bash
ruff format --check src/ tests/ examples/ benchmarks/calibration.py benchmarks/cuda_control.py benchmarks/cma_comparison.py
ruff check src/ tests/ examples/ benchmarks/calibration.py benchmarks/cuda_control.py benchmarks/cma_comparison.py
mypy src/
pytest tests/ -q
```

When touching optimizer math, budget accounting, or numerical boundaries, also run the Lean 4 and Z3 suite:

```bash
bash formal/verify.sh
```

## Repository rules

- **Pure PyTorch runtime.** No NumPy, SciPy, Pandas, JAX, or similar imports in `src/torch_dfo/` (enforced by `tests/test_no_cpu_binding_imports.py`).
- **No inline comments in library code.** Non-directive `#` comments are prohibited in `src/torch_dfo/` (enforced by `tests/test_no_inline_comments.py`). Use clear identifiers, small helpers, and docstrings for public contracts; keep inline comments limited to `# type: ignore`, `# noqa`, and `# pragma:`.
- **Ask/tell and budget contracts.** Optimizers expose `ask()`, `tell(candidates, fitness)`, `next_batch_size()`, and `state_dict()`/`load_state_dict()`.
- **Tests before merge.** Every bug fix or behavior change requires a regression test. Shared numeric tolerances and ceilings live in [tests/_thresholds.py](tests/_thresholds.py).
- **Minimal durable docs.** Keep documentation concise and tied to implemented behavior.

## Opening a PR

1. Branch off `master` with focused commits.
2. Summarize what changed, why, and how it was verified.
3. Include paired before-and-after benchmark numbers when changing optimizer math or performance paths.
4. Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## Reporting issues

Include `torch_dfo.__version__`, PyTorch version, device (CPU, CUDA, or MPS), and a minimal reproducer.
