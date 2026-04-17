# Changelog

All notable changes to torch-dfo.

## 0.9.0 — 2026-04-17

First public release.

The 0.9 series is feature-complete and interface-stable, but has not
yet been validated on real-world production workloads. Interfaces may
still move before 1.0.

### Optimizers

- `PhasedDFO` — 3-phase pipeline: SHADE-DE, IPOP-CMA-ES, polish
  (directional + coordinate + FD-BFGS).
- `CMAES` — Hansen 2016 reference CMA-ES with active covariance update,
  mirrored sampling, IPOP restarts.
- `SHADE` — success-history-based adaptive DE (Tanabe & Fukunaga 2014)
  with Levy-flight diversity.
- `NelderMead` — classical simplex (Nelder & Mead 1965).
- `DLRPortfolio` — GPU-native K-branch diagonal + low-rank CMA, no
  eigendecomposition.

### API

- `DFOOptimizer` — `torch.optim.Optimizer` wrapper. Accepts `closure`
  (sequential) or `closure_batched` (vectorized). `state_dict` /
  `load_state_dict` snapshot the wrapped inner optimizer and evaluation
  counter, so save/load round-trips preserve optimization progress.
  `step()` raises `RuntimeError` once the budget is consumed (callers
  that ignore `is_exhausted` get a loud failure instead of silently
  running past budget).
- `SearchSpace` with `Float`, `Int`, `Categorical` for typed parameter
  spaces.
- `normalize_bounds` validates that every dimension has positive span.
  Zero-span bounds (`bounds=0` or `bounds=(v, v)`) now raise `ValueError`
  at construction instead of crashing deep inside the CMA-ES
  eigendecomposition.

### Serialization

- `state_dict` / `load_state_dict` on every core optimizer, including
  full `PhasedDFO` serialization of all ~30 dynamic fields (nested
  SHADE / CMAES / portfolio states, elite pool, phase counters). A
  cross-phase round-trip test covers the transitions.
- Cross-device RNG handling: state dicts store `_rng_seed` and
  `_saved_device_type`. Same-device loads are bit-exact; cross-device
  loads re-seed from the stored seed (non-bit-exact, documented).

### Device support

- CPU, CUDA, MPS. Every public optimizer has an MPS smoke test.

### Benchmarks

- Classical suite at dims 10, 20, 40 (sphere, rosenbrock, rastrigin,
  ackley, griewank, levy, plus shifted and rotated variants at the
  stress dim). `benchmarks/run_benchmarks.py` runs the suite
  end-to-end.
- Scaling probe (`benchmarks/scaling_probe.py`) and multi-function
  sweep (`benchmarks/run_scaling_sweep.py`) across six classical
  functions, five optimizers, and a dim ladder up to 20 480.
  Per-optimizer scaling guidance from the v0.9 sweep is documented in
  `docs/benchmarks.rst`.

### Documentation

- Sphinx + Furo + myst-parser + sphinx-gallery site, hosted on
  ReadTheDocs at `torch-dfo.readthedocs.io`. Configuration in
  `docs/conf.py`; RTD build in `.readthedocs.yaml`.
- Five runnable examples in `examples/`: basic ask/tell, gradient-free
  NN training with `DFOOptimizer`, checkpointing / warm-start,
  multi-device usage, and a high-dimensional wall-clock benchmark.
- NumPy-style docstrings (Parameters, Attributes, Notes, References,
  Examples) on `CMAES`, `SHADE`, `NelderMead`, `PhasedDFO`, and
  `DLRPortfolio`.

### Tests

- 959 tests across 26 files. The suite went through two quality-focused
  audit passes before release: the first removed tautological and vacuous
  assertions (7 Critical + 21 High + 14 Medium findings fixed,
  consolidating several stale skipped tests); the second filled
  edge-case and error-path gaps (10 additional tests across construction
  validators, budget exhaustion, cross-device fallbacks, and zero-span
  bounds). Real bugs surfaced by the audit were fixed where they
  affected the user-facing API.

### Development

- Supported: Python 3.10, 3.11, 3.12, 3.13. Requires PyTorch ≥ 2.4.
- No NumPy at runtime; enforced by an AST guard
  (`tests/test_no_cpu_binding_imports.py`) that walks every file under
  `src/torch_dfo/` and fails on any import of `numpy`, `scipy`,
  `pandas`, `numba`, `jax`, etc.
- `py.typed` marker (PEP 561). Downstream type checkers see typed
  stubs.
- CI: three-Python CPU matrix on GitHub Actions; CUDA matrix on a
  self-hosted runner; PR-triggered docs build
  (`.github/workflows/docs.yml`) with `sphinx-build -W --keep-going`;
  tag-triggered release workflow (`.github/workflows/release.yml`) via
  OIDC Trusted Publisher with TestPyPI smoke test before PyPI publish.
- Lint: `ruff` with `E, F, I, UP, B, RUF, SIM, C4, PERF, RET`, plus
  `ruff format`. mypy strict on core modules; allowlist for `phased`,
  `_polish`, `dlr_cma`. Coverage gate `--cov-fail-under=85`; Codecov
  upload on the Python 3.12 lane.
- Community files: `CITATION.cff`, `SECURITY.md`, GitHub issue and PR
  templates, `.pre-commit-config.yaml`, `.codespellrc`.
