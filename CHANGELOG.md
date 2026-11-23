# Changelog

All notable changes to torch-dfo.

## [Unreleased]

### Optimizers and numerical invariants

- Fix `CMAES` and `DLRPortfolio` covariance learning rate `c_c` to Hansen (2016) Table 1 so `c_c < 1` holds at all batch sizes.
- Add Mahalanobis weight scaling and PSD bound to `CMAES` active covariance updates.
- Default `CMAES._normalize_on_decomp` to `False` and apply an exact three-variable gauge transform (`eigenvalues`, `sigma`, `p_c`) whenever eigensystem normalization triggers, preserving `sigma^2 * C` even under finite `sigma_min`/`sigma_max` bounds.
- Fix `DLRPortfolio` stationary diagonal variance (`c1 * delta_h`), pre-update `sigma_old` scaling, `restart_branch` generation reset, `K > 4` default sigma factors, `cma_budget` / `lam_max` bounds, unit-norm rank-1 `V` columns, and cross-device RNG serialization.
- Remove unused `_woodbury_solve` from `dlr_cma.py`.
- Harden `NaN` and `+inf` handling across `BaseOptimizer._update_best`, `DLRPortfolio.tell`, `NelderMead.tell`, `SHADE.tell`, `sanitize_fitness`, `multistart_basin_explore`, and `nm_polish`.
- Encode `SearchSpace` `Categorical` parameters at bin midpoints `(idx + 0.5) / len(choices)` for exact `float32`/`float64` round-trips.

### Polishing, orchestration, and wrappers

- Preserve warm-start simplex and per-dimension tensor bounds in `nm_polish`, and reuse accepted finite-difference gradients in `fd_bfgs_polish`.
- Enforce strict evaluation budgets, per-dimension tensor bounds, isolated RNG threading, and shared generator restoration in `PhasedDFO`.
- Add `BaseOptimizer.next_batch_size()` (overridden by `NelderMead`) so `SearchRun`, `minimize`, and `DFOOptimizer` support `NelderMead` cleanly across all dimensions; explicitly raise `TypeError` when `PhasedDFO` is passed to `SearchRun`.
- Restore `DFOOptimizer` parameters inside `try...finally` on closure errors and resize population buffers in `BaseOptimizer.load_state_dict()` when loading checkpoints saved after `SHADE` population reduction.
- Advance passed `torch.Generator` instances in `random_rotation_matrix` and `random_shift`.

### Verification and maintainability

- Add Lean 4 (`formal/lean/`) and Z3 (`formal/z3/verify_torch_dfo.py`) verification suite with runner `formal/verify.sh` and Tier-2 PyTorch refinement tests in `tests/test_formal_invariants.py`.
- Strip non-directive inline comments from `src/torch_dfo/` and add static lint guard `tests/test_no_inline_comments.py`.
- Tighten user documentation and roadmap to match implemented behavior.

## 0.11.0b1 — 2026-09-19

### Budgeted search

- Add `SearchRun` and `minimize` for CMA-ES, SHADE, and random search.
- Keep one pending batch, count logical evaluations, and preserve raw results.
- Add fixed repeats and planned-pause checkpoints with explicit fork lineage.
- Preserve the existing optimizer interfaces.

### Evaluation and maintenance

- Add a quantized thermal-control reference study with fixed train and held-out scenarios.
- Record evaluator probes, domain and random baselines, and bounded tuning trials.
- Correct budget, checkpoint, bounds, and benchmark-reporting defects found during the foundation review.
- Document the beta interface and its supported behavior.

## 0.10.0 — 2026-04-18

### Internal State

- Extracted SHADE success-history memory, archive, trial buffers, and
  initialization flag into `SHADEMemory`.
- Extracted CMA-ES evolution-path buffers into `CMAPathState` and
  Hansen adaptation constants into `CMAAdaptationRates`.
- Extracted PhasedDFO runtime state into four state bundles:
  `PhasedPolishState`, `PhasedValleyState`, `PhasedDEState`, and
  `PhasedCMAState`.
- Preserved compatibility accessors for existing internal test and
  downstream access patterns while moving real storage out of the flat
  optimizer instance layout.

### Serialization

- Nested SHADE memory now serializes under `state["memory"]`.
- Nested CMA-ES path state now serializes under `state["path"]`.
- PhasedDFO state serialization now delegates DE, valley, polish, and
  CMA phase state to the new state bundles, including CMA phase budgets
  and nested CMA-ES portfolio state.

### Quality

- Restored repo-wide `ruff check .`, `ruff format --check .`, and strict
  mypy cleanliness.
- Default optimizer construction now falls back to CPU when `device=None`
  would otherwise select MPS with unsupported `float64` tensors.

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
