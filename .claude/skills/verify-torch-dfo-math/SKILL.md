---
name: verify-torch-dfo-math
description: "Formally verify torch-dfo's mathematical, state-machine, and IEEE-754 floating-point invariants in Lean 4 and Z3, plus Tier-2 PyTorch refinement tests. Use whenever editing cmaes.py, dlr_cma.py, shade.py, nelder_mead.py, base.py, utils.py, space.py, run.py, optim/dfo_optimizer.py, _operators.py, _polish.py, or examples/06_engineering_control.py."
---

# Verify `torch-dfo` mathematical and boundary invariants

This skill runs the repository's machine-checked Lean 4 proofs (`formal/lean/*.lean`), the Z3 SMT IEEE-754 and real-arithmetic checks (`formal/z3/verify_torch_dfo.py`), and the Tier-2 PyTorch refinement suite (`tests/test_formal_invariants.py`).

## Sync commit and coverage record

- **Sync commit:** `a2b9ed0` (last confirmed commit where all Lean 4 models, Z3 assertions, and refinement tests match `src/torch_dfo/`).
- Check whether covered product code changed since the sync commit:
  ```bash
  git diff --stat a2b9ed0..HEAD -- \
    src/torch_dfo/base.py \
    src/torch_dfo/cmaes.py \
    src/torch_dfo/dlr_cma.py \
    src/torch_dfo/shade.py \
    src/torch_dfo/nelder_mead.py \
    src/torch_dfo/space.py \
    src/torch_dfo/utils.py \
    src/torch_dfo/run.py \
    src/torch_dfo/optim/dfo_optimizer.py \
    src/torch_dfo/_operators.py \
    src/torch_dfo/_polish.py \
    examples/06_engineering_control.py
  ```
- Move the sync commit forward **only** after verifying that every changed formula or branch in the covered files is reflected in the Lean/Z3 models and all checks pass.

### Coverage map

| Check file | Covered code | Property in plain English | Harm prevented |
| :--- | :--- | :--- | :--- |
| `formal/lean/CMAESAndDLR.lean` | `cmaes.py` (`_compute_adaptation_rates`, `_configure_strategy`, `tell`, `_update_eigensystem`), `dlr_cma.py` (`_compute_branch_rates`, `tell`) | Passive and Mahalanobis-scaled active CMA updates keep `C` symmetric and PSD; 3-variable gauge normalization (`C/s`, `sigma*sqrt(s)`, `p_c/sqrt(s)`) preserves `sigma^2 C`, `p_sigma`, and next-step evolution; Hansen `c_c` stays in `(0, 1)`; DLR diagonal update with `c1 * delta_h` and pre-update `sigma_old` preserves stationary variance. | Negative covariance eigenvalues, `p_c` collapse (`c_c > 2` at large batch sizes), sampling-covariance jumps on eigendecomp, and exponential diagonal variance blowup in `DLRPortfolio`. |
| `formal/lean/SearchRunInvariants.lean` | `base.py` (`next_batch_size`), `nelder_mead.py` (`next_batch_size`, `ask`), `run.py` (`SearchRun`), `optim/dfo_optimizer.py` (`DFOOptimizer`) | `SearchRun` preserves `completed <= attempted <= charged` and `charged + reserved <= max_evals` across any dynamic `next_batch_size()`; `NelderMead.next_batch_size()` equals `NelderMead.ask().shape[0]` in every state for all `dim >= 1`; `DFOOptimizer` never overruns `budget_remaining`. | Budget overruns, `SearchRun` shape-mismatch failures for `NelderMead` at `dim != 3`, and `AssertionError` crashes in `DFOOptimizer` at `dim in {1, 2}`. |
| `formal/lean/NelderMeadAndThermal.lean` | `nelder_mead.py` (`tell`), `_polish.py` (`nm_polish`), `examples/06_engineering_control.py` | Nelder-Mead's 4-way decision tree is mutually exclusive and exhaustive and strictly improves the worst vertex whenever no shrink occurs; `nm_polish` with `_initialized = True` evaluates the warm-started simplex; the CUDA thermal kernel equals the analytic recurrence, contracts by `47/50`, and respects its BIBO invariant box. | Silently discarding warm-start points during polish, non-decreasing simplex updates, or thermal kernel divergence. |
| `formal/lean/DEOperators.lean` | `_operators.py` (`de_current_to_pbest_mutation`, `de_binomial_crossover`, `opposition_init`) | Vectorized index maps `r1` and `r2` are bijections avoiding self/duplicate indices without wraparound or clamp distortion; binomial crossover forces `j_rand`; opposition mirror stays inside `[lb, ub]`. | Self-mutation degeneracy, out-of-bounds indices, or unmutated trial vectors in SHADE. |
| `formal/z3/verify_torch_dfo.py` | `space.py`, `shade.py`, `base.py`, `nelder_mead.py`, `utils.py`, `cmaes.py`, `dlr_cma.py` | Categorical `(idx + 0.5) / n` round-trips cleanly for `n = 1..50` in `f32` and `f64`; SHADE `+inf -> finite` fallback keeps `memory_F` and `memory_CR` finite; `BaseOptimizer._update_best` and `NelderMead.tell` mask `NaN` to `+inf`; `sanitize_fitness` uses `isfinite()`; Hansen rates and active PSD bound hold in `QF_NRA` for all `dim >= 1, mu_eff >= 1`. | Off-by-one categorical decode bugs, `NaN` poisoning of SHADE memory, silent loss of best finite solutions when one candidate is `NaN`, and invalid learning rates. |
| `tests/test_formal_invariants.py` | All of the above + `phased/`, `benchmarks/classical.py` | Tier-2 PyTorch refinement tests binding the Lean/Z3 models directly to live PyTorch execution. | Drift between formal models and actual Python/PyTorch implementation. |

## How to run

1. Run Lean 4 (4.16.0, pure `Init`, zero external deps) and Z3 (4.15.4+):
   ```bash
   LEAN_BIN="${LEAN_BIN:-lean}" \
   Z3_PYTHON="${Z3_PYTHON:-uv run --no-project --with z3-solver python}" \
   bash formal/verify.sh
   ```
2. Run the Tier-2 refinement tests:
   ```bash
   pytest -q --no-cov tests/test_formal_invariants.py
   ```

Both commands must exit `0` with zero warnings, zero `sorry`, and zero custom axioms.

## Telling a real bug from a modeling error

- **If Lean 4 or Z3 fails after you edited product code and updated the model to match:** replay the counterexample in `tests/test_formal_invariants.py` against real PyTorch. If PyTorch exhibits the failure, it is a **real bug** in `src/torch_dfo/`—fix the product code, do not weaken the property.
- **If Lean 4 or Z3 passes, but `tests/test_formal_invariants.py` fails:** the model drifted from the implementation or a numerical boundary was missed—inspect the real code diff against the coverage map.
- **When a promise intentionally changes:** update the model, the theorem statement, and the Tier-2 test together, prove the updated check fails on a broken variant, and advance the sync commit above.

## Scope and what is NOT covered

- Lean 4 proofs over `OrderedField` assume exact real/rational arithmetic; IEEE-754 rounding is checked by Z3 (`f32`/`f64`) for bounded operations (e.g. `n = 1..50` for Categorical, finite inputs within `±1e30`/`±1e100` for `sanitize_fitness`).
- Active CMA PSD proof assumes `C_invsqrt` matches `C^{-1/2}` (exact when `_decomp_frequency == 1`; approximate within ~10% between periodic eigendecompositions when `c_1 + c_mu <= 1 / (20 * dim)`).
- Full multi-phase convergence quality of `PhasedDFO` and GPU kernel timing are empirical, verified by `tests/` and `benchmarks/`, not by Lean 4 or Z3.
- Thermal plant `float32` quantization has a documented 1-ULP rounding boundary at `0.125 -> 0.25` (checked in `verify_torch_dfo.py`).
