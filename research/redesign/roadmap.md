# Roadmap

## Direction

Build a reliable, PyTorch-native derivative-free optimization library with machine-checked mathematical invariants and reproducible engineering benchmarks.

Core design principles:

- **Pure PyTorch core.** Zero CPU-bound third-party runtime dependencies in `src/torch_dfo/`.
- **Verified numerical contracts.** Maintain formal Lean 4 and Z3 proofs (`formal/verify.sh`) alongside behavioral PyTorch refinement tests (`tests/test_formal_invariants.py`) for covariance updates, step-size adaptation, budget accounting, and IEEE-754 boundary handling.
- **Measured performance.** Claims about convergence quality or GPU speed require paired, equal-budget benchmarks reporting both evaluation counts and end-to-end wall time, including compilation or setup cost.
- **Small, durable surface.** Keep public APIs explicit (`ask`, `tell`, `next_batch_size`, `SearchRun`, `minimize`, `state_dict`/`load_state_dict`) and add new capabilities only when concrete workloads require them.

## Current baseline

- Synchronous budgeted search (`SearchRun` and `minimize`) supporting `CMAES`, `SHADE`, `NelderMead`, and `RandomSearch`, with fixed repeats, raw observation logs, and planned-pause checkpoints.
- Self-budgeted `PhasedDFO` pipeline (`optimize()`) and `DLRPortfolio` for higher dimensions.
- Machine-checked mathematical and IEEE-754 invariants in Lean 4 and Z3 with automated CI checks.
- Reproducible benchmark harnesses in `benchmarks/` covering compiled CUDA thermal control, CMA-ES cross-library comparison, and simulation calibration.

## Next priorities

| Area | Scope | Verification gate |
| --- | --- | --- |
| GPU synchronization reduction | Profile and remove avoidable host scalar reads and Python sync barriers on CUDA hot paths in `CMAES`, `SHADE`, and `DLRPortfolio`. | Paired end-to-end CUDA benchmarks show lower wall time with identical numerical parity and checkpoint compatibility. |
| High-dimensional diagonal ES | Add a compact diagonal evolution strategy (e.g. PGPE / guarded ClipUp) when a high-dimensional control or policy workload demonstrates need beyond `SHADE` and `DLRPortfolio`. | Finite zero-gradient handling, exact budget/checkpoint parity, and equal-budget comparison against existing methods. |
| Constrained and noisy objectives | Extend `SearchRun` beyond unconstrained box bounds and fixed-repeat mean aggregation once an engineering benchmark requires constraint handling or adaptive resampling. | Explicit feasibility accounting, raw observation retention, and matched baseline comparisons on a frozen evaluator. |
| Discrete and mixed search | Add representation-aware mutation/sampling for integer and categorical spaces beyond unit-cube encoding, evaluated on discrete architecture-search benchmarks (e.g. NASBench201). | Track unique evaluated designs, duplicate rates, and held-out test metrics against random and categorical baselines. |
| Multi-objective search | Introduce Pareto archive selection under a dedicated multi-objective contract when a concrete trade-off study requires it. | Hypervolume/dominance invariants verified alongside bounded budget accounting. |

## Maintenance rules

- Keep `src/torch_dfo/` free of non-directive inline comments and CPU-binding imports.
- Update formal models in `formal/` whenever covered mathematical formulas or state transitions change.
- Retain failed or negative benchmark results alongside wins; do not cherry-pick tasks or budgets.
