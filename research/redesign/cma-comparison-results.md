# CMA-ES comparison on DGX Spark

September 20, 2026. This panel produced 100 complete runs and 20 EvoX failures.
It provides a pycma comparison and an EvoX compatibility result. It does not establish a general library ranking.

## Setup

The [protocol](cma-comparison-protocol.md) fixed four 16-dimensional tasks, population 32, 3,200 evaluations, and seeds 11 through 15.
All methods used float64. CMA-ES started at mean zero with sigma 0.6.
The host used NVIDIA GB10, PyTorch 2.13.0+cu130, NumPy 2.3.5, and one CPU thread.
The comparison used eager execution, including cached thermal scenario tensors.

[Raw results](results/cma-comparison/panel.json) include all 120 attempts, per-generation traces, selected candidates, source hashes, probes, and failures.
Every complete run used exactly 3,200 evaluations. Each EvoX attempt stopped after 32 evaluated candidates.
The panel's status remains `failed` because its EvoX runs did not complete.

The archive came from torch-dfo commit `48007c9` plus the comparison files in this change.
The remote archive omitted Git metadata, so `torch_dfo_head` is null. Source hashes identify the executed algorithms, evaluator, and comparison script.
Optional dependencies were installed separately with `--no-deps`; the existing Torch build stayed unchanged.
EvoX 1.4.0 and pycma 4.5.0 source hashes matched the pinned upstream files.

## Solution quality

These are median best-evaluated objective values across five runs. Lower is better.
Equal seed labels do not align candidate sequences between different optimizer implementations.
pycma's two modes use the same seeded CPU optimizer. Native CPU and CUDA use different random streams.

| Task | torch-dfo CPU | torch-dfo CUDA | pycma CPU | pycma CUDA objective | Random CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted sphere | 3.25e-6 | 6.76e-6 | 5.43e-5 | 5.43e-5 | 6.46 |
| Rotated ellipsoid, condition 1e6 | 823.50 | 864.97 | 825.08 | 825.08 | 62,137.39 |
| Rotated Rastrigin | 8.25 | 12.48 | 45.36 | 45.36 | 104.08 |
| Thermal train | 2.0712 | 2.0583 | 2.0531 | 2.0531 | 2.1837 |

No complete run reached a canonical target within the budget. All 25 complete thermal runs reached the existing threshold, including random search.
That thermal threshold therefore does not distinguish capable search methods at this budget.
The fixture probes and CPU audits passed, but they do not make that threshold a strong comparison metric.

Native CMA-ES had lower median scores on sphere and Rastrigin in this panel.
Ellipsoid results were similar; thermal results were close.
Rastrigin varied substantially: native CPU ranged from 5.99 to 9.59, native CUDA from 7.13 to 79.46, and pycma from 11.54 to 84.56.
Five seeds on one instance per task do not establish general superiority.

These implementations differ mathematically. Native CMA-ES normalizes covariance eigenvalues by their mean after clamping them.
EvoX decomposes periodically and clamps eigenvalues. pycma retains its own adaptation and decomposition rules.
Active covariance and mirroring were disabled where configurable. No method was tuned after these results.

Thermal held-out diagnostic medians were 4.7766 for native CPU, 4.9176 for native CUDA, 4.9379 for both pycma paths, and 4.9410 for random search.
This split was already exposed. The new unbounded latent search also differs from the earlier bounded thermal study.
None of the 25 selected candidates passed the held-out threshold of 3.5.
These scores do not replace that study's negative result against the PI controller or demonstrate engineering usefulness beyond it.

## Execution cost

Median setup plus optimizer initialization and search time, in milliseconds:

| Task | torch-dfo CPU | torch-dfo CUDA | pycma CPU | pycma CUDA objective | Random CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted sphere | 22.0 | 120.8 | 27.8 | 38.7 | 11.5 |
| Rotated ellipsoid | 22.1 | 121.8 | 28.3 | 41.2 | 12.0 |
| Rotated Rastrigin | 22.7 | 123.2 | 29.0 | 43.2 | 12.7 |
| Thermal train | 47.9 | 185.9 | 55.4 | 106.2 | 36.7 |

CPU execution was faster at this small population and objective cost.
pycma's optimizer stayed on CPU in both paths. Its CUDA path includes objective transfers.
Its two paths produced scores within 2.3e-13 for each seed and task. Their selected candidates match exactly across all 20 seed/task pairs. The code uses the same private random seed in both modes; treat them as timing variants rather than independent quality samples.
Timing includes counted observations, incumbent tracking, and GPU synchronization. It excludes package imports, source checks, preflight probes, and post-search CPU audits.
CUDA allocation measurements use PyTorch's allocator, not total device memory.
These short, eager runs do not establish peak throughput or compare compiled workflows.
The earlier compiled CUDA study measured a different execution question.

## EvoX findings

The isolated covariance probe reproduced a rank-one update defect on CPU and CUDA.
EvoX computes `p_c @ p_c.T` for a one-dimensional path, producing a scalar.
CMA-ES requires the outer product.

The one-line correction passed focused eager, vmap, and full-graph compiled vmap regression checks on CPU and CUDA.
The same tests failed against the original source. This validates the covariance correction, not the full upstream algorithm suite.

The comparison kept EvoX 1.4.0 unchanged. All its runs failed inside `torch.cond` with an input-aliasing error on this Torch build.
A separate [plain Sphere workflow probe](probe_evox_workflow.py), without our counter or custom monitor, reproduced the failure on CPU and CUDA.
The [CPU](results/cma-comparison/evox-workflow-cpu.json) and [CUDA](results/cma-comparison/evox-workflow-cuda.json) records preserve those errors.
This compatibility failure is separate from the covariance defect. No full-budget EvoX quality or timing result is available.

## What follows

This panel supports continuing torch-dfo as a small, maintained library, but does not prove its broader differentiation.
EvoX overlaps directly in PyTorch GPU evolutionary computation. pycma supplies a mature CMA-ES reference with a narrower library scope.
Our budget accounting, reproducible studies, and research tools must earn their value through useful workflows.

The next performance study should increase objective cost or batch size and compare compiled workflows after checking numerical parity.
A separate engineering study still needs to beat a strong domain baseline on unseen scenarios, or show another practical benefit.
Keep the existing PI result visible. Do not use the EvoX defects as a performance claim.

## Reproduction

Install the pinned optional packages beside a supported Torch build:

```sh
python -m pip install --target /tmp/torch-dfo-cma-deps --no-deps evox==1.4.0 cma==4.5.0
PYTHONPATH=/tmp/torch-dfo-cma-deps:. python benchmarks/cma_comparison.py --output cma-panel.json
```

On the measured Torch build, expect the EvoX failures and a nonzero panel exit status.
The [smoke record](results/cma-comparison/smoke.json) used two generations before the full panel.
The original focused test run had nine passes and one EvoX failure. That failure remains recorded; it is not a successful optimization result.

## Validation

The Spark regression suite passed 1,154 tests, skipped 13, and reported one expected EvoX failure.
Coverage collection was disabled because the container lacks pytest-cov.
The optional EvoX test recognizes only the observed aliasing error on EvoX 1.4.0 with Torch 2.13, after exactly one 32-candidate batch.
Other errors still fail. This expected failure does not change the saved panel's failed status.
The final focused rerun passed nine tests and reported the same expected EvoX failure after checking its exact evaluation counts.
Ruff lint and formatting checks pass for the new code. CI now includes the comparison script in static checks.
See the [independent review](cma-comparison-review.md) for the result audit and interpretation limits.
