# Public applications and maintenance value

19 September 2026.

torch-dfo is an independent maintained library. Public application workflows identify useful tasks and benchmark baselines. A new algorithm or a universal score advantage is not required. The next milestone is a reproducible application suite using the native torch-dfo API.

## Maintenance evidence

EvoTorch's latest default-branch commit and release are dated 14 May 2025. Its repository is not archived. Seven pull requests remain open, including contributions from 2026. These facts support concern about limited visible release activity. They do not establish abandonment.

Sources: [default-branch commit](https://github.com/nnaisense/evotorch/commit/cebcac4f20979078becf8b908016cd8e5e6714a4), [latest release](https://github.com/nnaisense/evotorch/releases/tag/v0.6.1), [contributions](https://github.com/nnaisense/evotorch/pulls). The adjacent JSON files preserve the GitHub API readings.

## Applications found

| Public application | EvoTorch use | Current evidence |
| --- | --- | --- |
| [calisim](https://github.com/Plant-Food-Research-Open/calisim/tree/fa6539b2d1e4a39210fef3d9dabee97b893ec69c) | Simulation calibration, including a predator-prey example | Executed the extracted public objective. The full package wrapper was not run. |
| [EvoIK](https://github.com/knowledgetechnologyuhh/evo_ik/tree/090ad201d821d9043764c6ad180e180d3a3fdcfe) | CMA-ES over batched robot forward kinematics | Reproduced a constructor incompatibility. Executed a repaired adapter and torch-dfo on a small pose set. |
| [Contimask](https://github.com/eth-siplab/contimask/blob/f536b4e8946adf529ec58330dbc4207c4f321ec3/attribution/mask_conti.py) | PGPE over neural perturbation masks | Inspected source. Synthetic-data reproduction remains future work. |
| [DynAMO](https://github.com/michael-s-yao/DynAMO/blob/52dfc8fe4cb4de6374635bc5f971974f027d4d1f/src/dynamo/optim/cmaes.py) | CMA-ES proposals for offline model-based design | Inspected source. Dataset preparation and surrogate training remain future work. |

## EvoIK execution

The original source failed before optimization. EvoTorch CMA-ES rejects the bounded `Problem` that EvoIK supplies. This is a reproduced integration incompatibility, not evidence of a newly introduced regression.

A separate adapter removed the Problem bounds and retained its penalty. It did not change the original source. The adapter and torch-dfo ran on Spark CUDA with Torch 2.14.0+cu130. Each solved one of two reachable poses under the upstream limits of 1 cm and 0.349 radians. Each failed a deliberately unreachable pose.

The budget was 20 populations of 32 candidates per case. The nominal 640 candidate count comes from the loop, not an instrumented callback counter. Final physical checks are additional work.

This is a smoke test. The adapter uses unbounded sampling with a penalty and reports the final population's best. torch-dfo uses bounded sampling and reports the best seen. Returned commands are clamped before physical checks. These differences prevent a fair ranking. The reported bounds check concerns the clamped command only.

A post-run evaluator check scored a known solution at zero error and confirmed a positive out-of-bounds penalty. This audit happened after the optimizer runs. It does not satisfy a claim that every autoresearch probe ran beforehand.

The dependency installation selected Torch 2.14.0 in the isolated target. The result does not validate the image's original Torch 2.13.0 environment. See the retained raw log and exact launch script.

## Calibration execution

The extracted calisim Lotka-Volterra example ran on the Spark CPU. An instrumented EvoTorch GA repeat made 2,000 objective calls. The first 1,100 count was inferred incorrectly and is superseded.

The current torch-dfo beta ran the same physical objective with normalized parameter coordinates. CMA-ES made 1,100 search calls and reached MSE 12.97734. A seeded uniform baseline made 1,100 calls and reached MSE 13.01270. Three final re-evaluations matched their stored values. The total was 2,203 objective calls.

This single-seed check shows executable objective compatibility. It does not establish a quality advantage, parameter recovery, or held-out prediction quality. The GA run had a different budget, algorithm, dtype, and random state. The complete calisim wrapper did not run.

The first torch-dfo attempt imported an older remote checkout. It was superseded by an unchanged rerun using beta 0.11.0b1. The remote CMA-ES and package initializer hashes match the current local beta. The retained beta stdout is the accepted result.

## Decision

The public integrations show existing application demand. The executions show that torch-dfo can evaluate real application objectives, with narrow limits. They do not yet establish complete application validation or superior results.

Build the next acceptance suite from these applications. Start with complete calibration and robot fixtures. Preserve the original upstream result and any required repair. Expand PGPE support through a released mask or policy example when that lane is ready.

Measure domain success, failed runs, evaluation counts, integration work, and runtime. Require stronger domain baselines only when claiming an advantage over those methods. Retain the thermal study's negative result without making it the sole test of the library's purpose.
