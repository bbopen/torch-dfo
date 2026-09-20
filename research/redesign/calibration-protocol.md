# calisim Lotka-Volterra calibration protocol

This protocol defines a small CPU comparison. It uses the public calisim
Lotka-Volterra evolutionary example at commit
`fa6539b2d1e4a39210fef3d9dabee97b893ec69c`.

The task adapts the source example at
https://github.com/Plant-Food-Research-Open/calisim/blob/fa6539b2d1e4a39210fef3d9dabee97b893ec69c/examples/evolutionary/evotorch_example.py.
The source is Apache-2.0. See `benchmarks/licenses/calisim-LICENSE.txt` for
the repository notice. This benchmark adapts a source task. It does not adapt
or require a calisim backend API.

The source example fits all 21 annual lynx observations from 1900 through
1920. This study changes that setup. It fits 1900 through 1915 and audits the
selected training incumbent once on 1916 through 1920. The last five years are
an exploratory temporal holdout inside a public example. They were exposed by
the source example. They are not a new confirmation set.

## Frozen setup

The benchmark copies the source model data and equations. It uses SciPy
`solve_ivp` in a CPU callback. It does not claim acceleration.

| Item | Fixed value |
| --- | --- |
| Train years | 1900 through 1915, 16 observations |
| Held-out years | 1916 through 1920, 5 observations |
| Fit value | Mean squared error on train observations only |
| Selected value | Lowest training incumbent seen during search |
| Audit | One held-out MSE call for the selected incumbent |
| Unit search box | `[0, 1]^2` |
| Physical alpha range | `[0.45, 0.55]` |
| Physical beta range | `[0.02, 0.03]` |
| Seeds | `0, 1, 2, 3, 4` |
| Candidate budget | 400 per algorithm and seed |
| Population size | 20 for every algorithm |
| Native CMA-ES start | Mean `[0.5, 0.5]`, `sigma0=0.1` on unit span one |
| EvoTorch CMA-ES start | Mean `[0.5, 0.5]`, `stdev_init=0.1` |
| Execution | CPU with one Torch thread |

Native `RandomSearch`, `CMAES`, and `SHADE` run through `SearchRun`. The
benchmark uses their public APIs only. It does not add a library algorithm or
change library code.

SciPy `differential_evolution` runs on the same unit box. It uses `popsize=10`,
which gives 20 population members for two parameters, `maxiter=19`,
`polish=False`, `tol=0`, `atol=0`, `workers=1`, and `updating="deferred"`.
That configuration makes 400 calls when SciPy completes its initial population
and 19 later populations. The report keeps the actual count if it stops early.
It tracks the lowest callback score itself and does not treat a final
population result as best seen.

EvoTorch CMA-ES is optional. It provides a workload baseline, not an API
compatibility check. EvoTorch CMA-ES uses an unbounded latent problem. The
benchmark clamps its proposals to `[0, 1]^2` before it maps them to alpha and
beta. Native optimizers propose inside their box. The report states this
difference. The EvoTorch adapter tracks the best score from every actual
callback. It does not use the final population as a substitute for best seen.

## Evaluator probes

Run these four train-only checks before any optimizer runs.

1. Simulate a known parameter pair. Replace train observations with that
   trajectory. Score the pair through the train objective. MSE must be near
   zero.
2. Score an all-zero prediction against the train observations. Its MSE must
   be positive. Record the mean-prediction MSE as a second null value.
3. Recheck a claimed cached fitness through the physical simulator. The result
   must differ from a false zero cache. The callback count must equal one.
4. Score a wrong bounded parameter pair through the same synthetic objective.
   It must score worse. Reject NaN, wrong-shape, and out-of-range inputs.

Any probe failure stops the benchmark. A non-finite solver value or malformed
candidate also stops its run. The command writes a small JSON failure record
before it exits with an error when an output path was supplied.

## Accounting and report

The report records each search's actual candidate calls, objective callback
calls, `SearchRun` charged evaluations, initialization and search seconds,
selected unit and physical parameters, train MSE, and the separate final audit
call. The native callback count must equal `SearchRun` accounting. External
methods record `null` for `SearchRun` accounting. The SciPy baseline flags an
actual count below 400. The EvoTorch callback count must equal its 400
candidate budget.

The report also records hashes for this benchmark and the checked calisim
source file, the exact calisim revision, source files for Torch DFO and optional
EvoTorch CMA-ES, CPU environment, raw results for each seed, and per-algorithm
medians. It records probe calls separately. It has no superiority flag. Five
held-out years from one public series do not show broad generalization.

Run it from the torch-dfo checkout after its benchmark dependencies are
available.

```sh
PYTHONPATH=src python benchmarks/calibration.py \
  --calisim-root ../downstream-calisim \
  --output research/redesign/results/calisim-calibration.json
```

Add `--evotorch --evotorch-src /path/to/evotorch/src` to run the optional
baseline. The command fails if requested EvoTorch imports are unavailable.
