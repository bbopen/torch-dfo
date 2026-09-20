# CUDA execution on DGX Spark

The prepared thermal-control scorer passed the frozen CUDA execution study.
Warm CMA-ES runs were 1.28 to 1.68 times faster than the reference eager scorer.
All 20 configuration-seed pairs had identical final reference scores.
Each configuration ran three execution paths, for 60 measured searches.

## What changed

`FrozenEvaluator.prepare(device, dtype, compile=False)` caches scenario tensors
on the selected device. Set `compile=True` to use full-graph PyTorch compilation
with CUDA graphs and eager-compatible arithmetic settings. The compiler generates
the GPU kernels. The public optimizer and `SearchRun` code remain unchanged.

The original evaluator remains the independent reference. The compiled scorer
clones its output so a later CUDA graph replay cannot overwrite a retained result.
No optional dependency was added to the core library.

The compiled example was tested with PyTorch 2.13.0+cu130 and Triton 3.7.1 on GB10.
It requires the Inductor options `emulate_precision_casts`,
`eager_numerics.division_rounding`, and `triton.cudagraphs`. Older PyTorch builds
without these options can use the eager scorer. The optional compilation tests
explicitly skip such builds; they do not establish compilation support there.

## Measured results

Each search used 20 generations through `SearchRun`. Seeds were 11 through 15.
The comparison rotated execution order and synchronized GPU timing boundaries.
Search timing includes the managed loop and result snapshot, but excludes optimizer
construction and compilation. Scores use the same frozen train objective.

| Dtype | Population | Eager search median | Compiled search median | Search speedup | Evaluator speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 12 | 39.48 ms | 23.54 ms | 1.68x | 10.33x |
| float32 | 4096 | 62.17 ms | 45.66 ms | 1.36x | 9.31x |
| float64 | 12 | 44.56 ms | 28.36 ms | 1.57x | 10.91x |
| float64 | 4096 | 78.02 ms | 61.19 ms | 1.28x | 5.43x |

Every paired compiled search was faster. All searches completed their exact
candidate budget without failures or uncertain accounting. The largest selected
score difference from the oracle was 2.39e-7 in float32. Final scores, audited by
the independent reference, matched exactly between execution paths.

Preparation plus the first compiled call cost 3.50, 2.83, and 2.13 seconds for
the first three configurations. Their measured savings repay that cost after
about 220, 172, and 132 searches of this length. The fourth configuration reused
compiler work and started in 1.62 ms. These are costs in study order, not four
independent cold starts. A single short search may be faster without compilation.

The profiler recorded 140 CUDA events per reference evaluator call and 7 for the
compiled path. Each managed generation still made 20 host scalar reads. The whole
search loop is not GPU graph-captured. Those synchronizations are a next target.

## Experiment ledger

| Attempt | Change | Outcome | Decision |
| --- | --- | --- | --- |
| 1 | Initial container command | Pytest coverage plugin absent; no tests or benchmark ran | Correct command; disable coverage collection for this run |
| 2 | Default compiler arithmetic | Tests passed; float32 population 4096 changed final scores for three seeds | Reject under the frozen paired-score rule |
| 3 | Disable FMA through an assumed option | PyTorch rejected the unsupported option; four compilation tests failed | Replace the option using the installed compiler configuration |
| 4 | Eager precision casts and division rounding | All four configurations passed quality, budget, and speed criteria | Keep |

Attempt 2's changed scores differed by 0.0048 to 0.0178 despite individual objective
scores remaining within tolerance. This is evidence that small numerical changes
can matter to adaptive search. The correction addresses arithmetic behavior.
It does not identify one flag as the sole cause.

The [protocol](cuda-protocol.md) retains the original acceptance criteria.
The [final JSON](results/cuda-control.json) records all samples, scores, selected
controls, source hashes, accounting, probes, and profiler summaries.
The [rejected JSON](results/cuda-control-default-arithmetic.json) retains the failed
comparison. Focused tests passed 23 cases, with two optional EvoTorch tests skipped
because that package was not on this container's import path. The final regression
snapshot passed 1,145 tests with 13 skips on Spark CPU/CUDA in 40.82 seconds.
Coverage collection was disabled because the container lacks the coverage plugin.
Ruff lint and formatting passed. The reproduction snippet completed 240 CUDA
evaluations; `results/cuda-control-doc-smoke.txt` records its runtime versions. An independent review checked the numerical
reference, source hashes, budget accounting, and every recorded acceptance result.

This study measures execution speed on one small thermal model. It does not
establish better optimization quality, superiority over the PI domain baseline,
or a general advantage over CPU execution or other DFO libraries.

## Reproduce

Use a CUDA-enabled PyTorch build with the compiler options above. From the checkout:

```sh
PYTHONPATH=src python -m pytest -o addopts= tests/test_engineering_cuda.py -q
PYTHONPATH=src python benchmarks/cuda_control.py --output cuda-control.json
```

From the checkout, start `PYTHONPATH=src python`. Load the engineering example
and prepare the objective once:

```python
import runpy
import torch
from torch_dfo import CMAES, SearchRun

FrozenEvaluator = runpy.run_path("examples/06_engineering_control.py")["FrozenEvaluator"]
evaluator = FrozenEvaluator("train")
objective = evaluator.prepare("cuda", torch.float64, compile=True)
optimizer = CMAES(16, 1.0, pop_size=12, device="cuda", dtype=torch.float64, seed=11)
run = SearchRun(optimizer, max_evals=240)
while not run.done:
    run.step(objective)
```

The first call compiles the objective. Keep the prepared callable alive across
repeated searches to amortize that cost.
