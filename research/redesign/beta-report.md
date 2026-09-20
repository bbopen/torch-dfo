# torch-dfo 0.11.0b1 beta candidate

The beta adds one budgeted run interface around CMA-ES, SHADE, and uniform random search.
It preserves existing optimizer APIs and keeps the core dependency list at PyTorch alone.
This is a locally validated release candidate. It has not been published to PyPI.

## What changed

`minimize` drives `SearchRun`. The run permits one pending batch, enforces logical evaluation caps,
retains raw observations, and stops on failed or invalid evaluations. Fixed repeats use their arithmetic mean.
Planned checkpoints restore the same candidate stream with a new run identity and parent lineage.

The implementation uses one module and small data classes. It adds no registry, evaluator hierarchy, or distributed scheduler.
The optimization loop does not copy its entire result history after each step. Callers request snapshots explicitly.
The standalone engineering example uses the public API. The benchmark adds experiment records and optional EvoTorch comparison.

The README, quickstart, run guide, API reference, changelog, package manifest, and CPU CI checks now cover the beta.
Earlier foundation corrections remain included. They cover budgets, serialization, bounds, and false benchmark success reports.

## Validation

| Check | Result |
| --- | --- |
| Full local CPU suite, Python 3.12 / Torch 2.14 | 1,125 passed, 4 skipped; 91.20% coverage |
| Full Spark CPU/CUDA suite, Torch 2.13 + CUDA 13.0, NVIDIA GB10 | 1,118 passed, 11 skipped |
| Final engineering integration tests on Spark | 8 passed, including real EvoTorch CPU and CUDA adapters |
| Independent adversarial review | Budgets, failures, candidate identity, RNG isolation, all three checkpoint paths passed |
| Ruff and strict mypy | Passed |
| Sphinx documentation build | Passed with warnings treated as errors |
| Wheel and source archive | Built; metadata checks passed |
| Clean wheel install | All three methods and checkpoint continuation passed |
| Documentation examples | All seven Python blocks passed against the installed wheel |

The full suites ran before the optional EvoTorch regression test was added.
The final focused tests cover that adapter change. The algorithm and run code did not change afterward.
The source archive now includes files needed by its tests and examples.

## Engineering study

The reference models a single thermal zone with 16 quantized heating/cooling commands.
It uses eight train scenarios and five held-out scenarios. The model is synthetic, not calibrated to a building.
The evaluator audit rejected the first provisional task. Version 2 passed the declared probes before search tuning.
All subsequent comparisons use the frozen version 2 evaluator.

Three train-only CMA-ES sigma trials precede five fixed seeds per search method.
Each search seed receives 96 evaluations. Every selected solution receives one held-out evaluation.
Simple PI and nine-point constant-control baselines remain in the comparison.

Mean held-out scores on the Spark are below. Lower is better. The synthetic audit gate is 3.50.

| Method | CPU score | CUDA score | Held-out gate passes |
| --- | ---: | ---: | --- |
| Mean-scenario PI | 3.241 | 3.241 | 1/1 on each device |
| Constant-control grid | 4.078 | 4.078 | 0/1 on each device |
| Random search | 7.556 | 7.322 | 0/5 on each device |
| torch-dfo CMA-ES | 4.872 | 4.877 | 0/5 on each device |
| torch-dfo SHADE | 4.845 | 5.033 | 0/5 on each device |
| EvoTorch CMA-ES | 4.964 | 4.876 | 0/5 on each device |

The DFO methods improved training scores but failed to generalize to this held-out split.
They did not beat the domain baseline. This study establishes a reproducible comparison, not an application win.
CPU and CUDA have different random streams. Train-only tuning selected sigma 0.30 on CPU and 0.15 on CUDA.

Each device study used 2,218 train evaluations, 22 validation evaluations, and 68 evaluator-audit evaluations.
That is 2,308 candidate evaluations and 18,398 scenario rollouts per study, including tuning and audit costs.
Separate performance probes and earlier failed integration attempts are not included in those study totals.

EvoTorch is pinned to `cebcac4f20979078becf8b908016cd8e5e6714a4`.
All 60 remote Python source files match that clean reference checkout.
Its CMA-ES adapts unbounded latent proposals; the physical objective clamps them before evaluation.
Torch-dfo samples within bounds. The report discloses this difference instead of claiming identical internal behavior.

## What the Spark accelerated

We timed the unchanged evaluator with identical CPU/CUDA input tensors.
Each configuration used three warmups and five timed repetitions, with CUDA synchronization.
The CPU reference is the faster of one and eight threads tested on the same Spark.

| Population | Float32 CPU time / CUDA time | Float64 CPU time / CUDA time |
| --- | ---: | ---: |
| 12 | 0.36 | 0.35 |
| 4,096 | 2.65 | 3.78 |

Small batches were faster on CPU. Large batches benefited from CUDA.
These are evaluator-throughput measurements, not end-to-end optimization speedups or general hardware claims.
CPU/CUDA score differences passed the predeclared absolute tolerances of 1e-3 for float32 and 1e-10 for float64.

The managed run also has measurable overhead versus direct optimizer calls.
Five paired trials preserved exactly the same evaluation counts and best scores.
Median managed/direct time ratios were 1.84 on local CPU and 1.50 on CUDA for Sphere.
For the synthetic compute objective, they were 1.33 and 1.38.
The proposed 10% overhead target is not met by these probes. The beta does not claim negligible bookkeeping cost.

## Limits and next work

The new interface supports scalar minimization, continuous box bounds, fixed repeats, and synchronous execution.
It does not add general constraints, multiobjective search, automatic retries, or distributed execution.
Raw observations remain in memory. Checkpoints are trusted Python payloads for a planned pause on the same device.

The bounded research loop selects among declared configurations and retains every trial.
General autonomous source editing and new EvoTorch-derived algorithms remain future work.
PGPE becomes useful when a high-dimensional application justifies it.

The next engineering study should optimize a feedback controller and include stronger uncertainty tests.
It needs a new evaluator version and fresh held-out scenarios. Do not tune the current task after seeing these losses.
A GPU flagship also needs enough parallel simulator work to outweigh scheduling and validation costs.

## Reproduce

From the checkout, install the development dependencies and run:

```bash
pip install -e '.[dev]'
pytest tests/ -q
python research/redesign/independent_beta_checks.py
python benchmarks/engineering_control.py --device cpu --output engineering-cpu.json
python benchmarks/engineering_control.py --device cuda --output engineering-cuda.json
python research/redesign/measure_overhead.py --device cuda
python research/redesign/measure_batches.py
```

The optional comparator needs the pinned EvoTorch checkout and its dependencies.
Add `--evotorch --evotorch-src /path/to/evotorch/src` to the study command.
The `results/` directory retains complete study records, rejected trials, performance samples, upstream provenance, and the final detached-job exit state.
