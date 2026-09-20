# NASBench201 quality and execution study

Freeze this protocol before search runs. Search quality and execution speed are
separate measurements. Neither measure establishes the other.

## Data and objective

Use EvoXBench commit `2f1ae28720a09fdf3e487bcd9de91433512fc106` and its
official NASBench201 database. Use all 15,625 encoded architectures, the three
datasets `cifar10`, `cifar100`, and `ImageNet16-120`, and fidelity 200.
The six edges each choose one of five operations in the upstream order.

Use raw classification error in percentage points. Search receives one sampled
validation trial per proposal, with replacement. Some records contain two trials
and others contain three. Respect each record's actual trial count.
Use `cifar10-valid` for CIFAR-10 validation and `cifar10` for its final test score.
Do not normalize against a Pareto front or combine error with resource costs.

Keep validation and test exports in separate files. Do not open the test export
inside search. After every search finishes, report the selected architecture's
mean recorded test error. Select it only by the lowest observed validation error.
Resolve equal validation errors by the earliest observation. Do not select from
test results or retrospective mean validation results.

Pin the archive, database, export, evaluator, protocol, and runner hashes in the
results. Preserve the data acquisition script and source manifest. Do not commit
the full database or score tables to this repository.

## Search quality

- Run seeds 0 through 29 on every dataset.
- Charge exactly 1,000 validation observations per method and seed.
- Charge repeated architectures, including repeats within one batch.
- Record unique architectures, repeats, decoder ties, and all failures.
- Use float64 and population 20 for all continuous solvers.
- Use six groups of five random keys in the box `[-1, 1]`, giving 30 coordinates.
- Decode each edge by its largest key. Before search, generate a uniform category
  priority permutation for each run seed and edge. Use it only to break exact
  maxima ties. Share it across methods and devices, and record tie counts.
- Freeze a private trial draw keyed by dataset, seed, architecture, and visit
  number. Keep this RNG independent from optimizer and decoder randomness.
  Assign visits in proposal order. Shared architecture visits get the same draw.

Run these methods on CPU and CUDA: torch-dfo CMA-ES, torch-dfo SHADE, stock EvoX
DE, and stock EvoX CMA-ES. Also run direct uniform categorical random search and
categorical aging evolution on CPU. Keep algorithm defaults except population,
dimension, bounds, device, dtype, and seed. EvoX CMA-ES uses mean zero and standard
deviation 0.6, matching the native initial scale. Its workflow clamps evaluated
keys to the box. Its latent update remains the upstream implementation.

Use aging evolution with population 100 and tournament size 25, as in Real et al.
Sample tournament members with replacement. Mutate exactly one uniformly selected
edge to a different operation. Remove the oldest population member each step.
Select the final candidate from the entire observed history.

Use EvoX commit `1c242cc9533fbd7e2e0c0342456f4e0f52449e5d` without algorithm patches.
Its DE samples differential indices with replacement. Treat these as distinct
implementations, not identical mathematical algorithms. Report an unsupported or
failed method with its consumed budget and error. Never convert it to a score or
speed result. Do not replace a failed comparator after inspecting its outcomes.

The quality runner uses the same exported host lookup for every method. Its total
runtime includes decoding, host lookup, transfers, and trace recording. This is
an integration measurement, not the stock Django query path or peak GPU speed.

## Separate CPU and CUDA performance panel

Use the same NASBench201 records, objective, private trial schedule, encoding,
population 20, float64, and budget of 1,000. Use seeds 0 through 4 on all datasets.
Run on the same DGX Spark, with one CPU thread and one measurement job at a time.
Record software versions, device, thread settings, and source hashes.

Preload recorded validation errors into tensors. Precompute the private trial
schedule and copy it to each device. Count visits on device, including duplicates
within a batch. Keep decoder priorities identical across execution paths.
This is an adapted device-resident EvoXBench scorer. It does not train networks,
measure inference latency, or claim acceleration of the stock database API.

Before timing, check every exported architecture and trial against the official
data. Verify CPU and CUDA lookup parity, repeated-query accounting, and decoder
parity on identical proposals. Run evaluator probes before optimization.

Measure three workloads separately:

1. Decode, trial selection, and lookup on identical fixed proposal batches.
2. Optimizer and managed-loop overhead using a trivial device-resident objective.
3. Full search with the device-resident NASBench201 scorer.

Compare CPU and CUDA within each method. Keep library comparisons separate from
algorithm-quality comparisons. Preserve native `SearchRun` costs in full torch-dfo
searches and normal `StdWorkflow` costs in EvoX searches. Audit traces after timing.
Do not add per-query host observers to the device-resident scorer.

Synchronize CUDA before and after each complete timed region. Warm the scorer
five times. Use a discarded search seed 419 to warm execution before measured
seeds. Rotate method and device order across seeds. Report setup, schedule build,
device staging, optimizer construction, and first-use costs separately.
EvoX DE may also use its supported compiled workflow. Report compile and first-use
cost separately from warm timing. Label eager and compiled measurements explicitly.
An eager comparison alone cannot establish either library's maximum performance.

CPU and CUDA RNGs need not generate identical proposals from the same seed.
Require scorer equality on identical inputs. Report final search quality separately;
do not require identical adaptive trajectories or claim they are paired identical work.

## Evaluation and stopping rule

Run fixture tests and score-blind decoder, duplicate, noise, and budget probes
before the full panel. A fixture is not a NASBench201 result. Preserve smoke runs
as smoke runs. Save every completed or failed run and its proposal trace.

Report every dataset and method. Summarize repeated-seed distributions, paired
quality differences against random and aging evolution, and raw timing samples.
Do not infer an advantage from one seed or remove a slow configuration.
Report implementation failures separately from poor optimization outcomes.

Complete one frozen panel and at most one implementation correction cycle.
Correct evaluator or accounting defects before accepting results, retain rejected
artifacts, and rerun every affected configuration. Do not tune algorithm parameters
from outcomes. A negative quality result or a GPU slowdown is a valid outcome.

## Sources

- [Pinned EvoXBench evaluator](https://github.com/EMI-Group/evoxbench/blob/2f1ae28720a09fdf3e487bcd9de91433512fc106/evoxbench/benchmarks/nb201.py)
- [Pinned EvoX DE](https://github.com/EMI-Group/evox/blob/1c242cc9533fbd7e2e0c0342456f4e0f52449e5d/src/evox/algorithms/so/de_variants/de.py)
- [Regularized evolution paper](https://ojs.aaai.org/index.php/AAAI/article/download/4405/4283)
