# First EvoX and pycma comparison

September 20, 2026. Freeze this protocol before inspecting optimization results.

## Question and limits

Compare implemented CMA-ES workflows on a small canonical panel and one thermal
task. Measure evaluation efficiency and complete execution cost on DGX Spark.
This is a bounded comparison of specified configurations, not a ranking of entire
libraries or a claim that they have identical product goals.

Use EvoX 1.4.0, source `1c242cc9533fbd7e2e0c0342456f4e0f52449e5d`, and
pycma 4.5.0, source `48a821b166eceef544847a0b9b66113ac26f1530`.
Install them in an isolated benchmark dependency directory without replacing Torch.

## Fixed panel

- Four 16-dimensional tasks: shifted sphere, rotated ellipsoid with condition
  number 1e6, rotated Rastrigin, and the existing frozen thermal train objective.
- Canonical tasks use shift `linspace(0.5, 1.5, 16)` and a CPU float64 orthogonal
  rotation generated with seed 20260920. Save its identity in the result.
- Population 32, budget 3,200 candidate evaluations, seeds 11 through 15.
- Float64 throughout. CMA-ES starts at mean zero and absolute sigma 0.6.
- Disable active covariance, mirroring, and restarts where configurable. Preserve
  implementation-specific adaptation and decomposition behavior, and disclose it.
- Compare torch-dfo CPU, torch-dfo CUDA, EvoX CUDA, pycma CPU, and pycma with a
  CUDA objective. pycma's optimizer remains on CPU in both cases.
- Include native uniform random search on CPU in the box [-2, 2].
- Use all 3,200 evaluations even after reaching a target. Record first target hit.
- Targets: 1e-8 for sphere, 1e-4 for ellipsoid and Rastrigin, and the thermal
  evaluator's existing `PASS_THRESHOLD`. These targets do not define scientific success.

CMA-ES uses unbounded latent requests. Native torch-dfo requires a finite box, so
use a guard radius of 1e6 and assert that no evaluated candidate reaches it.
Its relative sigma is `0.6 / 2e6`. Thermal actuator saturation and quantization
remain inside the shared evaluator. This differs from the earlier bounded thermal
search and must not be compared directly with its final scores.

## Correctness before timing

Check known optima, the shifted origin, a partially corrected candidate, and a
fixed random batch before search. Compare canonical CPU and CUDA scores with
absolute tolerance 1e-10 and relative tolerance 1e-12. Ellipsoid scores can be large.
Use absolute tolerance 1e-10 and zero relative tolerance for thermal scores. Preserve the thermal evaluator identity and existing probes.

An independent counter records actual candidate batches and the best evaluated
candidate for every method. Never substitute a final population or distribution
mean for that incumbent. Reject over-budget, malformed, or non-finite evaluations.
Native `SearchRun` must finish normally with matching attempted, completed, and
charged counts and no failures. Count EvoX initialization as a generation if it
evaluates a population. Record failed runs instead of dropping them.

Audit every selected candidate with the CPU reference after search. Thermal
validation uses the existing held-out split once per selected candidate. That split
has already been seen; this is a diagnostic check, not new confirmation data.

## Timing and interpretation

Use eager optimizer execution and eager objective tensors, with cached thermal
scenario tensors on each device. No full-workflow compilation comparison is claimed.
EvoX can compile internally through `torch.cond`; include that cost in the run.
Record initialization, first generation, remaining generations, total elapsed
time, target-hit time, and CUDA allocated-memory measurements. Synchronize GPU
timing boundaries. Include observer, transfer, and incumbent-tracking costs.
Use one CPU thread and rotate method order across seeds. Run one GPU job at a time.

Success for this phase means reproducible, complete, independently reviewed
results. A torch-dfo win is not required. Keep the evaluator, seeds, and budgets
fixed. Do not tune any method after seeing these results. Stop after one completed
panel and at most one adapter correction cycle; preserve failed setup attempts.

## Upstream implementation check

The isolated CPU and CUDA probe reproduced a scalar-dot-product error in EvoX's
rank-one covariance update before search. See `probe_evox_cma.py` and its saved
result. Keep EvoX unmodified and label its results as the observed 1.4.0 implementation. It cannot serve as a validated mathematical CMA-ES reference.
Do not present wins against that implementation as evidence of a better algorithm.
pycma supplies an independent comparison. Investigate any further failure openly.
