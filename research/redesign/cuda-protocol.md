# CUDA execution study

The target is the DGX Spark GB10. Keep the PyTorch API and optimize CUDA execution.
This phase tests the frozen thermal-control objective with prepared tensors and
compiled kernels. It does not change the optimizer, physical model, or score.

## Frozen comparison

- Use the existing `FrozenEvaluator` as the independent numerical reference.
- Keep its train scenarios, plant, quantization, penalties, and evaluator hash.
- Compare reference eager, prepared eager, and prepared compiled execution.
- Use float32 and float64, with populations 12 and 4096.
- Record preparation and first-call latency separately from steady-state latency.
- Warm each path five times. Rotate execution order across 15 timing samples.
- Synchronize CUDA before and after each timed evaluator call.
- Run CMA-ES through `SearchRun`, with 20 generations and seeds 11 through 15.
- Give each path the same population and exact candidate evaluation budget.
- Exclude optimizer construction from search timing, and state that scope.
- Record selected controls, reported scores, reference scores, and charged counts.
- Profile one warm evaluator call and one managed generation separately from timing.
- Run only one GPU measurement job at a time.

## Acceptance

Require reference parity on random inputs, saturation, quantization ties, and their
adjacent floating-point values. Use absolute score tolerances of 1e-3 for float32
and 1e-10 for float64, with zero relative tolerance. Check both scenario splits.

Keep the compiled path if one tested configuration achieves at least 1.10 times
the reference median end-to-end speed, with all five paired ratios above 1.0.
All runs must charge the exact budget. Selected scores must match the independent
reference and paired final scores must agree within the same tolerance.
Report every configuration, including losses. Report compilation cost and the
estimated number of searches needed to repay it. Do not claim CPU superiority
or general algorithm superiority from this within-GPU comparison.

The existing four objective probes must retain their results. A speedup cannot
repair the previously observed weakness against the PI domain baseline.

Stop after this implementation and one correction cycle if necessary. Record
failed attempts. Do not add a custom kernel unless profiling justifies it.

## Recorded correction

The default compiled path failed paired final-score parity for float32 with
population 4096. Three of five seeds differed by 0.0048 to 0.0178. Individual
objective scores passed tolerance. The other configurations passed paired parity.

The single correction enables eager precision casts and division rounding while
retaining CUDA graphs. It does not isolate which setting restores parity. Keep every acceptance criterion above unchanged.
First-call costs use one process in the declared configuration order. Later
configurations can reuse compiler work; these are not independent cold starts.
