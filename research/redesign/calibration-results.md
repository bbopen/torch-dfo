# Calibration across optimization libraries

19 September 2026.

torch-dfo is an independent maintained PyTorch library. Relevant libraries across stacks supply code, workloads, and comparison baselines. Drop-in compatibility is not required.

## Executed comparison

The public calisim predator-prey task now runs through native torch-dfo APIs. The frozen study fits 16 annual observations and checks the selected parameters against the last five. Earlier exploration used all 21 observations. This is an exploratory temporal split, not fresh confirmation data.

Each of five methods completed five seeds at exactly 400 candidate evaluations. Lower MSE is better.

| Method | Median fitting MSE | Median later-period MSE | Median initialization and search, seconds |
| --- | ---: | ---: | ---: |
| torch-dfo CMA-ES | 14.14373 | 12.60300 | 0.17207 |
| EvoTorch CMA-ES | 14.14363 | 12.63765 | 0.17590 |
| Uniform random | 14.23548 | 12.73577 | 0.16813 |
| SciPy differential evolution | 14.14365 | 12.60614 | 0.17326 |
| torch-dfo SHADE | 14.14567 | 12.50383 | 0.17211 |

The optimizers reach similar fitting scores on this small problem. These numbers do not establish a general quality or speed advantage. Random search is also competitive within this narrow two-parameter box. Different later-period scores do not establish a reliable ordering across methods.

All methods use the same CPU objective and fitting-only incumbent selection. SciPy uses differential evolution. EvoTorch and torch-dfo CMA-ES share initial mean and standard deviation, but handle proposals at bounds differently. See the protocol for each configuration. Timings exclude imports, evaluator probes, and final checks. This serial SciPy objective makes no GPU claim.

## Verification and cost

- Focused benchmark and SearchRun tests: 25 passed on the Spark.
- All four evaluator probes passed before optimization. A test confirms that a constant scorer fails the probes.
- 10,000 search evaluations, 25 final checks, and 3 probe candidate evaluations.
- One additional simulation generated the synthetic probe fixture. Total study simulator calls: 10,029.
- Python 3.12.3, PyTorch 2.13.0+cu130, SciPy 1.16.3, NumPy 2.3.5, torch-dfo 0.11.0b1.
- Saved snapshot Python hashes match the final local source files. The benchmark also rejects imports from another torch-dfo checkout.
- Independent source review accepted the corrected probes, accounting, failure records, timing scope, and source checks.

The first launch passed 24 tests but failed its Git source check because of container mount ownership. It stopped before optimization. The retry used a setting limited to the mounted calisim checkout. Its complete logs remain in the local delivered evidence bundle under evidence/results-attempt-1.

The core library did not change in this increment. The README and roadmap now describe cross-stack learning and the native application direction. CPU CI installs SciPy so the new benchmark tests run.

## Next use of the evidence

Keep this as a regression and application example. Use a more demanding robot, mask, policy, or constrained design problem to select the next algorithm improvement. Choose references by the missing capability, including pycma, Nevergrad, pymoo, evosax, pagmo, NLopt, SciPy, and EvoTorch. The reference strategy records the initial areas to inspect.

Protocol fingerprint: `c36a3d03e63c552e6189f72a2e7f5fb2aa00ab8dc5ead6b7d425ebd60fbb6421`.
