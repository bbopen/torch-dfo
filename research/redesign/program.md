# torch-dfo redesign program

Status: beta validation. The thermal-control v2 evaluator is frozen. No application superiority claim is accepted.
Baseline: foundation correction commit `e446f68`.
Branch: `redesign/contracts-20260919`.

## Overall goal

Build a maintained PyTorch-native derivative-free optimization library for research and engineering, with an autonomous research lab in the same repository. Algorithms, tests, benchmarks, and application demonstrations are integral parts of the project. Preserve useful implementations while allowing a new architecture.

Use EvoTorch directly as an implementation source. Inspect, reuse, adapt, and improve suitable code instead of requiring independent reimplementation. Record the source revision, preserve source attribution, and credit EvoTorch in the README. Compare each adaptation against its upstream behavior and relevant edge cases.

The library remains the main product. The lab uses it to solve engineering problems, investigate new applications, and return tested improvements to the library. Autonomous hardware design and space engineering are important application directions alongside biology and control. See `purpose.md` for the owner's clarified intent.

Completion requires a minimal working implementation, independent contract checks, reproducible canonical and application evaluations, and an honest comparative outcome, including no measured advantage. The applications must exercise different objective and representation requirements. Performance or novelty claims require evidence beyond the library tests. A useful library beta does not depend on benchmark superiority.

## Active phase

Validate and package `0.11.0b1`. The run API supports CMA-ES, SHADE, and uniform random search.
The engineering reference uses the public run API, fixed scenarios, simple domain baselines, and optional EvoTorch CMA-ES.

Completion evidence consists of independent contract checks, full CPU/CUDA regression results, a fresh wheel installation, executable documentation, and a cost-accounted study.
The study retains its held-out losses. A small open-loop schedule that fits train scenarios is not evidence of robust control.

Keep implementation simple. Use one pending batch and direct optimizer calls. Add no registry, scheduler, or application dependency to the core.
The research loop currently performs three declared train-only configuration trials. General autonomous source editing remains future work.

See `beta-report.md` for results and limitations. The detached Spark job survives loss of the local SSH connection.

## Research limits

For the current pilot, start with two application adapters and three development iterations per adapter. These limits bound one experiment, not the library's future scope. Do not add a third application to rescue a failed result within that experiment. Freeze evaluators and task splits before search tuning. Record failed and discarded trials.

Prioritize one engineering or control reference application. Keep a second application to test a different representation or evaluation requirement after selection gates pass. MyoSuite and Design-Bench are candidates, not mandatory dependencies. Existing uses are prior art, not proof of an unmet need.

Separate canonical correctness, application validity, and total-cost performance. Keep raw calls, unique designs, noisy repeats, simulator steps, and validation cost distinct.

## Experiment rules

Before tuning, run the autoresearch probes: degenerate input, achievable ceiling, cheap exploit, and null baseline. A missing or inconclusive probe is not a pass.

Version and hash the evaluator, dataset split, task configuration, and measurement protocol. Change only the declared mutable implementation during each iteration. Record any evaluator correction outside the loop and invalidate direct comparisons across versions.

Accept a change only against a criterion fixed before its run. Near-tie or noisy results require confirmation. Compare against EvoTorch and appropriate gradient, domain, random/Sobol, or Bayesian methods.

## Durable record

Keep the accepted decision and protocol here. Store evidence outside the source tree until its provenance is verified. Keep user-facing reports in the task outputs directory. Preserve baseline code and results throughout the redesign.
