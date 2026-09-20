# torch-dfo redesign program

Status: active. Evaluators are not frozen yet. No optimization claim is accepted from this phase.
Baseline: foundation correction commit `e446f68`.
Branch: `redesign/contracts-20260919`.

## Overall goal

Build a maintained PyTorch-native derivative-free optimization library for research and engineering. Preserve useful algorithms while allowing a new architecture.

Completion requires a minimal working implementation, independent contract checks, reproducible canonical and application evaluations, and an honest advantage or no-go decision. The applications must exercise different objective and representation requirements. Performance or novelty claims require evidence beyond the library tests.

## Active phase

Diagnose Spark access and define the architecture and evaluation protocol before implementation.

Phase completion requires:

- A tested connection or a documented execution fallback with recoverable job results.
- One architecture decision with independent challenge and explicit rejected alternatives.
- Exact initial application tasks, data sources, dependencies, and evaluator limits.
- A four-probe evaluator audit plan with predeclared outcomes.
- A scoped implementation plan with distinct file ownership and acceptance checks.

Use a ten-minute diagnostic ceiling for the first connectivity investigation. Preserve the system's VPN profiles and routes unless a correction has a demonstrated cause. Use isolated remote workspaces and containers.

## Research limits

Start with two application adapters and three development iterations per adapter. Do not add a third application to rescue a failed result. Freeze evaluators and task splits before search tuning. Record failed and discarded trials.

The tentative applications are one published MyoSuite control task and a benign discrete biological design task from Design-Bench. Select exact tasks after inspecting availability and validity. Existing uses are prior art, not proof of an unmet need.

Separate canonical correctness, application validity, and total-cost performance. Keep raw calls, unique designs, noisy repeats, simulator steps, and validation cost distinct.

## Experiment rules

Before tuning, run the autoresearch probes: degenerate input, achievable ceiling, cheap exploit, and null baseline. A missing or inconclusive probe is not a pass.

Version and hash the evaluator, dataset split, task configuration, and measurement protocol. Change only the declared mutable implementation during each iteration. Record any evaluator correction outside the loop and invalidate direct comparisons across versions.

Accept a change only against a criterion fixed before its run. Near-tie or noisy results require confirmation. Compare against EvoTorch and appropriate gradient, domain, random/Sobol, or Bayesian methods.

## Durable record

Keep the accepted decision and protocol here. Store evidence outside the source tree until its provenance is verified. Keep user-facing reports in the task outputs directory. Preserve baseline code and results throughout the redesign.
