# Scope and stepping stones

19 September 2026. This roadmap governs the next development stages. It is a plan, not a statement of implemented capabilities.

## Destination

Build a maintained PyTorch derivative-free optimization library with an autonomous research lab in the same repository. The library provides reusable methods. The lab solves engineering problems, evaluates methods, and returns tested improvements to the library.

Keep the ambition broad. Keep each implementation and experiment bounded. A frontier result is a research milestone, not a prerequisite for releasing a useful library.

torch-dfo owns its API and roadmap. Existing DFO libraries across stacks are sources of reusable implementations, workflows, and benchmark baselines. See `reference-strategy.md`. Public projects identify real tasks to support. Drop-in compatibility and replication of EvoTorch abstractions are not product goals.

## Starting point

The foundation correction batch is preserved at `e446f68`. It passed 1,096 local tests at 92.19% coverage and a 509-test Spark CPU/CUDA subset. These results apply to that batch, not the proposed architecture.

The `0.11.0b1` candidate implements scalar synchronous `SearchRun`, fixed repeats, checkpoints, and CMA-ES, SHADE, and random-search execution. See `beta-report.md` for the tested scope and environment. The thermal study did not show a held-out advantage over its PI-derived baseline. General autonomous code editing remains unimplemented.

Spark execution has a tested detached-job path with saved exit status. The Mac's intermittent Tailscale stop cause remains unresolved. Use detached execution and recoverable results; keep network diagnosis outside the library's critical path unless jobs themselves fail.

## Stepping stones

| Stage | Deliverable | Evidence required to advance |
| --- | --- | --- |
| 1. Working foundation | SearchRun, tensor evaluator, CMA-ES and random-search adapters, basic experiment records | Public-interface tests prove budgets, candidate identity, failure accounting, planned-pause checkpoints, and CPU/CUDA behavior. One saved run can be reproduced. |
| 2. Useful algorithm coverage | Bring SHADE through the same contracts; adapt EvoTorch PGPE and guarded ClipUp when the high-dimensional control case needs them | Characterization tests match intended upstream behavior. New edge-case tests prove each correction. Matched comparisons expose memory, quality, and total cost. |
| 3. Engineering reference study | One complete engineering or control application through the public interface, plus canonical numerical evaluation | An independent evaluator checks saved candidates. Baselines share the task, representation, initialization protocol, and declared cost. Failures remain in the report. |
| 4. Maintained library beta | Installable package, examples, reference documentation, support matrix, API change guidance, and CI | A fresh environment can install the package and reproduce documented examples. Supported behavior passes independent review and CPU/CUDA checks. Publish comparative results, including losses. |
| 5. Autonomous research lab | A bounded propose/run/measure/keep-or-revert loop over frozen evaluations | The loop reproduces a baseline, rejects bad changes, records interruption and every trial, and tests selected changes on held-out tasks. Correctly reporting no improvement is a pass. |
| 6. Frontier study | One difficult, precisely defined engineering research question | Literature review, strong baselines, a frozen protocol, independent validation, and replicated results support the stated claim. A negative finding is an acceptable research outcome. |

Research question selection can begin earlier. Each study must pass its evaluator gate before tuning. Stages do not require separate rewrites or disconnected implementations.

## First beta scope

The beta should offer a coherent path from an objective to a reproducible result.

- A simple minimize function and the same underlying ask/tell interface.
- Batched PyTorch objectives and a CPU evaluator path for external simulations.
- Bounded continuous search, explicit minimization semantics, and validated inputs.
- CMA-ES, SHADE, and random search through the managed run interface. Add PGPE when a selected application requires it. Keep Sobol as a possible baseline.
- Explicit evaluation caps, pending-batch rules, failure records, and stop reasons.
- Independent search and evaluator randomness, with reproducible scenario schedules.
- Quiescent checkpoints for planned continuation. Interrupted execution must not silently replay unknown work.
- Optional repeated evaluations with raw per-repeat values and a declared aggregate.
- Run manifests, evaluator fingerprints, traces, source versions, and cost records.
- Documentation that names supported devices, dtypes, representations, and method limitations.

Preserve existing public algorithms while the new path is evaluated. Do not route users onto an incomplete replacement by default. Decide migration and deprecation from compatibility tests and actual use.

Constraint fields in a result object do not establish constrained optimization support. Repeated evaluations do not establish adaptive noisy optimization. State these differences in the support matrix.

## Capability sequence

| Capability | Initial implementation | Next step and evidence gate |
| --- | --- | --- |
| Continuous box bounds | Required | Verify sampling, transforms, and candidate identity on CPU/CUDA. |
| General constraints | Record raw constraints; reject unsupported algorithm use | Add a declared constraint-aware selection policy for the first engineering task that needs it. Test feasibility, retained objective values, and algorithm-specific adaptation. |
| Noisy objectives | Fixed repeats, explicit aggregation, common scenario seeds, raw records | Add adaptive resampling or noise-aware selection only after a noisy reference study demonstrates the need. |
| Integer and categorical decisions | Retain the legacy adapter with its stated limits | Add direct representation-aware mutation or sampling. Compare against encoded search and count unique designs. |
| Large parameter vectors | Measure existing methods; keep dense covariance limits visible | Adapt EvoTorch diagonal PGPE/ClipUp for the selected control policy. Verify ties, zero gradients, bounds, sigma updates, and checkpoints. |
| Multiple objectives | Outside the initial scalar interface | Add Pareto selection and archive metrics with a concrete trade-off study. |
| Quality diversity | Research candidate | Require an application where diverse valid solutions are an explicit outcome. |
| Asynchronous or distributed search | Detached complete experiments only | Add in-flight evaluation recovery and worker scheduling after local execution shows a measured bottleneck. |

General constraints, noisy selection, and discrete operators are explicit next capabilities. They are not permanent exclusions. Prioritize them according to the first application rather than implementing all three speculatively.

## Applications and evaluations

Maintain three distinct kinds of evidence in the repository.

**Canonical evaluations** establish correctness and numerical behavior. Use official COCO identities and targets. Include random/Sobol and specialist baselines. Keep classical functions as fast controls. Report results by evaluation count and wall time.

**Engineering applications** establish usefulness. Prioritize control and hardware-related problems. The current source-backed candidates include MyoSuite Baoding P2 as a CPU control reference and MJX hand reach as a possible GPU evaluation path. Choose one initial study after a bounded runtime and reproducibility check. Do not claim the two tasks have identical semantics or accelerator behavior.

**Exploratory studies** test a specific new hypothesis. Candidate directions include robust hardware/controller co-design, designs under uncertain loads or environments, and resource planning. Biology remains in scope. Exact applications need accessible data, a meaningful objective, and an independent evaluator before becoming committed studies.

TFBind8 remains a possible representation and oracle-accounting control. Its current data availability and lookup-oracle limitations make it unsuitable as the first GPU flagship. It is not a dependency for core development or the engineering study.

The first application-selection cycle should inspect no more than three candidates. Select using objective availability, reference implementation, practical relevance, measured runtime, and a credible role for DFO. Record rejected candidates and reasons. Do not choose a task because torch-dfo happens to win on it.

For an application claimed to benefit from GPU acceleration, measure the full objective and search. A tensor optimizer around serial simulation is not enough. Include gradient or domain methods when they have useful access to the same problem.

## Implementation reuse strategy

Treat useful source from any relevant stack as reusable engineering work. Select implementations by the task and evidence, not their framework. For each candidate method:

1. Pin the source revision and identify the smallest useful implementation.
2. Preserve provenance and add characterization tests before changing behavior.
3. Adapt the code to our state, budget, device, and result contracts.
4. Correct demonstrated weaknesses with regression tests.
5. Measure parity and improvements against the upstream implementation.
6. Credit the source in code and the README.

The PGPE/ClipUp review provides the first concrete candidate. It identified the sampling, ranking, sigma update, and center-update code needed for a compact adaptation. The isolated ClipUp kernel produces NaNs for a zero gradient; the adapted rule must remain finite. Finite zero fitness is valid and must not be rejected.

Avoid a framework-wide graft before the smaller adaptation is evaluated. Direct code reuse is encouraged; importing unrelated execution machinery is not a prerequisite.

## Research lab scope

Start the record system during stage 1. It should save configuration, source and evaluator hashes, seeds, counts, timings, raw failures, and results. It needs a command-line entry point and readable reports. A dashboard is optional later work.

Begin automation with experiment execution and reporting. Add autonomous proposal and selection only after a manual baseline is reproducible and its evaluator passes the four autoresearch probes.

The first autonomous loop may tune one algorithm configuration or compare one implementation change. Limit the initial development cycle to three predeclared changes, then confirm the selected result on held-out instances. Count the entire outer search cost when evaluating recursive optimization.

The agent may change declared implementation or configuration files. It must not change the evaluator, test split, or acceptance rule within the same experiment. Failed and rejected trials remain in the ledger. On interruption, preserve the ledger and charge uncertain work conservatively. Mark an unresolved run terminal; any new run keeps its lineage and cost history. This does not promise transparent recovery of in-flight evaluations.

The lab passes its functional gate when it makes correct decisions, including rejecting every proposed change. A measured improvement is a research result, not a required output of every loop.

## Immediate work package

The calibration comparison is complete. Its repeated, equal-budget results are in
`calibration-results.md`. The first CUDA execution study is also complete. Its
compiled thermal scorer passes paired quality and speed checks on DGX Spark.
See `cuda-results.md` for setup cost, measured scope, and rejected attempts.

1. Use the CUDA profile to remove avoidable host synchronization from optimizer execution. Preserve checkpoint state, numerical behavior, and evaluation accounting. Measure each change separately.
2. Expand the robot task with GPU batches and an upstream baseline. Preserve independent position and orientation checks. Match incumbent selection and account for different boundary handling.
3. Use a released mask or policy workload to guide a compact PGPE implementation. Add only the capabilities that the workload demonstrates.

The next CUDA change must retain the current reference as a baseline. Repeat
paired searches and include setup cost. Keep runtime and application dependencies
outside the core library.

## Discrete architecture-search benchmark

EvoXBench is the selected next benchmark for discrete architecture search. Start with NASBench201 and a fixed scalar objective under counted evaluation budgets.
Compare against random search and a categorical evolutionary baseline. Keep validation scores for search and test scores for final assessment.
Record duplicate architectures, evaluator randomness, and unique evaluated designs. Add Pareto search only with a separate multi-objective protocol.

EvoXBench uses stored results or predictive evaluators. It does not establish CUDA acceleration or network latency on DGX Spark.
Measure selected networks on GB10 in a separate deployment study. Keep benchmark data and optional dependencies outside the core package.
This is planned work and does not block merging the foundation beta.

## Maintenance and decision rules

Keep one active implementation milestone and one bounded application-selection lane. Use independent review for meaningful numerical and state changes. Commit working increments and keep the verified baseline available.

Promote a feature when it has a clear public interface, relevant tests, documentation, and a reproducible use case. Keep experimental features visibly experimental until they meet that gate.

Release the beta for reliable, documented usefulness. A speed claim needs repeated end-to-end measurements. A quality claim needs matched costs and held-out confirmation. A frontier claim needs a precise literature-supported gap and independent scientific or engineering evaluation.
