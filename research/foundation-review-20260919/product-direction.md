# A direction for torch-dfo

19 September 2026. This is a research decision, not a performance or scientific novelty claim.

## What it should become

Build a maintained PyTorch library for bounded search when useful gradients are unavailable, unreliable, or incomplete. Make batched evaluation, strict resource budgets, reproducible continuation, and domain constraints part of the design.

Keep torch-dfo's algorithms and phased-search lineage where they earn their place. Use EvoTorch as a reference implementation and a baseline. Support adapters where an existing implementation is stronger. A new algorithm catalog alone would add little.

The central product hypothesis is that researchers need less work between a difficult objective and a trustworthy optimization experiment. The library must still improve optimization outcomes. Better experiment bookkeeping alone would justify a supporting package, not a new optimizer claim.

A rewrite is premature. The present code contains useful algorithms and tests, but review found failures in budget, state, validation, and evaluation contracts. Correct those first. Separate objective evaluation from search policy before adding more phases to PhasedDFO.

## What EvoTorch already established

EvoTorch's premise is scalable evolutionary computation for high-dimensional problems, built on PyTorch with GPU and parallel execution. Its scope includes reinforcement learning, engineering design, and industrial optimization. That is already close to the original torch-dfo premise. [EvoTorch paper](https://arxiv.org/abs/2302.12600).

The inspected source includes CMA-ES, PGPE, xNES, SNES, CEM, genetic algorithms, multi-objective selection, quality-diversity search, and distributed evaluation. Its functional API supports batched and vectorized use. The inspected release is 0.6.1 from May 2025. This establishes a maintenance opportunity, not abandonment or a technical advantage for torch-dfo. [Repository](https://github.com/nnaisense/evotorch), [releases](https://github.com/nnaisense/evotorch/releases).

The downstream audit found eight distinct repositories with configured EvoTorch calls. They cover simulator calibration, surrogate acquisition optimization, discrete graph search, inverse kinematics, model tuning, explanation masks, muscle-driven control, and game heuristics. Each has a pinned source link in [the usage audit](evotorch-downstream.md). None was executed in this review. Source integration does not prove deployment or a measured benefit.

Several users write custom evaluators, representations, operators, and result checks around EvoTorch. This is useful design evidence. It is not proof that EvoTorch cannot support those tasks.

## Where derivative-free search has a sound case

DFO is useful when the objective offers values but no useful derivative. Examples include discrete choices, thresholded rewards, external simulators, failed simulations, and hybrid systems with discontinuous behavior. It can also help explore multiple basins before local refinement.

GPU acceleration needs enough parallel work. It can come from evaluating many candidates, many scenarios per candidate, or several searches together. The optimizer running on CUDA is insufficient if evaluation remains a serial Python simulation.

There is no broad category that only DFO can solve. Discrete optimization, Bayesian optimization, reinforcement learning, differentiable relaxations, and hybrid methods often compete. Choose baselines according to the problem, not the library's identity.

| Situation | First comparison | Role for torch-dfo |
| --- | --- | --- |
| Smooth tensor objective with reliable gradients | Adam or L-BFGS, multiple starts | Global search or hybrid exploration must earn its extra evaluations |
| A small number of expensive simulator calls | Bayesian optimization and a strong domain method | Batch scheduling or robust constrained search if it improves quality per call |
| Cheap, discontinuous batched objectives | EvoTorch, differential evolution, random and Sobol search | Strong candidate for GPU population search |
| Discrete biological or engineering designs | Domain operators, discrete baselines, surrogate methods | Representation-aware variation and independent candidate validation |
| Noisy control or calibration | PGPE/CMA-ES, RL or system-identification baselines | Repeated scenario evaluation, uncertainty handling, robust objectives |

BoTorch already supports parameter and outcome constraints. An expensive-evaluation workflow should compare against it where appropriate. [BoTorch documentation](https://botorch.org/docs/constraints).

## Proposed architecture

Keep tensor `ask` and `tell` as the low-level interface. Make the following responsibilities explicit before expanding the public API.

| Responsibility | Required behavior | Current status |
| --- | --- | --- |
| Search state | Candidate generation, update, independent RNG, versioned checkpoint | Existing implementation; reviewed corrections cover several lost-state cases |
| Representation | Encode/decode, variable types, bounds, domain variation | Basic mixed-space support exists; ordinal categorical encoding has limitations |
| Evaluation | Candidate identity, fitness, constraints, failure status, measured cost | Needs an explicit result contract; scalar fitness alone loses information |
| Execution | Batch sizing, device selection, cancellation, retries, checkpoints | Local tensor batches exist; distributed execution should remain optional |
| Search policy | Restarts, populations, portfolios, hybrid local refinement | Existing lineage to retain and measure; extract policy from evaluator mechanics |
| Experiment record | Seeds, hashes, costs, feasibility, incumbent trace, independent verification | Research scripts demonstrate pieces; consolidate only after contracts are tested |

Do not make Ray, a simulator, or a biology package a core dependency. Keep application packages separate. Support EvoTorch and pycma through adapters for fair comparisons before attempting to reimplement every algorithm.

Constraints need first-class results, not only a hidden penalty coefficient. Noisy results need replicate counts and uncertainty. Multi-objective search needs a defined Pareto archive and metrics. These are proposed capabilities, not claims about today's package.

For very high-dimensional parameter search, test separable or low-rank strategies against full covariance. A GPU memory ceiling does not establish optimization quality. Add PGPE or another strategy only when a reference application demonstrates a need that current methods cannot meet.

## Application program

Start with two application families. Their different failure modes will test whether the shared library is useful.

### Robust musculoskeletal control and calibration

MyoSuite provides muscle-driven simulation with tasks and model variations, including fatigue and other physiological changes. EvoTorch already appears in a public MyoChallenge training script. This makes it a credible reference application, not a new field invented for torch-dfo. [MyoSuite tasks](https://myosuite.readthedocs.io/en/latest/suite.html), [verified training integration](https://github.com/PKU-MARL/MyoChallenge/blob/feef99336d2b09f6b8a57462ade30b3902017512/evotorch/train.py#L7-L87).

First task: optimize a compact controller for one published MyoSuite task across varied fatigue and model parameters, under explicit failure constraints. Treat simulator calibration as a separate later application. Hold out model variants and disturbances. Measure successful-task rate, constraint violations, worst-tail performance, simulator steps, and total time.

Compare EvoTorch PGPE/CMA-ES and suitable RL or calibration baselines. Include gradient methods when the chosen model provides useful derivatives. First reproduce an existing task. Then define a harder robust variant and check the literature before describing it as novel.

The GPU case remains unproven until the selected simulator or batched surrogate runs efficiently on the Spark. Parallel CPU rollouts and GPU policy inference are different costs. Simulation success would not establish patient benefit or clinical suitability.

### Discrete biological design with independent evaluation

Design-Bench provides standardized offline model-based optimization tasks derived from biology, materials, and robotics. Its sequence-design tasks make a useful starting point. Optimizing a learned predictor is easy to mistake for improving the underlying science. [Design-Bench paper](https://proceedings.mlr.press/v162/trabucco22a.html), [reference code](https://github.com/brandontrabucco/design-bench).

Proposed challenge: find a diverse set of valid sequences under mutation or composition constraints, using a fixed training dataset and a sealed evaluator. Measure independent scores, feasibility, diversity, duplicate rate, and distance from the training distribution. Record surrogate evaluations separately from expensive oracle queries.

Compare domain mutation, random search, EvoTorch, and published offline-design baselines such as the verified DynAMO integration. Test surrogate uncertainty and disagreement instead of rewarding the largest predicted score. [DynAMO CMA-ES implementation](https://github.com/michael-s-yao/DynAMO/blob/52dfc8fe4cb4de6374635bc5f971974f027d4d1f/src/dynamo/optim/cmaes.py#L28-L118).

A learned benchmark oracle is still a model. A high score does not establish a new functional molecule. Claims must name the assay, lookup data, simulator, or model that actually evaluated the candidate.

### Further research candidates

These are hypotheses to investigate after the first application gates, not a committed feature list.

- Artificial-life pattern discovery under perturbation and resource limits. Batched simulation could fit GPU population search. Require diversity and persistence, not only an attractive image score. The ASAL++ repository contains an experimental PyTorch/EvoTorch path, while its reported experiments use JAX. [Source](https://github.com/fredericowieser/ASALPlusPlus/blob/16a0f918251bfa0f8deaf7e25c23ab40586a5a86/README.md).
- Hybrid biological-model calibration with discrete model choices and continuous rates. Test parameter recovery on synthetic ground truth, then predictive error on held-out observations. Check identifiability before celebrating a low fitting loss.
- Co-design of morphology and control. Mixed discrete structure and continuous control can motivate domain operators and constraint handling. Reproduce a published benchmark before changing both model and evaluator.
- Robust schedules in mechanistic disease simulators. Keep work at the simulation-method level. Optimize across uncertain parameters and compare against established control methods. This review establishes no clinical use or new treatment.

Novelty must be earned by a precisely defined unsolved subproblem, a literature search, independent evaluation, and a reproducible result. An unfamiliar combination of tools is not enough.

## Evaluation program

1. Run contract checks before quality benchmarks. Budgets, tensor shape, finite inputs, checkpoint continuity, failure handling, and objective direction must be correct.
2. Use official COCO instances and targets for continuous numerical calibration. Keep classical functions as smoke tests. Include rotated, ill-conditioned, multimodal, and noisy families.
3. Report best-so-far quality versus both evaluations and elapsed time. Separate cold start, compilation, objective, optimizer, transfer, and validation costs. Synchronize GPU timings.
4. Use random/Sobol baselines and specialist competitors. Match bounds, initialization, dtype, seed protocol, budget, and termination. Explain algorithmic differences that cannot be matched.
5. Use several seeds and held-out instances. Publish all failures, raw traces, source hashes, device details, and uncertainty. Do not pick the best seed after seeing the result.
6. Evaluate application feasibility and independent quality. A generic BBOB win cannot validate a biological design or a controller.

A proposed first application study has a ceiling of two adapters, three development iterations per adapter, and ten confirmation seeds per selected task. Freeze task definitions and acceptance thresholds before confirmation. Stop feature expansion if the shared abstraction does not simplify both applications.

Choose success metrics with each application. A useful release should match a strong baseline's validated quality with lower total cost, or improve quality at the same cost. Report confidence intervals and practical effect size. Do not choose thresholds after seeing results.

## Recursive optimization

An outer optimizer can tune inner search settings, operators, or portfolio schedules. This is a valid DFO application, but it is also a route to benchmark overfitting.

Keep outer training, selection, and confirmation sets separate by problem family and instance. Freeze evaluator code. Tune on declared settings only. Count every inner evaluation, failed trial, and confirmation call in the outer experiment's cost. Compare outer DFO with random/Sobol search and a fixed expert default.

Earlier work in this session found a promising tuned SHADE configuration on held-out BBOB instances. It did not solve those targets or establish a universal default. The total search cost was substantial. Adoption needs broader family transfer and an amortized-cost comparison.

Use autoresearch to propose and test bounded changes. It must not change the evaluator, hide failures, or retune on confirmation results. Recursive improvement is a testable research program, not a guarantee of self-improvement.

## Decision

Continue torch-dfo as a focused, maintained library. Preserve the working lineage, repair its contracts, and use independent application evaluations to decide which features deserve a permanent API.

The next convincing result is a reproducible, useful application comparison with full cost accounting. It may involve a hybrid method. It need not claim that derivatives are always worse or that the whole scientific problem was previously unsolved.
