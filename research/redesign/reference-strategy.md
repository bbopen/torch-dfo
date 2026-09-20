# Learning from derivative-free optimization across stacks

19 September 2026.

## Decision

Build torch-dfo as an independent, maintained PyTorch library for engineering and research. Learn from relevant derivative-free optimization libraries, systems, papers, and public applications across stacks. EvoTorch is one reference among several.

Directly reuse and adapt useful implementations. Prefer a small, tested adaptation when it solves the task cleanly. The originating API, framework, and feature catalog do not define torch-dfo's design.

Distinguish three uses of a reference. Its implementation can supply code. Its applications can supply real workloads. Its executable algorithms can supply comparison baselines. One reference need not serve all three purposes.

## Initial reference set

The official sources below establish relevant capabilities. This table selects areas to investigate; it is not a completed code audit or a promise to implement every feature.

| Reference | Stack and relevant capabilities | What to examine for torch-dfo |
| --- | --- | --- |
| [EvoTorch](https://github.com/nnaisense/evotorch) | PyTorch evolutionary search and neuroevolution | PGPE, ClipUp, vectorized objectives, public policy and mask workloads |
| [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) | Python numerical optimization with differential evolution | A practical baseline, budget and stopping behavior, vectorized or parallel evaluation |
| [pycma](https://github.com/CMA-ES/pycma) | CMA-ES with bounds, constraints, noise, and integer-variable support | Numerical behavior, covariance and step-size handling, restarts, noise treatment |
| [Nevergrad](https://facebookresearch.github.io/nevergrad/optimization.html) | Gradient-free optimization with parametrization and ask/tell execution | Mixed search spaces, optimizer selection, noisy objectives, application benchmark design |
| [pymoo](https://pymoo.org/algorithms/) | Single- and multi-objective optimization, including constrained methods | Constraint comparison, Pareto methods, representations, task-specific operators |
| [evosax](https://github.com/RobertTLange/evosax) | JAX evolution strategies with jit, vmap, and scan | Functional optimizer state, batched independent searches, compilation and accelerator measurements |
| [pagmo/pygmo](https://github.com/esa/pagmo2) | C++ optimization with asynchronous island execution and Python access | Global-search portfolios, parallel search experiments, engineering problems |
| [NLopt](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) | Native numerical algorithms, including derivative-free local and constrained methods | Local refinement and model-based DFO beyond evolutionary search |

Use specialist implementations and domain methods when the problem warrants them. This list is a starting set, not a boundary around the field.

## How an implementation enters the library

1. Select a useful task and the missing capability it exposes.
2. Inspect and pin the relevant source and numerical specification.
3. Extract the smallest useful implementation. Record provenance and retain source notices.
4. Characterize its behavior before changing its mathematics or stopping rules.
5. Adapt it to torch-dfo's device, random-state, budget, and result conventions.
6. Benchmark quality and complete evaluation cost. Report numerical and boundary-handling differences.
7. Keep improvements that survive independent review and the frozen application evaluation.

The native library remains simple. Do not add a universal adapter registry or reproduce each upstream configuration system.

## Active benchmark

The first repeated application comparison uses a calibration objective adapted from calisim. It compares native CMA-ES, SHADE, and random search with SciPy differential evolution and optional EvoTorch CMA-ES.

The evaluator counts actual candidate calls. Every method selects its incumbent using fitting data only. Final checks score the same selected candidate on the fixed later time segment. The series was exposed in earlier exploration, so the split is diagnostic rather than independent confirmation.

Use the same CPU environment for this SciPy simulator. It cannot establish GPU acceleration. Future tensor workloads must measure transfer, compilation, warm-up, memory, and complete search time across stacks.

Broader algorithm coverage follows observed application needs. These references inform the work; they do not impose API compatibility or a requirement to beat every baseline.
