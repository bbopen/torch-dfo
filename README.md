# torch-dfo

A PyTorch derivative-free optimization library for engineering and research.

[![CI](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml/badge.svg)](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-dfo)](https://pypi.org/project/torch-dfo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Optimize bounded objectives that do not provide usable gradients. The core package requires PyTorch and supports CPU, CUDA, and MPS tensors.

## When to use it

Use derivative-free search for discrete decisions, discontinuous scores, external simulators, or objectives with unavailable or unreliable gradients. Batched tensor objectives can benefit from a GPU. Small populations and CPU simulators may not.

For smooth objectives with reliable gradients, compare against autograd-based methods. For expensive evaluations, compare against Bayesian optimization. The best method depends on evaluation cost and solution quality.

torch-dfo develops its own API, algorithms, and research tools. Application needs guide their design. The core depends only on PyTorch.

We study derivative-free optimization across Python, JAX, and native numerical libraries. EvoTorch, EvoX, SciPy, pycma, Nevergrad, pymoo, evosax, pagmo, and NLopt are candidate implementation and benchmark references. Completed cross-library studies currently cover SciPy and EvoTorch. Public applications supply real workloads. See [the reference strategy](research/redesign/reference-strategy.md).

As of September 20, 2026, EvoTorch's latest default-branch commit and release were dated May 14, 2025, about 16 months earlier. See its [latest commit at review](https://github.com/nnaisense/evotorch/commit/cebcac4f20979078becf8b908016cd8e5e6714a4) and [v0.6.1 release](https://github.com/nnaisense/evotorch/releases/tag/v0.6.1). Maintained PyTorch support is a torch-dfo priority; performance claims require separate benchmarks.

Use task results, evaluation cost, and runtime to judge an implementation. Compatibility with another library's API is not a design requirement.

## Install

This branch contains the `0.11.0b1` beta candidate. Install it from the checkout:

```bash
pip install -e .
```

Requires Python 3.10 or later and PyTorch 2.4 or later. Install a PyTorch build that supports your accelerator before selecting CUDA or MPS.

## Budgeted search

```python
from torch_dfo import CMAES, minimize

optimizer = CMAES(dim=10, bounds=(-5.0, 5.0), device="cpu", seed=42)
result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
print(result.best_value, result.charged_evals)
```

The objective receives a batch and returns one scalar per candidate. Lower is better.
`minimize` uses `SearchRun` to count evaluations and stop at the budget.
CMA-ES and SHADE require full generations. Random search can use the final partial batch.

Use `SearchRun` for manual ask/tell, fixed repeats, raw observations, and planned checkpoints.
See [the run guide](docs/runs.md). Existing optimizer-level ask/tell interfaces remain available.

Select `device="cuda"` when available. Use `dtype=torch.float32` on MPS.
GPU speed depends on the objective and population size.

## Algorithms

| Algorithm | Mechanism |
| --- | --- |
| `RandomSearch` | Uniform bounded search with a private random generator |
| `CMAES` | Full covariance adaptation with restart support |
| `SHADE` | Differential evolution with adaptive parameters |
| `NelderMead` | Simplex-based local search |
| `PhasedDFO` | SHADE exploration, CMA-ES search, and local polishing under an evaluation budget |
| `DLRPortfolio` | Multiple search branches with diagonal-plus-low-rank covariance approximations |

The optimizers expose `ask()` and `tell()`. Population sizes, budget handling, and supported parameters differ. Check the class documentation before switching algorithms.

`PhasedDFO` includes finite-difference local polishing. Its calls count against the evaluation budget; it does not use autograd.

## Model parameter search

`DFOOptimizer` supports the `shade`, `cmaes`, and `nelder_mead` algorithms. Each step evaluates several candidate parameter vectors. This can cost much more than one gradient step.

```python
import torch
from torch_dfo import DFOOptimizer

model = torch.nn.Linear(2, 1)
x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
y = torch.tensor([[1.0], [1.0]])
optimizer = DFOOptimizer(
    model.parameters(), algorithm="cmaes", bounds=(-2.0, 2.0), budget=100, seed=42
)

def closure():
    return (model(x) - y).square().mean()

while not optimizer.is_exhausted:
    loss = optimizer.step(closure)
```

Bounds are absolute parameter limits. The wrapper derives device and dtype from the model. A final budget residue can remain unused if a complete generation will not fit. Use a batched closure when the objective can evaluate candidate models together.

## Checkpointing

`SearchRun.state_dict()` saves a completed batch boundary. `SearchRun.from_checkpoint()` restores the search as a new run with parent lineage.
The beta uses trusted Python checkpoints on the same device. See [the checkpoint example](docs/runs.md#planned-checkpoints).

Optimizer-level `state_dict()` and `load_state_dict()` remain available. Their cross-device behavior can reinitialize the random generator.

## Accelerator and compilation limits

Tensor operations can remain on the selected device. Python control flow and scalar extraction can synchronize with the host. The package does not provide a fully asynchronous GPU search loop.

Optimizer compilation tests cover selected methods with graph breaks. They do not establish full-graph compilation or a general speedup. Measure objective time, optimizer time, and total time separately.

The thermal-control example has an optional compiled CUDA scorer. On one DGX Spark GB10 study, warm CMA-ES runs were 1.28 to 1.68 times faster at equal budgets and matching final reference scores. Compilation adds setup cost. These results apply to this workload. See [the CUDA study](research/redesign/cuda-results.md) and its reproduction command.

Historical memory-capacity measurements appear in [the benchmark notes](docs/benchmarks.rst). A dimension that fits in memory is not evidence that an optimizer can solve a problem at that dimension.

## Evaluation status

The benchmark harness supports COCO/BBOB, YAHPO, and Gymnasium. COCO's official target flag determines success. The harness does not estimate global optima with a local solver. It reports unknown precision as `null` in JSON.

The former README reported 14 of 16 classical problems solved. That historical claim lacks a pinned, independently verified result bundle here. It is not the current acceptance criterion. Classical examples are useful smoke tests; they do not establish general performance.

The YAHPO runner uses one torch-dfo seed and several random seeds. Its mean comparison is descriptive. Equalize repeated runs before making superiority claims.

Run bounded, seeded comparisons against random or Sobol search, EvoTorch, pycma, and suitable gradient or Bayesian methods. Report evaluation counts and wall time. Keep failures in the results and reserve unseen problem instances for confirmation.

## Structured search

`SearchSpace` supports `Float`, `Int`, and `Categorical` parameters. `PhasedDFO` can decode these parameters into individual trials.

```python
import torch
from torch_dfo import Float, Int, SearchSpace, PhasedDFO

space = SearchSpace([Float("scale", 0.1, 2.0), Int("count", 1, 8)])
search = PhasedDFO(space=space, budget=100, seed=42)
def objective(trials):
    return torch.tensor([
        (trial["scale"] - 1.0) ** 2 + abs(trial["count"] - 4)
        for trial in trials
    ], dtype=search.dtype, device=search.device)

best_encoded, best_score = search.optimize(objective)
```

Integer and categorical decoding does not replace domain-specific mutation or feasibility rules. General constraints and multi-objective search need further design and evaluation.

## Research lab

The [calibration benchmark](benchmarks/calibration.py) adapts a public predator-prey simulation task.
It compares native torch-dfo methods with SciPy differential evolution and optional EvoTorch CMA-ES.
The [protocol](research/redesign/calibration-protocol.md) specifies candidate budgets, seeds, evaluator checks, and an exploratory temporal split.
This CPU simulator benchmark does not measure GPU acceleration. Its dependencies stay outside the core package.

The repository includes a quantized thermal-control reference study.
It compares CMA-ES, SHADE, random search, and simple domain baselines on fixed train and held-out scenarios.
An optional EvoTorch baseline runs against the same objective.
The small RC model is a reproducible optimization example, not a validated building simulator.

The study audits its evaluator before tuning and records every bounded development trial.
See [the study protocol](research/redesign/engineering-study.md) and [the example](examples/06_engineering_control.py).
General autonomous code editing and frontier research remain later milestones in [the roadmap](research/redesign/roadmap.md).

## Acknowledgements

[EvoTorch](https://github.com/nnaisense/evotorch), developed by NNAISENSE and its contributors, is an implementation reference and source for this project. We study its algorithms and functional implementations and will adapt suitable code as the library develops. Reused implementations retain their source attribution.

The calibration benchmark adapts the Lotka-Volterra example and observed series from [calisim](https://github.com/Plant-Food-Research-Open/calisim). Its source revision and changes are recorded in the benchmark protocol. The benchmark retains the source license.

## Citation and license

```bibtex
@software{bonner2026torchdfo,
  author = {Bonner, Brett G.},
  title = {torch-dfo: Derivative-free optimization for PyTorch},
  year = {2026},
  url = {https://github.com/bbopen/torch-dfo},
  version = {0.11.0b1}
}
```

MIT. See [LICENSE](LICENSE).
