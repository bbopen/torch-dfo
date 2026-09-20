# torch-dfo

Derivative-free optimization with PyTorch tensors.

[![CI](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml/badge.svg)](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-dfo)](https://pypi.org/project/torch-dfo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Optimize bounded objectives that do not provide usable gradients. The core package requires PyTorch and supports CPU, CUDA, and MPS tensors.

## When to use it

Use derivative-free search for discrete decisions, discontinuous scores, external simulators, or objectives with unavailable or unreliable gradients. Batched tensor objectives can benefit from a GPU. Small populations and CPU simulators may not.

For smooth objectives with reliable gradients, compare against autograd-based methods. For expensive evaluations, compare against Bayesian optimization. The best method depends on evaluation cost and solution quality.

[EvoTorch](https://github.com/nnaisense/evotorch) already provides PyTorch-based evolutionary search under Apache 2.0. It includes distributed evaluation, neuroevolution, and algorithms beyond this package. torch-dfo provides a smaller tensor-based implementation, a phased search pipeline, structured search spaces, and a `torch.optim` wrapper. These differences do not establish performance superiority.

## Install

```bash
pip install torch-dfo
```

Requires Python 3.10 or later and PyTorch 2.4 or later. Install a PyTorch build that supports your accelerator before selecting CUDA or MPS.

## Batched search

```python
import torch
from torch_dfo import CMAES

opt = CMAES(dim=30, bounds=(-5.12, 5.12), device="cpu", seed=42)
for _ in range(100):
    candidates = opt.ask()
    fitness = candidates.square().sum(-1)
    opt.tell(candidates, fitness)
best_x, best_f = opt.best()
```

Select `device="cuda"` or `device="mps"` when available. The objective accepts a batch of candidates and returns one scalar fitness per candidate. Lower fitness is better.

## Algorithms

| Algorithm | Mechanism |
| --- | --- |
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

```python
state = opt.state_dict()
torch.save(state, "checkpoint.pt")
restored = CMAES(dim=30, bounds=(-5.12, 5.12), device="cpu", seed=42)
restored.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
```

Use matching optimizer configuration for continuation. Same-device tests check exact continuation. Cross-device loads can reinitialize the random generator from its seed, so continuation is not bit-exact.

## Accelerator and compilation limits

Tensor operations can remain on the selected device. Python control flow and scalar extraction can synchronize with the host. The package does not provide a fully asynchronous GPU search loop.

Compilation tests cover selected methods with graph breaks. They do not establish full-graph compilation or a general speedup. Measure objective time, optimizer time, and total time separately.

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

The redesign will include a research lab in this repository. Its program covers canonical evaluations, engineering applications, and autonomous experiments that propose, test, and retain measured improvements. See [the research program](research/redesign/program.md) for current scope and status.

## Acknowledgements

[EvoTorch](https://github.com/nnaisense/evotorch), developed by NNAISENSE and its contributors, is an implementation reference and source for this project. We study its algorithms and functional implementations and will adapt suitable code as the library develops. Reused implementations retain their source attribution.

## Citation and license

```bibtex
@software{bonner2026torchdfo,
  author = {Bonner, Brett G.},
  title = {torch-dfo: Derivative-free optimization for PyTorch},
  year = {2026},
  url = {https://github.com/bbopen/torch-dfo},
  version = {0.10.0}
}
```

MIT. See [LICENSE](LICENSE).
