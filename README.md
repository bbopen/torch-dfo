# torch-dfo

Derivative-free optimization for PyTorch.

[![CI](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml/badge.svg)](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-dfo)](https://pypi.org/project/torch-dfo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Optimize bounded objectives without gradients. The core package requires only PyTorch and runs on CPU, CUDA, and MPS tensors.

## When to use it

Use derivative-free search for discrete or quantized decisions, discontinuous scores, external simulators, or objectives without reliable gradients. Batched tensor objectives can benefit from a GPU. Small populations and serial CPU simulators usually run faster on CPU.

For smooth objectives with reliable gradients, compare against autograd methods. For expensive single-point evaluations, compare against Bayesian optimization.

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.4+. Install a PyTorch build matching your accelerator before selecting CUDA or MPS.

## Budgeted search

```python
from torch_dfo import CMAES, minimize

optimizer = CMAES(dim=10, bounds=(-5.0, 5.0), device="cpu", seed=42)
result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
print(result.best_value, result.charged_evals)
```

The objective receives a `[batch, dim]` tensor and returns one scalar per candidate. Lower is better. `minimize` delegates to `SearchRun` to enforce evaluation budgets, record raw observations, and support planned checkpoints. See [docs/runs.md](docs/runs.md).

`SearchRun` supports `CMAES`, `SHADE`, `NelderMead`, and `RandomSearch`. `CMAES`, `SHADE`, and `NelderMead` require complete steps (`optimizer.next_batch_size()`); `RandomSearch` can consume a partial final batch. `PhasedDFO` manages its own budget via `optimize()` and raises `TypeError` if passed to `SearchRun`.

## Algorithms

| Algorithm | Mechanism |
| --- | --- |
| `RandomSearch` | Uniform bounded search with a private random generator |
| `CMAES` | Full-covariance CMA-ES with active updates, mirrored sampling, and IPOP restarts |
| `SHADE` | Success-history adaptive differential evolution |
| `NelderMead` | Batched simplex local search |
| `PhasedDFO` | Self-budgeted SHADE, CMA-ES, and local polish pipeline via `optimize()` |
| `DLRPortfolio` | Multi-branch diagonal-plus-low-rank CMA without eigendecomposition |

All optimizers expose `ask()` and `tell()`, plus PyTorch-style `state_dict()` and `load_state_dict()`. See [docs/algorithms.md](docs/algorithms.md) and [docs/serialization.md](docs/serialization.md).

## Model parameter search

`DFOOptimizer` wraps `torch.nn.Module` parameters using `cmaes`, `shade`, or `nelder_mead`.

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

Use `closure_batched` when the objective can evaluate candidate parameter vectors together.

## Structured search

`SearchSpace` maps `Float`, `Int`, and `Categorical` parameters into unit-cube coordinates for `PhasedDFO`.

```python
import torch
from torch_dfo import Float, Int, PhasedDFO, SearchSpace

space = SearchSpace([Float("scale", 0.1, 2.0), Int("count", 1, 8)])
search = PhasedDFO(space=space, budget=100, seed=42)

def objective(trials):
    return torch.tensor(
        [(t["scale"] - 1.0) ** 2 + abs(t["count"] - 4) for t in trials],
        dtype=search.dtype,
        device=search.device,
    )

best_encoded, best_score = search.optimize(objective)
```

## GPU execution, verification, and roadmap

- **GPU execution:** Keep objective evaluation on-device and compile the batched objective rather than the outer Python ask/tell loop. See [docs/gpu.md](docs/gpu.md).
- **Formal verification:** Core covariance, step-size, budget, simplex, and IEEE-754 invariants are checked in Lean 4 and Z3 via `formal/verify.sh` and `tests/test_formal_invariants.py`.
- **Benchmarks and direction:** Benchmark harnesses live in `benchmarks/` (`cuda_control.py`, `cma_comparison.py`, `calibration.py`). Project direction and priorities live in [research/redesign/roadmap.md](research/redesign/roadmap.md).

## Acknowledgements

- [EvoTorch](https://github.com/nnaisense/evotorch) (NNAISENSE) is an implementation and benchmark reference.
- `benchmarks/calibration.py` adapts the Lotka-Volterra example and observed series from [calisim](https://github.com/Plant-Food-Research-Open/calisim).

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
