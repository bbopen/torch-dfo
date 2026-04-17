# torch-dfo

**GPU-accelerated derivative-free optimization for PyTorch.**

[![CI](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml/badge.svg)](https://github.com/bbopen/torch-dfo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-dfo)](https://pypi.org/project/torch-dfo/)
[![Python](https://img.shields.io/pypi/pyversions/torch-dfo)](https://pypi.org/project/torch-dfo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bbopen/torch-dfo/branch/master/graph/badge.svg)](https://codecov.io/gh/bbopen/torch-dfo)

Five optimizers running natively on PyTorch tensors. No NumPy at runtime. Works on CPU, CUDA, and MPS.

## Why torch-dfo

Existing derivative-free libraries are either CPU-bound (pycma, Nevergrad, scipy) or GPL-licensed (EvoX). torch-dfo fills the gap:

| | torch-dfo | pycma | Nevergrad | scipy.optimize |
|---|---|---|---|---|
| GPU-native | ✅ | ❌ | ❌ | ❌ |
| `torch.compile` support | ✅ | ❌ | ❌ | ❌ |
| Batched GPU objectives | ✅ | ❌ | ❌ | ❌ |
| MIT license | ✅ | ✅ | MIT | BSD |
| Ask/tell API | ✅ | ✅ | ✅ | ❌ |
| `torch.optim` wrapper | ✅ | ❌ | ❌ | ❌ |

## Install

**CPU (fastest install, great for development):**
```bash
pip install torch-dfo
```

**CUDA (recommended for production):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-dfo
```

**Apple Silicon (MPS):**
```bash
pip install torch  # standard PyPI build includes MPS support
pip install torch-dfo
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.4.

## 30-second example

```python
import torch
import torch_dfo

# Pick any optimizer; swap device="cpu" ↔ "cuda" ↔ "mps" with no other changes
opt = torch_dfo.CMAES(dim=30, bounds=(-5.12, 5.12), device="cuda")

for _ in range(1000):
    candidates = opt.ask()                    # (pop_size, 30) on GPU
    fitness    = (candidates ** 2).sum(-1)    # batched objective, stays on GPU
    opt.tell(candidates, fitness)

best_x, best_f = opt.best()
```

## Algorithms

| Algorithm | Paper | Best for |
|---|---|---|
| `CMAES` | Hansen (2001) | Medium-dim, ill-conditioned, curved valleys |
| `SHADE` | Tanabe & Fukunaga (2014) | Multimodal, self-adaptive, general-purpose |
| `NelderMead` | Nelder & Mead (1965) | Low-dim local polishing |
| `PhasedDFO` | 3-phase pipeline | General-purpose, strongest on benchmarks |
| `DLRPortfolio` | Loshchilov (2014) | High-dim, GPU-native, no eigendecomp |

All share the same `ask()` / `tell()` interface — swap optimizers with one line change.

## `torch.optim` wrapper

Use any torch-dfo optimizer as a drop-in `torch.optim.Optimizer` for gradient-free neural network training:

```python
from torch_dfo import DFOOptimizer, CMAES
import torch.nn as nn

model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
optimizer = DFOOptimizer(model.parameters(), algorithm=CMAES, dim=None, bounds=1.0)

def closure():
    loss = criterion(model(X), y)
    return loss

for _ in range(500):
    optimizer.step(closure)
```

## Checkpointing

Save and restore optimizer state across devices:

```python
# Save
state = opt.state_dict()
torch.save(state, "checkpoint.pt")

# Restore on any device (CPU ↔ CUDA ↔ MPS)
opt2 = torch_dfo.CMAES(dim=30, bounds=(-5.12, 5.12), device="cuda")
opt2.load_state_dict(torch.load("checkpoint.pt"))
```

Same-device round-trips are bit-exact. Cross-device loads fall back to seed-based RNG re-initialisation (non-bit-exact continuation, documented).

## GPU performance

torch-dfo is designed around batched tensor operations — all candidate evaluation, covariance updates, and selection happen on-device with no host round-trips. `DLRPortfolio` uses a diagonal-plus-low-rank covariance approximation (no eigendecomposition) to scale to high dimensions without falling back to CPU.

Scaling ceiling on a single NVIDIA RTX A4500 (19.6 GB usable, double precision):

| Optimizer | Max dim (no OOM) |
|---|---|
| `DLRPortfolio` | ≥ 20 480 (ceiling not hit, peak ≤ 123 MB) |
| `SHADE` | ≥ 20 480 (ceiling not hit, peak ≤ 151 MB) |
| `NelderMead` | 20 480 |
| `CMAES` | 10 240 (`O(d²)` covariance) |
| `PhasedDFO` | 5 120 (pop scales ~4·d) |

Full per-optimizer guidance, the quality caveat, and reproduction instructions live in [`docs/benchmarks.rst`](docs/benchmarks.rst).

## PhasedDFO pipeline

`PhasedDFO` is the library's flagship optimizer, combining three phases under a single budget-managed interface:

1. **SHADE-DE** — global exploration with opposition-based init, Levy flight perturbation, and success-history adaptive parameters.
2. **IPOP-CMA-ES** — exploitation with warm-started covariance from DE elites, mirrored sampling, active CMA updates, and population-doubling restarts.
3. **Polish** — directional basin search (CMA eigenvectors + PCA + elite displacements), coordinate basin search, and FD-BFGS local convergence.

### Benchmark results

16 classical benchmarks (sphere/rosenbrock/rastrigin/ackley + shifted + rotated + shifted-rotated variants), 10 runs each, budget = `dim × 5000` FE, blackbox evaluation (no autograd):

| Benchmark | Mean Fitness |
|---|---|
| Sphere 10d / 30d | 0.000000 |
| Rosenbrock 10d / 30d | 0.000000 |
| Rastrigin 10d / 30d | 0.000000 |
| Ackley 10d / 30d | 0.000000 |
| Shifted Sphere / Rosenbrock / Rastrigin / Ackley 30d | 0.000000 |
| Rotated Rastrigin 30d | ~4.278 |
| Rotated Ackley / ShiftRot Rastrigin / ShiftRot Ackley 30d | 0.000000 |
| **Mean across 16 benchmarks** | **0.267** |

14 of 16 solved to effectively zero. The residual mean is dominated by rotated Rastrigin 30d, where full covariance rotation combined with multimodal basin structure is the recognised hard case for DFO.

## Search space API

For structured (mixed continuous / integer / categorical) problems:

```python
from torch_dfo import SearchSpace, Float, Int, PhasedDFO

space = SearchSpace([
    Float("lr", 1e-5, 1e-1, log=True),
    Int("batch_size", 16, 512),
])

opt = PhasedDFO(space=space, budget=500)
for _ in range(opt.budget):
    trial = opt.ask()               # dict of {param: value}
    score = evaluate(trial)
    opt.tell(trial, score)
```

## Citation

```bibtex
@software{bonner2026torchdfo,
  author  = {Bonner, Brett G.},
  title   = {torch-dfo: GPU-accelerated derivative-free optimization for PyTorch},
  year    = {2026},
  url     = {https://github.com/bbopen/torch-dfo},
  version = {0.9.0}
}
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Hansen, N. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*.
- Tanabe, R. & Fukunaga, A. (2014). Improving the search performance of SHADE using linear population size reduction. *IEEE CEC*.
- Nelder, J. & Mead, R. (1965). A simplex method for function minimization. *Computer Journal*.
- Loshchilov, I. (2014). A computationally efficient limited memory CMA-ES for large scale optimization. *GECCO*.
- Zhang, J. & Sanderson, A. (2009). JADE: Adaptive differential evolution with optional external archive. *IEEE T-EC*.
