# Algorithms

## CMAES

Covariance Matrix Adaptation Evolution Strategy (Hansen 2001).
Full-covariance CMA-ES with mirrored sampling, active CMA updates,
IPOP population-doubling restarts, and an optional limited-memory
path sampler. Eigendecomposition runs on the target device.

```python
opt = torch_dfo.CMAES(
    dim=20,
    bounds=(-5.0, 5.0),
    pop_size=None,      # default: 4 + floor(3 ln dim)
    sigma0=0.3,
    mirrored=True,
    active=True,
    device="cuda",
)
```

**Ceiling:** O(d³) eigendecomposition per generation; prefer `SHADE` or
`DLRPortfolio` when full covariance updates dominate runtime at large dimension.

## SHADE

Success-History Based Adaptive Differential Evolution (Tanabe & Fukunaga 2014).
Self-adaptive F and CR via circular success-history memory.
Current-to-pbest/1 mutation with JADE-style external archive.
Scales well to high dimensions when population size is held fixed.

```python
opt = torch_dfo.SHADE(
    dim=100,
    bounds=(-5.0, 5.0),
    pop_size=80,
    memory_size=6,
    device="cuda",
)
```

## NelderMead

Classic Nelder-Mead simplex method adapted to the ask/tell protocol
via a batched `torch.where` decision chain. Best used as a local polisher
after a global search phase.

```python
opt = torch_dfo.NelderMead(dim=5, bounds=5.0, device="cpu")
```

## PhasedDFO

Three-phase budget-managed pipeline: SHADE-DE → IPOP-CMA-ES → Polish.
See {doc}`quickstart` and the README for a full description.

```python
opt = torch_dfo.PhasedDFO(
    dim=20,
    bounds=5.0,
    budget=100_000,     # default: dim * 5000
    device="cuda",
)
```

## DLRPortfolio

K-branch diagonal-plus-low-rank CMA portfolio (Loshchilov 2014).
No eigendecomposition — all operations stay on-device.
Designed for high-dimensional GPU-native optimization.

```python
import torch
from torch_dfo.dlr_cma import DLRPortfolio

dim = 200
lb  = torch.full((dim,), -5.0, dtype=torch.float64, device="cuda")
ub  = torch.full((dim,),  5.0, dtype=torch.float64, device="cuda")
rng = torch.Generator(device="cpu").manual_seed(42)

opt = DLRPortfolio(
    dim=dim, lb=lb, ub=ub,
    lambdas=(24, 12, 12, 12),
    sigma_fracs=(0.200, 0.043, 0.0093, 0.002),
    device=torch.device("cuda"),
    dtype=torch.float64,
    rng=rng,
)
```
