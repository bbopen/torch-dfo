# Algorithms

## RandomSearch

Uniform bounded search with an isolated `torch.Generator`. Useful as a baseline and compatible with partial final batches in `SearchRun`.

```python
opt = torch_dfo.RandomSearch(dim=20, bounds=(-5.0, 5.0), pop_size=32, device="cpu")
```

## CMAES

Full-covariance CMA-ES (Hansen 2016) with mirrored sampling, Mahalanobis-scaled active covariance updates, exact gauge-preserving eigensystem normalization when triggered, and IPOP restarts.

```python
opt = torch_dfo.CMAES(
    dim=20,
    bounds=(-5.0, 5.0),
    sigma0=0.3,
    mirrored=True,
    active=True,
    device="cuda",
)
```

Eigendecomposition costs `O(d^3)` per decomposition step. Prefer `SHADE` or `DLRPortfolio` when dense covariance updates dominate runtime at high dimension.

## SHADE

Success-history adaptive differential evolution (Tanabe & Fukunaga 2014) with current-to-pbest/1 mutation, an external archive, and optional linear population size reduction.

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

Batched simplex search using branch-free `torch.where` updates. `next_batch_size()` reports `dim + 1` on initial or post-shrink evaluations and `4` on standard iterations.

```python
opt = torch_dfo.NelderMead(dim=5, bounds=5.0, device="cpu")
```

## PhasedDFO

Self-budgeted three-stage pipeline combining SHADE exploration, CMA-ES adaptation, and local polishing (directional/coordinate basin search, smoothed envelope search, and finite-difference BFGS). Run via `optimize()` rather than `SearchRun`.

```python
opt = torch_dfo.PhasedDFO(
    dim=20,
    bounds=5.0,
    budget=100_000,
    device="cuda",
)
```

## DLRPortfolio

Multi-branch diagonal-plus-low-rank CMA portfolio with no eigendecomposition. All branch updates stay on-device.

```python
import torch
from torch_dfo import DLRPortfolio

opt = DLRPortfolio(
    dim=200,
    bounds=(-5.0, 5.0),
    lambdas=(24, 12, 12, 12),
    sigma_fracs=(0.200, 0.043, 0.0093, 0.002),
    device="cuda",
    dtype=torch.float64,
    seed=42,
)
```
