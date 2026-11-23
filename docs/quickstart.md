# Quickstart

## Install

Install a PyTorch build matching your device, then install `torch-dfo`:

```bash
pip install -e .
```

## Minimize a batched objective

```python
from torch_dfo import CMAES, minimize

optimizer = CMAES(dim=10, bounds=(-5.0, 5.0), device="cpu", seed=42)
result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
print(result.best_value, result.charged_evals)
```

The objective receives a `[population, dim]` tensor and returns one finite scalar per candidate. Lower is better. `CMAES`, `SHADE`, and `NelderMead` require complete steps (`optimizer.next_batch_size()`), so a small trailing budget residue can remain unused.

Select `device="cuda"` for CUDA builds or `device="mps", dtype=torch.float32` on Apple Silicon. Measure total wall time on your workload before moving small populations to GPU.

## Compare methods under equal budget

```python
from torch_dfo import NelderMead, RandomSearch, SHADE, minimize

for method in (RandomSearch, SHADE, NelderMead):
    optimizer = method(dim=10, bounds=5.0, device="cpu", seed=42)
    result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
    print(method.__name__, result.best_value, result.charged_evals)
```

Use {doc}`runs` for manual ask/tell loops, repeated evaluations, and planned checkpoints.
