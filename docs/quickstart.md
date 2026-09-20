# Quickstart

## Install

Install a PyTorch build for your device. From this beta checkout, run:

```bash
pip install -e .
```

The beta is version `0.11.0b1`. You can also install its built wheel.

## Minimize a batched objective

```python
from torch_dfo import CMAES, minimize

optimizer = CMAES(dim=10, bounds=(-5.0, 5.0), device="cpu", seed=42)
result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
print(result.best_value, result.charged_evals)
```

The objective receives a tensor with shape `[population, dimension]`.
It returns one finite scalar per candidate. Lower values are better.
A complete CMA-ES generation must fit the remaining budget, so some evaluations can remain unused.

Use `device="cuda"` for a CUDA-enabled PyTorch build. Use `dtype=torch.float32` with MPS.
GPU speed depends on objective cost and population size. Measure total elapsed time on your task.

## Select a method

Start by comparing CMA-ES, SHADE, and random search under the same evaluation budget.
CMA-ES learns correlations between variables. SHADE adapts differential-evolution proposals.
Random search provides a simple baseline and can use a partial final batch.

```python
from torch_dfo import RandomSearch, SHADE, minimize

for method in (RandomSearch, SHADE):
    optimizer = method(dim=10, bounds=5.0, pop_size=32, device="cpu", seed=42)
    result = minimize(lambda x: x.square().sum(-1), optimizer, max_evals=1000)
    print(method.__name__, result.best_value, result.charged_evals)
```

Compare smooth objectives against gradient methods when reliable gradients are available.
The thermal-control example demonstrates a quantized objective and a conventional controller baseline.
It is a synthetic reference study, not a validated building model.

## Control each batch

Use {doc}`runs` for ask/tell, repeated evaluations, and planned checkpoints.
Existing optimizer-level ask/tell code continues to work. That lower-level interface leaves evaluation accounting to the caller.
