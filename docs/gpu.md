# GPU guide

## Device selection

Pass `device` to any optimizer constructor. Valid values are any string or `torch.device` accepted by PyTorch:

```python
torch_dfo.CMAES(dim=20, bounds=5.0, device="cpu")
torch_dfo.CMAES(dim=20, bounds=5.0, device="cuda")
torch_dfo.CMAES(dim=20, bounds=5.0, device="cuda:1")  # second GPU
torch_dfo.CMAES(dim=20, bounds=5.0, device="mps")     # Apple Silicon
```

## Keep tensors on-device

The key to GPU performance is keeping all computation on-device. `ask()` returns a tensor on the optimizer's device; your objective should operate on it there and return a fitness tensor on the same device:

```python
opt = torch_dfo.SHADE(dim=50, bounds=5.0, device="cuda")

for _ in range(1000):
    x = opt.ask()                          # cuda tensor
    f = my_batched_objective(x)            # must stay on cuda
    opt.tell(x, f)
```

**Avoid:** `f = torch.tensor([my_fn(xi.cpu().numpy()) for xi in x])` — this kills GPU utilisation.

## dtype

All optimizers default to `torch.float64`. On CUDA, float64 throughput is ~8× lower than float32 on consumer GPUs (Ampere and older) but full-speed on datacenter cards (A100, H100). If precision is not critical:

```python
opt = torch_dfo.SHADE(dim=100, bounds=5.0, device="cuda", dtype=torch.float32)
```

Note: MPS does not support float64 — use `dtype=torch.float32` on Apple Silicon.

## torch.compile

The ask/tell hot path uses `torch.where` for branch-free operations and is compatible with `torch.compile`. This is most beneficial for large populations or high-frequency iterations:

```python
import torch
import torch_dfo

opt = torch_dfo.CMAES(dim=20, bounds=5.0, device="cuda")

@torch.compile
def step(opt, objective):
    x = opt.ask()
    f = objective(x)
    opt.tell(x, f)
```

## Scaling ceiling

GPU-specific scaling results (wall-clock vs. dimension, peak VRAM per optimizer) measured on a single NVIDIA RTX A4500 (19.6 GB usable, double precision) are summarised in {doc}`benchmarks`. The sweep is driven by [`benchmarks/scaling_probe.py`](https://github.com/bbopen/torch-dfo/blob/master/benchmarks/scaling_probe.py) and the GitHub Actions workflow at [`scaling-probe.yml`](https://github.com/bbopen/torch-dfo/actions/workflows/scaling-probe.yml).
