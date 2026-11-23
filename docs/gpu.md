# GPU guide

## Device selection

Pass `device` to any optimizer constructor using any string or `torch.device` accepted by PyTorch:

```python
torch_dfo.CMAES(dim=20, bounds=5.0, device="cpu")
torch_dfo.CMAES(dim=20, bounds=5.0, device="cuda")
torch_dfo.CMAES(dim=20, bounds=5.0, device="mps", dtype=torch.float32)
```

## Keep objective evaluation on-device

`ask()` returns a tensor on the optimizer's device. Evaluate the entire batch on that device and return a 1-D fitness tensor on the same device:

```python
opt = torch_dfo.SHADE(dim=50, bounds=5.0, device="cuda")

for _ in range(1000):
    x = opt.ask()
    f = my_batched_objective(x)
    opt.tell(x, f)
```

Avoid per-candidate `.cpu().numpy()` loops. Serial host evaluation removes GPU benefit.

## Precision

Optimizers default to `torch.float64`. On consumer GPUs with reduced FP64 throughput or on Apple Silicon MPS (which does not support FP64), pass `dtype=torch.float32`.

## Compilation

Compile the batched objective with `torch.compile`, not the outer Python ask/tell loop. Optimizer control flow contains host scalar reads and synchronization boundaries, so wrapping the outer search loop introduces graph breaks. Measure setup cost, objective time, and total search wall time separately.

See {doc}`benchmarks` for memory scaling notes across dimensions.
