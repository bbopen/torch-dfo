# Quickstart

## Installation

**CPU:**
```bash
pip install torch-dfo
```

**CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-dfo
```

**Apple Silicon (MPS):**
```bash
pip install torch torch-dfo
```

## First optimization

```python
import torch
import torch_dfo

opt = torch_dfo.CMAES(dim=10, bounds=(-5.0, 5.0), device="cuda")

for _ in range(500):
    x = opt.ask()                        # (pop_size, 10) tensor on CUDA
    f = (x ** 2).sum(dim=-1)            # sphere — evaluated on GPU, no CPU transfer
    opt.tell(x, f)

best_x, best_f = opt.best()
print(f"Best fitness: {best_f:.6f}")    # → ~0.0
```

## Swapping devices

The device is the only thing that changes:

```python
opt_cpu  = torch_dfo.SHADE(dim=20, bounds=5.0, device="cpu")
opt_cuda = torch_dfo.SHADE(dim=20, bounds=5.0, device="cuda")
opt_mps  = torch_dfo.SHADE(dim=20, bounds=5.0, device="mps")
```

## Choosing an optimizer

| Situation | Recommended optimizer |
|---|---|
| General-purpose, strongest convergence | `PhasedDFO` |
| High-dimensional (d > 100), GPU-first | `DLRPortfolio` |
| Moderate-dim, ill-conditioned | `CMAES` |
| Multimodal, robust | `SHADE` |
| Low-dim local polish | `NelderMead` |

## Checkpointing

```python
# Save
state = opt.state_dict()
torch.save(state, "checkpoint.pt")

# Restore
opt2 = torch_dfo.CMAES(dim=10, bounds=(-5.0, 5.0), device="cuda")
opt2.load_state_dict(torch.load("checkpoint.pt"))
```

Same-device loads are bit-exact. Cross-device loads (CPU → CUDA) fall back to re-seeding — see {doc}`serialization` for details.
