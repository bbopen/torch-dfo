# Serialization

All optimizers implement `state_dict()` and `load_state_dict()` following PyTorch conventions.

## Basic round-trip

```python
import torch
import torch_dfo

opt = torch_dfo.CMAES(dim=10, bounds=5.0, device="cuda", seed=42)
for _ in range(100):
    x = opt.ask()
    opt.tell(x, (x ** 2).sum(-1))

# Save
torch.save(opt.state_dict(), "checkpoint.pt")

# Restore on any compatible optimizer
opt2 = torch_dfo.CMAES(dim=10, bounds=5.0, device="cuda")
opt2.load_state_dict(torch.load("checkpoint.pt"))

# Bit-exact continuation (same device)
assert torch.allclose(opt.ask(), opt2.ask())
```

## Cross-device loading

Saving on one device and loading on another is supported:

```python
# CPU → CUDA
src = torch_dfo.SHADE(dim=5, bounds=5.0, device="cpu", seed=42)
state = src.state_dict()

dst = torch_dfo.SHADE(dim=5, bounds=5.0, device="cuda")
dst.load_state_dict(state)          # population & fitness restored on CUDA
```

**Important:** Cross-device loads are **non-bit-exact**. PyTorch's CPU and CUDA generators use different binary state representations, so the RNG is re-seeded from the stored seed rather than restored byte-for-byte. The optimizer will continue deterministically from the seed, but the sequence will diverge from what the original CPU run would have produced.

Same-device round-trips (CPU → CPU, CUDA → CUDA) remain bit-exact.

## PhasedDFO state

`PhasedDFO.state_dict()` captures all nested state:
- The SHADE sub-optimizer (phase 0)
- The CMAES sub-optimizer or portfolio (phase 1)
- All phase-machine fields (phase index, budget counters, stagnation trackers)
- The elite pool and search pool accumulated across phases

Load `PhasedDFO` state into an optimizer created with the same `dim`, `bounds`, and `budget`:

```python
opt1 = torch_dfo.PhasedDFO(dim=10, bounds=5.0, budget=50000, seed=42)
# ... run for a while ...
state = opt1.state_dict()

opt2 = torch_dfo.PhasedDFO(dim=10, bounds=5.0, budget=50000)
opt2.load_state_dict(state)         # full state, including nested SHADE/CMAES
```

## Security note

`load_state_dict` deserialises optimizer state from an arbitrary dict. **Do not call it on untrusted input.** Treat it with the same caution as `torch.load(weights_only=False)`.
