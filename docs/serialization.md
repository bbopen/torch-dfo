# Serialization

All optimizers implement PyTorch-style `state_dict()` and `load_state_dict()`.

## Same-device round-trip

```python
import torch
import torch_dfo

opt = torch_dfo.CMAES(dim=10, bounds=5.0, device="cuda", seed=42)
for _ in range(100):
    x = opt.ask()
    opt.tell(x, (x**2).sum(-1))

torch.save(opt.state_dict(), "checkpoint.pt")

opt2 = torch_dfo.CMAES(dim=10, bounds=5.0, device="cuda")
opt2.load_state_dict(torch.load("checkpoint.pt", weights_only=False))
assert torch.allclose(opt.ask(), opt2.ask())
```

Same-device restores (`cpu -> cpu`, `cuda -> cuda`) reproduce the candidate stream bit-for-bit. `BaseOptimizer.load_state_dict()` also resizes population buffers when restoring checkpoints saved after `SHADE` population reduction.

## Cross-device loading

Loading a checkpoint onto a different device type (`cpu -> cuda` or `cuda -> cpu`) restores all tensor state and re-seeds the random generator from `_rng_seed` because PyTorch CPU and CUDA generator state buffers are not binary-compatible. Continuation stays deterministic from the stored seed, not bit-identical to the source device. This applies across all optimizers including `DLRPortfolio`.

## PhasedDFO checkpoints

`PhasedDFO.state_dict()` saves the nested `SHADE`, `CMAES` or portfolio branches, phase counters, and elite pools. `load_state_dict()` restores all nested state and preserves the shared generator reference across nested CMA-ES instances.

## Security

Treat serialized optimizer checkpoints the same as `torch.load(..., weights_only=False)`. Do not load untrusted checkpoint files.
