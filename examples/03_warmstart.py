"""
Checkpointing and warm-start
=============================

All optimizers implement ``state_dict()`` / ``load_state_dict()`` following
PyTorch conventions.  This lets you:

* Save mid-run checkpoints and resume on any compatible device.
* Distribute a partially-warmed optimizer across machines.
* Keep the best seed from a short exploratory run and hand it off to a
  longer, more precise phase.
"""

import torch

import torch_dfo

device = "cuda" if torch.cuda.is_available() else "cpu"


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    A = 10.0
    return A * x.shape[-1] + (x**2 - A * torch.cos(2 * torch.pi * x)).sum(-1)


# --------------------------------------------------------------------- #
# 1. Run 100 generations then checkpoint                                  #
# --------------------------------------------------------------------- #

opt = torch_dfo.PhasedDFO(dim=10, bounds=5.0, budget=50_000, device=device, seed=42)

for _ in range(100):
    x = opt.ask()
    opt.tell(x, rastrigin(x))

state = opt.state_dict()
torch.save(state, "/tmp/torch_dfo_checkpoint.pt")

best_x, best_f = opt.best()
print(f"After 100 gens  — best f = {best_f.item():.4f}  phase = {opt.phase}")

# --------------------------------------------------------------------- #
# 2. Reload on the same (or a different) device and continue             #
# --------------------------------------------------------------------- #

opt2 = torch_dfo.PhasedDFO(dim=10, bounds=5.0, budget=50_000, device=device)
opt2.load_state_dict(torch.load("/tmp/torch_dfo_checkpoint.pt", weights_only=False))

assert opt2.phase == opt.phase
assert opt2.fe_count == opt.fe_count

for _ in range(100):
    x = opt2.ask()
    opt2.tell(x, rastrigin(x))

best_x2, best_f2 = opt2.best()
print(f"After 200 gens  — best f = {best_f2.item():.4f}  phase = {opt2.phase}")
