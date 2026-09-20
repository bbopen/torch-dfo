"""
Gradient-free neural network training
======================================

``DFOOptimizer`` wraps SHADE, CMA-ES, or Nelder-Mead with a
``torch.optim.Optimizer`` interface. Each step evaluates candidate models
without gradients.
Useful when gradients are unavailable or unreliable (e.g. non-differentiable
losses, RL reward signals, hyperparameter search over discrete configs).
"""

import torch
import torch.nn as nn

import torch_dfo

torch.manual_seed(0)

# --------------------------------------------------------------------- #
# 1. Tiny regression model                                                #
# --------------------------------------------------------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

# Synthetic dataset: y = sum(x)
X = torch.randn(64, 4, device=device)
y = X.sum(dim=-1, keepdim=True)

# --------------------------------------------------------------------- #
# 2. Wrap with DFOOptimizer                                               #
# --------------------------------------------------------------------- #
# ``bounds`` gives absolute limits for each parameter.
# ``pop_size`` is per-generation candidate count (default: auto).

dfo = torch_dfo.DFOOptimizer(
    model.parameters(),
    algorithm="shade",
    bounds=1.0,
    pop_size=40,
)

loss_fn = nn.MSELoss()

# --------------------------------------------------------------------- #
# 3. Training loop                                                        #
# --------------------------------------------------------------------- #

for step in range(50):

    def closure() -> torch.Tensor:
        pred = model(X)
        return loss_fn(pred, y)

    loss = dfo.step(closure)  # type: ignore[assignment]

    if step % 10 == 0:
        print(f"step {step:3d}  loss={loss:.4f}")

best_loss = closure()
print(f"\nFinal loss: {best_loss:.4f}")
