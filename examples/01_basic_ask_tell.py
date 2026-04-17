"""
Basic ask/tell loop
===================

The simplest way to use torch-dfo: ask for candidates, evaluate them,
tell the optimizer the fitness values. Works on CPU, CUDA, and MPS.
"""

import torch

import torch_dfo

# --------------------------------------------------------------------- #
# 1. Create an optimizer                                                  #
# --------------------------------------------------------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"

opt = torch_dfo.CMAES(
    dim=10,
    bounds=5.0,  # search in [-5, 5]^10
    device=device,
    seed=42,
)

# --------------------------------------------------------------------- #
# 2. Run the ask/tell loop                                                #
# --------------------------------------------------------------------- #


def sphere(x: torch.Tensor) -> torch.Tensor:
    """Sum of squares — global min 0 at origin."""
    return (x**2).sum(dim=-1)


for _ in range(200):
    candidates = opt.ask()  # (pop_size, dim) on device
    fitness = sphere(candidates)  # (pop_size,)   on device
    opt.tell(candidates, fitness)

# --------------------------------------------------------------------- #
# 3. Retrieve the best solution found                                     #
# --------------------------------------------------------------------- #

best_x, best_f = opt.best()
print(f"Best fitness after 200 generations: {best_f.item():.6e}")
print(f"Best x[:4] = {best_x[:4].tolist()}")
