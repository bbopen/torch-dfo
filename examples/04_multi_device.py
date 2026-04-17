"""
Multi-device usage
==================

torch-dfo optimizers are device-local: each instance lives on one device.
This example shows:

* Device-agnostic construction via a helper.
* Running the same problem on CPU and CUDA side by side.
* Cross-device state transfer (non-bit-exact; see serialization docs).
"""

import torch

import torch_dfo


def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1)


# --------------------------------------------------------------------- #
# 1. Device-agnostic helper                                               #
# --------------------------------------------------------------------- #


def make_optimizer(device: str | torch.device) -> torch_dfo.SHADE:
    return torch_dfo.SHADE(dim=20, bounds=5.0, pop_size=60, device=device, seed=7)


# --------------------------------------------------------------------- #
# 2. CPU run (always available)                                           #
# --------------------------------------------------------------------- #

cpu_opt = make_optimizer("cpu")
for _ in range(50):
    x = cpu_opt.ask()
    cpu_opt.tell(x, sphere(x))

_, cpu_best = cpu_opt.best()
print(f"CPU  best f = {cpu_best.item():.4e}")

# --------------------------------------------------------------------- #
# 3. CUDA run (skipped gracefully when unavailable)                       #
# --------------------------------------------------------------------- #

if torch.cuda.is_available():
    cuda_opt = make_optimizer("cuda")
    for _ in range(50):
        x = cuda_opt.ask()
        cuda_opt.tell(x, sphere(x))

    _, cuda_best = cuda_opt.best()
    print(f"CUDA best f = {cuda_best.item():.4e}")

    # ----------------------------------------------------------------- #
    # 4. Transfer CPU state to CUDA                                       #
    # ----------------------------------------------------------------- #
    # Cross-device loads re-seed the RNG from the stored seed rather than
    # restoring the binary generator state — they are non-bit-exact but
    # the population and fitness buffers are faithfully restored.

    cuda_opt2 = make_optimizer("cuda")
    cuda_opt2.load_state_dict(cpu_opt.state_dict())

    pop_cpu = cpu_opt.ask().to("cuda")
    pop_cuda = cuda_opt2.ask()

    print(f"\nPopulation shapes match: {pop_cpu.shape == pop_cuda.shape}")
    print("(Values may differ due to cross-device RNG re-seeding — this is expected.)")
else:
    print("CUDA not available — skipping GPU section.")
