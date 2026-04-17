"""Classical benchmark functions for derivative-free optimization.

All functions accept tensors of shape ``(N, D)`` (batched) or ``(D,)`` (single point)
and return tensors of shape ``(N,)`` or scalar respectively, using ``dim=-1``
reductions throughout.

References
----------
Surjanovic & Bingham, Virtual Library of Simulation Experiments.
    https://www.sfu.ca/~ssurjano/optimization.html

"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


def sphere(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum(x_i^2). Optimum: f(0) = 0."""
    return (x**2).sum(dim=-1)


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2). Optimum: f(1,...,1) = 0."""
    xi = x[..., :-1]
    xi1 = x[..., 1:]
    return (100.0 * (xi1 - xi**2) ** 2 + (1.0 - xi) ** 2).sum(dim=-1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i)). Optimum: f(0) = 0."""
    d = x.shape[-1]
    return 10.0 * d + (x**2 - 10.0 * torch.cos(2.0 * math.pi * x)).sum(dim=-1)


def ackley(x: torch.Tensor) -> torch.Tensor:
    """Ackley function. Optimum: f(0) = 0.

    f(x) = -20*exp(-0.2*sqrt(mean(x_i^2)))
           - exp(mean(cos(2*pi*x_i))) + 20 + e
    """
    d = x.shape[-1]
    sum_sq = (x**2).sum(dim=-1)
    sum_cos = torch.cos(2.0 * math.pi * x).sum(dim=-1)
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq / d))
    term2 = -torch.exp(sum_cos / d)
    return term1 + term2 + 20.0 + math.e


def griewank(x: torch.Tensor) -> torch.Tensor:
    """Griewank function. Optimum: f(0) = 0.

    f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1)))
    """
    d = x.shape[-1]
    # Build index tensor: 1, 2, ..., D  (1-based as in the standard formula)
    idx = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    sum_term = (x**2).sum(dim=-1) / 4000.0
    prod_term = torch.cos(x / torch.sqrt(idx)).prod(dim=-1)
    return 1.0 + sum_term - prod_term


def schwefel(x: torch.Tensor) -> torch.Tensor:
    r"""Schwefel function. Optimum: f(420.9687,...,420.9687) ~ 0.

    f(x) = 418.9829*D - sum(x_i * sin(sqrt(\|x_i\|)))
    """
    d = x.shape[-1]
    return 418.9829 * d - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum(dim=-1)


def levy(x: torch.Tensor) -> torch.Tensor:
    """Levy function. Optimum: f(1,...,1) = 0.

    ::

        w_i = 1 + (x_i - 1)/4
        f(x) = sin^2(pi*w_1)
               + sum_{i=1}^{D-1} (w_i - 1)^2 * (1 + 10*sin^2(pi*w_i + 1))
               + (w_D - 1)^2 * (1 + sin^2(2*pi*w_D))
    """
    w = 1.0 + (x - 1.0) / 4.0
    w1 = w[..., 0]
    wd = w[..., -1]
    wi = w[..., :-1]

    term1 = torch.sin(math.pi * w1) ** 2
    term2 = ((wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * wi + 1.0) ** 2)).sum(dim=-1)
    term3 = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * wd) ** 2)
    return term1 + term2 + term3


def zakharov(x: torch.Tensor) -> torch.Tensor:
    """Zakharov function. Optimum: f(0) = 0.

    f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
    where i is 1-based.
    """
    d = x.shape[-1]
    idx = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    sum_sq = (x**2).sum(dim=-1)
    weighted = (0.5 * idx * x).sum(dim=-1)
    return sum_sq + weighted**2 + weighted**4


# ---------------------------------------------------------------------------
# Shifted / rotated variant constructors
# ---------------------------------------------------------------------------


def make_shifted(
    fn: Callable[[torch.Tensor], torch.Tensor],
    shift: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create shifted version: g(x) = f(x - shift). *shift* is a ``(D,)`` tensor."""

    def shifted_fn(x: torch.Tensor) -> torch.Tensor:
        return fn(x - shift)

    return shifted_fn


def make_rotated(
    fn: Callable[[torch.Tensor], torch.Tensor],
    rotation_matrix: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create rotated version: g(x) = f(x @ R^T). *rotation_matrix* is ``(D, D)`` orthogonal."""

    def rotated_fn(x: torch.Tensor) -> torch.Tensor:
        return fn(x @ rotation_matrix.T)

    return rotated_fn


def random_rotation_matrix(
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Returns a ``(dim, dim)`` orthogonal matrix on the requested device/dtype.
    """
    # Generate on CPU (torch.linalg.qr may not be available on all backends for randn)
    # then move to the target device.
    cpu_gen = None
    if generator is not None:
        # Clone generator state to CPU if needed
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(generator.initial_seed())

    z = torch.randn(dim, dim, dtype=dtype, device="cpu", generator=cpu_gen)
    q, r = torch.linalg.qr(z)
    # Ensure a unique decomposition by making the diagonal of R positive
    d = torch.diag(r)
    ph = torch.diag(d / torch.abs(d))
    q = q @ ph
    if device is not None:
        q = q.to(device=device)
    return q  # type: ignore[no-any-return]


def random_shift(
    dim: int,
    lb: float,
    ub: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a random shift vector, centered at zero, within bounds.

    Shift is ``(rand - 0.5) * 0.4 * span`` then clamped with a 20% margin
    so the shifted optimum stays well within the search space.
    """
    cpu_gen = None
    if generator is not None:
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(generator.initial_seed())

    span = ub - lb
    margin = 0.2 * span
    raw = torch.empty(dim, dtype=dtype, device="cpu").uniform_(0.0, 1.0, generator=cpu_gen)
    shift = (raw - 0.5) * 0.4 * span
    shift = shift.clamp(lb + margin, ub - margin)
    if device is not None:
        shift = shift.to(device=device)
    return shift


# ---------------------------------------------------------------------------
# Benchmark problem dataclass and suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkProblem:
    """A single benchmark optimisation problem.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. ``"sphere_10d"``).
    fn : Callable
        Objective function mapping ``(N, D)`` -> ``(N,)``.
    dim : int
        Dimensionality of the search space.
    bounds : tuple[float, float]
        Per-dimension ``(lower, upper)`` bounds.
    known_optimum : float
        Known global minimum value.

    """

    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    dim: int
    bounds: tuple[float, float]
    known_optimum: float


# Canonical bounds for each function family
_BOUNDS: dict[str, tuple[float, float]] = {
    "sphere": (-5.12, 5.12),
    "rosenbrock": (-2.048, 2.048),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "levy": (-10.0, 10.0),
}

# Functions with their known global minimum values
_FUNCTIONS: dict[str, tuple[Callable[..., torch.Tensor], float]] = {
    "sphere": (sphere, 0.0),
    "rosenbrock": (rosenbrock, 0.0),
    "rastrigin": (rastrigin, 0.0),
    "ackley": (ackley, 0.0),
    "griewank": (griewank, 0.0),
    "levy": (levy, 0.0),
}


class BenchmarkSuite:
    """Factory for standard benchmark problem collections."""

    @staticmethod
    def classical(
        dims: tuple[int, ...] = (10, 30),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> list[BenchmarkProblem]:
        """Return benchmark problems for classical functions at each requested dimensionality.

        Produces ``len(dims)`` problems for each of the six core functions
        (sphere, rosenbrock, rastrigin, ackley, griewank, levy), yielding
        ``6 * len(dims)`` problems total by default.
        """
        problems: list[BenchmarkProblem] = []
        for name, (fn, opt) in _FUNCTIONS.items():
            problems.extend(
                BenchmarkProblem(
                    name=f"{name}_{d}d",
                    fn=fn,
                    dim=d,
                    bounds=_BOUNDS[name],
                    known_optimum=opt,
                )
                for d in dims
            )
        return problems

    @staticmethod
    def full(
        dims: tuple[int, ...] = (10, 30),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int = 42,
        stress_dim: int = 30,
    ) -> list[BenchmarkProblem]:
        """Return the 16-problem benchmark suite.

        The suite comprises:
        - classical problems for each requested dimension
        - 2 shifted variants  (sphere/rosenbrock at ``stress_dim``)
        - 2 rotated variants  (rastrigin/ackley at ``stress_dim``)
        """
        problems = BenchmarkSuite.classical(dims=dims, device=device, dtype=dtype)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        # Shifted sphere at the stress dimension
        shift_s = random_shift(
            stress_dim,
            *_BOUNDS["sphere"],
            device=device,
            dtype=dtype,
            generator=gen,
        )
        problems.append(
            BenchmarkProblem(
                name=f"shifted_sphere_{stress_dim}d",
                fn=make_shifted(sphere, shift_s),
                dim=stress_dim,
                bounds=_BOUNDS["sphere"],
                known_optimum=0.0,
            ),
        )

        # Shifted rosenbrock at the stress dimension
        shift_r = random_shift(
            stress_dim,
            *_BOUNDS["rosenbrock"],
            device=device,
            dtype=dtype,
            generator=gen,
        )
        problems.append(
            BenchmarkProblem(
                name=f"shifted_rosenbrock_{stress_dim}d",
                fn=make_shifted(rosenbrock, shift_r),
                dim=stress_dim,
                bounds=_BOUNDS["rosenbrock"],
                known_optimum=0.0,
            ),
        )

        # Rotated rastrigin at the stress dimension
        rot_rast = random_rotation_matrix(
            stress_dim,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        problems.append(
            BenchmarkProblem(
                name=f"rotated_rastrigin_{stress_dim}d",
                fn=make_rotated(rastrigin, rot_rast),
                dim=stress_dim,
                bounds=_BOUNDS["rastrigin"],
                known_optimum=0.0,
            ),
        )

        # Rotated ackley at the stress dimension
        rot_ack = random_rotation_matrix(
            stress_dim,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        problems.append(
            BenchmarkProblem(
                name=f"rotated_ackley_{stress_dim}d",
                fn=make_rotated(ackley, rot_ack),
                dim=stress_dim,
                bounds=_BOUNDS["ackley"],
                known_optimum=0.0,
            ),
        )

        return problems
