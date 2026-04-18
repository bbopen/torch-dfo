"""torch-dfo: GPU-accelerated derivative-free optimization for PyTorch.

Algorithms:
    CMAES: Covariance Matrix Adaptation Evolution Strategy (IPOP)
    SHADE: Success-History Based Adaptive Differential Evolution
    NelderMead: Nelder-Mead simplex method
    PhasedDFO: SHADE-DE -> IPOP-CMA-ES -> FD-BFGS polish pipeline
    DLRPortfolio: GPU-native K-branch diagonal+low-rank CMA (no eigendecomp)

Wrappers:
    DFOOptimizer: torch.optim.Optimizer interface for DFO algorithms

Benchmarks:
    sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy, zakharov
    BenchmarkSuite: collection of standard test problems
"""

__version__ = "0.10.0"
__author__ = "Brett G. Bonner"

from torch_dfo.base import BaseOptimizer
from torch_dfo.benchmarks import (
    BenchmarkSuite,
    ackley,
    griewank,
    levy,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    zakharov,
)
from torch_dfo.cmaes import CMAES
from torch_dfo.dlr_cma import DLRPortfolio
from torch_dfo.nelder_mead import NelderMead
from torch_dfo.optim import DFOOptimizer
from torch_dfo.phased import PhasedDFO
from torch_dfo.shade import SHADE
from torch_dfo.space import Categorical, Float, Int, SearchSpace

__all__ = [
    # Core
    "BaseOptimizer",
    # Algorithms
    "CMAES",
    "SHADE",
    "NelderMead",
    "PhasedDFO",
    "DLRPortfolio",
    # Wrappers
    "DFOOptimizer",
    # Search space
    "SearchSpace",
    "Float",
    "Int",
    "Categorical",
    # Benchmarks
    "sphere",
    "rosenbrock",
    "rastrigin",
    "ackley",
    "griewank",
    "schwefel",
    "levy",
    "zakharov",
    "BenchmarkSuite",
]
