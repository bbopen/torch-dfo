"""Check EvoX 1.4.0's rank-one covariance term against an explicit matrix."""

import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import torch
from evox.algorithms.so.es_variants import CMAES

torch.set_default_dtype(torch.float64)
source = Path(inspect.getsourcefile(CMAES))
report = {
    "evox": version("evox"),
    "torch": torch.__version__,
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "scope": "One isolated covariance update with a nonzero path and zero rank-mu term.",
    "results": [],
}
for device in ("cpu", "cuda"):
    algorithm = CMAES(torch.zeros(3, device=device), sigma=0.6, pop_size=6, device=device)
    path = torch.tensor([1.0, 2.0, -1.0], device=device)
    covariance = torch.eye(3, device=device)
    population = torch.zeros(algorithm.mu, 3, device=device)
    mean = torch.zeros(1, 3, device=device)
    actual = algorithm._update_covariance_matrix(
        covariance, path, population, mean, torch.ones((), device=device)
    )
    base = (1 - algorithm.c_1 - algorithm.c_mu) * covariance
    expected_rank_one = torch.tensor(
        [[1.0, 2.0, -1.0], [2.0, 4.0, -2.0], [-1.0, -2.0, 1.0]], device=device
    )
    expected = base + algorithm.c_1 * expected_rank_one
    observed_rank_one = (actual - base) / algorithm.c_1
    report["results"].append(
        {
            "device": device,
            "initialized_path_shape": list(algorithm.p_c.shape),
            "probe_path_shape": list(path.shape),
            "matches_expected": torch.allclose(actual, expected, rtol=0, atol=1e-12),
            "max_absolute_difference": (actual - expected).abs().max().item(),
            "observed_rank_one": observed_rank_one.tolist(),
            "expected_rank_one": expected_rank_one.tolist(),
        }
    )
print(json.dumps(report, indent=2))
