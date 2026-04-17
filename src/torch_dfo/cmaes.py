"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

Implements the (mu/mu_w, lambda)-CMA-ES from Hansen (2001) following the
2016 tutorial (arxiv:1604.00772). Supports mirrored (antithetic) sampling,
active CMA updates with negative weights, and IPOP restarts.

References
----------
Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
    arXiv:1604.00772v1

"""

from __future__ import annotations

import math
from typing import Any

import torch

from torch_dfo.base import BaseOptimizer
from torch_dfo.utils import clamp_to_bounds


def _default_pop_size(dim: int) -> int:
    """Default population size: 4 + floor(3 * ln(dim))."""
    return 4 + math.floor(3 * math.log(dim))


class CMAES(BaseOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy.

    Hansen (2001). Supports mirrored (antithetic) sampling, active CMA updates,
    and IPOP restarts via the restart() method.

    Parameters
    ----------
    dim : int
        Problem dimensionality.
    bounds : float | tuple[float, float]
        Search-space bounds (symmetric scalar or (lb, ub) tuple).
    pop_size : int | None
        Population size (lambda). ``None`` uses ``4 + floor(3 * ln(dim))``.
    device : str | torch.device | None
        Torch device.
    dtype : torch.dtype
        Floating-point dtype (float64 recommended).
    seed : int | None
        RNG seed for reproducibility.
    sigma0 : float
        Initial step size as a fraction of the search-space span.
    mirrored : bool
        If ``True``, use antithetic (mirrored) sampling.
    active : bool
        If ``True``, apply active CMA (negative-weight) covariance update.
    path_memory : int
        Number of recent normalized covariance paths to use as low-rank
        sampling directions. ``0`` disables the limited-memory path sampler.
    path_scale : float
        Standard-deviation multiplier for path-memory sampling.
    path_line_samples : int
        Number of candidate slots to overwrite with deterministic samples along
        the covariance evolution path. ``0`` disables evolution-path line
        sampling.
    path_line_scale : float
        Step multiplier for evolution-path line samples.

    Examples
    --------
    >>> import torch
    >>> import torch_dfo
    >>> opt = torch_dfo.CMAES(dim=10, bounds=5.0, seed=0)
    >>> for _ in range(50):
    ...     x = opt.ask()
    ...     opt.tell(x, (x ** 2).sum(-1))
    >>> best_x, best_f = opt.best()

    """

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float],
        pop_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
        sigma0: float = 0.3,
        mirrored: bool = False,
        active: bool = False,
        path_memory: int = 0,
        path_scale: float = 0.0,
        path_line_samples: int = 0,
        path_line_scale: float = 1.0,
    ):
        if pop_size is None:
            pop_size = _default_pop_size(dim)

        super().__init__(
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        self.mirrored = mirrored
        self.active = active
        self.path_memory = max(0, int(path_memory))
        self.path_scale = max(0.0, float(path_scale))
        self.path_line_samples = max(0, int(path_line_samples))
        self.path_line_scale = max(0.0, float(path_line_scale))

        # ---------- strategy parameters (Hansen 2016, Table 1) ----------
        self.mu = pop_size // 2
        mu = self.mu

        # Recombination weights (log-linear, positive only for now)
        raw = torch.log(torch.tensor((pop_size + 1) / 2, dtype=dtype)) - torch.log(
            torch.arange(1, mu + 1, dtype=dtype),
        )
        self.weights = (raw / raw.sum()).to(device=self.device)

        # Variance-effective selection mass
        self.mu_eff = (1.0 / (self.weights**2).sum()).item()
        mu_eff = self.mu_eff

        # Cumulation and adaptation rates
        self.c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
        self.c_c = (mu_eff + 2) / (dim + 4 + 2 * mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff),
        )

        # Expected norm of N(0, I)
        self.chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Active CMA parameters
        if active:
            mu_neg = min(mu, pop_size - mu)
            self._mu_neg = mu_neg
            neg_raw = torch.log(torch.tensor(mu_neg + 1, dtype=dtype)) - torch.log(
                torch.arange(1, mu_neg + 1, dtype=dtype),
            )
            self._neg_weights = (neg_raw / neg_raw.sum()).to(device=self.device)
            self._c_mu_neg = min(0.25 * self.c_mu, max(0.0, 1 - self.c_1 - self.c_mu))

        # Eigendecomposition frequency
        self._decomp_frequency = max(1, math.floor(1 / (10 * dim * (self.c_1 + self.c_mu))))

        # ---------- dynamic state ----------
        self._init_state(sigma0)

    def _init_state(self, sigma0: float) -> None:
        """Initialise or reset all mutable CMA-ES state tensors."""
        dim = self.dim

        # Distribution mean: center of the search space
        self.mean = ((self.lb + self.ub) / 2).clone()

        # Step size: sigma0 * average span
        span = (self.ub - self.lb).mean().item()
        self.sigma = sigma0 * span

        # Sigma bounds (set by PhasedDFO, defaults to no-op range)
        self.sigma_min = 1e-30  # effectively no lower bound
        self.sigma_max = float("inf")  # effectively no upper bound

        # Covariance normalization: when True, _update_eigensystem divides
        # eigenvalues by their mean. Standalone CMA-ES normalizes every decomp.
        # PhasedDFO sets this to False and normalizes C externally at phase
        # boundaries only.
        self._normalize_on_decomp = True

        # Covariance matrix and its decomposition
        self.C = torch.eye(dim, device=self.device, dtype=self.dtype)
        self.B = torch.eye(dim, device=self.device, dtype=self.dtype)
        self.D_diag = torch.ones(dim, device=self.device, dtype=self.dtype)
        self.C_invsqrt = torch.eye(dim, device=self.device, dtype=self.dtype)

        # Evolution paths
        self.p_sigma = torch.zeros(dim, device=self.device, dtype=self.dtype)
        self.p_c = torch.zeros(dim, device=self.device, dtype=self.dtype)
        if self.path_memory > 0:
            self._path_vectors = torch.zeros(
                self.path_memory,
                dim,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._path_vectors = torch.empty(0, dim, device=self.device, dtype=self.dtype)
        self._path_count = 0
        self._path_pos = 0

        # Generation of last eigendecomposition
        self._decomp_gen = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return CMA-ES state as a serializable dict."""
        state = super().state_dict()
        state.update(
            {
                "C": self.C.clone(),
                "B": self.B.clone(),
                "D_diag": self.D_diag.clone(),
                "C_invsqrt": self.C_invsqrt.clone(),
                "mean": self.mean.clone(),
                "sigma": self.sigma,
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "p_sigma": self.p_sigma.clone(),
                "p_c": self.p_c.clone(),
                "path_memory": self.path_memory,
                "path_scale": self.path_scale,
                "path_line_samples": self.path_line_samples,
                "path_line_scale": self.path_line_scale,
                "_path_vectors": self._path_vectors.clone(),
                "_path_count": self._path_count,
                "_path_pos": self._path_pos,
                "_decomp_gen": self._decomp_gen,
                "_normalize_on_decomp": self._normalize_on_decomp,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore CMA-ES state from a dict produced by :meth:`state_dict`."""
        super().load_state_dict(state)
        self.C.copy_(state["C"])
        self.B.copy_(state["B"])
        self.D_diag.copy_(state["D_diag"])
        self.C_invsqrt.copy_(state["C_invsqrt"])
        self.mean.copy_(state["mean"])
        self.sigma = state["sigma"]
        self.sigma_min = state["sigma_min"]
        self.sigma_max = state["sigma_max"]
        self.p_sigma.copy_(state["p_sigma"])
        self.p_c.copy_(state["p_c"])
        self.path_memory = int(state.get("path_memory", self.path_memory))
        self.path_scale = float(state.get("path_scale", self.path_scale))
        self.path_line_samples = int(state.get("path_line_samples", self.path_line_samples))
        self.path_line_scale = float(state.get("path_line_scale", self.path_line_scale))
        if "_path_vectors" in state:
            path_vectors = state["_path_vectors"].to(device=self.device, dtype=self.dtype)
            self._path_vectors = path_vectors.clone()
            self.path_memory = int(self._path_vectors.shape[0])
        elif self.path_memory > 0:
            self._path_vectors = torch.zeros(
                self.path_memory,
                self.dim,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self._path_vectors = torch.empty(0, self.dim, device=self.device, dtype=self.dtype)
        self._path_count = int(state.get("_path_count", 0))
        self._path_pos = int(state.get("_path_pos", 0))
        self._decomp_gen = state["_decomp_gen"]
        self._normalize_on_decomp = state["_normalize_on_decomp"]

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------
    def ask(self) -> torch.Tensor:
        """Sample a new population from the current search distribution.

        Returns
        -------
        candidates : Tensor of shape ``(pop_size, dim)``
            Candidate solutions clamped to bounds.

        """
        pop_size = self.pop_size
        dim = self.dim

        if self.mirrored:
            pair_count = pop_size // 2
            base_z = self._randn(pair_count, dim)
            z = torch.cat([base_z, -base_z], dim=0)
            if pop_size % 2 == 1:
                z = torch.cat([z, self._randn(1, dim)], dim=0)
            # Shuffle to break pairing structure
            perm = self._randperm(z.shape[0])
            z = z[perm]
        else:
            z = self._randn(pop_size, dim)

        # Transform: y = B @ diag(D) @ z^T  =>  candidates = mean + sigma * y^T
        # BD = B * D_diag (broadcasting column-wise)
        BD = self.B * self.D_diag.unsqueeze(0)  # (dim, dim)
        y = z @ BD.T  # (pop_size, dim)
        if self.path_memory > 0 and self.path_scale > 0.0 and self._path_count > 0:
            active_paths = min(self._path_count, self.path_memory)
            path_basis = self._path_vectors[:active_paths]
            if self.mirrored:
                pair_count = pop_size // 2
                base_coeff = self._randn(pair_count, active_paths)
                coeff = torch.cat([base_coeff, -base_coeff], dim=0)
                if pop_size % 2 == 1:
                    coeff = torch.cat([coeff, self._randn(1, active_paths)], dim=0)
                coeff = coeff[self._randperm(coeff.shape[0])]
            else:
                coeff = self._randn(pop_size, active_paths)
            y = y + (self.path_scale / math.sqrt(active_paths)) * (coeff @ path_basis)
        if self.path_line_samples > 0 and self.path_line_scale > 0.0:
            line_count = min(self.path_line_samples, pop_size)
            path_norm = torch.linalg.vector_norm(self.p_c)
            safe_norm = torch.clamp(path_norm, min=torch.finfo(self.dtype).eps)
            path_dir = self.p_c / safe_norm
            line_len = torch.clamp(path_norm, max=math.sqrt(dim)) * self.path_line_scale
            line_step = path_dir * line_len
            signs = torch.ones(line_count, device=self.device, dtype=self.dtype)
            signs[1::2] = -1.0
            line_steps = signs.unsqueeze(1) * line_step.unsqueeze(0)
            valid_path = torch.isfinite(path_norm) & (path_norm > 1e-12)
            y[:line_count] = torch.where(valid_path, line_steps, y[:line_count])
        candidates = self.mean.unsqueeze(0) + self.sigma * y

        # Clamp to bounds
        candidates = clamp_to_bounds(candidates, self.lb, self.ub)

        # Store into pre-allocated workspace
        self.population.copy_(candidates)
        return candidates

    # ------------------------------------------------------------------
    # tell
    # ------------------------------------------------------------------
    def tell(self, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        """Update the search distribution from evaluated candidates.

        Parameters
        ----------
        candidates : Tensor of shape ``(pop_size, dim)``
            Solutions returned by the last ``ask()`` call.
        fitness : Tensor of shape ``(pop_size,)``
            Objective values (lower is better).

        """
        self.fitness.copy_(fitness)
        pop_size = self.pop_size
        mu = self.mu
        dim = self.dim

        # 1. Sort by fitness and select mu-best
        sorted_indices = fitness.argsort()
        best_indices = sorted_indices[:mu]

        # 2. Update mean
        old_mean = self.mean.clone()
        selected = candidates[best_indices]  # (mu, dim)
        self.mean = (self.weights.unsqueeze(1) * selected).sum(dim=0)

        # Mean displacement (unscaled)
        mean_diff = self.mean - old_mean  # (dim,)

        # 3. Update evolution paths
        #    p_sigma update (conjugate evolution path)
        invsqrt_diff = self.C_invsqrt @ mean_diff / self.sigma  # (dim,)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(
            max(self.c_sigma * (2 - self.c_sigma) * self.mu_eff, 0.0),
        ) * invsqrt_diff

        #    Heaviside function for stalling detection
        gen = self._generation + 1  # 1-based for this formula
        lhs = torch.linalg.norm(self.p_sigma).item() / math.sqrt(
            1 - (1 - self.c_sigma) ** (2 * gen),
        )
        rhs = (1.4 + 2.0 / (dim + 1)) * self.chi_n
        h_sigma = 1.0 if lhs < rhs else 0.0

        #    p_c update (evolution path for rank-one update)
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            max(self.c_c * (2 - self.c_c) * self.mu_eff, 0.0),
        ) * mean_diff / self.sigma
        self._record_path_vector(self.p_c)

        # 4. Update covariance matrix
        y_selected = (selected - old_mean.unsqueeze(0)) / self.sigma  # (mu, dim)

        # Rank-mu update: weighted outer products of selected steps
        # rank_mu = y^T @ diag(w) @ y
        rank_mu = (y_selected * self.weights.unsqueeze(1)).T @ y_selected  # (dim, dim)

        # Rank-one update: outer product of p_c
        rank_one = torch.outer(self.p_c, self.p_c)

        # Correction for h_sigma == 0 (stalled step-size)
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)

        # Active CMA: compute negative rank update and balancing term
        if self.active:
            mu_neg = self._mu_neg
            worst_indices = sorted_indices[pop_size - mu_neg :].flip(0)
            y_worst = (candidates[worst_indices] - old_mean.unsqueeze(0)) / self.sigma
            rank_mu_neg = (y_worst * self._neg_weights.unsqueeze(1)).T @ y_worst
            c_mu_neg = self._c_mu_neg
        else:
            rank_mu_neg = torch.zeros_like(self.C)
            c_mu_neg = 0.0

        # Covariance update with active CMA trace-balancing term (c_mu_neg in old_C coeff)
        old_C = self.C
        self.C = (
            (1 - self.c_1 - self.c_mu + c_mu_neg) * old_C
            + self.c_1 * (rank_one + delta_h * old_C)
            + self.c_mu * rank_mu
            - c_mu_neg * rank_mu_neg
        )

        # 5. Update sigma (cumulative step-size adaptation)
        ps_norm = torch.linalg.norm(self.p_sigma).item()
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1),
        )
        self.sigma = min(max(self.sigma, self.sigma_min), self.sigma_max)

        # 6. Eigendecomposition (periodic, for efficiency)
        self._generation += 1
        if self._generation - self._decomp_gen >= self._decomp_frequency:
            self._update_eigensystem()
            self._decomp_gen = self._generation

        # 7. Track global best
        self._update_best(candidates, fitness)

    def _record_path_vector(self, path: torch.Tensor) -> None:
        """Store one normalized covariance path for limited-memory sampling."""
        if self.path_memory <= 0:
            return
        path_norm = torch.linalg.vector_norm(path)
        if not torch.isfinite(path_norm) or float(path_norm.item()) <= 1e-12:
            return
        self._path_vectors[self._path_pos] = path / path_norm
        self._path_pos = (self._path_pos + 1) % self.path_memory
        self._path_count = min(self.path_memory, self._path_count + 1)

    # ------------------------------------------------------------------
    # eigensystem
    # ------------------------------------------------------------------
    def _update_eigensystem(self) -> None:
        """Recompute the eigendecomposition of the covariance matrix C.

        Enforces symmetry and numerical stability before decomposition.
        Falls back to CPU for eigh when the device lacks native support (e.g. MPS).
        """
        # Enforce symmetry
        self.C = (self.C + self.C.T) / 2

        # Eigendecomposition (eigh returns ascending eigenvalues).
        # MPS does not support eigh natively; fall back to CPU.
        if self.device.type not in ("cpu", "cuda"):
            C_cpu = self.C.to("cpu")
            eigenvalues_cpu, B_cpu = torch.linalg.eigh(C_cpu)
            eigenvalues = eigenvalues_cpu.to(self.device)
            self.B = B_cpu.to(self.device)
        else:
            eigenvalues, self.B = torch.linalg.eigh(self.C)

        # Numerical safety: clamp small/negative eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)

        # Normalize covariance shape when requested. Normalize
        # only at phase boundaries (not every decomp) to avoid double-normalizing
        # during CMA-ES iterations, which would warp the learned orientation.
        if self._normalize_on_decomp:
            mean_eig = eigenvalues.mean()
            if mean_eig > 1e-12:
                eigenvalues = eigenvalues / mean_eig

        self.D_diag = torch.sqrt(eigenvalues)

        # C^{-1/2} = B @ diag(1/D) @ B^T
        self.C_invsqrt = self.B @ torch.diag(1.0 / self.D_diag) @ self.B.T

        # Reconstruct C from the clamped decomposition for consistency
        self.C = self.B @ torch.diag(eigenvalues) @ self.B.T

    # ------------------------------------------------------------------
    # restart (IPOP support)
    # ------------------------------------------------------------------
    def restart(
        self,
        new_pop_size: int | None = None,
        mean: torch.Tensor | None = None,
        sigma: float | None = None,
        C_init: torch.Tensor | None = None,
    ) -> None:
        """Restart CMA-ES, optionally with new parameters.

        Used by PhasedDFO for IPOP restarts (population doubling).

        Parameters
        ----------
        new_pop_size : int | None
            New population size (doubles for IPOP). If ``None``, keep current.
        mean : Tensor | None
            New distribution mean. If ``None``, reset to search-space center.
        sigma : float | None
            New step size. If ``None``, reuse the initial sigma0 fraction.
        C_init : Tensor | None
            New covariance matrix. If ``None``, reset to identity.

        """
        if new_pop_size is not None and new_pop_size != self.pop_size:
            self.pop_size = new_pop_size

            # Recompute mu and weights for the new population size
            self.mu = new_pop_size // 2
            mu = self.mu
            raw = torch.log(torch.tensor((new_pop_size + 1) / 2, dtype=self.dtype)) - torch.log(
                torch.arange(1, mu + 1, dtype=self.dtype),
            )
            self.weights = (raw / raw.sum()).to(device=self.device)
            self.mu_eff = (1.0 / (self.weights**2).sum()).item()

            # Recompute strategy parameters that depend on mu_eff
            mu_eff = self.mu_eff
            dim = self.dim
            self.c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            self.d_sigma = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
            self.c_c = (mu_eff + 2) / (dim + 4 + 2 * mu_eff / dim)
            self.c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
            self.c_mu = min(
                1 - self.c_1,
                2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff),
            )
            self._decomp_frequency = max(
                1,
                math.floor(1 / (10 * dim * (self.c_1 + self.c_mu))),
            )

            # Re-allocate workspace tensors for the new population size
            self.population = torch.empty(
                new_pop_size,
                self.dim,
                device=self.device,
                dtype=self.dtype,
            )
            self.fitness = torch.full(
                (new_pop_size,),
                float("inf"),
                device=self.device,
                dtype=self.dtype,
            )

            # Update active CMA weights if applicable
            if self.active:
                mu_neg = min(self.mu, new_pop_size - self.mu)
                self._mu_neg = mu_neg
                neg_raw = torch.log(torch.tensor(mu_neg + 1, dtype=self.dtype)) - torch.log(
                    torch.arange(1, mu_neg + 1, dtype=self.dtype),
                )
                self._neg_weights = (neg_raw / neg_raw.sum()).to(device=self.device)

        # Widen sigma_max for restart phases (CMA_ES_RESTART_SIGMA_MAX * span).
        # Only applies when sigma bounds were explicitly set (e.g. by PhasedDFO);
        # the default no-op bounds (inf) are left untouched.
        if math.isfinite(self.sigma_max):
            span = (self.ub - self.lb).mean().item()
            self.sigma_max = 0.30 * span

        # Reset evolution paths
        self.p_sigma.zero_()
        self.p_c.zero_()
        self._path_vectors.zero_()
        self._path_count = 0
        self._path_pos = 0

        # Set mean
        if mean is not None:
            self.mean = mean.to(device=self.device, dtype=self.dtype).clone()
        else:
            self.mean = ((self.lb + self.ub) / 2).clone()

        # Set sigma
        if sigma is not None:
            self.sigma = sigma
        else:
            span = (self.ub - self.lb).mean().item()
            self.sigma = 0.3 * span  # default fraction

        # Set covariance
        if C_init is not None:
            self.C = C_init.to(device=self.device, dtype=self.dtype).clone()
        else:
            self.C = torch.eye(self.dim, device=self.device, dtype=self.dtype)

        # Reset generation tracking
        self._generation = 0
        self._decomp_gen = 0

        # Recompute eigensystem from the (possibly new) covariance
        self._update_eigensystem()
