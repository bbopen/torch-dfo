"""GPU-native K-branch diagonal-plus-low-rank CMA portfolio.

Covariance parametrisation per branch:

    C = D + V Vᵀ    D = diag(exp(log_d)),  V ∈ ℝⁿˣᵏ

Sampling:   x = μ + σ * (√d ⊙ ε  +  V @ w),   ε ~ N(0,Iₙ),  w ~ N(0,Iₖ)

Covariance update (LM-CMA style, Loshchilov 2014):
  - V  = FIFO queue of last k p_c evolution paths (rank-1 direction history).
  - D  = EWA of squared mean shifts (diagonal scale adaptation).
  - CSA step-size with diagonal-only whitening (C ≈ D in evolution path norm).

All K branches run as a fused (K, …) batch — one ask() + tell() covers all
branches with no Python loop.  No eigendecomposition anywhere: all operations
stay on the target device (CUDA/MPS/CPU) without CPU fallback.

References
----------
Loshchilov (2014). "A Computationally Efficient Limited Memory CMA-ES for
    Large Scale Optimization."  GECCO 2014.
Glasmachers et al. (2010). "Exponential Natural Evolution Strategies."
    GECCO 2010.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from torch_dfo.utils import (
    clamp_to_bounds,
    make_generator,
    mean_span,
    normalize_bounds,
    resolve_device,
)

_DLR_K_LAMBDAS: tuple[int, ...] = (24, 12, 12, 12)
_DLR_K_SIGMA_FRACS: tuple[float, ...] = (0.200, 0.043, 0.0093, 0.002)
_DEFAULT_SIGMA_FACTORS: tuple[float, ...] = (1.4, 1.6, 2.0, 1.2)


def _default_sigma_factors(k_branches: int) -> tuple[float, ...]:
    """Return per-branch NIPOP sigma reduction factors for any K >= 1."""
    if k_branches <= len(_DEFAULT_SIGMA_FACTORS):
        return _DEFAULT_SIGMA_FACTORS[:k_branches]
    return tuple(1.2 * ((2.0 / 1.2) ** (i / (k_branches - 1))) for i in range(k_branches))


def _compute_branch_rates(n: int, mue: float) -> tuple[float, float, float, float, float]:
    """Compute Hansen (2016, Table 1) strategy rates for one DLR branch."""
    c_s = (mue + 2.0) / (n + mue + 5.0)
    d_s = 1.0 + 2.0 * max(0.0, math.sqrt((mue - 1.0) / (n + 1.0)) - 1.0) + c_s
    c_c = (4.0 + mue / n) / (n + 4.0 + 2.0 * mue / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mue)
    cmu = min(1.0 - c1, 2.0 * (mue - 2.0 + 1.0 / mue) / ((n + 2.0) ** 2 + mue))
    return c_s, d_s, c_c, c1, cmu


class DLRPortfolio:
    """K-branch diagonal-plus-low-rank CMA portfolio.  GPU-native, no eigendecomp.

    Each branch maintains a covariance approximation of the form:

        C = D + VVᵀ,   D = diag(exp(log_d)),  V ∈ ℝⁿˣᵏ

    Sampling is :math:`x = μ + σ(√d ⊙ ε + (rms(√d)/√k) Vw)` with :math:`ε ∼ 𝒩(0,Iₙ)` and
    :math:`w ∼ 𝒩(0,Iₖ)`.  All K branches are evaluated in a single fused
    ``ask()`` / ``tell()`` pass with no Python loop.  No eigendecomposition
    is performed anywhere — all operations stay on the target device.

    Parameters
    ----------
    dim : int
        Search-space dimensionality.
    lb : Tensor
        Lower bounds, shape ``(dim,)``.
    ub : Tensor
        Upper bounds, shape ``(dim,)``.
    lambdas : Sequence[int]
        Population size per branch.  Length determines K (number of branches).
    sigma_fracs : Sequence[float]
        Initial step-size σ per branch, expressed as a fraction of the
        search-space span ``(ub - lb).mean()``.
    k_rank : int, optional
        Low-rank dimension k (columns of V).  Defaults to ``min(dim//2, 16)``.
    sigma_factors : Sequence[float], optional
        Per-branch σ multipliers applied at NIPOP restarts.
        Defaults to ``[1.4, 1.6, 2.0, 1.2][:K]`` for ``K <= 4``, and a geometric
        spread across ``[1.2, 2.0]`` for ``K > 4``.
    cma_budget : int, optional
        Total CMA budget used to cap ``lam_max``.
    device : torch.device
        Target compute device (CUDA, CPU, or MPS).
    dtype : torch.dtype
        Floating-point dtype.  float64 recommended for numerical stability.
    rng : torch.Generator
        Pre-seeded random generator.  Must live on CPU (CUDA generators are
        not yet supported for per-branch draws).

    Attributes
    ----------
    K : int
        Number of parallel branches.
    k_rank : int
        Effective low-rank dimension.
    lam_max : int
        Maximum population size per branch (NIPOP ceiling).

    Notes
    -----
    Covariance update follows Loshchilov (2014) LM-CMA: V is a FIFO queue
    of the last *k* evolution-path directions; D is an EWA of squared mean
    shifts.  Step-size adaptation uses CSA with diagonal-only whitening.

    References
    ----------
    Loshchilov, I. (2014). A Computationally Efficient Limited Memory CMA-ES
        for Large Scale Optimization. GECCO 2014.

    Examples
    --------
    >>> import torch
    >>> from torch_dfo.dlr_cma import DLRPortfolio
    >>> dlr = DLRPortfolio(
    ...     dim=20, bounds=5.0,
    ...     lambdas=(24, 12, 12, 12),
    ...     sigma_fracs=(0.2, 0.043, 0.0093, 0.002),
    ...     seed=42,
    ... )
    >>> candidates = dlr.ask()          # shape (60, 20) — sum of lambdas
    >>> fitness = (candidates ** 2).sum(dim=-1)
    >>> dlr.tell(candidates, fitness)
    """

    def __init__(
        self,
        dim: int,
        bounds: float | tuple[float, float] | tuple[Tensor, Tensor] | None = None,
        *,
        lb: Tensor | None = None,
        ub: Tensor | None = None,
        lambdas: Sequence[int] = (12, 12),
        sigma_fracs: Sequence[float] = (0.3, 0.1),
        k_rank: int | None = None,
        sigma_factors: Sequence[float] | None = None,
        cma_budget: int | None = None,
        seed: int | None = None,
        rng: torch.Generator | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        device = resolve_device(device)

        has_lbub = lb is not None or ub is not None
        if bounds is not None and has_lbub:
            raise ValueError(
                "Pass either bounds= or (lb=, ub=), not both. "
                "lb/ub kwargs are deprecated; prefer bounds=."
            )
        if bounds is None and not has_lbub:
            raise ValueError("Must pass bounds= (preferred) or both lb= and ub=.")

        if bounds is not None:
            lb_t, ub_t = normalize_bounds(bounds, dim, device, dtype)
        else:
            if lb is None or ub is None:
                raise ValueError(
                    "When using the deprecated lb/ub API, both lb and ub are required."
                )
            warnings.warn(
                "lb/ub kwargs are deprecated; pass bounds= instead",
                DeprecationWarning,
                stacklevel=2,
            )
            lb_t, ub_t = normalize_bounds((lb, ub), dim, device, dtype)

        if seed is not None and rng is not None:
            raise ValueError("Pass either seed= or rng=, not both.")
        if rng is None:
            rng = make_generator(seed, device)

        if len(lambdas) == 0:
            raise ValueError("lambdas must be non-empty")
        if any(lam < 2 for lam in lambdas):
            raise ValueError("Each branch in lambdas must have population size >= 2")
        if len(sigma_fracs) != len(lambdas):
            raise ValueError(
                f"sigma_fracs length ({len(sigma_fracs)}) must match "
                f"lambdas length ({len(lambdas)})"
            )

        self.dim = dim
        self.K = len(lambdas)
        self.lambdas = list(lambdas)
        self.sigma_fracs = list(sigma_fracs)
        self.device = device
        self.dtype = dtype
        self._rng = rng
        self._rng_seed: int | None = (
            seed
            if seed is not None
            else (int(rng.initial_seed()) if hasattr(rng, "initial_seed") else None)
        )
        self._rng_device = (
            torch.device(rng.device) if hasattr(rng, "device") else torch.device("cpu")
        )
        self._last_active: list[int] = list(range(len(lambdas)))

        if k_rank is None:
            k_rank = max(4, min(dim // 2, 16))
        self.k_rank = k_rank

        self.lb = lb_t
        self.ub = ub_t
        span = mean_span(self.lb, self.ub)

        K = self.K
        self.lam_default = 4 + math.floor(3 * math.log(max(dim, 2)))
        if cma_budget is not None and cma_budget < 2 * K:
            raise ValueError(
                f"cma_budget ({cma_budget}) must be at least 2 * K ({2 * K}) "
                "to allocate at least 2 evaluations per branch"
            )
        _cma_budget = cma_budget if cma_budget is not None else 100000
        self.lam_max = max(2, min(dim * 128, _cma_budget // (K * 4)))
        self.sigma_default = sigma_fracs[0] * span
        if sigma_factors is not None:
            if len(sigma_factors) != K:
                raise ValueError(f"sigma_factors length ({len(sigma_factors)}) must match K ({K})")
            self.sigma_factors: tuple[float, ...] = tuple(sigma_factors)
        else:
            self.sigma_factors = _default_sigma_factors(K)
        self._nipop_level: list[int] = [0] * K

        K, n = self.K, dim
        lam_max = max(lambdas)
        self._lam_max = lam_max

        self._mu = [lam // 2 for lam in lambdas]
        mu_max = max(self._mu)
        self._mu_max = mu_max

        weights = torch.zeros(K, mu_max, device=device, dtype=dtype)
        mu_eff = []
        for ki in range(K):
            mu_k = self._mu[ki]
            lam_k = lambdas[ki]
            raw = torch.log(torch.tensor((lam_k + 1) / 2.0, dtype=dtype)) - torch.log(
                torch.arange(1, mu_k + 1, dtype=dtype)
            )
            raw = raw.clamp(min=0.0)
            w_k = raw / raw.sum()
            weights[ki, :mu_k] = w_k
            mu_eff.append((1.0 / (w_k**2).sum()).item())
        self._weights = weights
        self._mu_eff = mu_eff

        self._c_sigma = []
        self._d_sigma = []
        self._c_c = []
        self._c_1 = []
        self._c_mu = []
        self._chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        for ki in range(K):
            c_s, d_s, c_c_, c1, cmu = _compute_branch_rates(n, mu_eff[ki])
            self._c_sigma.append(c_s)
            self._d_sigma.append(d_s)
            self._c_c.append(c_c_)
            self._c_1.append(c1)
            self._c_mu.append(cmu)

        center = (0.5 * self.lb + 0.5 * self.ub).unsqueeze(0).expand(K, -1)
        self.means = center.clone()
        self.sigmas = torch.tensor([sf * span for sf in sigma_fracs], device=device, dtype=dtype)
        self.log_d = torch.zeros(K, n, device=device, dtype=dtype)
        self.V = torch.zeros(K, n, k_rank, device=device, dtype=dtype)
        self.p_sigma = torch.zeros(K, n, device=device, dtype=dtype)
        self.p_c = torch.zeros(K, n, device=device, dtype=dtype)
        self._generation = torch.zeros(K, dtype=torch.long, device=device)

        self._rank_mu_col = [0] * K
        self._oja_alpha = 0.3

        self.best_f_per_branch = torch.full((K,), float("inf"), device=device, dtype=dtype)
        self.stag_count = torch.zeros(K, dtype=torch.long, device=device)

        self.best_solution = (0.5 * self.lb + 0.5 * self.ub).clone()
        self.best_fitness = torch.tensor(float("inf"), device=device, dtype=dtype)

    def ask(self, remaining_budget: int | None = None) -> Tensor:
        """Sample active branches.  Returns (total_pop, dim) candidates.

        Parameters
        ----------
        remaining_budget : int | None
            If provided, include branches in configured order while their
            combined population fits this budget.  A branch that does not fit
            is skipped, so a later smaller branch can still be included.
            Returns an empty ``(0, dim)`` tensor when no branch fits.
        """
        K, n, k = self.K, self.dim, self.k_rank
        device, dtype = self.device, self.dtype

        if remaining_budget is not None:
            active = []
            used = 0
            for ki, lam_k in enumerate(self.lambdas):
                if used + lam_k <= remaining_budget:
                    active.append(ki)
                    used += lam_k
        else:
            active = list(range(K))

        self._last_active = active

        if not active:
            return torch.empty(0, n, device=device, dtype=dtype)

        lam_max = self._lam_max

        rdev = self._rng_device
        half = (lam_max + 1) // 2
        eps_base = torch.randn(K, half, n, device=rdev, dtype=dtype, generator=self._rng)
        w_base = torch.randn(K, half, k, device=rdev, dtype=dtype, generator=self._rng)
        if rdev != device:
            eps_base = eps_base.to(device)
            w_base = w_base.to(device)

        eps = torch.cat([eps_base, -eps_base], dim=1)[:, :lam_max, :]
        w = torch.cat([w_base, -w_base], dim=1)[:, :lam_max, :]

        d_val = torch.exp(self.log_d)
        sqrt_d = torch.sqrt(d_val)
        v_scale = torch.sqrt(d_val.mean(dim=-1, keepdim=True)) / math.sqrt(max(k, 1))
        Vw = torch.bmm(w, self.V.transpose(-1, -2)) * v_scale.unsqueeze(-1)
        y = sqrt_d.unsqueeze(1) * eps + Vw
        candidates_full = self.means.unsqueeze(1) + self.sigmas.view(K, 1, 1) * y

        parts = []
        for ki in active:
            lam_k = self.lambdas[ki]
            c_k = candidates_full[ki, :lam_k, :]
            c_k = clamp_to_bounds(c_k, self.lb, self.ub)
            parts.append(c_k)
        return torch.cat(parts, dim=0)

    def tell(self, candidates: Tensor, fitness: Tensor) -> None:
        """Update all K branches from evaluated (total_pop, dim) candidates.

        Parameters
        ----------
        candidates : (total_pop, n)
        fitness    : (total_pop,)  lower is better
        """
        K, n, k = self.K, self.dim, self.k_rank
        active = getattr(self, "_last_active", list(range(K)))
        offset = 0

        for ki in active:
            lam_k = self.lambdas[ki]
            mu_k = self._mu[ki]
            cands_k = candidates[offset : offset + lam_k]
            fit_k = fitness[offset : offset + lam_k]
            offset += lam_k

            if cands_k.shape[0] == 0:
                continue

            sorted_idx = fit_k.argsort()
            best_idx = sorted_idx[:mu_k]
            selected = cands_k[best_idx]
            w_k = self._weights[ki, :mu_k]

            finfo = torch.finfo(self.dtype)
            old_mean = self.means[ki].clone()
            displacements = selected - old_mean.unsqueeze(0)
            mean_diff = (w_k.unsqueeze(1) * displacements).sum(dim=0)
            new_mean = old_mean + mean_diff
            self.means[ki] = new_mean

            span_max = float((self.ub - self.lb).max().item())
            coord_scale = max(float(old_mean.abs().max().item()), span_max)
            sigma_floor = max(finfo.tiny * 1e6, 10.0 * finfo.eps * coord_scale)
            sigma_ceil = max(sigma_floor * 10.0, 10.0 * max(span_max, coord_scale))
            sigma_old = float(self.sigmas[ki].item())
            eff_sigma = max(sigma_old, sigma_floor)

            y_w = mean_diff / eff_sigma

            c_s = self._c_sigma[ki]
            d_s = self._d_sigma[ki]
            mue = self._mu_eff[ki]
            invsqrt_d = torch.exp(-0.5 * self.log_d[ki])
            whitened = invsqrt_d * y_w
            p_sigma_new = (1 - c_s) * self.p_sigma[ki] + math.sqrt(
                max(c_s * (2 - c_s) * mue, 0.0)
            ) * whitened
            if torch.isfinite(p_sigma_new).all():
                self.p_sigma[ki] = p_sigma_new
            ps_norm = torch.linalg.norm(self.p_sigma[ki]).item()
            gen_k = self._generation[ki].item() + 1
            exp_arg = (c_s / d_s) * (ps_norm / self._chi_n - 1)
            exp_arg = max(-20.0, min(20.0, exp_arg))
            next_sigma = eff_sigma * math.exp(exp_arg)
            self.sigmas[ki] = min(max(next_sigma, sigma_floor), sigma_ceil)

            denom = math.sqrt(1 - (1 - c_s) ** (2 * gen_k))
            lhs = ps_norm / (denom + 1e-30)
            rhs = (1.4 + 2.0 / (n + 1)) * self._chi_n
            h_sigma = 1.0 if lhs < rhs else 0.0

            c_c_ = self._c_c[ki]
            p_c_new = (1 - c_c_) * self.p_c[ki] + h_sigma * math.sqrt(
                max(c_c_ * (2 - c_c_) * mue, 0.0)
            ) * y_w
            if torch.isfinite(p_c_new).all():
                self.p_c[ki] = p_c_new

            c1 = self._c_1[ki]
            cmu = self._c_mu[ki]
            delta_h = (1 - h_sigma) * c_c_ * (2 - c_c_)
            d_old = torch.exp(self.log_d[ki])

            y_sel = displacements / eff_sigma
            y_scale = float(y_sel.abs().max().item())
            if y_scale > 1.0 and math.isfinite(y_scale):
                y_normed = y_sel / y_scale
                diag_mu = (w_k.unsqueeze(1) * (y_normed**2)).sum(dim=0) * (y_scale**2)
            else:
                y_normed = y_sel
                diag_mu = (w_k.unsqueeze(1) * (y_sel**2)).sum(dim=0)

            diag_1 = self.p_c[ki] ** 2

            d_cap = min(1e30, finfo.max**0.25)
            d_new = (1 - c1 - cmu + c1 * delta_h) * d_old + c1 * diag_1 + cmu * diag_mu
            d_new = torch.where(torch.isfinite(d_new), d_new, d_old).clamp(min=1e-30, max=d_cap)
            self.log_d[ki] = torch.log(d_new)

            k_half = k // 2

            if k_half > 0:
                rank1 = torch.roll(self.V[ki, :, :k_half], -1, dims=-1)
                pc_norm = self.p_c[ki].norm()
                if torch.isfinite(pc_norm) and pc_norm > 1e-30:
                    rank1[:, -1] = self.p_c[ki] / pc_norm
                else:
                    rank1[:, -1] = 0.0
                self.V[ki, :, :k_half] = rank1

            k_mu = k - k_half
            if k_mu > 0:
                mu_col = self._rank_mu_col[ki]
                v_old = self.V[ki, :, k_half + mu_col]
                v_old_norm = v_old.norm()

                if not torch.isfinite(v_old_norm) or v_old_norm < 1e-30:
                    v_new = (w_k.unsqueeze(1) * y_normed).sum(dim=0)
                    v_norm = v_new.norm()
                    if torch.isfinite(v_norm) and v_norm > 1e-30:
                        self.V[ki, :, k_half + mu_col] = v_new / v_norm
                else:
                    dots = y_normed @ v_old
                    v_new = (w_k * dots) @ y_normed
                    v_norm = v_new.norm()
                    if torch.isfinite(v_norm) and v_norm > 1e-30:
                        v_new = v_new / v_norm
                        v_blended = (1 - self._oja_alpha) * v_old + self._oja_alpha * v_new
                        vb_norm = v_blended.norm()
                        if torch.isfinite(vb_norm) and vb_norm > 1e-30:
                            self.V[ki, :, k_half + mu_col] = v_blended / vb_norm
                self._rank_mu_col[ki] = (mu_col + 1) % k_mu

            safe_fit_k = (
                torch.where(torch.isnan(fit_k), float("inf"), fit_k)
                if fit_k.is_floating_point()
                else fit_k
            )
            best_fit_k = safe_fit_k.min().item()
            if best_fit_k < self.best_fitness.item():
                bi = safe_fit_k.argmin()
                self.best_fitness = safe_fit_k[bi].to(dtype=self.dtype).clone()
                self.best_solution = cands_k[bi].clone()

            if best_fit_k < self.best_f_per_branch[ki].item() - 1e-14:
                self.best_f_per_branch[ki] = torch.tensor(
                    best_fit_k, device=self.device, dtype=self.dtype
                )
                self.stag_count[ki] = 0
            else:
                self.stag_count[ki] += 1

            self._generation[ki] += 1

    def _resize_branch(self, ki: int, new_lam: int) -> None:
        """Update branch ki for a new population size.

        Recomputes lambda, mu, weights, mu_eff, and all derived CMA strategy
        rates (c_sigma, d_sigma, c_c, c_1, c_mu).  Also widens the shared
        weights tensor and updates _lam_max / _mu_max if needed.
        """
        n = self.dim
        device, dtype = self.device, self.dtype

        self.lambdas[ki] = new_lam
        new_mu = new_lam // 2
        self._mu[ki] = new_mu

        raw = torch.log(torch.tensor((new_lam + 1) / 2.0, dtype=dtype)) - torch.log(
            torch.arange(1, new_mu + 1, dtype=dtype)
        )
        raw = raw.clamp(min=0.0)
        w_k = raw / raw.sum()
        mu_eff_k = (1.0 / (w_k**2).sum()).item()
        self._mu_eff[ki] = mu_eff_k

        new_mu_max = max(self._mu)
        if new_mu_max > self._mu_max:
            old_w = self._weights
            self._weights = torch.zeros(self.K, new_mu_max, device=device, dtype=dtype)
            self._weights[:, : old_w.shape[1]] = old_w
            self._mu_max = new_mu_max
        self._weights[ki, :] = 0.0
        self._weights[ki, :new_mu] = w_k.to(device=device)

        self._lam_max = max(self.lambdas)

        c_s, d_s, c_c_, c1, cmu = _compute_branch_rates(n, mu_eff_k)
        self._c_sigma[ki] = c_s
        self._d_sigma[ki] = d_s
        self._c_c[ki] = c_c_
        self._c_1[ki] = c1
        self._c_mu[ki] = cmu

    def restart_branch(self, ki: int, x0: Tensor, nipop_level: int = 0) -> None:
        """Restart branch ki at NIPOP level with scaled lambda and sigma.

        At level 0 (default), behaves identically to the original restart.
        At higher levels, lambda doubles and sigma shrinks per the NIPOP schedule::

            lambda = min(lam_default * 2^level, lam_max)
            sigma  = sigma_default / (sigma_factors[ki] ^ level)

        Parameters
        ----------
        ki : int
            Branch index.
        x0 : Tensor
            New mean (dim,).
        nipop_level : int
            NIPOP restart level.  0 = base, each increment doubles lambda.
        """
        n, k = self.dim, self.k_rank
        self._nipop_level[ki] = nipop_level

        new_lam = min(self.lam_default * (2**nipop_level), self.lam_max)
        new_sigma = self.sigma_default / (self.sigma_factors[ki] ** nipop_level)

        if new_lam != self.lambdas[ki]:
            self._resize_branch(ki, new_lam)

        self.means[ki] = x0.clone()
        self.sigmas[ki] = new_sigma
        self.log_d[ki] = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.V[ki] = torch.zeros(n, k, device=self.device, dtype=self.dtype)
        self.p_sigma[ki] = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.p_c[ki] = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.best_f_per_branch[ki] = torch.tensor(
            float("inf"), device=self.device, dtype=self.dtype
        )
        self._generation[ki] = 0
        self.stag_count[ki] = 0
        self._rank_mu_col[ki] = 0

    def state_dict(self) -> dict[str, Any]:
        """Return DLRPortfolio state as a serializable dict.

        All tensors are cloned so mutating the returned dict does not
        affect the portfolio.
        """
        return {
            "means": self.means.clone(),
            "sigmas": self.sigmas.clone(),
            "log_d": self.log_d.clone(),
            "V": self.V.clone(),
            "p_sigma": self.p_sigma.clone(),
            "p_c": self.p_c.clone(),
            "_generation": self._generation.clone(),
            "stag_count": self.stag_count.clone(),
            "best_f_per_branch": self.best_f_per_branch.clone(),
            "best_solution": self.best_solution.clone(),
            "best_fitness": self.best_fitness.clone(),
            "_nipop_level": list(self._nipop_level),
            "_rank_mu_col": list(self._rank_mu_col),
            "_last_active": list(self._last_active),
            "lambdas": list(self.lambdas),
            "_rng_state": self._rng.get_state(),
            "_rng_seed": self._rng_seed,
            "_saved_device_type": self._rng_device.type,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore DLRPortfolio state from a dict produced by :meth:`state_dict`."""
        self.means.copy_(state["means"])
        self.sigmas.copy_(state["sigmas"])
        self.log_d.copy_(state["log_d"])
        self.V.copy_(state["V"])
        self.p_sigma.copy_(state["p_sigma"])
        self.p_c.copy_(state["p_c"])
        self._generation.copy_(state["_generation"])
        self.stag_count.copy_(state["stag_count"])
        self.best_f_per_branch.copy_(state["best_f_per_branch"])
        self.best_solution.copy_(state["best_solution"])
        self.best_fitness.copy_(state["best_fitness"])
        self._nipop_level = list(state["_nipop_level"])
        self._rank_mu_col = list(state["_rank_mu_col"])
        self._last_active = list(state.get("_last_active", range(self.K)))
        for ki, new_lam in enumerate(state["lambdas"]):
            if new_lam != self.lambdas[ki]:
                self._resize_branch(ki, new_lam)
        saved_device_type = state.get("_saved_device_type", self._rng_device.type)
        if "_rng_seed" in state:
            self._rng_seed = state["_rng_seed"]
        if saved_device_type == self._rng_device.type:
            self._rng.set_state(state["_rng_state"])
        else:
            seed = state.get("_rng_seed")
            if seed is not None:
                self._rng.manual_seed(seed)

    def total_pop(self) -> int:
        return sum(self.lambdas)
