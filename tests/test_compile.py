"""Smoke tests for torch.compile compatibility.

Verifies that each optimizer's ask() method can be compiled with the 'eager'
backend and produces numerically identical results to the non-compiled version.
"""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import SMOKE_F_INIT_CMAES
from torch_dfo import CMAES, SHADE, NelderMead, PhasedDFO, sphere


@pytest.fixture
def cpu_device() -> torch.device:
    """torch.compile tests run on CPU (float64, best backend support)."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Parametrize over all ask/tell optimizers
# ---------------------------------------------------------------------------
OPTIMIZER_FACTORIES = {
    "CMAES": lambda d: CMAES(dim=5, bounds=5.0, seed=42, device=d),
    "SHADE": lambda d: SHADE(dim=5, bounds=5.0, seed=42, device=d),
    "NelderMead": lambda d: NelderMead(dim=5, bounds=5.0, seed=42, device=d),
    "PhasedDFO": lambda d: PhasedDFO(dim=5, bounds=5.0, budget=500, seed=42, device=d),
}


@pytest.fixture(params=OPTIMIZER_FACTORIES.keys())
def optimizer_name(request: pytest.FixtureRequest) -> str:
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCompileSmoke:
    """torch.compile smoke tests for each optimizer."""

    def test_compiled_ask_tell_loop(self, optimizer_name: str, cpu_device: torch.device) -> None:
        """Run 10 compiled ask/tell iterations without error.

        Note: fullgraph=True is not used because all optimizers hit a graph
        break on torch.Generator (not convertible to a compile proxy).
        The eager backend with fullgraph=False still validates that compiled
        code paths execute correctly.
        """
        opt = OPTIMIZER_FACTORIES[optimizer_name](cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="eager")
        for _ in range(10):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            f = sphere(x)
            opt.tell(x, f)
        best_x, best_f = opt.best()
        assert best_f.item() < SMOKE_F_INIT_CMAES, (
            "compiled optimizer produced a best_fitness no better than random init"
        )
        assert best_x.shape == (5,)

    def test_compiled_matches_eager(self, optimizer_name: str, cpu_device: torch.device) -> None:
        """Compiled (eager backend) produces identical results to non-compiled."""
        n_iters = 10

        # Eager run
        opt_eager = OPTIMIZER_FACTORIES[optimizer_name](cpu_device)
        for _ in range(n_iters):
            x = opt_eager.ask()
            if x.shape[0] == 0:
                break
            opt_eager.tell(x, sphere(x))
        _, f_eager = opt_eager.best()

        # Compiled run (same seed)
        opt_compiled = OPTIMIZER_FACTORIES[optimizer_name](cpu_device)
        compiled_ask = torch.compile(opt_compiled.ask, backend="eager")
        for _ in range(n_iters):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            opt_compiled.tell(x, sphere(x))
        _, f_compiled = opt_compiled.best()

        assert torch.allclose(f_eager, f_compiled), (
            f"Eager ({f_eager.item()}) != Compiled ({f_compiled.item()})"
        )

    def test_compiled_fitness_improves(self, optimizer_name: str, cpu_device: torch.device) -> None:
        """Compiled optimizer must strictly improve fitness over 10 iterations."""
        opt = OPTIMIZER_FACTORIES[optimizer_name](cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="eager")

        x = compiled_ask()
        if x.shape[0] == 0:
            pytest.skip("Optimizer returned empty candidates on first ask")
        f = sphere(x)
        opt.tell(x, f)
        _, f_first = opt.best()

        for _ in range(9):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            f = sphere(x)
            opt.tell(x, f)
        _, f_last = opt.best()

        assert f_last < f_first, (
            f"Fitness should strictly improve: first={f_first.item()}, last={f_last.item()}"
        )

    # Optimizers whose tell() can be compiled with the eager backend.
    # CMAES tell() uses .item() and math.exp() on a dynamic sigma that
    # changes each iteration, causing dynamo guard assertion failures.
    @pytest.mark.parametrize(
        "opt_name",
        ["SHADE", "NelderMead"],
    )
    def test_compiled_tell(self, opt_name: str, cpu_device: torch.device) -> None:
        """Compile tell() in addition to ask() and verify the loop works.

        Both ask and tell are compiled; results must match eager mode.
        CMAES and PhasedDFO are excluded because CMA-ES tell() uses .item()
        on dynamic sigma, which triggers dynamo guard assertion failures
        across iterations.  PhasedDFO uses CMA-ES internally.
        """
        n_iters = 10

        # Eager run
        opt_eager = OPTIMIZER_FACTORIES[opt_name](cpu_device)
        for _ in range(n_iters):
            x = opt_eager.ask()
            if x.shape[0] == 0:
                break
            opt_eager.tell(x, sphere(x))
        _, f_eager = opt_eager.best()

        # Compiled run (both ask and tell)
        opt_compiled = OPTIMIZER_FACTORIES[opt_name](cpu_device)
        compiled_ask = torch.compile(opt_compiled.ask, backend="eager")
        compiled_tell = torch.compile(opt_compiled.tell, backend="eager")
        for _ in range(n_iters):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            compiled_tell(x, sphere(x))
        _, f_compiled = opt_compiled.best()

        assert torch.allclose(f_eager, f_compiled), (
            f"Eager ({f_eager.item()}) != Compiled ask+tell ({f_compiled.item()})"
        )


class TestInductorBackend:
    """Tests using the inductor backend for real compilation.

    Each optimizer gets a sibling test parallel to ``test_inductor_backend_cmaes``.
    The CMAES header note about ``.item()`` on dynamic sigma applies to the
    CMAES ask() as well, but the 5-iteration ask-only pattern used here exercises
    the compile path without tripping the dynamo guard on tell().
    """

    @pytest.mark.skipif(
        "inductor" not in getattr(torch._dynamo, "list_backends", list)(),
        reason="inductor backend not available",
    )
    def test_inductor_backend_cmaes(self, cpu_device: torch.device) -> None:
        """CMAES with inductor backend for real compilation on sphere."""
        opt = CMAES(dim=5, bounds=5.0, seed=42, device=cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="inductor")
        for _ in range(5):
            x = compiled_ask()
            f = sphere(x)
            opt.tell(x, f)
        _, best_f = opt.best()
        assert best_f.item() < SMOKE_F_INIT_CMAES, (
            "inductor-compiled CMAES no better than random init"
        )
        assert torch.isfinite(best_f), "Best fitness is not finite"

    @pytest.mark.skipif(
        "inductor" not in getattr(torch._dynamo, "list_backends", list)(),
        reason="inductor backend not available",
    )
    def test_inductor_backend_shade(self, cpu_device: torch.device) -> None:
        """SHADE with inductor backend — 5 gens on sphere, must improve past random init."""
        opt = SHADE(dim=5, bounds=5.0, seed=42, device=cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="inductor")
        for _ in range(5):
            x = compiled_ask()
            f = sphere(x)
            opt.tell(x, f)
        _, best_f = opt.best()
        assert best_f.item() < SMOKE_F_INIT_CMAES, (
            "inductor-compiled SHADE no better than random init"
        )
        assert torch.isfinite(best_f)

    @pytest.mark.skipif(
        "inductor" not in getattr(torch._dynamo, "list_backends", list)(),
        reason="inductor backend not available",
    )
    def test_inductor_backend_nelder_mead(self, cpu_device: torch.device) -> None:
        """NelderMead with inductor backend — 5 iters on sphere."""
        opt = NelderMead(dim=5, bounds=5.0, seed=42, device=cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="inductor")
        for _ in range(5):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            f = sphere(x)
            opt.tell(x, f)
        _, best_f = opt.best()
        assert best_f.item() < SMOKE_F_INIT_CMAES, (
            "inductor-compiled NelderMead no better than random init"
        )
        assert torch.isfinite(best_f)

    @pytest.mark.skipif(
        "inductor" not in getattr(torch._dynamo, "list_backends", list)(),
        reason="inductor backend not available",
    )
    def test_inductor_backend_phased_dfo(self, cpu_device: torch.device) -> None:
        """PhasedDFO with inductor backend — 5 iters on sphere with small budget."""
        opt = PhasedDFO(dim=5, bounds=5.0, budget=500, seed=42, device=cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="inductor")
        for _ in range(5):
            x = compiled_ask()
            if x.shape[0] == 0:
                break
            f = sphere(x)
            opt.tell(x, f)
        _, best_f = opt.best()
        assert best_f.item() < SMOKE_F_INIT_CMAES, (
            "inductor-compiled PhasedDFO no better than random init"
        )
        assert torch.isfinite(best_f)


class TestCompiledTellCMAESKnownFailure:
    """Pin the CMAES + compiled tell() known-failure mode.

    CMAES.tell() uses ``.item()`` and ``math.exp()`` on a dynamic sigma that
    changes each iteration. Dynamo cannot trace this cleanly across multiple
    calls with different sigma values — guards fail on the second call.
    Rather than silently excluding CMAES from test_compiled_tell, this test
    pins the failure so a future regression (accidentally passing) would be
    noticed immediately.
    """

    def test_cmaes_compiled_tell_eventually_fails(self, cpu_device: torch.device) -> None:
        opt = CMAES(dim=5, bounds=5.0, seed=42, device=cpu_device)
        compiled_ask = torch.compile(opt.ask, backend="eager")
        compiled_tell = torch.compile(opt.tell, backend="eager")

        # First call may succeed (fresh trace); subsequent calls may hit a guard
        # failure or work fine depending on torch version.  We run enough
        # iterations that the dynamic-sigma guard has a chance to trip.  If NO
        # exception ever fires AND the loop completes, dynamo has become more
        # lenient — that would be an improvement worth investigating, so we
        # log via pytest.xfail rather than asserting a specific exception type.
        import pytest as _pytest

        try:
            for _ in range(10):
                x = compiled_ask()
                if x.shape[0] == 0:
                    break
                compiled_tell(x, sphere(x))
        except Exception as e:
            # Expected: dynamo guard failure on dynamic sigma
            msg = str(e).lower()
            # Don't pin exact text — torch versions vary the exact error
            assert any(k in msg for k in ("guard", "recompile", "dynamic", "trace"))
            return
        _pytest.xfail(
            "CMAES + compiled tell() no longer raises — dynamo may have improved. "
            "Review whether test_compiled_tell can now include CMAES."
        )


class TestPublicAPI:
    """Verify the public API surface is complete and consistent."""

    def test_all_exports_importable(self) -> None:
        """Every name in __all__ is importable."""
        import torch_dfo

        for name in torch_dfo.__all__:
            assert hasattr(torch_dfo, name), f"{name} listed in __all__ but not accessible"

    def test_version_string(self) -> None:
        import torch_dfo

        assert isinstance(torch_dfo.__version__, str)
        parts = torch_dfo.__version__.split(".")
        assert len(parts) == 3, "Version should be semver (major.minor.patch)"

    def test_optimizer_classes_have_ask_tell(self) -> None:
        """All optimizer classes implement the ask/tell interface."""
        from torch_dfo import CMAES, SHADE, NelderMead, PhasedDFO

        for cls in [CMAES, SHADE, NelderMead, PhasedDFO]:
            assert hasattr(cls, "ask"), f"{cls.__name__} missing ask()"
            assert hasattr(cls, "tell"), f"{cls.__name__} missing tell()"
            assert hasattr(cls, "best"), f"{cls.__name__} missing best()"

    def test_benchmark_functions_callable(self) -> None:
        """All exported benchmark functions accept (N, D) and return (N,)."""
        from torch_dfo import (
            ackley,
            griewank,
            levy,
            rastrigin,
            rosenbrock,
            schwefel,
            sphere,
            zakharov,
        )

        x = torch.zeros(10, 5)
        for fn in [sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy, zakharov]:
            result = fn(x)
            assert result.shape == (10,), f"{fn.__name__} should return (N,)"

    def test_benchmark_suite_available(self) -> None:
        """BenchmarkSuite produces named problems."""
        from torch_dfo import BenchmarkSuite

        suite = BenchmarkSuite.classical()
        assert len(suite) > 0
        for problem in suite:
            assert hasattr(problem, "name")
            assert hasattr(problem, "fn")
            assert hasattr(problem, "dim")
