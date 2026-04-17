"""Tests for torch_dfo.utils -- device helpers, RNG, bounds handling."""

from __future__ import annotations

import pytest
import torch

from torch_dfo.utils import (
    clamp_to_bounds,
    make_generator,
    normalize_bounds,
    resolve_device,
)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------
class TestResolveDevice:
    """Validate device resolution logic."""

    def test_cpu_string(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_torch_device_passthrough(self) -> None:
        dev = torch.device("cpu")
        assert resolve_device(dev) == dev

    def test_explicit_device_always_honoured(self, device: torch.device) -> None:
        """Whatever the environment, an explicit device string is returned as-is."""
        result = resolve_device(str(device))
        assert result == device

    def test_default_picks_available(self) -> None:
        dev = resolve_device()
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert dev.type == "mps"
        else:
            assert dev.type == "cpu"

    def test_resolve_device_invalid_string(self) -> None:
        """Passing an invalid device string should raise RuntimeError
        from the torch.device constructor.
        """
        with pytest.raises(RuntimeError):
            resolve_device("not_a_device")


# ---------------------------------------------------------------------------
# make_generator
# ---------------------------------------------------------------------------
class TestMakeGenerator:
    """Validate seeded RNG creation across devices."""

    def test_cpu_reproducibility(self) -> None:
        gen1 = make_generator(42, torch.device("cpu"))
        gen2 = make_generator(42, torch.device("cpu"))
        t1 = torch.randn(10, generator=gen1)
        t2 = torch.randn(10, generator=gen2)
        assert torch.equal(t1, t2)

    def test_different_seeds_differ(self) -> None:
        gen1 = make_generator(0, torch.device("cpu"))
        gen2 = make_generator(1, torch.device("cpu"))
        t1 = torch.randn(10, generator=gen1)
        t2 = torch.randn(10, generator=gen2)
        assert not torch.equal(t1, t2)

    def test_none_seed_does_not_raise(self, device: torch.device) -> None:
        gen = make_generator(None, device)
        assert isinstance(gen, torch.Generator)

    def test_generator_device_fallback(self, device: torch.device) -> None:
        """Non-CPU/CUDA devices fall back to a CPU generator."""
        gen = make_generator(42, device)
        if device.type in ("cpu", "cuda"):
            assert gen.device == device
        else:
            # MPS, XLA, etc. fall back to CPU generator
            assert gen.device == torch.device("cpu")

    def test_seeded_generator_on_device(self, device: torch.device) -> None:
        """Reproducibility: two generators with the same seed must yield
        identical sequences (generated on the generator's native device).
        """
        gen1 = make_generator(123, device)
        gen2 = make_generator(123, device)
        t1 = torch.randn(5, generator=gen1, device=gen1.device)
        t2 = torch.randn(5, generator=gen2, device=gen2.device)
        assert torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# clamp_to_bounds
# ---------------------------------------------------------------------------
class TestClampToBounds:
    """Validate broadcast-safe clamping."""

    def test_scalar_bounds(self, device: torch.device, default_dtype: torch.dtype) -> None:
        x = torch.tensor([-2.0, 0.5, 3.0], device=device, dtype=default_dtype)
        result = clamp_to_bounds(x, -1.0, 1.0)
        expected = torch.tensor([-1.0, 0.5, 1.0], device=device, dtype=default_dtype)
        assert torch.equal(result, expected)

    def test_tensor_bounds(self, device: torch.device, default_dtype: torch.dtype) -> None:
        x = torch.tensor([-5.0, 5.0, 0.0], device=device, dtype=default_dtype)
        lb = torch.tensor([-1.0, -2.0, -3.0], device=device, dtype=default_dtype)
        ub = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=default_dtype)
        result = clamp_to_bounds(x, lb, ub)
        expected = torch.tensor([-1.0, 2.0, 0.0], device=device, dtype=default_dtype)
        assert torch.equal(result, expected)

    def test_already_in_bounds(self, device: torch.device, default_dtype: torch.dtype) -> None:
        x = torch.tensor([0.0, 0.5, -0.5], device=device, dtype=default_dtype)
        result = clamp_to_bounds(x, -1.0, 1.0)
        assert torch.equal(result, x)

    def test_2d_input_with_broadcast(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """(pop, dim) tensor clamped by (dim,) bounds via broadcasting."""
        x = torch.tensor(
            [[-10.0, 10.0], [0.5, -0.5]],
            device=device,
            dtype=default_dtype,
        )
        lb = torch.tensor([-1.0, -1.0], device=device, dtype=default_dtype)
        ub = torch.tensor([1.0, 1.0], device=device, dtype=default_dtype)
        result = clamp_to_bounds(x, lb, ub)
        expected = torch.tensor(
            [[-1.0, 1.0], [0.5, -0.5]],
            device=device,
            dtype=default_dtype,
        )
        assert torch.equal(result, expected)

    def test_clamp_to_bounds_with_nan(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """NaN propagates through torch.clamp (PyTorch semantics)."""
        x = torch.tensor([float("nan"), 0.5, float("nan")], device=device, dtype=default_dtype)
        lb = torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=default_dtype)
        ub = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=default_dtype)
        result = clamp_to_bounds(x, lb, ub)
        assert torch.isnan(result[0]), "NaN at index 0 should propagate through clamp"
        assert result[1].item() == pytest.approx(0.5)
        assert torch.isnan(result[2]), "NaN at index 2 should propagate through clamp"


# ---------------------------------------------------------------------------
# normalize_bounds
# ---------------------------------------------------------------------------
class TestNormalizeBounds:
    """Validate bounds normalisation to (dim,) tensors."""

    def test_scalar_bounds(self, device: torch.device, default_dtype: torch.dtype) -> None:
        lb, ub = normalize_bounds(5.0, dim=3, device=device, dtype=default_dtype)
        assert lb.shape == (3,)
        assert ub.shape == (3,)
        assert torch.all(lb == -5.0)
        assert torch.all(ub == 5.0)
        assert lb.device.type == device.type
        assert ub.device.type == device.type

    def test_negative_scalar_uses_abs(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        lb, ub = normalize_bounds(-3.0, dim=2, device=device, dtype=default_dtype)
        assert torch.all(lb == -3.0)
        assert torch.all(ub == 3.0)

    def test_tuple_of_scalars(self, device: torch.device, default_dtype: torch.dtype) -> None:
        lb, ub = normalize_bounds((-2.0, 4.0), dim=4, device=device, dtype=default_dtype)
        assert lb.shape == (4,)
        assert torch.all(lb == -2.0)
        assert torch.all(ub == 4.0)

    def test_list_of_scalars(self, device: torch.device, default_dtype: torch.dtype) -> None:
        lb, ub = normalize_bounds([-1.0, 1.0], dim=3, device=device, dtype=default_dtype)
        assert torch.all(lb == -1.0)
        assert torch.all(ub == 1.0)

    def test_tuple_of_tensors(self, device: torch.device, default_dtype: torch.dtype) -> None:
        raw_lb = torch.tensor([-1.0, -2.0, -3.0])
        raw_ub = torch.tensor([1.0, 2.0, 3.0])
        lb, ub = normalize_bounds((raw_lb, raw_ub), dim=3, device=device, dtype=default_dtype)
        assert lb.device.type == device.type
        assert ub.device.type == device.type
        assert lb.dtype == default_dtype
        expected_lb = torch.tensor([-1.0, -2.0, -3.0], device=device, dtype=default_dtype)
        expected_ub = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=default_dtype)
        assert torch.equal(lb, expected_lb)
        assert torch.equal(ub, expected_ub)

    def test_int_scalar(self, device: torch.device, default_dtype: torch.dtype) -> None:
        lb, _ub = normalize_bounds(10, dim=2, device=device, dtype=default_dtype)
        assert lb.dtype == default_dtype
        assert torch.all(lb == -10.0)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported bounds type"):
            normalize_bounds("bad", dim=2, device=torch.device("cpu"), dtype=torch.float64)

    def test_dtype_preserved(self, device: torch.device) -> None:
        lb, ub = normalize_bounds(1.0, dim=2, device=device, dtype=torch.float32)
        assert lb.dtype == torch.float32
        assert ub.dtype == torch.float32

    def test_float64_on_cpu(self) -> None:
        """float64 must work on CPU regardless of other backends."""
        lb, ub = normalize_bounds(1.0, dim=2, device=torch.device("cpu"), dtype=torch.float64)
        assert lb.dtype == torch.float64
        assert ub.dtype == torch.float64

    def test_normalize_bounds_lb_greater_than_ub_raises(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """``bounds=(5.0, -5.0)`` (lb > ub) is rejected with a clean ValueError."""
        with pytest.raises(ValueError, match="positive span"):
            normalize_bounds((5.0, -5.0), dim=3, device=device, dtype=default_dtype)

    def test_normalize_bounds_zero_bounds_raises(
        self,
        device: torch.device,
        default_dtype: torch.dtype,
    ) -> None:
        """``bounds=0.0`` (zero span) is rejected with a clean ValueError."""
        with pytest.raises(ValueError, match="positive span"):
            normalize_bounds(0.0, dim=4, device=device, dtype=default_dtype)
