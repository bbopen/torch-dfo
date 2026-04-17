"""Tests for torch_dfo.space -- SearchSpace, Float, Int, Categorical."""

from __future__ import annotations

import pytest
import torch

from tests._thresholds import ATOL_F64_DEFAULT
from torch_dfo.space import Categorical, Float, Int, SearchSpace


# ---------------------------------------------------------------------------
# TestFloatParam
# ---------------------------------------------------------------------------
class TestFloatParam:
    def test_encode_linear_scale(self) -> None:
        space = SearchSpace([Float("x", 0.0, 10.0)])
        configs = [{"x": 0.0}, {"x": 5.0}, {"x": 10.0}]
        t = space.encode(configs)
        assert t.shape == (3, 1)
        assert torch.allclose(t[:, 0], torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))

    def test_encode_log_scale(self) -> None:
        space = SearchSpace([Float("x", 1.0, 100.0, log=True)])
        configs = [{"x": 1.0}, {"x": 10.0}, {"x": 100.0}]
        t = space.encode(configs)
        assert t.shape == (3, 1)
        expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        assert torch.allclose(t[:, 0], expected, atol=ATOL_F64_DEFAULT)

    def test_decode_linear_round_trip(self) -> None:
        space = SearchSpace([Float("x", -5.0, 5.0)])
        values = [-5.0, -2.5, 0.0, 2.5, 5.0]
        configs = [{"x": v} for v in values]
        encoded = space.encode(configs)
        decoded = space.decode(encoded)
        for orig, dec in zip(configs, decoded, strict=True):
            assert abs(dec["x"] - orig["x"]) < 1e-10

    def test_decode_log_round_trip(self) -> None:
        space = SearchSpace([Float("x", 1e-3, 1e3, log=True)])
        values = [1e-3, 1e-1, 1.0, 1e1, 1e3]
        configs = [{"x": v} for v in values]
        encoded = space.encode(configs)
        decoded = space.decode(encoded)
        for orig, dec in zip(configs, decoded, strict=True):
            assert abs(dec["x"] - orig["x"]) < 1e-10

    def test_encode_clamps_output(self) -> None:
        space = SearchSpace([Float("x", 0.0, 1.0)])
        # Values outside [lo, hi] should clamp to [0, 1]
        configs = [{"x": -1.0}, {"x": 2.0}]
        t = space.encode(configs)
        assert t[0, 0].item() == pytest.approx(0.0)
        assert t[1, 0].item() == pytest.approx(1.0)

    def test_decode_clips_input(self) -> None:
        space = SearchSpace([Float("x", 0.0, 10.0)])
        # Decode should clip -0.1 → lo and 1.1 → hi
        t = torch.tensor([[-0.1], [1.1]], dtype=torch.float64)
        decoded = space.decode(t)
        assert decoded[0]["x"] == pytest.approx(0.0)
        assert decoded[1]["x"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestIntParam
# ---------------------------------------------------------------------------
class TestIntParam:
    def test_decode_returns_int(self) -> None:
        space = SearchSpace([Int("n", 1, 10)])
        t = space.encode([{"n": 5}])
        decoded = space.decode(t)
        assert isinstance(decoded[0]["n"], int)

    def test_decode_round_trip_all_values(self) -> None:
        lo, hi = 2, 8
        space = SearchSpace([Int("n", lo, hi)])
        for v in range(lo, hi + 1):
            encoded = space.encode([{"n": v}])
            decoded = space.decode(encoded)
            assert decoded[0]["n"] == v

    def test_boundary_zero(self) -> None:
        space = SearchSpace([Int("n", 2, 8)])
        t = torch.tensor([[0.0]], dtype=torch.float64)
        decoded = space.decode(t)
        assert decoded[0]["n"] == 2

    def test_boundary_one(self) -> None:
        space = SearchSpace([Int("n", 2, 8)])
        eps = 1e-9
        t = torch.tensor([[1.0 - eps]], dtype=torch.float64)
        decoded = space.decode(t)
        assert decoded[0]["n"] == 8

    def test_log_int_decode_returns_int(self) -> None:
        space = SearchSpace([Int("n", 1, 100, log=True)])
        t = space.encode([{"n": 10}])
        decoded = space.decode(t)
        assert isinstance(decoded[0]["n"], int)


# ---------------------------------------------------------------------------
# TestCategoricalParam
# ---------------------------------------------------------------------------
class TestCategoricalParam:
    def test_decode_always_valid_choice(self) -> None:
        choices = ["a", "b", "c", "d"]
        space = SearchSpace([Categorical("c", choices)])
        # Test many encoded values
        t = torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(1)
        decoded = space.decode(t)
        for d in decoded:
            assert d["c"] in choices

    def test_encode_decode_round_trip(self) -> None:
        choices = ["relu", "tanh", "sigmoid"]
        space = SearchSpace([Categorical("act", choices)])
        configs = [{"act": c} for c in choices]
        encoded = space.encode(configs)
        decoded = space.decode(encoded)
        for orig, dec in zip(configs, decoded, strict=True):
            assert dec["act"] == orig["act"]

    def test_boundary_zero(self) -> None:
        choices = ["a", "b", "c"]
        space = SearchSpace([Categorical("c", choices)])
        t = torch.tensor([[0.0]], dtype=torch.float64)
        decoded = space.decode(t)
        assert decoded[0]["c"] == "a"

    def test_near_one(self) -> None:
        choices = ["a", "b", "c"]
        space = SearchSpace([Categorical("c", choices)])
        t = torch.tensor([[0.9999]], dtype=torch.float64)
        decoded = space.decode(t)
        assert decoded[0]["c"] == "c"

    def test_decode_clamps_above_one(self) -> None:
        choices = ["a", "b", "c"]
        space = SearchSpace([Categorical("c", choices)])
        t = torch.tensor([[1.5]], dtype=torch.float64)
        decoded = space.decode(t)
        # Should not raise IndexError; should be a valid choice
        assert decoded[0]["c"] in choices


# ---------------------------------------------------------------------------
# TestSearchSpace
# ---------------------------------------------------------------------------
class TestSearchSpace:
    def _make_mixed_space(self) -> SearchSpace:
        return SearchSpace(
            [
                Float("lr", 1e-4, 1e-1, log=True),
                Int("layers", 1, 5),
                Categorical("act", ["relu", "tanh", "sigmoid"]),
            ],
        )

    def test_dim_equals_len_params(self) -> None:
        space = self._make_mixed_space()
        assert space.dim == 3

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            SearchSpace([])

    def test_duplicate_names_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate parameter names"):
            SearchSpace([Float("x", 0.0, 1.0), Int("x", 1, 10)])

    def test_encode_shape(self) -> None:
        space = self._make_mixed_space()
        configs = [
            {"lr": 1e-3, "layers": 2, "act": "relu"},
            {"lr": 1e-2, "layers": 4, "act": "tanh"},
            {"lr": 5e-2, "layers": 1, "act": "sigmoid"},
        ]
        t = space.encode(configs)
        assert t.shape == (3, 3)

    def test_decode_length(self) -> None:
        space = self._make_mixed_space()
        t = torch.rand(5, 3, dtype=torch.float64)
        decoded = space.decode(t)
        assert len(decoded) == 5

    def test_encode_decode_mixed_round_trip(self) -> None:
        space = self._make_mixed_space()
        configs = [
            {"lr": 1e-3, "layers": 3, "act": "tanh"},
        ]
        encoded = space.encode(configs)
        decoded = space.decode(encoded)
        assert abs(decoded[0]["lr"] - 1e-3) < 1e-10
        assert decoded[0]["layers"] == 3
        assert decoded[0]["act"] == "tanh"

    def test_encode_batch_correctness(self) -> None:
        space = SearchSpace([Float("x", 0.0, 10.0)])
        values = list(range(10))
        configs = [{"x": float(v)} for v in values]
        t = space.encode(configs)
        assert t.shape == (10, 1)
        for i, v in enumerate(values):
            assert t[i, 0].item() == pytest.approx(v / 10.0)

    def test_decode_single_row_promoted(self) -> None:
        space = self._make_mixed_space()
        row = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)  # shape (dim,)
        decoded = space.decode(row)
        assert len(decoded) == 1
        assert isinstance(decoded[0], dict)
        assert set(decoded[0].keys()) == {"lr", "layers", "act"}

    def test_params_property_is_copy(self) -> None:
        space = self._make_mixed_space()
        params = space.params
        params.clear()
        assert space.dim == 3  # internal list unaffected

    def test_encode_device_kwarg(self) -> None:
        space = SearchSpace([Float("x", 0.0, 1.0)])
        configs = [{"x": 0.5}]
        t = space.encode(configs, device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_encode_dtype_kwarg(self) -> None:
        space = SearchSpace([Float("x", 0.0, 1.0)])
        configs = [{"x": 0.5}]
        t = space.encode(configs, dtype=torch.float32)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestSearchSpaceEdgeCases
# ---------------------------------------------------------------------------
class TestSearchSpaceEdgeCases:
    def test_log_float_lower_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            Float("x", 0.0, 1.0, log=True)

    def test_log_float_lower_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            Float("x", -1.0, 1.0, log=True)

    def test_unknown_categorical_raises(self) -> None:
        space = SearchSpace([Categorical("c", ["a", "b", "c"])])
        with pytest.raises(ValueError):
            space.encode([{"c": "unknown"}])

    def test_very_small_float_range(self) -> None:
        space = SearchSpace([Float("x", 1e-10, 1e-9)])
        configs = [{"x": 1e-10}, {"x": 5e-10}, {"x": 1e-9}]
        encoded = space.encode(configs)
        decoded = space.decode(encoded)
        for orig, dec in zip(configs, decoded, strict=True):
            assert abs(dec["x"] - orig["x"]) < 1e-20

    def test_single_param_space(self) -> None:
        space = SearchSpace([Float("x", -1.0, 1.0)])
        assert space.dim == 1
        encoded = space.encode([{"x": 0.0}])
        assert encoded.shape == (1, 1)
        decoded = space.decode(encoded)
        assert abs(decoded[0]["x"] - 0.0) < 1e-10

    def test_categorical_two_choices(self) -> None:
        space = SearchSpace([Categorical("flag", [True, False])])
        for choice in [True, False]:
            encoded = space.encode([{"flag": choice}])
            decoded = space.decode(encoded)
            assert decoded[0]["flag"] == choice

    def test_encode_missing_key_raises(self) -> None:
        space = SearchSpace([Float("x", 0.0, 1.0), Float("y", 0.0, 1.0)])
        with pytest.raises(KeyError, match="missing required parameter 'y'"):
            space.encode([{"x": 0.5}])  # missing 'y'

    # ------------------------------------------------------------------
    # Construction-validator coverage (round-2 edge-case audit).
    # Each raise site in space.py gets exactly one covering test.
    # ------------------------------------------------------------------

    def test_float_upper_not_greater_than_lower_raises(self) -> None:
        """space.py:23 — Float construction rejects upper <= lower."""
        with pytest.raises(ValueError, match=r"must be > lower"):
            Float("x", 1.0, 1.0)

    def test_int_upper_less_than_lower_raises(self) -> None:
        """space.py:41 — Int construction rejects upper < lower.

        Note: upper == lower is explicitly allowed (single-valued Int), which
        is why this uses a strictly inverted range.
        """
        with pytest.raises(ValueError, match=r"must be >= lower"):
            Int("n", 5, 1)

    def test_int_log_requires_positive_lower(self) -> None:
        """space.py:45 — Int with log=True rejects lower <= 0."""
        with pytest.raises(ValueError, match=r"log=True requires lower > 0"):
            Int("n", 0, 10, log=True)

    def test_categorical_empty_choices_raises(self) -> None:
        """space.py:57 — Categorical rejects empty choices list."""
        with pytest.raises(ValueError, match=r"choices must not be empty"):
            Categorical("c", [])
