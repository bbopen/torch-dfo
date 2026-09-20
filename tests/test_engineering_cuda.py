"""Parity checks for the prepared thermal-control score callable."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _reference():
    name = "test_engineering_cuda_reference"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "examples" / "06_engineering_control.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _controls(batch: int, dtype: torch.dtype) -> torch.Tensor:
    reference = _reference()
    generator = torch.Generator().manual_seed(91_027)
    controls = 2.4 * torch.rand(batch, reference.HORIZON, generator=generator, dtype=dtype) - 1.2
    ties = torch.tensor([-0.125, 0.125], dtype=dtype)
    lower = torch.nextafter(ties, torch.full_like(ties, -torch.inf))
    upper = torch.nextafter(ties, torch.full_like(ties, torch.inf))
    controls[0] = torch.cat(
        (
            torch.tensor([-1.5, 1.5], dtype=dtype),
            ties,
            lower,
            upper,
            torch.tensor([-0.875, -0.625, -0.375, -0.125], dtype=dtype),
            torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=dtype),
        )
    )
    return controls


def _atol(dtype: torch.dtype) -> float:
    return 1e-3 if dtype == torch.float32 else 1e-10


def _compiled_cuda_options_available() -> bool:
    required = {
        "emulate_precision_casts",
        "eager_numerics.division_rounding",
        "triton.cudagraphs",
    }
    try:
        available = set(torch._inductor.list_options())
    except AttributeError:
        return False
    return required <= available


@pytest.mark.parametrize("split", ["train", "heldout"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 12, 4096])
def test_prepared_score_matches_reference_on_cpu(
    split: str, dtype: torch.dtype, batch: int
) -> None:
    reference = _reference()
    evaluator = reference.FrozenEvaluator(split)
    prepared = evaluator.prepare("cpu", dtype)
    controls = _controls(batch, dtype)

    assert evaluator.prepare("cpu", dtype) is prepared
    torch.testing.assert_close(prepared(controls), evaluator(controls), rtol=0.0, atol=_atol(dtype))


def test_prepared_score_validates_before_the_hot_path() -> None:
    reference = _reference()
    evaluator = reference.FrozenEvaluator("train")
    prepared = evaluator.prepare("cpu", torch.float32)

    with pytest.raises(ValueError, match="shape"):
        prepared(torch.zeros(1, reference.HORIZON - 1, dtype=torch.float32))
    with pytest.raises(ValueError, match="dtype"):
        prepared(torch.zeros(1, reference.HORIZON, dtype=torch.float64))
    with pytest.raises(ValueError, match="CUDA"):
        evaluator.prepare("cpu", torch.float32, compile=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _compiled_cuda_options_available(),
    reason="requires CUDA and Inductor eager-arithmetic options",
)
@pytest.mark.parametrize("split", ["train", "heldout"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_cuda_score_matches_reference_and_keeps_outputs(
    split: str, dtype: torch.dtype
) -> None:
    reference = _reference()
    evaluator = reference.FrozenEvaluator(split)
    eager = evaluator.prepare("cuda", dtype)
    prepared = evaluator.prepare("cuda", dtype, compile=True)
    for batch in (12, 4096):
        first_controls = _controls(batch, dtype).cuda()
        second_controls = (-_controls(batch, dtype)).cuda()
        expected_first = evaluator(first_controls).clone()
        expected_second = evaluator(second_controls).clone()

        eager_first = eager(first_controls)
        eager_second = eager(second_controls)
        first_score = prepared(first_controls)
        second_score = prepared(second_controls)

        torch.testing.assert_close(eager_first, expected_first, rtol=0.0, atol=_atol(dtype))
        torch.testing.assert_close(eager_second, expected_second, rtol=0.0, atol=_atol(dtype))
        torch.testing.assert_close(first_score, expected_first, rtol=0.0, atol=_atol(dtype))
        torch.testing.assert_close(second_score, expected_second, rtol=0.0, atol=_atol(dtype))
