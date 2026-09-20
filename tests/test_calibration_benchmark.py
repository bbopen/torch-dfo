"""Small checks for the calisim temporal calibration benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torch_dfo

pytest.importorskip("scipy")

from benchmarks import calibration


def test_provenance_rejects_a_different_imported_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(torch_dfo, "__file__", str(tmp_path / "torch_dfo" / "__init__.py"))
    with pytest.raises(RuntimeError, match="must import this checkout"):
        calibration._source_metadata(tmp_path)


def test_normalization_round_trip_preserves_source_parameters() -> None:
    parameters = np.array([[0.45, 0.02], [0.52, 0.024], [0.55, 0.03]])
    restored = calibration.physical_from_unit(calibration.unit_from_physical(parameters))
    assert np.allclose(restored, parameters)


@pytest.mark.parametrize(
    "invalid",
    [np.array([[np.nan, 0.5]]), np.array([[0.5, 0.5, 0.5]]), np.array([[1.1, 0.5]])],
)
def test_unit_normalization_rejects_malformed_or_nonfinite_candidates(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        calibration.physical_from_unit(invalid)


def test_train_objective_does_not_include_heldout_observations() -> None:
    objective = calibration.CalibrationObjective("train")
    assert np.array_equal(objective.years, calibration.TRAIN_YEARS)
    assert np.array_equal(objective.observed, calibration.TRAIN_OBSERVATIONS)
    assert not np.intersect1d(objective.years, calibration.HELDOUT_YEARS).size
    scores = objective(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    assert scores.shape == (1,)
    assert objective.candidate_calls == 1
    assert objective.objective_calls == 1


def test_evaluator_probes_check_physical_score_and_rejections() -> None:
    report = calibration.run_evaluator_probes()
    probes = report["probes"]
    assert all(row["passes"] for row in probes.values())
    assert probes["cached_fitness_recheck"]["candidate_calls"] == 1
    assert probes["wrong_parameter_and_null_controls"]["rejected_malformed_inputs"] == 3
    assert report["cost_accounting"]["candidate_calls"] == 3


def test_evaluator_probes_reject_an_objective_with_constant_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration.CalibrationObjective,
        "score_unit",
        lambda self, candidates: np.ones(calibration._as_unit_matrix(candidates).shape[0]),
    )
    with pytest.raises(RuntimeError, match="required probes"):
        calibration.run_evaluator_probes()


def test_native_search_reports_the_same_candidate_count_as_searchrun() -> None:
    class QuadraticObjective:
        candidate_calls = 0
        objective_calls = 0

        def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
            self.candidate_calls += int(candidates.shape[0])
            self.objective_calls += 1
            return ((candidates - 0.5) ** 2).sum(dim=1)

    objective = QuadraticObjective()
    _, _, accounting = calibration.run_native_search("random", objective, budget=40, seed=0)
    assert accounting["candidate_evaluations"] == 40
    assert accounting["searchrun_charged_evaluations"] == 40
    assert accounting["objective_callback_calls"] == 2


def test_requested_evotorch_dependency_fails_loudly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("evotorch"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match="EvoTorch was requested"):
        calibration._run_evotorch_cmaes(
            calibration.CalibrationObjective("train"), budget=20, seed=0, evotorch_src=None
        )
