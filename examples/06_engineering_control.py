#!/usr/bin/env python
"""Quantized thermal-zone control
==================================

Derivative-free search on a batched thermal-control reference.

This reference problem uses a one-zone RC heat balance.  It is a small,
reproducible control study.  It is not a building simulator or a claim about
field HVAC performance.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

TASK_VERSION = "engineering-control-v2"
HORIZON = 16
TARGET_TEMPERATURE_C = 22.0
PASS_THRESHOLD = 3.50


@dataclass(frozen=True)
class ThermalPlant:
    """Parameters for a first-order zone heat balance."""

    step_seconds: float = 900.0
    thermal_capacitance_j_per_k: float = 3_000_000.0
    resistance_k_per_w: float = 0.005
    actuator_capacity_w: float = 2_500.0
    actuator_step: float = 0.25
    target_temperature_c: float = TARGET_TEMPERATURE_C

    @property
    def thermal_decay(self) -> float:
        """Return the explicit-Euler coefficient for the ambient term."""
        return self.step_seconds / (self.thermal_capacitance_j_per_k * self.resistance_k_per_w)

    @property
    def actuator_temperature_step(self) -> float:
        """Return the temperature change from full actuator output for one step."""
        return self.step_seconds * self.actuator_capacity_w / self.thermal_capacitance_j_per_k


DEFAULT_PLANT = ThermalPlant()


@dataclass(frozen=True)
class ScenarioSet:
    """Fixed exogenous trajectories used by one evaluator split."""

    name: str
    outdoor_temperature_c: tuple[tuple[float, ...], ...]
    internal_gain_w: tuple[tuple[float, ...], ...]
    initial_temperature_c: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.initial_temperature_c)

    def tensor(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        """Return scenario tensors on the evaluation device."""
        return (
            torch.tensor(self.outdoor_temperature_c, device=device, dtype=dtype),
            torch.tensor(self.internal_gain_w, device=device, dtype=dtype),
            torch.tensor(self.initial_temperature_c, device=device, dtype=dtype),
        )


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_scenario(index: int) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Build one deterministic weather and internal-gain trajectory."""
    base_outdoor = 21.5 + 1.2 * (index % 9)
    amplitude = 1.4 + 0.3 * (index % 4)
    phase = 0.37 * index
    outdoor = tuple(
        round(base_outdoor + amplitude * math.sin(phase + 2.0 * math.pi * step / HORIZON), 6)
        for step in range(HORIZON)
    )
    base_gain = 300.0 + 90.0 * ((3 * index) % 7)
    pulse_start = (2 * index + 3) % (HORIZON - 4)
    gains = tuple(
        round(base_gain + (450.0 if pulse_start <= step < pulse_start + 4 else 0.0), 6)
        for step in range(HORIZON)
    )
    initial = round(21.4 + 0.2 * ((5 * index) % 7), 6)
    return outdoor, gains, initial


def fixed_scenarios(split: Literal["train", "heldout"]) -> ScenarioSet:
    """Return the immutable training or held-out scenario split."""
    indices = tuple(range(8)) if split == "train" else tuple(range(8, 13))
    rows = tuple(_make_scenario(index) for index in indices)
    return ScenarioSet(
        name=split,
        outdoor_temperature_c=tuple(row[0] for row in rows),
        internal_gain_w=tuple(row[1] for row in rows),
        initial_temperature_c=tuple(row[2] for row in rows),
    )


def scenario_hash(scenarios: ScenarioSet) -> str:
    """Return the stable identity of a scenario split."""
    return _hash(asdict(scenarios))


def task_hash(plant: ThermalPlant = DEFAULT_PLANT) -> str:
    """Return the stable identity of the task and both fixed splits."""
    return _hash(
        {
            "version": TASK_VERSION,
            "plant": asdict(plant),
            "train": scenario_hash(fixed_scenarios("train")),
            "heldout": scenario_hash(fixed_scenarios("heldout")),
        }
    )


def effective_actuation(
    requested: torch.Tensor, plant: ThermalPlant = DEFAULT_PLANT
) -> torch.Tensor:
    """Clamp and quantize requested actuator fractions in [-1, 1]."""
    bounded = requested.clamp(-1.0, 1.0)
    return (torch.round(bounded / plant.actuator_step) * plant.actuator_step).clamp(-1.0, 1.0)


def simulate(
    requested_control: torch.Tensor,
    scenarios: ScenarioSet,
    plant: ThermalPlant = DEFAULT_PLANT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a batch of open-loop commands for every fixed scenario.

    The recurrence is ``T[k+1] = T[k] + dt/C * ((T_out - T[k]) / R + q_int + q_act)``.
    Commands are quantized after saturation.  This makes the objective
    piecewise constant in regions of the request space.
    """
    if requested_control.ndim == 1:
        requested_control = requested_control.unsqueeze(0)
    if requested_control.ndim != 2 or requested_control.shape[1] != HORIZON:
        raise ValueError(f"requested_control must have shape (batch, {HORIZON})")
    commands = effective_actuation(requested_control, plant)
    outdoor, internal, initial = scenarios.tensor(
        device=requested_control.device, dtype=requested_control.dtype
    )
    batch = commands.shape[0]
    temperature = initial.unsqueeze(0).expand(batch, -1)
    trace = [temperature]
    dt_over_c = plant.step_seconds / plant.thermal_capacitance_j_per_k
    for step in range(HORIZON):
        heat_w = commands[:, step].unsqueeze(1) * plant.actuator_capacity_w
        temperature = temperature + dt_over_c * (
            (outdoor[:, step].unsqueeze(0) - temperature) / plant.resistance_k_per_w
            + internal[:, step].unsqueeze(0)
            + heat_w
        )
        trace.append(temperature)
    return torch.stack(trace, dim=-1), commands


def analytic_linear_reference(
    effective_control: torch.Tensor,
    outdoor_temperature_c: torch.Tensor,
    internal_gain_w: torch.Tensor,
    initial_temperature_c: torch.Tensor,
    plant: ThermalPlant = DEFAULT_PLANT,
) -> torch.Tensor:
    """Solve the same unrolled linear recurrence without calling ``simulate``."""
    if effective_control.ndim == 1:
        effective_control = effective_control.unsqueeze(0)
    decay = 1.0 - plant.thermal_decay
    batch = effective_control.shape[0]
    scenario_count = outdoor_temperature_c.shape[0]
    state = initial_temperature_c.unsqueeze(0).expand(batch, -1)
    result = [state]
    dt_over_c = plant.step_seconds / plant.thermal_capacitance_j_per_k
    for step in range(HORIZON):
        forcing = dt_over_c * (
            outdoor_temperature_c[:, step].unsqueeze(0) / plant.resistance_k_per_w
            + internal_gain_w[:, step].unsqueeze(0)
            + effective_control[:, step].unsqueeze(1) * plant.actuator_capacity_w
        )
        state = decay * state + forcing
        if state.shape != (batch, scenario_count):
            raise RuntimeError("analytic recurrence produced an invalid state shape")
        result.append(state)
    return torch.stack(result, dim=-1)


@dataclass(frozen=True)
class Evaluation:
    """Scores from an immutable scenario evaluator."""

    score: torch.Tensor
    mean_comfort_cost: torch.Tensor
    worst_comfort_cost: torch.Tensor
    energy_cost: torch.Tensor
    switching_cost: torch.Tensor


class FrozenEvaluator:
    """Score controls against one fixed split.  This evaluator has no tuning knobs."""

    version = TASK_VERSION

    def __init__(self, split: Literal["train", "heldout"], plant: ThermalPlant = DEFAULT_PLANT):
        self.split = split
        self.plant = plant
        self.scenarios = fixed_scenarios(split)
        self.scenario_hash = scenario_hash(self.scenarios)
        self.evaluator_hash = _hash(
            {
                "version": self.version,
                "split": split,
                "scenario_hash": self.scenario_hash,
                "plant": asdict(plant),
                "weights": {
                    "comfort_band_c": 0.5,
                    "energy": 0.08,
                    "switching": 0.03,
                    "tail": 0.50,
                },
            }
        )

    def evaluate(self, requested_control: torch.Tensor) -> Evaluation:
        """Return the frozen robust objective for each requested control row."""
        temperatures, commands = simulate(requested_control, self.scenarios, self.plant)
        deviation = (temperatures[:, :, 1:] - self.plant.target_temperature_c).abs()
        comfort = torch.relu(deviation - 0.5).square().mean(dim=-1)
        mean_comfort = comfort.mean(dim=-1)
        worst_comfort = comfort.max(dim=-1).values
        energy = 0.08 * commands.abs().mean(dim=-1)
        switching = 0.03 * (commands[:, 1:] - commands[:, :-1]).abs().mean(dim=-1)
        score = mean_comfort + 0.50 * worst_comfort + energy + switching
        return Evaluation(score, mean_comfort, worst_comfort, energy, switching)

    def __call__(self, requested_control: torch.Tensor) -> torch.Tensor:
        return self.evaluate(requested_control).score

    def passes_reference_gate(self, requested_control: torch.Tensor) -> torch.Tensor:
        """Return whether candidates meet the fixed reference feasibility gate."""
        return self(requested_control) <= PASS_THRESHOLD


def mean_scenario_pi_control(
    scenarios: ScenarioSet,
    plant: ThermalPlant = DEFAULT_PLANT,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a fixed-gain PI plan for the mean disturbance trajectory.

    This is a domain baseline.  It uses the known RC model, but controls only
    the mean fixed scenario, so it returns one open-loop command plan.  The
    gains are fixed for the reference task and are not selected on held-out data.
    """
    target = torch.tensor(plant.target_temperature_c, device=device, dtype=dtype)
    outdoor, internal, initial = scenarios.tensor(device=torch.device(device), dtype=dtype)
    temperature = initial.mean()
    integral_error = torch.zeros((), device=device, dtype=dtype)
    commands: list[torch.Tensor] = []
    dt_over_c = plant.step_seconds / plant.thermal_capacitance_j_per_k
    for step in range(HORIZON):
        without_actuator = temperature + dt_over_c * (
            (outdoor[:, step].mean() - temperature) / plant.resistance_k_per_w
            + internal[:, step].mean()
        )
        error = target - without_actuator
        integral_error = integral_error + error
        request = (0.90 * error + 0.06 * integral_error) / plant.actuator_temperature_step
        command = effective_actuation(request, plant)
        commands.append(command)
        temperature = without_actuator + command * plant.actuator_temperature_step
    return torch.stack(commands).unsqueeze(0)


def mean_scenario_deadbeat_control(
    scenarios: ScenarioSet,
    plant: ThermalPlant = DEFAULT_PLANT,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the legacy name for the fixed-gain mean-scenario PI baseline."""
    return mean_scenario_pi_control(scenarios, plant, device=device, dtype=dtype)


def run_evaluator_probes(
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64
) -> dict[str, dict[str, float | bool]]:
    """Run the four evaluator probes before any optimizer benchmark."""
    # The probes use train only.  Held-out scenarios remain untouched until
    # the benchmark has selected each final candidate by the train score.
    evaluator = FrozenEvaluator("train")
    dev = torch.device(device)
    zero = torch.zeros(1, HORIZON, device=dev, dtype=dtype)
    ceiling = mean_scenario_pi_control(evaluator.scenarios, device=dev, dtype=dtype)
    constant_cooling = torch.full((1, HORIZON), -0.5, device=dev, dtype=dtype)
    all_heat = torch.ones(1, HORIZON, device=dev, dtype=dtype)
    generator = torch.Generator(device=dev).manual_seed(24_681)
    null_batch = 2.0 * torch.rand(64, HORIZON, generator=generator, device=dev, dtype=dtype) - 1.0
    zero_score = evaluator(zero)
    ceiling_score = evaluator(ceiling)
    cooling_score = evaluator(constant_cooling)
    heat_score = evaluator(all_heat)
    null_scores = evaluator(null_batch)
    return {
        "degenerate_zero": {
            "score": float(zero_score.item()),
            "passes": bool(zero_score.item() <= PASS_THRESHOLD),
        },
        "ceiling_domain_control": {
            "score": float(ceiling_score.item()),
            "passes": bool(ceiling_score.item() <= PASS_THRESHOLD),
        },
        "cheap_exploit_constant_cooling": {
            "score": float(cooling_score.item()),
            "passes": bool(cooling_score.item() <= PASS_THRESHOLD),
        },
        "runaway_all_heat": {
            "score": float(heat_score.item()),
            "passes": bool(heat_score.item() <= PASS_THRESHOLD),
        },
        "null_uniform_random": {
            "seed": 24_681,
            "sample_count": int(null_scores.numel()),
            "min_score": float(null_scores.min().item()),
            "median_score": float(null_scores.median().item()),
            "max_score": float(null_scores.max().item()),
            "mean_score": float(null_scores.mean().item()),
            "pass_rate": float((null_scores <= PASS_THRESHOLD).double().mean().item()),
        },
    }


def main() -> None:
    """Print a short reproducibility record and the evaluator audit."""
    print(
        json.dumps(
            {
                "task_version": TASK_VERSION,
                "task_hash": task_hash(),
                "train_scenario_hash": scenario_hash(fixed_scenarios("train")),
                "heldout_scenario_hash": scenario_hash(fixed_scenarios("heldout")),
                "probes": run_evaluator_probes(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
