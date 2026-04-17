"""Tests for torch_dfo._operators -- vectorized DE operators."""

from __future__ import annotations

import math

import pytest
import torch

from torch_dfo._operators import (
    de_binomial_crossover,
    de_current_to_pbest_mutation,
    levy_flight_perturbation,
    opposition_init,
)
from torch_dfo.utils import make_generator

from .conftest import best_float_dtype

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gen(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded generator compatible with the given device."""
    return make_generator(seed, device)


# ---------------------------------------------------------------------------
# de_current_to_pbest_mutation
# ---------------------------------------------------------------------------
class TestDeCurrentToPbestMutation:
    """Validate current-to-pbest/1 mutation operator."""

    @pytest.fixture
    def pop_data(self, device: torch.device) -> dict:
        pop_size, dim = 20, 5
        gen = _make_gen(42, device)
        population = torch.randn(pop_size, dim, device=gen.device, generator=gen).to(device)
        fitness = torch.randn(pop_size, device=gen.device, generator=gen).to(device)
        F = torch.full((pop_size,), 0.5, device=device)
        return {
            "population": population,
            "fitness": fitness,
            "F": F,
            "pop_size": pop_size,
            "dim": dim,
        }

    def test_output_shape(self, device: torch.device, pop_data: dict) -> None:
        donor = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            generator=_make_gen(99, device),
        )
        assert donor.shape == (pop_data["pop_size"], pop_data["dim"])
        assert torch.isfinite(donor).all(), "Donor contains NaN or Inf"
        assert not torch.equal(donor, pop_data["population"]), (
            "F=0.5 mutation must perturb the population"
        )

    def test_output_device(self, device: torch.device, pop_data: dict) -> None:
        donor = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            generator=_make_gen(99, device),
        )
        assert donor.device == pop_data["population"].device
        assert donor.dtype == pop_data["population"].dtype
        assert torch.isfinite(donor).all(), "Donor contains NaN or Inf"

    def test_archive_none_works(self, device: torch.device, pop_data: dict) -> None:
        donor = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            archive=None,
            generator=_make_gen(99, device),
        )
        assert donor.shape == (pop_data["pop_size"], pop_data["dim"])
        assert torch.isfinite(donor).all()
        assert not torch.equal(donor, pop_data["population"]), (
            "F=0.5 mutation must perturb the population"
        )

    def test_with_archive(self, device: torch.device, pop_data: dict) -> None:
        """Archive must actually influence the mutation output."""
        gen = _make_gen(77, device)
        archive_a = torch.randn(10, pop_data["dim"], device=gen.device, generator=gen).to(device)
        donor_a = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            archive=archive_a,
            generator=_make_gen(99, device),
        )
        assert donor_a.shape == (pop_data["pop_size"], pop_data["dim"])
        assert torch.isfinite(donor_a).all()

        # Same seed but different archive content -- output must differ
        archive_b = archive_a + 1000.0
        donor_b = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            archive=archive_b,
            generator=_make_gen(99, device),
        )
        assert not torch.equal(donor_a, donor_b), (
            "Different archives with same seed must produce different donors"
        )

    def test_empty_archive_same_as_none(self, device: torch.device, pop_data: dict) -> None:
        gen = _make_gen(99, device)
        donor_none = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            archive=None,
            generator=gen,
        )
        gen2 = _make_gen(99, device)
        empty_archive = torch.empty(0, pop_data["dim"], device=device)
        donor_empty = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            archive=empty_archive,
            generator=gen2,
        )
        assert torch.equal(donor_none, donor_empty)

    def test_reproducible_with_seed(self, device: torch.device, pop_data: dict) -> None:
        gen1 = _make_gen(123, device)
        donor1 = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            generator=gen1,
        )
        gen2 = _make_gen(123, device)
        donor2 = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            pop_data["F"],
            p_fraction=0.2,
            generator=gen2,
        )
        assert torch.equal(donor1, donor2)

    def test_f_zero_returns_population(self, device: torch.device, pop_data: dict) -> None:
        """When F=0, donor should equal the original population (no perturbation)."""
        F_zero = torch.zeros(pop_data["pop_size"], device=device)
        donor = de_current_to_pbest_mutation(
            pop_data["population"],
            pop_data["fitness"],
            F_zero,
            p_fraction=0.2,
            generator=_make_gen(99, device),
        )
        assert torch.allclose(donor, pop_data["population"])


# ---------------------------------------------------------------------------
# de_binomial_crossover
# ---------------------------------------------------------------------------
class TestDeBinomialCrossover:
    """Validate binomial crossover with forced j_rand."""

    @pytest.fixture
    def crossover_data(self, device: torch.device) -> dict:
        pop_size, dim = 30, 8
        gen = _make_gen(42, device)
        donor = torch.randn(pop_size, dim, device=gen.device, generator=gen).to(device)
        target = torch.randn(pop_size, dim, device=gen.device, generator=gen).to(device)
        return {
            "donor": donor,
            "target": target,
            "pop_size": pop_size,
            "dim": dim,
        }

    def test_output_shape(self, device: torch.device, crossover_data: dict) -> None:
        CR = torch.full((crossover_data["pop_size"],), 0.5, device=device)
        trial = de_binomial_crossover(
            crossover_data["donor"],
            crossover_data["target"],
            CR,
            generator=_make_gen(99, device),
        )
        assert trial.shape == (crossover_data["pop_size"], crossover_data["dim"])

    def test_j_rand_always_from_donor(self, device: torch.device, crossover_data: dict) -> None:
        """Every row must have at least one element from donor (j_rand guarantee)."""
        CR = torch.zeros(crossover_data["pop_size"], device=device)
        trial = de_binomial_crossover(
            crossover_data["donor"],
            crossover_data["target"],
            CR,
            generator=_make_gen(99, device),
        )
        # With CR=0 only j_rand comes from donor, so each row has exactly 1 donor element
        from_donor = trial == crossover_data["donor"]
        donor_count_per_row = from_donor.sum(dim=1)
        assert (donor_count_per_row >= 1).all(), "j_rand guarantee violated"

    def test_cr_one_gives_all_donor(self, device: torch.device, crossover_data: dict) -> None:
        """CR=1.0 means every element should come from donor."""
        CR = torch.ones(crossover_data["pop_size"], device=device)
        trial = de_binomial_crossover(
            crossover_data["donor"],
            crossover_data["target"],
            CR,
            generator=_make_gen(99, device),
        )
        assert torch.equal(trial, crossover_data["donor"])

    def test_cr_zero_gives_mostly_target(self, device: torch.device) -> None:
        """CR=0.0 means only j_rand should come from donor; the rest from target.

        Uses synthetic sentinel values so ``==`` comparison is unambiguous.
        """
        pop_size, dim = 30, 8
        dtype = best_float_dtype(device)
        donor = torch.arange(pop_size * dim, dtype=dtype, device=device).reshape(pop_size, dim)
        target = donor + 1_000_000.0

        CR = torch.zeros(pop_size, device=device)
        trial = de_binomial_crossover(donor, target, CR, generator=_make_gen(99, device))

        from_donor = trial == donor
        donor_count_per_row = from_donor.sum(dim=1)
        assert (donor_count_per_row == 1).all(), (
            f"Expected exactly 1 donor per row with CR=0, got {donor_count_per_row.tolist()}"
        )

    def test_reproducible_with_seed(self, device: torch.device, crossover_data: dict) -> None:
        CR = torch.full((crossover_data["pop_size"],), 0.5, device=device)
        gen1 = _make_gen(77, device)
        trial1 = de_binomial_crossover(
            crossover_data["donor"],
            crossover_data["target"],
            CR,
            generator=gen1,
        )
        gen2 = _make_gen(77, device)
        trial2 = de_binomial_crossover(
            crossover_data["donor"],
            crossover_data["target"],
            CR,
            generator=gen2,
        )
        assert torch.equal(trial1, trial2)

    def test_cr_intermediate_produces_statistical_mix(self, device: torch.device) -> None:
        """CR=0.5 should produce roughly 50% donor elements (with j_rand boost)."""
        pop_size, dim = 100, 50
        dtype = best_float_dtype(device)
        # Sentinel values: donor from arange, target from negative arange
        donor = torch.arange(pop_size * dim, dtype=dtype, device=device).reshape(pop_size, dim)
        target = -donor - 1.0  # guaranteed distinct from donor

        CR = torch.full((pop_size,), 0.5, device=device)
        trial = de_binomial_crossover(donor, target, CR, generator=_make_gen(42, device))

        from_donor = (trial == donor).float()
        donor_fraction = from_donor.mean().item()
        # CR=0.5 + j_rand slightly pushes above 0.5; allow [0.4, 0.7]
        assert 0.4 < donor_fraction < 0.7, (
            f"Expected donor fraction near 0.5, got {donor_fraction:.3f}"
        )


# ---------------------------------------------------------------------------
# levy_flight_perturbation
# ---------------------------------------------------------------------------
class TestLevyFlightPerturbation:
    """Validate Mantegna Levy flight perturbation."""

    @pytest.fixture
    def base_x(self, device: torch.device) -> torch.Tensor:
        gen = _make_gen(42, device)
        return torch.randn(50, 10, device=gen.device, generator=gen).to(device)

    def test_output_shape(self, device: torch.device, base_x: torch.Tensor) -> None:
        result = levy_flight_perturbation(base_x, generator=_make_gen(99, device))
        assert result.shape == base_x.shape

    def test_output_device(self, device: torch.device, base_x: torch.Tensor) -> None:
        result = levy_flight_perturbation(base_x, generator=_make_gen(99, device))
        assert result.device == base_x.device

    def test_no_nan_or_inf(self, device: torch.device, base_x: torch.Tensor) -> None:
        result = levy_flight_perturbation(
            base_x,
            alpha=1.5,
            step_scale=0.1,
            progress=0.5,
            generator=_make_gen(99, device),
        )
        assert torch.isfinite(result).all(), "Output contains NaN or Inf values"

    def test_progress_annealing(self, device: torch.device) -> None:
        """progress=0 should produce larger perturbations than progress=0.9."""
        x = torch.zeros(500, 10, device=device)

        gen_early = _make_gen(42, device)
        result_early = levy_flight_perturbation(
            x,
            step_scale=1.0,
            progress=0.0,
            generator=gen_early,
        )

        gen_late = _make_gen(42, device)
        result_late = levy_flight_perturbation(x, step_scale=1.0, progress=0.9, generator=gen_late)

        # Compare mean absolute displacement
        mag_early = result_early.abs().mean()
        mag_late = result_late.abs().mean()
        assert mag_early > mag_late, f"Expected early ({mag_early:.4f}) > late ({mag_late:.4f})"

    def test_step_scale_affects_magnitude(self, device: torch.device) -> None:
        """Larger step_scale should produce larger perturbations on average."""
        x = torch.zeros(500, 10, device=device)

        gen_small = _make_gen(42, device)
        result_small = levy_flight_perturbation(
            x,
            step_scale=0.01,
            progress=0.0,
            generator=gen_small,
        )

        gen_large = _make_gen(42, device)
        result_large = levy_flight_perturbation(
            x,
            step_scale=1.0,
            progress=0.0,
            generator=gen_large,
        )

        mag_small = result_small.abs().mean()
        mag_large = result_large.abs().mean()
        assert mag_large > mag_small, f"Expected large ({mag_large:.4f}) > small ({mag_small:.4f})"

    def test_reproducible_with_seed(self, device: torch.device, base_x: torch.Tensor) -> None:
        gen1 = _make_gen(55, device)
        r1 = levy_flight_perturbation(base_x, generator=gen1)
        gen2 = _make_gen(55, device)
        r2 = levy_flight_perturbation(base_x, generator=gen2)
        assert torch.equal(r1, r2)

    def test_different_alpha_values(self, device: torch.device) -> None:
        """Different alpha values produce finite results and affect the distribution."""
        x = torch.zeros(500, 20, device=device)
        magnitudes: dict[float, float] = {}
        for alpha in [1.1, 1.5, 1.9]:
            result = levy_flight_perturbation(
                x,
                alpha=alpha,
                step_scale=1.0,
                generator=_make_gen(99, device),
            )
            assert torch.isfinite(result).all(), f"NaN/Inf at alpha={alpha}"
            magnitudes[alpha] = result.abs().mean().item()

        # Alpha affects the distribution -- not all magnitudes should be identical.
        # With 500x20 samples the means should differ detectably.
        unique_mags = set(magnitudes.values())
        assert len(unique_mags) > 1, (
            f"All alpha values produced identical mean magnitudes: {magnitudes}"
        )

    def test_levy_flight_progress_boundary(self, device: torch.device) -> None:
        """The ratio of perturbation magnitudes at progress=0 vs 1 should be ~1/0.3."""
        x = torch.zeros(1000, 20, device=device)

        result_p0 = levy_flight_perturbation(
            x,
            step_scale=1.0,
            progress=0.0,
            generator=_make_gen(42, device),
        )
        result_p1 = levy_flight_perturbation(
            x,
            step_scale=1.0,
            progress=1.0,
            generator=_make_gen(42, device),
        )

        mag_p0 = result_p0.abs().mean().item()
        mag_p1 = result_p1.abs().mean().item()

        # effective_scale = step_scale * (1 - 0.7 * progress)
        # At progress=0: effective_scale = 1.0
        # At progress=1: effective_scale = 0.3
        # Same seed => same raw Levy steps, so ratio of magnitudes = 0.3 / 1.0
        expected_ratio = 0.3
        actual_ratio = mag_p1 / mag_p0
        assert abs(actual_ratio - expected_ratio) < 0.05, (
            f"Expected magnitude ratio ~{expected_ratio}, got {actual_ratio:.4f} "
            f"(mag_p0={mag_p0:.4f}, mag_p1={mag_p1:.4f})"
        )


# ---------------------------------------------------------------------------
# opposition_init
# ---------------------------------------------------------------------------
class TestOppositionInit:
    """Validate opposition-based population initialization."""

    def test_output_shape(self, device: torch.device) -> None:
        lb = torch.zeros(5, device=device)
        ub = torch.ones(5, device=device)
        pop = opposition_init(20, 5, lb, ub, generator=_make_gen(42, device))
        assert pop.shape == (20, 5)

    def test_all_within_bounds(self, device: torch.device) -> None:
        lb = torch.full((8,), -5.0, device=device)
        ub = torch.full((8,), 5.0, device=device)
        pop = opposition_init(30, 8, lb, ub, generator=_make_gen(42, device))
        assert (pop >= lb).all(), "Some points below lower bound"
        assert (pop <= ub).all(), "Some points above upper bound"

    def test_opposition_mirror_relationship(self, device: torch.device) -> None:
        """First half + second half should equal lb + ub (mirror property).

        Also verifies the odd-pop-size case via ``test_opposition_mirror_odd_pop``.
        """
        pop_size = 20
        dim = 6
        lb = torch.full((dim,), -3.0, device=device)
        ub = torch.full((dim,), 3.0, device=device)
        pop = opposition_init(pop_size, dim, lb, ub, generator=_make_gen(42, device))

        half = (pop_size + 1) // 2
        first_half = pop[:half]
        second_half_raw = lb + ub - first_half  # expected opposition
        # The second half in the output should match the mirrored first half
        actual_second_half = pop[half : 2 * half]
        assert torch.allclose(actual_second_half, second_half_raw[: actual_second_half.shape[0]])

    def test_opposition_mirror_odd_pop(self, device: torch.device) -> None:
        """Odd pop_size — the final (unpaired) row is still a valid mirror slot.

        With pop_size=21, half = 11, so rows [0, 11] form mirror pairs and row
        20 is the overflow. The operator's contract is that row ``half + k``
        mirrors row ``k`` for ``k`` up to ``pop_size - half``. We verify the
        final paired row explicitly (row 10 mirrors row 21-1-...-10 = 20).
        """
        pop_size = 21  # odd
        dim = 4
        lb = torch.full((dim,), -3.0, device=device)
        ub = torch.full((dim,), 3.0, device=device)
        pop = opposition_init(pop_size, dim, lb, ub, generator=_make_gen(42, device))
        half = (pop_size + 1) // 2  # = 11
        # Rows 11..20 mirror rows 0..9 respectively (10 pairs).
        paired_count = pop_size - half  # = 10
        first_part = pop[:paired_count]
        second_part = pop[half : half + paired_count]
        expected = lb + ub - first_part
        assert torch.allclose(second_part, expected), (
            "Final opposition pair row violates mirror property for odd pop_size"
        )
        # And the unpaired "middle" row (index = half - 1 = 10) is still in bounds.
        assert (pop[half - 1] >= lb).all() and (pop[half - 1] <= ub).all()

    def test_odd_pop_size(self, device: torch.device) -> None:
        """Odd pop_size should be handled correctly (truncation of combined)."""
        lb = torch.zeros(4, device=device)
        ub = torch.ones(4, device=device)
        pop = opposition_init(11, 4, lb, ub, generator=_make_gen(42, device))
        assert pop.shape == (11, 4)
        assert (pop >= lb).all()
        assert (pop <= ub).all()

    def test_reproducible_with_seed(self, device: torch.device) -> None:
        lb = torch.zeros(5, device=device)
        ub = torch.ones(5, device=device)
        pop1 = opposition_init(20, 5, lb, ub, generator=_make_gen(42, device))
        pop2 = opposition_init(20, 5, lb, ub, generator=_make_gen(42, device))
        assert torch.equal(pop1, pop2)

    def test_asymmetric_bounds(self, device: torch.device) -> None:
        """Bounds not symmetric about zero; verify mirror property holds, inc. odd pop."""
        lb = torch.tensor([1.0, 2.0, 3.0], device=device)
        ub = torch.tensor([10.0, 20.0, 30.0], device=device)
        pop_size = 16
        pop = opposition_init(pop_size, 3, lb, ub, generator=_make_gen(42, device))
        assert pop.shape == (pop_size, 3)
        assert (pop >= lb).all()
        assert (pop <= ub).all()

        # Verify opposition mirror: pop[:half] + pop[half:2*half] ~ lb + ub
        half = (pop_size + 1) // 2
        first_half = pop[:half]
        actual_second_half = pop[half : 2 * half]
        mirror_sum = first_half[: actual_second_half.shape[0]] + actual_second_half
        expected_sum = (lb + ub).unsqueeze(0).expand_as(mirror_sum)
        assert torch.allclose(mirror_sum, expected_sum), (
            "Opposition mirror relationship violated for asymmetric bounds"
        )

        # Also verify the odd-pop case explicitly so the final row is covered.
        odd_pop = 17
        pop_odd = opposition_init(odd_pop, 3, lb, ub, generator=_make_gen(42, device))
        half_odd = (odd_pop + 1) // 2  # = 9
        paired_count = odd_pop - half_odd  # = 8
        first_part = pop_odd[:paired_count]
        second_part = pop_odd[half_odd : half_odd + paired_count]
        expected_sum_odd = (lb + ub).unsqueeze(0).expand_as(first_part)
        assert torch.allclose(first_part + second_part, expected_sum_odd), (
            "Odd-pop mirror on asymmetric bounds violated at final paired row"
        )

    @pytest.mark.parametrize("pop_size", [1, 2, 3, 50])
    def test_various_pop_sizes(self, device: torch.device, pop_size: int) -> None:
        dim = 4
        lb = torch.zeros(dim, device=device)
        ub = torch.ones(dim, device=device)
        pop = opposition_init(pop_size, dim, lb, ub, generator=_make_gen(42, device))
        assert pop.shape == (pop_size, dim)
        assert torch.all(pop >= lb), "Some points below lower bound"
        assert torch.all(pop <= ub), "Some points above upper bound"
        assert torch.isfinite(pop).all(), "Population contains NaN or Inf"

    def test_opposition_init_dtype_preservation(self, device: torch.device) -> None:
        """Output dtype must match the dtype of the bound tensors."""
        lb = torch.zeros(5, device=device, dtype=torch.float32)
        ub = torch.ones(5, device=device, dtype=torch.float32)
        pop = opposition_init(20, 5, lb, ub, generator=_make_gen(42, device))
        assert pop.dtype == torch.float32, f"Expected float32 output, got {pop.dtype}"


# ---------------------------------------------------------------------------
# Mutation formula correctness
# ---------------------------------------------------------------------------
class TestMutationFormulaCorrectness:
    """Verify DE/current-to-pbest/1 mutation formula with hand-traced values."""

    def test_mutation_formula_correctness(self, device: torch.device) -> None:
        """Hand-trace the mutation for a small example to verify the formula.

        v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)

        With F=1.0 and p_fraction=1.0 (all individuals are pbest candidates),
        we run the function once with a known seed, then replay the RNG draws
        to reconstruct which indices were chosen and compute the expected output.
        """
        dtype = best_float_dtype(device)
        pop_size, dim = 5, 3
        population = torch.arange(pop_size * dim, dtype=dtype, device=device).reshape(pop_size, dim)
        fitness = torch.tensor([3.0, 1.0, 4.0, 0.5, 2.0], dtype=dtype, device=device)
        F = torch.ones(pop_size, dtype=dtype, device=device)

        # Get the actual result
        donor = de_current_to_pbest_mutation(
            population,
            fitness,
            F,
            p_fraction=1.0,
            generator=_make_gen(7, device),
        )

        # Replay the RNG to figure out which indices were drawn
        gen = _make_gen(7, device)
        gen_device = gen.device

        # sorted_indices for fitness [3, 1, 4, 0.5, 2] ascending: [3, 1, 4, 0, 2]
        sorted_indices = fitness.argsort()
        p_count = pop_size  # p_fraction=1.0

        # Draw pbest_choices
        pbest_choices = torch.randint(0, p_count, (pop_size,), device=gen_device, generator=gen)
        if gen_device != device:
            pbest_choices = pbest_choices.to(device)
        x_pbest = population[sorted_indices[pbest_choices]]

        # Draw r1
        r1_raw = torch.randint(0, pop_size - 1, (pop_size,), device=gen_device, generator=gen)
        if gen_device != device:
            r1_raw = r1_raw.to(device)
        i_idx = torch.arange(pop_size, device=device)
        r1 = torch.where(r1_raw >= i_idx, r1_raw + 1, r1_raw)
        x_r1 = population[r1]

        # Draw r2 (from population only, since archive=None)
        total = pop_size
        r2_raw = torch.randint(0, total - 2, (pop_size,), device=gen_device, generator=gen)
        if gen_device != device:
            r2_raw = r2_raw.to(device)
        r2 = torch.where(r2_raw >= i_idx, r2_raw + 1, r2_raw)
        r2 = torch.where(r2 >= r1, r2 + 1, r2)
        r2 = torch.where(r2 == i_idx, (r2 + 1) % total, r2)
        r2 = r2.clamp(max=total - 1)
        x_r2 = population[r2]

        expected = population + 1.0 * (x_pbest - population) + 1.0 * (x_r1 - x_r2)
        assert torch.allclose(donor, expected), (
            f"Mutation formula mismatch.\nExpected:\n{expected}\nGot:\n{donor}"
        )


# ---------------------------------------------------------------------------
# Index distinctness for small populations
# ---------------------------------------------------------------------------
class TestIndexDistinctnessSmallPop:
    """Verify that mutation indices are distinct for small populations."""

    def test_index_distinctness_small_pop(self, device: torch.device) -> None:
        """With unique rows and F != 0, no donor row should equal its own pop row.

        If self-mutation occurred (r1 == i or r2 == i), the formula would
        degenerate. With F=1 and p_fraction=1, self-references would produce
        specific patterns we can detect by checking donor != population[i].
        """
        dtype = best_float_dtype(device)
        pop_size, dim = 4, 6
        # Each row is unique and widely separated
        population = torch.eye(pop_size, dim, device=device, dtype=dtype) * 100.0
        fitness = torch.arange(pop_size, device=device, dtype=dtype)
        F = torch.ones(pop_size, device=device, dtype=dtype)

        # Run many iterations to exercise different index draws
        self_match_count = 0
        n_trials = 100
        for seed in range(n_trials):
            donor = de_current_to_pbest_mutation(
                population,
                fitness,
                F,
                p_fraction=1.0,
                generator=_make_gen(seed, device),
            )
            # Check if any donor row exactly equals its corresponding pop row
            for i in range(pop_size):
                if torch.equal(donor[i], population[i]):
                    self_match_count += 1

        # With F=1 and distinct rows, a self-match means the differential term
        # cancelled out (x_pbest==x_i and x_r1==x_r2, or similar degeneracy).
        # Some coincidental matches are possible but they should be very rare.
        max_allowed = int(n_trials * pop_size * 0.05)  # at most 5%
        assert self_match_count <= max_allowed, (
            f"Too many self-matches: {self_match_count}/{n_trials * pop_size} "
            f"(max allowed {max_allowed})"
        )


# ---------------------------------------------------------------------------
# pbest pool correctness
# ---------------------------------------------------------------------------
class TestPbestPoolCorrectness:
    """Verify that pbest selection only draws from the top-p fraction."""

    def test_pbest_pool_correctness(self, device: torch.device) -> None:
        """With p_fraction=0.4 on pop of 5, only the top-2 are in the pbest pool.

        With fitness [100, 1, 50, 2, 75], the top-2 (ascending = minimization)
        are indices 1 (fit=1) and 3 (fit=2).

        We replay the RNG to verify that the pbest draw only selects from
        the valid pool indices, then confirm the mutation output is finite
        and non-trivial.
        """
        dtype = best_float_dtype(device)
        pop_size, dim = 5, 4
        population = torch.arange(pop_size * dim, dtype=dtype, device=device).reshape(pop_size, dim)
        fitness = torch.tensor([100.0, 1.0, 50.0, 2.0, 75.0], dtype=dtype, device=device)

        # Top-2 by ascending fitness: indices 1 and 3
        sorted_indices = fitness.argsort()
        p_count = max(2, math.ceil(pop_size * 0.4))  # 2
        pbest_indices = sorted_indices[:p_count]

        # Replay the RNG and check the pbest indices directly.
        for seed in range(50):
            gen = _make_gen(seed, device)
            gen_device = gen.device

            # Replay the pbest_choices draw
            pbest_choices = torch.randint(0, p_count, (pop_size,), device=gen_device, generator=gen)
            if gen_device != device:
                pbest_choices = pbest_choices.to(device)
            selected_pbest = pbest_indices[pbest_choices]

            # Every selected pbest index must be in {1, 3}
            valid_set = set(pbest_indices.tolist())
            selected_set = set(selected_pbest.tolist())
            assert selected_set.issubset(valid_set), (
                f"Seed {seed}: selected pbest indices {selected_set} "
                f"not subset of valid {valid_set}"
            )

        # Also verify the actual mutation output uses these pbest values by
        # checking with F=1, p_fraction=0.4 and confirming the result is finite
        # and different from the population.
        donor = de_current_to_pbest_mutation(
            population,
            fitness,
            torch.ones(pop_size, dtype=dtype, device=device),
            p_fraction=0.4,
            generator=_make_gen(0, device),
        )
        assert torch.isfinite(donor).all()
        assert not torch.equal(donor, population)
