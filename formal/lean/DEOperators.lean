/-
  Formal Verification of Vectorized Differential Evolution Operators
  Target: torch-dfo `src/torch_dfo/_operators.py`
    - `de_current_to_pbest_mutation` (lines 45-70): index-shifting maps for r1 and r2
    - `opposition_init` (lines 154-185): mirror bounds preservation
    - `de_binomial_crossover` (lines 77-107): forced j_rand guarantee

  Machine-checked by Lean 4.16.0 (pure Init, zero external deps, zero `sorry`).
-/

namespace TorchDFO.DEOperators

/-! ### 1. Index map for `r1` (`_operators.py:46-50`) -/

/-- Exact model of: `r1 = torch.where(r1_raw >= i, r1_raw + 1, r1_raw)` -/
def r1Map (i r1_raw : Nat) : Nat :=
  if r1_raw >= i then r1_raw + 1 else r1_raw

theorem r1Map_in_bounds {popSize i r1_raw : Nat}
    (hi : i < popSize) (hr : r1_raw < popSize - 1) :
    r1Map i r1_raw < popSize := by
  unfold r1Map
  split <;> omega

theorem r1Map_ne_self {i r1_raw : Nat} :
    r1Map i r1_raw ≠ i := by
  unfold r1Map
  split <;> omega

theorem r1Map_injective {i a b : Nat}
    (h : r1Map i a = r1Map i b) : a = b := by
  unfold r1Map at h
  split at h <;> split at h <;> omega

theorem r1Map_surjective {popSize i y : Nat}
    (hi : i < popSize) (hy : y < popSize) (hne : y ≠ i) :
    ∃ r1_raw, r1_raw < popSize - 1 ∧ r1Map i r1_raw = y := by
  by_cases hlt : y < i
  · exact ⟨y, by omega, by unfold r1Map; split <;> omega⟩
  · exact ⟨y - 1, by omega, by unfold r1Map; split <;> omega⟩


/-! ### 2. Index map for `r2` (`_operators.py:60-69`) -/

/-- Exact model of `_operators.py:64-69`:
    ```python
    r2 = torch.where(r2 >= i_idx, r2 + 1, r2)
    r2 = torch.where(r2 >= r1, r2 + 1, r2)
    r2 = torch.where(r2 == i_idx, (r2 + 1) % total, r2)
    r2 = r2.clamp(max=total - 1)
    ``` -/
def r2Map (total i r1 r2_raw : Nat) : Nat :=
  let s1 := if r2_raw >= i then r2_raw + 1 else r2_raw
  let s2 := if s1 >= r1 then s1 + 1 else s1
  let s3 := if s2 = i then (s2 + 1) % total else s2
  if s3 <= total - 1 then s3 else total - 1

/-- Canonical order-preserving bijection from `{0..total-3}` onto `{0..total-1} \ {i, r1}`. -/
def r2Canonical (i r1 x : Nat) : Nat :=
  let lo := if i <= r1 then i else r1
  let hi := if i <= r1 then r1 else i
  let a := if x >= lo then x + 1 else x
  if a >= hi then a + 1 else a

/-- Core theorem: `r2Map` is IDENTICAL to the canonical order-preserving bijection
    `r2Canonical` for ALL `total ≥ 3`, `i < total`, `r1 < total`, `r1 ≠ i`, `r2_raw < total - 2`.
    (In particular, neither `% total` wraparound nor `.clamp(max=total-1)` ever activates.) -/
theorem r2Map_eq_canonical {total i r1 r2_raw : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (_hneq : r1 ≠ i) (hr2 : r2_raw < total - 2) :
    r2Map total i r1 r2_raw = r2Canonical i r1 r2_raw := by
  unfold r2Map r2Canonical
  dsimp only
  by_cases h1 : r2_raw ≥ i
  · simp only [h1, ↓reduceIte]
    by_cases h2 : r2_raw + 1 ≥ r1
    · simp only [h2, ↓reduceIte]
      have hne : r2_raw + 1 + 1 ≠ i := by omega
      have hle : r2_raw + 1 + 1 ≤ total - 1 := by omega
      simp only [hne, hle, ↓reduceIte]
      by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega
    · simp only [h2, ↓reduceIte]
      have hne : r2_raw + 1 ≠ i := by omega
      have hle : r2_raw + 1 ≤ total - 1 := by omega
      simp only [hne, hle, ↓reduceIte]
      by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega
  · simp only [h1, ↓reduceIte]
    by_cases h2 : r2_raw ≥ r1
    · simp only [h2, ↓reduceIte]
      by_cases h3 : r2_raw + 1 = i
      · simp only [h3, ↓reduceIte]
        have hlt : i + 1 < total := by omega
        rw [Nat.mod_eq_of_lt hlt]
        have hle : i + 1 ≤ total - 1 := by omega
        simp only [hle, ↓reduceIte]
        by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega
      · simp only [h3, ↓reduceIte]
        have hle : r2_raw + 1 ≤ total - 1 := by omega
        simp only [hle, ↓reduceIte]
        by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega
    · simp only [h2, ↓reduceIte]
      have hne : r2_raw ≠ i := by omega
      have hle : r2_raw ≤ total - 1 := by omega
      simp only [hne, hle, ↓reduceIte]
      by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega

theorem r2Map_in_bounds {total i r1 r2_raw : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (hneq : r1 ≠ i) (hr2 : r2_raw < total - 2) :
    r2Map total i r1 r2_raw < total := by
  rw [r2Map_eq_canonical htot hi hr1 hneq hr2]
  unfold r2Canonical; dsimp only
  by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega

theorem r2Map_ne_i {total i r1 r2_raw : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (hneq : r1 ≠ i) (hr2 : r2_raw < total - 2) :
    r2Map total i r1 r2_raw ≠ i := by
  rw [r2Map_eq_canonical htot hi hr1 hneq hr2]
  unfold r2Canonical; dsimp only
  by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega

theorem r2Map_ne_r1 {total i r1 r2_raw : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (hneq : r1 ≠ i) (hr2 : r2_raw < total - 2) :
    r2Map total i r1 r2_raw ≠ r1 := by
  rw [r2Map_eq_canonical htot hi hr1 hneq hr2]
  unfold r2Canonical; dsimp only
  by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] <;> (try split) <;> (try split) <;> omega

theorem r2Map_injective {total i r1 a b : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (hneq : r1 ≠ i) (ha : a < total - 2) (hb : b < total - 2)
    (heq : r2Map total i r1 a = r2Map total i r1 b) : a = b := by
  rw [r2Map_eq_canonical htot hi hr1 hneq ha,
      r2Map_eq_canonical htot hi hr1 hneq hb] at heq
  unfold r2Canonical at heq; dsimp only at heq
  by_cases hord : i ≤ r1 <;> simp only [hord, ↓reduceIte] at heq <;>
    (try split at heq) <;> (try split at heq) <;> (try split at heq) <;> (try split at heq) <;> omega

theorem r2Map_surjective {total i r1 y : Nat}
    (htot : total ≥ 3) (hi : i < total) (hr1 : r1 < total)
    (hneq : r1 ≠ i) (hy : y < total) (hyi : y ≠ i) (hyr1 : y ≠ r1) :
    ∃ r2_raw, r2_raw < total - 2 ∧ r2Map total i r1 r2_raw = y := by
  by_cases hord : i ≤ r1
  · by_cases h1 : y < i
    · refine ⟨y, by omega, ?_⟩
      rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
      unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega
    · by_cases h2 : y < r1
      · refine ⟨y - 1, by omega, ?_⟩
        rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
        unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega
      · refine ⟨y - 2, by omega, ?_⟩
        rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
        unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega
  · by_cases h1 : y < r1
    · refine ⟨y, by omega, ?_⟩
      rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
      unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega
    · by_cases h2 : y < i
      · refine ⟨y - 1, by omega, ?_⟩
        rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
        unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega
      · refine ⟨y - 2, by omega, ?_⟩
        rw [r2Map_eq_canonical htot hi hr1 hneq (by omega)]
        unfold r2Canonical; simp only [hord, ↓reduceIte]; (try split) <;> (try split) <;> omega


/-! ### 3. Binomial crossover forced `j_rand` guarantee (`_operators.py:98-107`) -/

def crossoverElem (mask : Nat → Bool) (j_rand : Nat) (donor target : Nat → Int) (j : Nat) : Int :=
  if (j == j_rand) || mask j then donor j else target j

theorem crossover_forces_j_rand (mask : Nat → Bool) (j_rand : Nat) (donor target : Nat → Int) :
    crossoverElem mask j_rand donor target j_rand = donor j_rand := by
  unfold crossoverElem
  simp

theorem crossover_exists_donor_coord {dim j_rand : Nat} (hj : j_rand < dim)
    (mask : Nat → Bool) (donor target : Nat → Int) :
    ∃ j, j < dim ∧ crossoverElem mask j_rand donor target j = donor j :=
  ⟨j_rand, hj, crossover_forces_j_rand mask j_rand donor target⟩


/-! ### 4. Opposition initialization mirror bounds (`_operators.py:178-185`) -/

theorem opposition_mirror_bounds {lb ub x : Int} (hl : lb ≤ x) (hu : x ≤ ub) :
    lb ≤ lb + ub - x ∧ lb + ub - x ≤ ub ∧ x + (lb + ub - x) = lb + ub := by
  omega

end TorchDFO.DEOperators
