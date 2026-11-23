/-
  Formal Verification of SearchRun and DFOOptimizer Budget Accounting State Machine
  Targets:
    - `src/torch_dfo/base.py` (`BaseOptimizer.next_batch_size`)
    - `src/torch_dfo/nelder_mead.py` (`NelderMead.next_batch_size`, `NelderMead.ask`)
    - `src/torch_dfo/run.py` (`SearchRun`)
    - `src/torch_dfo/optim/dfo_optimizer.py` (`DFOOptimizer`)

  Proves (pure Lean 4.16.0 Init, zero external deps, zero `sorry`):
  1. Inductive preservation of all accounting invariants across `init`, `stepAsk`, `stepSettle`
     for ANY dynamic `optimizer.next_batch_size()`.
  2. Exact shape parity: `NelderMead.next_batch_size()` equals `NelderMead.ask().shape[0]`
     in every `(initialized, needsFullEval)` state for all `dim ≥ 1`.
  3. `DFOOptimizer` terminal-residue guard using `next_batch_size()` never overruns
     `budgetRemaining` for any `dim ≥ 1` (closing the `dim ∈ {1, 2}` AssertionError).
-/

namespace TorchDFO.SearchRun

inductive RunState where
  | ready
  | stopped
  | failed
  deriving DecidableEq, Repr

structure Pending where
  count : Nat
  reservation : Nat
  deriving Repr

structure State where
  maxEvals : Nat
  repeats : Nat
  popSize : Nat
  isRandom : Bool
  status : RunState
  scheduled : Nat
  charged : Nat
  attempted : Nat
  completed : Nat
  reserved : Nat
  pending : Option Pending
  deriving Repr

/-- Core accounting invariant of `SearchRun` (`run.py`). -/
def Valid (s : State) : Prop :=
  s.repeats ≥ 1 ∧
  s.completed ≤ s.attempted ∧
  s.attempted ≤ s.charged ∧
  s.charged + s.reserved ≤ s.maxEvals ∧
  s.charged + s.reserved ≤ s.scheduled ∧
  match s.pending with
  | none => s.reserved = 0
  | some p => s.reserved = p.reservation ∧ p.reservation = p.count * s.repeats ∧ s.status = RunState.ready

def init (maxEvals repeats popSize : Nat) (isRandom : Bool) : State :=
  { maxEvals, repeats, popSize, isRandom
    status := RunState.ready
    scheduled := 0
    charged := 0
    attempted := 0
    completed := 0
    reserved := 0
    pending := none }

theorem init_valid {maxEvals repeats popSize : Nat} {isRandom : Bool} (hr : repeats ≥ 1) :
    Valid (init maxEvals repeats popSize isRandom) := by
  unfold Valid init
  exact ⟨hr, Nat.le_refl 0, Nat.le_refl 0, Nat.zero_le maxEvals, Nat.le_refl 0, rfl⟩

def remainingEvals (s : State) : Nat :=
  s.maxEvals - s.charged - s.reserved

/-- Exact model of fixed `SearchRun._next_batch_size()` (`run.py`) querying
    `optNextBatch = self.optimizer.next_batch_size()`. -/
def nextBatchSize (s : State) (optNextBatch : Nat) : Nat :=
  if s.isRandom then
    Nat.min s.popSize (remainingEvals s / s.repeats)
  else
    if optNextBatch * s.repeats ≤ remainingEvals s then optNextBatch else 0

theorem nextBatchSize_mul_le_remaining (s : State) (optNextBatch : Nat) :
    nextBatchSize s optNextBatch * s.repeats ≤ remainingEvals s := by
  unfold nextBatchSize
  split
  · have h1 : Nat.min s.popSize (remainingEvals s / s.repeats) ≤ remainingEvals s / s.repeats :=
      Nat.min_le_right _ _
    have h2 : Nat.min s.popSize (remainingEvals s / s.repeats) * s.repeats ≤
        (remainingEvals s / s.repeats) * s.repeats :=
      Nat.mul_le_mul_right s.repeats h1
    have h3 : (remainingEvals s / s.repeats) * s.repeats ≤ remainingEvals s := by
      rw [Nat.mul_comm]; exact Nat.mul_div_le (remainingEvals s) s.repeats
    exact Nat.le_trans h2 h3
  · split
    · assumption
    · simp

/-- `ask()` transition when quiescent and `ready`, parameterized by `optNextBatch`. -/
def stepAsk (s : State) (optNextBatch : Nat) : State :=
  if s.status ≠ RunState.ready ∨ s.pending.isSome then s
  else
    let count := nextBatchSize s optNextBatch
    if count = 0 then
      { s with status := RunState.stopped }
    else
      let res := count * s.repeats
      { s with
        pending := some { count, reservation := res }
        scheduled := s.scheduled + res
        reserved := res }

theorem stepAsk_preserves_valid {s : State} (optNextBatch : Nat) (hv : Valid s) :
    Valid (stepAsk s optNextBatch) := by
  unfold stepAsk
  by_cases hguard : s.status ≠ RunState.ready ∨ s.pending.isSome
  · simp only [hguard, ↓reduceIte]; exact hv
  · simp only [hguard, ↓reduceIte]
    have hstatus : s.status = RunState.ready := by
      cases hs : s.status with
      | ready => rfl
      | stopped => exfalso; apply hguard; left; intro hc; rw [hs] at hc; cases hc
      | failed => exfalso; apply hguard; left; intro hc; rw [hs] at hc; cases hc
    have hpend : s.pending = none := by
      cases hp : s.pending with
      | none => rfl
      | some _ => exfalso; apply hguard; right; simp [hp]
    obtain ⟨hr, hca, hac, hcm, hcs, hmatch⟩ := hv
    rw [hpend] at hmatch
    have hres0 : s.reserved = 0 := hmatch
    by_cases hcnt : nextBatchSize s optNextBatch = 0
    · simp only [hcnt, ↓reduceIte]
      unfold Valid
      simp only [hpend]
      exact ⟨hr, hca, hac, hcm, hcs, hres0⟩
    · simp only [hcnt, ↓reduceIte]
      have hmul := nextBatchSize_mul_le_remaining s optNextBatch
      unfold remainingEvals at hmul
      unfold Valid
      dsimp only
      exact ⟨hr, hca, hac, by omega, by omega, rfl, rfl, hstatus⟩

/-- `tell()` settlement transition (`_settle` in `run.py`) for both success and failure. -/
def stepSettle (s : State) (attemptedBatch completedBatch charge : Nat) (failed : Bool) : State :=
  match s.pending with
  | none => s
  | some _ =>
    { s with
      charged := s.charged + charge
      attempted := s.attempted + attemptedBatch
      completed := s.completed + completedBatch
      pending := none
      reserved := 0
      status := if failed then RunState.failed else s.status }

theorem stepSettle_preserves_valid {s : State} {p : Pending}
    (hv : Valid s) (hp : s.pending = some p)
    (hcb : completedBatch ≤ attemptedBatch)
    (hab : attemptedBatch ≤ charge)
    (hcr : charge ≤ p.reservation)
    (failed : Bool) :
    Valid (stepSettle s attemptedBatch completedBatch charge failed) := by
  unfold stepSettle Valid
  rw [hp]
  dsimp only
  obtain ⟨hr, hca, hac, hcm, hcs, hmatch⟩ := hv
  rw [hp] at hmatch
  obtain ⟨hres, _, _⟩ := hmatch
  refine ⟨hr, ?_, ?_, ?_, ?_, rfl⟩ <;> omega

theorem stepSettle_monotone {s : State} {p : Pending} (hp : s.pending = some p)
    (a c ch : Nat) (failed : Bool) :
    let s' := stepSettle s a c ch failed
    s.charged ≤ s'.charged ∧ s.attempted ≤ s'.attempted ∧ s.completed ≤ s'.completed := by
  unfold stepSettle
  rw [hp]
  dsimp only
  omega


/-! ### 2. Fixed `NelderMead.next_batch_size()` Parity (`nelder_mead.py`, `run.py`, `dfo_optimizer.py`) -/

def nmNextBatchSize (initialized needsFullEval : Bool) (dim : Nat) : Nat :=
  if !initialized || needsFullEval then dim + 1 else 4

def nmAskShape (initialized needsFullEval : Bool) (dim : Nat) : Nat :=
  if !initialized then dim + 1
  else if needsFullEval then dim + 1
  else 4

/-- Exact shape contract: `NelderMead.next_batch_size()` ALWAYS matches `NelderMead.ask().shape[0]`
    for every `dim ≥ 1` and every `(initialized, needsFullEval)` state. -/
theorem nm_nextBatchSize_eq_askShape (initialized needsFullEval : Bool) (dim : Nat) :
    nmNextBatchSize initialized needsFullEval dim = nmAskShape initialized needsFullEval dim := by
  cases initialized <;> cases needsFullEval <;> rfl

/-- Fixed `SearchRun` + `NelderMead`: whenever budget suffices for the next Nelder-Mead step,
    the reserved batch count equals the exact shape returned by `NelderMead.ask()`. -/
theorem searchRun_nelderMead_shape_parity
    {s : State} (initialized needsFullEval : Bool) (dim : Nat)
    (hnotrand : s.isRandom = false)
    (hfit : nmNextBatchSize initialized needsFullEval dim * s.repeats ≤ remainingEvals s) :
    nextBatchSize s (nmNextBatchSize initialized needsFullEval dim) =
    nmAskShape initialized needsFullEval dim := by
  unfold nextBatchSize
  simp only [hnotrand, Bool.false_eq_true, ↓reduceIte, hfit]
  exact nm_nextBatchSize_eq_askShape initialized needsFullEval dim

/-- Fixed `DFOOptimizer` guard: checking `next_batch_size() ≤ budgetRemaining` before calling
    `ask()` guarantees the returned batch NEVER exceeds `budgetRemaining` for any `dim ≥ 1`. -/
def dfoCanStep (budgetRemaining nextBatch : Nat) : Bool :=
  decide (nextBatch ≤ budgetRemaining)

theorem dfo_nelderMead_never_overruns_budget
    (initialized needsFullEval : Bool) (dim budgetRemaining : Nat)
    (hcan : dfoCanStep budgetRemaining (nmNextBatchSize initialized needsFullEval dim) = true) :
    nmAskShape initialized needsFullEval dim ≤ budgetRemaining := by
  unfold dfoCanStep at hcan
  rw [decide_eq_true_eq] at hcan
  rw [← nm_nextBatchSize_eq_askShape initialized needsFullEval dim]
  exact hcan

end TorchDFO.SearchRun
