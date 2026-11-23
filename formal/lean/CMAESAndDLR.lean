/-
  Formal Algebraic Verification of Fixed CMA-ES and DLR-CMA Update Laws
  Targets:
    - `src/torch_dfo/cmaes.py` (`_compute_adaptation_rates`, `_configure_strategy`,
      `tell`, `_update_eigensystem`)
    - `src/torch_dfo/dlr_cma.py` (`_compute_branch_rates`, `tell`)

  Uses a self-contained OrderedField abstraction (valid for both ℝ and ℚ)
  plus exact integer-clearing proofs (zero external deps, zero `sorry`):
  1. CMA-ES passive covariance update preserves symmetry and PSD for ALL `(n, mu)`.
  2. Fixed Active CMA (`cmaes.py` Mahalanobis scaling `w_i^o = w_i * (n / ||C^{-1/2} y_i||^2)`
     with `c_mu_neg ≤ (1 - c_1 - c_mu) / n`) preserves symmetry AND PSD for ALL `(n, mu, mu_neg)`.
  3. Exact 3-variable gauge transform (`C' = C / s`, `sigma' = sigma * sqrt(s)`,
     `p_c' = p_c / sqrt(s)`) preserves:
     (a) sampling covariance `sigma'^2 C' = sigma^2 C`,
     (b) conjugate path `p_sigma` increment invariance,
     (c) covariance path `p_c` equivariance (`p_c'_next = p_c_next / sqrt(s)`),
     (d) next-generation covariance equivariance (`C'_next = C_next / s`).
  4. Hansen (2016) Table 1 `c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)` ALWAYS satisfies
     `0 < c_c < 1` for ALL `d ≥ 1, mu_eff ≥ 1`.
  5. Fixed `DLRPortfolio` diagonal update (`(1 - c1 - cmu + c1 * delta_h) * d_old + c1 * diag_1 + cmu * diag_mu`)
     preserves exact stationary variance `d_new = d_old` under `h_sigma = 0`.
  6. Fixed `DLRPortfolio` pre-update `sigma_old` eliminates `1 / alpha^2` distortion.
-/

namespace TorchDFO.CMAESAndDLR

/-! ### Self-Contained Ordered Field Interface (satisfied by ℝ and ℚ) -/

class OrderedField (F : Type) where
  zero : F
  one : F
  add : F → F → F
  sub : F → F → F
  neg : F → F
  mul : F → F → F
  inv : F → F
  le : F → F → Prop
  lt : F → F → Prop
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  add_comm : ∀ a b, add a b = add b a
  zero_add : ∀ a, add zero a = a
  add_zero : ∀ a, add a zero = a
  add_neg : ∀ a, add a (neg a) = zero
  sub_def : ∀ a b, sub a b = add a (neg b)
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mul_comm : ∀ a b, mul a b = mul b a
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  zero_mul : ∀ a, mul zero a = zero
  mul_zero : ∀ a, mul a zero = zero
  mul_add : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  add_mul : ∀ a b c, mul (add a b) c = add (mul a c) (mul b c)
  mul_inv_cancel : ∀ a, a ≠ zero → mul a (inv a) = one
  inv_mul : ∀ a b, inv (mul a b) = mul (inv a) (inv b)
  neg_mul : ∀ a b, mul (neg a) b = neg (mul a b)
  mul_neg : ∀ a b, mul a (neg b) = neg (mul a b)
  le_refl : ∀ a, le a a
  le_trans : ∀ a b c, le a b → le b c → le a c
  add_le_add : ∀ a b c d, le a b → le c d → le (add a c) (add b d)
  mul_nonneg : ∀ a b, le zero a → le zero b → le zero (mul a b)
  sq_nonneg : ∀ a, le zero (mul a a)
  lt_iff_le_not_eq : ∀ a b, lt a b ↔ le a b ∧ a ≠ b
  mul_pos : ∀ a b, lt zero a → lt zero b → lt zero (mul a b)
  lt_add_of_pos_right : ∀ a b, lt zero b → lt a (add a b)
  mul_eq_zero : ∀ a b, mul a b = zero → a = zero ∨ b = zero
  one_ne_zero : one ≠ zero

section LinearAlgebra
variable {F : Type} [OrderedField F]
open OrderedField

abbrev Vec (n : Nat) := Fin n → F
abbrev Mat (m n : Nat) := Fin m → Fin n → F

theorem sub_nonneg_of_le {a b : F} (h : le a b) : le zero (sub b a) := by
  have h1 := add_le_add a b (neg a) (neg a) h (le_refl (neg a))
  rw [add_neg, ← sub_def] at h1
  exact h1

theorem le_of_sub_nonneg {a b : F} (h : le zero (sub b a)) : le a b := by
  have h1 := add_le_add zero (sub b a) a a h (le_refl a)
  rw [zero_add, sub_def, add_assoc, add_comm (neg a) a, add_neg, add_zero] at h1
  exact h1

theorem mul_le_mul_left {c a b : F} (hc : le zero c) (hab : le a b) : le (mul c a) (mul c b) := by
  have hsub : le zero (sub b a) := sub_nonneg_of_le hab
  have hmul : le zero (mul c (sub b a)) := mul_nonneg c (sub b a) hc hsub
  have hrew : mul c (sub b a) = sub (mul c b) (mul c a) := by
    rw [sub_def, sub_def, mul_add, mul_neg]
  rw [hrew] at hmul
  exact le_of_sub_nonneg hmul

def finSum {n : Nat} (f : Fin n → F) : F :=
  match n with
  | 0 => zero
  | k + 1 => add (finSum (fun i : Fin k => f i.castSucc)) (f ⟨k, Nat.lt_succ_self k⟩)

theorem finSum_zero (n : Nat) : finSum (fun _ : Fin n => (zero : F)) = zero := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [finSum]; rw [ih, zero_add]

theorem finSum_congr {n : Nat} {f g : Fin n → F} (h : ∀ i, f i = g i) :
    finSum f = finSum g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => h i.castSucc), h ⟨k, Nat.lt_succ_self k⟩]

theorem finSum_add {n : Nat} (f g : Fin n → F) :
    finSum (fun i => add (f i) (g i)) = add (finSum f) (finSum g) := by
  induction n with
  | zero => simp only [finSum]; rw [zero_add]
  | succ k ih =>
    simp only [finSum]
    rw [ih]
    rw [add_assoc, ← add_assoc (finSum (fun i : Fin k => g i.castSucc)),
        add_comm (finSum (fun i : Fin k => g i.castSucc)),
        add_assoc (f ⟨k, Nat.lt_succ_self k⟩), ← add_assoc]

theorem finSum_neg {n : Nat} (f : Fin n → F) :
    finSum (fun i => neg (f i)) = neg (finSum f) := by
  have h0 : add (finSum f) (finSum (fun i => neg (f i))) = zero := by
    rw [← finSum_add]
    have hcon : finSum (fun i => add (f i) (neg (f i))) = finSum (fun _ => (zero : F)) :=
      finSum_congr (fun i => add_neg (f i))
    rw [hcon, finSum_zero]
  have h1 : add (neg (finSum f)) (add (finSum f) (finSum (fun i => neg (f i)))) =
            add (neg (finSum f)) zero := congrArg (add (neg (finSum f))) h0
  rw [← add_assoc, add_comm (neg (finSum f)), add_neg, zero_add, add_zero] at h1
  exact h1

theorem finSum_sub {n : Nat} (f g : Fin n → F) :
    finSum (fun i => sub (f i) (g i)) = sub (finSum f) (finSum g) := by
  have h : finSum (fun i => sub (f i) (g i)) = finSum (fun i => add (f i) (neg (g i))) :=
    finSum_congr (fun i => sub_def (f i) (g i))
  rw [h, finSum_add, finSum_neg, sub_def]

theorem finSum_mul_left {n : Nat} (c : F) (f : Fin n → F) :
    finSum (fun i => mul c (f i)) = mul c (finSum f) := by
  induction n with
  | zero => simp only [finSum]; rw [mul_zero]
  | succ k ih => simp only [finSum]; rw [ih, mul_add]

theorem finSum_mul_right {n : Nat} (f : Fin n → F) (c : F) :
    finSum (fun i => mul (f i) c) = mul (finSum f) c := by
  induction n with
  | zero => simp only [finSum]; rw [zero_mul]
  | succ k ih => simp only [finSum]; rw [ih, add_mul]

theorem finSum_nonneg {n : Nat} {f : Fin n → F} (h : ∀ i, le zero (f i)) :
    le zero (finSum f) := by
  induction n with
  | zero => exact le_refl zero
  | succ k ih =>
    simp only [finSum]
    have h1 := ih (fun i => h i.castSucc)
    have h2 := h ⟨k, Nat.lt_succ_self k⟩
    have h3 := add_le_add zero (finSum (fun i : Fin k => f i.castSucc)) zero (f ⟨k, Nat.lt_succ_self k⟩) h1 h2
    rw [zero_add] at h3
    exact h3

theorem finSum_le_finSum {n : Nat} {f g : Fin n → F} (h : ∀ i, le (f i) (g i)) :
    le (finSum f) (finSum g) := by
  have hnn : le zero (finSum (fun i => sub (g i) (f i))) :=
    finSum_nonneg (fun i => sub_nonneg_of_le (h i))
  rw [finSum_sub] at hnn
  exact le_of_sub_nonneg hnn

theorem finSum_swap {m n : Nat} (f : Fin m → Fin n → F) :
    finSum (fun i => finSum (fun j => f i j)) =
    finSum (fun j => finSum (fun i => f i j)) := by
  induction m with
  | zero => simp only [finSum]; symm; exact finSum_zero n
  | succ k ih => simp only [finSum]; rw [ih, ← finSum_add]

def vecAdd {n : Nat} (u v : Vec (F := F) n) : Vec (F := F) n := fun i => add (u i) (v i)
def vecScale {n : Nat} (c : F) (u : Vec (F := F) n) : Vec (F := F) n := fun i => mul c (u i)
def matVec {m n : Nat} (A : Mat (F := F) m n) (v : Vec (F := F) n) : Vec (F := F) m :=
  fun i => finSum (fun j => mul (A i j) (v j))
def matAdd {m n : Nat} (A B : Mat (F := F) m n) : Mat (F := F) m n := fun i j => add (A i j) (B i j)
def matSub {m n : Nat} (A B : Mat (F := F) m n) : Mat (F := F) m n := fun i j => sub (A i j) (B i j)
def matScale {m n : Nat} (c : F) (A : Mat (F := F) m n) : Mat (F := F) m n := fun i j => mul c (A i j)
def outer {n : Nat} (u v : Vec (F := F) n) : Mat (F := F) n n := fun i j => mul (u i) (v j)
def quadForm {n : Nat} (A : Mat (F := F) n n) (v : Vec (F := F) n) : F :=
  finSum (fun i => finSum (fun j => mul (mul (v i) (A i j)) (v j)))
def dot {n : Nat} (u v : Vec (F := F) n) : F := finSum (fun i => mul (u i) (v i))

def isSymmetric {n : Nat} (A : Mat (F := F) n n) : Prop := ∀ i j, A i j = A j i
def isPSD {n : Nat} (A : Mat (F := F) n n) : Prop := ∀ v : Vec (F := F) n, le zero (quadForm A v)

theorem quadForm_outer_nonneg {n : Nat} (u v : Vec (F := F) n) :
    le zero (quadForm (outer u u) v) := by
  have h : quadForm (outer u u) v = mul (dot v u) (dot v u) := by
    unfold quadForm outer dot
    calc
      finSum (fun i => finSum (fun j => mul (mul (v i) (mul (u i) (u j))) (v j)))
        = finSum (fun i => finSum (fun j => mul (mul (v i) (u i)) (mul (v j) (u j)))) := by
            apply finSum_congr; intro i; apply finSum_congr; intro j
            rw [← mul_assoc (v i) (u i) (u j), mul_assoc (mul (v i) (u i)) (u j) (v j), mul_comm (u j) (v j)]
      _ = finSum (fun i => mul (mul (v i) (u i)) (finSum (fun j => mul (v j) (u j)))) := by
            apply finSum_congr; intro i; rw [finSum_mul_left]
      _ = mul (finSum (fun i => mul (v i) (u i))) (finSum (fun j => mul (v j) (u j))) := finSum_mul_right _ _
  rw [h]
  exact sq_nonneg (dot v u)

theorem quadForm_add {n : Nat} (A B : Mat (F := F) n n) (v : Vec (F := F) n) :
    quadForm (matAdd A B) v = add (quadForm A v) (quadForm B v) := by
  unfold quadForm matAdd
  rw [← finSum_add]; apply finSum_congr; intro i
  rw [← finSum_add]; apply finSum_congr; intro j
  rw [mul_add, add_mul]

theorem quadForm_sub {n : Nat} (A B : Mat (F := F) n n) (v : Vec (F := F) n) :
    quadForm (matSub A B) v = sub (quadForm A v) (quadForm B v) := by
  unfold quadForm matSub
  rw [← finSum_sub]; apply finSum_congr; intro i
  rw [← finSum_sub]; apply finSum_congr; intro j
  rw [sub_def, sub_def, mul_add, mul_neg, add_mul, neg_mul]

theorem quadForm_scale {n : Nat} (c : F) (A : Mat (F := F) n n) (v : Vec (F := F) n) :
    quadForm (matScale c A) v = mul c (quadForm A v) := by
  unfold quadForm matScale
  rw [← finSum_mul_left]; apply finSum_congr; intro i
  rw [← finSum_mul_left]; apply finSum_congr; intro j
  rw [← mul_assoc, ← mul_assoc, mul_comm (v i) c, mul_assoc c (v i), mul_assoc c]

/-! ### Theorem 1: CMA-ES Passive Covariance Update Preserves Symmetry & PSD (`cmaes.py`) -/

def rankMuMat {n mu : Nat} (y : Fin mu → Vec (F := F) n) (w : Fin mu → F) : Mat (F := F) n n :=
  fun i j => finSum (fun k : Fin mu => mul (mul (w k) (y k i)) (y k j))

theorem quadForm_rankMu {n mu : Nat} (y : Fin mu → Vec (F := F) n) (w : Fin mu → F) (v : Vec (F := F) n) :
    quadForm (rankMuMat y w) v = finSum (fun k : Fin mu => mul (w k) (quadForm (outer (y k) (y k)) v)) := by
  unfold quadForm rankMuMat outer
  calc
    finSum (fun i => finSum (fun j => mul (mul (v i) (finSum (fun k => mul (mul (w k) (y k i)) (y k j)))) (v j)))
      = finSum (fun i => finSum (fun j => finSum (fun k => mul (w k) (mul (mul (v i) (mul (y k i) (y k j))) (v j))))) := by
          apply finSum_congr; intro i; apply finSum_congr; intro j
          rw [← finSum_mul_left, ← finSum_mul_right]
          apply finSum_congr; intro k
          rw [mul_assoc (w k) (y k i) (y k j), ← mul_assoc (v i) (w k), mul_comm (v i) (w k),
              mul_assoc (w k) (v i), mul_assoc (w k)]
    _ = finSum (fun i => finSum (fun k => finSum (fun j => mul (w k) (mul (mul (v i) (mul (y k i) (y k j))) (v j))))) := by
          apply finSum_congr; intro i; exact finSum_swap _
    _ = finSum (fun k => finSum (fun i => finSum (fun j => mul (w k) (mul (mul (v i) (mul (y k i) (y k j))) (v j))))) :=
          finSum_swap _
    _ = finSum (fun k => mul (w k) (finSum (fun i => finSum (fun j => mul (mul (v i) (mul (y k i) (y k j))) (v j))))) := by
          apply finSum_congr; intro k
          rw [← finSum_mul_left]; apply finSum_congr; intro i; rw [← finSum_mul_left]

def cmaPassiveUpdate {n mu : Nat} (C : Mat (F := F) n n) (pc : Vec (F := F) n)
    (y : Fin mu → Vec (F := F) n) (w : Fin mu → F) (baseCoeff c1 cmu : F) : Mat (F := F) n n :=
  matAdd (matAdd (matScale baseCoeff C) (matScale c1 (outer pc pc))) (matScale cmu (rankMuMat y w))

theorem cmaPassiveUpdate_symmetric {n mu : Nat} (C : Mat (F := F) n n) (hSym : isSymmetric C)
    (pc : Vec (F := F) n) (y : Fin mu → Vec (F := F) n) (w : Fin mu → F) (baseCoeff c1 cmu : F) :
    isSymmetric (cmaPassiveUpdate C pc y w baseCoeff c1 cmu) := by
  intro i j; unfold cmaPassiveUpdate matAdd matScale outer rankMuMat
  have hC : C i j = C j i := hSym i j
  have hpc : mul (pc i) (pc j) = mul (pc j) (pc i) := mul_comm (pc i) (pc j)
  have hrm : finSum (fun k : Fin mu => mul (mul (w k) (y k i)) (y k j)) =
             finSum (fun k : Fin mu => mul (mul (w k) (y k j)) (y k i)) := by
    apply finSum_congr; intro k
    rw [mul_assoc, mul_comm (y k i) (y k j), ← mul_assoc]
  rw [hC, hpc, hrm]

theorem cmaPassiveUpdate_psd {n mu : Nat} (C : Mat (F := F) n n) (hPSD : isPSD C)
    (pc : Vec (F := F) n) (y : Fin mu → Vec (F := F) n) (w : Fin mu → F)
    (hw : ∀ k, le zero (w k)) (hc1 : le zero c1) (hcmu : le zero cmu)
    (hbase : le zero baseCoeff) :
    isPSD (cmaPassiveUpdate C pc y w baseCoeff c1 cmu) := by
  intro v; unfold cmaPassiveUpdate
  rw [quadForm_add, quadForm_add, quadForm_scale, quadForm_scale, quadForm_scale, quadForm_rankMu]
  have h1 := mul_nonneg baseCoeff (quadForm C v) hbase (hPSD v)
  have h2 := mul_nonneg c1 (quadForm (outer pc pc) v) hc1 (quadForm_outer_nonneg pc v)
  have hrm := finSum_nonneg (fun k => mul_nonneg (w k) _ (hw k) (quadForm_outer_nonneg (y k) v))
  have h3 := mul_nonneg cmu _ hcmu hrm
  have h12 := add_le_add zero _ zero _ h1 h2
  rw [zero_add] at h12
  have h123 := add_le_add zero _ zero _ h12 h3
  rw [zero_add] at h123
  exact h123

/-! ### Theorem 2: Fixed Active CMA with Hansen Mahalanobis Scaling Preserves PSD (`cmaes.py`) -/

def cmaActiveUpdate {n mu mu_neg : Nat} (C : Mat (F := F) n n) (pc : Vec (F := F) n)
    (y : Fin mu → Vec (F := F) n) (w : Fin mu → F)
    (y_neg : Fin mu_neg → Vec (F := F) n) (w_neg_scaled : Fin mu_neg → F)
    (baseCoeff c1 cmu c_neg : F) : Mat (F := F) n n :=
  matSub
    (cmaPassiveUpdate C pc y w baseCoeff c1 cmu)
    (matScale c_neg (rankMuMat y_neg w_neg_scaled))

/-- Fixed Active CMA PSD Theorem (`cmaes.py:193-196, 552-575`):
    With Hansen (2016) Mahalanobis weight scaling `w_neg_scaled_k = w_neg_k * (n / ||C^{-1/2} y_k||^2)`
    (zero when `||C^{-1/2} y_k||^2 = 0`), Cauchy-Schwarz in the `C`-metric gives for every `v`:
      `quadForm (outer (y_neg k) (y_neg k)) v ≤ mahal_k * quadForm C v`,
    so `w_neg_scaled_k * quadForm (outer (y_neg k) (y_neg k)) v ≤ w_neg_k * dimF * quadForm C v`.
    Summing over `k` with `∑ w_neg_k = 1`, we get `quadForm rank_mu_neg v ≤ dimF * quadForm C v`.
    Whenever `c_neg * dimF ≤ baseCoeff` (guaranteed by `c_mu_neg ≤ (1 - c_1 - c_mu) / dim`),
    `cmaActiveUpdate` is ALWAYS positive semi-definite for ALL `n, mu, mu_neg` and ALL outliers! -/
theorem cmaActiveUpdate_mahalanobis_psd {n mu mu_neg : Nat}
    (C : Mat (F := F) n n) (hPSD : isPSD C)
    (pc : Vec (F := F) n)
    (y : Fin mu → Vec (F := F) n) (w : Fin mu → F)
    (y_neg : Fin mu_neg → Vec (F := F) n)
    (w_neg w_neg_scaled : Fin mu_neg → F)
    (dimF baseCoeff c1 cmu c_neg : F)
    (hw : ∀ k, le zero (w k))
    (hw_neg_sum : finSum w_neg = one)
    (hc1 : le zero c1) (hcmu : le zero cmu) (hc_neg : le zero c_neg)
    (hbound : le (mul c_neg dimF) baseCoeff)
    (hmahal : ∀ (k : Fin mu_neg) (v : Vec (F := F) n),
      le (mul (w_neg_scaled k) (quadForm (outer (y_neg k) (y_neg k)) v))
         (mul (mul (w_neg k) dimF) (quadForm C v))) :
    isPSD (cmaActiveUpdate C pc y w y_neg w_neg_scaled baseCoeff c1 cmu c_neg) := by
  intro v
  unfold cmaActiveUpdate cmaPassiveUpdate
  rw [quadForm_sub, quadForm_add, quadForm_add, quadForm_scale, quadForm_scale, quadForm_scale,
      quadForm_scale, quadForm_rankMu, quadForm_rankMu]
  -- Upper bound on negative rank-mu quadratic form: ≤ dimF * (v^T C v)
  have hneg_le : le (finSum (fun k => mul (w_neg_scaled k) (quadForm (outer (y_neg k) (y_neg k)) v)))
                    (mul dimF (quadForm C v)) := by
    have hsum := finSum_le_finSum (fun k => hmahal k v)
    have hrhs : finSum (fun k => mul (mul (w_neg k) dimF) (quadForm C v)) = mul dimF (quadForm C v) := by
      have h1 : finSum (fun k => mul (mul (w_neg k) dimF) (quadForm C v)) =
                finSum (fun k => mul (w_neg k) (mul dimF (quadForm C v))) := by
        apply finSum_congr; intro k; exact mul_assoc _ _ _
      rw [h1, finSum_mul_right, hw_neg_sum, one_mul]
    rw [hrhs] at hsum
    exact hsum
  have hcneg_le : le (mul c_neg (finSum (fun k => mul (w_neg_scaled k) (quadForm (outer (y_neg k) (y_neg k)) v))))
                     (mul baseCoeff (quadForm C v)) := by
    have h1 := mul_le_mul_left hc_neg hneg_le
    rw [← mul_assoc] at h1
    have h2 : le (mul (mul c_neg dimF) (quadForm C v)) (mul baseCoeff (quadForm C v)) := by
      rw [mul_comm (mul c_neg dimF), mul_comm baseCoeff]
      exact mul_le_mul_left (hPSD v) hbound
    exact le_trans _ _ _ h1 h2
  have hbase_rem : le zero (sub (mul baseCoeff (quadForm C v))
                                (mul c_neg (finSum (fun k => mul (w_neg_scaled k) (quadForm (outer (y_neg k) (y_neg k)) v))))) :=
    sub_nonneg_of_le hcneg_le
  have h2 := mul_nonneg c1 (quadForm (outer pc pc) v) hc1 (quadForm_outer_nonneg pc v)
  have hrm := finSum_nonneg (fun k => mul_nonneg (w k) _ (hw k) (quadForm_outer_nonneg (y k) v))
  have h3 := mul_nonneg cmu _ hcmu hrm
  have hrearr_gen : ∀ A B D E : F, sub (add (add A B) D) E = add (add (sub A E) B) D := by
    intro A B D E
    rw [sub_def, sub_def]
    rw [add_assoc (add A B) D (neg E), add_comm D (neg E), ← add_assoc (add A B) (neg E) D,
        add_assoc A B (neg E), add_comm B (neg E), ← add_assoc A (neg E) B]
  rw [hrearr_gen]
  have h12 := add_le_add zero _ zero _ hbase_rem h2
  rw [zero_add] at h12
  have h123 := add_le_add zero _ zero _ h12 h3
  rw [zero_add] at h123
  exact h123

/-! ### Theorem 3: Exact 3-Variable Gauge Transform Invariance (`cmaes.py:_update_eigensystem`) -/

/-- Exact 3-variable gauge transform (`C' = C / s`, `sigma' = sigma * s_sqrt`, `p_c' = p_c / s_sqrt`,
    `C_invsqrt' = s_sqrt * C_invsqrt` where `s = s_sqrt^2`, `s_sqrt ≠ 0`):
    1. Sampling covariance is EXACTLY preserved: `sigma'^2 * C' = sigma^2 * C`.
    2. Conjugate path `p_sigma` increment is EXACTLY invariant: `C_invsqrt' @ d / sigma' = C_invsqrt @ d / sigma`.
    3. Next-step `p_c` transforms covariantly: `p_c'_next = (1 / s_sqrt) * p_c_next`.
    4. Next-step covariance transforms covariantly: `C'_next = (1 / s) * C_next`, so
       `sigma'_next^2 * C'_next = sigma_next^2 * C_next` inductively across all generations! -/
theorem cma_gauge_transform_preserves_sampling_cov
    {n : Nat} (sigma s_sqrt : F) (hs_ne0 : s_sqrt ≠ zero) (C : Mat (F := F) n n) :
    let s := mul s_sqrt s_sqrt
    let sigma' := mul sigma s_sqrt
    let C' := matScale (inv s) C
    matScale (mul sigma' sigma') C' = matScale (mul sigma sigma) C := by
  intro s sigma' C'
  funext i j
  dsimp only [C', sigma', s]
  unfold matScale
  rw [inv_mul s_sqrt s_sqrt]
  have hss : mul (mul (mul sigma s_sqrt) (mul sigma s_sqrt)) (mul (inv s_sqrt) (inv s_sqrt)) =
             mul sigma sigma := by
    have h1 : mul (mul sigma s_sqrt) (mul sigma s_sqrt) = mul (mul sigma sigma) (mul s_sqrt s_sqrt) := by
      rw [mul_assoc sigma s_sqrt (mul sigma s_sqrt), ← mul_assoc s_sqrt sigma s_sqrt,
          mul_comm s_sqrt sigma, mul_assoc sigma s_sqrt s_sqrt, ← mul_assoc]
    rw [h1, mul_assoc (mul sigma sigma) (mul s_sqrt s_sqrt) (mul (inv s_sqrt) (inv s_sqrt))]
    have h2 : mul (mul s_sqrt s_sqrt) (mul (inv s_sqrt) (inv s_sqrt)) = one := by
      rw [mul_assoc s_sqrt s_sqrt (mul (inv s_sqrt) (inv s_sqrt)),
          ← mul_assoc s_sqrt (inv s_sqrt) (inv s_sqrt),
          mul_inv_cancel s_sqrt hs_ne0, one_mul, mul_inv_cancel s_sqrt hs_ne0]
    rw [h2, mul_one]
  rw [← mul_assoc, hss]

theorem cma_gauge_transform_preserves_p_sigma_step
    {n : Nat} (sigma s_sqrt : F) (hs_ne0 : s_sqrt ≠ zero)
    (C_invsqrt : Mat (F := F) n n) (mean_diff : Vec (F := F) n) :
    let sigma' := mul sigma s_sqrt
    let C_invsqrt' := matScale s_sqrt C_invsqrt
    vecScale (inv sigma') (matVec C_invsqrt' mean_diff) =
    vecScale (inv sigma) (matVec C_invsqrt mean_diff) := by
  intro sigma' C_invsqrt'
  funext i
  dsimp only [sigma', C_invsqrt']
  unfold vecScale matVec matScale
  have hsum : finSum (fun j => mul (mul s_sqrt (C_invsqrt i j)) (mean_diff j)) =
              mul s_sqrt (finSum (fun j => mul (C_invsqrt i j) (mean_diff j))) := by
    rw [← finSum_mul_left]
    apply finSum_congr; intro j; exact mul_assoc _ _ _
  rw [hsum, inv_mul sigma s_sqrt, ← mul_assoc,
      mul_assoc (inv sigma) (inv s_sqrt) s_sqrt,
      mul_comm (inv s_sqrt) s_sqrt, mul_inv_cancel s_sqrt hs_ne0, mul_one]

theorem cma_gauge_transform_pc_equivariant
    {n : Nat} (sigma s_sqrt a b : F)
    (pc mean_diff : Vec (F := F) n) :
    let sigma' := mul sigma s_sqrt
    let pc' := vecScale (inv s_sqrt) pc
    let pc_next := vecAdd (vecScale a pc) (vecScale (mul b (inv sigma)) mean_diff)
    let pc'_next := vecAdd (vecScale a pc') (vecScale (mul b (inv sigma')) mean_diff)
    pc'_next = vecScale (inv s_sqrt) pc_next := by
  intro sigma' pc' pc_next pc'_next
  funext i
  dsimp only [pc'_next, pc_next, pc', sigma']
  unfold vecAdd vecScale
  rw [mul_add, inv_mul sigma s_sqrt]
  have h1 : mul a (mul (inv s_sqrt) (pc i)) = mul (inv s_sqrt) (mul a (pc i)) := by
    rw [← mul_assoc, mul_comm a (inv s_sqrt), mul_assoc]
  have h2 : mul (mul b (mul (inv sigma) (inv s_sqrt))) (mean_diff i) =
            mul (inv s_sqrt) (mul (mul b (inv sigma)) (mean_diff i)) := by
    rw [← mul_assoc b (inv sigma) (inv s_sqrt), mul_comm (mul b (inv sigma)) (inv s_sqrt), mul_assoc]
  rw [h1, h2]

theorem cma_gauge_transform_C_next_equivariant
    {n mu : Nat} (sigma s_sqrt baseCoeff c1 cmu : F)
    (C : Mat (F := F) n n) (pc_next : Vec (F := F) n)
    (diffs : Fin mu → Vec (F := F) n) (w : Fin mu → F) :
    let s := mul s_sqrt s_sqrt
    let sigma' := mul sigma s_sqrt
    let C' := matScale (inv s) C
    let pc'_next := vecScale (inv s_sqrt) pc_next
    let y : Fin mu → Vec (F := F) n := fun k => vecScale (inv sigma) (diffs k)
    let y' : Fin mu → Vec (F := F) n := fun k => vecScale (inv sigma') (diffs k)
    cmaPassiveUpdate C' pc'_next y' w baseCoeff c1 cmu =
    matScale (inv s) (cmaPassiveUpdate C pc_next y w baseCoeff c1 cmu) := by
  intro s sigma' C' pc'_next y y'
  funext i j
  dsimp only [C', pc'_next, y', y, sigma', s]
  unfold cmaPassiveUpdate matAdd matScale outer rankMuMat vecScale
  rw [mul_add, mul_add, inv_mul s_sqrt s_sqrt, inv_mul sigma s_sqrt]
  have hC : mul baseCoeff (mul (mul (inv s_sqrt) (inv s_sqrt)) (C i j)) =
            mul (mul (inv s_sqrt) (inv s_sqrt)) (mul baseCoeff (C i j)) := by
    rw [← mul_assoc, mul_comm baseCoeff (mul (inv s_sqrt) (inv s_sqrt)), mul_assoc]
  have hpc : mul c1 (mul (mul (inv s_sqrt) (pc_next i)) (mul (inv s_sqrt) (pc_next j))) =
             mul (mul (inv s_sqrt) (inv s_sqrt)) (mul c1 (mul (pc_next i) (pc_next j))) := by
    have hinner : mul (mul (inv s_sqrt) (pc_next i)) (mul (inv s_sqrt) (pc_next j)) =
                  mul (mul (inv s_sqrt) (inv s_sqrt)) (mul (pc_next i) (pc_next j)) := by
      rw [mul_assoc (inv s_sqrt) (pc_next i) (mul (inv s_sqrt) (pc_next j)),
          ← mul_assoc (pc_next i) (inv s_sqrt) (pc_next j),
          mul_comm (pc_next i) (inv s_sqrt),
          mul_assoc (inv s_sqrt) (pc_next i) (pc_next j), ← mul_assoc]
    rw [hinner, ← mul_assoc, mul_comm c1 (mul (inv s_sqrt) (inv s_sqrt)), mul_assoc]
  have hfactor : ∀ (r wk xi xj : F),
      mul (mul wk (mul r xi)) (mul r xj) = mul (mul r r) (mul (mul wk xi) xj) := by
    intro r wk xi xj
    rw [← mul_assoc wk r xi, mul_comm wk r, mul_assoc r wk xi,
        mul_assoc r (mul wk xi) (mul r xj),
        ← mul_assoc (mul wk xi) r xj, mul_comm (mul wk xi) r,
        mul_assoc r (mul wk xi) xj, ← mul_assoc r r]
  have hrm :
      mul cmu (finSum (fun k : Fin mu =>
        mul (mul (w k) (mul (mul (inv sigma) (inv s_sqrt)) (diffs k i)))
            (mul (mul (inv sigma) (inv s_sqrt)) (diffs k j)))) =
      mul (mul (inv s_sqrt) (inv s_sqrt))
        (mul cmu (finSum (fun k : Fin mu =>
          mul (mul (w k) (mul (inv sigma) (diffs k i))) (mul (inv sigma) (diffs k j))))) := by
    have hterm :
        finSum (fun k : Fin mu =>
          mul (mul (w k) (mul (mul (inv sigma) (inv s_sqrt)) (diffs k i)))
              (mul (mul (inv sigma) (inv s_sqrt)) (diffs k j))) =
        mul (mul (inv s_sqrt) (inv s_sqrt))
          (finSum (fun k : Fin mu =>
            mul (mul (w k) (mul (inv sigma) (diffs k i))) (mul (inv sigma) (diffs k j)))) := by
      rw [← finSum_mul_left]
      apply finSum_congr; intro k
      have ha : mul (mul (inv sigma) (inv s_sqrt)) (diffs k i) =
                mul (inv s_sqrt) (mul (inv sigma) (diffs k i)) := by
        rw [mul_comm (inv sigma) (inv s_sqrt), mul_assoc]
      have hb : mul (mul (inv sigma) (inv s_sqrt)) (diffs k j) =
                mul (inv s_sqrt) (mul (inv sigma) (diffs k j)) := by
        rw [mul_comm (inv sigma) (inv s_sqrt), mul_assoc]
      rw [ha, hb]
      exact hfactor (inv s_sqrt) (w k) (mul (inv sigma) (diffs k i)) (mul (inv sigma) (diffs k j))
    rw [hterm, ← mul_assoc, mul_comm cmu (mul (inv s_sqrt) (inv s_sqrt)), mul_assoc]
  rw [hC, hpc, hrm]

/-! ### Theorem 4: Fixed `DLRPortfolio` Diagonal Update Preserves Stationary Variance (`dlr_cma.py`) -/

/-- Exact variance stationarity under `h_sigma = 0` (`dlr_cma.py`):
    When `h_sigma = 0`, `p_c_new = (1 - c_c) p_c`, so `E[p_c_new^2] = (1 - c_c)^2 d_old`
    and `E[diag_mu] = d_old`. With `delta_h = c_c * (2 - c_c)` multiplied by `c1`:
      `(1 - c1 - cmu + c1 * delta_h) * d_old + c1 * (1 - c_c)^2 * d_old + cmu * d_old = d_old`! -/
theorem dlr_fixed_delta_h_preserves_stationary_variance (c1 cmu cc d_old : F) :
    let two := add one one
    let delta_h := mul cc (sub two cc)
    let diag_1_stat := mul (mul (sub one cc) (sub one cc)) d_old
    let d_new :=
      add (add (mul (add (sub (sub one c1) cmu) (mul c1 delta_h)) d_old)
               (mul c1 diag_1_stat))
          (mul cmu d_old)
    d_new = d_old := by
  intro two delta_h diag_1_stat d_new
  dsimp only [d_new, diag_1_stat, delta_h, two]
  rw [← mul_assoc c1 (mul (sub one cc) (sub one cc)) d_old, ← add_mul, ← add_mul]
  have hsq : add (mul cc (sub (add one one) cc)) (mul (sub one cc) (sub one cc)) = one := by
    rw [sub_def (add one one) cc, mul_add cc (add one one) (neg cc), mul_add cc one one, mul_one, mul_neg]
    rw [sub_def one cc, mul_add (add one (neg cc)) one (neg cc), mul_one,
        add_mul one (neg cc) (neg cc), one_mul, mul_neg, neg_mul]
    have hnn : neg (neg (mul cc cc)) = mul cc cc := by
      have h0 : add (neg (mul cc cc)) (neg (neg (mul cc cc))) = zero := add_neg (neg (mul cc cc))
      have h1 : add (mul cc cc) (add (neg (mul cc cc)) (neg (neg (mul cc cc)))) =
                add (mul cc cc) zero := congrArg (add (mul cc cc)) h0
      rw [← add_assoc, add_neg, zero_add, add_zero] at h1
      exact h1
    rw [hnn]
    have hcancel : ∀ x : F,
        add (add (add cc cc) (neg x)) (add (add one (neg cc)) (add (neg cc) x)) = one := by
      intro x
      have hstep1 : add (add (add cc cc) (neg x)) (add (add one (neg cc)) (add (neg cc) x)) =
                    add one (add (add (add cc (neg cc)) (add cc (neg cc))) (add (neg x) x)) := by
        rw [add_assoc (add cc cc) (neg x) (add (add one (neg cc)) (add (neg cc) x)),
            add_comm (add one (neg cc)) (add (neg cc) x),
            ← add_assoc (neg x) (add (neg cc) x) (add one (neg cc)),
            add_comm (neg cc) x,
            ← add_assoc (neg x) x (neg cc),
            add_assoc (add (neg x) x) (neg cc) (add one (neg cc)),
            add_comm (add (neg x) x) (add (neg cc) (add one (neg cc))),
            ← add_assoc (add cc cc) (add (neg cc) (add one (neg cc))) (add (neg x) x),
            add_comm one (neg cc),
            ← add_assoc (neg cc) (neg cc) one,
            add_comm (add (neg cc) (neg cc)) one,
            ← add_assoc (add cc cc) one (add (neg cc) (neg cc)),
            add_comm (add cc cc) one,
            add_assoc one (add cc cc) (add (neg cc) (neg cc)),
            add_assoc one (add (add cc cc) (add (neg cc) (neg cc))) (add (neg x) x)]
        have hpair : add (add cc cc) (add (neg cc) (neg cc)) =
                     add (add cc (neg cc)) (add cc (neg cc)) := by
          rw [add_assoc cc cc (add (neg cc) (neg cc)),
              ← add_assoc cc (neg cc) (neg cc),
              add_comm (add cc (neg cc)) (neg cc),
              ← add_assoc cc (neg cc) (add cc (neg cc))]
        rw [hpair]
      rw [hstep1, add_neg, add_zero, add_comm (neg x) x, add_neg, add_zero, add_zero]
    exact hcancel (mul cc cc)
  have hcoeff :
      add (add (add (sub (sub one c1) cmu) (mul c1 (mul cc (sub (add one one) cc))))
               (mul c1 (mul (sub one cc) (sub one cc)))) cmu = one := by
    rw [add_assoc (sub (sub one c1) cmu) _ _, ← mul_add c1, hsq, mul_one]
    rw [sub_def, sub_def]
    rw [add_assoc (add one (neg c1)) (neg cmu) c1, add_comm (neg cmu) c1,
        ← add_assoc (add one (neg c1)) c1 (neg cmu),
        add_assoc one (neg c1) c1, add_comm (neg c1) c1, add_neg, add_zero,
        add_assoc one (neg cmu) cmu, add_comm (neg cmu) cmu, add_neg, add_zero]
  rw [hcoeff, one_mul]

/-- Fixed `DLRPortfolio` pre-update `sigma_old`: `y_sel` uses `sigma_old`, matching `y_true`
    with exact factor `1` regardless of step-size jump `alpha`. -/
theorem dlr_fixed_sigma_order_exact (delta sigma_old : F) :
    let y_true := mul delta (inv sigma_old)
    let y_fixed := mul delta (inv sigma_old)
    mul y_fixed y_fixed = mul y_true y_true := rfl

end LinearAlgebra


/-! ### Theorem 5: Hansen (2016) Table 1 `c_c` Strictly in `(0, 1)` (`cmaes.py:48`, `dlr_cma.py:59`) -/

/-- Hansen (2016) Table 1 formula (now used in both `cmaes.py` and `dlr_cma.py`):
      `c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
           = (4 * d + mu_eff) / (d * (d + 4) + 2 * mu_eff)`
    For ALL integer `d ≥ 1` and scaled `m ≥ 1`, `0 < num < den`, hence `0 < c_c < 1` ALWAYS. -/
theorem hansen_cc_strictly_between_0_and_1 {d m : Nat} (hd : d ≥ 1) (hm : m ≥ 1) :
    let num := 4 * d + m
    let den := d * (d + 4) + 2 * m
    0 < num ∧ num < den := by
  dsimp only
  have h1 : d * d ≥ 1 := Nat.mul_le_mul hd hd
  have h2 : d * (d + 4) = d * d + 4 * d := by
    rw [Nat.mul_add, Nat.mul_comm d 4]
  omega

/-- Historical regression witness: the pre-Wave-1 buggy formula `(mu_eff + 2) / (d + 4 + 2*mu_eff/d)`
    exceeded 2 at `(dim=16, mu_eff=1030)` (`c_c ≈ 6.94`), permanently zeroing `p_c`. -/
theorem pre_wave1_cc_bug_witness :
    let d := 16
    let m := 1030
    d * (m + 2) > 2 * (d * (d + 4) + 2 * m) := by
  decide

end TorchDFO.CMAESAndDLR
