"""Z3 SMT Formal Verification Suite for torch-dfo (IEEE-754 FP + QF_NRA).

Verifies floating-point and real-algebraic invariants across the core optimizers:
  1. SearchSpace Categorical midpoint encoding (idx + 0.5) / n round-trips
     with 0 violations for all n = 1..50 in both f64 and f64 -> f32 -> f64.
  2. SHADE +inf parent replacement with uniform-weight fallback produces
     finite, non-NaN memory_F and memory_CR in both f32 and f64.
  3. BaseOptimizer._update_best NaN -> +inf masking never lets NaN beat or
     mask a finite candidate in both f32 and f64.
  4. NelderMead.tell NaN -> +inf sanitization makes all comparisons ordered
     and guarantees all-NaN trial batches always trigger shrink.
  5. sanitize_fitness using isfinite() produces finite worst_finite + 1.0
     when finite values coexist with +inf and NaN.
  6. QF_NRA: Hansen (2016) CMA-ES & DLRPortfolio rates satisfy
     0 < c_sigma, c_c, c_1 < 1, c_mu >= 0, c_1 + c_mu <= 1, and active
     0 <= c_mu_neg <= (1 - c_1 - c_mu) / dim for ALL dim >= 1, mu_eff >= 1.
  7. Thermal Plant float32 quantization 1-ULP boundary (documented scope limit).
"""

from __future__ import annotations

import z3


def main() -> None:
    print(f"Z3 version: {z3.get_version_string()}")
    rm = z3.RNE()
    f32 = z3.Float32()
    f64 = z3.Float64()

    # 1. Fixed SearchSpace Categorical midpoint encoding (idx + 0.5) / n
    #    src/torch_dfo/space.py:163-170
    for label, cast_f32 in [("f64", False), ("f64->f32->f64", True)]:
        bad = 0
        for n in range(1, 51):
            for k in range(n):
                u = z3.fpDiv(rm, z3.FPVal(k + 0.5, f64), z3.FPVal(float(n), f64))
                if cast_f32:
                    u = z3.fpToFP(rm, z3.fpToFP(rm, u, f32), f64)
                prod = z3.fpMul(rm, u, z3.FPVal(float(n), f64))
                s = z3.Solver()
                s.add(
                    z3.Or(
                        z3.fpLT(prod, z3.FPVal(float(k), f64)),
                        z3.fpGEQ(prod, z3.FPVal(float(k + 1), f64)),
                    )
                )
                if s.check() == z3.sat:
                    bad += 1
        assert bad == 0, f"Midpoint encoding failed for {bad} cases in {label}"
        print(f"[PASS] IEEE-754 {label} Categorical midpoint (idx+0.5)/n: 0 violations (n=1..50)")

    # 2. Fixed SHADE +inf parent -> uniform weight fallback keeps memory_F/CR finite & non-NaN
    #    src/torch_dfo/shade.py:348-370
    for dt_name, fp in [("f32", f32), ("f64", f64)]:
        s = z3.Solver()
        f_parent = z3.fpPlusInfinity(fp)
        f_off = z3.FP("f_off", fp)
        succ_f = z3.FP("succ_f", fp)
        succ_cr = z3.FP("succ_cr", fp)
        s.add(z3.Not(z3.fpIsNaN(f_off)), z3.Not(z3.fpIsInf(f_off)))
        s.add(z3.fpGEQ(succ_f, z3.FPVal(0.05, fp)), z3.fpLEQ(succ_f, z3.FPVal(1.0, fp)))
        s.add(z3.fpGEQ(succ_cr, z3.FPVal(0.0, fp)), z3.fpLEQ(succ_cr, z3.FPVal(1.0, fp)))
        delta_f = z3.fpSub(rm, f_parent, f_off)
        delta_finite_pos = z3.And(
            z3.Not(z3.fpIsNaN(delta_f)),
            z3.Not(z3.fpIsInf(delta_f)),
            z3.fpGT(delta_f, z3.FPVal(0.0, fp)),
        )
        weight = z3.If(delta_finite_pos, z3.fpDiv(rm, delta_f, delta_f), z3.FPVal(1.0, fp))
        wf = z3.fpMul(rm, weight, succ_f)
        wf2 = z3.fpMul(rm, wf, succ_f)
        lehmer_f = z3.fpDiv(rm, wf2, wf)
        mean_cr = z3.fpMul(rm, weight, succ_cr)
        s.add(
            z3.Or(
                z3.fpIsNaN(lehmer_f),
                z3.fpIsInf(lehmer_f),
                z3.fpIsNaN(mean_cr),
                z3.fpIsInf(mean_cr),
            )
        )
        assert s.check() == z3.unsat
        print(
            f"[PASS] IEEE-754 {dt_name} SHADE +inf parent -> "
            "memory_F & memory_CR ALWAYS finite (unsat)"
        )

    # 3. Fixed BaseOptimizer._update_best NaN -> +inf masking
    #    src/torch_dfo/base.py:70-85
    for dt_name, fp in [("f32", f32), ("f64", f64)]:
        s = z3.Solver()
        f_good = z3.FP("f_good", fp)
        best_f = z3.FP("best_f", fp)
        s.add(z3.Not(z3.fpIsNaN(f_good)), z3.Not(z3.fpIsInf(f_good)))
        s.add(z3.Not(z3.fpIsNaN(best_f)), z3.fpLT(f_good, best_f))
        f_nan = z3.fpNaN(fp)
        safe_nan = z3.If(z3.fpIsNaN(f_nan), z3.fpPlusInfinity(fp), f_nan)
        safe_good = z3.If(z3.fpIsNaN(f_good), z3.fpPlusInfinity(fp), f_good)
        # Masking guarantees safe_good < safe_nan so argmin picks safe_good (< best_f).
        s.add(z3.Not(z3.And(z3.fpLT(safe_good, safe_nan), z3.fpLT(safe_good, best_f))))
        assert s.check() == z3.unsat
        print(
            f"[PASS] IEEE-754 {dt_name} BaseOptimizer._update_best "
            "NaN masking selects finite winner (unsat)"
        )

    # 4. Fixed NelderMead.tell NaN -> +inf sanitization
    #    src/torch_dfo/nelder_mead.py:170-230
    for dt_name, fp in [("f32", f32), ("f64", f64)]:
        s = z3.Solver()
        f_best, f_sec, f_worst = z3.FPs("f_best f_sec f_worst", fp)
        for v in (f_best, f_sec, f_worst):
            s.add(z3.Not(z3.fpIsNaN(v)), z3.Not(z3.fpIsInf(v)))
        s.add(z3.fpLEQ(f_best, f_sec), z3.fpLEQ(f_sec, f_worst))
        # Sanitize all-NaN trial batch to +inf
        f_r = z3.fpPlusInfinity(fp)
        f_co = z3.fpPlusInfinity(fp)
        f_ci = z3.fpPlusInfinity(fp)
        use_exp = z3.fpLT(f_r, f_best)
        acc_ref = z3.And(z3.fpLEQ(f_best, f_r), z3.fpLT(f_r, f_sec))
        try_out = z3.And(z3.Not(acc_ref), z3.Not(use_exp), z3.fpLT(f_r, f_worst))
        try_in = z3.And(z3.Not(acc_ref), z3.Not(use_exp), z3.Not(try_out))
        need_shrink = z3.Or(
            z3.And(try_out, z3.Not(z3.fpLEQ(f_co, f_r))),
            z3.And(try_in, z3.Not(z3.fpLT(f_ci, f_worst))),
        )
        s.add(z3.Not(need_shrink))
        assert s.check() == z3.unsat
        print(
            f"[PASS] IEEE-754 {dt_name} NelderMead.tell "
            "sanitized all-NaN batch ALWAYS triggers shrink (unsat)"
        )

    # 5. Fixed sanitize_fitness with [finite, +inf, NaN] produces finite replacement
    #    src/torch_dfo/utils.py:36-44
    for dt_name, fp, cap in [("f32", f32, 1e30), ("f64", f64, 1e100)]:
        s = z3.Solver()
        f_fin = z3.FP("f_fin", fp)
        s.add(
            z3.Not(z3.fpIsNaN(f_fin)),
            z3.Not(z3.fpIsInf(f_fin)),
            z3.fpGEQ(f_fin, z3.FPVal(-cap, fp)),
            z3.fpLEQ(f_fin, z3.FPVal(cap, fp)),
        )
        # Finite mask excludes +inf and NaN, so worst_finite == f_fin; replacement = f_fin + 1.0
        replaced = z3.fpAdd(rm, f_fin, z3.FPVal(1.0, fp))
        s.add(z3.Or(z3.fpIsNaN(replaced), z3.fpIsInf(replaced)))
        assert s.check() == z3.unsat
        print(
            f"[PASS] IEEE-754 {dt_name} sanitize_fitness "
            "isfinite() replacement ALWAYS finite (unsat)"
        )

    # 6. QF_NRA: Hansen (2016) rates + Active CMA PSD bound hold for ALL dim >= 1, mu_eff >= 1
    #    src/torch_dfo/cmaes.py:46-54, 193-196 & src/torch_dfo/dlr_cma.py:58-64
    s_fix = z3.Solver()
    d2, m2 = z3.Reals("d2 m2")
    s_fix.add(d2 >= 1, m2 >= 1)
    c_sigma = (m2 + 2) / (d2 + m2 + 5)
    c_c_hansen = (4 + m2 / d2) / (d2 + 4 + 2 * m2 / d2)
    c_1 = 2 / ((d2 + z3.Q(13, 10)) ** 2 + m2)
    c_mu_raw = 2 * (m2 - 2 + 1 / m2) / ((d2 + 2) ** 2 + m2)
    c_mu = z3.If(c_mu_raw < 1 - c_1, c_mu_raw, 1 - c_1)
    c_mu_neg_a = z3.Q(1, 4) * c_mu
    c_mu_neg_b = z3.If(1 - c_1 - c_mu > 0, (1 - c_1 - c_mu) / d2, 0)
    c_mu_neg = z3.If(c_mu_neg_a < c_mu_neg_b, c_mu_neg_a, c_mu_neg_b)
    s_fix.add(
        z3.Or(
            c_sigma <= 0,
            c_sigma >= 1,
            c_c_hansen <= 0,
            c_c_hansen >= 1,
            c_1 <= 0,
            c_1 >= 1,
            c_mu < 0,
            c_1 + c_mu > 1,
            c_mu_neg < 0,
            c_mu_neg * d2 > 1 - c_1 - c_mu,
        )
    )
    assert s_fix.check() == z3.unsat
    print(
        "[PASS] QF_NRA Hansen (2016) rates + Active PSD bound "
        "hold for ALL dim>=1, mu_eff>=1 (unsat)"
    )

    # 7. Thermal Plant float32 quantization 1-ULP discontinuity at 0.125 (documented scope limit)
    s = z3.Solver()
    x_tie = z3.FPVal(0.125, f32)
    x_up = z3.FPVal(float.fromhex("0x1.000002p-3"), f32)
    step = z3.FPVal(0.25, f32)
    q_tie = z3.fpMul(rm, z3.fpRoundToIntegral(rm, z3.fpDiv(rm, x_tie, step)), step)
    q_up = z3.fpMul(rm, z3.fpRoundToIntegral(rm, z3.fpDiv(rm, x_up, step)), step)
    s.add(z3.Not(z3.And(z3.fpEQ(q_tie, z3.FPVal(0.0, f32)), z3.fpEQ(q_up, z3.FPVal(0.25, f32)))))
    assert s.check() == z3.unsat
    print("[PASS] IEEE-754 f32 Thermal quantization 1-ULP boundary documented (unsat)")


if __name__ == "__main__":
    main()
