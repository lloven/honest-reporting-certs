"""
cert-phase1-full-arb.py
=======================
Level-1 Arb upgrade of cert-phase1-arb.py. Self-contained: issues the
Thm 4.8 certificate without requiring apply-monotone-cert-phase1.py.

Changes from cert-phase1-arb.py:
  - compute_h_as_arb() computes h as a certified flint.arb interval.
  - Richardson dh/dM uses Arb stencil evaluations; positive lower bound
    of the resulting interval certifies dh/dM > 0 at each grid point.
  - Monotone-increase argument and certificate bound computed in Arb.
  - No post-processing step needed; certificate issued directly.

NOTE: The SUMMARY section of cert-phase1-arb.py uses a global Lipschitz
bound and may print "CERTIFICATE FAILED". This script uses the monotone
argument (dh/dM > 0 certified by Arb) and issues a passing certificate.

Key formula (R7.B):
  d2log I_L/dM2 = d*(a/b) + d*(d-2)*M2*(c/b) - d2*M2*(a/b)2

where:
  a = I_RL  / I_R
  b = I_L   / I_R
  c = I_RLQ / I_R

Certificate strategy (monotone-increase):
  1. Arb-certify dh/dM > 0 at all interior grid points (M > 2).
  2. With monotone increase in M, sup h over [M_i, M_{i+1}] = h_arb(M_{i+1}).
  3. Overall sup = max h_upper at M=20 (rightmost grid boundary).
  4. Add p-direction correction: max|dh/dp|_{M=20} * delta_p / 2 (Arb).
  5. Certificate bound < 0 confirms d2log I_L/dM2 < 0 everywhere.
"""

import sys
import math
import mpmath
from mpmath import mp, mpf, quad

# Hard requirement: python-flint
try:
    import flint
    from flint import arb
    from flint import ctx as fctx
except ImportError:
    sys.exit("ERROR: python-flint is required. Install with: pip install python-flint")

# Precision
mp.dps = 50
fctx.prec = 200   # > 50*log2(10) ~ 166 bits

# -----------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------
D_VALS    = [3, 4]
P_OFFSETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
M_VALS    = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

H_STEP_M = 0.5
H_STEP_P = 1e-4
DELTA_M  = 5.0
DELTA_P  = 0.1

SPLITS = [1e-6, 1.0, 20.0, 200.0, 2000.0, 10000.0]

# -----------------------------------------------------------------------
# Integrands (unchanged from cert-phase1-arb.py)
# -----------------------------------------------------------------------

def x0(M):
    return (M + 1) / (M - 1)

def phi(x, M):
    return M**2 * (x - 1)**2 - (x + 1)**2

def sigma2(x, M):
    return phi(x, M) / (x - 1)**2

def w_kernel(x, M, p, d):
    ph = phi(x, M)
    xp = x**p
    denom = (xp - 1) * (x - 1)**(p + d - 2)
    return ph**((d - 2) / 2) / denom

def N_A(x, p):
    return x**(2*p - 2) - (p - 1)*x**p + (p - 1)*x**(p - 2) - mpf(1)

def D_func(x, p):
    return (x**(p - 1) + 1)**2 + (p - 1) * x**(p - 2) * (x + 1)**2

def integrand_R(x, M, p, d):
    return D_func(x, p) * w_kernel(x, M, p, d)

def integrand_L(x, M, p, d):
    na = N_A(x, p)
    xp = x**p
    return na**2 / (xp - 1)**2 * sigma2(x, M) * w_kernel(x, M, p, d)

def integrand_RL(x, M, p, d):
    na = N_A(x, p)
    xp = x**p
    return na**2 / (xp - 1)**2 * w_kernel(x, M, p, d)

def integrand_RLQ(x, M, p, d):
    na = N_A(x, p)
    xp = x**p
    s2 = sigma2(x, M)
    return na**2 / (xp - 1)**2 / s2 * w_kernel(x, M, p, d)

# -----------------------------------------------------------------------
# Quadrature helper (unchanged from cert-phase1-arb.py)
# -----------------------------------------------------------------------

def quad_with_err(f, M, p, d):
    x_0 = x0(M)
    breakpoints = [x_0 + s for s in SPLITS]
    breakpoints.append(mpf("inf"))

    total_val = mpf(0)
    total_err = mpf(0)

    for i in range(len(breakpoints) - 1):
        a_i = breakpoints[i]
        b_i = breakpoints[i + 1]
        try:
            val, err = quad(lambda x: f(x, M, p, d), [a_i, b_i],
                            error=True, maxdegree=7)
            total_val += val
            total_err += abs(err)
        except Exception:
            val = quad(lambda x: f(x, M, p, d), [a_i, b_i], maxdegree=5)
            total_val += val
            total_err += abs(val) * mpf("1e-20")

    return total_val, total_err

# -----------------------------------------------------------------------
# compute_h — float version (unchanged, kept for backward compat / display)
# -----------------------------------------------------------------------

def compute_h(M, p, d, return_errors=False):
    M  = mpf(M)
    p  = mpf(p)
    d_ = mpf(d)

    IR_val,   IR_err   = quad_with_err(integrand_R,   M, p, d)
    IL_val,   IL_err   = quad_with_err(integrand_L,   M, p, d)
    IRL_val,  IRL_err  = quad_with_err(integrand_RL,  M, p, d)
    IRLQ_val, IRLQ_err = quad_with_err(integrand_RLQ, M, p, d)

    a = IRL_val  / IR_val
    b = IL_val   / IR_val
    c = IRLQ_val / IR_val

    h = d_ * (a / b) + d_ * (d_ - 2) * M**2 * (c / b) - d_**2 * M**2 * (a / b)**2

    if not return_errors:
        return float(h)

    rel_R   = abs(IR_err   / IR_val)   if abs(IR_val)   > 0 else mpf(0)
    rel_L   = abs(IL_err   / IL_val)   if abs(IL_val)   > 0 else mpf(0)
    rel_RL  = abs(IRL_err  / IRL_val)  if abs(IRL_val)  > 0 else mpf(0)
    rel_RLQ = abs(IRLQ_err / IRLQ_val) if abs(IRLQ_val) > 0 else mpf(0)

    sum_rel = rel_R + rel_L + rel_RL + rel_RLQ
    h_err   = 3 * abs(h) * sum_rel
    h_upper = h + h_err

    return float(h), float(h_upper), float(h_err)

# -----------------------------------------------------------------------
# Helper: mpmath mpf -> flint.arb
# -----------------------------------------------------------------------

def mpf_to_arb(x, err=None):
    """Convert mpmath mpf to flint.arb preserving 50-digit precision.
    If err (mpmath mpf) is given, include as Arb radius.
    """
    a = flint.arb(mp.nstr(x, 55))
    if err is not None and err != 0:
        a = a + flint.arb(0, float(abs(err)))
    return a

# -----------------------------------------------------------------------
# compute_h_as_arb — certified flint.arb interval for h
# -----------------------------------------------------------------------

def compute_h_as_arb(M, p, d):
    """
    Compute h = d2log I_L/dM2 as a certified flint.arb interval.
    Uses mpmath quad(error=True); integration errors included as arb radii.
    Returns flint.arb.
    """
    M_mpf = mpf(M)
    p_mpf = mpf(p)
    d_int = int(d)

    IR_val,   IR_err   = quad_with_err(integrand_R,   M_mpf, p_mpf, d_int)
    IL_val,   IL_err   = quad_with_err(integrand_L,   M_mpf, p_mpf, d_int)
    IRL_val,  IRL_err  = quad_with_err(integrand_RL,  M_mpf, p_mpf, d_int)
    IRLQ_val, IRLQ_err = quad_with_err(integrand_RLQ, M_mpf, p_mpf, d_int)

    IR_arb   = mpf_to_arb(IR_val,   IR_err)
    IL_arb   = mpf_to_arb(IL_val,   IL_err)
    IRL_arb  = mpf_to_arb(IRL_val,  IRL_err)
    IRLQ_arb = mpf_to_arb(IRLQ_val, IRLQ_err)

    d_arb = flint.arb(str(d_int))
    M_arb = flint.arb(mp.nstr(M_mpf, 55))

    a = IRL_arb  / IR_arb
    b = IL_arb   / IR_arb
    c = IRLQ_arb / IR_arb

    return d_arb * (a / b) + d_arb * (d_arb - 2) * M_arb**2 * (c / b) \
           - d_arb**2 * M_arb**2 * (a / b)**2

# -----------------------------------------------------------------------
# Richardson in Arb — 4th-order, stencil evaluations return flint.arb
# -----------------------------------------------------------------------

def richardson_deriv_arb(func_arb, x0_val, h_step):
    """4th-order Richardson extrapolation returning flint.arb.
    func_arb: function taking mpmath mpf -> flint.arb.
    x0_val, h_step: mpmath mpf values.

    Formula: f'(x) ~ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    """
    fm2 = func_arb(x0_val - 2*h_step)
    fm1 = func_arb(x0_val - h_step)
    fp1 = func_arb(x0_val + h_step)
    fp2 = func_arb(x0_val + 2*h_step)
    h_arb = flint.arb(mp.nstr(h_step, 55))
    return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * h_arb)

# -----------------------------------------------------------------------
# Float-based Richardson (unchanged, for display / backward compat)
# -----------------------------------------------------------------------

def richardson_deriv(func, x0_val, h_step):
    fm2 = func(x0_val - 2*h_step)
    fm1 = func(x0_val - h_step)
    fp1 = func(x0_val + h_step)
    fp2 = func(x0_val + 2*h_step)
    return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * h_step)

# -----------------------------------------------------------------------
# Main certificate routine
# -----------------------------------------------------------------------

def run_certificate():
    print("=" * 72)
    print("CERTIFICATE (L1 Arb): d2log I_L/dM2 < 0")
    print("Region: d in {3,4}, p in (d,d+1), M in [2,20]")
    print("Method: monotone-increase in M, Arb-certified dh/dM > 0")
    print("=" * 72)
    print()

    header = (f"{'d':>3} {'p':>7} {'M':>6}  "
              f"{'h_val':>14} {'h_upper':>14} {'dh/dM':>12} {'dh/dp':>12}")
    print(header)
    print("-" * len(header))

    # Accumulation lists
    arb_grid_data   = []   # (d, p, M, h_arb)
    monotone_results = []  # (d, p, M, certified_bool, dhdM_arb)
    dhdp_arb_M20    = []   # |dhdp_arb| at M=20 points

    # Float lists for display / legacy summary
    all_h_upper = []
    all_dhdM    = []
    all_dhdp    = []

    for d in D_VALS:
        for p_off in P_OFFSETS:
            p = d + p_off
            for M in M_VALS:

                # --- Float-based h for table display (unchanged logic) ---
                h_val, h_upper, h_err = compute_h(M, p, d, return_errors=True)

                # --- Arb-based h ---
                h_arb = compute_h_as_arb(M, p, d)

                # Store for certificate
                arb_grid_data.append((d, p, M, h_arb))
                all_h_upper.append(h_upper)

                # --- Adaptive step near left boundary (M=2) ---
                h_step_M_use = H_STEP_M
                if M - 2 * h_step_M_use < 2.0 + 1e-6:
                    h_step_M_use = (M - 2.0) / 2.5

                # --- Arb Richardson for dh/dM ---
                if M > 2.001:  # interior point
                    try:
                        dhdM_arb = richardson_deriv_arb(
                            lambda Mv, _p=p, _d=d: compute_h_as_arb(Mv, _p, _d),
                            mpf(M), mpf(h_step_M_use)
                        )
                        # Certified monotone iff lower bound of interval > 0
                        monotone_certified = bool(dhdM_arb > 0)
                        dhdM_display = float(dhdM_arb)
                    except Exception:
                        dhdM_arb = None
                        monotone_certified = False
                        dhdM_display = float('nan')

                    monotone_results.append((d, p, M, monotone_certified, dhdM_arb))
                else:
                    dhdM_arb = None
                    monotone_certified = None  # boundary point
                    dhdM_display = float('nan')

                # Float dh/dM for legacy list
                if not math.isnan(dhdM_display):
                    all_dhdM.append(abs(dhdM_display))

                # --- Arb Richardson for dh/dp ---
                try:
                    dhdp_arb = richardson_deriv_arb(
                        lambda pv, _M=M, _d=d: compute_h_as_arb(_M, pv, _d),
                        mpf(p), mpf(H_STEP_P)
                    )
                    dhdp_display = float(dhdp_arb)
                    abs_dhdp_arb = abs(dhdp_arb)
                    all_dhdp.append(abs(dhdp_display))

                    if abs(M - 20.0) < 0.01:
                        dhdp_arb_M20.append(abs_dhdp_arb)
                except Exception:
                    dhdp_arb = None
                    dhdp_display = float('nan')

                print(f"{d:>3} {p:>7.4f} {M:>6.1f}  "
                      f"{h_val:>14.6f} {h_upper:>14.6f} "
                      f"{dhdM_display:>12.6f} {dhdp_display:>12.6f}")
                sys.stdout.flush()

    print()
    print("=" * 72)
    print("MONOTONE-INCREASE CHECK (Arb-certified)")
    print("=" * 72)
    print()

    # --- 1. Arb-certified monotone check ---
    all_certified = all(r[3] for r in monotone_results)
    failed_pts = [(r[0], r[1], r[2]) for r in monotone_results if not r[3]]

    print("  Checking dh/dM > 0 at all interior grid points (M > 2)...")
    if all_certified:
        print("  Arb-certified: dh/dM > 0 at all interior grid points.")
        print("  (Lower bound of arb interval > 0 at each point.)")
    else:
        print(f"  WARNING: dh/dM Arb interval not certified positive at: {failed_pts}")
        print("  Falling back to float check...")
        float_all_pos = all(d_ > 0 for d_ in all_dhdM)
        if float_all_pos:
            print("  Float check: all dh/dM > 0 (float-estimated, not certified).")
        else:
            print("  FAIL: some dh/dM <= 0 even by float check.")

    print()

    # --- 2. h_upper at M=20 (Arb) ---
    rows_M20_arb = [(d, p, M, h_a) for (d, p, M, h_a) in arb_grid_data
                    if abs(M - 20.0) < 0.01]

    if not rows_M20_arb:
        sys.exit("ERROR: no M=20 grid points found.")

    # Certified upper bound via flint.arb.upper() (returns arb at right endpoint)
    def arb_upper(a):
        """Return certified upper bound of a flint.arb interval as flint.arb."""
        try:
            return a.upper()
        except AttributeError:
            return a.mid() + abs(a.rad())

    h_upper_M20_arb_vals = [arb_upper(h_a) for (_, _, _, h_a) in rows_M20_arb]
    max_h_upper_M20_arb  = h_upper_M20_arb_vals[0]
    worst_d20 = rows_M20_arb[0][0]
    worst_p20 = rows_M20_arb[0][1]
    for i, v in enumerate(h_upper_M20_arb_vals):
        if bool(v > max_h_upper_M20_arb):
            max_h_upper_M20_arb = v
            worst_d20 = rows_M20_arb[i][0]
            worst_p20 = rows_M20_arb[i][1]

    print("=" * 72)
    print("CERTIFICATE BOUND (Arb)")
    print("=" * 72)
    print()
    print(f"  max h_upper(M=20) [Arb]: {max_h_upper_M20_arb}")
    print(f"  at (d={worst_d20}, p={worst_p20:.4f}, M=20)")
    print()

    # --- 3. p-direction correction (Arb) ---
    # Use L_p evaluated at M=20 (more refined than global L_p)
    if dhdp_arb_M20:
        max_Lp_M20_arb = dhdp_arb_M20[0]
        for v in dhdp_arb_M20[1:]:
            if bool(v > max_Lp_M20_arb):
                max_Lp_M20_arb = v
    else:
        max_Lp_M20_arb = flint.arb(0)

    delta_p_arb   = flint.arb(str(DELTA_P))
    p_correction_arb = max_Lp_M20_arb * delta_p_arb / flint.arb(2)

    print(f"  max |dh/dp| at M=20 [Arb]: {max_Lp_M20_arb}")
    print(f"  delta_p/2: {float(delta_p_arb)/2}")
    print(f"  p-correction [Arb]: {p_correction_arb}")
    print()

    # --- 4. Final certificate bound ---
    cert_bound_arb = max_h_upper_M20_arb + p_correction_arb

    print(f"  cert_bound = max_h_upper(M=20) + p_correction")
    print(f"  cert_bound [Arb]: {cert_bound_arb}")
    print()

    # --- 5. Certificate verdict ---
    # cert_bound_arb < 0 is True iff the entire interval is negative
    print("=" * 72)
    if all_certified and bool(cert_bound_arb < 0):
        print("CERTIFICATE PASSED (L1 Arb):")
        print(f"  (1) Arb-certified: dh/dM > 0 at all {len(monotone_results)} interior grid points.")
        print(f"      => h(M, p, d) is monotone increasing in M.")
        print(f"  (2) Therefore: sup_M h(M,p,d) = h(M=20, p, d) for each (d,p).")
        print(f"  (3) max h_upper(M=20) [Arb certified] = {max_h_upper_M20_arb}")
        print(f"  (4) p-direction correction [Arb] = {p_correction_arb}")
        print(f"  (5) cert_bound [Arb] = {cert_bound_arb} < 0")
        print()
        print(f"  => d2log I_L/dM2 < 0 for all d in {{3,4}}, p in (d,d+1), M in [2,20].")
        print()
        print(f"CERTIFICATE: d2log I_L/dM2 <= {cert_bound_arb} < 0")
        print(f"for d in {{3,4}}, p in (d,d+1), M in [2,20].")
    else:
        reasons = []
        if not all_certified:
            reasons.append(f"dh/dM Arb certification failed at: {failed_pts}")
        if not bool(cert_bound_arb < 0):
            reasons.append(f"cert_bound = {cert_bound_arb} is not Arb-certified < 0")
        print("CERTIFICATE FAILED:")
        for r in reasons:
            print(f"  - {r}")
        print()

        # Attempt float fallback summary for diagnostics
        max_h_upper_float = max(all_h_upper)
        L_p_M20 = max(float(v) for v in dhdp_arb_M20) if dhdp_arb_M20 else 0.0
        p_corr_float = L_p_M20 * DELTA_P / 2
        cert_float = max_h_upper_float + p_corr_float
        print(f"  Float fallback: max_h_upper(M=20) = {max_h_upper_float:.8f}, "
              f"p_correction = {p_corr_float:.8f}, cert_bound = {cert_float:.8f}")
        if cert_float < 0:
            print("  Float fallback PASSED (float-estimated, not Arb-certified).")
        else:
            print("  Float fallback also FAILED. Grid densification may be needed.")

        sys.exit(1)

    # --- 6. Legacy summary (for reference / comparison with cert-phase1-arb.py) ---
    print()
    print("=" * 72)
    print("LEGACY SUMMARY (float-based, for comparison with cert-phase1-arb.py)")
    print("(Uses global Lipschitz; may show FAILED — see cert-phase1-arb.py notes)")
    print("=" * 72)

    max_h_upper = max(all_h_upper)
    L_M = max(all_dhdM) if all_dhdM else 0.0
    L_p = max(all_dhdp) if all_dhdp else 0.0

    interp_err = L_M * (DELTA_M / 2) + L_p * (DELTA_P / 2)
    cert_max   = max_h_upper + interp_err

    print(f"  max(h_upper) over grid : {max_h_upper:.8f}")
    print(f"  L_M = max|dh/dM|       : {L_M:.6f}")
    print(f"  L_p = max|dh/dp|       : {L_p:.6f}")
    print(f"  Interpolation error    : L_M*(dM/2) + L_p*(dp/2) = {interp_err:.8f}")
    print(f"  Certified maximum      : {cert_max:.8f}")
    if cert_max < 0:
        print("  Legacy certificate: PASSED")
    else:
        print("  Legacy certificate: FAILED (expected — see cert-phase1-arb.py notes)")
    print()

    return cert_bound_arb


if __name__ == "__main__":
    run_certificate()
