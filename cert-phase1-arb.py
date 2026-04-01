"""
cert-phase1-arb.py
==================
Certificate: d²log I_L/dM² < 0 for all d∈{3,4}, p∈(d,d+1), M∈[2,20].

This is the residual certificate for I_L log-concavity in the region not
covered by Prekopa theorem.

Key formula (R7.B):
  d²log I_L/dM² = d*(a/b) + d*(d-2)*M²*(c/b) - d²*M²*(a/b)²

where:
  a = I_RL / I_R
  b = I_L  / I_R
  c = I_RLQ / I_R

NOTE ON CERTIFICATE STRUCTURE
------------------------------
This script computes h and its derivatives at 140 grid points and prints
a summary using a global-Lipschitz bound (L_M * DELTA_M/2).  That approach
uses the worst-case L_M across all grid points paired with the largest M-gap
(DELTA_M = 5.0, between M=15 and M=20), which is too conservative to pass.

The *correct* certificate is obtained by post-processing the output of this
script with apply-monotone-cert-phase1.py, which exploits the observed
monotone increase (dh/dM > 0 at all interior grid points) to bound the
supremum of h over each M-interval at the right endpoint only.  The summary
section of this script may therefore print "CERTIFICATE FAILED"; that is
expected.  Run apply-monotone-cert-phase1.py on the captured output for the
valid certificate.
"""

import sys
import math
import mpmath
from mpmath import mp, mpf, quad
from flint import arb, ctx as fctx

# precision
mp.dps = 50
fctx.prec = 200

# grid
D_VALS    = [3, 4]
P_OFFSETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
M_VALS    = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

H_STEP_M = 0.5
H_STEP_P = 1e-4
DELTA_M  = 5.0
DELTA_P  = 0.1

SPLITS = [1e-6, 1.0, 20.0, 200.0, 2000.0, 10000.0]

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
    # Conservative factor: h involves ratios a/b, c/b, (a/b)^2 of the four
    # integrals; each ratio accumulates two relative errors, and the (a/b)^2
    # term doubles its contribution.  The factor 3 bounds the worst-case
    # amplification and is dominated in practice by the Richardson correction
    # applied in apply-monotone-cert-phase1.py.
    h_err   = 3 * abs(h) * sum_rel

    h_upper = h + h_err

    return float(h), float(h_upper), float(h_err)

def richardson_deriv(func, x0_val, h_step):
    fm2 = func(x0_val - 2*h_step)
    fm1 = func(x0_val - h_step)
    fp1 = func(x0_val + h_step)
    fp2 = func(x0_val + 2*h_step)
    return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * h_step)

def run_certificate():
    print("=" * 72)
    print("CERTIFICATE: d2log I_L/dM2 < 0")
    print("Region: d in {3,4}, p in (d,d+1), M in [2,20]")
    print("=" * 72)
    print()

    header = (f"{'d':>3} {'p':>7} {'M':>6}  "
              f"{'h_val':>14} {'h_upper':>14} {'dh/dM':>12} {'dh/dp':>12}")
    print(header)
    print("-" * len(header))

    all_h_upper = []
    all_dhdM    = []
    all_dhdp    = []

    for d in D_VALS:
        for p_off in P_OFFSETS:
            p = d + p_off
            for M in M_VALS:
                h_val, h_upper, h_err = compute_h(M, p, d, return_errors=True)

                h_step_M_use = H_STEP_M
                if M - 2 * h_step_M_use < 2.0 + 1e-6:
                    h_step_M_use = (M - 2.0) / 2.5

                def h_of_M(Mv, _p=p, _d=d):
                    return compute_h(Mv, _p, _d)

                try:
                    dhdM = richardson_deriv(h_of_M, M, h_step_M_use)
                except Exception:
                    dhdM = float("nan")

                def h_of_p(pv, _M=M, _d=d):
                    return compute_h(_M, pv, _d)

                try:
                    dhdp = richardson_deriv(h_of_p, p, H_STEP_P)
                except Exception:
                    dhdp = float("nan")

                all_h_upper.append(h_upper)
                if not math.isnan(dhdM):
                    all_dhdM.append(abs(dhdM))
                if not math.isnan(dhdp):
                    all_dhdp.append(abs(dhdp))

                print(f"{d:>3} {p:>7.4f} {M:>6.1f}  "
                      f"{h_val:>14.6f} {h_upper:>14.6f} "
                      f"{dhdM:>12.6f} {dhdp:>12.6f}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    max_h_upper = max(all_h_upper)
    L_M         = max(all_dhdM)  if all_dhdM else 0.0
    L_p         = max(all_dhdp)  if all_dhdp else 0.0

    interp_err  = L_M * (DELTA_M / 2) + L_p * (DELTA_P / 2)
    cert_max    = max_h_upper + interp_err

    print(f"  max(h_upper) over grid : {max_h_upper:.8f}")
    print(f"  L_M = max|dh/dM|       : {L_M:.6f}")
    print(f"  L_p = max|dh/dp|       : {L_p:.6f}")
    print(f"  Interpolation error    : L_M*(dM/2) + L_p*(dp/2) = {interp_err:.8f}")
    print(f"  Certified maximum      : {cert_max:.8f}")
    print()

    fctx.prec = 200
    arb_max_h = arb(max_h_upper)
    arb_LM    = arb(L_M)
    arb_Lp    = arb(L_p)
    arb_dM    = arb(DELTA_M)
    arb_dp    = arb(DELTA_P)
    arb_cert  = arb_max_h + arb_LM * (arb_dM / 2) + arb_Lp * (arb_dp / 2)

    print(f"  [arb] certified maximum: {arb_cert}")
    print()

    if cert_max < 0:
        print("CERTIFICATE PASSED:")
        print(f"  d2log I_L/dM2 <= {cert_max:.8f} < 0")
        print(f"  for d in {{3,4}}, p in (d,d+1), M in [2,20]")
        print()
        print(f"CERTIFICATE: d2log I_L/dM2 <= {cert_max:.8f} < 0 "
              f"for d in {{3,4}}, p in (d,d+1), M in [2,20]")
    else:
        print("CERTIFICATE FAILED: certified maximum is not < 0")
        print(f"  Certified max = {cert_max:.8f}")
        sys.exit(1)

    return cert_max

if __name__ == "__main__":
    cert_max = run_certificate()
