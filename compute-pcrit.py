#!/usr/bin/env python3
"""
Compute p_crit(d) for d in {5, 6, 7, 8} by bisection on p.

p_crit(d) = inf{p in (d, d+1) : sup_M R(M,p,d) >= 1}

For each p, R_peak = sup_M R(M,p,d) is found by golden-section search on M.
Then bisection on p locates the crossing R_peak = 1.
"""
import mpmath as mp
mp.dps = 30  # 30 digits sufficient for locating p_crit

def x0(M):
    return (M + 1) / (M - 1)

def phi_fn(x, M):
    return M*M*(x-1)**2 - (x+1)**2

def sigma2_fn(x, M):
    return phi_fn(x, M) / (x - 1)**2

def NA(x, p):
    return x**(2*p-2) - (p-1)*x**p + (p-1)*x**(p-2) - 1

def D_fn(x, p):
    return (x**(p-1)+1)**2 + (p-1)*x**(p-2)*(x+1)**2

def w_fn(x, M, p, d):
    ph = phi_fn(x, M)
    if ph <= 0:
        return mp.mpf(0)
    xpm1 = x**p - 1
    if xpm1 <= 0:
        return mp.mpf(0)
    if d == 2:
        phi_pow = mp.mpf(1)
    else:
        phi_pow = ph**((d-2)*mp.mpf('0.5'))
    return phi_pow / (xpm1 * (x-1)**(p+d-2))

def compute_R(M, p, d):
    """Compute R(M,p,d) = p/(d-1) * I_L/I_R."""
    M = mp.mpf(M)
    p = mp.mpf(p)
    x0v = x0(M)

    def f_IR(x):
        ph = phi_fn(x, M)
        if ph <= 0:
            return mp.mpf(0)
        return D_fn(x, p) * w_fn(x, M, p, d)

    def f_IL(x):
        ph = phi_fn(x, M)
        if ph <= 0:
            return mp.mpf(0)
        xpm1 = x**p - 1
        if xpm1 <= 0:
            return mp.mpf(0)
        na = NA(x, p)
        s2 = sigma2_fn(x, M)
        w = w_fn(x, M, p, d)
        return na**2 / xpm1**2 * s2 * w

    IR = mp.quad(f_IR, [x0v, mp.inf])
    IL = mp.quad(f_IL, [x0v, mp.inf])

    if IR == 0:
        return mp.mpf(0)
    return p / (d - 1) * IL / IR

def find_R_peak(p, d, M_lo=1.5, M_hi=50.0, tol=0.05):
    """Find sup_M R(M,p,d) by golden-section search."""
    gr = (mp.sqrt(5) + 1) / 2
    a, b = mp.mpf(M_lo), mp.mpf(M_hi)

    while (b - a) > tol:
        c = b - (b - a) / gr
        d_pt = a + (b - a) / gr
        Rc = compute_R(c, p, d)
        Rd = compute_R(d_pt, p, d)
        if Rc > Rd:
            b = d_pt
        else:
            a = c

    M_peak = (a + b) / 2
    R_peak = compute_R(M_peak, p, d)
    return float(R_peak), float(M_peak)

def find_p_crit(d, p_lo=None, p_hi=None, tol=0.001):
    """Find p_crit(d) by bisection: R_peak(p_lo) < 1, R_peak(p_hi) >= 1."""
    if p_lo is None:
        p_lo = float(d) + 0.1
    if p_hi is None:
        p_hi = float(d) + 0.99

    # Verify bracket
    R_lo, _ = find_R_peak(p_lo, d)
    R_hi, _ = find_R_peak(p_hi, d)
    print(f"  Bracket: R_peak({p_lo:.2f})={R_lo:.4f}, R_peak({p_hi:.2f})={R_hi:.4f}")

    if R_lo >= 1:
        print(f"  WARNING: R_peak already >= 1 at p_lo={p_lo}")
        return p_lo, p_lo
    if R_hi < 1:
        print(f"  R_peak < 1 at p_hi={p_hi} — True-KL0 holds for all p in ({d},{d+1})")
        return None, None

    while (p_hi - p_lo) > tol:
        p_mid = (p_lo + p_hi) / 2
        R_mid, M_mid = find_R_peak(p_mid, d)
        print(f"  bisect: p={p_mid:.4f}, R_peak={R_mid:.6f} (M_peak={M_mid:.2f})")
        if R_mid < 1:
            p_lo = p_mid
        else:
            p_hi = p_mid

    return p_lo, p_hi

# Main
print("=" * 70)
print("Computing p_crit(d) for d in {5, 6, 7, 8}")
print("=" * 70)

for d in [5, 6, 7, 8]:
    print(f"\n--- d = {d} ---")
    lo, hi = find_p_crit(d)
    if lo is not None:
        print(f"  => p_crit({d}) in ({lo:.4f}, {hi:.4f})")
        print(f"     midpoint: {(lo+hi)/2:.4f}, half-width: {(hi-lo)/2:.4f}")
    else:
        print(f"  => True-KL0 holds unconditionally for d={d}")
