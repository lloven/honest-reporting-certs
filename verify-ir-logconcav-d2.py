#!/usr/bin/env python3
"""
Verify I_R log-concavity for d=2: check G'(M)*I_R(M)/G(M)^2 < 1
for all p in (2,3) and M in [1.001, 20].

For d=2, I_R(M,p,2) = int_1^M G(y;p) dy, where G(y;p) > 0 is M-independent.
Log-concavity d^2 log I_R / dM^2 < 0 iff G'(M;p) * I_R(M) < G(M;p)^2.
"""
import mpmath as mp
mp.mp.dps = 50

def NA(x, p):
    return x**(2*p-2) - (p-1)*x**p + (p-1)*x**(p-2) - 1

def D_fn(x, p):
    return (x**(p-1)+1)**2 + (p-1)*x**(p-2)*(x+1)**2

def G_y(y, p):
    """G(y;p) = 2*D(x(y),p)*(y-1)^{p-2} / (2^p * (x(y)^p - 1)^2), x=(y+1)/(y-1)"""
    x = (y + 1) / (y - 1)
    D = D_fn(x, p)
    xp1 = x**p - 1
    return 2 * D * (y - 1)**(p - 2) / (mp.mpf(2)**p * xp1**2)

def G_y_prime(y, p, h=mp.mpf('1e-8')):
    """Numerical derivative of G(y;p) w.r.t. y using central differences."""
    return (G_y(y + h, p) - G_y(y - h, p)) / (2 * h)

def I_R_d2(M, p):
    """I_R(M,p,2) = int_1^M G(y;p) dy"""
    if M <= 1:
        return mp.mpf(0)
    return mp.quad(lambda y: G_y(y, p), [mp.mpf(1) + mp.mpf('1e-12'), M])

# Grid
p_offsets = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
M_vals = [1.001, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 5.75,
          6.5, 8.0, 10.0, 12.0, 15.0, 20.0]

print("d=2 I_R log-concavity check: G'(M)*I_R(M)/G(M)^2 < 1")
print("=" * 90)
print(f"{'p':>6}  {'M':>8}  {'G(M)':>14}  {'G_prime(M)':>14}  {'I_R(M)':>14}  {'ratio':>10}  {'ok':>4}")
print("-" * 90)

worst_ratio = mp.mpf(0)
worst_pt = None
max_deriv_ratio = mp.mpf(0)  # for Lipschitz of the ratio itself

all_ratios = []

for p_off in p_offsets:
    p = mp.mpf(2) + mp.mpf(str(p_off))
    for M in M_vals:
        M = mp.mpf(str(M))
        Gval = G_y(M, p)
        Gprime = G_y_prime(M, p)
        IR = I_R_d2(M, p)

        if Gval == 0 or IR == 0:
            ratio = mp.mpf(0)
        else:
            ratio = Gprime * IR / Gval**2

        ok = "OK" if ratio < 1 else "FAIL"
        all_ratios.append((float(p), float(M), float(ratio)))

        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_pt = (float(p), float(M))

        print(f"{float(p):6.2f}  {float(M):8.3f}  {float(Gval):14.6e}  {float(Gprime):14.6e}  {float(IR):14.6e}  {float(ratio):10.6f}  {ok}")

print("=" * 90)
print(f"Worst ratio: {float(worst_ratio):.6f} at p={worst_pt[0]:.2f}, M={worst_pt[1]:.3f}")
print(f"Margin from 1: {float(1 - worst_ratio):.6f}")

# Estimate Lipschitz constant of the ratio
# Use consecutive M pairs to estimate |d(ratio)/dM|
print("\n--- Lipschitz estimate for ratio function ---")
max_L = 0.0
for p_off in p_offsets:
    p_val = 2.0 + p_off
    pts = [(pv, mv, rv) for (pv, mv, rv) in all_ratios if abs(pv - p_val) < 0.001]
    for i in range(len(pts) - 1):
        _, m1, r1 = pts[i]
        _, m2, r2 = pts[i + 1]
        if m2 > m1:
            L = abs(r2 - r1) / (m2 - m1)
            if L > max_L:
                max_L = L
                max_L_pt = (p_val, m1, m2)

print(f"Max |d(ratio)/dM| estimate: {max_L:.4f}")
print(f"  at p={max_L_pt[0]:.2f}, M in [{max_L_pt[1]:.3f}, {max_L_pt[2]:.3f}]")
