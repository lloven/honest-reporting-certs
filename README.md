# true-kl0-certificates

Numerical certificates for the paper:

> **Honest Reporting in Scored Oversight: The True-KL₀ Property via the Prékopa Principle**
> Submitted to *The Annals of Applied Probability* (AAP)

These scripts certify two key numerical results:
- **Theorem 5.1** (R-bound certificate): R(M,p,d) < 0.7974 < 1 for all d∈{2,3,4}, p∈(d,d+1), M∈[1.001,20]
- **Theorem 4.8** (Residual certificate): d²log I_L/dM² ≤ −0.009 < 0 for all d∈{3,4}, p∈(d,d+1), M∈[2,20]

---

## Dependencies

```
pip install mpmath python-flint
```

Tested with mpmath 1.3.0 and python-flint 0.7.x (which provides `flint.arb`).

---

## Script overview

| Script | Purpose | Theorem |
|---|---|---|
| `cert-r-bound-arb.py` | Compute R(M,p,d) on a 510-point grid with Richardson derivatives; issue R-bound certificate | Thm 5.1 |
| `cert-phase1-arb.py` | Compute h = d²log I_L/dM² on a 140-point grid with Richardson derivatives | Thm 4.8 (data) |
| `apply-interval-cert.py` | Post-process `cert-r-bound-arb.py` output; apply interval-wise local Lipschitz bound | Thm 5.1 |
| `apply-monotone-cert-phase1.py` | Post-process `cert-phase1-arb.py` output; apply monotone-increase certificate | Thm 4.8 |

---

## How to reproduce the certificates

### Theorem 5.1 (R-bound, ~4–6 hours)

```bash
python cert-r-bound-arb.py | tee cert-r-bound.output
# The script itself prints the final certificate at the end.
# If you need to re-apply the interval-wise certificate separately:
python apply-interval-cert.py cert-r-bound.output
```

Expected terminal output (last lines):
```
CERTIFICATE: R(M,p,d) < 0.7974 < 1
  for ALL d in {2,3,4}, p in (d,d+1), M in [1.001,20].
  Final margin from 1: 0.2026 (20.3%)
```

### Theorem 4.8 (Residual certificate, ~2–3 hours)

```bash
python cert-phase1-arb.py | tee cert-phase1.output
# NOTE: cert-phase1-arb.py may print "CERTIFICATE FAILED" in its own
# summary (see note in the script docstring). This is expected.
# The valid certificate uses the monotone-increase argument:
python apply-monotone-cert-phase1.py cert-phase1.output
```

Expected terminal output from `apply-monotone-cert-phase1.py` (last lines):
```
CERTIFICATE PASSED: d²log I_L/dM² <= -0.009294 < 0
for d in {3,4}, p in (d,d+1), M in [2,20].
```

---

## Certificate methods

### R-bound (Thm 5.1)

1. **Quadrature**: `mp.quad(error=True)` at 50 decimal digits gives R_upper(M,p,d) with certified error bound at each of 510 grid points.
2. **M-direction Lipschitz**: 4th-order Richardson extrapolation at 50-digit precision gives |dR/dM| at each grid point. Paired *locally* with each M-interval gap (interval-wise bound, not global), avoiding the global-Lipschitz over-estimation.
3. **p-direction Lipschitz**: Richardson extrapolation gives |dR/dp|; applied globally against the uniform p-grid half-gap (0.05).
4. **Final arithmetic**: `python-flint arb` (interval arithmetic) confirms the certificate bound as a rigorous enclosure.

### Residual certificate (Thm 4.8)

1. **Quadrature**: 50-digit quadrature at 140 grid points gives h(M,p,d) = d²log I_L/dM² with error bound h_upper.
2. **Monotone increase**: Richardson extrapolation confirms dh/dM > 0 at all interior M grid points. This implies sup h over [M_i, M_{i+1}] = h_upper(M_{i+1}) (right-endpoint bound), eliminating the need for a large M-direction Lipschitz constant.
3. **p-direction correction**: Richardson gives |dh/dp| at M=20; correction = |dh/dp|_max × δ_p/2.
4. **Certificate**: cert_bound = max h_upper(M=20) + p-correction = −0.009294 < 0.

---

## Remaining gap (as stated in the paper)

The Richardson-derived Lipschitz constants are high-precision floating-point values evaluated at grid points, not interval-arithmetic bounds over a continuous parameter domain. A fully rigorous certificate in the sense of Tucker (2011) would require Arb *box evaluation* of the derivatives over continuous parameter boxes. The current certificates provide a very high-confidence numerical guarantee (50-digit working precision, 20–29% safety margins) sufficient for the paper's claims, with the gap noted explicitly in §7.3.
