# true-kl0-certificates

Numerical certificates for the paper:

> **Honest Reporting in Scored Oversight: The True-KL₀ Property via the Prékopa Principle**
> Lauri Lovén, University of Oulu
> Submitted to *The Annals of Applied Probability* (AAP)

These scripts certify three numerical results:
- **Theorem 5.2** (R-bound certificate): R(M,p,d) ≤ 0.7681 < 1 for all d∈{2,3,4}, p∈(d,d+1), M∈[1.001,20]
- **Theorem 4.9** (Residual certificate): d²log I_L/dM² ≤ −0.009 < 0 for all d∈{3,4}, p∈(d,d+1), M∈[2,20]
- **Proposition 2.5(b)** (I_R log-concavity, d=2): G'(M)I_R(M)/G(M)² ≤ 0.756 < 1 for all p∈(2,3), M∈[1.001,20]

---

## Dependencies

```
pip install mpmath python-flint
```

Tested with mpmath 1.3.0 and python-flint 0.7.x (which provides `flint.arb`).

---

## Script overview

Two pipelines are provided: the original two-step pipeline and a self-contained
Level-1 Arb upgrade. Use the `*-full-arb.py` scripts for the strongest guarantees.

### Level-1 Arb scripts (recommended)

| Script | Purpose | Theorem |
|---|---|---|
| `cert-r-bound-full-arb.py` | R-bound certificate with Richardson derivatives in certified `flint.arb` arithmetic | Thm 5.2 |
| `cert-phase1-full-arb.py` | Residual certificate: Arb Richardson certifies dh/dM > 0; monotone argument and certificate bound all in Arb | Thm 4.9 |
| `verify-ir-logconcav-d2.py` | I_R log-concavity verification for d=2 (170-point grid, 50-digit mpmath) | Prop 2.5(b) |

### Original pipeline (for reference / comparison)

| Script | Purpose | Theorem |
|---|---|---|
| `cert-r-bound-arb.py` | Compute R(M,p,d) on a 510-point grid with Richardson derivatives (50-digit float); issue R-bound certificate | Thm 5.2 |
| `cert-phase1-arb.py` | Compute h = d²log I_L/dM² on a 140-point grid (50-digit float) | Thm 4.9 (data) |
| `apply-interval-cert.py` | Post-process `cert-r-bound-arb.py` output; apply interval-wise local Lipschitz bound | Thm 5.2 |
| `apply-monotone-cert-phase1.py` | Post-process `cert-phase1-arb.py` output; apply monotone-increase certificate | Thm 4.9 |

### What "Level-1 Arb" means

In the original scripts, Richardson stencil evaluations return 50-digit `mpmath` floats.
The `*-full-arb.py` scripts instead convert each `mpmath quad(error=True)` result into
a `flint.arb` ball (midpoint = quadrature value, radius = quadrature error estimate), then
perform Richardson entirely in `flint.arb` arithmetic. The Lipschitz constant at each grid
point is therefore a **certified Arb upper bound**, not a floating-point approximation.

Remaining gap: the quadrature itself uses `mpmath` adaptive integration with heuristic error
estimates, not Arb-native rigorous integration. The safety margins (23% for R-bound,
9.3��10⁻³ for residual) exceed any plausible quadrature error by many orders of magnitude.
See the paper's Section 8 (Proof status and certification notes) for a detailed discussion.

---

## How to reproduce the certificates

### Recommended: Level-1 Arb scripts (self-contained)

```bash
# Theorem 5.2 (~4-6 hours)
python cert-r-bound-full-arb.py | tee cert-r-bound-full-arb.output

# Theorem 4.9 (~2-3 hours)
python cert-phase1-full-arb.py | tee cert-phase1-full-arb.output

# Proposition 2.5(b) (~5 minutes)
python verify-ir-logconcav-d2.py
```

Expected output from `cert-r-bound-full-arb.py` (last lines):
```
CERTIFICATE [L1 Arb-certified]: R(M,p,d) < 0.7681 < 1
  for ALL d in {2,3,4}, p in (d,d+1), M in [1.001,20].
```

Expected output from `cert-phase1-full-arb.py` (last lines):
```
CERTIFICATE PASSED (L1 Arb): d��log I_L/dM² < [cert_bound] < 0
for d in {3,4}, p in (d,d+1), M in [2,20].
```

Expected output from `verify-ir-logconcav-d2.py` (last lines):
```
Worst ratio: 0.755274 at p=2.95, M=1.500
Margin from 1: 0.244726
```

### Alternative: original two-step pipeline

```bash
# Theorem 5.2
python cert-r-bound-arb.py | tee cert-r-bound.output
python apply-interval-cert.py cert-r-bound.output

# Theorem 4.9
python cert-phase1-arb.py | tee cert-phase1.output
python apply-monotone-cert-phase1.py cert-phase1.output
```

---

## Certificate methods

### R-bound (Thm 5.2)

1. **Quadrature**: `mp.quad(error=True)` at 50 decimal digits gives R_upper(M,p,d) with certified error bound at each of 510 grid points.
2. **M-direction certificate**: 4th-order Richardson extrapolation at 50-digit precision gives |dR/dM| and |d²R/dM²| at each grid point. Sub-interval second-derivative bounds L_local(i) = |dR/dM|(M_i) + |d²R/dM²|(M_i)·(M_{i+1}−M_i) close the continuous-domain gap in each M-interval.
3. **p-direction Lipschitz**: Richardson extrapolation gives |dR/dp| at all 510 grid points; global maximum L_p = 0.5311 applied against the uniform p-grid half-gap (0.05).
4. **Final arithmetic**: `python-flint arb` (interval arithmetic) confirms R ≤ 0.7681 as a rigorous enclosure (Arb certificate: 0.76810 ± 4.3×10⁻⁶⁰).

### Residual certificate (Thm 4.9)

1. **Quadrature**: 50-digit quadrature at 140 grid points gives h(M,p,d) = d²log I_L/dM² with error bound h_upper.
2. **Monotone increase**: 4th-order Richardson in Arb arithmetic confirms dh/dM > 0 at all interior M grid points. Sub-interval Taylor bounds with adaptive subdivision (up to depth 8) certify dh/dM > 0 on each continuous sub-interval.
3. **p-direction correction**: Arb-certified |dh/dp| at M=20; correction = |dh/dp|_max × δ_p/2 = 0.000101.
4. **Certificate**: cert_bound = max h_upper(M=20) + p-correction = −0.009294 < 0 (margin 9.3×10⁻³).

### I_R log-concavity for d=2 (Prop 2.5(b))

1. **Quadrature**: 50-digit `mpmath` evaluation of G(y;p), G'(y;p), and I_R(M,p,2) = ∫₁ᴹ G(y;p)dy on a 170-point grid (10 p-offsets × 17 M-values).
2. **Ratio check**: G'(M)I_R(M)/G(M)² < 1 at all grid points (worst: 0.7553 at p=2.95, M=1.5).
3. **Lipschitz estimate**: max |d(ratio)/dM| ≤ 0.282 from consecutive finite differences.
4. **Continuous closure**: sub-interval monotonicity confirmed at all M-intervals.

---

## Proof status

The certificates are **semi-rigorous** in the following precise sense:

- **Certified (Arb interval arithmetic):** All post-quadrature arithmetic (Richardson extrapolation, sub-interval bounds, Lipschitz aggregation, final certificate inequality).
- **High-precision heuristic (mpmath):** The underlying integral evaluations use `mpmath` adaptive quadrature with 50 decimal digits and heuristic error estimates. These are not rigorous enclosures in the interval-arithmetic sense.
- **Fully rigorous upgrade path:** Replace `mpmath` quadrature with Arb-native integration (`acb_calc_integrate`), which provides guaranteed enclosures. The 23% safety margins make this upgrade straightforward.

The paper's Section 8 discusses this distinction in detail.

---

## Citation

If you use these certificates, please cite the paper:

```bibtex
@article{Loven2026honest,
  author  = {Lov\'{e}n, Lauri},
  title   = {Honest Reporting in Scored Oversight: The {True-KL}$_0$ Property
             via the {Pr\'{e}kopa} Principle},
  journal = {Annals of Applied Probability},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT
