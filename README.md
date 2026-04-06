# true-kl0-certificates

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19435617.svg)](https://doi.org/10.5281/zenodo.19435617)

Numerical certificates for the paper:

> **Honest Reporting in Scored Oversight: The True-KL₀ Property via the Prékopa Principle**
> Lauri Lovén, University of Oulu
> Submitted to *The Annals of Applied Probability* (AAP)

These scripts certify three numerical results:
- **Theorem 5.2** (R-bound certificate): R(M,p,d) ≤ 0.7681 < 1 for all d∈{2,3,4}, p∈(d,d+1), M∈[1.001,20]
- **Theorem 4.9** (Residual certificate): d²log I_L/dM² ≤ −0.009 < 0 for all d∈{3,4}, p∈(d,d+1), M∈[2,20]
- **Proposition 2.5(b)** (I_R log-concavity, d=2): G'(M)I_R(M)/G(M)² ≤ 0.756 < 1 for all p∈(2,3), M∈[1.001,20]

Additionally, `compute-pcrit.py` locates the dimensional boundary p_crit(d) for d ≥ 5.

---

## Dependencies

```
pip install mpmath python-flint
```

Tested with mpmath 1.3.0 and python-flint 0.7.x (which provides `flint.arb`).

---

## Scripts

| Script | Purpose | Reference |
|---|---|---|
| `cert-r-bound-full-arb.py` | R-bound certificate: 510-point grid with Richardson derivatives in certified `flint.arb` arithmetic, sub-interval second-derivative bounds | Thm 5.2 |
| `cert-phase1-full-arb.py` | Residual certificate: Arb Richardson certifies dh/dM > 0 at each grid point; monotone argument and certificate bound all in Arb | Thm 4.9 |
| `verify-ir-logconcav-d2.py` | I_R log-concavity verification for d=2 (170-point grid, 50-digit mpmath) | Prop 2.5(b) |
| `compute-pcrit.py` | Locate p_crit(d) by bisection for d ≥ 5; finds d=5 is the unique transitional dimension | Sec 7 |

The certificate scripts use `mpmath` 50-digit quadrature for integral evaluation and
`python-flint` (Arb) ball arithmetic for all post-quadrature steps (Richardson
extrapolation, sub-interval bounds, Lipschitz aggregation, final certificate inequality).
The Lipschitz constant at each grid point is a **certified Arb upper bound**.

Remaining gap: the quadrature itself uses `mpmath` adaptive integration with heuristic
error estimates, not Arb-native rigorous integration. The safety margins (23% for
R-bound, 9.3×10⁻³ for residual) exceed any plausible quadrature error by many orders
of magnitude. See the paper's Section 8 for a detailed discussion.

---

## How to reproduce

```bash
# Theorem 5.2 (~4-6 hours)
python cert-r-bound-full-arb.py | tee cert-r-bound-full-arb.output

# Theorem 4.9 (~2-3 hours)
python cert-phase1-full-arb.py | tee cert-phase1-full-arb.output

# Proposition 2.5(b) (~5 minutes)
python verify-ir-logconcav-d2.py

# Dimensional boundary (~30 minutes)
python compute-pcrit.py
```

Expected outputs:

```
# cert-r-bound-full-arb.py
CERTIFICATE [L1 Arb-certified]: R(M,p,d) < 0.7681 < 1
  for ALL d in {2,3,4}, p in (d,d+1), M in [1.001,20].

# cert-phase1-full-arb.py
CERTIFICATE PASSED (L1 Arb): d²log I_L/dM² < [cert_bound] < 0
for d in {3,4}, p in (d,d+1), M in [2,20].

# verify-ir-logconcav-d2.py
Worst ratio: 0.755274 at p=2.95, M=1.500
Margin from 1: 0.244726

# compute-pcrit.py
p_crit(5) in (5.5719, 5.5728)
d >= 6: R_peak > 1 for all p in (d, d+1)
```

Reference outputs from our runs are included as `.output` files.

---

## Certificate methods

### R-bound (Thm 5.2)

1. **Quadrature**: `mp.quad(error=True)` at 50 decimal digits gives R_upper(M,p,d) with certified error bound at each of 510 grid points.
2. **M-direction certificate**: 4th-order Richardson extrapolation gives |dR/dM| and |d²R/dM²| at each grid point. Sub-interval second-derivative bounds L_local(i) = |dR/dM|(M_i) + |d²R/dM²|(M_i)·(M_{i+1}−M_i) close the continuous-domain gap.
3. **p-direction Lipschitz**: Global maximum L_p = 0.5311 applied against uniform p-grid half-gap (0.05).
4. **Final arithmetic**: `flint.arb` confirms R ≤ 0.7681 (Arb certificate: 0.76810 ± 4.3×10⁻⁶⁰).

### Residual certificate (Thm 4.9)

1. **Quadrature**: 50-digit quadrature at 140 grid points gives h = d²log I_L/dM².
2. **Monotone increase**: 4th-order Richardson in Arb confirms dh/dM > 0 at all interior grid points. Adaptive subdivision (depth 8) certifies continuity.
3. **p-direction correction**: Arb-certified |dh/dp| at M=20; correction = 0.000101.
4. **Certificate**: cert_bound = −0.009294 < 0 (margin 9.3×10⁻³).

### I_R log-concavity, d=2 (Prop 2.5(b))

1. **Quadrature**: 50-digit evaluation of G(y;p), G'(y;p), I_R(M,p,2) on 170-point grid.
2. **Ratio check**: G'(M)I_R(M)/G(M)² < 1 at all points (worst: 0.7553 at p=2.95, M=1.5; margin 24.5%).

### Dimensional boundary (Sec 7)

Bisection on p with golden-section search for R_peak in M (30-digit mpmath).
Key finding: for d ≥ 6, R_peak > 1 for all p ∈ (d, d+1); d = 5 is the unique transitional case.

---

## Proof status

The certificates are **semi-rigorous**:

- **Certified (Arb):** All post-quadrature arithmetic.
- **High-precision heuristic (mpmath):** Integral evaluations (50 digits, heuristic error estimates).
- **Upgrade path:** Replace `mpmath` quadrature with Arb-native `acb_calc_integrate` for full rigour.

---

## Citation

```bibtex
@article{Loven2026honest,
  author  = {Lov\'{e}n, Lauri},
  title   = {Honest Reporting in Scored Oversight: The {True-KL}$_0$ Property
             via the {Pr\'{e}kopa} Principle},
  journal = {Annals of Applied Probability},
  year    = {2026},
  note    = {Submitted}
}

@software{Loven2026certs,
  author  = {Lov\'{e}n, Lauri},
  title   = {true-kl0-certificates},
  year    = {2026},
  doi     = {10.5281/zenodo.19435617},
  url     = {https://doi.org/10.5281/zenodo.19435617}
}
```

## License

MIT
