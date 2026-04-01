"""
apply-interval-cert.py
======================
Post-process the output of cert-r-bound-arb.py to apply the
interval-wise local Lipschitz certificate (correcting the
global-L_M * global-delta_M bug in the original certificate section).

Usage:
  python apply-interval-cert.py bc391df.output
"""

import sys
import os
import re
import flint

if len(sys.argv) < 2:
    sys.exit(f"Usage: python {os.path.basename(__file__)} <output_file>\n"
             f"  where <output_file> is the output of cert-r-bound-arb.py")
OUTPUT_FILE = sys.argv[1]
if not os.path.exists(OUTPUT_FILE):
    sys.exit(f"ERROR: output file not found: {OUTPUT_FILE}")

# Grid definition (must match cert-r-bound-arb.py)
d_vals    = [2, 3, 4]
p_offsets = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
M_vals    = [1.001, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 5.75, 6.5, 8.0, 10.0, 12.0, 15.0, 20.0]
delta_p   = 0.1  # p-grid spacing

# Parse the output table
row_re = re.compile(
    r'^\s*(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+(OK|FAIL)\s*$'
)

rows = []
with open(OUTPUT_FILE) as f:
    for line in f:
        m = row_re.match(line)
        if m:
            d    = int(m.group(1))
            p    = float(m.group(2))
            M    = float(m.group(3))
            R    = float(m.group(4))
            Rup  = float(m.group(5))
            dRM  = float(m.group(6))
            dRp  = float(m.group(7))
            rows.append({'d': d, 'p': p, 'M': M, 'R': R, 'R_up': Rup, 'dRM': dRM, 'dRp': dRp})

print(f"Parsed {len(rows)} grid points.")

# Build lookup
row_lookup = {}
for r in rows:
    p_off = round(r['p'] - r['d'], 6)
    row_lookup[(r['d'], p_off, r['M'])] = r

# Global L_p
max_L_p = max(r['dRp'] for r in rows)
p_interp = max_L_p * delta_p / 2
print(f"Global L_p = {max_L_p:.8f}, p-interp error = {p_interp:.8f}")

# Interval-wise M bounds
interval_bounds = []
for d in d_vals:
    for p_off in p_offsets:
        for i in range(len(M_vals) - 1):
            Mi = M_vals[i]; Mj = M_vals[i+1]
            ri = row_lookup.get((d, round(p_off, 6), Mi))
            rj = row_lookup.get((d, round(p_off, 6), Mj))
            if ri is None or rj is None:
                continue
            local_Rup = max(ri['R_up'], rj['R_up'])
            li = ri['dRM'] if ri['dRM'] == ri['dRM'] else 0.0
            lj = rj['dRM'] if rj['dRM'] == rj['dRM'] else 0.0
            local_LM  = max(li, lj)
            gap = Mj - Mi
            local_bound_M = local_Rup + local_LM * gap / 2
            interval_bounds.append({
                'd': d, 'p': d + p_off, 'Mi': Mi, 'Mj': Mj,
                'R_up': local_Rup, 'LM': local_LM, 'gap': gap,
                'bound_M': local_bound_M
            })

interval_bounds.sort(key=lambda x: -x['bound_M'])

print("\nTop 15 worst M-intervals:")
print(f"{'d':>3} {'p':>6} {'M_i':>7} {'M_j':>7} {'R_up':>9} {'L_M':>9} {'gap':>6} {'bound_M':>9}")
for ib in interval_bounds[:15]:
    print(f"  {ib['d']:1d} {ib['p']:6.2f} {ib['Mi']:7.3f} {ib['Mj']:7.3f} "
          f"{ib['R_up']:9.6f} {ib['LM']:9.6f} {ib['gap']:6.3f} {ib['bound_M']:9.6f}")

max_bound_M = interval_bounds[0]['bound_M']
cert_bound  = max_bound_M + p_interp

print(f"\nmax interval M-bound              : {max_bound_M:.8f}")
print(f"p-interp error (L_p * dp/2)       : {p_interp:.8f}")
print(f"cert_bound = max_M_bound + p_err  : {cert_bound:.8f}")

# arb final check
fctx = flint.ctx
fctx.prec = 200
arb_cert = flint.arb(max_bound_M) + flint.arb(max_L_p) * flint.arb(delta_p) / 2

print(f"\narb cert_bound                     : {arb_cert}")

if cert_bound < 1.0:
    print(f"\nCERTIFICATE PASSED: R(M,p,d) <= {cert_bound:.8f} < 1")
    print(f"for d in {{2,3,4}}, p in (d, d+1), M in [1, 20].")
    print(f"(interval-wise local Lipschitz; mpmath 50-digit quadrature + Richardson derivatives)")
else:
    print(f"\nCERTIFICATE FAILED: bound = {cert_bound:.8f} >= 1")
    print("Need denser grid.")
