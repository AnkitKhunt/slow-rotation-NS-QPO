#!/usr/bin/env python3
"""
paper_checks.py -- numbers quoted in the text of Paper I that are not read
off a figure or a table.

  1. Accuracy of the HT metric: HT frequencies with q = -j^2 against the
     exact Kerr frequencies at the same orbital frequency (Sec. VII D).
  2. The Kerr-equivalent spin and moment of inertia for the 4U 1728-34
     triplet nu_U = 1098.4 Hz with IOPB-I, M = 1.922 Msun (Sec. VI B).
  3. Bayesian evidence of the EOS fits to 4U 1728-34 (Sec. VI B).

Units: G = c = 1, lengths in km, x = r/M, frequencies in Hz.

    python3 paper_checks.py
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import brentq

from qpo import kerr_ht_angular, find_radius_for_nu2, qpo_at_radius
from qpo_fit import StarTable, M_SUN_KM

C_KM_S = 2.99792458e5
I45 = 1.3466e43 / 1e45          # I [km^3] -> [1e45 g cm^2]


# ---------------------------------------------------------------------
# Exact Kerr, Boyer-Lindquist radius x = r/M, prograde orbits
# ---------------------------------------------------------------------
def kerr_angular(x, a):
    """(Omega_phi, Omega_r, Omega_theta) in units of 1/M."""
    Op = 1.0 / (x ** 1.5 + a)
    fr = 1 - 6 / x + 8 * a * x ** -1.5 - 3 * a * a / x ** 2
    ft = 1 - 4 * a * x ** -1.5 + 3 * a * a / x ** 2
    return Op, Op * np.sqrt(max(fr, 0.0)), Op * np.sqrt(max(ft, 0.0))


def kerr_at_Omega(Op, a):
    """Kerr frequencies at the radius where Omega_phi = Op (1/M units)."""
    x = brentq(lambda y: kerr_angular(y, a)[0] - Op, 1.5, 500.0, xtol=1e-13)
    return x, kerr_angular(x, a)


# ---------------------------------------------------------------------
# 1. HT (q = -j^2) against exact Kerr
# ---------------------------------------------------------------------
def check_ht_vs_kerr(js=(0.1, 0.178, 0.2), x_min=6.0, x_max=20.0):
    """
    Compare at equal orbital frequency, which is coordinate independent
    (HT and Boyer-Lindquist radii differ at O(j^2)).  Returns, for each j,
    the largest differences of nu_r/nu_phi and nu_theta/nu_phi and the
    largest relative difference of nu_nod for x_min <= x <= x_max.
    """
    print('1. HT (q = -j^2) against exact Kerr at equal nu_phi, '
          f'{x_min} <= x <= {x_max}')
    out = {}
    for a in js:
        dr = dt = dn = 0.0
        for x in np.linspace(x_min, x_max, 80):
            Op, Or, Ot = kerr_ht_angular(x, a, -a * a)
            _, (Ok, Ork, Otk) = kerr_at_Omega(Op, a)
            dr = max(dr, abs(Or / Op - Ork / Ok))
            dt = max(dt, abs(Ot / Op - Otk / Ok))
            dn = max(dn, abs((Op - Ot) - (Ok - Otk)) / (Ok - Otk))
        out[a] = (dr, dt, dn)
        print(f'   j = {a:5.3f}:  d(nu_r/nu_phi) = {dr:.1e}   '
              f'd(nu_th/nu_phi) = {dt:.1e}   d(nu_nod)/nu_nod = {dn:.1e}')
    return out


# ---------------------------------------------------------------------
# 2. Kerr-equivalent spin for the 4U 1728-34 triplet
# ---------------------------------------------------------------------
def check_kerr_equivalent(eos='eos/IOPB', M=1.922, nu_s=363.0,
                          nu_U=1098.4):
    st = StarTable(eos).star(M, nu_s)
    j, q, I = st['j'], st['q'], st['I'] * I45
    Mkm = M * M_SUN_KM
    conv = C_KM_S / (2 * np.pi * Mkm)               # (1/M) -> Hz
    r0 = find_radius_for_nu2(Mkm, j, q, nu_U, r_lo_km=1.01 * st['R_eq'])
    f = qpo_at_radius(Mkm, j, q, r0)
    Op = nu_U / conv

    def nod_kerr(a):
        _, (Ok, _, Otk) = kerr_at_Omega(Op, a)
        return (Ok - Otk) * conv

    jK = brentq(lambda a: nod_kerr(a) - f['nu_nod'], 1e-4, 0.6, xtol=1e-10)
    x = r0 / Mkm
    nu_LT = j * Mkm ** 2 / (np.pi * r0 ** 3) * C_KM_S   # J/(pi r^3), Eq. (nuLT)
    nu_Q = 1.5 * q / x ** 2 * f['nu_phi']               # (3/2)(q/x^2) nu_phi
    print(f'2. 4U 1728-34 with {os.path.basename(eos)}, M = {M} Msun, '
          f'nu_s = {nu_s:.0f} Hz:  j = {j:.4f}, Qbar = {-q / j**2:.3f}, '
          f'I = {I:.3f}e45 g cm^2')
    print(f'   r0 = {r0:.3f} km (x = {x:.3f}):  nu_nod(HT) = '
          f'{f["nu_nod"]:.3f} Hz,  nu_per = {f["nu_peri"]:.2f} Hz,  '
          f'nu_r = {f["nu_r"]:.2f} Hz')
    print(f'   weak field: nu_LT = {nu_LT:.2f} Hz,  nu_Q = {nu_Q:.2f} Hz')
    Oh, _, Oth = kerr_ht_angular(x, j, -j * j)
    _, (Ok, _, Otk) = kerr_at_Omega(Oh, j)
    d2 = 2 * f['nu_nod'] * abs((Oh - Oth) / (Ok - Otk) - 1)
    print(f'   neglected O(j^3) terms (HT vs Kerr, q = -j^2, this orbit): '
          f'd(2 nu_nod) = {d2:.3f} Hz')
    print(f'   Kerr, same j:  nu_nod = {nod_kerr(j):.3f} Hz')
    print(f'   Kerr matching nu_nod: j = {jK:.4f} ({100 * (jK / j - 1):+.1f}%)'
          f'  ->  I = {I * jK / j:.3f}e45 g cm^2 (same M and nu_s)')
    return dict(j=j, jK=jK, I=I, IK=I * jK / j, nu_nod=f['nu_nod'],
                r0=r0, x=x)


# ---------------------------------------------------------------------
# 3. Evidence of the EOS fits to 4U 1728-34
# ---------------------------------------------------------------------
def check_evidence(src='4U1728-34'):
    print(f'3. Evidence of the EOS fits to {src} (grid quadrature)')
    for label in ('frac2', 'kHz'):
        rows = []
        for p in sorted(glob.glob(f'grid_fit_out/eos/{src}/*_{label}.json')):
            d = json.load(open(p))
            eta = d.get('eta_LF') or d.get('eta')
            rows.append((d['eos'], d['lnZ'], eta['median'] if eta else None))
        if not rows:
            print(f'   {label}: no results (run run_all.py step 4)')
            continue
        z0 = max(r[1] for r in rows)
        for eos, lnZ, e in rows:
            s = f'   {label:5s} {eos:10s} lnZ = {lnZ:9.2f}  dlnZ = {z0 - lnZ:5.2f}'
            if label == 'frac2' and e is not None:
                s += f'   eta_LF = {100 * 10 ** e:4.1f}%'
            print(s)


if __name__ == '__main__':
    check_ht_vs_kerr()
    print()
    check_kerr_equivalent()
    print()
    check_evidence()
