#!/usr/bin/env python3
"""
run_qpo.py

QPO pipeline using the full Hartle-Thorne interior solution.

Fix the star by ONE of:
    --rho_c   central density        [g/cm^3]
    --mass    gravitational mass     [M_sun]
    --radius  circumferential radius [km]

Optional:
    --spin    spin frequency [Hz]    (default 400)
    --eos     EOS file               (default eos/eosAU)
    --outdir  output directory       (default qpo_out)
    --plots   also generate PNG plots via plots.py

v3: --mass is the TOTAL mass of the rotating star (M_TOV + delta M); the
central density is found by ht_common.solve_rho_c_for_mass on the stable
branch.  The inner edge is max(R_eq, r_ISCO) in the exterior coordinate;
the "ISCO" row (nu_r = 0) is written only if a real ISCO lies outside the
star, otherwise the first row is the stellar surface.  The nodal
precession is labelled nu_nod (it includes the quadrupole precession).
"""

import argparse
import os
import numpy as np

from eos_loader import EOS
from ht_full import HTModel
from qpo import (qpo_table, qpo_at_radius, find_inner_edge_km,
                 find_radius_for_nu2)
import ht_common


# =====================================================================
# Mass / radius target solvers
# =====================================================================
def find_max_mass(eos, rho_lo=1.0e14, rho_hi=None, n_scan=80):
    """TOV maximum mass: (rho_c_at_Mmax, M_max, hit_table_edge, rho_hi)."""
    rho_at_max, M_max, hit = ht_common.find_max_mass(eos, rho_lo, rho_hi,
                                                     n_scan)
    return rho_at_max, M_max, hit, (eos.rho[-1] if rho_hi is None
                                    else rho_hi)


def solve_rho_c_for_mass(eos, M_target, rho_lo=1.0e14, rho_hi=None,
                         tol=1e-5, verbose=True, spin_hz=0.0):
    """
    Stable-branch rho_c for TOTAL mass M_target at spin_hz (default 0:
    TOV mass, the v2 behaviour).  Thin wrapper of ht_common (code fix 2).
    """
    return ht_common.solve_rho_c_for_mass(eos, M_target, spin_hz,
                                          rho_lo=rho_lo, rho_hi=rho_hi,
                                          tol=tol, verbose=verbose)


def solve_rho_c_for_radius(eos, R_target,
                           rho_lo=1.0e14, rho_hi=None, tol=1e-5,
                           verbose=True):
    if rho_hi is None:
        rho_hi = eos.rho[-1]
    from tov import TOV
    tov = TOV(eos)
    rho_mid = 0.5 * (rho_lo + rho_hi)
    for _ in range(80):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        try:
            tov.solve(rho_mid)
            R = tov.R_km
        except Exception:
            rho_hi = rho_mid
            continue
        if abs(R - R_target) / R_target < tol:
            if verbose:
                print(f"  Converged: rho_c = {rho_mid:.4e}  "
                      f"R = {R:.3f} km")
            return rho_mid
        if R > R_target:
            rho_lo = rho_mid
        else:
            rho_hi = rho_mid
    return rho_mid


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eos',    default='eos/eosAU')
    p.add_argument('--rho_c',  type=float, default=None)
    p.add_argument('--mass',   type=float, default=None)
    p.add_argument('--radius', type=float, default=None)
    p.add_argument('--spin',   type=float, default=400.0)
    p.add_argument('--outdir', default='qpo_out')
    p.add_argument('--rmax',   type=float, default=25.0)
    p.add_argument('--npts',   type=int,   default=25)
    p.add_argument('--nu-upper', action='append', type=float, default=None)
    p.add_argument('--plots', action='store_true',
                   help='also generate PNG plots via plots.py')
    p.add_argument('--plot-npts', type=int, default=60,
                   help='M-R curve points if --plots is used')
    args = p.parse_args()

    n_fix = sum(x is not None for x in (args.rho_c, args.mass, args.radius))
    if n_fix == 0:
        args.mass = 1.8
        print("[default] using --mass 1.8")
    elif n_fix > 1:
        p.error("Specify at most one of --rho_c, --mass, --radius")

    if args.nu_upper is None:
        args.nu_upper = [500.0, 700.0, 900.0, 1050.0]

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Header ----
    print("=" * 72)
    print("  Full Hartle-Thorne QPO pipeline")
    print("=" * 72)
    print(f"  EOS             : {args.eos}")
    if args.rho_c is not None:
        print(f"  Central density : {args.rho_c:.4e} g/cm^3")
    if args.mass is not None:
        print(f"  Target mass     : {args.mass:.4f} M_sun")
    if args.radius is not None:
        print(f"  Target radius   : {args.radius:.4f} km")
    print(f"  Spin frequency  : {args.spin:.1f} Hz")
    print(f"  Output dir      : {args.outdir}")
    print("=" * 72)

    # ---- [1/6] Load EOS ----
    print("\n[1/6] Load EOS")
    eos = EOS(args.eos)
    print(f"      {eos}")

    # ---- [2/6] Determine central density ----
    print("\n[2/6] Determine central density")
    if args.rho_c is not None:
        rho_c = args.rho_c
        print(f"      Using user-supplied rho_c = {rho_c:.4e}")
    elif args.mass is not None:
        print(f"      Solving for rho_c such that M = {args.mass} M_sun ...")
        try:
            rho_c = solve_rho_c_for_mass(eos, args.mass,
                                         spin_hz=args.spin)
        except ValueError as e:
            p.error(str(e))
    else:
        print(f"      Solving for rho_c such that R = {args.radius} km ...")
        rho_c = solve_rho_c_for_radius(eos, args.radius)

    # ---- [3/6] Solve TOV ----
    print("\n[3/6] Solve TOV")
    from tov import TOV
    tov = TOV(eos)
    tov.solve(rho_c)
    print(f"      M_TOV   = {tov.M_solar:.4f} M_sun  (non-rotating, same rho_c)")
    print(f"      R_TOV   = {tov.R_km:.3f} km")
    if tov.M_solar < 1.2:
        print("      *** WARNING: M < 1.2 M_sun ***")

    # ---- [4/6] Hartle-Thorne rotation ----
    print("\n[4/6] Hartle-Thorne rotation parameters")
    model = HTModel(eos, rho_c, args.spin)
    print(f"      M_TOV   = {model.M0_solar:.5f} M_sun  (same rho_c, "
          f"no rotation)")
    print(f"      dM      = {model.dM_geo / 1.4766:+.5f} M_sun  "
          f"(dM/M = {model.dM_geo / model.M0_geo:+.3e})")
    print(f"      M_rot   = {model.M_solar:.5f} M_sun  (used in exterior)")
    print(f"      R_eq    = {model.R_eq_km:.4f} km (coordinate), "
          f"{model.R_eq_circ_km:.4f} km (circumferential)")
    print(f"      R_pol   = {model.R_pol_km:.4f} km")
    print(f"      M_geo   = {model.M_geo:.4f} km")
    print(f"      Omega   = {model.Omega * 2.99792458e5:.3f} rad/s")
    print(f"      J_geo   = {model.J_geo:+.6f} km^2")
    print(f"      I_geo   = {model.I_geo:+.6f} km^3")
    print(f"      Q_geo   = {model.Q_geo:+.6f} km^3")
    print(f"      j       = {model.j:+.6f}")
    print(f"      q       = {model.q:+.6f}")
    if model.j != 0:
        ratio = model.q / model.j**2
        print(f"      q/j^2   = {ratio:+.4f}   "
              f"(Kerr/BH: -1.0000; real NS more negative, "
              f"EOS-dependent)")

    # ---- [5/6] ISCO and QPO table ----
    print("\n[5/6] Find inner edge and build QPO table")
    r_isco, edge = find_inner_edge_km(model.M_geo, model.j, model.q,
                                      model.r_surface)
    if edge == 'isco':
        print(f"      r_ISCO  = {r_isco:.3f} km  "
              f"(= {r_isco / model.M_geo:.3f} M)  > R_eq = "
              f"{model.r_surface:.3f} km")
    else:
        print(f"      no ISCO outside the star: inner edge = R_eq = "
              f"{r_isco:.3f} km (= {r_isco / model.M_geo:.3f} M)")

    r_min = 1.02 * r_isco
    table = qpo_table(model.M_geo, model.j, model.q,
                      r_min, args.rmax, args.npts)

    isco_row = qpo_at_radius(model.M_geo, model.j, model.q, r_isco)
    isco_row['is_isco'] = True
    isco_row['edge'] = edge
    if edge == 'isco':
        isco_row['nu_r'] = 0.0
        isco_row['nu_peri'] = isco_row['nu_phi']
    for row in table:
        row['is_isco'] = False
        row['edge'] = ''
    table = [isco_row] + table

    W = 78
    print()
    print("-" * W)
    print("QPO Frequencies (Stella-Vietri / RPM)")
    print("-" * W)
    print(f"{'r(km)':<9}{'x=r/M':<9}{'nu_phi':<11}{'nu_r':<11}"
          f"{'nu_theta':<11}{'nu_peri':<11}{'nu_nod':<11}")
    print("-" * W)
    for row in table:
        flag = (f"  <-- {row['edge'].upper()}" if row.get('is_isco')
                else "")
        print(f"{row['r_km']:<9.2f}{row['x']:<9.3f}"
              f"{row['nu_phi']:<11.1f}{row['nu_r']:<11.1f}"
              f"{row['nu_theta']:<11.1f}{row['nu_peri']:<11.1f}"
              f"{row['nu_nod']:<11.1f}{flag}")
    print("-" * W)

    m_isco = isco_row
    print()
    print(f"Inner edge ({edge}): x = {r_isco / model.M_geo:.3f}  "
          f"({r_isco:.2f} km)")
    print()
    print(f"QPO frequencies at the inner edge ({edge}, Stella-Vietri):")
    print(f"  nu_2  (orbital)        = {m_isco['nu_phi']:8.1f} Hz")
    print(f"  nu_1  (periastron)     = {m_isco['nu_peri']:8.1f} Hz")
    print(f"  Dnu   (= nu_r)         = {m_isco['nu_r']:8.1f} Hz")
    print(f"  nu_th (vertical)       = {m_isco['nu_theta']:8.1f} Hz")
    print(f"  nu_nod (nodal prec.)   = {m_isco['nu_nod']:8.1f} Hz")
    print()
    print(f"  Range of nu_2          : "
          f"{min(row['nu_phi'] for row in table):.1f} - "
          f"{max(row['nu_phi'] for row in table):.1f} Hz")
    print(f"  Range of Delta_nu=nu_r : "
          f"{min(row['nu_r'] for row in table):.1f} - "
          f"{max(row['nu_r'] for row in table):.1f} Hz")
    print(f"  Range of nu_peri=nu_1  : "
          f"{min(row['nu_peri'] for row in table):.1f} - "
          f"{max(row['nu_peri'] for row in table):.1f} Hz")
    print(f"  Range of nu_nod        : "
          f"{min(row['nu_nod'] for row in table):.1f} - "
          f"{max(row['nu_nod'] for row in table):.1f} Hz")
    print()

    # ---- [5b/6] Match observed Sco X-1 nu_2 = 1000 Hz ----
    print("[5b/6] Match observed Sco X-1 nu_2 = 1000 Hz")
    r_match = find_radius_for_nu2(model.M_geo, model.j, model.q,
                                  1000.0,
                                  r_lo_km=r_isco, r_hi_km=args.rmax)
    if r_match > 0:
        m = qpo_at_radius(model.M_geo, model.j, model.q, r_match)
        print(f"      r where nu_2 = 1000 Hz : {r_match:.2f} km")
        print(f"      nu_2  = {m['nu_phi']:.1f} Hz")
        print(f"      nu_1  = {m['nu_peri']:.1f} Hz")
        print(f"      Dnu   = {m['nu_r']:.1f} Hz")
        print(f"      nu_nod= {m['nu_nod']:.1f} Hz")

    # ---- Additional observed frequencies ----
    print("\nObserved nu_2 matching (from --nu-upper):")
    for nu_obs in args.nu_upper:
        r_match = find_radius_for_nu2(model.M_geo, model.j, model.q,
                                      nu_obs,
                                      r_lo_km=r_isco, r_hi_km=args.rmax)
        if r_match <= 0:
            continue
        m = qpo_at_radius(model.M_geo, model.j, model.q, r_match)
        print(f"      nu_2 = {nu_obs:6.1f} Hz  ->  r = {r_match:6.2f} km   "
              f"nu_r = {m['nu_r']:6.1f}   nu_nod = {m['nu_nod']:6.1f}")

    # ---- Save table ----
    out = os.path.join(args.outdir, 'qpo_table.dat')
    with open(out, 'w') as f:
        f.write(f"# Full Hartle-Thorne QPO table\n")
        f.write(f"# EOS   = {args.eos}\n")
        f.write(f"# rho_c = {rho_c:.6e} g/cm^3\n")
        f.write(f"# M     = {model.M_solar:.5f} Msun (rotating, total)\n")
        f.write(f"# M_TOV = {model.M0_solar:.5f} Msun\n")
        f.write(f"# R     = {model.R_km:.5f} km (TOV)\n")
        f.write(f"# R_eq  = {model.R_eq_km:.5f} km (coord), "
                f"{model.R_eq_circ_km:.5f} km (circ)\n")
        f.write(f"# J_geo = {model.J_geo:+.6f} km^2\n")
        f.write(f"# I_geo = {model.I_geo:+.6f} km^3\n")
        f.write(f"# Q_geo = {model.Q_geo:+.6f} km^3\n")
        f.write(f"# j     = {model.j:+.6f}\n")
        f.write(f"# q     = {model.q:+.6f}\n")
        if model.j != 0:
            f.write(f"# q/j^2 = {model.q / model.j**2:+.5f}\n")
        f.write(f"# r_in  = {r_isco:.5f} km ({edge})\n")
        f.write("#\n")
        f.write(f"{'r_km':<9}{'x':<9}{'nu_phi':<11}{'nu_r':<11}"
                f"{'nu_theta':<11}{'nu_peri':<11}{'nu_nod':<11}\n")
        for row in table:
            flag = row['edge'].upper() if row.get('is_isco') else ""
            f.write(f"{row['r_km']:<9.3f}{row['x']:<9.4f}"
                    f"{row['nu_phi']:<11.3f}{row['nu_r']:<11.3f}"
                    f"{row['nu_theta']:<11.3f}{row['nu_peri']:<11.3f}"
                    f"{row['nu_nod']:<11.3f}{flag}\n")
    print(f"\n      Saved: {out}")

    # ---- [6/6] Optional plots ----
    if args.plots:
        print("\n[6/6] Generating plots ...")
        make_qpo_plots(eos, model, table, r_isco, rho_c, args)
    else:
        print("\n[6/6] Plots skipped (use --plots to enable)")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


# =====================================================================
# Plot wrapper
# =====================================================================
def make_qpo_plots(eos, model, table, r_isco, rho_c, args):
    from tov import TOV
    from veff import (HartleThorneMetric, veff, dveff_dr, d2veff_dr2,
                      circular_omega)
    from plots import make_all_plots

    source_name = (f"{os.path.basename(args.eos)}  "
                   f"M={model.M_solar:.2f}Msun  spin={args.spin:.0f}Hz")

    r_veff = np.linspace(model.r_surface, args.rmax, 200)
    metric = HartleThorneMetric(model.M_geo, model.j, model.q)

    veff_vals   = np.array([veff(metric, r) for r in r_veff])
    d1veff_vals = np.array([dveff_dr(metric, r) for r in r_veff])
    d2veff_vals = np.array([d2veff_dr2(metric, r) for r in r_veff])
    omega_vals  = np.array([circular_omega(metric, r) for r in r_veff])

    rho_lo, rho_hi = eos.rho[0] * 2, eos.rho[-1]
    rho_scan = np.logspace(np.log10(rho_lo), np.log10(rho_hi),
                           args.plot_npts)
    tov_mr = TOV(eos)
    M_arr, R_arr = tov_mr.mass_radius_curve(rho_scan)

    make_all_plots(
        table, eos, M_arr, R_arr, source_name, args.outdir,
        r_veff=r_veff, veff_vals=veff_vals,
        d1veff_vals=d1veff_vals, d2veff_vals=d2veff_vals,
        omega_vals=omega_vals,
        r_isco=r_isco, r_star=model.r_surface)


if __name__ == '__main__':
    main()