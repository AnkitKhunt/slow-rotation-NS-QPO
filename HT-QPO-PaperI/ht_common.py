"""
ht_common.py

Shared helpers for the Hartle-Thorne QPO code base (new in v3).

    find_max_mass(eos)                  TOV maximum mass (cached)
    solve_rho_c_for_mass(eos, M, spin)  stable-branch rho_c giving the TOTAL
                                        (rotating) mass M = M_TOV + delta M
    solve_ht_mass(eos_file, M, spin)    cached HTModel at fixed total mass
    inner_edge(model)                   (r_in, 'isco' | 'surface')
    kerr_isco_over_M(j)                 exact prograde Kerr ISCO (BL radius)
    qpo_interpolators(model)            splines nu_phi(r), nu_per(r), ...
    precompute_grid(...)                (M, spin) -> j, q, R_eq, ... table
    convert_rmf_table(...)              MeV fm^-3 tables -> cgs columns

Code fixes 2, 4, 7, 9 of the paper plan live here; every plotting script
now uses solve_ht_mass instead of its own private bisection.

Mass convention: "mass" everywhere means the gravitational mass of the
ROTATING star, M_rot = M_TOV(rho_c) + delta M(rho_c, spin).  At spin = 0
this is the TOV mass.
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

from eos_loader import EOS
from tov import TOV
from ht_full import HTModel, M_SUN_GEO_KM
from qpo import qpo_at_radius, find_inner_edge_km

# ---------------------------------------------------------------------
# Unit conversions for relativistic mean-field (RMF) EOS tables
# ---------------------------------------------------------------------
MEV_FM3_TO_G_CM3 = 1.782662e12     # energy density / c^2
MEV_FM3_TO_DYN_CM2 = 1.602177e33   # pressure
FM3_TO_G_CM3_REST = 1.660539e15    # n_B [fm^-3] * atomic mass unit


# =====================================================================
# Maximum mass and stable-branch rho_c(M)
# =====================================================================
_MMAX_CACHE = {}


def find_max_mass(eos, rho_lo=1.0e14, rho_hi=None, n_scan=80):
    """
    Scan rho_c for the TOV maximum mass.
    Returns (rho_c_at_Mmax, M_max [Msun], hit_table_edge).
    """
    if rho_hi is None:
        rho_hi = eos.rho[-1]
    key = (id(eos), float(rho_lo), float(rho_hi), int(n_scan))
    if key in _MMAX_CACHE:
        return _MMAX_CACHE[key]
    tov = TOV(eos)
    rhos = np.logspace(np.log10(rho_lo), np.log10(rho_hi), n_scan)
    M_max, rho_at_max = -1.0, rhos[0]
    for rc in rhos:
        try:
            tov.solve(rc)
        except Exception:
            continue
        if tov.M_solar > M_max:
            M_max, rho_at_max = tov.M_solar, rc
    # refine the turning point by golden-section search on [prev, next]
    i = int(np.argmin(np.abs(rhos - rho_at_max)))
    a, b = rhos[max(i - 1, 0)], rhos[min(i + 1, len(rhos) - 1)]
    gr = 0.5 * (np.sqrt(5.0) - 1.0)

    def M_of(rc):
        try:
            tov.solve(rc)
            return tov.M_solar
        except Exception:
            return -1.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    Mc, Md = M_of(c), M_of(d)
    for _ in range(25):
        if Mc > Md:
            b, d, Md = d, c, Mc
            c = b - gr * (b - a)
            Mc = M_of(c)
        else:
            a, c, Mc = c, d, Md
            d = a + gr * (b - a)
            Md = M_of(d)
    rho_ref = 0.5 * (a + b)
    M_ref = M_of(rho_ref)
    if M_ref > M_max:
        M_max, rho_at_max = M_ref, rho_ref
    hit_edge = rho_at_max > 0.95 * rho_hi
    _MMAX_CACHE[key] = (rho_at_max, M_max, hit_edge)
    return _MMAX_CACHE[key]


def _rho_c_tov(eos, M_target, rho_lo, rho_at_max, tol):
    """Bisection on the stable branch for the TOV (non-rotating) mass."""
    tov = TOV(eos)
    lo, hi = rho_lo, rho_at_max
    mid = 0.5 * (lo + hi)
    for _ in range(100):
        mid = np.sqrt(lo * hi)                  # bisect in log rho
        try:
            tov.solve(mid)
            M = tov.M_solar
        except Exception:
            hi = mid
            continue
        if abs(M - M_target) / M_target < tol:
            return mid
        if M < M_target:
            lo = mid
        else:
            hi = mid
    return mid


def solve_rho_c_for_mass(eos, M_target, spin_hz=0.0, rho_lo=1.0e14,
                         rho_hi=None, tol=1e-5, verbose=True,
                         include_l0=True, return_model=False):
    """
    Central density of the STABLE-branch star whose total gravitational
    mass equals M_target [Msun] at spin frequency spin_hz [Hz].

    spin_hz = 0 (or include_l0=False): plain TOV bisection.
    spin_hz > 0: secant iteration in ln(rho_c) on HTModel.M_solar
    (= M_TOV + delta M), started from the TOV solution.

    Raises ValueError if M_target exceeds what the EOS supports.
    """
    rho_at_max, M_max, hit_edge = find_max_mass(eos, rho_lo, rho_hi)
    if verbose:
        print(f"  TOV M_max = {M_max:.4f} Msun at rho_c = {rho_at_max:.3e}")
    if hit_edge:
        print(f"  [warning] maximum mass at the edge of the EOS table "
              f"(rho_max = {eos.rho[-1]:.3e}); M_max may be a table "
              f"artefact.")

    rotating = include_l0 and spin_hz > 0.0
    if M_target > M_max and not rotating:
        raise ValueError(
            f"M = {M_target:.4f} Msun exceeds the TOV maximum mass "
            f"({M_max:.4f} Msun at rho_c = {rho_at_max:.3e} g/cm^3).")

    rho0 = _rho_c_tov(eos, min(M_target, M_max * 0.9999), rho_lo,
                      rho_at_max, tol)
    if not rotating:
        if verbose:
            print(f"  rho_c = {rho0:.5e} g/cm^3 (TOV)")
        return (rho0, HTModel(eos, rho0, spin_hz, include_l0)) \
            if return_model else rho0

    # --- secant on the rotating mass ---------------------------------
    def f(lnrho):
        m = HTModel(eos, np.exp(lnrho), spin_hz, include_l0)
        return m.M_solar - M_target, m

    x0 = np.log(rho0)
    f0, m0 = f(x0)
    if abs(f0) / M_target < tol:
        return (rho0, m0) if return_model else rho0
    # TOV slope dM/dln rho at rho0 for the first step
    tov = TOV(eos)
    tov.solve(rho0 * 1.01)
    M1 = tov.M_solar
    tov.solve(rho0)
    slope = (M1 - tov.M_solar) / np.log(1.01)
    x1 = x0 - f0 / slope if slope > 0 else x0 - 0.02
    x1 = min(x1, np.log(rho_at_max))
    f1, m1 = f(x1)
    for _ in range(25):
        if abs(f1) / M_target < tol:
            break
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = min(x2, np.log(rho_at_max))
        x0, f0 = x1, f1
        x1 = x2
        f1, m1 = f(x1)
    if abs(f1) / M_target > 10 * tol:
        raise ValueError(
            f"no stable rotating star with M = {M_target:.4f} Msun at "
            f"{spin_hz:.0f} Hz on this EOS (closest: {m1.M_solar:.4f}).")
    rho = float(np.exp(x1))
    if verbose:
        print(f"  rho_c = {rho:.5e} g/cm^3  (M_TOV = {m1.M0_solar:.5f}, "
              f"dM = {m1.dM_geo / M_SUN_GEO_KM:+.5f}, "
              f"M_rot = {m1.M_solar:.5f} Msun)")
    return (rho, m1) if return_model else rho


# =====================================================================
# Cached model at fixed total mass  (used by all plotting scripts)
# =====================================================================
_EOS_CACHE = {}
_HT_CACHE = {}


def get_eos(eos_file):
    e = _EOS_CACHE.get(eos_file)
    if e is None:
        e = EOS(eos_file)
        _EOS_CACHE[eos_file] = e
    return e


def solve_ht_mass(eos_file, mass, spin, include_l0=True, verbose=False):
    """
    HTModel of total mass `mass` [Msun] at `spin` [Hz] for EOS file
    `eos_file`, on the stable branch.  Cached.  Returns None on failure.
    """
    key = (eos_file, round(float(mass), 6), round(float(spin), 6),
           bool(include_l0))
    if key in _HT_CACHE:
        return _HT_CACHE[key]
    try:
        eos = get_eos(eos_file)
        _, model = solve_rho_c_for_mass(eos, mass, spin, verbose=verbose,
                                        include_l0=include_l0,
                                        return_model=True)
    except Exception as e:
        print(f"    [warning] {os.path.basename(str(eos_file))} "
              f"M={mass} spin={spin}: {e}")
        model = None
    _HT_CACHE[key] = model
    return model


# =====================================================================
# Inner edge, Kerr reference
# =====================================================================
def inner_edge(model):
    """(r_in [km], 'isco' | 'surface') using the equatorial surface."""
    return find_inner_edge_km(model.M_geo, model.j, model.q,
                              model.r_surface)


def kerr_isco_over_M(j):
    """
    Exact prograde Kerr ISCO in Boyer-Lindquist radius, r/M
    (Bardeen, Press & Teukolsky 1972).  For small j:
    6 - 4 sqrt(2/3) j - (7/18) j^2 + ...
    """
    a = np.asarray(j, dtype=float)
    z1 = 1.0 + np.cbrt(1.0 - a * a) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


# =====================================================================
# Sampler-ready interface
# =====================================================================
def qpo_interpolators(model, r_max_km=None, n=300):
    """
    Tabulate the RPM frequencies from the inner edge outward and return
    CubicSpline interpolants (Hz versus r in km):

        out['nu_phi'](r), out['nu_r'](r), out['nu_theta'](r),
        out['nu_per'](r), out['nu_nod'](r)

    plus out['r'] (grid), out['r_in'], out['edge'] ('isco'|'surface'),
    and the star's (M_solar, j, q, R_eq_km).  Use these inside a Bilby
    likelihood so that no ODE is solved per likelihood call.
    """
    r_in, kind = inner_edge(model)
    if r_max_km is None:
        r_max_km = max(4.0 * r_in, 60.0)
    rr = np.linspace(r_in * (1.0 + 1e-6), r_max_km, n)
    rows = [qpo_at_radius(model.M_geo, model.j, model.q, r) for r in rr]
    out = {'r': rr, 'r_in': r_in, 'edge': kind,
           'M_solar': model.M_solar, 'j': model.j, 'q': model.q,
           'R_eq_km': model.R_eq_km}
    for k_out, k_in in (('nu_phi', 'nu_phi'), ('nu_r', 'nu_r'),
                        ('nu_theta', 'nu_theta'), ('nu_per', 'nu_peri'),
                        ('nu_nod', 'nu_nod')):
        y = np.array([row[k_in] for row in rows])
        out[k_out] = CubicSpline(rr, y)
        out[k_out + '_arr'] = y
    return out


def precompute_grid(eos_file, masses, spins, outfile=None, verbose=True):
    """
    Table of stellar properties on a (mass, spin) grid for one EOS, for
    fast interpolation in fits.  Columns:
        M, spin, rho_c, M0, dM_over_M, R, R_eq, R_eq_circ, j, q,
        q_over_j2, I_geo, r_in, is_isco
    Saved as .npz (key 'table' + 'columns') if outfile is given.
    """
    cols = ['M', 'spin', 'rho_c', 'M0', 'dM_over_M', 'R', 'R_eq',
            'R_eq_circ', 'j', 'q', 'q_over_j2', 'I_geo', 'r_in', 'is_isco']
    rows = []
    for M in masses:
        for s in spins:
            m = solve_ht_mass(eos_file, M, s)
            if m is None:
                continue
            r_in, kind = inner_edge(m)
            rows.append([m.M_solar, s, m.rho_c, m.M0_solar,
                         m.dM_geo / m.M0_geo, m.R_km, m.R_eq_km,
                         m.R_eq_circ_km, m.j, m.q,
                         m.q / m.j ** 2 if m.j != 0 else np.nan,
                         m.I_geo, r_in, 1.0 if kind == 'isco' else 0.0])
            if verbose:
                print(f"  M={m.M_solar:.3f} spin={s:5.0f}  j={m.j:.4f}  "
                      f"q/j^2={rows[-1][10]:+.3f}  r_in={r_in:.3f} ({kind})")
    table = np.array(rows)
    if outfile:
        np.savez(outfile, table=table, columns=np.array(cols))
        if verbose:
            print(f"  Saved: {outfile}")
    return table, cols


# =====================================================================
# RMF / CompOSE-style table converter
# =====================================================================
def convert_rmf_table(infile, outfile, col_nb=0, col_eps=1, col_P=2,
                      skip_header=0, eps_includes_rest_mass=True):
    """
    Convert an EOS table with columns
        n_B [fm^-3],  eps [MeV fm^-3],  P [MeV fm^-3]
    (column indices selectable) into the RNS format read by eos_loader.EOS
    and by the RNS code of Paper II:

        line 1:  number of rows
        then:    eps/c^2 [g cm^-3]   P [dyn cm^-2]   H [cm^2 s^-2]   n_B [cm^-3]

    eps is the TOTAL energy density (rest mass included); pass
    eps_includes_rest_mass=False if the table gives the internal energy
    only.  H = c^2 Int dP / (eps c^2 + P) is the enthalpy column of RNS
    (only eps and P enter the TOV and HT equations).  Rows are sorted by
    density; non-monotonic points are dropped.
    """
    a = np.loadtxt(infile, skiprows=skip_header, ndmin=2)
    nb, eps, P = a[:, col_nb], a[:, col_eps], a[:, col_P]
    if not eps_includes_rest_mass:
        eps = eps + nb * 931.494                     # add m_u c^2 n_B
    o = np.argsort(nb)
    nb, eps, P = nb[o], eps[o], P[o]
    keep = np.concatenate([[True], (np.diff(P) > 0) & (np.diff(eps) > 0)])
    nb, eps, P = nb[keep], eps[keep], P[keep]
    eps_cgs = eps * MEV_FM3_TO_G_CM3                  # eps/c^2 [g cm^-3]
    P_cgs = P * MEV_FM3_TO_DYN_CM2                    # [dyn cm^-2]
    c2 = 2.99792458e10 ** 2
    w = 1.0 / (eps_cgs * c2 + P_cgs)                  # 1/(eps c^2 + P)
    H = np.concatenate([[1.0], 1.0 + c2 * np.cumsum(
        0.5 * (w[1:] + w[:-1]) * np.diff(P_cgs))])
    with open(outfile, 'w') as f:
        f.write(f'{len(nb)}\n')
        for row in zip(eps_cgs, P_cgs, H, nb * 1e39):
            f.write('%.6e %.6e %.15e %.15e\n' % row)
    print(f"  Wrote {outfile} (RNS format): {len(nb)} rows, "
          f"n_B = {nb[0]:.3e} .. {nb[-1]:.3e} fm^-3, "
          f"eps/c^2 = {eps_cgs[0]:.3e} .. {eps_cgs[-1]:.3e} g/cm^3")
    return outfile


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='ht_common utilities')
    sub = p.add_subparsers(dest='cmd')
    c = sub.add_parser('convert', help='convert an (n_B, eps, P) table in '
                       'MeV fm^-3 to the RNS format of eos/')
    c.add_argument('infile')
    c.add_argument('outfile')
    c.add_argument('--cols', type=int, nargs=3, default=[0, 1, 2],
                   help='column indices of n_B, eps, P')
    c.add_argument('--skip', type=int, default=0,
                   help='header lines to skip')
    c.add_argument('--no-rest-mass', action='store_true',
                   help='eps column excludes the rest-mass energy')
    g = sub.add_parser('grid', help='precompute a (M, spin) grid')
    g.add_argument('--eos', default='eos/eosAU')
    g.add_argument('--masses', type=float, nargs='+',
                   default=[1.4, 1.6, 1.8, 2.0])
    g.add_argument('--spins', type=float, nargs='+',
                   default=[0, 200, 363, 400, 581, 600])
    g.add_argument('--out', default='ht_grid.npz')
    args = p.parse_args()
    if args.cmd == 'convert':
        convert_rmf_table(args.infile, args.outfile, *args.cols,
                          skip_header=args.skip,
                          eps_includes_rest_mass=not args.no_rest_mass)
    elif args.cmd == 'grid':
        precompute_grid(args.eos, args.masses, args.spins, args.out)
    else:
        p.print_help()
