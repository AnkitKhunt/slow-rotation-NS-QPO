"""
qpo.py

QPO frequencies in the relativistic precession model (RPM), computed
from the SAME Hartle-Thorne exterior metric that orbits.py / veff.py use.

    nu_2     = nu_phi
    nu_1     = nu_peri = nu_phi - nu_r
    Delta_nu = nu_r
    nu_nod   = nu_phi - nu_theta        (HBO ~ 2 nu_nod)

v3: the nodal precession is returned as 'nu_nod'.  It is NOT pure
Lense-Thirring (the spin-induced quadrupole drives a retrograde classical
precession), so the old key 'nu_LT' is kept only as an alias of 'nu_nod'
for backward compatibility.  New helper find_inner_edge_km() tells whether
the inner edge of the disc is the ISCO or the stellar surface.

v2 CHANGE
---------
The old version used closed-form "Kerr + ad-hoc q" expressions
(fac_r = 1 - 6/x + 8j/x^1.5 - 3j^2/x^2 - q/x^3, Omega_phi without any q).
Those do not have the right quadrupole scaling (a Newtonian quadrupole
enters at q/x^2, not q/x^3), Omega_phi had no quadrupole term at all, and
they were not the epicyclic frequencies of the metric used for the
particle orbits.  Here the three frequencies are obtained exactly (to
O(j^2, q)) from the metric:

    Omega_phi  : circular-orbit condition  g_tt' + 2 Omega g_tphi' + Omega^2 g_phiphi' = 0
    kappa_r^2  = +H_rr     / (2 g_rr)     (proper time, E and L fixed)
    kappa_th^2 = +H_thth   / (2 g_thth)
    H(r,th;E,L) = (g_pp E^2 + 2 g_tp E L + g_tt L^2) / (g_tt g_pp - g_tp^2)

and divided by u^t to give coordinate-time frequencies.  Public API
(qpo_at_radius, qpo_table, find_isco_km, find_radius_for_nu2,
kerr_ht_angular, _fac_r, _fac_t) is unchanged.

Sign convention: q = Q/M^3 in the Geroch-Hansen sense (Kerr: q = -j^2,
oblate neutron star: q < 0), exactly as produced by ht_full.HTModel.
"""

import numpy as np

from ht_full import HTExterior

C_KM_S = 2.99792458e5

_CACHE = {}


def _ext(M, j, q):
    key = (float(M), float(j), float(q))
    e = _CACHE.get(key)
    if e is None:
        if len(_CACHE) > 256:
            _CACHE.clear()
        e = HTExterior(M, j * M * M, q * M ** 3)
        _CACHE[key] = e
    return e


def _d1(f, r, h):
    """4th-order central first derivative."""
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def _d2(f, r, h):
    """4th-order central second derivative."""
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r)
            + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)


# =====================================================================
# Circular orbit + epicyclic frequencies from the metric
# =====================================================================
def circular_orbit(ext, r):
    """
    Prograde circular equatorial orbit.  Returns dict with Omega, u^t, E, L
    (geometric units, same length unit as r) or None if no orbit exists.
    """
    h = 1e-3 * r
    gtt_r = _d1(lambda x: ext.metric_components(x)[0], r, h)
    gtp_r = _d1(lambda x: ext.metric_components(x)[1], r, h)
    gpp_r = _d1(lambda x: ext.metric_components(x)[2], r, h)
    if not (np.isfinite(gtt_r) and np.isfinite(gtp_r) and np.isfinite(gpp_r)):
        return None
    disc = gtp_r ** 2 - gtt_r * gpp_r
    if disc < 0.0 or gpp_r == 0.0:
        return None
    Om = (-gtp_r + np.sqrt(disc)) / gpp_r
    gtt, gtp, gpp, grr, gth = ext.metric_components(r)
    n2 = -(gtt + 2.0 * gtp * Om + gpp * Om * Om)
    if not np.isfinite(n2) or n2 <= 0.0:
        return None
    ut = 1.0 / np.sqrt(n2)
    return {'Omega': Om, 'ut': ut,
            'E': -(gtt + gtp * Om) * ut, 'L': (gtp + gpp * Om) * ut}


def _H(ext, r, th, E, L):
    gtt, gtp, gpp, _, _ = ext.metric_components(r, th)
    return (gpp * E * E + 2.0 * gtp * E * L + gtt * L * L) / (gtt * gpp - gtp * gtp)


def epicyclic(ext, r):
    """
    (Omega_phi, Omega_r, Omega_theta) in 1/length (coordinate time = length).
    Omega_r = 0 or Omega_theta = 0 if the corresponding mode is unstable.
    """
    co = circular_orbit(ext, r)
    if co is None:
        return np.nan, 0.0, 0.0
    E, L, ut = co['E'], co['L'], co['ut']
    grr = ext.metric_components(r)[3]
    gth = ext.metric_components(r)[4]

    Hrr = _d2(lambda x: _H(ext, x, np.pi / 2, E, L), r, 2e-3 * r)
    Hthth = _d2(lambda t: _H(ext, r, t, E, L), np.pi / 2, 2e-3)

    kr2 = Hrr / (2.0 * grr)
    kt2 = Hthth / (2.0 * gth)
    Or = np.sqrt(kr2) / ut if kr2 > 0 else 0.0
    Ot = np.sqrt(kt2) / ut if kt2 > 0 else 0.0
    return co['Omega'], Or, Ot


# =====================================================================
# Epicyclic factors  (API kept)
# =====================================================================
def _fac_r(x, j, q):
    """Radial epicyclic factor  Omega_r^2 / Omega_phi^2  (x = r/M)."""
    Op, Or, Ot = _epi_signed(x, j, q)
    return Or


def _fac_t(x, j, q):
    """Vertical epicyclic factor Omega_theta^2 / Omega_phi^2  (x = r/M)."""
    Op, Or, Ot = _epi_signed(x, j, q)
    return Ot


def _epi_signed(x, j, q):
    """(Omega_phi, fac_r, fac_t) with SIGNED factors (negative = unstable)."""
    ext = _ext(1.0, j, q)
    co = circular_orbit(ext, x)
    if co is None:
        return np.nan, -1.0, -1.0
    E, L, ut = co['E'], co['L'], co['ut']
    grr = ext.metric_components(x)[3]
    gth = ext.metric_components(x)[4]
    Hrr = _d2(lambda y: _H(ext, y, np.pi / 2, E, L), x, 2e-3 * x)
    Hthth = _d2(lambda t: _H(ext, x, t, E, L), np.pi / 2, 2e-3)
    kr2 = Hrr / (2.0 * grr) / ut ** 2
    kt2 = Hthth / (2.0 * gth) / ut ** 2
    Om2 = co['Omega'] ** 2
    return co['Omega'], kr2 / Om2, kt2 / Om2


# =====================================================================
# Angular frequencies
# =====================================================================
def kerr_ht_angular(x, j, q):
    """
    Return (Omega_phi, Omega_r, Omega_theta) in geometric units (1/M).
    (Name kept for compatibility; this is the full HT result.)
    """
    Op, fr, ft = _epi_signed(x, j, q)
    Or = Op * np.sqrt(fr) if fr > 0 else 0.0
    Ot = Op * np.sqrt(ft) if ft > 0 else 0.0
    return Op, Or, Ot


def qpo_at_radius(M_geo_km, j, q, r_km):
    """QPO frequencies in Hz at physical radius r_km."""
    x = r_km / M_geo_km
    Op, Or, Ot = kerr_ht_angular(x, j, q)
    conv = C_KM_S / (2.0 * np.pi * M_geo_km)

    nu_phi   = Op * conv
    nu_r     = Or * conv
    nu_theta = Ot * conv

    return {
        'r_km':     r_km,
        'x':        x,
        'nu_phi':   nu_phi,
        'nu_r':     nu_r,
        'nu_theta': nu_theta,
        'nu_peri':  nu_phi - nu_r,
        'nu_nod':   nu_phi - nu_theta,
        'nu_LT':    nu_phi - nu_theta,     # old alias of nu_nod, NOT the frame-dragging nu_LT of the paper
    }


def qpo_table(M_geo_km, j, q, r_min_km, r_max_km, n_pts=20):
    rs = np.linspace(r_min_km, r_max_km, n_pts)
    return [qpo_at_radius(M_geo_km, j, q, r) for r in rs]


# =====================================================================
# ISCO finder -- outermost zero of fac_r(x)
# =====================================================================
def find_isco_km(M_geo_km, j, q, r_lo=3.0, r_hi=30.0, r_surface=None):
    """
    ISCO radius [km]: the outermost radius where the radial epicyclic
    frequency vanishes (Omega_r^2 = 0), found by scanning inward from
    r_hi and bisecting the first sign change.

    Returns r_lo if the ISCO lies inside r_lo (e.g. inside the star when
    r_surface is given) and -1.0 if no stable orbit exists at r_hi.
    """
    r_lo = max(r_lo, 2.05 * M_geo_km)
    if r_surface is not None and r_surface > r_lo:
        r_lo = r_surface * 1.001
    r_hi = max(r_hi, 40.0 * M_geo_km)

    def fr(r_km):
        return _fac_r(r_km / M_geo_km, j, q)

    f_lo = fr(r_lo)
    if f_lo > 0:
        return r_lo
    f_hi = fr(r_hi)
    if not f_hi > 0:
        return -1.0

    grid = np.linspace(r_hi, r_lo, 120)
    prev_r, prev_f = grid[0], f_hi
    a = b = None
    for r in grid[1:]:
        f = fr(r)
        if not f > 0:               # includes NaN
            a, b = r, prev_r        # f(a) <= 0 < f(b)
            break
        prev_r, prev_f = r, f
    if a is None:
        return r_lo
    for _ in range(60):
        mid = 0.5 * (a + b)
        if fr(mid) > 0:
            b = mid
        else:
            a = mid
        if abs(b - a) < 1e-9:
            break
    return 0.5 * (a + b)


def find_inner_edge_km(M_geo_km, j, q, r_surface):
    """
    Inner edge of the disc, r_in = max(R_surface, r_ISCO), and its kind.

    Returns (r_in_km, 'isco') if a marginally stable orbit exists outside
    the star, else (r_surface_km, 'surface').  r_surface should be the
    equatorial coordinate radius HTModel.r_surface (HT coordinate, same as
    the exterior metric).
    """
    if _fac_r(r_surface / M_geo_km, j, q) > 0:
        return r_surface, 'surface'
    r_isco = find_isco_km(M_geo_km, j, q, r_surface=None)
    if r_isco <= r_surface:
        return r_surface, 'surface'
    return r_isco, 'isco'


# =====================================================================
# Optional -- find radius where nu_phi matches an observed value
# =====================================================================
def find_radius_for_nu2(M_geo_km, j, q, nu2_target_hz,
                        r_lo_km=3.0, r_hi_km=50.0):
    """Bisection on nu_phi(r) = nu2_target."""
    def f(r):
        return qpo_at_radius(M_geo_km, j, q, r)['nu_phi'] - nu2_target_hz

    f_lo = f(r_lo_km)
    f_hi = f(r_hi_km)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return -1.0

    for _ in range(60):
        r_mid = 0.5 * (r_lo_km + r_hi_km)
        fm = f(r_mid)
        if abs(fm) < 1e-4:
            return r_mid
        if fm * f_lo > 0:
            r_lo_km = r_mid
            f_lo = fm
        else:
            r_hi_km = r_mid
    return 0.5 * (r_lo_km + r_hi_km)
