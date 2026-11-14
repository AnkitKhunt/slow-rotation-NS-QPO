"""
ht_full.py

Full Hartle-Thorne slow-rotation neutron star.

    * TOV background          (tov.py)
    * Frame-dragging ODE      -> J, I
    * l=0 spherical sector    -> delta M (spin-induced mass shift), xi_0
    * l=2 oblateness system   -> Q, q, xi_2
    * Equatorial / polar radius of the rotating surface
    * Exterior HT metric      -> g_tt, g_tphi, g_phiphi, g_rr, g_thth

References:
    Hartle (1967) ApJ 150, 1005
    Hartle & Thorne (1968) ApJ 153, 807
    Berti, White, Maniopoulou & Bruni (2005) MNRAS 358, 923

v3 CHANGES (code fixes 1, 3, 5 of the paper plan)
-------------------------------------------------
1. NEW l = 0 sector.  The spin-induced mass shift
        delta M = m0(R) + J^2 / R^3
   is now computed, and the exterior metric uses the TOTAL mass
        M_rot = M_TOV + delta M,
   so j = J / M_rot^2 and q = Q / M_rot^3.  The equatorial radius of the
   rotating star (coordinate and circumferential) and the polar radius are
   computed from xi_0, xi_2 (Hartle 1967).  HTModel.r_surface is the
   equatorial coordinate radius, i.e. the inner edge to compare with r_ISCO
   in the SAME radial coordinate as the exterior metric.
   Pass include_l0=False to recover the v2 behaviour (TOV mass).

3. The interior ODE coefficients are now CubicSpline interpolants built
   once (the v2 right-hand sides looked up the nearest grid point with
   np.argmin: piecewise-constant coefficients and O(N) cost per call).
   nu' is evaluated analytically, nu' = 2 (m + 4 pi r^3 P) / (r (r - 2m)).

5. The frame-dragging result now also carries jbar(r) on its grid, so
   the enclosed angular momentum J(r) = r^4 jbar dW/dr / 6 can be plotted
   correctly inside the star.

Attributes of HTModel (all lengths in km, geometric units):
    M_geo, M_solar     total (rotating) mass   <- used by the exterior
    M0_geo, M0_solar   non-rotating TOV mass at the same rho_c
    dM_geo             delta M
    R_km               TOV (spherical) radius = interior grid edge
    R_eq_km            equatorial coordinate radius  (= r_surface)
    R_eq_circ_km       equatorial circumferential radius (compare to NICER)
    R_pol_km           polar coordinate radius
    J_geo, I_geo, Q_geo, j, q
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator

from tov import TOV

C_KM_S       = 2.99792458e5
M_SUN_GEO_KM = 1.4766
G_CGS        = 6.67430e-8
C_CGS        = 2.99792458e10
KM_CM        = 1.0e5


# =====================================================================
# Associated Legendre functions of the second kind (x > 1)
#
# Hartle-Thorne (1968) conventions:
#     Q_2^2(x) = (x^2-1)   d^2 Q_2/dx^2        -> (8/5) x^-3  at large x
#     Q_2^1(x) = -(x^2-1)^(1/2) dQ_2/dx  > 0   -> (2/5) x^-3
# with Q_2(x) = (3x^2-1)/4 ln[(x+1)/(x-1)] - 3x/2.
# Convergent 1/x^2 series for x >= 3, closed forms below that.
# =====================================================================
_NSER = 80


def _q2_series_coeffs():
    c, out = 2.0 / 15.0, []
    for k in range(_NSER):
        out.append(c)
        c *= (1.5 + k) * (2.0 + k) / ((3.5 + k) * (k + 1.0))
    return np.array(out)


_A2 = _q2_series_coeffs()
_KK = np.arange(_NSER)


def _Q2_2(x):
    """Q_2^2(x), x > 1  (HT convention, positive)."""
    if x <= 1.0:
        return 0.0
    if x >= 3.0:
        u = 1.0 / (x * x)
        return (x * x - 1.0) * float(np.sum(
            (3 + 2 * _KK) * (4 + 2 * _KK) * _A2 * u ** _KK)) * x ** -5
    L = np.log((x + 1.0) / (x - 1.0))
    return 1.5 * (x * x - 1.0) * L - (3.0 * x ** 3 - 5.0 * x) / (x * x - 1.0)


def _Q2_1(x):
    """Q_2^1(x), x > 1  (HT convention, positive)."""
    if x <= 1.0:
        return 0.0
    if x >= 3.0:
        u = 1.0 / (x * x)
        return np.sqrt(x * x - 1.0) * float(np.sum(
            (3 + 2 * _KK) * _A2 * u ** _KK)) * x ** -4
    L = np.log((x + 1.0) / (x - 1.0))
    return np.sqrt(x * x - 1.0) * ((3.0 * x * x - 2.0) / (x * x - 1.0)
                                   - 1.5 * x * L)


# =====================================================================
# Background splines (built once per star)
# =====================================================================
_I0 = 10          # first grid point used as the regular-centre start


class _Background:
    """CubicSpline interpolants of the TOV profile (geometric km units)."""

    def __init__(self, profile):
        r = np.asarray(profile['r_geo'], dtype=float)
        m = np.asarray(profile['m_geo'], dtype=float)
        P = np.clip(np.asarray(profile['P_geo'], dtype=float), 0.0, None)
        e = np.clip(np.asarray(profile['rho_km2'], dtype=float), 0.0, None)
        nu = np.asarray(profile['nu'], dtype=float)     # Phi = nu_Hartle / 2
        e[~np.isfinite(e)] = 0.0

        inner = np.clip(1.0 - 2.0 * m / np.maximum(r, 1e-30), 1e-12, None)
        jb = np.exp(-nu) * np.sqrt(inner)                # e^{-nu_H/2} sqrt(1-2m/r)

        self.r, self.m, self.P, self.e, self.nu, self.jb = r, m, P, e, nu, jb
        self.R, self.M = r[-1], m[-1]
        self.s_m = CubicSpline(r, m)
        self.s_P = CubicSpline(r, P)
        self.s_e = CubicSpline(r, e)
        self.s_j = CubicSpline(r, jb)
        self.s_dj = self.s_j.derivative()

    def dnu(self, r_):
        """Hartle nu' = d ln(-g_tt)/dr = 2 Phi'  (analytic)."""
        m_ = float(self.s_m(r_))
        P_ = max(float(self.s_P(r_)), 0.0)
        return 2.0 * (m_ + 4.0 * np.pi * r_ ** 3 * P_) / (r_ * (r_ - 2.0 * m_))


# =====================================================================
# Frame dragging
# =====================================================================
def solve_frame_dragging(profile, Omega, bg=None):
    """
    Solve (1/r^4) d/dr[ r^4 jbar dW/dr ] + (4/r)(d jbar/dr) W = 0
    with W = Omega - omega, jbar = e^{-nu_H/2} sqrt(1 - 2m/r).

    Match at R:
        W(R) = Omega - 2 J / R^3
        J    = (1/6) R^4 (dW/dr)_R
    """
    bg = _Background(profile) if bg is None else bg
    r = bg.r
    r_start = r[_I0]

    def rhs(r_, y):
        W, dWdr = y
        j_l = float(bg.s_j(r_))
        djd_l = float(bg.s_dj(r_))
        coef1 = 4.0 / r_ + djd_l / j_l
        coef0 = (4.0 / r_) * djd_l / j_l
        return [dWdr, -coef1 * dWdr - coef0 * W]

    sol = solve_ivp(rhs, [r_start, r[-1]], [1.0, 0.0],
                    t_eval=r[_I0:], method='DOP853',
                    rtol=1e-10, atol=1e-14)

    W_raw, dWdr_raw = sol.y[0], sol.y[1]
    R = r[-1]
    denom = W_raw[-1] + (1.0 / 3.0) * R * dWdr_raw[-1]

    out_r = r[_I0:]
    jb_on = bg.jb[_I0:]
    if np.abs(denom) < 1e-30 or Omega == 0.0:
        return {'J_geo': 0.0, 'I_geo': 0.0,
                'obar': np.zeros_like(W_raw), 'dobar': np.zeros_like(W_raw),
                'r': out_r, 'jbar': jb_on}

    scale = Omega / denom
    W, dWdr = scale * W_raw, scale * dWdr_raw

    J = (1.0 / 6.0) * R ** 4 * dWdr[-1]
    I = J / Omega

    return {'J_geo': J, 'I_geo': I, 'obar': W, 'dobar': dWdr,
            'r': out_r, 'jbar': jb_on}


def _W_splines(fd):
    r_fd = np.asarray(fd['r'])
    sW = CubicSpline(r_fd, fd['obar'])
    sdW = CubicSpline(r_fd, fd['dobar'])
    return sW, sdW


# =====================================================================
# l = 0 sector: spin-induced mass shift
# =====================================================================
def _deps_dP(profile, eos):
    """
    d eps / d P (dimensionless) on the TOV grid, from the EOS itself
    (centred difference in P).  eps here is the total energy density.
    """
    P_km2 = np.clip(np.asarray(profile['P_geo'], dtype=float), 0.0, None)
    P_dyn = P_km2 / KM_CM ** 2 * C_CGS ** 4 / G_CGS
    out = np.full_like(P_dyn, np.nan)
    h = 1e-4
    for i, p in enumerate(P_dyn):
        if p <= 0.0:
            continue
        ep = eos.energy_density(p * (1.0 + h))
        em = eos.energy_density(p * (1.0 - h))
        out[i] = (ep - em) / (2.0 * h * p) * C_CGS ** 2
    good = np.isfinite(out) & (out > 0)
    if not good.any():
        return np.ones_like(P_dyn)
    # fill the surface point(s) with the last valid value
    idx = np.where(good)[0]
    out = np.interp(np.arange(len(out)), idx, out[idx])
    return out


def solve_spin_mass(profile, fd, eos, bg=None):
    """
    Hartle (1967) l = 0 equations, regular at the centre (m0 = p0* = 0 at
    fixed central density):

      dm0/dr  = 4 pi r^2 (d eps/dP)(eps+P) p0*
                + r^4 jbar^2 W'^2 / 12 - r^3 (jbar^2)' W^2 / 3
      dp0*/dr = - m0 (1 + 8 pi r^2 P)/(r-2m)^2
                - 4 pi (eps+P) r^2 p0* / (r-2m)
                + r^4 jbar^2 W'^2 / (12 (r-2m))
                + (1/3) d/dr [ r^3 jbar^2 W^2 / (r-2m) ]

    delta M = m0(R) + J^2 / R^3.
    Exterior check: m0 = delta M - J^2/r^3 satisfies the first equation
    with jbar = 1, W' = 6J/r^4.
    """
    bg = _Background(profile) if bg is None else bg
    J = fd['J_geo']
    r_fd = np.asarray(fd['r'])
    if J == 0.0:
        z = np.zeros_like(r_fd)
        return {'dM_geo': 0.0, 'm0': z, 'p0': z, 'r': r_fd}

    sW, sdW = _W_splines(fd)
    dedp = _deps_dP(profile, eos)
    # monotone C1 interpolant: no overshoot at EOS kinks (crust/core joins)
    # and no derivative jumps at every grid node (which a linear interpolant
    # would create, forcing tiny integrator steps)
    s_dedp = PchipInterpolator(bg.r, dedp)

    # F(r) = r^3 jbar^2 W^2 / (r - 2m)  and its derivative
    jb = np.interp(r_fd, bg.r, bg.jb)
    mm = np.interp(r_fd, bg.r, bg.m)
    F = r_fd ** 3 * jb ** 2 * fd['obar'] ** 2 / (r_fd - 2.0 * mm)
    s_dF = CubicSpline(r_fd, F).derivative()

    def rhs(r_, y):
        m0, p0 = y
        m_ = float(bg.s_m(r_))
        P_ = max(float(bg.s_P(r_)), 0.0)
        e_ = max(float(bg.s_e(r_)), 0.0)
        j_ = float(bg.s_j(r_))
        dj_ = float(bg.s_dj(r_))
        W_ = float(sW(r_))
        dW_ = float(sdW(r_))
        dj2 = 2.0 * j_ * dj_
        den = r_ - 2.0 * m_
        dm0 = (4.0 * np.pi * r_ ** 2 * float(s_dedp(r_)) * (e_ + P_) * p0
               + r_ ** 4 * j_ ** 2 * dW_ ** 2 / 12.0
               - r_ ** 3 * dj2 * W_ ** 2 / 3.0)
        dp0 = (-m0 * (1.0 + 8.0 * np.pi * r_ ** 2 * P_) / den ** 2
               - 4.0 * np.pi * (e_ + P_) * r_ ** 2 * p0 / den
               + r_ ** 4 * j_ ** 2 * dW_ ** 2 / (12.0 * den)
               + float(s_dF(r_)) / 3.0)
        return [dm0, dp0]

    sol = solve_ivp(rhs, [r_fd[0], r_fd[-1]], [0.0, 0.0], t_eval=r_fd,
                    method='DOP853', rtol=1e-9, atol=1e-14)
    m0, p0 = sol.y[0], sol.y[1]
    R = r_fd[-1]
    dM = m0[-1] + J ** 2 / R ** 3
    return {'dM_geo': dM, 'm0': m0, 'p0': p0, 'r': r_fd}


# =====================================================================
# l = 2 oblateness
# =====================================================================
def solve_oblateness(profile, fd, Omega, bg=None):
    """
    Coupled (h2, v2) system, split into complementary + particular:

        h2 = h2_P + A h2_C ,   v2 = v2_P + A v2_C

    Match at R against exterior (x = R/M - 1):

        h2^ext(R) = + J^2/(M R^3)(1 + M/R) + K Q2_2(x)
        v2^ext(R) = - J^2/R^4              + K (2M/sqrt(R(R-2M))) Q2_1(x)

    Q_HT = J^2/M + (8/5) K M^3 (oblate > 0);  Q = -Q_HT (Geroch-Hansen);
    q = Q / M^3.   For a rotating NS: K > 0, Q < 0, q/j^2 < -1.
    """
    bg = _Background(profile) if bg is None else bg
    r = bg.r
    J = fd['J_geo']
    r_start = r[_I0]
    sW, sdW = _W_splines(fd)

    def coeffs(r_):
        m_ = float(bg.s_m(r_))
        P_ = max(float(bg.s_P(r_)), 0.0)
        e_ = max(float(bg.s_e(r_)), 0.0)
        dnu = bg.dnu(r_)
        den = r_ - 2.0 * m_
        bracket = 8.0 * np.pi * (e_ + P_) - 4.0 * m_ / r_ ** 3
        coeff_h2 = -dnu + (r_ / den) * (1.0 / dnu) * bracket
        return m_, den, dnu, coeff_h2

    def rhs_P(r_, y):
        h2, v2 = y
        m_, den, dnu, coeff_h2 = coeffs(r_)
        if den <= 0 or abs(dnu) < 1e-30:
            return [0.0, 0.0]
        j_ = float(bg.s_j(r_))
        dj_ = float(bg.s_dj(r_))
        W_ = float(sW(r_))
        dW_ = float(sdW(r_))
        d_j2_dr = 2.0 * j_ * dj_

        # Hartle (1967) Eq. for v2
        S_v = (1.0 / r_ + 0.5 * dnu) * (
            (1.0 / 6.0) * r_ ** 4 * j_ ** 2 * dW_ ** 2
            - (1.0 / 3.0) * r_ ** 3 * W_ ** 2 * d_j2_dr)
        dv2dr = -dnu * h2 + S_v

        # Hartle (1967) Eq. for h2
        term_j2 = (1.0 / 6.0) * (0.5 * r_ * dnu - 1.0 / (den * dnu)) \
            * r_ ** 3 * j_ ** 2 * dW_ ** 2
        term_dj2 = (1.0 / 3.0) * (0.5 * r_ * dnu + 1.0 / (den * dnu)) \
            * r_ ** 2 * d_j2_dr * W_ ** 2
        dh2dr = (coeff_h2 * h2 - 4.0 * v2 / (r_ * den * dnu)
                 + term_j2 - term_dj2)
        return [dh2dr, dv2dr]

    def rhs_C(r_, y):
        h2, v2 = y
        m_, den, dnu, coeff_h2 = coeffs(r_)
        if den <= 0 or abs(dnu) < 1e-30:
            return [0.0, 0.0]
        return [coeff_h2 * h2 - 4.0 * v2 / (r_ * den * dnu), -dnu * h2]

    e_c, P_c = bg.e[0], bg.P[0]
    h2_C0 = r_start ** 2
    v2_C0 = -(2.0 * np.pi / 3.0) * (e_c + 3.0 * P_c) * r_start ** 4

    t_eval = r[_I0:]
    solC = solve_ivp(rhs_C, [r_start, r[-1]], [h2_C0, v2_C0], t_eval=t_eval,
                     method='DOP853', rtol=1e-10, atol=1e-16)
    solP = solve_ivp(rhs_P, [r_start, r[-1]], [0.0, 0.0], t_eval=t_eval,
                     method='DOP853', rtol=1e-10, atol=1e-16)
    h2_C, v2_C = solC.y
    h2_P, v2_P = solP.y

    R, M = bg.R, bg.M
    x = R / M - 1.0
    coeff_h = _Q2_2(x)
    coeff_v = 2.0 * M / np.sqrt(R * (R - 2.0 * M)) * _Q2_1(x)
    rhs_h = J ** 2 / (M * R ** 3) * (1.0 + M / R) - h2_P[-1]
    rhs_v = -J ** 2 / R ** 4 - v2_P[-1]

    det = h2_C[-1] * (-coeff_v) - (-coeff_h) * v2_C[-1]
    if abs(det) < 1e-30 or J == 0.0:
        return {'Q_geo': 0.0, 'q': 0.0, 'K': 0.0, 'A': 0.0, 'Q_HT_geo': 0.0,
                'h2': np.zeros_like(t_eval), 'v2': np.zeros_like(t_eval),
                'r': t_eval}

    A = (rhs_h * (-coeff_v) - (-coeff_h) * rhs_v) / det
    K = (h2_C[-1] * rhs_v - v2_C[-1] * rhs_h) / det

    Q_HT = J ** 2 / M + (8.0 / 5.0) * K * M ** 3
    Q = -Q_HT
    return {'Q_geo': Q, 'q': Q / M ** 3, 'K': K, 'A': A, 'Q_HT_geo': Q_HT,
            'h2': h2_P + A * h2_C, 'v2': v2_P + A * v2_C, 'r': t_eval}


# =====================================================================
# Surface of the rotating star
# =====================================================================
def surface_radii(bg, fd, l0, ob):
    """
    Hartle (1967): r(theta) = R + xi_0 + xi_2 P_2(cos theta), with
        xi_l = - p_l* (eps + P) / (dP/dr)
             = p_l* r (r - 2m) / (m + 4 pi r^3 P)          (TOV)
        p_2* = - h_2 - (1/3) r^2 e^{-nu_H} W^2 .
    At the surface (P = 0):  xi_l = p_l*(R) R (R - 2M) / M.

    Returns (R_eq_coord, R_eq_circ, R_pol):
        R_eq_coord = R + xi_0 - xi_2/2          (HT coordinate, equator)
        R_eq_circ  = R_eq_coord [1 - (v_2 - h_2)/2]   (circumferential)
        R_pol      = R + xi_0 + xi_2
    """
    R, M = bg.R, bg.M
    if fd['J_geo'] == 0.0:
        return R, R, R
    fac = R * (R - 2.0 * M) / M
    xi0 = l0['p0'][-1] * fac
    W_R = fd['obar'][-1]
    h2_R, v2_R = ob['h2'][-1], ob['v2'][-1]
    p2 = -h2_R - (1.0 / 3.0) * R ** 2 * np.exp(-2.0 * bg.nu[-1]) * W_R ** 2
    xi2 = p2 * fac
    R_eq = R + xi0 - 0.5 * xi2
    R_eq_circ = R_eq * (1.0 - 0.5 * (v2_R - h2_R))
    R_pol = R + xi0 + xi2
    return R_eq, R_eq_circ, R_pol


# =====================================================================
# Exterior HT metric  (complete: all five non-zero components)
#
#   ds^2 = -e^nu [1 + 2(h0 + h2 P2)] dt^2
#          + e^lam [1 + 2(m0 + m2 P2)/(r-2M)] dr^2
#          + r^2 [1 + 2 k2 P2] ( dtheta^2 + sin^2 th (dphi - w dt)^2 )
#
#   exterior:  w  = 2J/r^3
#              h0 = J^2 / (r^3 (r-2M))          m0 = -J^2 / r^3
#              h2 = J^2/(M r^3)(1+M/r) + K Q2^2(x)
#              v2 = -J^2/r^4 + K 2M/sqrt(r(r-2M)) Q2^1(x)
#              m2 = (r-2M)(-h2 + 6 J^2/r^4),   k2 = v2 - h2
#   x = r/M - 1,  K = -(5/8)(Q + J^2/M)/M^3  (Geroch-Hansen Q = -Q_HT).
#   M is the TOTAL mass (M_TOV + delta M) since v3.
# =====================================================================
class HTExterior:
    def __init__(self, M_geo, J_geo, Q_geo):
        self.M = M_geo
        self.J = J_geo
        self.Q = Q_geo
        self.j = J_geo / M_geo ** 2
        self.q = Q_geo / M_geo ** 3
        self.K = -(5.0 / 8.0) * (Q_geo + J_geo ** 2 / M_geo) / M_geo ** 3

    def _k1(self, r):
        """h2^ext(r)."""
        M, J = self.M, self.J
        if r <= 2.0 * M:
            return 0.0
        x = r / M - 1.0
        return J ** 2 / (M * r ** 3) * (1.0 + M / r) + self.K * _Q2_2(x)

    def _v2(self, r):
        M, J = self.M, self.J
        if r <= 2.0 * M:
            return 0.0
        x = r / M - 1.0
        return (-J ** 2 / r ** 4
                + self.K * 2.0 * M / np.sqrt(r * (r - 2.0 * M)) * _Q2_1(x))

    def _k2(self, r):
        return self._k1(r) - self._v2(r)

    def metric_components(self, r, theta=np.pi / 2):
        """(g_tt, g_tphi, g_phiphi, g_rr, g_thth).  NaN inside r <= 2M."""
        M, J = self.M, self.J
        if r <= 2.0 * M:
            return (np.nan,) * 5
        c2 = np.cos(theta) ** 2
        s2 = 1.0 - c2
        P2 = 0.5 * (3.0 * c2 - 1.0)
        f = 1.0 - 2.0 * M / r
        k1 = self._k1(r)
        k2 = k1 - self._v2(r)
        g_tt = -f * (1.0 + 2.0 * k1 * P2
                     + 2.0 / f * J ** 2 / r ** 4 * (2.0 * c2 - 1.0))
        g_tp = -2.0 * J / r * s2
        ang = 1.0 - 2.0 * k2 * P2
        g_pp = r * r * s2 * ang
        g_th = r * r * ang
        g_rr = (1.0 / f) * (1.0 - 2.0 * (k1 - 6.0 * J ** 2 / r ** 4) * P2
                            - 2.0 / f * J ** 2 / r ** 4)
        return g_tt, g_tp, g_pp, g_rr, g_th

    def g_tt(self, r, theta=np.pi / 2):
        return self.metric_components(r, theta)[0]

    def g_tphi(self, r, theta=np.pi / 2):
        return -2.0 * self.J / r * np.sin(theta) ** 2

    def g_phiphi(self, r, theta=np.pi / 2):
        return self.metric_components(r, theta)[2]

    def g_rr(self, r, theta=np.pi / 2):
        return self.metric_components(r, theta)[3]

    def g_thth(self, r, theta=np.pi / 2):
        return self.metric_components(r, theta)[4]


# =====================================================================
# Full model
# =====================================================================
class HTModel:
    def __init__(self, eos, rho_c_cgs, spin_hz, include_l0=True):
        self.eos = eos
        self.rho_c = rho_c_cgs
        self.spin_hz = spin_hz
        self.include_l0 = include_l0

        self.tov = TOV(eos)
        self.tov.solve(rho_c_cgs)
        bg = _Background(self.tov.profile)

        Omega = 2.0 * np.pi * spin_hz / C_KM_S
        self.Omega = Omega

        self.fd = solve_frame_dragging(self.tov.profile, Omega, bg)
        self.ob = solve_oblateness(self.tov.profile, self.fd, Omega, bg)
        self.l0 = solve_spin_mass(self.tov.profile, self.fd, eos, bg)

        M0 = self.tov.M_geo
        dM = self.l0['dM_geo'] if include_l0 else 0.0
        M_tot = M0 + dM
        J_geo = self.fd['J_geo']
        Q_geo = self.ob['Q_geo']

        self.exterior = HTExterior(M_tot, J_geo, Q_geo)

        self.M0_geo = M0
        self.M0_solar = self.tov.M_solar
        self.dM_geo = dM
        self.M_geo = M_tot
        self.M_solar = M_tot / M_SUN_GEO_KM
        self.R_km = self.tov.R_km
        R_eq, R_eq_circ, R_pol = surface_radii(bg, self.fd, self.l0, self.ob)
        if not include_l0:
            R_eq = R_eq_circ = R_pol = self.R_km
        self.R_eq_km = R_eq
        self.R_eq_circ_km = R_eq_circ
        self.R_pol_km = R_pol
        self.r_surface = R_eq          # inner edge in exterior coordinates
        self.J_geo = J_geo
        self.Q_geo = Q_geo
        self.j = self.exterior.j
        self.q = self.exterior.q
        self.I_geo = self.fd['I_geo']


# =====================================================================
# Self-test
# =====================================================================
if __name__ == '__main__':
    from eos_loader import EOS

    eos = EOS('eos/eosAU')
    model = HTModel(eos, 1.0e15, 400.0)

    print("=" * 64)
    print("  Full Hartle-Thorne model (v3)")
    print("=" * 64)
    print(f"  M_TOV  = {model.M0_solar:.5f} Msun")
    print(f"  dM     = {model.dM_geo / M_SUN_GEO_KM:+.5f} Msun  "
          f"(dM/M = {model.dM_geo / model.M0_geo:+.3e})")
    print(f"  M_rot  = {model.M_solar:.5f} Msun")
    print(f"  R_TOV  = {model.R_km:.4f} km")
    print(f"  R_eq   = {model.R_eq_km:.4f} km (coord), "
          f"{model.R_eq_circ_km:.4f} km (circumferential)")
    print(f"  R_pol  = {model.R_pol_km:.4f} km")
    print(f"  J_geo  = {model.J_geo:+.6f} km^2")
    print(f"  Q_geo  = {model.Q_geo:+.6f} km^3")
    print(f"  j      = {model.j:+.6f}")
    print(f"  q      = {model.q:+.6f}")
    if model.j != 0:
        ratio = model.q / model.j ** 2
        print(f"  q/j^2  = {ratio:+.4f}   (Kerr: -1)")
    print()
    print("  Sanity checks:")
    print(f"    Q < 0 ?            {'YES' if model.Q_geo < 0 else 'NO  <-- BAD'}")
    print(f"    dM > 0 ?           {'YES' if model.dM_geo > 0 else 'NO  <-- BAD'}")
    print(f"    R_eq > R_pol ?     "
          f"{'YES' if model.R_eq_km > model.R_pol_km else 'NO  <-- BAD'}")
    if model.j != 0:
        ok = -25.0 < model.q / model.j ** 2 < -1.0
        print(f"    q/j^2 in [-25,-1]? {'YES' if ok else 'NO  <-- CHECK'}")
