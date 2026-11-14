"""
veff.py

Hartle-Thorne metric components (equatorial plane) and the effective
potential for equatorial geodesics in a stationary axisymmetric spacetime.

v2 CHANGES
----------
1. The metric is now the SAME one used by orbits.py / qpo.py
   (ht_full.HTExterior).  The old class re-implemented the Abramowicz et
   al. (2003) expressions with a dimensionally inconsistent term
   (r^2 - 2M in A1) and had no g_rr.

2. The old "Veff = g_tt + 2 Omega g_tphi + Omega^2 g_phiphi" with
   Omega = Omega(r) is just -1/(u^t)^2 along the circular-orbit family.
   It has no extremum at the ISCO, so d2V/dr2 never signalled stability.
   Veff is now the true radial effective potential at fixed angular
   momentum L,

        V(r; L) = E_+(r, L)
                = [ -g_tphi L + sqrt( Delta (L^2 + g_phiphi) ) ] / g_phiphi,
        Delta   = g_tphi^2 - g_tt g_phiphi ,

   i.e. the energy at which a particle with angular momentum L has
   dr/dtau = 0.  Allowed motion: E >= V(r; L).  Circular orbits sit at
   dV/dr = 0, are stable if d2V/dr2 > 0, and at the ISCO
   dV/dr = d2V/dr2 = 0.

   Signature kept: veff(metric, r) still works; when no L is given the
   ISCO angular momentum L_isco of the metric is used, so the plotted
   curve has its inflection point exactly at r_ISCO, as the plots marking
   r_ISCO imply.

Sign convention: q = Q/M^3 in the Geroch-Hansen sense (Kerr: q = -j^2,
oblate NS: q < 0), as returned by ht_full.HTModel.
"""

import numpy as np

from ht_full import HTExterior
from qpo import circular_orbit, find_isco_km


# =====================================================================
# Hartle-Thorne metric (equatorial plane)
# =====================================================================
class HartleThorneMetric:
    """
    Parameters
    ----------
    M_geo_km : gravitational mass in geometric units (km)
    j        : dimensionless spin  J/M^2
    q        : dimensionless quadrupole  Q/M^3 (Geroch-Hansen sign)
    """

    def __init__(self, M_geo_km, j, q):
        self.M = M_geo_km
        self.j = j
        self.q = q
        self.ext = HTExterior(M_geo_km, j * M_geo_km**2, q * M_geo_km**3)
        self._ref = None

    # ---- components ------------------------------------------------------
    def g_tt(self, r):
        return self.ext.metric_components(r)[0]

    def g_tphi(self, r):
        return self.ext.metric_components(r)[1]

    def g_phiphi(self, r):
        return self.ext.metric_components(r)[2]

    def g_rr(self, r):
        return self.ext.metric_components(r)[3]

    # ---- radial derivatives (4th order) -----------------------------------
    @staticmethod
    def _d1(f, r, dr):
        return (-f(r + 2 * dr) + 8 * f(r + dr)
                - 8 * f(r - dr) + f(r - 2 * dr)) / (12.0 * dr)

    def dg_tt_dr(self, r, dr=None):
        dr = 1e-3 * r if dr is None else dr
        return self._d1(self.g_tt, r, dr)

    def dg_tphi_dr(self, r, dr=None):
        dr = 1e-3 * r if dr is None else dr
        return self._d1(self.g_tphi, r, dr)

    def dg_phiphi_dr(self, r, dr=None):
        dr = 1e-3 * r if dr is None else dr
        return self._d1(self.g_phiphi, r, dr)

    # ---- ISCO reference orbit ---------------------------------------------
    @property
    def ref(self):
        """(r_isco, E_isco, L_isco); fallback r = 6M if no ISCO is found."""
        if self._ref is None:
            r_i = find_isco_km(self.M, self.j, self.q)
            if r_i <= 0:
                r_i = 6.0 * self.M
            co = circular_orbit(self.ext, r_i)
            if co is None:
                co = circular_orbit(self.ext, 6.0 * self.M)
                r_i = 6.0 * self.M
            self._ref = (r_i, co['E'], co['L'])
        return self._ref


# =====================================================================
# Circular orbits
# =====================================================================
def circular_omega(metric, r):
    """Prograde circular-orbit angular velocity; NaN if none exists."""
    gtt_r = metric.dg_tt_dr(r)
    gtp_r = metric.dg_tphi_dr(r)
    gpp_r = metric.dg_phiphi_dr(r)
    if (not np.isfinite(gtt_r) or not np.isfinite(gtp_r)
            or not np.isfinite(gpp_r) or gpp_r == 0.0):
        return np.nan
    disc = gtp_r**2 - gtt_r * gpp_r
    if disc < 0.0:
        return np.nan
    return (-gtp_r + np.sqrt(disc)) / gpp_r


def circular_EL(metric, r):
    """(E, L) per unit mass of the circular orbit at r; (nan, nan) if none."""
    co = circular_orbit(metric.ext, r)
    if co is None:
        return np.nan, np.nan
    return co['E'], co['L']


def isco_radius(metric):
    """ISCO radius [km] of this metric (outermost zero of Omega_r^2)."""
    return metric.ref[0]


# =====================================================================
# Effective potential
# =====================================================================
def veff(metric, r, E=None, L=None):
    """
    V(r; L) = E_+(r, L)  (energy per unit mass at a radial turning point).

    If L is None the ISCO angular momentum of the metric is used.
    (E is accepted only for symmetry with veff_radial_term.)
    """
    if L is None:
        L = metric.ref[2]
    gtt, gtp, gpp, _, _ = metric.ext.metric_components(r)
    if not (np.isfinite(gtt) and np.isfinite(gtp) and np.isfinite(gpp)):
        return np.nan
    Delta = gtp * gtp - gtt * gpp
    arg = Delta * (L * L + gpp)
    if Delta <= 0.0 or arg < 0.0:
        return np.nan
    return (-gtp * L + np.sqrt(arg)) / gpp


def dveff_dr(metric, r, dr=None, L=None):
    """First radial derivative of V(r; L)."""
    dr = 1e-4 * r if dr is None else dr
    vp, vm = veff(metric, r + dr, L=L), veff(metric, r - dr, L=L)
    if not (np.isfinite(vp) and np.isfinite(vm)):
        return np.nan
    return (vp - vm) / (2.0 * dr)


def d2veff_dr2(metric, r, dr=None, L=None):
    """Second radial derivative of V(r; L)."""
    dr = 2e-3 * r if dr is None else dr
    vp, v0, vm = (veff(metric, r + dr, L=L), veff(metric, r, L=L),
                  veff(metric, r - dr, L=L))
    if not (np.isfinite(vp) and np.isfinite(v0) and np.isfinite(vm)):
        return np.nan
    return (vp - 2.0 * v0 + vm) / (dr**2)
