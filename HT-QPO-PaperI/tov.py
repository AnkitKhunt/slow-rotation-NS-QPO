"""
tov.py

Solve the Tolman-Oppenheimer-Volkoff equations for a non-rotating
spherically symmetric neutron star.

Everything is stored in km (geometric units).
"""

import numpy as np
from scipy.integrate import solve_ivp

G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
M_SUN = 1.98847e33
KM    = 1.0e5
M_SUN_GEO_KM = 1.4766


class TOV:
    """Solve TOV equations for a given EOS and central density."""

    def __init__(self, eos):
        self.eos = eos
        self.M_solar = None
        self.R_km    = None
        self.M_geo   = None
        self.R_geo   = None
        self.profile = None

    def _rhs(self, r, y):
        m, P = y
        if P <= 0:
            return [0.0, 0.0]
        if r <= 0 or r - 2*m <= 0:
            return [0.0, 0.0]

        P_cgs = P * C_CGS**4 / G_CGS
        rho_cgs = self.eos.energy_density(P_cgs)
        rho_geo = rho_cgs * G_CGS / C_CGS**2

        dmdr = 4.0 * np.pi * r**2 * rho_geo
        dPdr = -(rho_geo + P) * (m + 4.0*np.pi*r**3 * P) / (r * (r - 2*m))
        return [dmdr, dPdr]

    def solve(self, rho_c_cgs, r_max_km=30.0, n_points=4000,
             _r_max_cap_km=300.0):
        P_c_cgs = self.eos.pressure(rho_c_cgs)
        P_c_geo = P_c_cgs * G_CGS / C_CGS**4

        r0 = 1.0e2
        r_max = r_max_km * KM
        y0 = [0.0, P_c_geo]

        r_grid = np.linspace(r0, r_max, n_points)

        # v3.2 FIX: P is ~1e-14 cm^-2 in these units, so the old scalar
        # atol=1e-12 left the pressure integration uncontrolled near the
        # surface (TOV radii scattered by ~0.1 km between neighbouring
        # rho_c).  Use a per-component absolute tolerance scaled to P_c.
        sol = solve_ivp(self._rhs, [r0, r_max], y0, t_eval=r_grid,
                        method='DOP853', rtol=1e-9,
                        atol=[1e-6, max(abs(P_c_geo) * 1e-13, 1e-60)])

        r = sol.t
        m = sol.y[0]
        P = sol.y[1]

        idx = np.where(P <= 0.0)[0]
        if len(idx) == 0:
            # The true surface (P -> 0) was never reached within
            # r_max_km: P is still positive at the last grid point.
            # Returning that point as "the surface" would silently
            # fabricate a truncated (R, M) pair (this is what produced
            # jagged, artifact points -- e.g. a vertical drop at
            # r_max_km -- in mass-radius curves). Instead, widen the
            # integration range and retry, up to a hard cap.
            if r_max_km < _r_max_cap_km:
                return self.solve(rho_c_cgs, r_max_km=2.0 * r_max_km,
                                  n_points=n_points,
                                  _r_max_cap_km=_r_max_cap_km)
            raise RuntimeError(
                f"TOV surface not found for rho_c={rho_c_cgs:.3e} "
                f"g/cm^3 even out to r_max={r_max_km:.0f} km "
                f"(P is still positive there). This EOS/rho_c "
                f"combination likely isn't a bound, finite star -- "
                f"skip this point rather than using it."
            )
        i_surf = idx[0]

        r_km = r[:i_surf+1] / KM
        m_km = m[:i_surf+1] / KM
        P_geo = P[:i_surf+1]

        rho_cgs = np.array([self.eos.energy_density(
                                p * C_CGS**4 / G_CGS)
                            for p in P_geo])

        rho_km2 = rho_cgs * G_CGS / C_CGS**2 * KM**2
        P_km2 = P_geo * KM**2

        nu = np.zeros_like(r_km)
        for i in range(1, len(r_km)):
            r_i = 0.5 * (r_km[i] + r_km[i-1])
            m_i = 0.5 * (m_km[i] + m_km[i-1])
            P_i = 0.5 * (P_km2[i] + P_km2[i-1])
            denom = r_i * (r_i - 2.0 * m_i)
            if denom <= 0:
                nu[i] = nu[i-1]
                continue
            dnu = (m_i + 4.0 * np.pi * r_i**3 * P_i) / denom
            nu[i] = nu[i-1] + dnu * (r_km[i] - r_km[i-1])

        R = r_km[-1]
        M = m_km[-1]
        if 1.0 - 2.0 * M / R > 0:
            nu_ext_R = 0.5 * np.log(1.0 - 2.0 * M / R)
            nu += (nu_ext_R - nu[-1])

        self.M_solar = m_km[-1] / M_SUN_GEO_KM
        self.R_km    = r_km[-1]
        self.M_geo   = m_km[-1]
        self.R_geo   = r_km[-1]

        self.profile = {
            'r_geo':   r_km,
            'm_geo':   m_km,
            'P_geo':   P_km2,
            'P_cgs':   P_geo,
            'r_km':    r_km,
            'nu':      nu,
            'rho_cgs': rho_cgs,
            'rho_km2': rho_km2,
        }
        return self.profile

    def mass_radius_curve(self, rho_c_list):
        M_list, R_list = [], []
        for rc in rho_c_list:
            try:
                self.solve(rc)
                M_list.append(self.M_solar)
                R_list.append(self.R_km)
            except Exception:
                continue
        return np.array(M_list), np.array(R_list)