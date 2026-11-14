"""
eos_loader.py

Load an equation-of-state (EOS) table and provide interpolation.

Expected file format (whitespace separated):
    rho [g/cm^3]   P [dyn/cm^2]   H [cm^2/s^2]   n_b [1/cm^3]

Leading integer headers and '#' comment lines are ignored.
"""

import numpy as np

G_CGS = 6.67430e-8      # cm^3 g^-1 s^-2
C_CGS = 2.99792458e10   # cm/s
M_SUN = 1.98847e33      # g


class EOS:
    """Read, convert, and interpolate an EOS table."""

    def __init__(self, filename):
        self.filename = filename
        raw = self._load(filename)
        self.rho, self.P, self.H, self.nb = self._clean(raw)

        # Convert to geometric units (G = c = 1, length in cm)
        self.rho_geo = self.rho * G_CGS / C_CGS**2
        self.P_geo   = self.P   * G_CGS / C_CGS**4

    # ------------------------------------------------------------------
    def _load(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) >= 2:
                    row = (vals[:4] if len(vals) >= 4
                           else [vals[0], vals[1], 0.0, 0.0])
                    data.append(row)
        return np.array(data)

    # ------------------------------------------------------------------
    def _clean(self, data):
        rho = data[:, 0]
        P   = data[:, 1]
        H   = data[:, 2]
        nb  = data[:, 3]
        mask = (rho > 0) & (P > 0) & np.isfinite(rho) & np.isfinite(P)
        rho, P, H, nb = rho[mask], P[mask], H[mask], nb[mask]
        order = np.argsort(rho)
        return rho[order], P[order], H[order], nb[order]

    # ------------------------------------------------------------------
    def pressure(self, rho_cgs):
        return np.interp(rho_cgs, self.rho, self.P)

    def energy_density(self, P_cgs):
        return np.interp(P_cgs, self.P, self.rho)

    def __repr__(self):
        return (f"EOS({self.filename}, N={len(self.rho)}, "
                f"rho=[{self.rho[0]:.2e},{self.rho[-1]:.2e}] g/cm^3)")