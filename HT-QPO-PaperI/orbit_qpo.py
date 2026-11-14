#!/usr/bin/env python3
"""
orbit_qpo.py -- a test-particle orbit around a Hartle-Thorne star and the
QPO peaks of a hot spot moving on it.

Physics
-------
* Star: HT model (O(Omega^2)) of a given EOS, total mass M and spin nu_s.
* Orbit: timelike geodesic of the HT exterior, started on the equator with a
  small radial kick (eccentricity) and a small tilt (inclination).  It is a
  circular orbit plus the two epicyclic oscillations, so in the inertial frame
      the pericentre advances at  nu_per = nu_phi - nu_r ,
      the orbital plane precesses at nu_nod = nu_phi - nu_theta .
  The orbit starts at r0 with the velocity of the circular orbit plus a
  small radial kick, so to first order in the kick it oscillates about r0,
  and the epicyclic frequencies are evaluated at r0 (the time-averaged
  radius <r> is printed as a check).  Matching L_z to an equatorial
  circular orbit would be wrong here: for a tilted orbit L_z = L cos(iota),
  and near the ISCO, where L(r) is flat, a small deficit in L_z moves the
  matched radius far inside the orbit.
* Hot spot: bolometric flux F ~ g^4 with the redshift factor
      g = 1 / [u^t (1 - n.v)],      v = dx/dt (coordinate velocity, c = 1),
  for a distant observer in direction n (inclination i_obs to the spin axis).
  Light bending is neglected; the Roemer delay t_obs = t - n.x is included.
  g^4 is a non-linear function of the orbit, so its power spectrum contains
  the orbital frequency and the combination frequencies, among them the
  RPM QPOs nu_U = nu_phi, nu_L = nu_per and nu_nod.

Usage
-----
    python3 orbit_qpo.py --eos eos/IOPB --mass 1.922 --spin 363 \
        --nu-target 1098.4 --ecc 0.015 --inc 8 --T 0.5 --out results/orbit.npz

--nu-target sets the start radius so that the circular orbital frequency is
the observed upper kHz QPO (default: a 4U 1728-34 triplet).
"""
import argparse
import time

import numpy as np
_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy 2 / numpy 1
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from eos_loader import EOS
from ht_full import HTModel
from run_qpo import solve_rho_c_for_mass
from orbits import HTMetricFull, integrate_geodesic
from qpo import qpo_at_radius, find_isco_km, circular_orbit
from fft_qpo import pericentre_precession

C_KM_S = 2.99792458e5


def build_star(eos_path, M, nu_s):
    eos = EOS(eos_path)
    rho_c = solve_rho_c_for_mass(eos, M, verbose=False, spin_hz=nu_s)
    return HTModel(eos, rho_c, nu_s)


def radius_for_nu_phi(model, nu_target):
    """Circular-orbit radius [km] with nu_phi = nu_target [Hz]."""
    M, j, q = model.M_geo, model.j, model.q
    f = lambda r: qpo_at_radius(M, j, q, r)['nu_phi'] - nu_target
    return brentq(f, 1.01 * model.r_surface, 60.0 * M, xtol=1e-10)


def hot_spot(traj, i_obs_deg, n_grid):
    """Uniformly sampled (in observer time) flux of a hot spot, F ~ g^4."""
    t, r, th, ph = traj['t'], traj['r'], traj['theta'], traj['phi']
    ut, ur, uth, uph = traj['ut'], traj['ur'], traj['uth'], traj['uph']
    rd, thd, phd = ur / ut, uth / ut, uph / ut          # d/dt
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    x, y, z = r * st * cp, r * st * sp, r * ct
    vx = rd * st * cp + r * ct * cp * thd - r * st * sp * phd
    vy = rd * st * sp + r * ct * sp * thd + r * st * cp * phd
    vz = rd * ct - r * st * thd
    io = np.radians(i_obs_deg)
    n = np.array([np.sin(io), 0.0, np.cos(io)])
    nv = n[0] * vx + n[1] * vy + n[2] * vz
    g = 1.0 / (ut * (1.0 - nv))
    t_obs = t - (n[0] * x + n[1] * y + n[2] * z)       # Roemer delay
    F = g ** 4
    tg = np.linspace(t_obs[0], t_obs[-1], n_grid)
    Fg = CubicSpline(t_obs, F)(tg)
    return tg / C_KM_S, Fg, g


def psd(ts, F):
    """Normalised power spectrum (Hann window)."""
    w = np.hanning(len(F))
    P = np.abs(np.fft.rfft((F - F.mean()) * w)) ** 2
    f = np.fft.rfftfreq(len(F), ts[1] - ts[0])
    return f, P / P[f > 5].max()


def peak_near(f, P, f0, rel=0.06):
    """Frequency of the highest PSD bin within |f - f0| < rel f0 (parabolic)."""
    sel = np.where(np.abs(f - f0) < rel * f0)[0]
    if len(sel) < 3:
        return np.nan
    i = sel[np.argmax(P[sel])]
    a, b, c = np.log(P[i - 1:i + 2] + 1e-300)
    den = a - 2 * b + c
    return f[i] + (0.5 * (a - c) / den if den != 0 else 0.0) * (f[1] - f[0])


def node_and_pericentre(traj, n_grid=2 ** 18):
    """Angles of the ascending node and of the pericentre vs time [s]."""
    t = traj['t']
    tg = np.linspace(t[0], t[-1], n_grid)
    cs = {k: CubicSpline(t, traj[k]) for k in ('r', 'theta', 'phi')}
    r, th, ph = cs['r'](tg), cs['theta'](tg), cs['phi'](tg)
    x, y, z = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), \
        r * np.cos(th)
    X = {k: CubicSpline(tg, v) for k, v in (('x', x), ('y', y), ('z', z))}
    vx, vy, vz = X['x'](tg, 1), X['y'](tg, 1), X['z'](tg, 1)
    lx, ly = y * vz - z * vy, z * vx - x * vz
    # ascending node: direction of z_hat x l  ->  angle atan2(lx, -ly)
    node = np.unwrap(np.arctan2(lx, -ly))
    ts = tg / C_KM_S
    # pericentres: local minima of r
    d = np.diff(r)
    im = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    t_p, phi_p = ts[im], ph[im]
    return ts, node, t_p, phi_p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--eos', default='eos/IOPB')
    ap.add_argument('--mass', type=float, default=1.922)
    ap.add_argument('--spin', type=float, default=363.0)
    ap.add_argument('--nu-target', type=float, default=1098.4)
    ap.add_argument('--ecc', type=float, default=0.015,
                    help='radial kick / orbital speed')
    ap.add_argument('--inc', type=float, default=8.0,
                    help='orbital inclination [deg]')
    ap.add_argument('--iobs', type=float, default=60.0,
                    help='observer inclination [deg]')
    ap.add_argument('--T', type=float, default=0.5,
                    help='integration time [s]')
    ap.add_argument('--out', default='results/orbit.npz')
    a = ap.parse_args()

    model = build_star(a.eos, a.mass, a.spin)
    M, j, q = model.M_geo, model.j, model.q
    r_isco = find_isco_km(M, j, q, r_surface=model.r_surface)
    r0 = radius_for_nu_phi(model, a.nu_target)
    print(f'star: M = {model.M_solar:.4f} Msun, nu_s = {a.spin} Hz, '
          f'j = {j:.4f}, q = {q:.4f}, Qbar = {-q / j**2:.3f}, '
          f'R_eq = {model.r_surface:.3f} km, r_ISCO = {r_isco:.3f} km '
          f'(x = {r_isco / M:.3f})')
    print(f'start radius r0 = {r0:.4f} km (x = {r0 / M:.4f})')

    mf = HTMetricFull(model.exterior)
    co = circular_orbit(model.exterior, r0)
    n_orb = a.T * C_KM_S * co['Omega'] / (2 * np.pi)
    n_pts = int(n_orb * 160)
    t0 = time.time()
    tr = integrate_geodesic(mf, r0, 90.0 - a.inc, n_orb, n_pts=n_pts,
                            ecc=a.ecc, r_stop=model.r_surface)
    print(f'integrated {n_orb:.0f} orbits, {len(tr["t"])} points, '
          f'{time.time() - t0:.0f} s; terminated = {tr["terminated"]}')
    print(f'max |u.u + 1| = {np.max(np.abs(tr["norm_t"] + 1)):.2e}, '
          f'dE/E = {np.ptp(tr["E_t"]) / abs(tr["E"]):.2e}, '
          f'dL/L = {np.ptp(tr["Lz_t"]) / abs(tr["Lz"]):.2e}')

    # epicyclic frequencies at the start radius; <r> as a check
    r_mean = _trapz(tr['r'], tr['t']) / (tr['t'][-1] - tr['t'][0])
    pr = qpo_at_radius(M, j, q, r0)
    print(f'r range {tr["r"].min():.3f}-{tr["r"].max():.3f} km, time-averaged '
          f'<r> = {r_mean:.4f} km ({100 * (r_mean / r0 - 1):+.2f}% from r0)')

    # light curve and spectrum
    ts, F, g = hot_spot(tr, a.iobs, 2 ** 20)
    f, P = psd(ts, F)

    # precession angles
    tn, node, t_p, phi_p = node_and_pericentre(tr)
    nu_nod_dir = np.polyfit(tn, node, 1)[0] / (2 * np.pi)
    nu_per_dir = np.polyfit(t_p, phi_p - 2 * np.pi * np.arange(len(t_p)),
                            1)[0] / (2 * np.pi)

    names = ('nu_nod', 'nu_r', 'nu_peri', 'nu_theta', 'nu_phi')
    print(f'{"":10s} {"epicyclic(r0)":>16s} {"PSD peak":>10s} '
          f'{"direct":>10s}')
    meas = {}
    for k in names:
        rel = {'nu_nod': 0.25, 'nu_theta': 0.008, 'nu_phi': 0.008}.get(k, 0.05)
        fp = peak_near(f, P, pr[k], rel)
        meas[k] = fp
        dr = {'nu_nod': nu_nod_dir, 'nu_peri': nu_per_dir}.get(k, np.nan)
        print(f'{k:10s} {pr[k]:16.3f} {fp:10.3f} {dr:10.3f}')

    np.savez(a.out, t=tr['t'] / C_KM_S, r=tr['r'], theta=tr['theta'],
             phi=tr['phi'], M_km=M, j=j, q=q, R_eq=model.r_surface,
             r_isco=r_isco, R_gc=r0, r_mean=r_mean, r0=r0, M_sun=model.M_solar,
             nu_s=a.spin, inc=a.inc, ecc=a.ecc, iobs=a.iobs,
             ts=ts[::8], F=F[::8], f=f, P=P,
             tn=tn[::16], node=node[::16], t_p=t_p, phi_p=phi_p,
             nu_nod_dir=nu_nod_dir, nu_per_dir=nu_per_dir,
             pred=np.array([pr[k] for k in names]),
             meas=np.array([meas[k] for k in names]),
             names=np.array(names))
    print('saved', a.out)


if __name__ == '__main__':
    main()
