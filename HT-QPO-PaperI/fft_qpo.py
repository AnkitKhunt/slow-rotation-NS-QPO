#!/usr/bin/env python3
"""
fft_qpo.py

FFT study of INTEGRATED geodesics in the Hartle-Thorne spacetime and
comparison with the relativistic precession model (RPM) frequencies of qpo.py.

    nu_2   = nu_phi                  (upper kHz QPO)
    nu_1   = nu_per = nu_phi - nu_r  (lower kHz QPO)
    nu_HBO = 2 nu_nod = 2 (nu_phi - nu_theta)

Uses only the existing code base:
    eos_loader.EOS, ht_full.HTModel, qpo.{find_isco_km, qpo_at_radius,
    circular_orbit}, orbits.{HTMetricFull, integrate_geodesic},
    run_qpo.{solve_rho_c_for_mass, solve_rho_c_for_radius}

What is done
------------
1. A small-amplitude geodesic (radial kick --ecc, inclination --inc-deg) is
   integrated with orbits.integrate_geodesic (Christoffel form, DOP853).
2. Its GUIDING radius R_gc is found: the circular orbit with the same L_z
   (epicyclic theory predicts the frequencies there, not at the start radius).
3. The orbit is resampled on a uniform coordinate-time grid (cubic spline) and
   Fourier transformed (Hann window, parabolic peak interpolation):
        r(t)             -> nu_r
        cos(theta)(t)    -> nu_theta
        r sin(th) e^{i phi}  -> nu_phi and the sideband nu_phi - nu_r = nu_per
4. Direct (non-FFT) cross-checks of the precessions, which need far less time
   than 1/nu_nod:  nu_per from successive pericentres, nu_nod from the phase
   of the orbital-angular-momentum vector l_x + i l_y.
5. Everything is compared with qpo.qpo_at_radius at R_gc, over a scan of radii.

Outputs (in --outdir)
    fft_spectra.png          spectra of r(t), cos(theta)(t), x+iy  + table
    fft_qpo_view.png         the three RPM QPOs read off the spectra
    fft_scan.png / .dat      FFT vs RPM over radius (+ residuals)
    fft_qpo_correlation.png  geodesic points on the (nu_1, nu_2) and
                             (nu_1, nu_HBO) planes, with the RPM curves
    fft_amplitude_test.png   (--amp-test) finite-amplitude shift of the peaks

Usage
    python3 fft_qpo.py --eos eos/eosAU --mass 1.8 --spin 400 --outdir fft_out
    python3 fft_qpo.py --eos eos/eosAU --rho_c 1.0e15 --spin 600 --scan-n 8
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from eos_loader import EOS
from ht_full import HTModel
from qpo import find_isco_km, qpo_at_radius, circular_orbit
from orbits import HTMetricFull, integrate_geodesic
from run_qpo import solve_rho_c_for_mass, solve_rho_c_for_radius
import ht_common

C_KM_S = 2.99792458e5


# =====================================================================
# style (same look as plots.py / orbits.py)
# =====================================================================
def set_style():
    plt.rcParams.update({
        'figure.figsize': (7.0, 5.5), 'figure.dpi': 120,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 12, 'axes.linewidth': 1.0,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
        'grid.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'legend.fontsize': 10, 'legend.frameon': True, 'legend.framealpha': 0.9,
        'legend.edgecolor': '0.7', 'lines.linewidth': 1.8,
    })


# =====================================================================
# model
# =====================================================================
def build_model(args):
    eos = EOS(args.eos)
    if args.rho_c is not None:
        rho_c = args.rho_c
    elif args.radius is not None:
        rho_c = solve_rho_c_for_radius(eos, args.radius, verbose=False)
    else:
        rho_c = solve_rho_c_for_mass(eos, args.mass if args.mass else 1.8,
                                     verbose=False, spin_hz=args.spin)
    model = HTModel(eos, rho_c, args.spin)
    return eos, model, rho_c


# =====================================================================
# orbit -> measured frequencies
# =====================================================================
def guiding_radius(ext, L, R0):
    """Radius of the equatorial circular orbit with angular momentum L.

    Valid only for equatorial orbits: for a tilted orbit L_z = L cos(iota)
    is smaller than L, and near the ISCO this moves the root far inside the
    orbit.  The scripts therefore use the start radius instead.
    """
    def f(r):
        co = circular_orbit(ext, r)
        return np.nan if co is None else co['L'] - L

    for fac in (0.04, 0.10, 0.20):
        a, b = R0 * (1 - fac), R0 * (1 + fac)
        fa, fb = f(a), f(b)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return brentq(f, a, b, xtol=1e-9)
    return R0


def spectrum_peak(t, x, f_guess=None, rel_window=0.15, pad=8):
    """
    Peak frequency [cycles per unit of t] of a real or complex uniformly
    sampled signal: Hann window, zero padding, parabolic interpolation of
    log|F|.  If f_guess is given the search is limited to
    |f - f_guess| < rel_window * |f_guess|.
    """
    x = np.asarray(x)
    n = len(x)
    dt = t[1] - t[0]
    w = np.hanning(n)
    npad = n * pad
    if np.iscomplexobj(x):
        F = np.abs(np.fft.fft((x - x.mean()) * w, npad))
        fr = np.fft.fftfreq(npad, dt)
        o = np.argsort(fr)
        fr, F = fr[o], F[o]
    else:
        F = np.abs(np.fft.rfft((x - x.mean()) * w, npad))
        fr = np.fft.rfftfreq(npad, dt)
    sel = np.ones_like(fr, dtype=bool)
    if f_guess is not None:
        sel = np.abs(fr - f_guess) < rel_window * abs(f_guess)
    idx = np.where(sel)[0]
    if len(idx) < 3:
        return np.nan
    i = idx[np.argmax(F[idx])]
    if 0 < i < len(F) - 1:
        a, b, c = np.log(F[i - 1:i + 2] + 1e-300)
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return fr[i] + d * (fr[1] - fr[0])
    return fr[i]


def norm_spectrum(t, x, complex_sig=False, pad=4):
    n = len(x)
    dt = t[1] - t[0]
    w = np.hanning(n)
    npad = n * pad
    if complex_sig:
        F = np.abs(np.fft.fftshift(np.fft.fft((x - x.mean()) * w, npad)))
        fr = np.fft.fftshift(np.fft.fftfreq(npad, dt))
    else:
        F = np.abs(np.fft.rfft((x - x.mean()) * w, npad))
        fr = np.fft.rfftfreq(npad, dt)
    return fr, F / F.max()


def pericentre_precession(tg, R, phi):
    """
    nu_per from successive pericentres:  advance beyond 2 pi per radial
    period divided by the period  (= Omega_phi - Omega_r).  [cycles / t-unit]
    """
    d = np.diff(R)
    imin = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    if len(imin) < 3:
        return np.nan, len(imin)
    tp, pp = [], []
    for i in imin:                              # parabolic refinement
        a, b, c = R[i - 1], R[i], R[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        dt = tg[1] - tg[0]
        tp.append(tg[i] + off * dt)
        pp.append(phi[i] + off * (phi[i + 1] - phi[i - 1]) * 0.5)
    tp, pp = np.array(tp), np.array(pp)
    delta = np.diff(pp) - 2 * np.pi
    return np.sum(delta) / np.sum(np.diff(tp)) / (2 * np.pi), len(imin)


def analyse_orbit(traj, model, n_grid=2 ** 16):
    """
    Measured frequencies (Hz) of one integrated geodesic and the RPM
    prediction at the start radius r0 (the orbit starts circular plus a
    small radial kick, so it oscillates about r0 to first order).  Returns a
    dict (incl. arrays for plots).
    """
    M, j, q = model.M_geo, model.j, model.q
    R_gc = traj['r0']
    pred = qpo_at_radius(M, j, q, R_gc)

    t = traj['t']                                  # km (coordinate time)
    tg = np.linspace(t[0], t[-1], n_grid)
    cs = {k: CubicSpline(t, traj[k]) for k in ('r', 'theta', 'phi')}
    r, th, ph = cs['r'](tg), cs['theta'](tg), cs['phi'](tg)
    ts = tg / C_KM_S                               # seconds
    to_hz = lambda f_per_km: f_per_km * C_KM_S     # cycles/km -> Hz

    x, y, z = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), \
        r * np.cos(th)
    xy = r * np.sin(th) * np.exp(1j * ph)

    out = {'R_gc': R_gc, 'pred': pred, 't': ts, 'r': r, 'theta': th,
           'phi': ph, 'xy': xy, 'x': x, 'y': y, 'z': z}

    # --- FFT peaks (frequencies in Hz: t in seconds) ---
    out['nu_phi_mean'] = (ph[-1] - ph[0]) / (2 * np.pi * (ts[-1] - ts[0]))
    out['nu_phi_fft'] = spectrum_peak(ts, xy, pred['nu_phi'], 0.10)
    out['nu_r_fft'] = spectrum_peak(ts, r, pred['nu_r'], 0.15) \
        if pred['nu_r'] > 0 else np.nan
    out['nu_th_fft'] = spectrum_peak(ts, np.cos(th), pred['nu_theta'], 0.10)
    out['nu_per_fft'] = spectrum_peak(ts, xy, pred['nu_peri'], 0.12) \
        if pred['nu_peri'] > 0 else np.nan

    # --- direct precession measurements ---
    nu_per_p, n_peri = pericentre_precession(ts, r, ph)
    out['nu_per_peri'] = nu_per_p
    out['n_pericentres'] = n_peri

    # orbital angular momentum direction l = r x v  (v = d r / d t)
    sx, sy, sz = (CubicSpline(t, traj['r'] * np.sin(traj['theta'])
                              * np.cos(traj['phi'])),
                  CubicSpline(t, traj['r'] * np.sin(traj['theta'])
                              * np.sin(traj['phi'])),
                  CubicSpline(t, traj['r'] * np.cos(traj['theta'])))
    vx, vy, vz = sx(tg, 1), sy(tg, 1), sz(tg, 1)
    lx = y * vz - z * vy
    ly = z * vx - x * vz
    node = lx + 1j * ly
    ang = np.unwrap(np.angle(node))
    slope = np.polyfit(ts, ang, 1)[0] / (2 * np.pi)       # Hz
    out['node_phase'] = ang
    out['node_fit'] = np.polyval(np.polyfit(ts, ang, 1), ts)
    out['nu_nod_phase'] = slope
    out['nu_nod_diff'] = out['nu_phi_mean'] - out['nu_th_fft']
    T = ts[-1] - ts[0]
    out['T'] = T
    out['nod_cycles'] = abs(pred['nu_nod']) * T
    out['nu_nod_fft'] = (spectrum_peak(ts, node, pred['nu_nod'], 0.5, pad=16)
                         if out['nod_cycles'] > 2.5 else np.nan)
    return out


def run_one(model, mf, R0, ecc, inc_deg, n_orbits, n_pts=24000):
    tr = integrate_geodesic(mf, R0, 90.0 - inc_deg, n_orbits, n_pts=n_pts,
                            ecc=ecc, r_stop=model.r_surface)
    if tr is None or len(tr['t']) < 100:
        return None, None
    if tr.get('terminated'):
        return None, None
    return tr, analyse_orbit(tr, model)


# =====================================================================
# plots
# =====================================================================
def _lines(ax, vals, ymax=1.5):
    for v, name, col in vals:
        ax.axvline(v, color=col, ls='--', lw=1)
        ax.text(v, ymax, name, color=col, ha='left', va='bottom', fontsize=9)


def table_rows(res):
    p = res['pred']
    return [
        ('nu_r     peak of r(t)', res['nu_r_fft'], p['nu_r'], 'pct'),
        ('nu_theta peak of cos(th)', res['nu_th_fft'], p['nu_theta'], 'pct'),
        ('nu_phi   mean dphi/dt', res['nu_phi_mean'], p['nu_phi'], 'pct'),
        ('nu_phi   peak of x+iy', res['nu_phi_fft'], p['nu_phi'], 'pct'),
        ('nu_per   sideband of x+iy', res['nu_per_fft'], p['nu_peri'], 'pct'),
        ('nu_per   pericentre advance', res['nu_per_peri'], p['nu_peri'], 'pct'),
        ('nu_nod   phase of l_x+i l_y', res['nu_nod_phase'], p['nu_nod'], 'hz'),
        ('nu_nod   nu_phi - nu_theta', res['nu_nod_diff'], p['nu_nod'], 'hz'),
    ]


def fmt_table(rows):
    s = f"{'':<30s} {'geodesic':>10s} {'RPM':>10s}  diff\n"
    for n, m, p, kind in rows:
        if not (np.isfinite(m) and np.isfinite(p)):
            s += f"{n:<30s} {'n/a':>10s} {p:10.3f}\n"
        elif kind == 'pct':
            s += f"{n:<30s} {m:10.3f} {p:10.3f}  {100*(m/p-1):+8.4f}%\n"
        else:
            s += f"{n:<30s} {m:10.3f} {p:10.3f}  {m-p:+8.4f} Hz\n"
    return s


def plot_spectra(res, title, path):
    set_style()
    ts, p = res['t'], res['pred']
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5))

    f, s = norm_spectrum(ts, res['r'])
    ax = axes[0]
    ax.semilogy(f, s + 1e-12, color='#0066CC', lw=1.2)
    _lines(ax, [(p['nu_r'], r'$\nu_r$', '#CC3300'),
                (2 * p['nu_r'], r'$2\nu_r$', '#CC3300')], 1.5)
    ax.set_xlim(0, 3.3 * p['nu_r']); ax.set_ylim(1e-6, 5)
    ax.set_ylabel(r'$|\widetilde{r}|$ (norm.)')
    ax.set_title(r'FFT of $r(t)$ (radial epicycle)')

    f, s = norm_spectrum(ts, np.cos(res['theta']))
    ax = axes[1]
    ax.semilogy(f, s + 1e-12, color='#8c564b', lw=1.2)
    _lines(ax, [(p['nu_theta'], r'$\nu_\theta$', '#CC3300')], 1.5)
    ax.set_xlim(0, 1.5 * p['nu_phi']); ax.set_ylim(1e-6, 5)
    ax.set_ylabel(r'$|\widetilde{\cos\theta}|$ (norm.)')
    ax.set_title(r'FFT of $z/r=\cos\theta$ (vertical epicycle)')

    f, s = norm_spectrum(ts, res['xy'], complex_sig=True)
    ax = axes[2]
    ax.semilogy(f, s + 1e-12, color='#6600CC', lw=1.2)
    _lines(ax, [(p['nu_phi'], r'$\nu_\phi$', '#CC3300'),
                (p['nu_peri'], r'$\nu_{\rm per}=\nu_\phi-\nu_r$', '#22AA22'),
                (p['nu_phi'] + p['nu_r'], r'$\nu_\phi+\nu_r$', '#FF8800')], 1.5)
    ax.set_xlim(0, 1.8 * p['nu_phi']); ax.set_ylim(1e-6, 5)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|\widetilde{x+iy}|$ (norm.)')
    ax.set_title(r'FFT of $x+iy$ (sidebands at $\nu_\phi\pm\nu_r$)')

    fig.text(0.5, 0.005, fmt_table(table_rows(res)), ha='center', va='bottom',
             fontsize=7.5, family='monospace')
    fig.suptitle(title, fontsize=12.5)
    plt.tight_layout(rect=(0, 0.17, 1, 0.97))
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_qpo_view(res, title, path):
    """The three RPM QPOs (nu_2, nu_1, nu_HBO) read off the orbit."""
    set_style()
    ts, p = res['t'], res['pred']
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 1.0])

    # (a) kHz QPOs from x+iy
    ax = fig.add_subplot(gs[0])
    f, s = norm_spectrum(ts, res['xy'], complex_sig=True, pad=8)
    m = (f > 0.5 * p['nu_peri']) & (f < 1.2 * p['nu_phi'])
    ax.semilogy(f[m], s[m] + 1e-12, color='#6600CC', lw=1.3)
    for v, nm, col in ((p['nu_phi'], r'$\nu_2=\nu_\phi$', '#CC3300'),
                       (p['nu_peri'], r'$\nu_1=\nu_{\rm per}$', '#22AA22')):
        ax.axvline(v, color=col, ls='--', lw=1)
        ax.text(v, 2.0, nm, color=col, ha='center', fontsize=10)
    ax.axvspan(res['nu_per_fft'] - 3, res['nu_per_fft'] + 3, color='#22AA22',
               alpha=0.12) if np.isfinite(res['nu_per_fft']) else None
    ax.set_ylim(1e-6, 8)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('norm. amplitude')
    ax.set_title('(a) kHz QPOs: orbital motion + periastron sideband')

    # (b) nodal precession from the phase of l_x + i l_y
    ax = fig.add_subplot(gs[1])
    ax.plot(ts * 1e3, res['node_phase'] / (2 * np.pi), color='#000000',
            label='node longitude of $\\hat l$')
    ax.plot(ts * 1e3, np.polyval(np.polyfit(ts, res['node_phase'], 1), ts)
            / (2 * np.pi), 'r--', lw=1.2,
            label=fr"slope = {res['nu_nod_phase']:.2f} Hz")
    ax.plot(ts * 1e3, (res['node_phase'][0]
                       + 2 * np.pi * p['nu_nod'] * (ts - ts[0])) / (2 * np.pi),
            ':', color='#1f77b4', lw=1.6,
            label=fr"RPM $\nu_{{\rm nod}}$ = {p['nu_nod']:.2f} Hz")
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('node longitude (turns)')
    ax.set_title(r'(b) nodal precession $\rightarrow$ HBO $=2\nu_{\rm nod}$')
    ax.legend(fontsize=8, loc='best')

    # (c) summary of the three QPOs
    ax = fig.add_subplot(gs[2])
    names = [r'$\nu_2$', r'$\nu_1$', r'$\Delta\nu$', r'HBO']
    meas = [res['nu_phi_fft'], res['nu_per_peri'],
            res['nu_phi_fft'] - res['nu_per_fft'], 2 * res['nu_nod_phase']]
    pr = [p['nu_phi'], p['nu_peri'], p['nu_r'], 2 * p['nu_nod']]
    dev = [100 * (a / b - 1) if b > 0 and np.isfinite(a) else np.nan
           for a, b in zip(meas, pr)]
    xs = np.arange(len(names))
    ax.bar(xs, dev, color=['#CC3300', '#22AA22', '#FF8800', '#1f77b4'])
    ax.axhline(0, color='k', lw=0.8)
    for xx, d, a, b in zip(xs, dev, meas, pr):
        ax.text(xx, d, f"\n{a:.1f}\n(RPM {b:.1f})", ha='center',
                va='top' if (np.isfinite(d) and d < 0) else 'bottom',
                fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    lim = max(0.3, np.nanmax(np.abs(dev)) * 2.2)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel('geodesic / RPM - 1  (%)')
    ax.set_title('(c) RPM QPOs: geodesic vs prediction')

    fig.suptitle(title, fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# scan over radius
# =====================================================================
def fft_scan(model, mf, r_in, n, ecc, inc_deg, n_orbits, outdir):
    Rs = np.linspace(1.12 * r_in, 3.0 * r_in, n)
    keys = ['R_gc', 'nu_phi', 'nu_phi_m', 'nu_r', 'nu_r_m', 'nu_th', 'nu_th_m',
            'nu_per', 'nu_per_m', 'nu_nod', 'nu_nod_m']
    rows = []
    for R0 in Rs:
        try:
            tr, res = run_one(model, mf, R0, ecc, inc_deg, n_orbits)
            if res is None:
                print(f"    R0={R0:6.2f}: orbit terminated / failed")
                continue
        except Exception as e:                       # keep the scan going
            print(f"    R0={R0:6.2f}: skipped ({e})")
            continue
        p = res['pred']
        rows.append([res['R_gc'], p['nu_phi'], res['nu_phi_mean'],
                     p['nu_r'], res['nu_r_fft'], p['nu_theta'],
                     res['nu_th_fft'], p['nu_peri'], res['nu_per_peri'],
                     p['nu_nod'], res['nu_nod_phase']])
        print(f"    R={res['R_gc']:7.2f} km  nu_r {res['nu_r_fft']:8.2f}/"
              f"{p['nu_r']:8.2f}  nu_th {res['nu_th_fft']:8.2f}/"
              f"{p['nu_theta']:8.2f}  nu_phi {res['nu_phi_mean']:8.2f}/"
              f"{p['nu_phi']:8.2f}  nu_nod {res['nu_nod_phase']:6.2f}/"
              f"{p['nu_nod']:6.2f}")
    if not rows:
        return None
    a = np.array(rows)
    dat = os.path.join(outdir, 'fft_scan.dat')
    with open(dat, 'w') as f:
        f.write("# R_gc  | RPM, geodesic pairs (Hz): nu_phi nu_r nu_theta "
                "nu_per nu_nod\n# " + "  ".join(keys) + "\n")
        for r in a:
            f.write(" ".join(f"{x:12.5f}" for x in r) + "\n")
    print(f"  Saved: {dat}")
    return a


def plot_scan(a, model, path):
    set_style()
    Rf = np.linspace(a[:, 0].min(), a[:, 0].max(), 150)
    pf = [qpo_at_radius(model.M_geo, model.j, model.q, r) for r in Rf]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 9.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.5]})
    ax = axes[0]
    spec = [('nu_phi', 1, 2, '#1f77b4', r'$\nu_\phi$'),
            ('nu_r', 3, 4, '#d62728', r'$\nu_r$'),
            ('nu_theta', 5, 6, '#8c564b', r'$\nu_\theta$'),
            ('nu_peri', 7, 8, '#2ca02c', r'$\nu_{\rm per}$')]
    for key, ip, im, col, nm in spec:
        ax.plot(Rf, [d[key] for d in pf], '-', color=col, label=nm + ' RPM')
        ax.plot(a[:, 0], a[:, im], 'o', color=col, mfc='none', ms=8,
                label=nm + ' geodesic')
    ax.plot(Rf, [d['nu_nod'] for d in pf], '-', color='k', label=r'$\nu_{\rm nod}$ RPM')
    ax.plot(a[:, 0], a[:, 10], 'o', color='k', mfc='none', ms=8,
            label=r'$\nu_{\rm nod}$ geodesic')
    ax.set_yscale('log')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title('Geodesic (FFT / precession fits) vs RPM over radius')
    ax.legend(ncol=2, fontsize=8)
    ax = axes[1]
    for key, ip, im, col, nm in spec:
        ax.plot(a[:, 0], 100 * (a[:, im] / a[:, ip] - 1), 'o-', color=col,
                label=nm)
    ax.plot(a[:, 0], 100 * (a[:, 10] / a[:, 9] - 1), 'o-', color='k',
            label=r'$\nu_{\rm nod}$')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'guiding radius $R_{\rm gc}$ (km)')
    ax.set_ylabel('geodesic/RPM - 1 (%)')
    ax.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation(a, model, r_in, path):
    """Geodesic points on the (nu_1, nu_2) and (nu_1, nu_HBO) planes."""
    set_style()
    Rf = np.linspace(r_in * 1.0, a[:, 0].max() * 1.15, 200)
    pf = [qpo_at_radius(model.M_geo, model.j, model.q, r) for r in Rf]
    nu1 = np.array([d['nu_peri'] for d in pf])
    nu2 = np.array([d['nu_phi'] for d in pf])
    hbo = 2 * np.array([d['nu_nod'] for d in pf])
    ok = nu1 > 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.plot(nu1[ok], nu2[ok], 'k-', label='RPM (qpo.py)')
    ax.plot(a[:, 8], a[:, 2], 'o', color='#d62728', mfc='none', ms=9,
            label='integrated geodesics')
    ax.set_xlabel(r'$\nu_1=\nu_{\rm per}$ (Hz)')
    ax.set_ylabel(r'$\nu_2=\nu_\phi$ (Hz)')
    ax.set_title(r'$\nu_2$ vs $\nu_1$')
    ax.legend()
    ax = axes[1]
    ax.plot(nu1[ok], hbo[ok], 'k-', label='RPM (qpo.py)')
    ax.plot(a[:, 8], 2 * a[:, 10], 'o', color='#1f77b4', mfc='none', ms=9,
            label='integrated geodesics')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\nu_1=\nu_{\rm per}$ (Hz)')
    ax.set_ylabel(r'$\nu_{\rm HBO}=2\nu_{\rm nod}$ (Hz)')
    ax.set_title('HBO vs lower kHz QPO')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# amplitude test (optional)
# =====================================================================
def amplitude_test(model, mf, R0, n_orbits, path, ecc0=0.003, inc0=0.4):
    """
    Peak shifts versus overall amplitude: radial kick and inclination are
    scaled together by s.  The shifts must vanish like s^2 (non-linear
    epicycles), which separates real discrepancies from finite amplitude.
    """
    set_style()
    scales = [1.0, 2.0, 4.0, 8.0, 16.0]
    rows = []
    for sc in scales:
        tr, res = run_one(model, mf, R0, ecc0 * sc, inc0 * sc, n_orbits)
        if res is None:
            continue
        p = res['pred']
        rows.append((sc, 100 * (res['nu_r_fft'] / p['nu_r'] - 1),
                     100 * (res['nu_th_fft'] / p['nu_theta'] - 1),
                     100 * (res['nu_phi_mean'] / p['nu_phi'] - 1)))
        print(f"    scale s = {sc:5.1f} (ecc {ecc0*sc:.4f}, inc {inc0*sc:.1f} deg):"
              f" nu_r {rows[-1][1]:+.4f}%  nu_th {rows[-1][2]:+.4f}%  "
              f"nu_phi {rows[-1][3]:+.4f}%")
    if not rows:
        return
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(a[:, 0], np.abs(a[:, 1]), 'o-', color='#d62728', label=r'$\nu_r$')
    ax.plot(a[:, 0], np.abs(a[:, 2]), 's-', color='#8c564b', label=r'$\nu_\theta$')
    ax.plot(a[:, 0], np.abs(a[:, 3]), '^-', color='#1f77b4',
            label=r'$\nu_\phi$ (mean)')
    ref = np.abs(a[0, 2]) * (a[:, 0] / a[0, 0]) ** 2
    ax.plot(a[:, 0], ref, 'k:', label=r'$\propto s^2$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'amplitude scale s  (ecc = {ecc0:g} s, inclination = {inc0:g} s deg)')
    ax.set_ylabel('|measured/RPM - 1|  (%)')
    ax.set_title(f'Finite-amplitude shift at R0 = {R0:.1f} km (expected ~ s^2)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =====================================================================
# main
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='FFT study of integrated HT geodesics vs the RPM')
    p.add_argument('--eos', default='eos/eosAU')
    p.add_argument('--rho_c', type=float, default=None)
    p.add_argument('--mass', type=float, default=None)
    p.add_argument('--radius', type=float, default=None)
    p.add_argument('--spin', type=float, default=400.0)
    p.add_argument('--outdir', default='fft_out')
    p.add_argument('--R0', type=float, default=None,
                   help='start radius [km] (default 1.6 r_ISCO)')
    p.add_argument('--ecc', type=float, default=0.003,
                   help='radial kick (fraction of orbital speed)')
    p.add_argument('--inc-deg', type=float, default=0.4,
                   help='orbital inclination [deg]')
    p.add_argument('--n-orbits', type=float, default=60.0)
    p.add_argument('--scan-n', type=int, default=6,
                   help='radii in the scan (0 = skip)')
    p.add_argument('--scan-n-orbits', type=float, default=40.0)
    p.add_argument('--amp-test', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 72)
    print("  FFT of integrated Hartle-Thorne geodesics vs RPM")
    print("=" * 72)
    eos, model, rho_c = build_model(args)
    print(f"  M = {model.M_solar:.4f} Msun (M_TOV = {model.M0_solar:.4f})   "
          f"R_eq = {model.r_surface:.3f} km   spin = {args.spin:.0f} Hz")
    print(f"  j = {model.j:+.5f}   q = {model.q:+.5f} (Geroch-Hansen sign)   "
          f"rho_c = {rho_c:.4e}")

    r_in, edge = ht_common.inner_edge(model)
    inner = 'ISCO' if edge == 'isco' else 'stellar surface'
    print(f"  inner edge = {inner}: {r_in:.3f} km "
          f"(x = {r_in / model.M_geo:.3f} M)")

    mf = HTMetricFull(model.exterior)
    R0 = args.R0 if args.R0 else 1.6 * r_in
    tag = (f"{os.path.basename(args.eos)}, M={model.M_solar:.2f} Msun, "
           f"{args.spin:.0f} Hz")

    # ---- single orbit ----
    print(f"\n[1] Small-amplitude orbit: R0 = {R0:.2f} km, ecc = {args.ecc}, "
          f"inc = {args.inc_deg} deg, {args.n_orbits:g} orbits")
    tr, res = run_one(model, mf, R0, args.ecc, args.inc_deg, args.n_orbits,
                      n_pts=40000)
    if res is None:
        raise SystemExit("integration failed or hit the star; "
                         "increase --R0 or reduce --ecc")
    print(f"    guiding radius R_gc = {res['R_gc']:.3f} km "
          f"(r range {tr['r'].min():.3f} .. {tr['r'].max():.3f} km); "
          f"duration {res['T'] * 1e3:.1f} ms")
    print("\n" + fmt_table(table_rows(res)))
    if res['nod_cycles'] < 2.5:
        print(f"    note: only {res['nod_cycles']:.2f} nodal cycles in this run, "
              "so nu_nod is taken from the phase slope of l_x + i l_y "
              "(an FFT peak would be unresolved)")
    plot_spectra(res, f'FFT of an integrated geodesic vs RPM — {tag}, '
                      f'$R_{{gc}}$ = {res["R_gc"]:.2f} km',
                 os.path.join(args.outdir, 'fft_spectra.png'))
    plot_qpo_view(res, f'RPM QPOs from an integrated geodesic — {tag}, '
                       f'$R_{{gc}}$ = {res["R_gc"]:.2f} km',
                  os.path.join(args.outdir, 'fft_qpo_view.png'))

    # ---- scan ----
    if args.scan_n > 0:
        print(f"\n[2] Scan over {args.scan_n} radii")
        a = fft_scan(model, mf, r_in, args.scan_n, args.ecc, args.inc_deg,
                     args.scan_n_orbits, args.outdir)
        if a is not None and len(a) >= 2:
            plot_scan(a, model, os.path.join(args.outdir, 'fft_scan.png'))
            plot_correlation(a, model, r_in,
                             os.path.join(args.outdir,
                                          'fft_qpo_correlation.png'))

    # ---- amplitude test ----
    if args.amp_test:
        print("\n[3] Amplitude test")
        amplitude_test(model, mf, R0, args.n_orbits,
                       os.path.join(args.outdir, 'fft_amplitude_test.png'),
                       ecc0=args.ecc, inc0=args.inc_deg)

    print("\n" + "=" * 72)
    print(f"  Done. Outputs in {args.outdir}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
