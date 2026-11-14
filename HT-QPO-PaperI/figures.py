#!/usr/bin/env python3
"""
figures.py -- all figures and tables of Paper I (slow rotation, HT).

    fig_stars.pdf      M-R, j(M), Qbar(M), ISCO and stellar radius
    fig_freqs.pdf      epicyclic frequencies and nodal precession vs radius
    fig_nodal.pdf      HT/Kerr nodal ratio and saturation with spin
    fig_orbit.pdf      3D geodesic, epicycle (top view), precession angles
    fig_psd.pdf        hot-spot light curve and its power spectrum
    fig_tracks_atoll.pdf, fig_tracks_z.pdf   data and best RPM tracks
    fig_corner.pdf     4U 1728-34, EOS-free fit (M, j, Qbar, I)
    tab_eos.tex, tab_fits.tex

Inputs: qpo_fit_cache/ (frequency and star tables), results/orbit.npz
(orbit_qpo.py), grid_fit_out/ (qpo_grid_fit.py) and the Bilby posterior of
the EOS-free 4U 1728-34 fit (qpo_fit.py).

Usage:  python3 figures.py [--out paper_out] [--only stars,orbit,...]
"""
import argparse
import glob
import json
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter

import plot_style as ps
from plot_style import S, lab, finish, panel_tag, label_panels, COL, LS, GREY
from qpo_fit import FreqTable, StarTable, hz_curves, QPOSource, M_SUN_KM, \
    C_KM_S
from qpo import qpo_at_radius, _epi_signed

OUT = 'paper_out'
logging.getLogger('fontTools').setLevel(logging.ERROR)   # quiet font embedding
EOS = [(t, ps.EOS_LABEL[t]) for t in ps.EOS_ORDER]
GDIR = 'grid_fit_out'
I45 = 1.3466e43 / 1e45          # I [km^3] -> [1e45 g cm^2]
INFO = {}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('  saved', os.path.join(OUT, name))


def mlab(sym, unit=None):
    return lab(S[sym] if sym in S else sym, unit)


MSUN = r'$\boldsymbol{M}_{\boldsymbol{\odot}}$'


# =====================================================================
# 1. Stars
# =====================================================================
def star_sequence(st, nu, Mgrid):
    rows = []
    for M in Mgrid:
        s = st.star(M, nu)
        if s is None:
            continue
        rows.append([M, s['R'], s['R_eq_circ'], s['j'], s['q_over_j2'],
                     s['x_surf'], s['I'], s['q']])
    return np.array(rows)


def fig_stars(ftab):
    Mg = np.linspace(1.0, 2.25, 126)
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    rows = []
    for k, (tag, name) in enumerate(EOS):
        st = StarTable(f'eos/{tag}')
        a0 = star_sequence(st, 0.0, Mg)
        a3 = star_sequence(st, 363.0, Mg)
        a5 = star_sequence(st, 581.0, Mg)
        kw = dict(color=COL[k], ls=LS[k])
        ax[0, 0].plot(a0[:, 1], a0[:, 0], label=name, **kw)
        ax[0, 1].plot(a3[:, 0], a3[:, 3], **kw)
        ax[0, 1].plot(a5[:, 0], a5[:, 3], color=COL[k], ls=LS[k], lw=0.9)
        ax[1, 0].plot(a3[:, 0], -a3[:, 4], **kw)
        xin = []
        for M, j, q in zip(a5[:, 0], a5[:, 3], a5[:, 7]):
            xin.append(ftab.curves(j, q, 0.0, 25.0, 50)['x_in'])
        ax[1, 1].plot(a5[:, 0], xin, **kw)
        ax[1, 1].plot(a5[:, 0], a5[:, 5], color=COL[k], ls=LS[k], lw=0.9)
        s14, s19 = st.star(1.4, 0.0), st.star(1.9, 363.0)
        s14r = st.star(1.4, 363.0)
        rows.append((name, st.M_max(0.0), s14['R'], s19['R'],
                     s14r['I'] / M_SUN_KM ** 3 / 1.4 ** 3, s19['I'] * I45,
                     s19['j'], -s19['q_over_j2'], s19['x_surf']))
    ax[0, 0].set_xlabel(mlab('R', 'km'))
    ax[0, 0].set_ylabel(lab(S['M'], MSUN))
    ax[0, 0].set_xlim(9.5, 14.5)
    ax[0, 0].set_ylim(1.0, 2.3)
    ax[0, 0].legend(loc='lower left')
    ax[0, 1].set_xlabel(lab(S['M'], MSUN))
    ax[0, 1].set_ylabel(mlab('j'))
    ax[0, 1].text(0.05, 0.90, r'$\boldsymbol{\nu}_{\boldsymbol{s}}=581$ Hz (thin)',
                  transform=ax[0, 1].transAxes, fontsize=9)
    ax[0, 1].text(0.05, 0.80, r'$\boldsymbol{\nu}_{\boldsymbol{s}}=363$ Hz (thick)',
                  transform=ax[0, 1].transAxes, fontsize=9)
    ax[1, 0].set_xlabel(lab(S['M'], MSUN))
    ax[1, 0].set_ylabel(lab(S['Qbar'] + r'=-\boldsymbol{q}/\boldsymbol{j}^{2}'))
    ax[1, 0].axhline(1.0, color=GREY, lw=1.0, ls=(0, (1, 2)))
    ax[1, 0].text(1.05, 1.25, 'Kerr', fontsize=9, color=GREY)
    ax[1, 0].set_ylim(0, 9)
    ax[1, 1].set_xlabel(lab(S['M'], MSUN))
    ax[1, 1].set_ylabel(lab(r'\boldsymbol{r}/\boldsymbol{M}'))
    ax[1, 1].text(0.05, 0.08, r'thick: ISCO;  thin: $\boldsymbol{R}_{\mathbf{eq}}/\boldsymbol{M}$'
                  '\n' r'($\boldsymbol{\nu}_{\boldsymbol{s}}=581$ Hz)',
                  transform=ax[1, 1].transAxes, fontsize=9)
    for a in ax.ravel():
        finish(a)
    fig.tight_layout()
    label_panels(fig, ax)
    save(fig, 'fig_stars.pdf')

    with open(os.path.join(OUT, 'tab_eos.tex'), 'w') as f:
        f.write('\\begin{table*}[t]\n\\caption{\\label{tab:eos}Stellar '
                'properties for the four EOSs.  $M_{\\max}$ is the maximum '
                'mass of nonrotating stars, $R_{1.4}$ and $R_{1.9}$ are '
                'nonrotating radii, and $\\bar I_{1.4}=I/M^{3}$ is given at '
                '$1.4\\,M_\\odot$.  $I$, $j$, $\\bar Q=-q/j^{2}$ and '
                '$R_{\\rm eq}/M$ are given at $M=1.9\\,M_\\odot$ and '
                '$\\nu_s=363$~Hz.}\n\\begin{ruledtabular}\n'
                '\\begin{tabular}{lcccccccc}\n'
                'EOS & $M_{\\max}\\,[M_\\odot]$ & $R_{1.4}$ [km] & $R_{1.9}$ '
                '[km] & $\\bar I_{1.4}$ & $I_{1.9}\\,[10^{45}\\,{\\rm g\\,cm^2}]$ '
                '& $j_{1.9}$ & $\\bar Q_{1.9}$ & $R_{\\rm eq}/M$ \\\\\n'
                '\\hline\n')
        for r in rows:
            f.write(f'{r[0]} & {r[1]:.3f} & {r[2]:.2f} & {r[3]:.2f} & '
                    f'{r[4]:.2f} & {r[5]:.2f} & {r[6]:.3f} & {r[7]:.2f} & '
                    f'{r[8]:.2f} \\\\\n')
        f.write('\\end{tabular}\n\\end{ruledtabular}\n\\end{table*}\n')
    INFO['stars'] = rows
    print('  saved tab_eos.tex')


# =====================================================================
# 2. Frequencies
# =====================================================================
def fig_freqs(ftab, eos='IOPB', M=1.922, nu=363.0):
    st = StarTable(f'eos/{eos}').star(M, nu)
    j, q = st['j'], st['q']
    Mkm = M * M_SUN_KM
    fc = ftab.curves(j, q, st['x_surf'], 20.0, 50)
    xs = np.linspace(fc['x_in'] * 1.0005, 20.0, 300)
    d = {k: np.array([qpo_at_radius(Mkm, j, q, x * Mkm)[k] for x in xs])
         for k in ('nu_phi', 'nu_theta', 'nu_r', 'nu_peri', 'nu_nod')}
    nuK = C_KM_S / (2 * np.pi * Mkm) * xs ** -1.5
    nuLT = nuK * 2 * j * xs ** -1.5
    nuQ = nuK * 1.5 * q * xs ** -2
    r_km = xs * Mkm
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for i, (k, s) in enumerate([('nu_phi', S['nuphi'] + '=' + S['nuU']),
                                ('nu_theta', S['nuth']),
                                ('nu_r', S['nur']),
                                ('nu_peri', S['nuper'] + '=' + S['nuL'])]):
        ax[0].plot(r_km, d[k], color=COL[i], ls=LS[i], label=lab(s))
    ax[0].set_yscale('log')
    ax[0].set_ylim(60, 3000)
    ax[0].legend(loc='upper right', ncol=1)
    ax[1].plot(r_km, 2 * d['nu_nod'], color=COL[0], label=lab(
        r'2' + S['nunod'] + r'\ (\mathbf{HT})'))
    ax[1].plot(r_km, 2 * nuLT, color=COL[1], ls='--', label=lab(
        r'2' + S['nuLT']))
    ax[1].plot(r_km, -2 * nuQ, color=COL[2], ls='-.', label=lab(
        r'-2' + S['nuQ']))
    ax[1].plot(r_km, 2 * (nuLT + nuQ), color=COL[3], ls=':', label=lab(
        r'2(' + S['nuLT'] + '+' + S['nuQ'] + r')'))
    ax[1].set_yscale('log')
    ax[1].set_ylim(1.0, 150)
    ax[1].legend(loc='upper right')
    for a in ax:
        a.axvspan(0, st['R_eq'], color='#e8e8e8', lw=0)
        a.axvline(fc['x_in'] * Mkm, color=GREY, lw=1.0, ls=(0, (1, 2)))
        a.set_xlim(st['R_eq'] - 1.0, 40.0)
        a.set_xlabel(mlab('r', 'km'))
        a.set_ylabel(lab(r'\boldsymbol{\nu}', 'Hz'))
        a.text(fc['x_in'] * Mkm + 0.4, a.get_ylim()[0] * 1.25, 'ISCO',
               fontsize=9, color=GREY)
        finish(a)
    fig.tight_layout()
    label_panels(fig, ax)
    save(fig, 'fig_freqs.pdf')
    INFO['freqs'] = {'eos': eos, 'M': M, 'nu_s': nu, 'j': j, 'q': q,
                     'Qbar': -q / j ** 2, 'x_isco': fc['x_in'],
                     'r_isco': fc['x_in'] * Mkm, 'R_eq': st['R_eq']}


def fig_nodal(ftab, M=1.9, nu=363.0):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    xs = np.linspace(5.6, 14.0, 80)
    out = {}
    for k, (tag, name) in enumerate(EOS):
        stt = StarTable(f'eos/{tag}')
        st = stt.star(M, nu)
        j, q = st['j'], st['q']
        rat = []
        for x in xs:
            Op, fr, ft = _epi_signed(x, j, q)
            Opk, frk, ftk = _epi_signed(x, j, -j * j)
            rat.append((1 - np.sqrt(ft)) / (1 - np.sqrt(ftk)) * Op / Opk)
        ax[0].plot(xs, rat, color=COL[k], ls=LS[k], label=name)
        nus = np.linspace(150, 750, 25)
        y = []
        for n in nus:
            s = stt.star(M, n)
            if s is None:
                y.append(np.nan)
                continue
            fc = ftab.curves(s['j'], s['q'], s['x_surf'], 25.0, 1500)
            cur = hz_curves(fc, M)
            y.append(np.interp(900.0, cur['nu_U'][::-1], cur['nu_HBO'][::-1]))
        y = np.array(y)
        ax[1].plot(nus, y, color=COL[k], ls=LS[k], label=name)
        out[name] = {'ratio_min': float(np.min(rat)),
                     'ratio_max': float(np.max(rat)),
                     'max_2nod_900': float(np.nanmax(y)),
                     'nu_at_max': float(nus[np.nanargmax(y)])}
    ax[0].axhline(1.0, color=GREY, lw=1.0, ls=(0, (1, 2)))
    ax[0].set_xlabel(lab(r'\boldsymbol{x}=\boldsymbol{r}/\boldsymbol{M}'))
    ax[0].set_ylabel(lab(S['nunod'] + r'^{\mathbf{HT}}/' + S['nunod']
                         + r'^{\mathbf{Kerr}}'))
    ax[0].legend(loc='lower right')
    ax[0].text(0.05, 0.90, r'$\boldsymbol{M}=1.9\,\boldsymbol{M}_{\boldsymbol{\odot}}$,'
               r' $\boldsymbol{\nu}_{\boldsymbol{s}}=363$ Hz',
               transform=ax[0].transAxes, fontsize=9)
    ax[1].set_xlabel(mlab('nus', 'Hz'))
    ax[1].set_ylabel(lab(r'2' + S['nunod'], 'Hz'))
    ax[1].text(0.05, 0.90, r'$\boldsymbol{M}=1.9\,\boldsymbol{M}_{\boldsymbol{\odot}}$,'
               r' $\boldsymbol{\nu}_{\mathbf{U}}=900$ Hz',
               transform=ax[1].transAxes, fontsize=9)
    for a in ax:
        finish(a)
    fig.tight_layout()
    label_panels(fig, ax)
    save(fig, 'fig_nodal.pdf')
    INFO['nodal'] = out


# =====================================================================
# 3. Orbit and QPO peaks
# =====================================================================
def _sphere(ax, R, color='#c9c9c9'):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v),
                    R * np.cos(v), color=color, alpha=0.55, lw=0,
                    shade=True, rstride=1, cstride=1)


def fig_orbit(path='results/orbit.npz', t3d=0.02):
    d = np.load(path)
    t, r, th, ph = d['t'], d['r'], d['theta'], d['phi']
    x, y, z = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), \
        r * np.cos(th)
    R_eq, r_isco, r0 = float(d['R_eq']), float(d['r_isco']), float(d['r0'])
    pred = dict(zip([str(n) for n in d['names']], d['pred']))

    fig = plt.figure(figsize=(7.2, 3.3))
    # (a) 3D orbit, true scale
    ax = fig.add_axes([-0.04, 0.10, 0.48, 0.92], projection='3d')
    sel = t <= t3d
    pts = np.array([x[sel], y[sel], z[sel]]).T.reshape(-1, 1, 3)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap='viridis', lw=1.0)
    lc.set_array(t[sel][:-1] * 1e3)
    ax.add_collection(lc)
    _sphere(ax, R_eq)
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_isco * np.cos(a), r_isco * np.sin(a), 0 * a, color=GREY,
            lw=1.0, ls='--')
    ax.plot([0, 0], [0, 0], [R_eq, 1.75 * R_eq], color='k', lw=1.8)
    ax.text(0, 0, 1.85 * R_eq, 'spin axis', fontsize=9, ha='center')
    L = 19.0
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L * 0.6, L * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=14, azim=-58)
    ax.set_xlabel(mlab('x', 'km'), labelpad=-3, fontsize=10.5)
    ax.set_ylabel(mlab('y', 'km'), labelpad=-3, fontsize=10.5)
    ax.set_zlabel(mlab('z', 'km'), labelpad=-6, fontsize=10.5)
    ax.tick_params(labelsize=8, pad=-2)
    ax.set_xticks([-15, 0, 15]); ax.set_yticks([-15, 0, 15])
    ax.set_zticks([-10, 0, 10])
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.set_facecolor((1, 1, 1, 0))
        p.pane.set_edgecolor('#bbbbbb')
        p._axinfo['grid']['color'] = (0.92, 0.92, 0.92, 1)
        p._axinfo['tick']['inward_factor'] = 0.4
        p._axinfo['tick']['outward_factor'] = 0.0
    cax = fig.add_axes([0.07, 0.10, 0.22, 0.03])
    cb = fig.colorbar(lc, cax=cax, orientation='horizontal')
    cb.set_label(lab(S['t'], 'ms'), fontsize=10, labelpad=1)
    cb.ax.tick_params(labelsize=8.5, direction='in', which='both')
    ax.text2D(0.08, 0.93, '(a)', transform=ax.transAxes, fontsize=11)

    # (b) radial and vertical epicycles
    tm = 8e-3
    m = t <= tm
    bx = fig.add_axes([0.52, 0.60, 0.185, 0.32])
    ex = fig.add_axes([0.52, 0.13, 0.185, 0.32])
    bx.plot(t[m] * 1e3, r[m], color='k', lw=1.1)
    bx.axhline(r0, color=COL[1], lw=1.0, ls='--')          # start radius r0
    bx.set_ylabel(mlab('r', 'km'), labelpad=1)
    bx.set_xticklabels([])
    bx.set_title(lab(r'1/' + S['nur']) + f' = {1e3 / pred["nu_r"]:.2f} ms',
                 loc='right', fontsize=8.5, pad=2)
    ex.plot(t[m] * 1e3, z[m], color='k', lw=1.1)
    ex.set_xlabel(lab(S['t'], 'ms'), labelpad=1)
    ex.set_ylabel(mlab('z', 'km'), labelpad=1)
    ex.set_title(lab(r'1/' + S['nuth']) + f' = {1e3 / pred["nu_theta"]:.3f} ms',
                 loc='right', fontsize=8.5, pad=2)
    for a_ in (bx, ex):
        a_.set_xlim(0, tm * 1e3)
        a_.tick_params(labelsize=8.5)
        finish(a_)

    # (d), (e) longitude of pericentre and of the ascending node
    cx = fig.add_axes([0.795, 0.60, 0.195, 0.32])
    dx = fig.add_axes([0.795, 0.13, 0.195, 0.32])
    tp = d['t_p']
    w = (d['phi_p'] - 2 * np.pi * np.arange(len(tp))) / (2 * np.pi)
    mm = tp <= 0.02
    cx.plot(tp[mm] * 1e3, w[mm], 'o', ms=3.2, mfc='w', mec='k', mew=1.0,
            zorder=5)
    tt = np.linspace(0, 20, 50)
    cx.plot(tt, w[0] + pred['nu_peri'] * (tt * 1e-3 - tp[0]), color=COL[1],
            lw=1.5)
    cx.set_xlabel(lab(S['t'], 'ms'), labelpad=0)
    cx.set_ylabel(lab(r'\boldsymbol{\varpi}/2\boldsymbol{\pi}'), labelpad=1)
    cx.text(0.06, 0.82, lab(r'\mathbf{slope}=' + S['nuper']),
            transform=cx.transAxes, fontsize=8.5, color=COL[1])
    tn, node = d['tn'], d['node'] / (2 * np.pi)
    dx.plot(tn, node - node[0], color='k', lw=1.1)
    dx.plot(tn, pred['nu_nod'] * (tn - tn[0]), color=COL[0], lw=1.5,
            ls='--')
    dx.set_xlabel(lab(S['t'], 's'), labelpad=1)
    dx.set_ylabel(lab(r'\boldsymbol{\Omega}_{\mathbf{N}}/2\boldsymbol{\pi}'),
                  labelpad=1)
    dx.text(0.06, 0.82, lab(r'\mathbf{slope}=' + S['nunod']),
            transform=dx.transAxes, fontsize=8.5, color=COL[0])
    for a_ in (cx, dx):
        a_.tick_params(labelsize=8.5)
        finish(a_)
    label_panels(fig, [bx, ex], ['(b)', '(c)'], fontsize=10, where='top')
    label_panels(fig, [cx, dx], ['(d)', '(e)'], fontsize=10)
    save(fig, 'fig_orbit.pdf')
    INFO['orbit'] = {k: float(d[k]) for k in ('M_sun', 'j', 'q', 'R_eq',
                     'r_isco', 'r0', 'inc', 'ecc', 'iobs', 'M_km',
                     'nu_nod_dir', 'nu_per_dir')}
    INFO['orbit']['r_mean'] = float(d['r_mean']) if 'r_mean' in d.files else None
    INFO['orbit']['r_min'] = float(r.min())
    INFO['orbit']['r_max'] = float(r.max())
    INFO['orbit']['z_max'] = float(np.abs(z).max())
    INFO['orbit']['pred'] = pred
    INFO['orbit']['psd_peaks'] = dict(zip([str(n) for n in d['names']],
                                          map(float, d['meas'])))


def fig_psd(path='results/orbit.npz', obs=None):
    d = np.load(path)
    pred = dict(zip([str(n) for n in d['names']], d['pred']))
    obs = obs or {'LF': (38.60, 1.715), 'L': (734.94, 1.235),
                  'U': (1098.42, 11.015)}
    fig = plt.figure(figsize=(7.2, 2.8))
    a0 = fig.add_axes([0.075, 0.17, 0.24, 0.76])
    a1 = fig.add_axes([0.40, 0.17, 0.585, 0.76])
    ts, F = d['ts'], d['F']
    m = ts - ts[0] <= 0.006
    a0.plot((ts[m] - ts[0]) * 1e3, F[m] / F.mean(), color='k', lw=1.1)
    a0.set_xlabel(lab(S['t'] + r'_{\mathbf{obs}}', 'ms'), labelpad=1)
    a0.set_ylabel(lab(r'\boldsymbol{F}/\langle\boldsymbol{F}\rangle'))
    a0.set_xlim(0, 6)
    f, P = d['f'], d['P']
    s = (f > 4) & (f < 3000)
    a1.plot(f[s], P[s], color='k', lw=0.8)
    a1.set_xscale('log'); a1.set_yscale('log')
    a1.set_xlim(8, 3000); a1.set_ylim(1e-8, 8)
    marks = [('nu_nod', S['nunod'], COL[0], 3.5, 'left'),
             ('nu_nod2', '2' + S['nunod'], COL[0], 3.5, 'left'),
             ('nu_r', S['nur'], GREY, 3.5, 'left'),
             ('nu_peri', S['nuper'], COL[1], 3.5, 'right'),
             ('nu_theta', S['nuth'], GREY, 4e-3, 'right'),
             ('nu_phi', S['nuphi'], COL[2], 3.5, 'left')]
    pred['nu_nod2'] = 2 * pred['nu_nod']
    for k, s_, c, yt, ha in marks:
        a1.axvline(pred[k], color=c, lw=1.0, zorder=0,
                   ls=':' if k == 'nu_nod2' else '--')
        xo = 0.96 if ha == 'right' else 1.04
        a1.text(pred[k] * xo, yt, lab(s_), fontsize=9, color=c,
                rotation=90, ha=ha, va='top')
    for key, (v, e) in obs.items():
        a1.axvspan(v - max(e, 0.012 * v), v + max(e, 0.012 * v),
                   color=COL[3], alpha=0.45, lw=0, zorder=0)
    a1.text(obs['LF'][0] * 1.06, 2.5e-8, r'$\mathbf{LF}$', fontsize=9,
            color='#a06d00')
    a1.text(obs['L'][0] * 0.955, 2.5e-8, r'$\mathbf{L}$', fontsize=9,
            color='#a06d00', ha='right')
    a1.text(obs['U'][0] * 1.06, 2.5e-8, r'$\mathbf{U}$', fontsize=9,
            color='#a06d00')
    a1.set_xlabel(lab(S['f'], 'Hz'), labelpad=1)
    a1.set_ylabel(lab(r'\boldsymbol{P}(\boldsymbol{f})/\boldsymbol{P}_{\mathbf{max}}'))
    for a in (a0, a1):
        finish(a)
    label_panels(fig, [a0], ['(a)'], where='top')
    label_panels(fig, [a1], ['(b)'])
    save(fig, 'fig_psd.pdf')


# =====================================================================
# 4. Data and tracks
# =====================================================================
def load_fit(src, tag, label=''):
    p = os.path.join(GDIR, 'eos', src, f'{tag}{label}.json')
    return json.load(open(p)) if os.path.isfile(p) else None


def model_curve(ftab, tag, M, nu, xmax=25.0):
    st = StarTable(f'eos/{tag}').star(M, nu)
    if st is None:
        return None
    fc = ftab.curves(st['j'], st['q'], st['x_surf'], xmax, 800)
    return hz_curves(fc, M)


def errbar(ax, x, y, xe, ye):
    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', ms=3.4, mfc='k', mec='k',
                ecolor='#8a8a8a', elinewidth=0.9, capsize=1.5, capthick=0.9,
                zorder=5)


FITS = {'4U1636-53': ('', 581.0), '4U1728-34': ('_frac2', 363.0),
        'GX5-1': ('_frac2', None), 'GX340+0': ('_frac2', None)}


def fig_tracks_atoll(ftab):
    s36 = QPOSource('4U1636-53', 'data/4U1636-53.csv')
    s28 = QPOSource('4U1728-34', 'data/4U1728-34.csv')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.75))
    d = s36.data
    errbar(ax[0], d['nu_U'], d['nu_L'], d['sig_U'], d['sig_L'])
    d2 = s28.data
    errbar(ax[1], d2['nu_U'], d2['nu_L'], d2['sig_U'], d2['sig_L'])
    errbar(ax[2], d2['nu_L'], d2['nu_HBO'], d2['sig_L'], d2['sig_HBO'])
    for k, (tag, name) in enumerate(EOS):
        f = load_fit('4U1636-53', tag)
        c = model_curve(ftab, tag, f['best']['M'], 581.0)
        ax[0].plot(c['nu_U'], c['nu_L'], color=COL[k], ls=LS[k],
                   label=f"{name}, {f['best']['M']:.2f}")
        f = load_fit('4U1728-34', tag, '_frac2')
        c = model_curve(ftab, tag, f['best']['M'], 363.0)
        ax[1].plot(c['nu_U'], c['nu_L'], color=COL[k], ls=LS[k],
                   label=f"{name}, {f['best']['M']:.3f}")
        ax[2].plot(c['nu_L'], c['nu_HBO'], color=COL[k], ls=LS[k])
    ax[0].set_xlim(820, 1340); ax[0].set_ylim(480, 1050)
    ax[1].set_xlim(1000, 1180); ax[1].set_ylim(640, 880)
    ax[2].set_xlim(640, 880); ax[2].set_ylim(25, 55)
    ax[0].set_title(r'4U 1636$-$53,  $\boldsymbol{\nu}_{\boldsymbol{s}}=581$ Hz',
                    fontsize=10)
    ax[1].set_title(r'4U 1728$-$34,  $\boldsymbol{\nu}_{\boldsymbol{s}}=363$ Hz',
                    fontsize=10)
    ax[2].set_title(r'4U 1728$-$34,  LF QPO', fontsize=10)
    for a in ax[:2]:
        a.set_xlabel(mlab('nuU', 'Hz'))
        a.set_ylabel(mlab('nuL', 'Hz'))
    ax[2].set_xlabel(mlab('nuL', 'Hz'))
    ax[2].set_ylabel(mlab('nuLF', 'Hz'))
    ax[0].legend(loc='lower right', fontsize=7.2, handlelength=1.8,
                 borderaxespad=0.3)
    ax[1].legend(loc='upper left', fontsize=7.5)
    for a in ax:
        a.tick_params(labelsize=9.5)
        finish(a)
    fig.tight_layout(w_pad=0.8)
    label_panels(fig, ax, fontsize=10.5)
    save(fig, 'fig_tracks_atoll.pdf')


def fig_tracks_z(ftab):
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for r, sname, title in ((0, 'GX5-1', r'GX 5$-$1'),
                            (1, 'GX340+0', 'GX 340+0')):
        s = QPOSource(sname, f'data/{sname}.csv')
        d = s.data
        errbar(ax[r, 0], d['nu_U'], d['nu_L'], d['sig_U'], d['sig_L'])
        errbar(ax[r, 1], d['nu_L'], d['nu_HBO'], d['sig_L'], d['sig_HBO'])
        for k, (tag, name) in enumerate(EOS):
            b = load_fit(sname, tag, '_frac2')['best']
            c = model_curve(ftab, tag, b['M'], b['nu_s'])
            ax[r, 0].plot(c['nu_U'], c['nu_L'], color=COL[k], ls=LS[k],
                          label=f"{name}, {b['M']:.2f}, {b['nu_s']:.0f} Hz")
            ax[r, 1].plot(c['nu_L'], c['nu_HBO'], color=COL[k], ls=LS[k])
        b = load_fit(sname, 'IOPB', '_kHz')['best']
        c = model_curve(ftab, 'IOPB', b['M'], b['nu_s'])
        ax[r, 1].plot(c['nu_L'], c['nu_HBO'], color=GREY, lw=1.2,
                      ls=(0, (1, 1.5)),
                      label=f"IOPB-I, kHz QPOs only ({b['nu_s']:.0f} Hz)")
        ax[r, 0].set_xlim(400, 900); ax[r, 0].set_ylim(100, 640)
        ax[r, 1].set_xlim(100, 640); ax[r, 1].set_ylim(0, 58)
        ax[r, 0].set_xlabel(mlab('nuU', 'Hz'))
        ax[r, 0].set_ylabel(mlab('nuL', 'Hz'))
        ax[r, 1].set_xlabel(mlab('nuL', 'Hz'))
        ax[r, 1].set_ylabel(mlab('nuHBO', 'Hz'))
        ax[r, 0].set_title(title, fontsize=10)
        ax[r, 1].set_title(title + r',  HBO $=2\boldsymbol{\nu}_{\mathbf{nod}}$',
                           fontsize=10)
        ax[r, 0].legend(loc='upper left', fontsize=7.5)
        ax[r, 1].legend(loc='upper left', fontsize=7.5)
        for a in ax[r]:
            finish(a)
    fig.tight_layout()
    label_panels(fig, ax)
    save(fig, 'fig_tracks_z.pdf')


# =====================================================================
# 5. One corner plot: 4U 1728-34 without an EOS
# =====================================================================
def _levels(H, fracs=(0.95, 0.68)):
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    return [h[np.searchsorted(c, f)] for f in fracs]


def _q(x):
    p = np.percentile(x, [16, 50, 84])
    return p[1], p[2] - p[1], p[1] - p[0]


def fig_corner(ftab, csv=None, nu_s=363.0):
    csv = csv or sorted(glob.glob('bilby_out/4U1728_free_frac2/**/*'
                                  'posterior*.csv', recursive=True))[0]
    import pandas as pd
    df = pd.read_csv(csv)
    col = lambda k: df[k].values if k in df else df[f'{k}_4U1728-34'].values
    s = {'M': col('M'), 'j': col('j'), 'Qbar': col('Qbar')}
    Om = 2 * np.pi * nu_s / C_KM_S
    s['I'] = s['j'] * (s['M'] * M_SUN_KM) ** 2 / Om * I45
    keys = ['M', 'j', 'Qbar', 'I']
    labels = {'M': lab(S['M'], MSUN), 'j': lab(S['j']),
              'Qbar': lab(S['Qbar']),
              'I': lab(S['I'], r'$\mathbf{10^{45}}$ g cm$^{\mathbf{2}}$')}
    rng = {'M': (1.875, 1.975), 'j': (0.13, 0.25), 'Qbar': (1.0, 6.5),
           'I': (1.7, 3.2)}
    # EOS sequences at 363 Hz
    seq = {}
    for tag, name in EOS:
        st = StarTable(f'eos/{tag}')
        a = []
        for M in np.linspace(1.80, 2.05, 60):
            x = st.star(M, nu_s)
            if x is not None:
                a.append([M, x['j'], -x['q_over_j2'], x['I'] * I45])
        seq[name] = np.array(a)
    n = len(keys)
    fig, ax = plt.subplots(n, n, figsize=(6.6, 6.6))
    col = '#2a3f8f'
    for i, ki in enumerate(keys):
        for jj, kj in enumerate(keys):
            a = ax[i, jj]
            if jj > i:
                a.axis('off')
                continue
            if i == jj:
                h, e = np.histogram(s[ki], bins=45, range=rng[ki],
                                    density=True)
                h = gaussian_filter(h, 1.0)
                a.fill_between(0.5 * (e[1:] + e[:-1]), h, step=None,
                               color=col, alpha=0.25, lw=0)
                a.plot(0.5 * (e[1:] + e[:-1]), h, color=col, lw=1.5)
                med, up, lo = _q(s[ki])
                for v in (med - lo, med, med + up):
                    a.axvline(v, color=col, lw=1.0,
                              ls='-' if v == med else '--')
                nd = 3 if ki in ('M', 'j') else 2
                a.set_title(lab(S[ki] if ki in S else ki) + ' = '
                            + rf'${med:.{nd}f}^{{+{up:.{nd}f}}}_{{-{lo:.{nd}f}}}$',
                            fontsize=9.5)
                a.set_yticks([])
                a.tick_params(which='both', left=False, right=False)
                a.set_xlim(rng[ki])
            else:
                H, xe, ye = np.histogram2d(s[kj], s[ki], bins=40,
                                           range=[rng[kj], rng[ki]])
                H = gaussian_filter(H, 1.2)
                lv = _levels(H)
                xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                a.contourf(xc, yc, H.T, levels=[lv[0], lv[1], H.max()],
                           colors=[col], alpha=0.0)
                a.contourf(xc, yc, H.T, levels=[lv[0], lv[1]],
                           colors=['#9fb3e6'], alpha=0.8)
                a.contourf(xc, yc, H.T, levels=[lv[1], H.max() * 1.01],
                           colors=['#4f6fc8'], alpha=0.9)
                a.contour(xc, yc, H.T, levels=lv, colors=[col], linewidths=1.0)
                idx = {'M': 0, 'j': 1, 'Qbar': 2, 'I': 3}
                for k, (tag, name) in enumerate(EOS):
                    q_ = seq[name]
                    a.plot(q_[:, idx[kj]], q_[:, idx[ki]], color=COL[k],
                           ls=LS[k], lw=1.6)
                a.set_xlim(rng[kj]); a.set_ylim(rng[ki])
            if i == n - 1:
                a.set_xlabel(labels[kj], fontsize=11)
            else:
                a.set_xticklabels([])
            if jj == 0 and i > 0:
                a.set_ylabel(labels[ki], fontsize=11)
            elif jj > 0 or i == 0:
                a.set_yticklabels([])
            a.tick_params(labelsize=8.5)
            for lbl in a.get_xticklabels():
                lbl.set_rotation(40)
            finish(a)
            if i == jj:
                a.tick_params(which='both', left=False, right=False)
    hs = [plt.Line2D([], [], color=COL[k], ls=LS[k], lw=1.6,
                     label=f'{name} ' + r'($\boldsymbol{\nu}_{\boldsymbol{s}}$ = 363 Hz)')
          for k, (_, name) in enumerate(EOS)]
    hs.append(plt.Rectangle((0, 0), 1, 1, fc='#4f6fc8', label='68%'))
    hs.append(plt.Rectangle((0, 0), 1, 1, fc='#9fb3e6', label='95%'))
    fig.legend(handles=hs, loc='upper right', bbox_to_anchor=(0.97, 0.95),
               fontsize=9, title=r'4U 1728$-$34, no EOS assumed',
               title_fontsize=10)
    fig.subplots_adjust(hspace=0.07, wspace=0.07, left=0.11, bottom=0.11,
                        right=0.98, top=0.96)
    save(fig, 'fig_corner.pdf')
    INFO['corner'] = {k: _q(s[k]) for k in keys}
    INFO['corner_csv'] = csv


# =====================================================================
# 6. Table of fits
# =====================================================================
def _pm(d, nd=3, scale=1.0):
    m = d['median'] * scale
    up, lo = (d['hi68'] - d['median']) * scale, (d['median'] - d['lo68']) * scale
    while nd < 4 and min(up, lo) < 0.5 * 10 ** (-nd):
        nd += 1
    return f'${m:.{nd}f}^{{+{up:.{nd}f}}}_{{-{lo:.{nd}f}}}$'


def tab_fits():
    names = {'4U1636-53': '4U 1636$-$53', '4U1728-34': '4U 1728$-$34',
             'GX5-1': 'GX 5$-$1', 'GX340+0': 'GX 340+0'}
    lines = []
    for src, (lbl, nu) in FITS.items():
        first = True
        for tag, name in EOS:
            f = load_fit(src, tag, lbl)
            s_ = names[src] if first else ''
            first = False
            nus = (f'{nu:.0f}' if nu else _pm(f['nu_s'], 0))
            qb = dict(f['q_over_j2'])
            qb = {k: -v for k, v in qb.items() if k != 'param'}
            qb['hi68'], qb['lo68'] = -f['q_over_j2']['lo68'], \
                -f['q_over_j2']['hi68']
            lines.append(f"{s_} & {name} & {_pm(f['M'])} & {nus} & "
                         f"{_pm(f['j'])} & {_pm(qb, 2)} & "
                         f"{_pm(f['I'], 2, I45)} \\\\")
        lines.append('\\hline')
    lines = lines[:-1]
    with open(os.path.join(OUT, 'tab_fits.tex'), 'w') as fh:
        fh.write('\\begin{table*}[t]\n\\caption{\\label{tab:fits}Masses and '
                 'spins from the QPOs (medians and 68\\% intervals).  For the '
                 'atoll sources $\\nu_s$ is fixed by the burst oscillations; '
                 'for the Z sources it is a free parameter.  $j$, '
                 '$\\bar Q=-q/j^{2}$ and $I$ follow from $M$, $\\nu_s$ and the '
                 'EOS.  4U 1636$-$53: kHz QPOs only; the other sources: kHz '
                 'QPOs and the low-frequency QPO or HBO, identified with '
                 '$2\\nu_{\\rm nod}$.}\n\\begin{ruledtabular}\n'
                 '\\begin{tabular}{llccccc}\n'
                 'Source & EOS & $M\\,[M_\\odot]$ & $\\nu_s$ [Hz] & $j$ & '
                 '$\\bar Q$ & $I\\,[10^{45}\\,{\\rm g\\,cm^{2}}]$ \\\\\n'
                 '\\hline\n')
        fh.write('\n'.join(lines) + '\n')
        fh.write('\\end{tabular}\n\\end{ruledtabular}\n\\end{table*}\n')
    print('  saved tab_fits.tex')


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--only', default='stars,freqs,nodal,orbit,psd,tracks,'
                    'corner,table')
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    ps.use()
    todo = a.only.split(',')
    ftab = FreqTable()
    if 'stars' in todo:
        fig_stars(ftab)
    if 'freqs' in todo:
        fig_freqs(ftab)
    if 'nodal' in todo:
        fig_nodal(ftab)
    if 'orbit' in todo:
        fig_orbit()
    if 'psd' in todo:
        fig_psd()
    if 'tracks' in todo:
        fig_tracks_atoll(ftab)
        fig_tracks_z(ftab)
    if 'corner' in todo:
        fig_corner(ftab)
    if 'table' in todo:
        tab_fits()
    p = os.path.join(OUT, 'figures_info.json')
    old = json.load(open(p)) if os.path.isfile(p) else {}
    old.update(INFO)
    json.dump(old, open(p, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
