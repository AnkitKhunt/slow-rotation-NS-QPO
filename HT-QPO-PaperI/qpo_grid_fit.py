#!/usr/bin/env python3
"""
qpo_grid_fit.py

Fast, deterministic grid fitter for kHz / low-frequency QPOs in the
relativistic precession model (RPM) on the Hartle-Thorne (HT) spacetime.

It uses the SAME tables and likelihood ingredients as qpo_fit.py
(FreqTable, StarTable, QPOSource), so the two codes can be cross-checked:
qpo_fit.py samples with Bilby/dynesty, this script integrates the posterior
on a grid (no sampler noise, evidence by direct quadrature, ~1 min per
source and EOS on one core).  Use it to explore; use qpo_fit.py for the
final, sampled posteriors.

---------------------------------------------------------------------------
MODEL  (identical to qpo_fit.py except for the scatter option)
---------------------------------------------------------------------------
  nu_U = nu_phi(r),  nu_L = nu_phi - nu_r,  nu_LF (HBO) = 2 nu_nod = 2 (nu_phi - nu_theta)

Each QPO set i comes from its own radius x_i = r_i/M, marginalised with a
uniform prior on [x_in, x_max], x_in = max(R_eq/M, x_ISCO):

  L_i = 1/(x_max - x_in) Int dx  prod_k N(d_ik | m_k(x), s_ik^2)

Scatter (choose with --scatter):
  frac : s_ik^2 = sigma_ik^2 + (eta d_ik)^2       [default; eta log-uniform]
  abs  : s_ik^2 = sigma_ik^2 + sigma_int^2         [as in qpo_fit.py]
A fractional scatter is the natural choice when kHz QPOs (~10^3 Hz) and the
LF QPO / HBO (~10-50 Hz) are fitted together: an additive scatter large
enough to absorb kHz residuals would make the LF QPO uninformative.

Modes
  eos   parameters per source: M, nu_s (fixed if its prior width < 5 Hz), eta
  free  parameters per source: M, j, Qbar (q = -Qbar j^2), eta;
        --qbar-fixed 1 gives the Kerr-like (q = -j^2) analysis.

Priors: M ~ U(M_min, M_max); nu_s ~ U(spin_lo, spin_hi) if unknown;
        eta ~ LogU(eta_min, eta_max); j ~ U(0, 0.45); Qbar ~ U(1, 12).
Where an EOS cannot support (M, nu_s), the likelihood is zero (this enters
the evidence, exactly as in qpo_fit.py).

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
  # tables first (once):  python3 qpo_fit.py build-tables --eos eos/G3 ...
  python3 qpo_grid_fit.py eos  --config configs/atoll.json --eos eos/G3 eos/IOPB
  python3 qpo_grid_fit.py free --config configs/atoll.json
  python3 qpo_grid_fit.py free --config configs/atoll.json --qbar-fixed 1
  python3 qpo_grid_fit.py predict --eos eos/G3 --mass 1.8 --spin 363
  python3 qpo_grid_fit.py scan-spin --eos eos/G3 --mass 2.0 --nu-u 800 --nu-hbo 40

Outputs in --outdir/<mode>/<source>/:
  <label>.json        evidence, best fit, medians, 68/95% intervals, derived
  <label>_grid.npz    full log-likelihood grid (for your own plots)
  <label>_post.png    marginals and 2D posterior
  <label>_track.png   data with the posterior-median RPM track
"""

import argparse
import json
import os
import sys

import numpy as np
_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy 2 / numpy 1
from scipy.special import logsumexp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qpo_fit import (FreqTable, StarTable, load_config, hz_curves,
                     eos_tag, set_style, M_SUN_KM, C_KM_S, _log_phi_diff,
                     loglike_split)


# =====================================================================
# Likelihood on many scatter values at once
# =====================================================================
def _chan(src, k):
    d = src.data['nu_' + k]
    s = src.data['sig_' + k]
    return d, s, np.isfinite(d)


def _terms(src, cur, chans, etas, frac):
    """
    Segment terms for the channels `chans`, for all scatter values:
    A, B, C of shape (nE, n, K-1) and the normalisation (nE, n).
    """
    x = cur['x']
    nE, n = len(etas), src.n
    A = np.zeros((nE, n, len(x) - 1))
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    norm = np.zeros((nE, n))
    for k in chans:
        if not src.has[k]:
            continue
        d, s, ok = _chan(src, k)
        m = cur['nu_' + k]
        b = np.diff(m) / np.diff(x)                               # (K-1,)
        r = np.where(ok[:, None], d[:, None] - m[None, :-1], 0.0)  # (n, K-1)
        dd = np.where(ok, d, 0.0)
        ss = np.where(ok, s, 1.0)
        if frac:
            var = ss[None, :] ** 2 + (etas[:, None] * dd[None, :]) ** 2
        else:
            var = ss[None, :] ** 2 + etas[:, None] ** 2
        w = np.where(ok[None, :], 1.0 / var, 0.0)                 # (nE, n)
        norm += np.where(ok[None, :], -0.5 * np.log(2 * np.pi * var), 0.0)
        A += w[:, :, None] * (b * b)[None, None, :]
        B += w[:, :, None] * (b[None, :] * r)[None]
        C += w[:, :, None] * (r * r)[None]
    return A, B, C, norm


def _seg(A, B, C, h):
    """ln sum_j Int_0^h_j exp(-(A t^2 - 2 B t + C)/2) dt over the last axis."""
    A = np.maximum(A, 1e-300)
    sA = np.sqrt(A)
    ua = -B / sA
    ub = ua + sA * h
    lnI = (-0.5 * (C - B * B / A) + 0.5 * np.log(2.0 * np.pi / A)
           + _log_phi_diff(ua, ub))
    return logsumexp(lnI, axis=-1)


def loglike_eta(src, cur, etas, frac=True):
    """
    ln L(eta) for one source and one model curve set, for all eta at once.
    The radius x of each QPO set is marginalised EXACTLY for the
    piecewise-linear model curves (seg_log_integral in qpo_fit.py).
    Returns array of shape (len(etas),).
    """
    etas = np.atleast_1d(etas)
    if getattr(src, 'asym', False):                 # split-normal errors
        return loglike_split(src, cur, etas, None, frac)
    x = cur['x']
    A, B, C, norm = _terms(src, cur, ('U', 'L', 'HBO'), etas, frac)
    li = _seg(A, B, C, np.diff(x))
    return np.sum(norm + li, axis=1) - src.n * np.log(x[-1] - x[0])


def loglike_eta2(src, cur, etas):
    """
    ln L for a fractional scatter eta_kHz on (nu_U, nu_L) and a separate
    fractional scatter eta_LF on the LF QPO / HBO.  Returns (nE, nE) array
    indexed [eta_kHz, eta_LF].
    """
    etas = np.atleast_1d(etas)
    if getattr(src, 'asym', False):                 # split-normal errors
        return loglike_split(src, cur, etas, etas, True)
    x = cur['x']
    A1, B1, C1, n1 = _terms(src, cur, ('U', 'L'), etas, True)
    A2, B2, C2, n2 = _terms(src, cur, ('HBO',), etas, True)
    li = _seg(A1[:, None] + A2[None], B1[:, None] + B2[None],
              C1[:, None] + C2[None], np.diff(x))             # (nE, nE, n)
    norm = n1[:, None] + n2[None]
    return np.sum(norm + li, axis=2) - src.n * np.log(x[-1] - x[0])


def curves_for(ftab, M, j, q, x_floor, x_max, K):
    fc = ftab.curves(j, q, x_floor, x_max, K)
    if fc is None:
        return None, None
    return hz_curves(fc, M), fc


# =====================================================================
# Grid statistics
# =====================================================================
def wquant(x, w, qs=(0.025, 0.16, 0.5, 0.84, 0.975)):
    """
    Weighted quantiles of a 1D marginal tabulated on grid x (ascending).
    The mass w_k belongs to the cell around node k (edges at the midpoints
    between nodes), so the CDF is interpolated between cell EDGES.
    v3.2 fix: the first version interpolated the CDF at the nodes, which
    shifts every quantile down by about half a grid cell (0.05-0.07 dex for
    the scatter grids, i.e. eta 10-15 % too small).
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return [np.nan] * len(qs)
    if len(x) == 1:
        return [float(x[0])] * len(qs)
    e = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    c = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    return [float(np.interp(qq, c, e)) for qq in qs]


def summarise_1d(name, x, logpost_marg):
    p = np.exp(logpost_marg - logpost_marg.max())
    q = wquant(x, p)
    return {'param': name, 'median': q[2], 'lo68': q[1], 'hi68': q[3],
            'lo95': q[0], 'hi95': q[4], 'map': float(x[np.argmax(p)])}


def log_trapz_weights(x):
    """ln of trapezoid weights for a 1D grid."""
    w = np.zeros_like(x, dtype=float)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return np.log(w)


# =====================================================================
# EOS mode
# =====================================================================
def fit_eos(src, star, ftab, args):
    two = args.scatter == 'frac2'
    frac = args.scatter in ('frac', 'frac2')
    # the grid may cover only part of the prior range (--M-grid, --spin-grid)
    # to resolve narrow posteriors; the prior normalisation always uses the
    # full ranges [M_min, M_max] and [spin_lo, spin_hi].  The likelihood must
    # be negligible outside the grid (check the marginals).
    Mg = args.M_grid if args.M_grid else (args.M_min, args.M_max)
    Ms = np.linspace(Mg[0], Mg[1], args.nM)
    M_width = args.M_max - args.M_min
    if src.spin is not None and src.spin[1] < 5.0:
        spins = np.array([float(src.spin[0])])
        s_width = 1.0
    elif src.spin is not None:
        mu, sd = src.spin
        spins = np.linspace(mu - 3 * sd, mu + 3 * sd, args.nS)
        s_width = 6 * sd
    else:
        sg = args.spin_grid if args.spin_grid else (args.spin_lo, args.spin_hi)
        spins = np.linspace(sg[0], sg[1], args.nS)
        s_width = args.spin_hi - args.spin_lo
    etas = (np.geomspace(args.eta_min, args.eta_max, args.nE) if frac
            else np.geomspace(args.sig_min, args.sig_max, args.nE))

    nEt = len(etas) ** 2 if two else len(etas)
    lnL = np.full((len(Ms), len(spins), nEt), -np.inf)
    der = {k: np.full((len(Ms), len(spins)), np.nan) for k in
           ('j', 'q', 'q_over_j2', 'R_eq', 'R_eq_circ', 'r_in', 'x_in',
            'nuU_edge', 'is_isco', 'I')}
    for a, M in enumerate(Ms):
        for b, nu in enumerate(spins):
            st = star.star(M, nu)
            if st is None or not ftab.in_range(st['j'], st['q']):
                continue
            cur, fc = curves_for(ftab, M, st['j'], st['q'], st['x_surf'],
                                 args.x_max, args.K)
            if cur is None:
                continue
            if two:
                lnL[a, b, :] = loglike_eta2(src, cur, etas).ravel()
            else:
                lnL[a, b, :] = loglike_eta(src, cur, etas, frac)
            der['j'][a, b] = st['j']
            der['q'][a, b] = st['q']
            der['q_over_j2'][a, b] = st['q_over_j2']
            der['R_eq'][a, b] = st['R_eq']
            der['R_eq_circ'][a, b] = st['R_eq_circ']
            der['x_in'][a, b] = fc['x_in']
            der['r_in'][a, b] = fc['x_in'] * M * M_SUN_KM
            der['nuU_edge'][a, b] = cur['nu_U'][0]
            der['is_isco'][a, b] = 1.0 if fc['edge'] == 'isco' else 0.0
            der['I'][a, b] = st['I']
    # priors: uniform M, uniform nu_s (if gridded), log-uniform eta
    lnw = (log_trapz_weights(Ms)[:, None, None] - np.log(M_width))
    if len(spins) > 1:
        lnw = lnw + (log_trapz_weights(spins)[None, :, None]
                     - np.log(s_width))
    le = np.log(etas)
    w1 = log_trapz_weights(le) - np.log(le[-1] - le[0])
    if two:
        w1 = (w1[:, None] + w1[None, :]).ravel()
    lnw = lnw + w1[None, None, :]
    lnZ = float(logsumexp(lnL + lnw)) if np.isfinite(lnL).any() else -np.inf
    return {'Ms': Ms, 'spins': spins, 'etas': etas, 'lnL': lnL, 'lnw': lnw,
            'lnZ': lnZ, 'derived': der, 'frac': frac, 'two': two}


def summarise_eos(res):
    lp = res['lnL'] + res['lnw']
    out = {'lnZ': res['lnZ']}
    if not np.isfinite(res['lnZ']):
        out['status'] = 'no support: the EOS cannot reproduce the data'
        return out
    i = np.unravel_index(np.nanargmax(res['lnL']), res['lnL'].shape)
    nE = len(res['etas'])
    if res.get('two'):
        ia, ib = np.unravel_index(i[2], (nE, nE))
        eb = {'eta_kHz': float(res['etas'][ia]), 'eta_LF': float(res['etas'][ib])}
    else:
        eb = {'eta': float(res['etas'][i[2]])}
    out['best'] = {'M': float(res['Ms'][i[0]]), 'nu_s': float(res['spins'][i[1]]),
                   'lnL_max': float(res['lnL'][i]), **eb}
    for k, v in res['derived'].items():
        out['best'][k] = float(v[i[0], i[1]])
    out['M'] = summarise_1d('M', res['Ms'], logsumexp(lp, axis=(1, 2)))
    if len(res['spins']) > 1:
        out['nu_s'] = summarise_1d('nu_s', res['spins'],
                                   logsumexp(lp, axis=(0, 2)))
    le = np.log10(res['etas'])
    if res.get('two'):
        lpe = logsumexp(lp, axis=(0, 1)).reshape(nE, nE)
        out['eta'] = summarise_1d('log10_eta_kHz', le, logsumexp(lpe, axis=1))
        out['eta_LF'] = summarise_1d('log10_eta_LF', le, logsumexp(lpe, axis=0))
    else:
        out['eta'] = summarise_1d('eta', le, logsumexp(lp, axis=(0, 1)))
        out['eta']['param'] = ('log10_eta' if res['frac']
                               else 'log10_sigma_int')
    # derived quantities: posterior weights on the (M, nu) plane
    w2 = np.exp(logsumexp(lp, axis=2) - np.max(lp))
    for k, v in res['derived'].items():
        ok = np.isfinite(v) & (w2 > 0)
        if not ok.any():
            continue
        if k == 'is_isco':
            out['P_isco'] = float(np.sum(w2[ok] * v[ok]) / np.sum(w2[ok]))
            continue
        x, w = v[ok], w2[ok]
        o = np.argsort(x)
        q = wquant(x[o], w[o])
        out[k] = {'median': q[2], 'lo68': q[1], 'hi68': q[3],
                  'lo95': q[0], 'hi95': q[4]}
    return out


# =====================================================================
# Free (EOS-agnostic) mode
# =====================================================================
def fit_free(src, ftab, args):
    frac = args.scatter == 'frac'
    Mg = args.M_grid if args.M_grid else (args.M_min, args.M_max_free)
    jg = args.j_grid if args.j_grid else (0.0, args.j_max)
    Qg = args.qbar_grid if args.qbar_grid else (args.qbar_lo, args.qbar_hi)
    Ms = np.linspace(Mg[0], Mg[1], args.nM)
    js = np.linspace(jg[0], jg[1], args.nj)
    if args.qbar_fixed is not None:
        Qbs = np.array([args.qbar_fixed])
    else:
        Qbs = np.linspace(Qg[0], Qg[1], args.nQ)
    wM = args.M_max_free - args.M_min
    wj = args.j_max
    wQ = args.qbar_hi - args.qbar_lo
    etas = (np.geomspace(args.eta_min, args.eta_max, args.nE) if frac
            else np.geomspace(args.sig_min, args.sig_max, args.nE))
    lnL = np.full((len(Ms), len(js), len(Qbs), len(etas)), -np.inf)
    for b, j in enumerate(js):
        for c, Qb in enumerate(Qbs):
            q = -Qb * j * j
            if not ftab.in_range(j, q):
                continue
            fc = ftab.curves(j, q, args.x_floor_free, args.x_max, args.K)
            if fc is None:
                continue
            for a, M in enumerate(Ms):
                cur = hz_curves(fc, M)
                lnL[a, b, c, :] = loglike_eta(src, cur, etas, frac)
    lnw = log_trapz_weights(Ms)[:, None, None, None] - np.log(wM)
    lnw = lnw + log_trapz_weights(js)[None, :, None, None] - np.log(wj)
    if len(Qbs) > 1:
        lnw = lnw + (log_trapz_weights(Qbs)[None, None, :, None]
                     - np.log(wQ))
    le = np.log(etas)
    lnw = lnw + log_trapz_weights(le)[None, None, None, :] - np.log(np.ptp(le))
    lnZ = float(logsumexp(lnL + lnw))
    return {'Ms': Ms, 'js': js, 'Qbs': Qbs, 'etas': etas, 'lnL': lnL,
            'lnw': lnw, 'lnZ': lnZ, 'frac': frac}


def summarise_free(res):
    lp = res['lnL'] + res['lnw']
    i = np.unravel_index(np.nanargmax(res['lnL']), res['lnL'].shape)
    out = {'lnZ': res['lnZ'],
           'best': {'M': float(res['Ms'][i[0]]), 'j': float(res['js'][i[1]]),
                    'Qbar': float(res['Qbs'][i[2]]),
                    'eta': float(res['etas'][i[3]]),
                    'lnL_max': float(res['lnL'][i])}}
    out['M'] = summarise_1d('M', res['Ms'], logsumexp(lp, axis=(1, 2, 3)))
    out['j'] = summarise_1d('j', res['js'], logsumexp(lp, axis=(0, 2, 3)))
    if len(res['Qbs']) > 1:
        out['Qbar'] = summarise_1d('Qbar', res['Qbs'],
                                   logsumexp(lp, axis=(0, 1, 3)))
    out['eta'] = summarise_1d('log10_eta', np.log10(res['etas']),
                              logsumexp(lp, axis=(0, 1, 2)))
    # derived: J = j M^2 -> I_45 / (nu_s) is source dependent; report
    # j*M^2 (km^2) so I = J/Omega can be formed for any assumed spin
    Mg, jg = np.meshgrid(res['Ms'], res['js'], indexing='ij')
    w = np.exp(logsumexp(lp, axis=(2, 3)) - np.max(lp))
    Jkm2 = jg * (Mg * M_SUN_KM) ** 2
    o = np.argsort(Jkm2.ravel())
    q = wquant(Jkm2.ravel()[o], w.ravel()[o])
    out['J_km2'] = {'median': q[2], 'lo68': q[1], 'hi68': q[3]}
    return out


# =====================================================================
# Plots
# =====================================================================
def plot_eos_post(res, src, path, title):
    set_style()
    lp = res['lnL'] + res['lnw']
    has_spin = len(res['spins']) > 1
    fig, axes = plt.subplots(1, 3 if has_spin else 2,
                             figsize=(13 if has_spin else 9, 3.8))
    pM = np.exp(logsumexp(lp, axis=(1, 2)) - lp.max())
    axes[0].plot(res['Ms'], pM / _trapz(pM, res['Ms']), color='#1f4e79')
    axes[0].set_xlabel(r'$M\ [M_\odot]$')
    axes[0].set_ylabel('posterior density')
    pE = logsumexp(lp, axis=(0, 1))
    if res.get('two'):
        n = len(res['etas'])
        pE = logsumexp(pE.reshape(n, n), axis=1)
    pE = np.exp(pE - pE.max())
    le = np.log10(res['etas'])
    axes[1].plot(le, pE / _trapz(pE, le), color='#1f4e79')
    axes[1].set_xlabel(r'$\log_{10}\eta$' if res['frac']
                       else r'$\log_{10}\sigma_{\rm int}$ [Hz]')
    if has_spin:
        p2 = np.exp(logsumexp(lp, axis=2) - lp.max())
        axes[2].contourf(res['spins'], res['Ms'], p2, levels=20, cmap='Blues')
        axes[2].set_xlabel(r'$\nu_s$ [Hz]')
        axes[2].set_ylabel(r'$M\ [M_\odot]$')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_track(src, ftab, star, M, nu, j, q, x_floor, args, path, title):
    set_style()
    cur, fc = curves_for(ftab, M, j, q, x_floor, args.x_max, 600)
    d = src.data
    has_h = src.has['HBO']
    fig, axes = plt.subplots(1, 2 if has_h else 1,
                             figsize=(11 if has_h else 6, 4.4))
    axes = np.atleast_1d(axes)
    axes[0].errorbar(d['nu_U'], d['nu_L'], xerr=d['sig_U'], yerr=d['sig_L'],
                     fmt='o', ms=4, color='#CC3300', ecolor='0.5')
    if cur is not None:
        axes[0].plot(cur['nu_U'], cur['nu_L'], 'k', lw=1.6)
    axes[0].set_xlabel(r'$\nu_U$ [Hz]')
    axes[0].set_ylabel(r'$\nu_L$ [Hz]')
    if has_h:
        axes[1].errorbar(d['nu_L'], d['nu_HBO'], xerr=d['sig_L'],
                         yerr=d['sig_HBO'], fmt='s', ms=4, color='#2ca02c',
                         ecolor='0.5')
        if cur is not None:
            axes[1].plot(cur['nu_L'], cur['nu_HBO'], 'k', lw=1.6)
        axes[1].set_xlabel(r'$\nu_L$ [Hz]')
        axes[1].set_ylabel(r'$\nu_{\rm LF}=2\nu_{\rm nod}$ [Hz]')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# =====================================================================
# predict / scan-spin helpers  (for exploring by hand)
# =====================================================================
def weak_field(x, j, q):
    """Leading weak-field pieces of the nodal precession, in units of Omega_K."""
    return {'LT': 2.0 * j * x ** -1.5, 'Q': 1.5 * q * x ** -2}


def cmd_predict(args):
    import ht_common
    from qpo import qpo_at_radius
    m = ht_common.solve_ht_mass(args.eos, args.mass, args.spin, verbose=True)
    if m is None:
        sys.exit('no stable star')
    r_in, edge = ht_common.inner_edge(m)
    print(f"\n  EOS {eos_tag(args.eos)}  M = {m.M_solar:.4f} Msun  "
          f"nu_s = {args.spin} Hz")
    print(f"  j = {m.j:.4f}  q = {m.q:.4f}  q/j^2 = {m.q / m.j ** 2:.3f}  "
          f"R_eq = {m.R_eq_km:.3f} km  R_eq,circ = {m.R_eq_circ_km:.3f} km")
    print(f"  I = {m.I_geo * 1.3466e43 / 1e45:.3f} x 10^45 g cm^2   "
          f"inner edge r_in = {r_in:.3f} km ({edge}), x_in = "
          f"{r_in / m.M_geo:.3f}\n")
    hdr = (f"{'r[km]':>7} {'x':>6} {'nu_phi':>8} {'nu_r':>8} {'nu_th':>8} "
           f"{'nu_per':>8} {'nu_nod':>7} {'nu_LT':>7} {'nu_Qd':>7} "
           f"{'2nu_nod':>8}")
    print(hdr)
    rows = []
    for r in np.linspace(r_in * 1.0001, args.rmax * m.M_geo, args.n):
        f = qpo_at_radius(m.M_geo, m.j, m.q, r)
        x = r / m.M_geo
        nk = C_KM_S / (2 * np.pi * m.M_geo) * x ** -1.5
        wf = weak_field(x, m.j, m.q)
        row = [r, x, f['nu_phi'], f['nu_r'], f['nu_theta'], f['nu_peri'],
               f['nu_nod'], nk * wf['LT'], nk * wf['Q'], 2 * f['nu_nod']]
        rows.append(row)
        print(f"{r:7.3f} {x:6.3f} {row[2]:8.2f} {row[3]:8.2f} {row[4]:8.2f} "
              f"{row[5]:8.2f} {row[6]:7.3f} {row[7]:7.3f} {row[8]:7.3f} "
              f"{row[9]:8.3f}")
    if args.out:
        np.savetxt(args.out, np.array(rows), header=hdr, fmt='%.6g')
        print(f"\n  saved {args.out}")


def cmd_scan_spin(args):
    """For a target (nu_U, nu_HBO): which spin does each EOS need at mass M?"""
    ftab = FreqTable()
    for e in args.eos:
        star = StarTable(e)
        print(f"\n  EOS {star.tag}, M = {args.mass}")
        for nu in np.arange(args.spin_lo, args.spin_hi + 1, args.dspin):
            st = star.star(args.mass, nu)
            if st is None:
                print(f"   nu_s={nu:5.0f}: no star")
                continue
            fc = ftab.curves(st['j'], st['q'], st['x_surf'], 40.0, 2000)
            cur = hz_curves(fc, args.mass)
            if not (cur['nu_U'][-1] <= args.nu_u <= cur['nu_U'][0]):
                print(f"   nu_s={nu:5.0f}: nu_U={args.nu_u} not reachable")
                continue
            h = np.interp(args.nu_u, cur['nu_U'][::-1], cur['nu_HBO'][::-1])
            l = np.interp(args.nu_u, cur['nu_U'][::-1], cur['nu_L'][::-1])
            print(f"   nu_s={nu:5.0f}: j={st['j']:.3f} q/j2={st['q_over_j2']:.2f}"
                  f"  at nu_U={args.nu_u}: nu_L={l:7.1f}  2nu_nod={h:6.2f}"
                  f"  (target {args.nu_hbo})")


def cmd_resummarise(args):
    """Rebuild every <label>.json from <label>_grid.npz (keeps metadata)."""
    import glob
    n = 0
    for npz in sorted(glob.glob(os.path.join(args.root, '**', '*_grid.npz'),
                                recursive=True)):
        js = npz.replace('_grid.npz', '.json')
        if not os.path.isfile(js):
            continue
        old = json.load(open(js))
        g = np.load(npz)
        lnZ = float(logsumexp(g['lnL'] + g['lnw']))
        if 'spins' in g.files:                                 # eos mode
            der = {k[4:]: g[k] for k in g.files if k.startswith('der_')}
            res = {'Ms': g['Ms'], 'spins': g['spins'], 'etas': g['etas'],
                   'lnL': g['lnL'], 'lnw': g['lnw'], 'lnZ': lnZ,
                   'derived': der,
                   'frac': old.get('scatter', 'frac') != 'abs',
                   'two': g['lnL'].shape[2] == len(g['etas']) ** 2}
            new = summarise_eos(res)
        else:                                                  # free mode
            res = {'Ms': g['Ms'], 'js': g['js'], 'Qbs': g['Qbs'],
                   'etas': g['etas'], 'lnL': g['lnL'], 'lnw': g['lnw'],
                   'lnZ': lnZ}
            new = summarise_free(res)
        for k, v in old.items():
            if k not in new:
                new[k] = v
        json.dump(new, open(js, 'w'), indent=1)
        n += 1
    print(f'  re-summarised {n} fits under {args.root}')


# =====================================================================
# Driver
# =====================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--config', required=True)
        sp.add_argument('--outdir', default='grid_fit_out')
        sp.add_argument('--scatter', choices=['frac', 'frac2', 'abs'],
                        default='frac',
                        help='frac: one fractional scatter; frac2: separate '
                             'fractional scatters for kHz and LF/HBO (eos mode); '
                             'abs: additive scatter in Hz')
        sp.add_argument('--eta-min', type=float, default=1e-4)
        sp.add_argument('--eta-max', type=float, default=0.2)
        sp.add_argument('--sig-min', type=float, default=0.1)
        sp.add_argument('--sig-max', type=float, default=300.0)
        sp.add_argument('--nE', type=int, default=36)
        sp.add_argument('--nM', type=int, default=161)
        sp.add_argument('--M-min', type=float, default=1.0)
        sp.add_argument('--M-max', type=float, default=2.6)
        sp.add_argument('--x-max', type=float, default=25.0)
        sp.add_argument('--K', type=int, default=600)
        sp.add_argument('--label', default=None)
        sp.add_argument('--only', nargs='*', default=None,
                        help='fit only these source names')
        sp.add_argument('--M-grid', type=float, nargs=2, default=None,
                        help='mass range of the grid (prior stays M-min..M-max)')

    e = sub.add_parser('eos', help='EOS-consistent HT fit (M, nu_s, eta)')
    common(e)
    e.add_argument('--eos', nargs='+', required=True)
    e.add_argument('--nS', type=int, default=56)
    e.add_argument('--spin-lo', type=float, default=200.0)
    e.add_argument('--spin-hi', type=float, default=750.0)
    e.add_argument('--spin-grid', type=float, nargs=2, default=None,
                   help='spin range of the grid (prior stays spin-lo..spin-hi)')

    f = sub.add_parser('free', help='EOS-agnostic fit (M, j, Qbar, eta)')
    common(f)
    f.add_argument('--nj', type=int, default=46)
    f.add_argument('--nQ', type=int, default=23)
    f.add_argument('--j-max', type=float, default=0.45)
    f.add_argument('--qbar-lo', type=float, default=1.0)
    f.add_argument('--qbar-hi', type=float, default=12.0)
    f.add_argument('--qbar-fixed', type=float, default=None)
    f.add_argument('--j-grid', type=float, nargs=2, default=None,
                   help='j range of the grid (prior stays 0..j-max)')
    f.add_argument('--qbar-grid', type=float, nargs=2, default=None,
                   help='Qbar range of the grid (prior stays qbar-lo..qbar-hi)')
    f.add_argument('--M-max-free', type=float, default=3.0)
    f.add_argument('--x-floor-free', type=float, default=3.0)

    rs = sub.add_parser('resummarise',
                        help='recompute the JSON summaries from saved grids')
    rs.add_argument('--root', default='grid_fit_out')

    pr = sub.add_parser('predict', help='RPM frequencies vs radius for one star')
    pr.add_argument('--eos', required=True)
    pr.add_argument('--mass', type=float, required=True)
    pr.add_argument('--spin', type=float, required=True)
    pr.add_argument('--rmax', type=float, default=15.0, help='outer r/M')
    pr.add_argument('--n', type=int, default=20)
    pr.add_argument('--out', default=None)

    sc = sub.add_parser('scan-spin', help='spin needed for a (nu_U, nu_HBO) pair')
    sc.add_argument('--eos', nargs='+', required=True)
    sc.add_argument('--mass', type=float, required=True)
    sc.add_argument('--nu-u', type=float, required=True)
    sc.add_argument('--nu-hbo', type=float, required=True)
    sc.add_argument('--spin-lo', type=float, default=200.0)
    sc.add_argument('--spin-hi', type=float, default=750.0)
    sc.add_argument('--dspin', type=float, default=50.0)

    args = p.parse_args()
    if args.cmd == 'resummarise':
        return cmd_resummarise(args)
    if args.cmd == 'predict':
        return cmd_predict(args)
    if args.cmd == 'scan-spin':
        return cmd_scan_spin(args)

    sources, _ = load_config(args.config)
    if args.only:
        sources = [s for s in sources if s.name in args.only]
    ftab = FreqTable()
    label_sfx = f"_{args.label}" if args.label else ''
    summary = {}
    for s in sources:
        od = os.path.join(args.outdir, args.cmd, s.name)
        os.makedirs(od, exist_ok=True)
        if args.cmd == 'eos':
            for ef in args.eos:
                star = StarTable(ef)
                lab = star.tag + label_sfx
                res = fit_eos(s, star, ftab, args)
                out = summarise_eos(res)
                out.update({'source': s.name, 'eos': star.tag,
                            'scatter': args.scatter, 'x_max': args.x_max,
                            'prior_M': [args.M_min, args.M_max],
                            'M_max_eos_static': star.M_max(0.0)})
                json.dump(out, open(os.path.join(od, lab + '.json'), 'w'),
                          indent=1)
                np.savez(os.path.join(od, lab + '_grid.npz'),
                         Ms=res['Ms'], spins=res['spins'], etas=res['etas'],
                         lnL=res['lnL'], lnw=res['lnw'],
                         **{f'der_{k}': v for k, v in res['derived'].items()})
                summary[(s.name, lab)] = out
                if np.isfinite(out['lnZ']):
                    plot_eos_post(res, s, os.path.join(od, lab + '_post.png'),
                                  f'{s.name} / {lab}')
                    b = out['best']
                    st = star.star(b['M'], b['nu_s'])
                    plot_track(s, ftab, star, b['M'], b['nu_s'], st['j'],
                               st['q'], st['x_surf'], args,
                               os.path.join(od, lab + '_track.png'),
                               f"{s.name} / {lab}: best fit M={b['M']:.3f}, "
                               f"nu_s={b['nu_s']:.0f} Hz")
                    print(f"  {s.name:10s} {lab:10s} lnZ = {out['lnZ']:9.2f}  "
                          f"M = {out['M']['median']:.3f} "
                          f"[{out['M']['lo68']:.3f}, {out['M']['hi68']:.3f}]  "
                          + (f"nu_s = {out['nu_s']['median']:.0f}  "
                             if 'nu_s' in out else '')
                          + f"log10 eta = {out['eta']['median']:.2f}", flush=True)
                else:
                    print(f"  {s.name:10s} {lab:10s} no support", flush=True)
        else:
            lab = ('kerr' if args.qbar_fixed == 1.0 else 'free') + label_sfx
            res = fit_free(s, ftab, args)
            out = summarise_free(res)
            out.update({'source': s.name, 'mode': lab,
                        'scatter': args.scatter})
            json.dump(out, open(os.path.join(od, lab + '.json'), 'w'),
                      indent=1)
            np.savez(os.path.join(od, lab + '_grid.npz'), Ms=res['Ms'],
                     js=res['js'], Qbs=res['Qbs'], etas=res['etas'],
                     lnL=res['lnL'], lnw=res['lnw'])
            summary[(s.name, lab)] = out
            print(f"  {s.name:10s} {lab}: lnZ = {out['lnZ']:.2f}  "
                  f"M = {out['M']['median']:.3f} [{out['M']['lo68']:.3f}, "
                  f"{out['M']['hi68']:.3f}]  j = {out['j']['median']:.3f} "
                  f"[{out['j']['lo68']:.3f}, {out['j']['hi68']:.3f}]", flush=True)

    if args.cmd == 'eos':
        print('\n  ln Bayes factors relative to the best EOS, per source:')
        for s in sources:
            z = {k[1]: v['lnZ'] for k, v in summary.items() if k[0] == s.name}
            best = max(z.values())
            print('   ' + s.name + ': ' + ', '.join(
                f"{k} {v - best:+.2f}" for k, v in z.items()))


if __name__ == '__main__':
    main()
