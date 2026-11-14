#!/usr/bin/env python3
"""
qpo_fit.py

Bayesian parameter estimation of kHz QPOs with the relativistic precession
model (RPM) in the Hartle-Thorne spacetime, built on the existing code base
(ht_full, ht_common, qpo).  Bilby + dynesty, corner plots, posterior
predictive checks, EOS evidences and Bayes factors, optional NICER terms.

---------------------------------------------------------------------------
MODEL
---------------------------------------------------------------------------
For one source with QPO detections i = 1..N (any of nu_U, nu_L, nu_HBO may
be missing in a row):

    nu_U = nu_phi(r),  nu_L = nu_per(r) = nu_phi - nu_r,
    nu_HBO = 2 nu_nod(r) = 2 (nu_phi - nu_theta)

Each detection comes from its own (unknown) radius x_i = r_i / M, which is
marginalised with a uniform prior on [x_in, x_max]:

    L_s = prod_i  1/(x_max - x_in) Int dx  prod_k N(d_ik | m_k(x), s_ik^2 + sigma_int^2)

x_in = max(R_eq/M, x_ISCO) is the inner edge (stellar surface or ISCO).
sigma_int is an intrinsic scatter (QPO widths + RPM model error); a large
sigma_int is itself a statement about how well the RPM fits.

Two modes
  eos   : the star is fixed by (EOS, M, nu_s).  j, q, R_eq follow from the HT
          solution (ht_full) -> parameters per source: M, nu_s, sigma_int.
          One run per EOS -> evidence Z_EOS -> Bayes factors.
  free  : EOS-agnostic.  Parameters per source: M, j, Qbar (q = -Qbar j^2),
          sigma_int.  Inner edge = ISCO (surface unknown).  Gives the M-j
          degeneracy that each EOS line j(M, nu_s) must cross.

Speed (no ODE in the likelihood):
  * frequency table: M Omega_phi x^{3/2}, Omega_r^2/Omega_phi^2,
    Omega_theta^2/Omega_phi^2 on a (j, q, x) grid with M = 1, computed once
    from qpo._epi_signed (the same metric derivatives as qpo.py);
    frequencies scale as 1/M.
  * star table per EOS: one HTModel per rho_c along the stable branch at a
    reference spin nu_ref.  HT is a perturbation in Omega, so at any spin
        J ~ nu,  Q ~ nu^2,  delta M ~ nu^2,  R_eq - R ~ nu^2   (exact in HT),
    and M_rot(rho_c, nu) = M_TOV + dM_ref (nu/nu_ref)^2 is inverted by
    interpolation.  Both tables are cached in qpo_fit_cache/.

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
  # 1. build tables (once; parallel with --npool)
  python3 qpo_fit.py build-tables --eos eos/eosA eos/eosB --npool 8

  # 2. check table accuracy against direct qpo.py evaluation
  python3 qpo_fit.py check-table --eos eos/eosA

  # 3. injection-recovery test with mock data
  python3 qpo_fit.py mock --eos eos/eosA --mass 1.8 --spin 581 --n 40 \\
          --sigma 6 --out data/mock_4U1636.csv
  python3 qpo_fit.py fit --config sources_mock.json --mode eos \\
          --eos eos/eosA --truth M_4U1636=1.8 nu_s_4U1636=581

  # 4. real data: one fit per EOS, evidences, Bayes factors, corner plots
  python3 qpo_fit.py fit --config sources.json --mode eos \\
          --eos eos/G3U eos/FSUGarnetU eos/IOPBIU eos/NITRU --nlive 800

  # 5. EOS-agnostic M-j-Qbar fit
  python3 qpo_fit.py fit --config sources.json --mode free

Config (JSON):
  {
    "sources": [
      {"name": "4U1636", "data": "data/4U1636.csv", "spin": [581.0, 1.0]},
      {"name": "4U1728", "data": "data/4U1728.csv", "spin": [363.0, 1.0]},
      {"name": "4U1820", "data": "data/4U1820.csv", "spin": null}
    ],
    "nicer": [
      {"name": "J0740", "samples": "nicer/J0740.txt", "spin": 346,
       "mass_prior": [2.08, 0.07]},
      {"name": "J0437", "samples": "nicer/J0437.txt", "spin": 174,
       "mass_prior": [1.418, 0.037]},
      {"name": "J0030", "samples": "nicer/J0030.txt", "spin": 205}
    ]
  }
  "spin": [mean, sigma] -> truncated Gaussian prior; null -> Uniform(200, 750).
  NICER sample files: two columns M [Msun], R_eq [km] (equal-weight samples).

Data CSV (header required, any subset of columns, NaN = missing):
  nu_L,sig_L,nu_U,sig_U,nu_HBO,sig_HBO

Outputs (in --outdir/<label>/):
  bilby result (.json), bilby corner, corner with derived parameters
  (j, q/j^2, R_eq, r_in, nu_U at the inner edge), posterior predictive
  plots, summary.txt, summary.tex; for several EOSs: evidence table
  (evidence.txt/.tex), Bayes-factor plot, mass-posterior comparison.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp
_trapz = getattr(np, "trapezoid", None) or np.trapz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_KM_S = 2.99792458e5
M_SUN_KM = 1.4766
CACHE = 'qpo_fit_cache'

# v3.2: scatter model.  'abs': s^2 = sigma^2 + sigma_int^2   (original)
#                       'frac': s^2 = sigma^2 + (eta d)^2, eta = sigma_int_<src>
# Set from the command line with --scatter.  'frac' is the right choice when
# kHz QPOs (~1e3 Hz) and LF QPOs / HBOs (~10-50 Hz) are fitted together.
SCATTER = 'abs'


# =====================================================================
# Style (same family as the other scripts)
# =====================================================================
def set_style():
    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.grid': True, 'grid.alpha': 0.3,
        'grid.linestyle': '--', 'xtick.direction': 'in',
        'ytick.direction': 'in', 'legend.fontsize': 9,
    })


def eos_tag(path):
    return os.path.splitext(os.path.basename(path))[0]


# =====================================================================
# 1. Frequency table  (M = 1, dimensionless)
# =====================================================================
J_GRID = np.linspace(0.0, 0.5, 26)
Q_GRID = np.linspace(-2.6, 0.1, 55)
X_GRID = np.concatenate([np.linspace(3.0, 12.0, 181)[:-1],
                         np.geomspace(12.0, 45.0, 90)])


def _freq_row(args):
    """Table row for one (j, q): A = Omega_phi x^1.5, fr, ft (signed)."""
    from qpo import _epi_signed
    j, q = args
    out = np.empty((len(X_GRID), 3))
    for k, x in enumerate(X_GRID):
        try:
            Op, fr, ft = _epi_signed(x, j, q)
        except Exception:
            Op, fr, ft = np.nan, -1.0, -1.0
        if not np.isfinite(Op):
            out[k] = (np.nan, -1.0, -1.0)
        else:
            out[k] = (Op * x ** 1.5, fr, ft)
    return out


def build_freq_table(path=None, npool=1):
    path = path or os.path.join(CACHE, 'freq_table.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tasks = [(j, q) for j in J_GRID for q in Q_GRID]
    print(f"  building frequency table: {len(J_GRID)} x {len(Q_GRID)} "
          f"(j, q) x {len(X_GRID)} x  ({len(tasks) * len(X_GRID)} points)")
    if npool > 1:
        with Pool(npool) as pool:
            rows = pool.map(_freq_row, tasks, chunksize=4)
    else:
        rows = [_freq_row(t) for t in tasks]
    tab = np.array(rows).reshape(len(J_GRID), len(Q_GRID), len(X_GRID), 3)
    np.savez(path, j=J_GRID, q=Q_GRID, x=X_GRID, tab=tab)
    print(f"  saved {path}")
    return path


class FreqTable:
    """Interpolated dimensionless RPM frequencies Omega(x; j, q), M = 1."""

    def __init__(self, path=None):
        path = path or os.path.join(CACHE, 'freq_table.npz')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found: run  python3 qpo_fit.py build-tables")
        d = np.load(path)
        self.j, self.q, self.x = d['j'], d['q'], d['x']
        tab = d['tab'].copy()
        # NaN A (no circular orbit) -> 0, fr/ft -> -1  (marked unstable)
        bad = ~np.isfinite(tab[..., 0])
        tab[..., 0][bad] = 0.0
        tab[..., 1][bad] = -1.0
        tab[..., 2][bad] = -1.0
        self.interp = RegularGridInterpolator((self.j, self.q), tab,
                                              bounds_error=False,
                                              fill_value=None)
        self.jmax, self.qmin, self.qmax = self.j[-1], self.q[0], self.q[-1]

    def in_range(self, j, q):
        return (0.0 <= j <= self.jmax) and (self.qmin <= q <= self.qmax)

    def curves(self, j, q, x_floor, x_max, K=1200):
        """
        Dimensionless Omega_phi, Omega_r, Omega_theta on K points between
        the inner edge x_in = max(x_floor, x_ISCO) and x_max.
        Returns dict or None if no stable orbit region exists.
        """
        v = self.interp([[j, q]])[0]                  # (nx, 3)
        A, fr, ft = v[:, 0], v[:, 1], v[:, 2]
        x = self.x
        neg = np.where(fr <= 0.0)[0]
        if len(neg):
            i = neg.max()
            if i >= len(x) - 1:
                return None
            x_isco = x[i] + (0.0 - fr[i]) * (x[i + 1] - x[i]) / \
                (fr[i + 1] - fr[i])
        else:
            x_isco = x[0]
        x_in = max(x_floor, x_isco)
        edge = 'isco' if x_isco >= x_floor else 'surface'
        if x_in >= x_max:
            return None
        xf = np.linspace(x_in, x_max, K)
        Af = np.interp(xf, x, A)
        frf = np.clip(np.interp(xf, x, fr), 0.0, None)
        ftf = np.clip(np.interp(xf, x, ft), 0.0, None)
        Op = Af * xf ** -1.5
        return {'x': xf, 'Op': Op, 'Or': Op * np.sqrt(frf),
                'Ot': Op * np.sqrt(ftf), 'x_in': x_in, 'edge': edge,
                'x_isco': x_isco}


def hz_curves(fc, M_sun):
    """Dimensionless curves -> Hz for mass M_sun."""
    conv = C_KM_S / (2.0 * np.pi * M_sun * M_SUN_KM)
    nphi = fc['Op'] * conv
    nr = fc['Or'] * conv
    nth = fc['Ot'] * conv
    return {'nu_U': nphi, 'nu_L': nphi - nr, 'nu_HBO': 2.0 * (nphi - nth),
            'nu_r': nr, 'nu_theta': nth, 'x': fc['x']}


# =====================================================================
# 2. Star table per EOS  (one HTModel per rho_c at nu_ref)
# =====================================================================
_POOL_EOS = None


def _init_pool(eos_file):
    global _POOL_EOS
    from eos_loader import EOS
    _POOL_EOS = EOS(eos_file)


def _star_row(args):
    from ht_full import HTModel
    rho, nu_ref = args
    try:
        m = HTModel(_POOL_EOS, rho, nu_ref)
        return [rho, m.M0_geo, m.dM_geo, m.J_geo, m.Q_geo, m.R_km,
                m.R_eq_km, m.R_eq_circ_km, m.I_geo]
    except Exception:
        return [rho] + [np.nan] * 8


def build_star_table(eos_file, nu_ref=400.0, n_rho=64, M_min=0.9,
                     npool=1, path=None):
    import ht_common
    path = path or os.path.join(CACHE, f'star_{eos_tag(eos_file)}.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eos = ht_common.get_eos(eos_file)
    rho_max, M_max, _ = ht_common.find_max_mass(eos)
    rho_min = ht_common._rho_c_tov(eos, M_min, 1.0e14, rho_max, 1e-4)
    rhos = np.geomspace(rho_min, rho_max, n_rho)
    print(f"  {eos_tag(eos_file)}: {n_rho} HT models, rho_c "
          f"{rho_min:.3e} .. {rho_max:.3e} (TOV M_max = {M_max:.4f})")
    tasks = [(r, nu_ref) for r in rhos]
    if npool > 1:
        with Pool(npool, initializer=_init_pool,
                  initargs=(eos_file,)) as pool:
            rows = pool.map(_star_row, tasks)
    else:
        _init_pool(eos_file)
        rows = [_star_row(t) for t in tasks]
    a = np.array(rows, dtype=float)
    a = a[np.all(np.isfinite(a), axis=1)]
    np.savez(path, table=a, nu_ref=nu_ref,
             columns=np.array(['rho_c', 'M0', 'dM_ref', 'J_ref', 'Q_ref',
                               'R', 'Req_ref', 'Reqc_ref', 'I']))
    print(f"  saved {path}")
    return path


class StarTable:
    """
    (M_rot, nu_s) -> j, q, R_eq for one EOS, exact to O(Omega^2) by the
    HT scalings J ~ nu, Q, dM, dR ~ nu^2 at fixed rho_c.
    """

    def __init__(self, eos_file, path=None):
        self.eos_file = eos_file
        self.tag = eos_tag(eos_file)
        path = path or os.path.join(CACHE, f'star_{self.tag}.npz')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found: run  python3 qpo_fit.py build-tables "
                f"--eos {eos_file}")
        d = np.load(path)
        a = d['table']
        self.nu_ref = float(d['nu_ref'])
        (self.rho, self.M0, self.dM, self.J, self.Q, self.R, self.Req,
         self.Reqc, self.I) = a.T

    def star(self, M, nu):
        """Dict of stellar properties, or None if M is not supported."""
        s = nu / self.nu_ref
        s2 = s * s
        Mr = (self.M0 + self.dM * s2) / M_SUN_KM          # Msun
        if not np.all(np.diff(Mr) > 0):                   # keep stable part
            k = int(np.argmax(Mr)) + 1
            Mr = Mr[:k]
        else:
            k = len(Mr)
        if M < Mr[0] or M > Mr[-1]:
            return None

        def at(arr):
            return float(np.interp(M, Mr, arr[:k]))
        M_km = M * M_SUN_KM
        R = at(self.R)
        J = at(self.J) * s
        Q = at(self.Q) * s2
        Req = R + (at(self.Req) - R) * s2
        Reqc = R + (at(self.Reqc) - R) * s2
        j = J / M_km ** 2
        q = Q / M_km ** 3
        return {'j': j, 'q': q, 'R': R, 'R_eq': Req, 'R_eq_circ': Reqc,
                'x_surf': Req / M_km, 'rho_c': at(self.rho),
                'I': at(self.I), 'q_over_j2': q / j ** 2 if j > 0 else np.nan}

    def M_max(self, nu=0.0):
        s2 = (nu / self.nu_ref) ** 2
        return float(np.max((self.M0 + self.dM * s2) / M_SUN_KM))


# =====================================================================
# 3. Data
# =====================================================================
class QPOSource:
    COLS = ('nu_L', 'sig_L', 'nu_U', 'sig_U', 'nu_HBO', 'sig_HBO')

    def __init__(self, name, path, spin=None):
        self.name = name
        self.path = path
        self.spin = spin                    # None or (mean, sigma)
        d = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
        d = np.atleast_1d(d)
        n = len(d)
        get = (lambda c: np.asarray(d[c], float)
               if c in d.dtype.names else np.full(n, np.nan))
        self.data = {c: get(c) for c in self.COLS}
        for k in ('L', 'U', 'HBO'):
            v, s = self.data['nu_' + k], self.data['sig_' + k]
            bad = ~(np.isfinite(v) & np.isfinite(s) & (s > 0))
            v[bad], s[bad] = np.nan, np.nan
        self.has = {k: bool(np.isfinite(self.data['nu_' + k]).any())
                    for k in ('L', 'U', 'HBO')}
        # v3.2: optional asymmetric errors, columns sigp_<k> (upper) and
        # sigm_<k> (lower).  If present the likelihood is a split normal
        # (loglike_split); sig_<k> is then only used where they are missing.
        self.asym = False
        for k in ('L', 'U', 'HBO'):
            sp, sm = get('sigp_' + k), get('sigm_' + k)
            s = self.data['sig_' + k]
            sp = np.where(np.isfinite(sp) & (sp > 0), sp, s)
            sm = np.where(np.isfinite(sm) & (sm > 0), sm, s)
            self.data['sigp_' + k], self.data['sigm_' + k] = sp, sm
            ok = np.isfinite(self.data['nu_' + k])
            if np.any(np.abs(sp[ok] - sm[ok]) > 1e-12 * np.abs(sp[ok])):
                self.asym = True
        self.n = n
        print(f"  source {name}: {n} rows  (nu_U: "
              f"{np.isfinite(self.data['nu_U']).sum()}, nu_L: "
              f"{np.isfinite(self.data['nu_L']).sum()}, nu_HBO: "
              f"{np.isfinite(self.data['nu_HBO']).sum()})")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p)
    sources = [QPOSource(s['name'], resolve(s['data']),
                         tuple(s['spin']) if s.get('spin') else None)
               for s in cfg['sources']]
    nicer = []
    for p in cfg.get('nicer', []):
        p = dict(p)
        p['samples'] = resolve(p['samples'])
        nicer.append(p)
    return sources, nicer


# =====================================================================
# 4. Likelihood
# =====================================================================
from scipy.special import log_ndtr


def _log_phi_diff(ua, ub):
    """ln[Phi(ub) - Phi(ua)] for ub > ua, numerically stable."""
    pos = ua > 0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        l1 = log_ndtr(-ua) + np.log1p(-np.exp(log_ndtr(-ub) - log_ndtr(-ua)))
        l2 = log_ndtr(ub) + np.log1p(-np.exp(log_ndtr(ua) - log_ndtr(ub)))
    return np.where(pos, l1, l2)


def seg_log_integral(x, r_list, b_list, w_list):
    """
    v3.2: EXACT radius marginalisation for piecewise-linear model curves.

    On segment [x_j, x_j+1] each model frequency is m_k = a_k + b_k t,
    t in [0, h_j].  With residuals r_k = d_k - a_k and inverse variances
    w_k = 1/s_k^2, the integrand exp(-1/2 sum_k w_k (r_k - b_k t)^2) is a
    Gaussian in t and integrates in closed form (error functions).

    r_list, b_list, w_list: lists over channels of arrays broadcastable to a
    common shape (..., n, K-1); w = 0 for a missing channel.
    Returns ln of sum_j Int_0^h_j exp(...) dt, shape (..., n).

    WHY: the previous version averaged the integrand over K grid points.
    When the measurement error (e.g. 1 Hz on nu_L of 4U 1728-34) is smaller
    than the change of the model between grid points (several Hz for
    K = 600-1200), that sum is undersampled: the likelihood becomes jagged
    in M and the posteriors are biased.  The closed form has no such limit.
    """
    h = np.diff(x)
    A = sum(w * b * b for w, b in zip(w_list, b_list))
    B = sum(w * b * r for w, b, r in zip(w_list, b_list, r_list))
    C = sum(w * r * r for w, r in zip(w_list, r_list))
    A = np.maximum(A, 1e-300)
    sA = np.sqrt(A)
    ua = -B / sA
    ub = ua + sA * h
    lnI = (-0.5 * (C - B * B / A) + 0.5 * np.log(2.0 * np.pi / A)
           + _log_phi_diff(ua, ub))
    return logsumexp(lnI, axis=-1)


def channel_terms(src, cur, k, var):
    """r, b, w arrays (n, K-1) for channel k given variance var (n,)."""
    d = src.data['nu_' + k]
    m = cur['nu_' + k]
    x = cur['x']
    ok = np.isfinite(d)
    b = (np.diff(m) / np.diff(x))[None, :]
    r = np.where(ok[:, None], d[:, None] - m[None, :-1], 0.0)
    w = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)[:, None]
    lognorm = np.where(ok, -0.5 * np.log(2 * np.pi * np.where(ok, var, 1.0)),
                       0.0)
    return r, b, w, lognorm


def loglike_split(src, cur, etaA, etaB=None, frac=True):
    """
    v3.2: exact radius marginalisation with ASYMMETRIC errors (split normal).

    For a datum d with errors +sp/-sm and a model value m, the likelihood is
        2 / (sqrt(2 pi) (s+ + s-)) * exp(-(m - d)^2 / (2 s^2)),
        s = s+ if m > d else s-,   s+-^2 = sp^2 + (eta d)^2   (frac)
                                    s+-^2 = sp^2 + eta^2       (abs)
    The piecewise-linear model curves cross d at most once per channel; the
    radius grid is refined at these crossings so that on every interval each
    channel uses a single variance, and the Gaussian integral over the
    interval is done in closed form as in seg_log_integral.

    etaA: scatter values for the kHz QPOs (and for all QPOs if etaB is None).
    etaB: scatter values for the LF QPO / HBO (two-scatter model) or None.
    Returns ln L summed over the QPO sets, shape (nA,) or (nA, nB).
    For symmetric errors it equals source_loglike / loglike_eta(2).
    """
    etaA = np.atleast_1d(np.asarray(etaA, float))
    two = etaB is not None
    etaB = np.atleast_1d(np.asarray(etaB, float)) if two else None
    x = cur['x']
    shape = (len(etaA), len(etaB)) if two else (len(etaA),)
    total = np.zeros(shape)
    for i in range(src.n):
        chans = [k for k in ('U', 'L', 'HBO')
                 if src.has[k] and np.isfinite(src.data['nu_' + k][i])]
        if not chans:
            continue
        xc = [x]
        for k in chans:
            m = cur['nu_' + k]
            dm = m - src.data['nu_' + k][i]
            idx = np.nonzero(dm[:-1] * dm[1:] < 0)[0]
            if len(idx):
                xc.append(x[idx] - dm[idx] * (x[idx + 1] - x[idx])
                          / (m[idx + 1] - m[idx]))
        xr = np.unique(np.concatenate(xc))
        h = np.diff(xr)
        A = B = C = norm = 0.0
        for k in chans:
            d = src.data['nu_' + k][i]
            sp = src.data['sigp_' + k][i]
            sm = src.data['sigm_' + k][i]
            mr = np.interp(xr, x, cur['nu_' + k])
            b = np.diff(mr) / h
            r = d - mr[:-1]
            up = 0.5 * (mr[:-1] + mr[1:]) > d       # model above the datum
            sig = np.where(up, sp, sm)               # (K',)
            eta = etaB if (two and k == 'HBO') else etaA
            ex = (eta * d) ** 2 if frac else eta ** 2  # (n_eta,)
            var = sig[None, :] ** 2 + ex[:, None]    # (n_eta, K')
            w = 1.0 / var
            nk = (np.log(2.0) - 0.5 * np.log(2 * np.pi)
                  - np.log(np.sqrt(sp ** 2 + ex) + np.sqrt(sm ** 2 + ex)))
            a_, b_, c_ = w * b * b, w * b * r, w * r * r
            if two:
                if k == 'HBO':
                    a_, b_, c_, nk = (a_[None], b_[None], c_[None],
                                      nk[None, :])
                else:
                    a_, b_, c_, nk = (a_[:, None], b_[:, None], c_[:, None],
                                      nk[:, None])
            A, B, C, norm = A + a_, B + b_, C + c_, norm + nk
        A = np.maximum(A, 1e-300)
        sA = np.sqrt(A)
        ua = -B / sA
        ub = ua + sA * h
        lnI = (-0.5 * (C - B * B / A) + 0.5 * np.log(2.0 * np.pi / A)
               + _log_phi_diff(ua, ub))
        total = total + norm + logsumexp(lnI, axis=-1)
    return total - src.n * np.log(x[-1] - x[0])


def source_loglike(src, curves_hz, sigma_int, eta_lf=None):
    """
    ln L for one source: exact radius marginalisation (v3.2).
    SCATTER = 'abs'  : s^2 = sigma^2 + sigma_int^2          (Hz)
              'frac' : s^2 = sigma^2 + (sigma_int d)^2       (one eta)
              'frac2': as 'frac' for nu_U, nu_L; the LF QPO / HBO uses its
                       own fractional scatter eta_lf          (two-scatter)
    """
    if getattr(src, 'asym', False):
        two = SCATTER == 'frac2' and src.has['HBO']
        return float(np.ravel(loglike_split(
            src, curves_hz, [sigma_int], [eta_lf] if two else None,
            frac=SCATTER != 'abs'))[0])
    x = curves_hz['x']
    rl, bl, wl = [], [], []
    norm = np.zeros(src.n)
    for k in ('U', 'L', 'HBO'):
        if not src.has[k]:
            continue
        d = src.data['nu_' + k]
        s = src.data['sig_' + k]
        if SCATTER == 'frac2' and k == 'HBO':
            var = s ** 2 + (eta_lf * d) ** 2
        elif SCATTER in ('frac', 'frac2'):
            var = s ** 2 + (sigma_int * d) ** 2
        else:
            var = s ** 2 + sigma_int ** 2
        r, b, w, ln = channel_terms(src, curves_hz, k, var)
        rl.append(r); bl.append(b); wl.append(w)
        norm += ln
    li = seg_log_integral(x, rl, bl, wl)
    return float(np.sum(norm + li - np.log(x[-1] - x[0])))


def source_loglike_gridsum(src, curves_hz, sigma_int):
    """v3.1 likelihood (uniform-grid average); kept for comparison only."""
    K = len(curves_hz['x'])
    total = np.zeros((src.n, K))
    for k in ('U', 'L', 'HBO'):
        if not src.has[k]:
            continue
        d = src.data['nu_' + k][:, None]
        if SCATTER == 'frac':
            var = src.data['sig_' + k][:, None] ** 2 + (sigma_int * d) ** 2
        else:
            var = src.data['sig_' + k][:, None] ** 2 + sigma_int ** 2
        term = -0.5 * (d - curves_hz['nu_' + k][None, :]) ** 2 / var \
            - 0.5 * np.log(2.0 * np.pi * var)
        total += np.where(np.isfinite(term), term, 0.0)
    return float(np.sum(logsumexp(total, axis=1) - np.log(K)))


def model_for(src, p, mode, ftab, star, x_max, K):
    """Model curves (Hz) + derived quantities for one source."""
    n = src.name
    M = p[f'M_{n}']
    if mode == 'eos':
        nu = p[f'nu_s_{n}']
        st = star.star(M, nu)
        if st is None:
            return None, None
        j, q, x_floor = st['j'], st['q'], st['x_surf']
    elif mode == 'kerr':                 # Kerr-like: q = -j^2 (Qbar = 1)
        j = p[f'j_{n}']
        q = -j * j
        x_floor, st = 3.0, {'j': j, 'q': q}
    else:                                # free: (M, j, Qbar)
        j = p[f'j_{n}']
        q = -p[f'Qbar_{n}'] * j * j
        x_floor, st = 3.0, {'j': j, 'q': q}
    if not ftab.in_range(j, q):
        return None, None
    fc = ftab.curves(j, q, x_floor, x_max, K)
    if fc is None:
        return None, None
    st = dict(st)
    st.update({'x_in': fc['x_in'], 'edge': fc['edge'],
               'r_in': fc['x_in'] * M * M_SUN_KM})
    return hz_curves(fc, M), st


try:
    import bilby

    class RPMLikelihood(bilby.Likelihood):
        def __init__(self, sources, mode, ftab, star=None, x_max=25.0,
                     K=1200):
            # parameters are supplied by bilby from the prior keys
            super().__init__()
            self.sources, self.mode = sources, mode
            self.ftab, self.star = ftab, star
            self.x_max, self.K = x_max, K

        def log_likelihood(self, parameters=None):
            # bilby >= 2.5 passes `parameters`; older versions use
            # self.parameters
            p = self.parameters if parameters is None else parameters
            tot = 0.0
            for s in self.sources:
                cur, _ = model_for(s, p, self.mode, self.ftab, self.star,
                                   self.x_max, self.K)
                if cur is None:
                    return -np.inf
                tot += source_loglike(s, cur, p[f'sigma_int_{s.name}'],
                                      p.get(f'eta_LF_{s.name}'))
            return tot if np.isfinite(tot) else -np.inf
except ImportError:                                    # pragma: no cover
    bilby = None


def make_priors(sources, mode, M_range=(1.0, 2.6), sig_range=(0.1, 300.0)):
    pr = bilby.core.prior.PriorDict()
    for s in sources:
        n = s.name
        pr[f'M_{n}'] = bilby.core.prior.Uniform(
            *M_range, name=f'M_{n}', latex_label=rf'$M_{{\rm {n}}}$')
        if mode == 'eos':
            if s.spin is not None:
                mu, sd = s.spin
                pr[f'nu_s_{n}'] = bilby.core.prior.TruncatedGaussian(
                    mu, sd, max(mu - 10 * sd, 1.0), mu + 10 * sd,
                    name=f'nu_s_{n}', latex_label=rf'$\nu_{{s,\rm {n}}}$')
            else:
                pr[f'nu_s_{n}'] = bilby.core.prior.Uniform(
                    200.0, 750.0, name=f'nu_s_{n}',
                    latex_label=rf'$\nu_{{s,\rm {n}}}$')
        else:
            pr[f'j_{n}'] = bilby.core.prior.Uniform(
                0.0, 0.45, name=f'j_{n}', latex_label=rf'$j_{{\rm {n}}}$')
            if mode == 'free':
                pr[f'Qbar_{n}'] = bilby.core.prior.Uniform(
                    1.0, 12.0, name=f'Qbar_{n}',
                    latex_label=rf'$\bar Q_{{\rm {n}}}$')
        if SCATTER in ('frac', 'frac2'):
            pr[f'sigma_int_{n}'] = bilby.core.prior.LogUniform(
                1e-4, 0.2, name=f'sigma_int_{n}',
                latex_label=rf'$\eta_{{\rm {n}}}$')
            if SCATTER == 'frac2' and s.has['HBO']:
                pr[f'eta_LF_{n}'] = bilby.core.prior.LogUniform(
                    1e-4, 0.2, name=f'eta_LF_{n}',
                    latex_label=rf'$\eta_{{\rm LF,{n}}}$')
        else:
            pr[f'sigma_int_{n}'] = bilby.core.prior.LogUniform(
                *sig_range, name=f'sigma_int_{n}',
                latex_label=rf'$\sigma_{{\rm int,{n}}}$')
    return pr


# =====================================================================
# 5. NICER evidence term (EOS mode)
# =====================================================================
def nicer_log_evidence(star, nicer, M_range=(1.0, 2.6)):
    """
    ln Z_NICER(EOS) = sum_p ln Int dM pi(M) KDE_p(M, R_eq,circ(M; EOS, nu_p)).
    pi(M): Gaussian from radio timing if given, else uniform on M_range.
    """
    from scipy.stats import gaussian_kde
    out = {}
    for p in nicer:
        a = np.loadtxt(p['samples'], comments='#', usecols=(0, 1))
        kde = gaussian_kde(a.T)
        nu = float(p.get('spin', 0.0))
        Mg = np.linspace(0.8, 2.9, 600)
        if p.get('mass_prior'):
            mu, sd = p['mass_prior']
            pi = np.exp(-0.5 * ((Mg - mu) / sd) ** 2)
        else:
            pi = ((Mg >= M_range[0]) & (Mg <= M_range[1])).astype(float)
        pi /= _trapz(pi, Mg)
        dens = np.zeros_like(Mg)
        for i, M in enumerate(Mg):
            st = star.star(M, nu)
            if st is not None:
                dens[i] = kde([M, st['R_eq_circ']])[0]
        z = _trapz(pi * dens, Mg)
        out[p['name']] = np.log(z) if z > 0 else -np.inf
    return out


# =====================================================================
# 6. Post-processing: derived parameters, corner, predictive, summary
# =====================================================================
def add_derived(post, sources, mode, ftab, star, x_max, K=600):
    """Add j, q, q/j^2, R_eq, r_in, nu_U at the inner edge per source."""
    for s in sources:
        n = s.name
        cols = {k: np.full(len(post), np.nan) for k in
                ('j', 'q', 'q_over_j2', 'Qbar', 'R_eq', 'r_in', 'nuU_edge',
                 'is_isco', 'I45')}
        nu_known = s.spin[0] if s.spin is not None else np.nan
        for i, row in enumerate(post.to_dict('records')):
            cur, st = model_for(s, row, mode, ftab, star, x_max, K)
            if cur is None:
                continue
            M = row[f'M_{n}']
            cols['j'][i], cols['q'][i] = st['j'], st['q']
            cols['q_over_j2'][i] = st['q'] / st['j'] ** 2 \
                if st['j'] > 0 else np.nan
            cols['Qbar'][i] = -cols['q_over_j2'][i]
            cols['R_eq'][i] = st.get('R_eq_circ', np.nan)  # circumferential
            cols['r_in'][i] = st['r_in']
            cols['nuU_edge'][i] = cur['nu_U'][0]
            cols['is_isco'][i] = 1.0 if st['edge'] == 'isco' else 0.0
            # moment of inertia in 1e45 g cm^2.  EOS mode: from the star;
            # free/kerr: I = J/Omega = j M^2 / (2 pi nu_s / c) for a known spin
            if mode == 'eos':
                cols['I45'][i] = st['I'] * I_KM3_TO_45
            else:
                Om = 2.0 * np.pi * nu_known / C_KM_S
                cols['I45'][i] = (st['j'] * (M * M_SUN_KM) ** 2 / Om
                                  * I_KM3_TO_45)
        for k, v in cols.items():
            if mode != 'eos' and k in ('j', 'R_eq'):
                continue
            if not np.any(np.isfinite(v)):      # e.g. I45 for unknown spin
                continue
            if mode == 'free' and k == 'Qbar':
                continue
            if mode == 'kerr' and k in ('Qbar', 'q_over_j2'):
                continue
            post[f'{k}_{n}'] = v
    return post


I_KM3_TO_45 = 1.3466e43 / 1e45     # I [km^3, G=c=1] -> I [1e45 g cm^2]

LATEX = {'M': r'$M\,[M_\odot]$', 'nu_s': r'$\nu_s\,[{\rm Hz}]$',
         'eta_LF': r'$\log_{10}\eta_{\rm LF}$',
         'I45': r'$I\,[10^{45}\,{\rm g\,cm^2}]$',
         'sigma_int': r'$\sigma_{\rm int}\,[{\rm Hz}]$', 'j': r'$j$',
         'Qbar': r'$\bar Q$', 'q': r'$q$', 'q_over_j2': r'$q/j^2$',
         'R_eq': r'$R_{\rm eq,circ}\,[{\rm km}]$', 'r_in': r'$r_{\rm in}\,[{\rm km}]$',
         'nuU_edge': r'$\nu_U(r_{\rm in})\,[{\rm Hz}]$'}


def corner_derived(post, s, mode, path, truths=None, title=''):
    import corner
    keys = (['M', 'nu_s', 'sigma_int', 'eta_LF', 'j', 'Qbar', 'R_eq',
             'r_in', 'I45', 'nuU_edge'] if mode == 'eos'
            else ['M', 'j', 'Qbar', 'sigma_int', 'eta_LF', 'I45', 'r_in',
                  'nuU_edge'])
    cols = [f'{k}_{s.name}' for k in keys if f'{k}_{s.name}' in post
            and np.any(np.isfinite(post[f'{k}_{s.name}'].to_numpy(float)))]
    data = post[cols].to_numpy().astype(float)
    # scatter parameters are shown as log10
    for i, c in enumerate(cols):
        if c.startswith('sigma_int_') or c.startswith('eta_LF_'):
            data[:, i] = np.log10(data[:, i])
    ok = np.all(np.isfinite(data), axis=1)
    data = data[ok]
    keep = [i for i in range(data.shape[1]) if np.ptp(data[:, i]) > 0]
    data = data[:, keep]
    cols = [cols[i] for i in keep]
    labels = [LATEX.get(c[:-(len(s.name) + 1)], c) for c in cols]
    eta_lab = {'frac': r'$\log_{10}\eta$',
               'frac2': r'$\log_{10}\eta_{\rm kHz}$'}.get(SCATTER)
    labels = [eta_lab if (eta_lab and c.startswith('sigma_int_')) else l
              for c, l in zip(cols, labels)]
    tv = None
    if truths:          # 'M_<src>=1.8' (per source) or 'M=1.8' (all)
        tv = [truths.get(c, truths.get(c[:-(len(s.name) + 1)], None))
              for c in cols]
    fig = corner.corner(data, labels=labels, truths=tv, show_titles=True,
                        title_fmt='.3g', quantiles=[0.16, 0.5, 0.84],
                        levels=(0.393, 0.865), smooth=1.0, bins=30,
                        color='#1f4e79', truth_color='#CC3300',
                        title_kwargs={'fontsize': 10},
                        label_kwargs={'fontsize': 12},
                        hist_kwargs={'density': True})
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predictive(post, s, mode, ftab, star, x_max, path, title='',
                    n_draw=150, seed=1):
    """Data with posterior-predictive RPM tracks (nu_L vs nu_U, HBO)."""
    set_style()
    rng = np.random.default_rng(seed)
    rows = post.to_dict('records')
    idx = rng.choice(len(rows), min(n_draw, len(rows)), replace=False)
    has_hbo = s.has['HBO']
    fig, axes = plt.subplots(1, 2 if has_hbo else 1,
                             figsize=(12.5 if has_hbo else 7, 5.4))
    axes = np.atleast_1d(axes)
    for i in idx:
        cur, st = model_for(s, rows[i], mode, ftab, star, x_max, 400)
        if cur is None:
            continue
        axes[0].plot(cur['nu_U'], cur['nu_L'], color='#1f77b4', alpha=0.05,
                     lw=1)
        axes[0].plot(cur['nu_U'][0], cur['nu_L'][0], '.', color='#CC3300',
                     alpha=0.15, ms=3)
        if has_hbo:
            axes[1].plot(cur['nu_U'], cur['nu_HBO'], color='#1f77b4',
                         alpha=0.05, lw=1)
    med = {k: np.median(post[k]) for k in post.columns
           if np.issubdtype(post[k].dtype, np.number)}
    cur, st = model_for(s, med, mode, ftab, star, x_max, 400)
    if cur is not None:
        axes[0].plot(cur['nu_U'], cur['nu_L'], color='k', lw=1.8,
                     label='median parameters')
    d = s.data
    axes[0].errorbar(d['nu_U'], d['nu_L'], xerr=d['sig_U'], yerr=d['sig_L'],
                     fmt='o', ms=4, color='#CC3300', ecolor='0.4',
                     label=f'{s.name} data', zorder=5)
    axes[0].plot([], [], color='#1f77b4', label='posterior draws')
    axes[0].plot([], [], '.', color='#CC3300', label='inner edge')
    axes[0].set_xlabel(r'$\nu_U=\nu_\phi$ (Hz)')
    axes[0].set_ylabel(r'$\nu_L=\nu_\phi-\nu_r$ (Hz)')
    axes[0].legend(loc='upper left')
    lo = np.nanmin(d['nu_U']) * 0.85 if s.has['U'] else None
    hi = np.nanmax(d['nu_U']) * 1.15 if s.has['U'] else None
    if lo:
        axes[0].set_xlim(lo, hi)
    if s.has['L']:
        axes[0].set_ylim(np.nanmin(d['nu_L']) * 0.7,
                         np.nanmax(d['nu_L']) * 1.25)
    if has_hbo:
        axes[1].errorbar(d['nu_U'], d['nu_HBO'], xerr=d['sig_U'],
                         yerr=d['sig_HBO'], fmt='s', ms=4, color='#2ca02c',
                         ecolor='0.4')
        if cur is not None:
            axes[1].plot(cur['nu_U'], cur['nu_HBO'], 'k', lw=1.8)
        axes[1].set_xlabel(r'$\nu_U$ (Hz)')
        axes[1].set_ylabel(r'$\nu_{\rm HBO}=2\nu_{\rm nod}$ (Hz)')
        if lo:
            axes[1].set_xlim(lo, hi)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def summarise(post, sources, mode, extra=''):
    keys = (['M', 'nu_s', 'sigma_int', 'eta_LF', 'j', 'q', 'Qbar', 'R_eq',
             'r_in', 'I45', 'nuU_edge', 'is_isco'] if mode == 'eos'
            else ['M', 'j', 'Qbar', 'sigma_int', 'eta_LF', 'q', 'I45',
                  'r_in', 'nuU_edge', 'is_isco'])
    lines, tex = [extra], []
    for s in sources:
        lines.append(f"\n[{s.name}]")
        for k in keys:
            c = f'{k}_{s.name}'
            if c not in post:
                continue
            v = post[c].to_numpy()
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            if k == 'is_isco':
                lines.append(f"  P(inner edge = ISCO) = {v.mean():.3f}")
                continue
            lo, me, hi = np.percentile(v, [16, 50, 84])
            lines.append(f"  {k:10s} = {me:10.4g}  +{hi - me:.3g} "
                         f"-{me - lo:.3g}")
            tex.append((s.name, k, me, hi - me, me - lo))
    return '\n'.join(lines), tex


def write_tex(rows, path, caption):
    keys = []
    for r in rows:
        if r[1] not in keys:
            keys.append(r[1])
    srcs = []
    for r in rows:
        if r[0] not in srcs:
            srcs.append(r[0])
    with open(path, 'w') as f:
        f.write('\\begin{table}\n\\centering\n\\caption{' + caption +
                '}\n\\begin{tabular}{l' + 'c' * len(srcs) + '}\n\\hline\n')
        f.write('Parameter & ' + ' & '.join(srcs) + r' \\' + '\n\\hline\n')
        for k in keys:
            cells = []
            for sname in srcs:
                m = [r for r in rows if r[0] == sname and r[1] == k]
                cells.append(f'${m[0][2]:.4g}^{{+{m[0][3]:.2g}}}'
                             f'_{{-{m[0][4]:.2g}}}$' if m else '--')
            f.write(LATEX.get(k, k) + ' & ' + ' & '.join(cells) +
                    r' \\' + '\n')
        f.write('\\hline\n\\end{tabular}\n\\end{table}\n')
    print(f"  Saved: {path}")


# =====================================================================
# 7. Driver
# =====================================================================
def run_one(sources, mode, ftab, star, args, label, truths=None,
            nicer=None):
    outdir = os.path.join(args.outdir, label)
    os.makedirs(outdir, exist_ok=True)
    like = RPMLikelihood(sources, mode, ftab, star, args.x_max, args.K)
    pri = make_priors(sources, mode, (args.M_min, args.M_max))
    kw = dict(likelihood=like, priors=pri, sampler=args.sampler,
              outdir=outdir, label=label, resume=args.resume,
              plot=False, save=True)
    if args.sampler == 'dynesty':
        kw.update(nlive=args.nlive, sample=args.sample, dlogz=0.1)
        if args.npool > 1:
            kw['npool'] = args.npool
    elif args.sampler in ('emcee', 'zeus'):
        kw.update(nwalkers=args.nwalkers, nsteps=args.nsteps)
    res = bilby.run_sampler(**kw)

    post = add_derived(res.posterior.copy(), sources, mode, ftab, star,
                       args.x_max)
    post.to_csv(os.path.join(outdir, f'{label}_posterior_derived.csv'),
                index=False)
    try:
        res.plot_corner(filename=os.path.join(outdir,
                                              f'{label}_corner_sampled.png'))
    except Exception as e:
        print(f"  (bilby corner skipped: {e})")
    for s in sources:
        tr = None
        if truths:
            tr = {k: v for k, v in truths.items()}
        corner_derived(post, s, mode,
                       os.path.join(outdir, f'{label}_corner_{s.name}.png'),
                       truths=tr, title=f'{s.name}: {label}')
        plot_predictive(post, s, mode, ftab, star, args.x_max,
                        os.path.join(outdir,
                                     f'{label}_predictive_{s.name}.png'),
                        title=f'{s.name}: RPM posterior predictive ({label})')

    lnz_n = {}
    if nicer and star is not None:
        lnz_n = nicer_log_evidence(star, nicer, (args.M_min, args.M_max))
    head = (f"run {label}  mode={mode}\n"
            f"ln Z_QPO = {res.log_evidence:.3f} +- "
            f"{res.log_evidence_err:.3f}\n"
            + ''.join(f"ln Z_NICER[{k}] = {v:.3f}\n"
                      for k, v in lnz_n.items()))
    text, tex = summarise(post, sources, mode, head)
    print(text)
    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    write_tex(tex, os.path.join(outdir, 'summary.tex'),
              f'RPM parameter estimates ({label}); median and 68\\% '
              f'credible interval.')
    return {'label': label, 'lnZ': res.log_evidence,
            'lnZ_err': res.log_evidence_err,
            'lnZ_nicer': float(sum(lnz_n.values())) if lnz_n else 0.0,
            'post': post}


def compare_eos(results, sources, outdir):
    set_style()
    labs = [r['label'] for r in results]
    lnq = np.array([r['lnZ'] for r in results])
    err = np.array([r['lnZ_err'] for r in results])
    lnn = np.array([r['lnZ_nicer'] for r in results])
    tot = lnq + lnn
    best = int(np.argmax(tot))
    with open(os.path.join(outdir, 'evidence.txt'), 'w') as f:
        f.write(f"{'EOS':<16}{'lnZ_QPO':>12}{'err':>8}{'lnZ_NICER':>12}"
                f"{'lnZ_tot':>12}{'lnB_vs_best':>13}\n")
        for i, l in enumerate(labs):
            f.write(f"{l:<16}{lnq[i]:12.3f}{err[i]:8.3f}{lnn[i]:12.3f}"
                    f"{tot[i]:12.3f}{tot[i] - tot[best]:13.3f}\n")
    print(open(os.path.join(outdir, 'evidence.txt')).read())
    with open(os.path.join(outdir, 'evidence.tex'), 'w') as f:
        f.write('\\begin{table}\n\\centering\n\\caption{Evidences and Bayes '
                'factors relative to the preferred EOS.}\n'
                '\\begin{tabular}{lcccc}\n\\hline\n'
                'EOS & $\\ln\\mathcal{Z}_{\\rm QPO}$ & '
                '$\\ln\\mathcal{Z}_{\\rm NICER}$ & '
                '$\\ln\\mathcal{Z}_{\\rm tot}$ & $\\ln B$ \\\\\n\\hline\n')
        for i, l in enumerate(labs):
            f.write(f'{l} & ${lnq[i]:.2f}\\pm{err[i]:.2f}$ & {lnn[i]:.2f} '
                    f'& {tot[i]:.2f} & {tot[i] - tot[best]:.2f} \\\\\n')
        f.write('\\hline\n\\end{tabular}\n\\end{table}\n')

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(labs))
    ax.bar(xs - 0.2, lnq - lnq.max(), 0.4, yerr=err, color='#1f77b4',
           label='QPO only')
    if np.any(lnn != 0):
        ax.bar(xs + 0.2, tot - tot.max(), 0.4, color='#CC3300',
               label='QPO + NICER')
    for y, txt in ((-1.0, 'substantial'), (-2.3, 'strong'),
                   (-4.6, 'decisive')):
        ax.axhline(y, color='0.5', ls=':', lw=0.9)
        ax.text(len(labs) - 0.5, y, txt, va='bottom', ha='right',
                fontsize=8, color='0.4')
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r'$\ln B$ relative to the best EOS')
    ax.legend()
    plt.tight_layout()
    p = os.path.join(outdir, 'bayes_factors.png')
    plt.savefig(p)
    plt.close(fig)
    print(f"  Saved: {p}")

    for s in sources:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for r in results:
            v = r['post'][f'M_{s.name}'].to_numpy()
            ax.hist(v, bins=40, density=True, histtype='step', lw=1.8,
                    label=f"{r['label']} ({np.median(v):.3f})")
        ax.set_xlabel(r'$M\ [M_\odot]$')
        ax.set_ylabel('posterior density')
        ax.set_title(f'{s.name}: mass posterior per EOS')
        ax.legend()
        plt.tight_layout()
        p = os.path.join(outdir, f'mass_posteriors_{s.name}.png')
        plt.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")


# =====================================================================
# 8. Mock data and table check
# =====================================================================
def make_mock(args):
    """
    Synthetic QPO pairs from the FULL code path (ht_common + qpo.py, not the
    tables), so a fit to them tests both the tables and the sampler.
    """
    import ht_common
    from qpo import qpo_at_radius
    rng = np.random.default_rng(args.seed)
    m = ht_common.solve_ht_mass(args.eos, args.mass, args.spin)
    r_in, edge = ht_common.inner_edge(m)
    x_in = r_in / m.M_geo
    xs = rng.uniform(x_in * 1.02, x_in * args.xspan, args.n)
    rows = []
    for x in xs:
        f = qpo_at_radius(m.M_geo, m.j, m.q, x * m.M_geo)
        sU = args.sigma * rng.uniform(0.6, 1.4)
        sL = args.sigma * rng.uniform(0.6, 1.4)
        row = [f['nu_peri'] + rng.normal(0, sL), sL,
               f['nu_phi'] + rng.normal(0, sU), sU]
        if args.hbo:
            sH = max(0.5, 0.05 * 2 * f['nu_nod'])
            row += [2 * f['nu_nod'] + rng.normal(0, sH), sH]
        rows.append(row)
    hdr = 'nu_L,sig_L,nu_U,sig_U' + (',nu_HBO,sig_HBO' if args.hbo else '')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savetxt(args.out, np.array(rows), delimiter=',', header=hdr,
               comments='', fmt='%.4f')
    print(f"  wrote {args.out}: {args.n} rows")
    print(f"  truth: M = {m.M_solar:.4f}, spin = {args.spin}, j = {m.j:.4f}, "
          f"q = {m.q:.4f}, q/j^2 = {m.q / m.j ** 2:.3f}, "
          f"R_eq = {m.R_eq_circ_km:.3f} km, r_in = {r_in:.3f} km ({edge})")


def check_table(args):
    """Compare table frequencies with direct qpo.py evaluation."""
    import ht_common
    from qpo import qpo_at_radius
    ftab = FreqTable()
    star = StarTable(args.eos)
    worst = 0.0
    print(f"  {'M':>5} {'spin':>5} {'r/M':>6} {'quantity':>9} "
          f"{'table':>10} {'direct':>10} {'rel.err':>9}")
    for M, nu in ((1.4, 363.0), (1.8, 581.0), (2.0, 400.0)):
        if M > star.M_max(nu):
            continue
        m = ht_common.solve_ht_mass(args.eos, M, nu)
        st = star.star(M, nu)
        print(f"  star M={M} nu={nu}: j {st['j']:.5f} vs {m.j:.5f}, "
              f"q {st['q']:.5f} vs {m.q:.5f}, R_eq {st['R_eq']:.4f} vs "
              f"{m.R_eq_km:.4f}")
        fc = ftab.curves(st['j'], st['q'], st['x_surf'], 25.0, 3000)
        cur = hz_curves(fc, M)
        for xr in (1.05, 1.3, 1.8, 2.5):
            x = fc['x_in'] * xr
            if x > 25:
                continue
            d = qpo_at_radius(m.M_geo, m.j, m.q, x * m.M_geo)
            for key, kd in (('nu_U', 'nu_phi'), ('nu_L', 'nu_peri'),
                            ('nu_HBO', None)):
                tv = np.interp(x, cur['x'], cur[key])
                dv = d[kd] if kd else 2 * d['nu_nod']
                e = abs(tv / dv - 1)
                worst = max(worst, e if key != 'nu_HBO' else 0)
                print(f"  {M:5.2f} {nu:5.0f} {x:6.2f} {key:>9} {tv:10.3f} "
                      f"{dv:10.3f} {e:9.2e}")
    print(f"\n  worst relative error (nu_U, nu_L): {worst:.2e}")


# =====================================================================
# CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser(
        description='Bayesian RPM fits of kHz QPOs in the HT spacetime',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build-tables', help='frequency + star tables')
    b.add_argument('--eos', nargs='*', default=[])
    b.add_argument('--nu-ref', type=float, default=400.0)
    b.add_argument('--n-rho', type=int, default=64)
    b.add_argument('--npool', type=int, default=1)
    b.add_argument('--skip-freq', action='store_true')

    c = sub.add_parser('check-table', help='table vs direct qpo.py')
    c.add_argument('--eos', required=True)

    m = sub.add_parser('mock', help='synthetic QPO data from the full code')
    m.add_argument('--eos', required=True)
    m.add_argument('--mass', type=float, required=True)
    m.add_argument('--spin', type=float, required=True)
    m.add_argument('--n', type=int, default=40)
    m.add_argument('--sigma', type=float, default=6.0)
    m.add_argument('--xspan', type=float, default=1.9,
                   help='radii drawn in [1.02, xspan] x r_in')
    m.add_argument('--hbo', action='store_true')
    m.add_argument('--seed', type=int, default=42)
    m.add_argument('--out', required=True)

    f = sub.add_parser('fit', help='run the sampler')
    f.add_argument('--config', required=True)
    f.add_argument('--mode', choices=['eos', 'free', 'kerr'], default='eos',
                   help='eos: (M, nu_s) with an EOS; free: (M, j, Qbar); '
                        'kerr: (M, j) with Qbar = 1')
    f.add_argument('--eos', nargs='*', default=[])
    f.add_argument('--outdir', default='qpo_fit_out')
    f.add_argument('--sampler', default='dynesty')
    f.add_argument('--sample', default='act-walk',
                   help="dynesty sampling method (bilby name)")
    f.add_argument('--nlive', type=int, default=500)
    f.add_argument('--nwalkers', type=int, default=40)
    f.add_argument('--nsteps', type=int, default=3000)
    f.add_argument('--npool', type=int, default=1)
    f.add_argument('--x-max', type=float, default=25.0,
                   help='outer radius r/M of the QPO region (prior edge)')
    f.add_argument('--K', type=int, default=1200,
                   help='grid points for the radius marginalisation')
    f.add_argument('--M-min', type=float, default=1.0)
    f.add_argument('--M-max', type=float, default=2.6)
    f.add_argument('--truth', nargs='*', default=[],
                   help='true values for the corner plots, per source '
                        '(M_4U1636=1.8 nu_s_4U1636=581) or for all (M=1.8)')
    f.add_argument('--resume', action='store_true')
    f.add_argument('--scatter', choices=['abs', 'frac', 'frac2'],
                   default='abs',
                   help="abs: sigma_int in Hz; frac: sigma_int is a "
                        "fractional scatter eta; frac2: eta for the kHz QPOs "
                        "and a separate eta_LF for the LF QPO / HBO (v3.2)")

    args = p.parse_args()

    if args.cmd == 'build-tables':
        if not args.skip_freq:
            build_freq_table(npool=args.npool)
        for e in args.eos:
            build_star_table(e, args.nu_ref, args.n_rho, npool=args.npool)
        return
    if args.cmd == 'check-table':
        check_table(args)
        return
    if args.cmd == 'mock':
        make_mock(args)
        return

    if bilby is None:
        sys.exit("bilby is required:  pip install bilby corner dynesty")
    global SCATTER
    SCATTER = args.scatter
    sources, nicer = load_config(args.config)
    ftab = FreqTable()
    truths = {}
    for t in args.truth:
        k, v = t.split('=')
        truths[k] = float(v)
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode in ('free', 'kerr'):
        run_one(sources, args.mode, ftab, None, args, args.mode, truths)
        return

    if not args.eos:
        sys.exit("--mode eos needs --eos <files>")
    results = []
    for e in args.eos:
        star = StarTable(e)
        print(f"\n=== EOS {star.tag}: M_max(spin 0) = {star.M_max():.4f} "
              f"Msun ===")
        results.append(run_one(sources, 'eos', ftab, star, args, star.tag,
                               truths, nicer))
    if len(results) > 1 or nicer:
        compare_eos(results, sources, args.outdir)


if __name__ == '__main__':
    main()
