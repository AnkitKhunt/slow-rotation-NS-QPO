#!/usr/bin/env python3
"""
orbits.py

Particle orbits in the Hartle-Thorne spacetime.

v2 (physics-corrected)
----------------------
  * metric: full Hartle-Thorne exterior (ht_full.HTExterior) incl. g_rr,
    g_thth and the K*Q2^1 term -- same metric as qpo.py / veff.py
  * 4-velocity properly normalised (u.u = -1), inclination = orbit tilt,
    optional radial kick (--ecc) so a periastron rosette really appears
  * vectorised Christoffels, 4th-order derivatives
  * x-y projection r sin(th); phase colouring consistent with colourbar
  * conservation plot shows the actual drift of E, L_z and u.u
  * new plots: radial motion, effective potential, E(r)/L(r), Poincare
    section, FFT frequency check against qpo.py (nu_phi, nu_r, nu_theta)

Neutron-star image placed at the centre, auto-cropped only against
fully transparent borders, sized to fill_fraction * r_star.
Accretion-disc fog drawn as a translucent red -> tomato -> brown
annulus outside the ISCO.  Blue dotted ISCO circle.
Azimuthal-phase colorbar.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp

from eos_loader import EOS
from ht_full import HTModel
from qpo import find_isco_km, qpo_at_radius, circular_orbit
import ht_common


# =====================================================================
# CONFIG
# =====================================================================
CONFIG = {
    'eos':             'eos/eosAU',
    'mass':            1.8,
    'spin':            400.0,

    'r_circular':      None,
    'n_orbits_circ':   3,

    'r_start':         None,
    'theta_start_deg': 45.0,
    'n_orbits_geo':    60.0,
    'ecc':             0.1,   # radial kick, fraction of orbital speed

    'outdir':          'orbits_out',
}

CMAP_PHI   = 'plasma'
STAR_IMAGE = None          # optional picture drawn over the star (--star-image)

# Accretion-disc background configuration
DISC_INNER_FACTOR   = 1.0      # inner edge = DISC_INNER_FACTOR * r_ISCO
DISC_OUTER_FACTOR   = 3.20     # outer edge = DISC_OUTER_FACTOR * r_ISCO
DISC_LAYERS         = int(140) # radial gradient rings

# Colour ramp:  deep red inside -> tomato mid -> dark brown outside
DISC_HOT_COLOR      = (0.85, 0.05, 0.05)
DISC_MID_COLOR      = (1.00, 0.38, 0.28)
DISC_BASE_COLOR     = (0.35, 0.10, 0.05)

DISC_MAX_ALPHA      = 0.55
DISC_FALLOFF_POWER  = 1.2

DISC_CLOUD_N        = int(3500)
DISC_CLOUD_SIZE     = 70
DISC_CLOUD_ALPHA    = 0.22

DISC_SWIRL_N        = 0        # no spiral arms
DISC_SWIRL_POINTS   = int(500) # unused when DISC_SWIRL_N = 0
# =====================================================================


# =====================================================================
# Style
# =====================================================================
def set_style():
    plt.rcParams.update({
        'figure.figsize':      (8.0, 7.0),
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.facecolor':   'white',
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'font.family':         'serif',
        'font.serif':          ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset':    'dejavuserif',
        'font.size':           11,
        'axes.labelsize':      12,
        'axes.titlesize':      13,
        'axes.linewidth':      1.0,
        'axes.grid':           True,
        'grid.alpha':          0.35,
        'grid.linestyle':      '--',
        'grid.linewidth':      0.6,
        'grid.color':          '#888888',
        'lines.linewidth':     1.8,
        'legend.fontsize':     10,
        'legend.frameon':      True,
        'legend.framealpha':   0.95,
        'legend.edgecolor':    '0.7',
        'xtick.direction':     'in',
        'ytick.direction':     'in',
    })


# =====================================================================
# Accretion-disc background
#   Radial ramp:  deep red inside -> tomato mid -> dark brown outside
# =====================================================================
def _disc_color_at(t):
    """Piecewise colour ramp."""
    if t <= 0.5:
        u = t / 0.5
        r = DISC_HOT_COLOR[0] * (1 - u) + DISC_MID_COLOR[0] * u
        g = DISC_HOT_COLOR[1] * (1 - u) + DISC_MID_COLOR[1] * u
        b = DISC_HOT_COLOR[2] * (1 - u) + DISC_MID_COLOR[2] * u
    else:
        u = (t - 0.5) / 0.5
        r = DISC_MID_COLOR[0] * (1 - u) + DISC_BASE_COLOR[0] * u
        g = DISC_MID_COLOR[1] * (1 - u) + DISC_BASE_COLOR[1] * u
        b = DISC_MID_COLOR[2] * (1 - u) + DISC_BASE_COLOR[2] * u
    return (r, g, b)


def draw_accretion_disc(ax, r_isco, r_outer=None, zorder=1, seed=42):
    """Smooth dense accretion-disc background, no isolated blobs."""
    rng = np.random.default_rng(seed)

    r_in  = DISC_INNER_FACTOR * r_isco
    r_out = (DISC_OUTER_FACTOR * r_isco) if r_outer is None else r_outer

    # Guard against invalid range
    if r_out <= r_in * 1.02:
        r_out = DISC_OUTER_FACTOR * r_isco

    n = max(int(DISC_CLOUD_N), 1)

    # ---- 1. Radial gradient ----
    radii = np.linspace(r_in, r_out, int(DISC_LAYERS))
    for i in range(int(DISC_LAYERS) - 1):
        r_a = radii[i]
        r_b = radii[i + 1]
        t = i / max(int(DISC_LAYERS) - 2, 1)
        col = _disc_color_at(t)
        alpha = DISC_MAX_ALPHA * (1.0 - t) ** DISC_FALLOFF_POWER
        ax.add_patch(
            Wedge((0, 0), r_b, 0, 360,
                  width=(r_b - r_a),
                  facecolor=col, edgecolor='none',
                  alpha=alpha, zorder=zorder)
        )

    # ---- 2. Spiral arms (only if DISC_SWIRL_N > 0) ----
    for arm in range(int(DISC_SWIRL_N)):
        phi0 = rng.uniform(0.0, 2.0 * np.pi)
        r_arm = np.geomspace(r_in * 1.02, r_out * 0.98,
                             int(DISC_SWIRL_POINTS))
        phi_arm = phi0 + 2.0 * np.pi * np.log(r_arm / r_in) / np.log(1.8)
        x_arm = r_arm * np.cos(phi_arm)
        y_arm = r_arm * np.sin(phi_arm)
        for j in range(int(DISC_SWIRL_POINTS) - 1):
            t = j / (int(DISC_SWIRL_POINTS) - 1)
            col = _disc_color_at(t)
            alpha_j = 0.45 * (1.0 - t) ** 1.3
            ax.plot(x_arm[j:j+2], y_arm[j:j+2],
                    color=col, alpha=alpha_j,
                    linewidth=2.4, solid_capstyle='butt',
                    zorder=zorder + 1)

    # ---- 3. Dense cloud blobs ----
    r_cloud = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    phi_cloud = rng.uniform(0.0, 2.0 * np.pi, n)
    x_cloud = r_cloud * np.cos(phi_cloud)
    y_cloud = r_cloud * np.sin(phi_cloud)

    t = (r_cloud - r_in) / max(r_out - r_in, 1e-12)

    mask = t <= 0.5
    cols = np.empty((n, 3))
    u = np.where(mask, t / 0.5, 0.0)
    cols[mask, 0] = DISC_HOT_COLOR[0] * (1 - u[mask]) + DISC_MID_COLOR[0] * u[mask]
    cols[mask, 1] = DISC_HOT_COLOR[1] * (1 - u[mask]) + DISC_MID_COLOR[1] * u[mask]
    cols[mask, 2] = DISC_HOT_COLOR[2] * (1 - u[mask]) + DISC_MID_COLOR[2] * u[mask]

    u2 = np.where(~mask, (t - 0.5) / 0.5, 0.0)
    cols[~mask, 0] = DISC_MID_COLOR[0] * (1 - u2[~mask]) + DISC_BASE_COLOR[0] * u2[~mask]
    cols[~mask, 1] = DISC_MID_COLOR[1] * (1 - u2[~mask]) + DISC_BASE_COLOR[1] * u2[~mask]
    cols[~mask, 2] = DISC_MID_COLOR[2] * (1 - u2[~mask]) + DISC_BASE_COLOR[2] * u2[~mask]

    colors = np.concatenate(
        [cols, np.full((n, 1), DISC_CLOUD_ALPHA)], axis=1)

    ax.scatter(x_cloud, y_cloud,
               s=DISC_CLOUD_SIZE,
               c=colors, edgecolors='none',
               zorder=zorder + 2)


def draw_accretion_disc_3d(ax, r_isco, r_outer=None, zorder=1, seed=42):
    """3D version: same red->tomato->brown ramp."""
    rng = np.random.default_rng(seed)
    r_in  = DISC_INNER_FACTOR * r_isco
    r_out = (DISC_OUTER_FACTOR * r_isco) if r_outer is None else r_outer
    if r_out <= r_in * 1.02:
        r_out = DISC_OUTER_FACTOR * r_isco

    n = max(int(DISC_CLOUD_N), 1)

    radii = np.linspace(r_in, r_out, int(DISC_LAYERS))
    theta = np.linspace(0.0, 2.0 * np.pi, 220)
    for i in range(int(DISC_LAYERS) - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        t = i / max(int(DISC_LAYERS) - 2, 1)
        col = _disc_color_at(t)
        alpha = DISC_MAX_ALPHA * (1.0 - t) ** DISC_FALLOFF_POWER
        ax.plot(r_mid * np.cos(theta),
                r_mid * np.sin(theta),
                np.zeros_like(theta),
                color=col, alpha=alpha, linewidth=4.0,
                zorder=zorder)

    r_cloud = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    phi_cloud = rng.uniform(0.0, 2.0 * np.pi, n)
    z_cloud = rng.normal(0.0, 0.03 * r_out, n)
    x_cloud = r_cloud * np.cos(phi_cloud)
    y_cloud = r_cloud * np.sin(phi_cloud)

    t = (r_cloud - r_in) / max(r_out - r_in, 1e-12)
    mask = t <= 0.5
    cols = np.empty((n, 3))
    u = np.where(mask, t / 0.5, 0.0)
    cols[mask, 0] = DISC_HOT_COLOR[0] * (1 - u[mask]) + DISC_MID_COLOR[0] * u[mask]
    cols[mask, 1] = DISC_HOT_COLOR[1] * (1 - u[mask]) + DISC_MID_COLOR[1] * u[mask]
    cols[mask, 2] = DISC_HOT_COLOR[2] * (1 - u[mask]) + DISC_MID_COLOR[2] * u[mask]

    u2 = np.where(~mask, (t - 0.5) / 0.5, 0.0)
    cols[~mask, 0] = DISC_MID_COLOR[0] * (1 - u2[~mask]) + DISC_BASE_COLOR[0] * u2[~mask]
    cols[~mask, 1] = DISC_MID_COLOR[1] * (1 - u2[~mask]) + DISC_BASE_COLOR[1] * u2[~mask]
    cols[~mask, 2] = DISC_MID_COLOR[2] * (1 - u2[~mask]) + DISC_BASE_COLOR[2] * u2[~mask]

    colors = np.concatenate(
        [cols, np.full((n, 1), DISC_CLOUD_ALPHA)], axis=1)

    ax.scatter(x_cloud, y_cloud, z_cloud,
               s=DISC_CLOUD_SIZE,
               c=colors, edgecolors='none',
               depthshade=False,
               zorder=zorder + 2)


# =====================================================================
# Neutron star -- image based, transparent-crop only
# =====================================================================
def draw_neutron_star_image(ax, r_star, image_path=None,
                             zorder=6, fill_fraction=0.7,
                             crop_transparent=True,
                             zoom_max=50.0):
    """
    Place a PNG/JPG image at the origin (0,0) with footprint equal
    to radius r_star * fill_fraction in data coordinates.

    Only fully-transparent borders are auto-cropped.  White or light
    borders are preserved, so the visible content lands exactly
    where you drew it in the PNG.
    """
    image_path = image_path or STAR_IMAGE   # value of --star-image, read at call time
    if not image_path:
        return                      # no picture requested
    if not os.path.isfile(image_path):
        print(f"  [warning] star image not found: {image_path}")
        return

    img = imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0

    # ---- Auto-crop fully transparent borders only ----
    if crop_transparent and img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        rows = np.where(alpha.max(axis=1) > 0.01)[0]
        cols = np.where(alpha.max(axis=0) > 0.01)[0]
        if len(rows) and len(cols):
            img = img[rows[0]:rows[-1] + 1,
                      cols[0]:cols[-1] + 1, :]

    # ---- Compute zoom so image radius = r_star * fill_fraction ----
    fig = ax.figure
    fig.canvas.draw()

    try:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((r_star * fill_fraction, 0.0))
        r_pix = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    except Exception:
        r_pix = 0.0

    h_pix, w_pix = img.shape[:2]
    zoom = (2.0 * r_pix) / max(h_pix, w_pix)

    if not np.isfinite(zoom) or zoom <= 0:
        zoom = 0.5
    zoom = min(zoom, zoom_max)

    oi = OffsetImage(img, zoom=zoom, zorder=zorder)
    ab = AnnotationBbox(oi, (0.0, 0.0),
                        frameon=False,
                        xycoords='data',
                        box_alignment=(0.5, 0.5),
                        pad=0.0, zorder=zorder)
    ax.add_artist(ab)


def draw_neutron_star_3d_image(ax, r_star, image_path=None,
                                zorder=6):
    image_path = image_path or STAR_IMAGE
    if not image_path:
        return
    if not os.path.isfile(image_path):
        print(f"  [warning] star image not found: {image_path}")
        return

    img = imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 80)
    U, V = np.meshgrid(u, v)

    X = r_star * np.cos(U) * np.sin(V)
    Y = r_star * np.sin(U) * np.sin(V)
    Z = r_star * np.cos(V)

    h, w = img.shape[:2]
    img_v = (V / np.pi * (h - 1)).astype(int)
    img_u = (U / (2.0 * np.pi) * (w - 1)).astype(int)
    facecolors = img[img_v, img_u]

    ax.plot_surface(X, Y, Z,
                    facecolors=facecolors,
                    linewidth=0, antialiased=True,
                    shade=False, zorder=zorder)


def draw_isco(ax, r_isco, color='#0066CC', ls=':', lw=1.6,
              zorder=8, label=None):
    ax.add_patch(Circle((0, 0), r_isco, fill=False,
                         edgecolor=color, linestyle=ls,
                         linewidth=lw, zorder=zorder, label=label))


def draw_isco_3d(ax, r_isco, color='#0066CC'):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r_isco * np.cos(theta), r_isco * np.sin(theta),
            np.zeros_like(theta), color=color,
            linestyle=':', linewidth=1.6, alpha=0.95, zorder=8)


# =====================================================================
# Start / end markers
# =====================================================================
def _star_marker(ax, x, y, color='#22CC22', edge='#111111',
                  size=420, zorder=12, label=None):
    ax.scatter([x], [y], marker='*', s=size,
               facecolor=color, edgecolor=edge,
               linewidth=1.0, zorder=zorder, label=label)
    ax.scatter([x], [y], marker='o', s=size * 0.10,
               facecolor=edge, edgecolor=edge,
               linewidth=0.0, zorder=zorder + 1)


def mark_start_end_2d(ax, x, y, label_start='start', label_end='end',
                       zorder=12):
    _star_marker(ax, x[0], y[0], color='#22CC22', edge='#111111',
                  size=420, zorder=zorder, label=label_start)
    _star_marker(ax, x[-1], y[-1], color='#E02020', edge='#111111',
                  size=420, zorder=zorder, label=label_end)


def _star_marker_3d(ax, x, y, z, color='#22CC22', edge='#111111',
                     size=420, zorder=12, label=None):
    ax.scatter([x], [y], [z], marker='*', s=size,
               facecolor=color, edgecolor=edge,
               linewidth=1.0, depthshade=False,
               zorder=zorder, label=label)
    ax.scatter([x], [y], [z], marker='o', s=size * 0.10,
               facecolor=edge, edgecolor=edge,
               linewidth=0.0, depthshade=False,
               zorder=zorder + 1)


def mark_start_end_3d(ax, x, y, z, label_start='start', label_end='end'):
    _star_marker_3d(ax, x[0], y[0], z[0], color='#22CC22',
                     edge='#111111', size=420, label=label_start)
    _star_marker_3d(ax, x[-1], y[-1], z[-1], color='#E02020',
                     edge='#111111', size=420, label=label_end)


# =====================================================================
# Colored line helpers
# =====================================================================
def _colored_line_2d(ax, x, y, cmap_name=CMAP_PHI, lw=2.0,
                      alpha=1.0, zorder=10, values=None):
    """Line coloured by `values` (default: sample index)."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    if values is None:
        values = np.arange(len(x), dtype=float)
    values = np.asarray(values, dtype=float)
    seg_vals = 0.5 * (values[:-1] + values[1:])
    norm = Normalize(float(np.min(values)), float(np.max(values)))
    cmap = plt.get_cmap(cmap_name)
    lc = LineCollection(segs, cmap=cmap, norm=norm,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(seg_vals)
    ax.add_collection(lc)
    return lc


def _colored_line_3d(ax, x, y, z, cmap_name=CMAP_PHI, lw=1.6,
                      alpha=1.0, zorder=10):
    cmap = plt.get_cmap(cmap_name)
    n = len(x)
    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                color=cmap(i / max(n - 1, 1)),
                linewidth=lw, alpha=alpha, zorder=zorder)


def add_phi_colorbar(fig, ax, phi_vals, cmap_name=CMAP_PHI,
                     label=r'azimuthal phase $\phi$ (rad)'):
    norm = Normalize(vmin=float(np.min(phi_vals)),
                     vmax=float(np.max(phi_vals)))
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array(phi_vals)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    return cbar




# =====================================================================
# Metric wrapper (full HT exterior, equatorial AND off-equatorial)
# =====================================================================
class HTMetricFull:
    """
    4x4 metric (t, r, theta, phi) built from ht_full.HTExterior.

    v2: g_rr and g_thth now carry the O(j^2, q) Hartle-Thorne terms and
    g_phiphi/g_thth the K*Q2^1 term (previous version: Schwarzschild g_rr,
    g_thth = r^2).  Derivatives: 4th-order central differences.
    """

    def __init__(self, exterior):
        self.ext = exterior
        self.M = exterior.M
        self.r_min = 2.05 * exterior.M

    def _comp(self, r, th):
        return np.array(self.ext.metric_components(r, th))

    def g_munu(self, r, theta):
        gtt, gtp, gpp, grr, gth = self.ext.metric_components(r, theta)
        g = np.zeros((4, 4))
        g[0, 0] = gtt
        g[0, 3] = g[3, 0] = gtp
        g[1, 1] = grr
        g[2, 2] = gth
        g[3, 3] = gpp
        return g

    def g_inv(self, r, theta):
        gtt, gtp, gpp, grr, gth = self.ext.metric_components(r, theta)
        det = gtt * gpp - gtp * gtp
        gi = np.zeros((4, 4))
        gi[0, 0] = gpp / det
        gi[0, 3] = gi[3, 0] = -gtp / det
        gi[3, 3] = gtt / det
        gi[1, 1] = 1.0 / grr
        gi[2, 2] = 1.0 / gth
        return gi

    def dg_dx(self, r, theta):
        """dg[k, a, b] = d g_ab / d x^k  (only k = r, theta non-zero)."""
        hr, ht = 1e-3 * r, 1e-3
        c = self._comp
        dr = (-c(r + 2 * hr, theta) + 8 * c(r + hr, theta)
              - 8 * c(r - hr, theta) + c(r - 2 * hr, theta)) / (12 * hr)
        dth = (-c(r, theta + 2 * ht) + 8 * c(r, theta + ht)
               - 8 * c(r, theta - ht) + c(r, theta - 2 * ht)) / (12 * ht)
        dg = np.zeros((4, 4, 4))
        for k, d in ((1, dr), (2, dth)):
            dg[k, 0, 0] = d[0]
            dg[k, 0, 3] = dg[k, 3, 0] = d[1]
            dg[k, 3, 3] = d[2]
            dg[k, 1, 1] = d[3]
            dg[k, 2, 2] = d[4]
        return dg

    def christoffel(self, r, theta):
        """Gamma^mu_ab, shape (4, 4, 4)."""
        gi = self.g_inv(r, theta)
        dg = self.dg_dx(r, theta)
        S = dg + dg.transpose(1, 0, 2) - dg.transpose(1, 2, 0)
        # S[a,b,n] = d_a g_bn + d_b g_an - d_n g_ab
        return 0.5 * np.einsum('mn,abn->mab', gi, S)

    def conserved(self, r, theta, u):
        """E = -u_t, Lz = u_phi, u.u  for 4-velocity u = (u^t,u^r,u^th,u^ph)."""
        g = self.g_munu(r, theta)
        ul = g @ u
        return -ul[0], ul[3], float(u @ ul)


# =====================================================================
# Circular orbit (same orbit definition as qpo.py)
# =====================================================================
def circular_omega(metric_full, r):
    co = circular_orbit(metric_full.ext, r)
    return np.nan if co is None else co['Omega']


def build_circular_orbit(metric_full, r0, n_orbits, n_pts_per_orbit=400):
    co = circular_orbit(metric_full.ext, r0)
    if co is None:
        return None
    Om = co['Omega']
    T = 2.0 * np.pi / Om
    n_pts = int(n_orbits * n_pts_per_orbit)
    t = np.linspace(0.0, n_orbits * T, n_pts)
    return {
        'lam': t / co['ut'], 't': t,
        'r': np.full(n_pts, r0),
        'theta': np.full(n_pts, np.pi / 2),
        'phi': Om * t,
        'Omega': Om, 'T': T, 'ut': co['ut'], 'E': co['E'], 'Lz': co['L'],
    }


# =====================================================================
# Geodesic equation  d u^mu / d lam = -Gamma^mu_ab u^a u^b
# =====================================================================
def geodesic_rhs(lam, y, metric_full):
    r, th = y[1], y[2]
    u = y[4:8]
    if not np.isfinite(r) or r <= metric_full.r_min:
        return np.zeros(8)
    G = metric_full.christoffel(r, th)
    acc = -np.einsum('mab,a,b->m', G, u, u)
    return np.concatenate([u, acc])


def integrate_geodesic(metric_full, r0, tilt_deg, n_orbits, n_pts=6000,
                       ecc=0.0, r_stop=None):
    """
    Timelike geodesic starting at (r0, theta = 90 deg, phi = 0), at a node.

    tilt_deg = theta_start: the orbit reaches colatitude ~ theta_start, i.e.
               inclination i = 90 - tilt_deg  (CLI/CONFIG meaning unchanged).
    ecc      : initial radial speed as a fraction of the circular orbital
               speed (0 = start on the circular radius, no radial epicycle).

    The angular speed equals the circular-orbit Omega(r0), split into
    phi-dot = Omega cos i and theta-dot = Omega sin i, and u^t is then fixed
    by u.u = -1 (the old code skipped this step: u.u was ~ -0.988).
    """
    ext = metric_full.ext
    co = circular_orbit(ext, r0)
    if co is None:
        return None
    w = co['Omega']
    inc = np.radians(90.0 - tilt_deg)
    phid, thd, rd = w * np.cos(inc), w * np.sin(inc), ecc * w * r0

    gtt, gtp, gpp, grr, gth = ext.metric_components(r0, np.pi / 2)
    n2 = -(gtt + 2 * gtp * phid + gpp * phid**2 + grr * rd**2 + gth * thd**2)
    if n2 <= 0:
        return None
    ut = 1.0 / np.sqrt(n2)
    u0 = np.array([ut, ut * rd, ut * thd, ut * phid])
    E0, L0, _ = metric_full.conserved(r0, np.pi / 2, u0)

    y0 = np.concatenate([[0.0, r0, np.pi / 2, 0.0], u0])
    lam_max = n_orbits * 2.0 * np.pi / u0[3]
    r_stop = metric_full.r_min if r_stop is None else r_stop

    def hit_surface(lam, y, *a):
        return y[1] - r_stop
    hit_surface.terminal = True
    hit_surface.direction = -1

    sol = solve_ivp(geodesic_rhs, [0.0, lam_max], y0, args=(metric_full,),
                    t_eval=np.linspace(0.0, lam_max, n_pts),
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    events=hit_surface)
    if sol.y.shape[1] < 10:
        return None

    t, r, th, ph, ut_, ur, uth, uph = sol.y
    E_t = np.empty(len(r)); L_t = np.empty(len(r)); N_t = np.empty(len(r))
    for k in range(len(r)):
        E_t[k], L_t[k], N_t[k] = metric_full.conserved(
            r[k], th[k], sol.y[4:8, k])

    return {
        'lam': sol.t, 't': t, 'r': r, 'theta': th, 'phi': ph,
        'ut': ut_, 'ur': ur, 'uth': uth, 'uph': uph,
        'E': E0, 'Lz': L0, 'E_t': E_t, 'Lz_t': L_t, 'norm_t': N_t,
        'inc_deg': 90.0 - tilt_deg, 'ecc': ecc, 'r0': r0,
        'terminated': bool(sol.t_events[0].size) if sol.t_events else False,
    }


def verify_circular(metric_full, r0, n_orbits=3):
    """Integrate the geodesic from circular-orbit data: r must stay r0."""
    tr = integrate_geodesic(metric_full, r0, 90.0, n_orbits,
                            n_pts=600, ecc=0.0)
    if tr is None:
        return None
    return {'dr_max': float(np.max(np.abs(tr['r'] - r0)) / r0),
            'dE': float(np.ptp(tr['E_t']) / abs(tr['E'])),
            'dnorm': float(np.max(np.abs(tr['norm_t'] + 1.0)))}


# =====================================================================
# Frequencies measured from the trajectory (for the qpo.py cross-check)
# =====================================================================
def _peak_freq(t_u, sig, f_lo, f_hi, pad=32):
    """Dominant frequency (cycles per unit t) in [f_lo, f_hi]."""
    s = sig - np.mean(sig)
    s = s * np.hanning(len(s))
    n = len(s) * pad
    F = np.abs(np.fft.rfft(s, n))
    f = np.fft.rfftfreq(n, t_u[1] - t_u[0])
    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return np.nan
    i = np.where(m)[0][np.argmax(F[m])]
    if 0 < i < len(F) - 1:                      # parabolic refinement
        a, b, c = np.log(F[i - 1] + 1e-300), np.log(F[i] + 1e-300), \
                  np.log(F[i + 1] + 1e-300)
        d = 0.5 * (a - c) / (a - 2 * b + c)
        return f[i] + d * (f[1] - f[0])
    return f[i]


def measure_frequencies(traj, M_geo, j, q):
    """Measured vs predicted (qpo.py) nu_phi, nu_r, nu_theta in Hz."""
    C = 2.99792458e5
    t, r, th, ph = traj['t'], traj['r'], traj['theta'], traj['phi']
    tu = np.linspace(t[0], t[-1], 2 ** 14)
    ru, thu = np.interp(tu, t, r), np.interp(tu, t, th)
    pred = qpo_at_radius(M_geo, j, q, traj['r0'])

    nu_phi = (ph[-1] - ph[0]) / (2 * np.pi * (t[-1] - t[0])) * C
    out = {'nu_phi': (nu_phi, pred['nu_phi'])}
    if traj['ecc'] > 0 and pred['nu_r'] > 0:
        f = _peak_freq(tu, ru, 0.5 * pred['nu_r'] / C, 1.5 * pred['nu_r'] / C)
        out['nu_r'] = (f * C, pred['nu_r'])
    if traj['inc_deg'] > 0:
        f = _peak_freq(tu, thu, 0.5 * pred['nu_theta'] / C,
                       1.5 * pred['nu_theta'] / C)
        out['nu_theta'] = (f * C, pred['nu_theta'])
    return out, tu, ru, thu


# =====================================================================
# Plots kept from v1 (unchanged apart from phase colouring)
# =====================================================================
def plot_circular_xy(traj, r_star, r_isco, out_path, title):
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    r, phi = traj['r'], traj['phi']
    x, y = r * np.cos(phi), r * np.sin(phi)

    R_plot = 1.2 * max(np.abs(x).max(), np.abs(y).max(), r_star)
    draw_accretion_disc(ax, r_isco, r_outer=R_plot * 0.98)

    draw_isco(ax, r_isco, label=r'$r_{\rm ISCO}$')
    _colored_line_2d(ax, x, y, cmap_name=CMAP_PHI, lw=2.6, alpha=0.95)
    mark_start_end_2d(ax, x, y)

    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')
    ax.set_title(title)
    ax.set_xlim(-R_plot, R_plot); ax.set_ylim(-R_plot, R_plot)
    ax.set_aspect('equal')

    draw_neutron_star_image(ax, r_star, fill_fraction=0.7)

    ax.legend(loc='upper right', fontsize=11)
    add_phi_colorbar(fig, ax, phi)

    plt.tight_layout(); plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_geodesic_xz(traj, r_star, r_isco, out_path, title):
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 7))
    r, phi, theta = traj['r'], traj['phi'], traj['theta']
    x = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)

    R_plot = 1.2 * max(np.abs(x).max(), np.abs(z).max(), r_star)
    draw_accretion_disc(ax, r_isco, r_outer=R_plot * 0.98)

    draw_isco(ax, r_isco, label=r'$r_{\rm ISCO}$')
    _colored_line_2d(ax, x, z, cmap_name=CMAP_PHI, lw=1.9, alpha=0.95,
                     values=phi)
    mark_start_end_2d(ax, x, z)

    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$z$ (km)')
    ax.set_title(title)
    ax.set_xlim(-R_plot, R_plot); ax.set_ylim(-R_plot, R_plot)
    ax.set_aspect('equal')

    draw_neutron_star_image(ax, r_star, fill_fraction=0.7)

    ax.legend(loc='upper right', fontsize=11, ncol=2)
    add_phi_colorbar(fig, ax, phi)

    plt.tight_layout(); plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_geodesic_3d(traj, r_star, r_isco, out_path, title):
    set_style()
    fig = plt.figure(figsize=(10.5, 9))
    ax = fig.add_subplot(111, projection='3d')

    r, phi, theta = traj['r'], traj['phi'], traj['theta']
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    R = 1.25 * max(np.abs(x).max(), np.abs(y).max(),
                    np.abs(z).max(), r_star)

    draw_accretion_disc_3d(ax, r_isco, r_outer=R * 0.95)

    draw_neutron_star_3d_image(ax, r_star)
    draw_isco_3d(ax, r_isco)
    _colored_line_3d(ax, x, y, z, cmap_name=CMAP_PHI, lw=1.8, alpha=0.95)
    mark_start_end_3d(ax, x, y, z)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)

    ax.set_xlabel(r'$x$ (km)', labelpad=8)
    ax.set_ylabel(r'$y$ (km)', labelpad=8)
    ax.set_zlabel(r'$z$ (km)', labelpad=8)
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_box_aspect([1, 1, 1])

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor('#BBBBBB')

    ax.grid(True, color='#CCCCCC', linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, ncol=2)
    ax.view_init(elev=22, azim=35)

    norm = Normalize(vmin=float(np.min(phi)), vmax=float(np.max(phi)))
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP_PHI))
    sm.set_array(phi)
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, fraction=0.03, shrink=0.7)
    cbar.set_label(r'azimuthal phase $\phi$ (rad)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_isco_zoom(traj, r_star, r_isco, out_path, title):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 8))
    r, phi = traj['r'], traj['phi']
    x, y = r * np.cos(phi), r * np.sin(phi)

    R = r_isco * 1.6
    draw_accretion_disc(ax, r_isco, r_outer=R * 0.98)

    draw_isco(ax, r_isco, lw=2.0, label=r'$r_{\rm ISCO}$')
    _colored_line_2d(ax, x, y, cmap_name=CMAP_PHI, lw=2.3, alpha=0.95)
    mark_start_end_2d(ax, x, y)

    ax.set_xlabel(r'$x$ (km)')
    ax.set_ylabel(r'$y$ (km)')
    ax.set_title(title)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect('equal')

    draw_neutron_star_image(ax, r_star, fill_fraction=0.7)

    ax.legend(loc='upper right', fontsize=11, ncol=2)
    add_phi_colorbar(fig, ax, phi)

    plt.tight_layout(); plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# New / corrected plots
# =====================================================================
def _xy_geodesic(traj):
    r, th, ph = traj['r'], traj['theta'], traj['phi']
    return r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph)


def plot_geodesic_xy(traj, r_star, r_isco, out_path, title):
    """x-y projection (r sin theta), line coloured by phi."""
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 9))
    x, y = _xy_geodesic(traj)
    phi = traj['phi']
    R_plot = 1.2 * max(np.abs(x).max(), np.abs(y).max(), r_star)
    draw_accretion_disc(ax, r_isco, r_outer=R_plot * 0.98)
    draw_isco(ax, r_isco, label=r'$r_{\rm ISCO}$')
    _colored_line_2d(ax, x, y, cmap_name=CMAP_PHI, lw=1.9, alpha=0.95,
                     values=phi)
    mark_start_end_2d(ax, x, y)
    ax.set_xlabel(r'$x$ (km)'); ax.set_ylabel(r'$y$ (km)')
    ax.set_title(title)
    ax.set_xlim(-R_plot, R_plot); ax.set_ylim(-R_plot, R_plot)
    ax.set_aspect('equal')
    draw_neutron_star_image(ax, r_star, fill_fraction=0.7)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    add_phi_colorbar(fig, ax, phi)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_time_slices_xy(traj, r_star, r_isco, out_path, title, n_slices=8):
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 9))
    x, y = _xy_geodesic(traj)
    t = traj['t']
    R_plot = 1.2 * max(np.abs(x).max(), np.abs(y).max(), r_star)
    draw_accretion_disc(ax, r_isco, r_outer=R_plot * 0.98)
    draw_isco(ax, r_isco, label=r'$r_{\rm ISCO}$')
    cmap = plt.get_cmap(CMAP_PHI)
    edges = np.linspace(0, len(x) - 1, n_slices + 1).astype(int)
    for i in range(n_slices):
        a, b = edges[i], edges[i + 1]
        ax.plot(x[a:b + 1], y[a:b + 1], color=cmap(i / max(n_slices - 1, 1)),
                linewidth=2.3, alpha=0.95, zorder=10)
    mark_start_end_2d(ax, x, y)
    ax.set_xlabel(r'$x$ (km)'); ax.set_ylabel(r'$y$ (km)')
    ax.set_title(title)
    ax.set_xlim(-R_plot, R_plot); ax.set_ylim(-R_plot, R_plot)
    ax.set_aspect('equal')
    draw_neutron_star_image(ax, r_star, fill_fraction=0.7)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    add_phi_colorbar(fig, ax, t / 2.99792458e5 * 1e3,
                     label='coordinate time $t$ (ms)')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_theta_oscillation(traj, out_path, title):
    """Latitude vs time.  theta < 90 deg  <=>  z = r cos(theta) > 0  (north)."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t_ms = traj['t'] / 2.99792458e5 * 1e3
    theta_deg = np.degrees(traj['theta'])
    ax.axhspan(0, 90, color='#87CEFA', alpha=0.25, label='north of equator')
    ax.axhspan(90, 180, color='#FFB6C1', alpha=0.25, label='south of equator')
    ax.axhline(90, color='k', lw=1.0, ls='--')
    ax.plot(t_ms, theta_deg, color='#004C99', lw=1.8)
    pad = max(1.0, 0.3 * np.ptp(theta_deg))
    ax.set_ylim(theta_deg.min() - pad, theta_deg.max() + pad)
    ax.set_xlabel(r'$t$ (ms)'); ax.set_ylabel(r'$\theta$ (deg)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_velocity_profile(traj, out_path, title):
    """Coordinate velocities d(r, theta, phi)/d lambda from the solution."""
    set_style()
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    lam = traj['lam']
    for ax, key, lab, c in zip(
            axs, ('ur', 'uth', 'uph'),
            (r'$dr/d\lambda$', r'$d\theta/d\lambda$ (km$^{-1}$)',
             r'$d\phi/d\lambda$ (km$^{-1}$)'),
            ('#CC0000', '#0066CC', '#228B22')):
        ax.plot(lam, traj[key], color=c, lw=1.5)
        ax.set_ylabel(lab)
    axs[-1].set_xlabel(r'affine parameter $\lambda$ (km)')
    axs[0].set_title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_energy_conservation(traj, out_path, title):
    """Actual drift of E, L_z and u.u along the integrated geodesic."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lam = traj['lam']
    eps = 1e-17
    ax.semilogy(lam, np.abs(traj['E_t'] / traj['E'] - 1) + eps,
                label=r'$|E/E_0-1|$', lw=1.8)
    ax.semilogy(lam, np.abs(traj['Lz_t'] / traj['Lz'] - 1) + eps,
                label=r'$|L_z/L_{z,0}-1|$', lw=1.8)
    ax.semilogy(lam, np.abs(traj['norm_t'] + 1) + eps,
                label=r'$|u\cdot u+1|$', lw=1.8, ls='--')
    ax.set_xlabel(r'affine parameter $\lambda$ (km)')
    ax.set_ylabel('relative drift')
    ax.set_title(title)
    ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_radial_motion(traj, r_star, r_isco, out_path, title):
    """r(t) and r(phi): radial epicycle and periastron precession."""
    set_style()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    t_ms = traj['t'] / 2.99792458e5 * 1e3
    r = traj['r']
    axs[0].plot(t_ms, r, color='#CC0000', lw=1.6)
    axs[1].plot(traj['phi'] / (2 * np.pi), r, color='#CC0000', lw=1.6)
    for ax in axs:
        ax.axhline(r_isco, color='#0066CC', ls=':', lw=1.6, label=r'$r_{\rm ISCO}$')
        ax.axhline(r_star, color='#22AA22', ls='--', lw=1.4, label='surface')
        ax.set_ylabel(r'$r$ (km)')
    axs[0].set_xlabel(r'$t$ (ms)')
    axs[1].set_xlabel(r'azimuth $\phi/2\pi$ (orbits)')
    axs[0].legend(fontsize=10)
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_effective_potential(metric_v, traj, r_star, r_isco, out_path, title):
    """Equatorial V(r;L) = E_+ at the orbit's L_z, with E and turning points."""
    set_style()
    from veff import veff
    fig, ax = plt.subplots(figsize=(9, 6))
    rr = np.linspace(max(metric_v.M * 2.2, r_star * 0.6), 1.2 * traj['r'].max(), 400)
    for lab, L, ls, c in ((r'$L=L_{\rm ISCO}$', None, '--', '#888888'),
                          (r'$L=L_z$ of orbit', traj['Lz'], '-', '#004C99')):
        V = np.array([veff(metric_v, r, L=L) for r in rr])
        ax.plot(rr, V, ls, color=c, lw=2, label=lab)
    ax.axhline(traj['E'], color='#CC0000', lw=1.5, label=r'$E$ of orbit')
    ax.axvline(r_isco, color='#0066CC', ls=':', lw=1.4, label=r'$r_{\rm ISCO}$')
    ax.axvline(r_star, color='#22AA22', ls='--', lw=1.4, label='surface')
    ax.axvspan(traj['r'].min(), traj['r'].max(), color='#FFD27F', alpha=0.3,
               label='radial range of orbit')
    ax.set_xlabel(r'$r$ (km)'); ax.set_ylabel(r'$V_{\rm eff}$ (energy per unit mass)')
    ax.set_title(title + ('\n(equatorial potential; approximate for i > 0)'
                          if traj['inc_deg'] > 0 else ''))
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_circular_EL(metric_v, r_star, r_isco, out_path, title):
    """E(r), L(r) of circular orbits; ISCO = minimum of both."""
    set_style()
    from veff import circular_EL
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    rr = np.linspace(max(metric_v.M * 2.3, r_star), 6 * r_isco, 300)
    EL = np.array([circular_EL(metric_v, r) for r in rr])
    for ax, i, lab in ((axs[0], 0, r'$E(r)$'), (axs[1], 1, r'$L(r)$')):
        ax.plot(rr, EL[:, i], color='#004C99', lw=2)
        ax.axvline(r_isco, color='#0066CC', ls=':', lw=1.6, label=r'$r_{\rm ISCO}$ (qpo.py)')
        ax.axvline(r_star, color='#22AA22', ls='--', lw=1.4, label='surface')
        ok = np.isfinite(EL[:, i])
        k = np.argmin(EL[ok, i]); ax.plot(rr[ok][k], EL[ok, i][k], 'ro',
                                          label='min of curve')
        ax.set_xlabel(r'$r$ (km)'); ax.set_ylabel(lab)
    axs[0].legend(fontsize=10)
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_poincare(traj, out_path, title):
    """(r, dr/dt) at equatorial crossings with d(theta)/d(lambda) > 0."""
    set_style()
    th, r, ur, ut, uth = (traj['theta'], traj['r'], traj['ur'],
                          traj['ut'], traj['uth'])
    s = th - np.pi / 2
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(idx) < 2:
        print("  Poincare section: <2 crossings, skipped"); return
    w = -s[idx] / (s[idx + 1] - s[idx])
    rc = r[idx] + w * (r[idx + 1] - r[idx])
    vc = (ur / ut)[idx] + w * ((ur / ut)[idx + 1] - (ur / ut)[idx])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(rc, vc, c=np.arange(len(rc)), cmap=CMAP_PHI, s=40)
    ax.set_xlabel(r'$r$ at equator crossing (km)')
    ax.set_ylabel(r'$dr/dt$')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='crossing number')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def plot_frequency_check(freqs, tu, ru, thu, traj, M_geo, j, q, out_path, title):
    """FFT spectra of r(t), theta(t) with qpo.py predictions overplotted."""
    set_style()
    C = 2.99792458e5
    pred = qpo_at_radius(M_geo, j, q, traj['r0'])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sig, key, lab in ((axs[0], ru, 'nu_r', r'$r(t)$'),
                              (axs[1], thu, 'nu_theta', r'$\theta(t)$')):
        s = (sig - sig.mean()) * np.hanning(len(sig))
        F = np.abs(np.fft.rfft(s, len(s) * 8))
        f = np.fft.rfftfreq(len(s) * 8, tu[1] - tu[0]) * C
        ax.semilogy(f, F / F.max() + 1e-12, color='#004C99', lw=1.3)
        ax.axvline(pred[key], color='#CC0000', ls='--', lw=1.6,
                   label=f"qpo.py {key} = {pred[key]:.1f} Hz")
        ax.set_xlim(0, 2.5 * max(pred['nu_phi'], 1))
        ax.set_ylim(1e-6, 2)
        ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('FFT of ' + lab)
        ax.legend(fontsize=10)
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description='Particle orbits in Hartle-Thorne spacetime.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--eos',  default=CONFIG['eos'])
    p.add_argument('--mass', type=float, default=CONFIG['mass'])
    p.add_argument('--spin', type=float, default=CONFIG['spin'])
    p.add_argument('--r-circular', type=float, default=CONFIG['r_circular'])
    p.add_argument('--n-orbits-circ', type=int, default=CONFIG['n_orbits_circ'])
    p.add_argument('--r-start', type=float, default=CONFIG['r_start'])
    p.add_argument('--theta-start', type=float, default=CONFIG['theta_start_deg'],
                   help='orbit reaches colatitude ~theta-start (inclination = 90 - this)')
    p.add_argument('--n-orbits-geo', type=int, default=CONFIG['n_orbits_geo'])
    p.add_argument('--ecc', type=float, default=CONFIG['ecc'],
                   help='initial radial speed / orbital speed (0 = no radial epicycle)')
    p.add_argument('--star-image', default=STAR_IMAGE,
                   help='Path to neutron-star image file')
    p.add_argument('--outdir', default=CONFIG['outdir'])
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global STAR_IMAGE
    STAR_IMAGE = args.star_image

    print("=" * 72)
    print("  Particle orbits in Hartle-Thorne spacetime")
    print("=" * 72)
    print(f"  EOS       = {args.eos}")
    print(f"  M_target  = {args.mass:.3f} Msun")
    print(f"  spin      = {args.spin:.1f} Hz")

    eos = EOS(args.eos)
    from run_qpo import solve_rho_c_for_mass          # handles M_max safely
    # v3: --mass is the total (rotating) mass; stable branch only
    rho_c = solve_rho_c_for_mass(eos, args.mass, verbose=False,
                                 spin_hz=args.spin)

    model = HTModel(eos, rho_c, args.spin)
    r_star = model.r_surface          # equatorial surface (HT coordinate)
    r_isco, edge = ht_common.inner_edge(model)

    print(f"  rho_c     = {rho_c:.4e} g/cm^3")
    print(f"  M         = {model.M_solar:.4f} Msun (M_TOV = "
          f"{model.M0_solar:.4f}, dM/M = "
          f"{model.dM_geo / model.M0_geo:+.2e})")
    print(f"  R_eq      = {r_star:.3f} km  (R_TOV = {model.R_km:.3f})")
    print(f"  j         = {model.j:+.5f}")
    print(f"  q         = {model.q:+.5f}   (Geroch-Hansen sign)")
    print(f"  q/j^2     = {model.q / model.j**2:+.4f}")
    print(f"  r_in      = {r_isco:.3f} km  ({edge})")
    print("=" * 72)

    from veff import HartleThorneMetric
    metric_full = HTMetricFull(model.exterior)
    metric_v = HartleThorneMetric(model.M_geo, model.j, model.q)

    # -------- [1/3] Circular orbit --------
    r_circ = args.r_circular if args.r_circular is not None else 1.5 * r_isco
    print(f"\n[1/3] Circular equatorial orbit:  r = {r_circ:.2f} km, "
          f"{args.n_orbits_circ} loops")
    traj_circ = build_circular_orbit(metric_full, r_circ, args.n_orbits_circ)
    if traj_circ is not None:
        T = traj_circ['T']
        print(f"  Omega  = {traj_circ['Omega']:.6e} km^-1")
        print(f"  Period = {T:.3f} km  =  {T/2.99792458e5*1e3:.4f} ms")
        v = verify_circular(metric_full, r_circ, args.n_orbits_circ)
        if v:
            print(f"  geodesic check: max|dr|/r = {v['dr_max']:.2e}, "
                  f"dE/E = {v['dE']:.2e}, |u.u+1| = {v['dnorm']:.2e}")
        plot_circular_xy(traj_circ, r_star, r_isco,
            os.path.join(args.outdir, 'circular_orbit_xy.png'),
            title=f'Circular orbit: r = {r_circ:.2f} km, '
                  f'{args.n_orbits_circ} loops  '
                  f'(M={model.M_solar:.2f} M$_\\odot$, spin={args.spin:.0f} Hz)')
        plot_isco_zoom(traj_circ, r_star, r_isco,
            os.path.join(args.outdir, 'circular_orbit_isco_zoom.png'),
            title=f'ISCO zoom: r = {r_circ:.2f} km  (r_ISCO = {r_isco:.2f} km)')
    else:
        print("  Circular orbit failed.")

    # -------- [2/3] Circular-orbit E(r), L(r) --------
    print("\n[2/3] Circular-orbit energetics")
    plot_circular_EL(metric_v, r_star, r_isco,
        os.path.join(args.outdir, 'circular_E_L_vs_r.png'),
        title='Circular-orbit E(r), L(r): ISCO = minimum')

    # -------- [3/3] Inclined (eccentric) geodesic --------
    r_geo = args.r_start if args.r_start is not None else 1.5 * r_isco
    print(f"\n[3/3] Inclined geodesic:  r = {r_geo:.2f} km, "
          f"i = {90 - args.theta_start:.1f} deg, ecc = {args.ecc:.2f}, "
          f"{args.n_orbits_geo} orbits")
    traj_geo = integrate_geodesic(metric_full, r_geo, args.theta_start,
                                  args.n_orbits_geo, ecc=args.ecc,
                                  r_stop=max(r_star, metric_full.r_min))
    if traj_geo is None:
        print("  Geodesic integration failed.")
    else:
        if traj_geo['terminated']:
            print("  [warning] orbit hit the stellar surface / horizon "
                  "region; trajectory truncated.")
        th = np.degrees(traj_geo['theta'])
        print(f"  E   = {traj_geo['E']:.6f}")
        print(f"  Lz  = {traj_geo['Lz']:.6f}")
        print(f"  theta range: {th.min():.2f} to {th.max():.2f} deg")
        print(f"  r range    : {traj_geo['r'].min():.2f} to "
              f"{traj_geo['r'].max():.2f} km")
        print(f"  drift: dE/E = {np.ptp(traj_geo['E_t'])/abs(traj_geo['E']):.2e}, "
              f"dLz/Lz = {np.ptp(traj_geo['Lz_t'])/abs(traj_geo['Lz']):.2e}, "
              f"max|u.u+1| = {np.max(np.abs(traj_geo['norm_t']+1)):.2e}")

        freqs, tu, ru, thu = measure_frequencies(
            traj_geo, model.M_geo, model.j, model.q)
        print("  Frequency check (trajectory vs qpo.py at r0; ~O(ecc^2, i^2) agreement):")
        for k, (meas, pred) in freqs.items():
            print(f"    {k:9s} measured {meas:8.2f} Hz   qpo.py {pred:8.2f} Hz   "
                  f"diff {100*(meas-pred)/pred:+.2f} %")

        o = args.outdir
        ttl_i = f'r={r_geo:.1f} km, i={90-args.theta_start:.0f}°, ecc={args.ecc:.2f}'
        plot_geodesic_xy(traj_geo, r_star, r_isco, os.path.join(o, 'geodesic_xy.png'),
            title=f'Geodesic (x-y projection) — {ttl_i}, {args.n_orbits_geo} orbits')
        plot_geodesic_xz(traj_geo, r_star, r_isco, os.path.join(o, 'geodesic_xz.png'),
            title=f'Geodesic (x-z) — {ttl_i}')
        plot_geodesic_3d(traj_geo, r_star, r_isco, os.path.join(o, 'geodesic_3d.png'),
            title=f'Geodesic (3D) — {ttl_i}')
        plot_time_slices_xy(traj_geo, r_star, r_isco,
            os.path.join(o, 'geodesic_time_slices.png'),
            title='Time-sliced trajectory (rainbow = time flow)')
        plot_theta_oscillation(traj_geo, os.path.join(o, 'geodesic_theta_vs_t.png'),
            title='Nodal precession: colatitude vs time')
        plot_velocity_profile(traj_geo, os.path.join(o, 'geodesic_velocity.png'),
            title='Coordinate velocities vs affine parameter')
        plot_energy_conservation(traj_geo, os.path.join(o, 'geodesic_conservation.png'),
            title='Drift of E, L$_z$ and u·u along the geodesic')
        plot_radial_motion(traj_geo, r_star, r_isco, os.path.join(o, 'geodesic_radial.png'),
            title='Radial epicycle and periastron precession')
        plot_effective_potential(metric_v, traj_geo, r_star, r_isco,
            os.path.join(o, 'geodesic_veff.png'), title='Effective potential')
        plot_poincare(traj_geo, os.path.join(o, 'geodesic_poincare.png'),
            title='Poincaré section: equatorial crossings')
        plot_frequency_check(freqs, tu, ru, thu, traj_geo, model.M_geo,
            model.j, model.q, os.path.join(o, 'geodesic_frequency_check.png'),
            title='FFT of the trajectory vs qpo.py epicyclic frequencies')

    print(f"\n  Plots in: {args.outdir}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
