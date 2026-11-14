"""
plots.py

Publication-quality individual plots for the QPO pipeline.

Files produced (in out_dir):

    QPO frequency plots:
        qpo_vs_radius.png           — all frequencies vs r
        qpo_phi_vs_radius.png       — nu_phi alone
        qpo_r_vs_radius.png         — nu_r alone
        qpo_peri_vs_radius.png      — nu_peri alone
        qpo_nod_vs_radius.png       — nu_nod alone (nodal precession)
        qpo_deltanu_vs_nu2.png      — Delta nu vs nu_2
        qpo_nu2_vs_nu1.png          — nu_2 vs nu_1
        qpo_hbo_vs_nu2.png          — 2 nu_nod vs nu_2

    Veff-family plots (all starting from the stellar surface):
        veff.png                    — Effective potential Veff(r)
        veff_first_deriv.png        — d Veff / dr
        veff_second_deriv.png       — d^2 Veff / dr^2
        circular_omega.png          — Omega(r)

    EOS and Mass-Radius:
        mass_radius.png             — M(R)
        eos_curve.png               — P(rho)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator


# =====================================================================
# Global style
# =====================================================================

def set_style():
    plt.rcParams.update({
        'figure.figsize':      (7.0, 5.5),
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'font.family':         'serif',
        'font.serif':          ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset':    'dejavuserif',
        'font.size':           11,
        'axes.labelsize':      12,
        'axes.titlesize':      12,
        'axes.linewidth':      1.0,
        'axes.grid':           True,
        'grid.alpha':          0.3,
        'grid.linestyle':      '--',
        'grid.linewidth':      0.6,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.major.size':    5,
        'ytick.major.size':    5,
        'xtick.minor.size':    2.5,
        'ytick.minor.size':    2.5,
        'legend.fontsize':     10,
        'legend.frameon':      True,
        'legend.framealpha':   0.9,
        'legend.edgecolor':    '0.7',
        'legend.fancybox':     False,
        'lines.linewidth':     1.8,
        'lines.solid_capstyle': 'round',
    })


def _add_minor_ticks(ax):
    """
    Add minor ticks on both axes, using the locator appropriate to
    each axis's scale. AutoMinorLocator only supports linear axes; a
    log-scaled axis needs LogLocator (with subs= for the in-between
    decade ticks), otherwise matplotlib emits a UserWarning and
    silently skips adding the minor ticks.
    """
    if ax.get_xscale() == 'log':
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                      numticks=12))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if ax.get_yscale() == 'log':
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                      numticks=12))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _extract_arrays(table):
    return {
        'r':    np.array([row['r_km']     for row in table]),
        'x':    np.array([row['x']        for row in table]),
        'phi':  np.array([row['nu_phi']   for row in table]),
        'r_r':  np.array([row['nu_r']     for row in table]),
        'th':   np.array([row['nu_theta'] for row in table]),
        'peri': np.array([row['nu_peri']  for row in table]),
        'lt':   np.array([row.get('nu_nod', row.get('nu_LT'))
                          for row in table]),
    }


# =====================================================================
# QPO frequency plots
# =====================================================================

def plot_all_frequencies_vs_radius(table, title, out_path):
    set_style()
    d = _extract_arrays(table)

    fig, ax = plt.subplots()
    ax.plot(d['r'], d['phi'],  color='#1f77b4', label=r'$\nu_\phi$ (orbital)')
    ax.plot(d['r'], d['peri'], color='#2ca02c', label=r'$\nu_{\rm peri}$ (periastron)')
    ax.plot(d['r'], d['r_r'],  color='#d62728', label=r'$\nu_r$ (radial)')
    ax.plot(d['r'], d['lt'],   color='#000000', label=r'$\nu_{\rm nod}$ (nodal precession)')

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'Frequency (Hz)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlim(d['r'].min() * 0.98, d['r'].max() * 1.02)
    ax.legend(loc='best', ncol=2)
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_single_frequency_vs_radius(table, key, ylabel, title, out_path,
                                     color='#1f77b4'):
    set_style()
    d = _extract_arrays(table)

    fig, ax = plt.subplots()
    ax.plot(d['r'], d[key], color=color)

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(d['r'].min() * 0.98, d['r'].max() * 1.02)
    ax.set_ylim(bottom=0)
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_deltanu_vs_nu2(table, title, out_path):
    set_style()
    d = _extract_arrays(table)

    fig, ax = plt.subplots()
    ax.plot(d['phi'], d['r_r'], color='#000000')

    ax.set_xlabel(r'$\nu_2 = \nu_\phi$ (Hz)')
    ax.set_ylabel(r'$\Delta\nu = \nu_r$ (Hz)')
    ax.set_title(title)
    ax.set_xlim(d['phi'].min() * 0.98, d['phi'].max() * 1.02)
    ax.set_ylim(bottom=0)
    _add_minor_ticks(ax)

    isco_mask = d['r_r'] < 1.0
    if isco_mask.any():
        idx = np.argmin(d['r_r'])
        ax.plot(d['phi'][idx], d['r_r'][idx], 'o',
                color='#d62728', markersize=7, zorder=5, label='ISCO')
        ax.legend(loc='best')

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_nu2_vs_nu1(table, title, out_path):
    set_style()
    d = _extract_arrays(table)

    fig, ax = plt.subplots()
    ax.plot(d['phi'], d['peri'], color='#9467bd')

    ax.set_xlabel(r'$\nu_2 = \nu_\phi$ (Hz)')
    ax.set_ylabel(r'$\nu_1 = \nu_{\rm peri}$ (Hz)')
    ax.set_title(title)
    ax.set_xlim(d['phi'].min() * 0.98, d['phi'].max() * 1.02)
    ax.set_ylim(bottom=0)
    _add_minor_ticks(ax)

    idx = np.argmin(d['r_r'])
    ax.plot(d['phi'][idx], d['peri'][idx], 'o',
            color='#d62728', markersize=7, zorder=5, label='ISCO')
    ax.legend(loc='best')

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_hbo_vs_nu2(table, title, out_path):
    set_style()
    d = _extract_arrays(table)

    fig, ax = plt.subplots()
    ax.plot(d['phi'], 2.0 * d['lt'], color='#ff7f0e')

    ax.set_xlabel(r'$\nu_2 = \nu_\phi$ (Hz)')
    ax.set_ylabel(r'$\nu_{\rm HBO} = 2\,\nu_{\rm nod}$ (Hz)')
    ax.set_title(title)
    ax.set_xlim(d['phi'].min() * 0.98, d['phi'].max() * 1.02)
    ax.set_ylim(bottom=0)
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# Veff family plots — all start at the stellar surface
# =====================================================================

def _mark_surface_and_isco(ax, r_vals, y_vals, r_star, r_isco):
    """
    Draw the standard markers used on all Veff-family plots:
        * blue filled square at r = r_star (start of the curve)
        * blue vertical line at r = r_star
        * red dashed vertical line at r = r_isco
        * red circle at r = r_isco on the curve
    """
    # --- Surface marker ------------------------------------------------
    if r_star is not None and r_star > 0:
        ax.axvline(r_star, color='#1f77b4', linestyle='-',
                   linewidth=1.2, alpha=0.75)
        if np.isfinite(y_vals[0]):
            ax.plot(r_vals[0], y_vals[0], 's',
                    color='#1f77b4', markersize=9,
                    markeredgecolor='white', markeredgewidth=1.0,
                    zorder=6,
                    label=f'$R_{{\\rm surf}}$ = {r_star:.2f} km')

    # --- ISCO marker ---------------------------------------------------
    if r_isco is not None and r_isco > 0:
        ax.axvline(r_isco, color='#d62728', linestyle='--',
                   linewidth=1.4)
        i_isco = int(np.argmin(np.abs(r_vals - r_isco)))
        if np.isfinite(y_vals[i_isco]):
            ax.plot(r_vals[i_isco], y_vals[i_isco], 'o',
                    color='#d62728', markersize=9,
                    markeredgecolor='white', markeredgewidth=1.0,
                    zorder=5,
                    label=f'$r_{{\\rm ISCO}}$ = {r_isco:.2f} km')


def plot_veff(r_vals, veff_vals, r_isco=None, r_star=None,
              out_path=None, title='Effective potential'):
    """
    Plot Veff(r) from the stellar surface outward.
    """
    set_style()
    fig, ax = plt.subplots()

    ax.plot(r_vals, veff_vals, color='#000000',
            label=r'$V_{\rm eff}(r)$')

    _mark_surface_and_isco(ax, r_vals, veff_vals, r_star, r_isco)

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$V_{\rm eff}(r)$')
    ax.set_title(title)
    ax.set_xlim(r_vals.min() * 0.98, r_vals.max() * 1.02)
    ax.legend(loc='best')
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_veff_first_derivative(r_vals, d1_vals, r_isco=None, r_star=None,
                               out_path=None,
                               title=r'$dV_{\rm eff}/dr$'):
    """
    First derivative of Veff.  It vanishes at the minimum of Veff,
    the location of the innermost stable circular orbit.
    """
    set_style()
    fig, ax = plt.subplots()

    ax.plot(r_vals, d1_vals, color='#000000',
            label=r'$dV_{\rm eff}/dr$')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)

    _mark_surface_and_isco(ax, r_vals, d1_vals, r_star, r_isco)

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$dV_{\rm eff}/dr$')
    ax.set_title(title)
    ax.set_xlim(r_vals.min() * 0.98, r_vals.max() * 1.02)
    ax.legend(loc='best')
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_veff_second_derivative(r_vals, d2_vals, r_isco=None, r_star=None,
                                out_path=None,
                                title=r'$d^2V_{\rm eff}/dr^2$'):
    """
    Second derivative of Veff.  It crosses zero at the ISCO.
    """
    set_style()
    fig, ax = plt.subplots()

    ax.plot(r_vals, d2_vals, color='#000000',
            label=r'$d^2V_{\rm eff}/dr^2$')
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)

    _mark_surface_and_isco(ax, r_vals, d2_vals, r_star, r_isco)

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$d^2V_{\rm eff}/dr^2$')
    ax.set_title(title)
    ax.set_xlim(r_vals.min() * 0.98, r_vals.max() * 1.02)
    ax.legend(loc='best')
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_circular_omega(r_vals, omega_vals, r_isco=None, r_star=None,
                        out_path=None,
                        title=r'Circular-orbit angular velocity'):
    """
    Angular velocity of circular equatorial orbits, Omega(r).
    """
    set_style()
    fig, ax = plt.subplots()

    ax.plot(r_vals, omega_vals, color='#000000',
            label=r'$\Omega(r)$')

    _mark_surface_and_isco(ax, r_vals, omega_vals, r_star, r_isco)

    ax.set_xlabel(r'$r$ (km)')
    ax.set_ylabel(r'$\Omega$ (rad/s)')
    ax.set_title(title)
    ax.set_xlim(r_vals.min() * 0.98, r_vals.max() * 1.02)
    ax.legend(loc='best')
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
# =====================================================================
# EOS / Mass-Radius
# =====================================================================

def plot_mass_radius(M, R, out_path):
    set_style()
    fig, ax = plt.subplots()
    ax.plot(R, M, color='#1f77b4')
    ax.set_xlabel(r'$R$ (km)')
    ax.set_ylabel(r'$M$ ($M_\odot$)')
    ax.set_title('Mass–Radius relation')
    ax.set_ylim(bottom=0)
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_eos(rho, P, out_path):
    set_style()
    fig, ax = plt.subplots()
    ax.loglog(rho, P, color='#000000')
    ax.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylabel(r'$P$ (dyn cm$^{-2}$)')
    ax.set_title('Equation of State')
    ax.grid(which='both', alpha=0.3, linestyle='--', linewidth=0.6)
    _add_minor_ticks(ax)

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# =====================================================================
# Master wrapper
# =====================================================================

def make_all_plots(table, eos, M_arr, R_arr, source_name, out_dir,
                   r_veff=None, veff_vals=None,
                   d1veff_vals=None, d2veff_vals=None,
                   omega_vals=None,
                   r_isco=None, r_star=None):
    """
    Generate all individual plots and save them in out_dir.

    All Veff-family plots start at r_star (the stellar surface).
    """
    os.makedirs(out_dir, exist_ok=True)

    print("\nGenerating individual plots ...")

    # ---- QPO frequency plots ----------------------------------------
    plot_all_frequencies_vs_radius(
        table, f'{source_name}: QPO frequencies',
        os.path.join(out_dir, 'qpo_vs_radius.png'))

    plot_single_frequency_vs_radius(
        table, 'phi', r'$\nu_\phi$ (Hz)',
        f'{source_name}: orbital frequency',
        os.path.join(out_dir, 'qpo_phi_vs_radius.png'),
        color='#1f77b4')

    plot_single_frequency_vs_radius(
        table, 'r_r', r'$\nu_r$ (Hz)',
        f'{source_name}: radial epicyclic frequency',
        os.path.join(out_dir, 'qpo_r_vs_radius.png'),
        color='#d62728')

    plot_single_frequency_vs_radius(
        table, 'peri', r'$\nu_{\rm peri}$ (Hz)',
        f'{source_name}: periastron precession',
        os.path.join(out_dir, 'qpo_peri_vs_radius.png'),
        color='#2ca02c')

    plot_single_frequency_vs_radius(
        table, 'lt', r'$\nu_{\rm nod}$ (Hz)',
        f'{source_name}: nodal precession (LT + quadrupole)',
        os.path.join(out_dir, 'qpo_nod_vs_radius.png'),
        color='#000000')

    plot_deltanu_vs_nu2(
        table, f'{source_name}: RPM fingerprint',
        os.path.join(out_dir, 'qpo_deltanu_vs_nu2.png'))

    plot_nu2_vs_nu1(
        table, f'{source_name}: QPO pair relation',
        os.path.join(out_dir, 'qpo_nu2_vs_nu1.png'))

    plot_hbo_vs_nu2(
        table, f'{source_name}: HBO correlation',
        os.path.join(out_dir, 'qpo_hbo_vs_nu2.png'))

    # ---- Veff family --------------------------------------------------
    if r_veff is not None and veff_vals is not None:
        plot_veff(
            r_veff, veff_vals,
            r_isco=r_isco, r_star=r_star,
            out_path=os.path.join(out_dir, 'veff.png'),
            title=f'{source_name}: Effective potential')

    if r_veff is not None and d1veff_vals is not None:
        plot_veff_first_derivative(
            r_veff, d1veff_vals,
            r_isco=r_isco, r_star=r_star,
            out_path=os.path.join(out_dir, 'veff_first_deriv.png'),
            title=f'{source_name}: $dV_{{eff}}/dr$')

    if r_veff is not None and d2veff_vals is not None:
        plot_veff_second_derivative(
            r_veff, d2veff_vals,
            r_isco=r_isco, r_star=r_star,
            out_path=os.path.join(out_dir, 'veff_second_deriv.png'),
            title=f'{source_name}: $d^2V_{{eff}}/dr^2$')

    if r_veff is not None and omega_vals is not None:
        plot_circular_omega(
            r_veff, omega_vals,
            r_isco=r_isco, r_star=r_star,
            out_path=os.path.join(out_dir, 'circular_omega.png'),
            title=f'{source_name}: Circular-orbit $\\Omega(r)$')

    # ---- EOS and mass-radius -----------------------------------------
    plot_mass_radius(
        M_arr, R_arr,
        os.path.join(out_dir, 'mass_radius.png'))

    plot_eos(
        eos.rho, eos.P,
        os.path.join(out_dir, 'eos_curve.png'))