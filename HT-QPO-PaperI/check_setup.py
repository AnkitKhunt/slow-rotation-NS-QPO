#!/usr/bin/env python3
"""
check_setup.py -- check that this computer can run the HT QPO code.

    python check_setup.py

It checks the Python and package versions, the shipped data files, the
fonts and math labels used in the figures, and computes the QPO
frequencies of one star.  It takes a few seconds and writes nothing except
a test figure in a temporary folder.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

ok = True


def report(good, msg, fix=''):
    global ok
    print(('  [ok]   ' if good else '  [FAIL] ') + msg)
    if not good:
        ok = False
        if fix:
            print('         fix: ' + fix)


def version(mod):
    v = getattr(mod, '__version__', '0')
    out = []
    for p in v.split('.')[:3]:
        n = ''.join(ch for ch in p if ch.isdigit())
        out.append(int(n) if n else 0)
    return tuple(out), v


print('Python and packages')
report(sys.version_info >= (3, 10),
       f'Python {sys.version.split()[0]} (need >= 3.10)',
       'install Python 3.10 or newer (python.org, conda or uv)')

need = {'numpy': (1, 24), 'scipy': (1, 10), 'matplotlib': (3, 8),
        'pandas': (1, 5)}
for name, vmin in need.items():
    try:
        mod = __import__(name)
        t, v = version(mod)
        report(t >= vmin, f'{name} {v} (need >= {".".join(map(str, vmin))})',
               f'pip install -U "{name}>={".".join(map(str, vmin))}"')
    except ImportError:
        report(False, f'{name} not installed', 'pip install -r requirements.txt')

print('Optional (only for step 5, nested sampling)')
for name in ('bilby', 'dynesty', 'corner'):
    try:
        mod = __import__(name)
        t, v = version(mod)
        if name == 'dynesty' and t >= (3, 2):
            print(f'  [warn] dynesty {v}: bilby 2.8 fails with dynesty >= 3.2;'
                  ' run  pip install "dynesty<3.2"')
        else:
            print(f'  [ok]   {name} {v}')
    except Exception as e:      # e.g. bilby with a mismatched dynesty
        print(f'  [--]   {name} not usable ({type(e).__name__}); '
              'steps 1-4 and 6-7 do not need it')

print('Data files')
for p in ('eos/IOPB', 'eos/FSUGarnet', 'eos/G3', 'eos/eosAU',
          'qpo_fit_cache/freq_table.npz', 'qpo_fit_cache/star_IOPB.npz',
          'data', 'configs/atoll.json', 'results/orbit.npz'):
    report(os.path.exists(p), p,
           'unzip the whole package and run this script from its folder')

print('Figures')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import plot_style as ps
    f = font_manager.findfont('cmb10', fallback_to_default=False)
    report(os.path.exists(f), 'Computer Modern font cmb10 (ships with Matplotlib)')
    ps.use()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([1, 2], [1, 2])
    ax.set_xlabel(ps.lab(ps.S['nus'], 'Hz'))
    ax.set_ylabel(ps.lab(r'\bar{\boldsymbol{Q}}=-\boldsymbol{q}/\boldsymbol{j}^{2}'))
    ps.finish(ax)
    ps.label_panels(fig, [ax])
    out = os.path.join(tempfile.gettempdir(), 'ht_qpo_check.pdf')
    fig.savefig(out)
    plt.close(fig)
    report(True, f'bold math labels and panel labels (test figure: {out})')
except Exception as e:
    report(False, f'figure test failed: {type(e).__name__}: {e}',
           'pip install -U "matplotlib>=3.8"')

print('Physics')
try:
    from qpo_fit import StarTable, M_SUN_KM
    from qpo import qpo_at_radius
    st = StarTable('eos/IOPB').star(1.922, 363.0)
    M = 1.922 * M_SUN_KM
    fr = qpo_at_radius(M, st['j'], st['q'], 17.394)
    good = (abs(st['j'] - 0.1776) < 2e-3 and abs(fr['nu_phi'] - 1098.4) < 1.0
            and abs(fr['nu_nod'] - 19.31) < 0.05)
    report(good, f"IOPB-I, 1.922 Msun, 363 Hz: j = {st['j']:.4f}, "
                 f"nu_phi = {fr['nu_phi']:.1f} Hz, nu_nod = {fr['nu_nod']:.2f} Hz "
                 '(paper: 0.1776, 1098.4, 19.31)')
except Exception as e:
    report(False, f'frequency test failed: {type(e).__name__}: {e}')

print()
if ok:
    print('All checks passed.  Next:  python run_all.py 2 3 6 7')
else:
    print('Some checks failed; see the fixes above.')
sys.exit(0 if ok else 1)
