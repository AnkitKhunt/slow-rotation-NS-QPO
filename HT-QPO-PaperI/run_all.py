#!/usr/bin/env python3
"""
run_all.py -- reproduce every number, figure and table of Paper I
(slow rotation, Hartle-Thorne).

    python3 run_all.py            # all steps
    python3 run_all.py 2 3 6      # selected steps
    python3 run_all.py list       # show the steps

Steps
  1  tables: frequency table + star table per EOS   (~10 min; cached in
     qpo_fit_cache/, already shipped -- skip unless you add an EOS)
  2  physics figures: stars, frequencies, nodal precession (+ tab_eos.tex)
  3  orbit: integrate the geodesic, hot-spot light curve and spectrum
     (~1 min), then fig_orbit.pdf and fig_psd.pdf
  4  RPM fits with each EOS (grid quadrature; ~15-30 min)
  5  fits without an EOS (Bilby/dynesty; ~15-30 min each)
  6  data figures, corner plot, tab_fits.tex
  7  checks of numbers quoted in the text (paper_checks.py, ~2 s)

Figures and tables go to paper_out/.  Set NPOOL for step 1 (default 2).
"""
import os
import subprocess
import sys

PY = sys.executable
E = ['eos/eosAU', 'eos/G3', 'eos/FSUGarnet', 'eos/IOPB']
NP = os.environ.get('NPOOL', '2')
G = [PY, 'qpo_grid_fit.py']
B = [PY, 'qpo_fit.py', 'fit']

STEPS = {
    1: ('frequency and star tables', [
        [PY, 'qpo_fit.py', 'build-tables', '--eos', *E, '--npool', NP]]),
    2: ('physics figures', [
        [PY, 'figures.py', '--only', 'stars,freqs,nodal']]),
    3: ('orbit and hot-spot spectrum', [
        [PY, 'orbit_qpo.py', '--eos', 'eos/IOPB', '--mass', '1.922',
         '--spin', '363', '--nu-target', '1098.4', '--ecc', '0.01',
         '--inc', '8', '--iobs', '60', '--T', '0.5',
         '--out', 'results/orbit.npz'],
        [PY, 'figures.py', '--only', 'orbit,psd']]),
    4: ('RPM fits with each EOS (grid)', [
        # 4U 1636-53: kHz QPOs, one scatter
        G + ['eos', '--config', 'configs/atoll.json', '--eos', *E,
             '--only', '4U1636-53', '--M-grid', '1.6', '2.25', '--nM', '261'],
        # 4U 1728-34: kHz + LF QPOs, separate kHz and LF scatters
        G + ['eos', '--config', 'configs/atoll.json', '--eos', *E,
             '--only', '4U1728-34', '--scatter', 'frac2', '--nE', '24',
             '--label', 'frac2', '--M-grid', '1.75', '2.05', '--nM', '121'],
        # 4U 1728-34: kHz QPOs only
        G + ['eos', '--config', 'configs/kHz_only.json', '--eos', *E,
             '--only', '4U1728-34', '--label', 'kHz', '--M-grid', '1.75',
             '2.05', '--nM', '121'],
        # Z sources: kHz QPOs only (spin free)
        G + ['eos', '--config', 'configs/kHz_only.json', '--eos', *E,
             '--only', 'GX5-1', 'GX340+0', '--label', 'kHz', '--K', '300',
             '--M-grid', '1.75', '2.25', '--nM', '51', '--nS', '23'],
        # Z sources: kHz QPOs + HBO, two scatters, spin free
        G + ['eos', '--config', 'configs/zsources.json', '--eos', *E,
             '--scatter', 'frac2', '--nE', '16', '--label', 'frac2',
             '--K', '300', '--M-grid', '1.9', '2.25', '--nM', '71',
             '--spin-grid', '600', '750', '--nS', '31'],
        # 4U 1636-53 without an EOS (text of Sec. VI B)
        G + ['free', '--config', 'configs/atoll.json', '--only', '4U1636-53',
             '--nE', '24', '--K', '300', '--M-min', '1.4', '--M-max-free',
             '2.6', '--nM', '49', '--nj', '31', '--nQ', '12']]),
    5: ('fits without an EOS (Bilby)', [
        B + ['--config', 'configs/bilby_1728.json', '--mode', 'free',
             '--scatter', 'frac2', '--nlive', '400', '--K', '300',
             '--outdir', 'bilby_out/4U1728_free_frac2'],
        B + ['--config', 'configs/bilby_GX5-1.json', '--mode', 'free',
             '--scatter', 'frac2', '--nlive', '300', '--K', '300',
             '--M-min', '1.4', '--M-max', '3.2',
             '--outdir', 'bilby_out/GX5-1_free_frac2'],
        B + ['--config', 'configs/bilby_GX340+0.json', '--mode', 'free',
             '--scatter', 'frac2', '--nlive', '300', '--K', '300',
             '--M-min', '1.4', '--M-max', '3.2',
             '--outdir', 'bilby_out/GX340+0_free_frac2']]),
    6: ('data figures, corner plot and table', [
        [PY, 'figures.py', '--only', 'tracks,corner,table']]),
    7: ('numbers quoted in the text (HT vs Kerr, Kerr-equivalent I, evidence)', [
        [PY, 'paper_checks.py']]),
}


def run(cmd):
    print('  $', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f'step failed: {" ".join(cmd)}')


def main():
    if sys.argv[1:] == ['list']:
        for k, (d, _) in STEPS.items():
            print(f'{k}: {d}')
        return
    os.makedirs('results', exist_ok=True)
    sel = [int(a) for a in sys.argv[1:]] or list(STEPS)
    for k in sel:
        d, cmds = STEPS[k]
        print(f'\n=== step {k}: {d} ===', flush=True)
        for c in cmds:
            run(c)


if __name__ == '__main__':
    main()
