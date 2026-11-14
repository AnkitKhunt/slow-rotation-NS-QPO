# How to run each script

Run every command from the top folder of the repository (the folder that
contains `run_all.py`).  On Windows use `python` (or `py`) where `python3` is
written, and put each command on one line (the `\` at a line end is the
Linux/macOS continuation).  Units: masses in M_sun, spin and QPO frequencies
in Hz, radii in km unless stated; x = r/M.

EOS names used below: `eos/eosAU` (WFF1), `eos/G3`, `eos/FSUGarnet`,
`eos/IOPB` (IOPB-I).  Sources: `data/<name>.csv`.  Fit configurations:
`configs/*.json`.

Times are for one core of a laptop.

---

## 0. Setup and the whole paper

| command | what it does | time |
|---|---|---|
| `python3 check_setup.py` | checks Python, packages, data files, fonts, and one star against the paper | seconds |
| `python3 run_all.py list` | lists the seven steps | — |
| `python3 run_all.py 2 3 6 7` | all figures, tables and checked numbers from the shipped results | ~2 min |
| `python3 run_all.py` | everything, including all fits | 1–2 h |
| `python3 run_all.py 4` | only the grid fits (Table III, evidences) | 15–30 min |
| `python3 paper_checks.py` | numbers quoted in the text: HT vs Kerr accuracy, Kerr-equivalent I, evidences | seconds |

Steps of `run_all.py`: 1 frequency and star tables (shipped in
`qpo_fit_cache/`; rerun only for a new EOS), 2 physics figures, 3 orbit and
hot-spot spectrum, 4 grid fits with each EOS, 5 fits without an EOS (Bilby),
6 data figures, corner plot and Table III, 7 `paper_checks.py`.  Step 1 uses
`NPOOL` processes (default 2): `export NPOOL=8` (Linux/macOS),
`set NPOOL=8` (Windows cmd), `$env:NPOOL=8` (PowerShell).

---

## 1. One star: structure and QPO frequencies

### `ht_full.py` — self-test of one HT star
    python3 ht_full.py
Builds a WFF1 star (ρ_c = 10¹⁵ g cm⁻³, 400 Hz) and prints M, δM, R_eq,
R_pol, J, Q, j, q, q/j² with sanity checks (Q < 0, δM > 0, R_eq > R_pol).

### `run_qpo.py` — a star of given mass and spin, and its RPM table
    python3 run_qpo.py --eos eos/IOPB --mass 1.9 --spin 363 --outdir out_star
    python3 run_qpo.py --eos eos/IOPB --mass 1.9 --spin 363 --outdir out_star --plots
Options: `--mass` (or `--rho_c` central density in g cm⁻³, or `--radius`),
`--spin` [Hz], `--rmax` outer radius [r/M], `--npts` number of radii,
`--nu-upper` mark a given ν_U, `--plots` adds PNG plots (EOS, M–R curve,
effective potential and its derivatives, all frequencies vs radius, ν_L vs
ν_U, HBO vs ν_U), `--plot-npts` points on the M–R curve.  Output:
`out_star/qpo_table.dat` with ν_φ, ν_r, ν_θ, ν_per, ν_nod at each radius.
(~25 s)

### `qpo_grid_fit.py predict` — RPM frequencies vs radius (fast, from the cached tables)
    python3 qpo_grid_fit.py predict --eos eos/IOPB --mass 1.92 --spin 363 --n 60 --out predict.txt
Options: `--rmax` outer r/M, `--n` number of radii.

### `qpo_grid_fit.py scan-spin` — spin needed for an observed (ν_U, HBO) pair
    python3 qpo_grid_fit.py scan-spin --eos eos/IOPB eos/G3 --mass 2.0 --nu-u 800 --nu-hbo 40
Options: `--spin-lo`, `--spin-hi`, `--dspin` [Hz].  Prints, for each EOS and
spin, j, q/j², ν_L and 2ν_nod at the radius where ν_φ = ν_U, next to the
observed HBO.

### `ht_common.py grid` — properties on a (mass, spin) grid
    python3 ht_common.py grid --eos eos/IOPB --masses 1.4 1.6 1.9 --spins 300 450 600 --out grid_IOPB.npz
Stores, for every (M, ν_s): central density, M₀, δM/M, R, R_eq (coordinate
and circumferential), j, q, q/j², I, and the inner disc edge r_in with a
flag telling whether it is the ISCO or the surface.  (~10 s per star)

### Python, inside your own script
```python
from qpo_fit import StarTable, M_SUN_KM
from qpo import qpo_at_radius, find_isco_km
st = StarTable('eos/IOPB').star(1.92, 363.0)        # M [M_sun], nu_s [Hz]
M = 1.92 * M_SUN_KM                                # M in km
print(st['j'], -st['q_over_j2'], st['R_eq'], st['I'])
print(find_isco_km(M, st['j'], st['q'], r_surface=st['R_eq']))
print(qpo_at_radius(M, st['j'], st['q'], 6.5 * M)) # dict of frequencies in Hz
```
`qpo_at_radius` returns `nu_phi, nu_r, nu_theta, nu_peri, nu_nod` (and
`nu_LT`, an old alias of `nu_nod`, not the frame-dragging term of the paper).

---

## 2. Orbits and QPO peaks

### `orbit_qpo.py` — tilted eccentric orbit, precession and hot-spot spectrum (Figs. 4, 5)
    python3 orbit_qpo.py --eos eos/IOPB --mass 1.922 --spin 363 \
        --nu-target 1098.4 --ecc 0.01 --inc 8 --iobs 60 --T 0.5 --out results/my_orbit.npz
Options: `--nu-target` sets the start radius r₀ through ν_φ(r₀);
`--ecc` radial kick as a fraction of the orbital speed; `--inc` orbital tilt
[deg]; `--iobs` observer inclination [deg]; `--T` duration [s] (frequency
resolution 1/T).  Prints the epicyclic frequencies at r₀, the spectral peaks
and the directly measured precession rates.  (~1 min for 0.5 s)

Figures for your own orbit:

    python3 -c "import os, figures as f, plot_style as ps; ps.use(); os.makedirs('my_out', exist_ok=True); f.OUT='my_out'; f.fig_orbit('results/my_orbit.npz'); f.fig_psd('results/my_orbit.npz')"

### `orbits.py` — circular orbits and a general geodesic, many diagnostic plots
    python3 orbits.py --eos eos/IOPB --mass 1.9 --spin 363 --ecc 0.05 \
        --theta-start 80 --n-orbits-geo 20 --outdir out_orbits
Options: `--r-circular` radius of the circular orbit [km],
`--n-orbits-circ`, `--r-start` start radius of the geodesic [km],
`--theta-start` colatitude reached (inclination = 90° − this), `--ecc`,
`--n-orbits-geo`, `--star-image` optional picture of the star.  Output:
PNG plots of the orbit in 2D and 3D, r(t), θ(t), conservation of E, L_z and
u·u, the effective potential, a Poincaré section and a frequency check.
(~30 s)

### `fft_qpo.py` — check that integrated orbits precess at the epicyclic rates
    python3 fft_qpo.py --eos eos/IOPB --mass 1.922 --spin 363 --R0 17.39 \
        --n-orbits 60 --scan-n 5 --outdir out_fft
Options: `--R0` start radius [km] (default 1.6 r_ISCO), `--ecc`,
`--inc-deg`, `--n-orbits`, `--scan-n` number of radii in a scan (0 = skip),
`--scan-n-orbits`, `--amp-test` (shift of the frequencies with amplitude).
Prints measured vs epicyclic ν_r, ν_θ, ν_φ, ν_nod at each radius.  (~1 min)

---

## 3. Fits to QPO data

### `qpo_grid_fit.py eos` — RPM fit with an EOS (grid quadrature; Table III)
    python3 qpo_grid_fit.py eos --config configs/atoll.json --eos eos/IOPB eos/G3 \
        --only 4U1728-34 --scatter frac2 --nE 24 --label frac2 --M-grid 1.75 2.05 --nM 121
Options: `--only` source names from the config; `--scatter frac` (one
fractional scatter), `frac2` (separate kHz and LF/HBO scatters) or `abs`
[Hz]; `--nE` scatter grid; `--M-grid lo hi` and `--nM` mass grid;
`--spin-grid lo hi` and `--nS` when the spin is free (Z sources); `--K`
radius grid; `--x-max` outer radius r/M; `--label` tag of the output.
Output: `grid_fit_out/eos/<source>/<EOS>_<label>.json` (lnZ, medians,
intervals) and `_grid.npz`.  The exact commands of the paper are in
`run_all.py`, step 4.  (seconds to minutes per source and EOS)

### `qpo_grid_fit.py free` — fit without an EOS: (M, j, Q̄)
    python3 qpo_grid_fit.py free --config configs/atoll.json --only 4U1636-53 \
        --nE 24 --K 300 --M-min 1.4 --M-max-free 2.6 --nM 49 --nj 31 --nQ 12
Options as above plus `--nj`, `--j-max`, `--nQ`, `--qbar-lo`, `--qbar-hi`,
`--qbar-fixed 1` (Kerr).

### `qpo_grid_fit.py resummarise` — rebuild the JSON summaries from saved grids
    python3 qpo_grid_fit.py resummarise --root grid_fit_out

### `qpo_fit.py fit` — nested sampling with Bilby/dynesty (corner plot, Fig. 7)
    python3 qpo_fit.py fit --config configs/bilby_1728.json --mode free \
        --scatter frac2 --nlive 400 --K 300 --npool 4 --outdir bilby_out/4U1728_free_frac2
Options: `--mode eos` (M, ν_s with `--eos`), `free` (M, j, Q̄) or `kerr`
(Q̄ = 1); `--nlive` live points; `--npool` processes; `--M-min`,
`--M-max`; `--resume` continue an interrupted run.  Output: posterior CSV,
corner plots, `summary.tex`.  (15–30 min; `--nlive 40` gives a quick test
in ~1 min)

### `qpo_fit.py build-tables` — frequency and star tables (step 1)
    python3 qpo_fit.py build-tables --eos eos/IOPB eos/G3 --npool 4
Options: `--nu-ref` reference spin of the star table, `--n-rho` number of
central densities, `--skip-freq` rebuild only the star tables.  Output:
`qpo_fit_cache/`.  (~10 min; needed only for a new EOS)

### `qpo_fit.py check-table` — compare the cached tables with the direct calculation
    python3 qpo_fit.py check-table --eos eos/IOPB
Prints the largest relative error of ν_U and ν_L (~10⁻⁴).  (~45 s)

### `qpo_fit.py mock` — synthetic QPO data from a known star (test of the fits)
    python3 qpo_fit.py mock --eos eos/IOPB --mass 1.9 --spin 400 --n 15 --sigma 5 --hbo --seed 1 --out data/mock.csv
Options: `--n` number of QPO sets, `--sigma` error [Hz], `--xspan` radii
drawn in [1.02, xspan] × r_in, `--hbo` add 2ν_nod.  Fit it with a config
that points to `data/mock.csv`.

---

## 4. Figures and tables

### `figures.py`
    python3 figures.py                                   # all, into paper_out/
    python3 figures.py --only stars,freqs,nodal          # Figs. 1–3 and tab_eos.tex
    python3 figures.py --only orbit,psd                  # Figs. 4–5 (needs results/orbit.npz)
    python3 figures.py --only tracks,corner,table        # Figs. 6–8 and tab_fits.tex
    python3 figures.py --out my_figs
Names for `--only`: `stars, freqs, nodal, orbit, psd, tracks, corner, table`.
`paper_out/figures_info.json` holds the numbers read off the figures.

### `plot_style.py` — the figure style, for your own plots
```python
import matplotlib.pyplot as plt
import plot_style as ps
ps.use()                                   # fonts, bold ticks, inward ticks
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].plot([1, 2], [1, 4])
ax[0].set_xlabel(ps.lab(ps.S['M'], r'M$_\odot$'))
for a in ax: ps.finish(a)
fig.tight_layout()
ps.label_panels(fig, ax)                   # (a), (b) inside, without overlap
fig.savefig('my_plot.pdf')
```

---

## 5. Your own EOS or source

New EOS from a table of (n_B, ε, P) in MeV fm⁻³:

    python3 ht_common.py convert my_eos.dat eos/MyEOS --cols 0 1 2 --skip 1
    python3 qpo_fit.py build-tables --eos eos/MyEOS --npool 4

(`--no-rest-mass` if ε excludes the rest-mass energy.)  Add `eos/MyEOS` to
`E` in `run_all.py` and to `EOS_ORDER`/`EOS_LABEL` in `plot_style.py` to
include it in the paper figures.

New source: write `data/<name>.csv` with columns
`nu_L,sig_L,nu_U,sig_U,nu_HBO,sig_HBO` (`nan` for a missing QPO), add an
entry to a config in `configs/`, for example

    {"name": "MySource", "data": "../data/MySource.csv", "spin": [400.0, 1.0]}

(the data path is relative to the config file; `spin` is [value, error] in
Hz, or `null` to fit the spin), and run `qpo_grid_fit.py eos` with
`--only MySource`.

## Library modules (imported, not run)

| module | content |
|---|---|
| `eos_loader.py` | reads an RNS-format EOS table; `pressure(ρ)`, `energy_density(P)` |
| `tov.py` | TOV solver |
| `ht_full.py` | HT interior (ℓ = 0, 1, 2) and exterior metric |
| `ht_common.py` | mass solver, EOS conversion, grids |
| `qpo.py` | circular orbits, epicyclic frequencies, ISCO, RPM frequencies |
| `veff.py` | effective potential and its derivatives |
| `orbits.py` | 4×4 HT metric, Christoffel symbols, geodesic integrator |
| `plots.py` | plots for `run_qpo.py --plots` |
| `qpo_fit.py` | cached tables, data reader, likelihood, Bilby fits |
| `plot_style.py` | figure style and panel labels |
