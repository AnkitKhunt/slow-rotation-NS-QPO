# Relativistic precession around slowly rotating neutron stars

Code and data for the paper

> A. C. Khunt, *Relativistic precession around rotating neutron stars.
> I. Slow rotation: Hartle–Thorne spacetimes, particle orbits and the QPOs of
> atoll and Z sources* (2026, in preparation).

The code builds rotating neutron stars in the Hartle–Thorne (HT)
approximation from an equation of state (EOS), computes the orbital and
epicyclic frequencies of test particles in their exterior spacetime,
integrates tilted eccentric orbits and the light curve of a hot spot on
them, and fits the relativistic precession model (RPM) to the quasi-periodic
oscillations (QPOs) of the atoll sources 4U 1636−53 and 4U 1728−34 and the
Z sources GX 5−1 and GX 340+0.  It reproduces every figure, table and quoted
number of the paper.

## Quick start

    git clone https://github.com/<your-username>/HT-QPO-PaperI.git
    cd HT-QPO-PaperI
    python3 -m venv .venv
    source .venv/bin/activate            # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python3 check_setup.py               # a few seconds
    python3 run_all.py 2 3 6 7           # all figures and tables, ~2 min

Output goes to `paper_out/`.  **How to run each script, with all options:
[docs/USAGE.md](docs/USAGE.md).**

## Physics in one paragraph

A rigidly rotating star is built to O(Ω²) (Hartle 1967; Hartle & Thorne
1968): TOV background, frame dragging (ℓ = 1 → J, I), spherical deformation
(ℓ = 0 → δM) and quadrupolar deformation (ℓ = 2 → Q, R_eq).  The exterior
metric depends only on (M, J, Q); we use j = J/M², q = Q/M³ (Geroch–Hansen
sign: oblate star q < 0, Kerr q = −j²) and Q̄ = −q/j².  For circular
equatorial geodesics we compute, without expanding in j or q,

    ν_φ (orbital),  ν_r (radial epicyclic),  ν_θ (vertical epicyclic),
    ν_per = ν_φ − ν_r (periastron precession),  ν_nod = ν_φ − ν_θ (nodal),

and use the RPM: ν_U = ν_φ, ν_L = ν_per, ν_LF/HBO = 2ν_nod, all at one
radius per set of QPOs.  Units: G = c = 1 (lengths in km), frequencies in Hz,
masses in M_sun.

## Requirements

Python 3.10 or newer; NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.8, pandas
≥ 1.5; Bilby, dynesty (< 3.2) and corner only for the nested-sampling fits.
No LaTeX installation is needed for the figures.  Tested on Linux with
Python 3.10 (oldest supported versions) and 3.12/3.13 (current versions),
with worker processes started the Windows/macOS way.  Conda works too:
`conda create -n htqpo python=3.12 && conda activate htqpo && pip install -r requirements.txt`.

## Reproducing the paper

    python3 run_all.py list
    python3 run_all.py 2 3 6 7     # figures and checks from the shipped results (~2 min)
    python3 run_all.py             # everything, fits included (~1–2 h on 2 cores)

| step | content | paper |
|---|---|---|
| 1 | frequency and star tables (shipped in `qpo_fit_cache/`) | — |
| 2 | stars, frequencies, nodal precession | Figs. 1–3, Table I |
| 3 | orbit, precession, hot-spot spectrum | Figs. 4–5 |
| 4 | RPM fits with each EOS (grid quadrature) | Table III, evidences |
| 5 | fits without an EOS (Bilby/dynesty) | Fig. 7 |
| 6 | QPO tracks, corner plot, fit table | Figs. 6–8, Table III |
| 7 | numbers quoted in the text (`paper_checks.py`) | Secs. VI B, VII D |

## Repository layout

| path | content |
|---|---|
| `run_all.py` | reproduces the paper |
| `check_setup.py` | checks the installation |
| `figures.py`, `plot_style.py` | paper figures and tables; figure style and panel labels |
| `orbit_qpo.py` | tilted eccentric orbit, precession, hot-spot light curve and spectrum |
| `orbits.py`, `veff.py` | HT metric, geodesic integrator, effective potential, orbit plots |
| `fft_qpo.py` | check that integrated orbits precess at the epicyclic rates |
| `run_qpo.py`, `plots.py` | one star of given mass and spin, its RPM table and plots |
| `qpo_fit.py` | cached tables, data reader, likelihood, Bilby fits, mock data |
| `qpo_grid_fit.py` | the same likelihood by grid quadrature; `predict`, `scan-spin` |
| `paper_checks.py` | HT vs Kerr accuracy, Kerr-equivalent I, evidences |
| `eos_loader.py`, `tov.py`, `ht_full.py`, `ht_common.py`, `qpo.py` | EOS, TOV, HT star, frequencies |
| `eos/` | EOS tables (see below) |
| `data/` | QPO data used in the fits; `data/raw/` with sources and original errors |
| `configs/` | source lists for the fits |
| `qpo_fit_cache/` | precomputed frequency and star tables |
| `results/`, `grid_fit_out/`, `bilby_out/` | outputs used by the paper (orbit, grid fits, posteriors) |
| `docs/USAGE.md` | how to run each script |
| `paper_out/` | created by the scripts (figures, tables) |

## Data sources

QPO frequencies (each file in `data/raw/` states its source and table):

* 4U 1636−53: twin kHz QPOs from RXTE, Zhang et al. (2017), MNRAS 465, 5003,
  as tabulated by Shi et al. (2024), A&A 691, A187.
* 4U 1728−34: QPO triplets from AstroSat/LAXPC, Anand et al. (2024), ApJ
  967, 129.
* GX 5−1: Jonker et al. (2002), MNRAS 333, 665.
* GX 340+0: Jonker et al. (2000), ApJ 537, 374.

Asymmetric errors are symmetrised in `data/*.csv`.  Check the values
against the original tables before using them for other work.

EOS tables:

* `eos/G3`, `eos/FSUGarnet`, `eos/IOPB` — unified EOSs (crust and core from
  the same relativistic mean-field functional) of V. Parmar, H. C. Das,
  A. Kumar, M. K. Sharma and S. K. Patra, Phys. Rev. D 105, 043017 (2022),
  arXiv:2111.07278, provided by H. C. Das and available at
  https://github.com/hcdas/Unified_eos.  Parameter sets: G3 (Kumar et al.
  2017), FSUGarnet (Chen & Piekarewicz 2015), IOPB-I (Kumar et al. 2018).
* `eos/eosAU` — WFF1 (Wiringa, Fiks & Fabrocini 1988), from the EOS
  collection distributed with the RNS code of N. Stergioulas (Stergioulas &
  Friedman 1995); `eos/README` is the index of that collection.
* Format (RNS): energy density/c² [g cm⁻³], pressure [dyn cm⁻²], enthalpy
  [cm² s⁻²], baryon number density [cm⁻³].

Please cite the original papers when you use these tables.

## Troubleshooting

* `Unknown symbol: \boldsymbol` — Matplotlib is older than 3.8:
  `pip install -U "matplotlib>=3.8"`.
* `cannot import name 'apply_reflect' from 'dynesty.utils'` — dynesty 3.2
  does not work with Bilby 2.8: `pip install "dynesty<3.2"`.
* `FileNotFoundError` for `eos/...` or `qpo_fit_cache/...` — run the scripts
  from the top folder of the repository.
* Bilby is only needed for step 5.  Without it everything else runs
  (`python3 run_all.py 1 2 3 4 6 7`), and step 6 uses the shipped posterior.
* On Windows use `python` instead of `python3`.

## Caveats

* The HT star is correct to O(Ω²); results at ν_s ≳ 600 Hz (Z sources)
  should be checked with numerically exact rotating stars (Paper II, in
  preparation).
* The hot spot is a kinematic illustration (no light bending); it shows
  which frequencies an orbit imprints on the flux, not their amplitudes in a
  real accretion flow.

## Citation and licence

If you use this code, please cite the paper above (see `CITATION.cff`).
The code is released under the MIT licence (`LICENSE`).  The EOS tables and
QPO data belong to their original authors and are included for
reproducibility.

Contact: Ankit C. Khunt, ack.gravity@gmail.com
