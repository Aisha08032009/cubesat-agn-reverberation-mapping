# Cadence, CubeSats, and Certainty

**Research code for:** *Cadence, CubeSats, and Certainty: How Constellation Size Affects Reverberation-Mapped Black Hole Mass Uncertainty*  
**Author:** Aisha Shermatova  
**Manuscript:** [`paper/main.pdf`](paper/main.pdf) (the final 17-page PDF)

This repository contains the Python implementation, recorded numerical outputs, figures, and LaTeX files for a **synthetic study of AGN reverberation mapping with CubeSat constellations**. It combines DRW continuum variability, a broad-line transfer function, ICCF lag recovery, FR/RSS uncertainty estimation, a representative photon budget, and a dynamic-orbit visibility calculation.

The code is a documented reconstruction of the study's computational workflow; it is **not a historical archive of the original scripts**. The current manuscript was generated from the numerical results distributed in `results/` using `paper/make_paper.py`.

## Key results in the accompanying manuscript

**Controlled experiment:** injected lag = 10 days, 3% photometric noise, 100 independent AGN realizations and 1,200 FR/RSS trials per realization.

| Satellites | Cadence (days) | Mean recovered lag (days) | Mean lag uncertainty (days) | Fractional lag uncertainty | Fractional mass uncertainty |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.667 | 10.77 | 4.46 | 41.4% | 46.0% |
| 6 | 0.333 | 10.60 | 2.80 | 26.4% | 33.1% |
| 12 | 0.167 | 9.75 | 1.80 | 18.5% | 27.2% |

**Mission-model example:** NGC 5548, representative 10 cm photometer, simulated V = 17 target, 4% intrinsic variability, 30 realizations and 600 trials per configuration.

| Satellites | 1 | 3 | 6 | 12 |
|---|---:|---:|---:|---:|
| Fractional lag uncertainty | 37.7% | 19.6% | 11.5% | 8.0% |

These percentages refer to **modeled FR/RSS lag uncertainty**, not measured in-orbit spacecraft performance or demonstrated black hole mass accuracy. The full manuscript also reports bias, RMSE, interval coverage, seasonal observing gaps, and sensitivity to noise and AGN variability.

## Repository layout

| Path | Contents |
|---|---|
| `cubesat_rm/` | DRW simulation, transfer functions, ICCF and FR/RSS, orbital geometry, photon budget, mission scheduling, figures, CLI |
| `results/` | JSON files recording the experiments used by the manuscript |
| `figures/` | Figure assets; seven figures appear in the manuscript and two legacy orbital figures are supplementary |
| `paper/main.pdf` | Final manuscript PDF |
| `paper/main.tex`, `paper/main.tex.in` | Generated LaTeX source and its parameterized template |
| `paper/make_paper.py` | Rebuilds LaTeX tables and numerical statements from `results/*.json` |
| `tests/` | Automated unit and release-consistency checks |

## Installation

Python 3.10+ is recommended. In the repository's root directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Reproducing experiments

The committed `results/*.json` and `figures/*.png` preserve the outputs used for the manuscript. The computationally intensive commands below regenerate them; the default seed is **42** in the controlled experiments. Run these from the repository root:

```bash
python -m cubesat_rm baseline          # 100 realizations x 1200 trials; 3/6/12 sats
python -m cubesat_rm sensitivity       # true-lag and noise sensitivity (30 realizations)
python -m cubesat_rm diagnostics --n-real 100 --n-mc 600
python -m cubesat_rm mc-convergence
python -m cubesat_rm.experiments budget
python -m cubesat_rm.experiments targets "NGC 5548" "NGC 4151" "NGC 7469" "Mrk 509" "NGC 3783" "Fairall 9"
python -m cubesat_rm.experiments mission "NGC 5548:17:0.04" "NGC 5548:17:0.10" "NGC 5548:16:0.04" "NGC 5548:15:0.01" "NGC 5548:15:0.10"
python -m cubesat_rm.experiments sweeps base tau:20 tau:100 tau:200 amp:0.01 amp:0.10 amp:0.15 transfer:lognormal:4 transfer:lognormal:8
python -m cubesat_rm figures
python paper/make_paper.py
```

The baseline diagnostic and mission runs have different experiment settings, detailed in the Methods section. Full runs can be computationally expensive. A small execution check is:

```bash
python -m cubesat_rm baseline --n-real 3 --n-mc 20 --out /tmp/rm-smoke
```

The `/tmp` example is for Unix-like systems; on Windows provide a normal writable output folder instead. This smoke run **does not** reproduce the paper's published numerical precision.

To compile the LaTeX in `paper/`, run `pdflatex main.tex` twice from inside that folder. Exact pagination and fonts may depend on the LaTeX environment; `paper/main.pdf` is the finalized manuscript supplied with this release.

## Model definitions and assumptions

- **AGN variability:** exact-update damped random walk, stationary fractional RMS `amp_drw = 0.04` and damping timescale `tau_drw = 50` days. The controlled photometric noise is 3% (6% and 10% sensitivity cases).
- **BLR response:** a 10-day mean lag and 4-day top-hat transfer function by default, with alternative true lags and skewed transfer kernels tested separately.
- **Lag estimator:** ICCF evaluated every 0.5 days, with the centroid defined using all lag bins at least 80% of the maximum correlation. The contiguous-centroid alternative is included in the diagnostics.
- **Mass conversion:** `M ∝ tau × (velocity width)^2`, with a stipulated 10% velocity-width uncertainty, giving a 20% statistical floor in mass uncertainty; systematic uncertainties in the virial factor are outside the model.
- **Mission model:** 500 km circular orbit, 51.6° inclination, evenly phased satellites in one plane, J2 nodal regression, Sun avoidance, Earth occultation, eclipse/thermal recovery, one ground station, representative instrument parameters, and finite downlink capacity.
- **Targets:** six AGN with approximate J2000 RA/Dec listed in `cubesat_rm/orbit_dynamic.py`. The positions have been cross-checked at the precision used here against [SIMBAD](https://simbad.cds.unistra.fr/simbad/); the magnitudes in the experiment are controlled assumptions, not each object's measured magnitude.

The photon budget is illustrative, **not a verified flight-instrument design**. The simulations do not model continuum contamination in the line channel, attitude/pointing stability, Earth-limb constraints, the South Atlantic Anomaly, or alternative reverberation-mapping estimators such as JAVELIN/PyROA. The Results and Discussion in the manuscript describe these limitations and the distinction between estimated lag precision and estimator accuracy.

## Figures

Manuscript Figure 1 uses `fig1_example_light_curves.png`; Figure 2 `fig2_example_ccf.png`; Figure 3 `fig3_frrss_distributions.png`; Figure 4 `fig4_combined_uncertainty.png`; Figure 5 `fig9_parameter_sweeps.png`; Figure 6 `fig7_catalogue_targets.png`; Figure 7 `fig8_mission_photon_budget.png`. The assets `fig5_orbital_validation_combined.png` and `fig6_full_visibility_timeline.png` represent an earlier fixed-geometry model and are supplementary, **not the dynamic-orbit calculations reported as the mission model**.

## Citation

Shermatova, A. *Cadence, CubeSats, and Certainty: How Constellation Size Affects Reverberation-Mapped Black Hole Mass Uncertainty*. Research manuscript and accompanying simulation repository. Citation details can be updated upon formal publication.

## License

The source code is provided under the [MIT License](LICENSE). The manuscript and third-party cited literature retain their respective rights.
