# Scientific code audit and validation record

**Input:** Original uploaded `cubesat_rm_code(1).zip` and corresponding 13-page research manuscript.  
**Output status:** Corrected exploratory computational pipeline; **not** a fully reproduced or independently validated version of the paper.

## Scientific audit

1. **Legacy DRW inconsistency (confirmed):** Under the legacy code with tau=50 and `sf_inf=0.2`, the initialization SD was 0.20 while the stochastic recursion's stationary SD was 0.04. A 2,000-day trajectory using the legacy implementation measured roughly 0.0419 SD (expected legacy recursion 0.04).
2. **Corrected OU convention:** This version defines `sf_inf` as the infinite-lag structure function. For 0.2, the stationary SD is `0.2/sqrt(2) ≈ 0.1414`. This convention is **not** interchangeable with the manuscript's unspecific “20% variability amplitude”; confirm intended meaning before a scientific corrigendum.
3. **Echo endpoint behavior (confirmed):** The original same-mode convolution imposed implicit zeros beyond the generated light curve, artificially depressing the line flux at campaign boundaries. The new echo generator extends edge values, passing the constant-signal invariance test. A prior-epoch burn-in should replace this approximation in a more physical simulation.
4. **Monte Carlo settings:** The supplied `run_simulation.py` used 100 realizations × 1,200 trials, while `full_constraints_analysis.py` used 100 × 300; the latter now uses 1,200, and the new runner makes modes explicit. All reported results are labeled with their actual run settings.
5. **Sky coordinates:** The uploaded four-constraint orbit model mixed a target defined by an orbital pole angle with a separately specified ecliptic target for Sun exclusion. The revised model uses the ecliptic target consistently in both calculations. Legacy geometry is still available via `consistent_target_geometry=False` for comparison.
6. **ICCF numerical acceleration:** A Numba kernel matches the original NumPy two-direction ICCF to ≲2×10⁻¹⁵ absolute error on three seeded test datasets. A regression test retains this equivalence check.

## Tests and executed experiments

- Regression suite: **7 passed** (`PYTHONPATH=. pytest -q tests/test_scientific_core.py`).
- 500-km circular orbit: period ~94.47 minutes, in agreement with two-body Keplerian period.
- Original visibility code reproduced the manuscript's attempted/successful observation counts: 78/180, 170/360, 346/720 for 3, 6, and 12 satellites respectively. The consistent-target revised geometry yields the same counts for these three configurations in the diagnostic schedule, but this does not validate all physics assumptions.
- **Smoke run:** 4 independent realizations and 40 FR/RSS trials per realization, both ideal and full-constraint. Numerical CSV and plots are in `results/smoke/`.
- **Diagnostic run:** 30 independent realizations and 300 FR/RSS trials per realization, both ideal and full-constraint. This run completed successfully; tables and figures are in `results/diagnostic/`.
- **Sensitivity smoke:** 6 independent realizations and 100 FR/RSS trials per configuration, 3/6/12 satellites at multiple true lags and noise levels. This run completed; see `results/sensitivity_smoke/`. This is intended as a functional smoke test, not a robust sensitivity conclusion.
- **Full paper-scale corrected experiment:** **NOT RUN HERE.** The `--mode full` preset is supplied and is configured for 100 independent realizations × 1,200 FR/RSS trials for each baseline and four-constraint configuration. It requires a substantially larger compute allocation. No full-scale output files have been fabricated or implied.

## Diagnostic results (corrected simulation)

| N satellites | Idealized uncertainty | Full-constraint uncertainty |
|---:|---:|---:|
| 3 | 8.69% | 19.71% |
| 6 | 6.20% | 11.57% |
| 12 | 3.75% | 16.06% |

These percentages quantify the code-defined mean FR/RSS lag SD divided by mean recovered lag, **not black-hole mass uncertainty**. The corrected full-constraint diagnostic is not monotonic between N=6 and N=12; this undermines a strong diminishing-returns inference without more runs, a physical review, and confidence intervals.

## Reproducibility instructions

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests    # Windows: python -m pytest -q tests
python run_corrected_experiments.py --mode smoke
python run_corrected_experiments.py --mode diagnostic
python run_corrected_experiments.py --mode full   # very computationally expensive
python run_sensitivity_diagnostic.py
```

Numerical results are saved as CSV/JSON together with plots. Each corrected-run `run_metadata.json` includes configuration, creation time, assumptions, and caveats. The original manuscript PDF remains unchanged. The original code is preserved verbatim in `original_source/original_python_scripts.zip`.

## Outstanding work before claiming manuscript reproduction

- Resolve whether “20% asymptotic variability” means RMS or the structure-function amplitude.
- Reproduce sensitivity and orbital results at the manuscript's stated statistical power after the corrections; compute independent confidence intervals and report all failed/filtered realizations.
- Ensure a single physically consistent target coordinates and orbital geometry throughout; add perturbations and space-mission subsystem constraints as appropriate.
- Model pre-campaign continuum history; compare the ICCF centroid against a contiguous-main-peak implementation on multimodal CCFs.
- Compare against independent reverberation-mapping packages and/or publicly available observed AGN light curves.
- Review quantitative changes with the research mentor; if the manuscript is published/accepted, pursue corrections through the relevant publication process if warranted.

**Provenance:** Original manuscript numerical results and newly computed corrected results must not be silently merged or substituted for each other.
