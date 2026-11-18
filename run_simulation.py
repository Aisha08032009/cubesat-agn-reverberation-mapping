"""Baseline simulation: lag and mass uncertainty for 3, 6, and 12 satellites."""
import numpy as np
from drw import simulate_drw, echo_light_curve
from iccf import iccf_lag, fr_rss_uncertainty

TAU_TRUE = 10.0
TOPHAT_WIDTH = 4.0
TAU_DRW = 50.0
SF_INF = 0.20
BASELINE = 120.0
FLUX_ERR_FRAC = 0.03
LINE_SCALE = 0.5
LINE_SCATTER = 0.05

# Idealized cadence: one visit per 2 days per satellite, divided by constellation size.
BASE_SINGLE_SAT_CADENCE = 2.0
CONSTELLATIONS = {3: BASE_SINGLE_SAT_CADENCE / 3,
                   6: BASE_SINGLE_SAT_CADENCE / 6,
                   12: BASE_SINGLE_SAT_CADENCE / 12}

N_REALIZATIONS = 100
N_FRRSS = 1200

C_LIGHT_DAYS_TO_CM = 2.59e15
G_CGS = 6.674e-8
MSUN_G = 1.989e33
F_VIRIAL = 1.4
DELTA_V_KMS = 3000.0
DELTA_V_FRAC_ERR = 0.10

results = {}

for n_sats, cadence in CONSTELLATIONS.items():
    recovered_lags = []
    lag_uncertainties = []

    for real in range(N_REALIZATIONS):
        seed = 1000 * n_sats + real
        t_fine = np.arange(0, BASELINE, 0.1)
        continuum_fine = simulate_drw(t_fine, tau=TAU_DRW, sf_inf=SF_INF, mean=1.0, seed=seed)
        line_fine = echo_light_curve(t_fine, continuum_fine, lag=TAU_TRUE,
                                      tophat_width=TOPHAT_WIDTH, scale=LINE_SCALE,
                                      extra_scatter=LINE_SCATTER, seed=seed + 1)

        t_obs = np.arange(0, BASELINE, cadence)
        rng = np.random.default_rng(seed + 2)
        t_obs = t_obs + rng.uniform(-cadence * 0.1, cadence * 0.1, len(t_obs))
        t_obs = np.clip(np.sort(t_obs), 0, BASELINE - 0.01)

        cont_obs = np.interp(t_obs, t_fine, continuum_fine)
        line_obs = np.interp(t_obs, t_fine, line_fine)

        cont_err = FLUX_ERR_FRAC * np.abs(cont_obs)
        line_err = FLUX_ERR_FRAC * np.abs(line_obs)
        cont_obs_noisy = cont_obs + cont_err * rng.standard_normal(len(cont_obs))
        line_obs_noisy = line_obs + line_err * rng.standard_normal(len(line_obs))

        peak, centroid, _, _ = iccf_lag(t_obs, cont_obs_noisy, t_obs, line_obs_noisy,
                                         lag_min=-5, lag_max=40, lag_step=0.5)
        if np.isnan(centroid):
            continue

        centroids_mc = fr_rss_uncertainty(t_obs, cont_obs_noisy, cont_err,
                                           t_obs, line_obs_noisy, line_err,
                                           n_sims=N_FRRSS, lag_min=-5, lag_max=40,
                                           lag_step=0.5, seed=seed + 3)
        if len(centroids_mc) < 10:
            continue

        sigma_lag = np.std(centroids_mc)
        recovered_lags.append(centroid)
        lag_uncertainties.append(sigma_lag)

    recovered_lags = np.array(recovered_lags)
    lag_uncertainties = np.array(lag_uncertainties)

    mean_lag = np.mean(recovered_lags)
    mean_sigma_lag = np.mean(lag_uncertainties)
    lag_scatter_across_realizations = np.std(recovered_lags)

    R_cm = mean_lag * C_LIGHT_DAYS_TO_CM
    dV_cms = DELTA_V_KMS * 1e5
    M_g = F_VIRIAL * R_cm * dV_cms**2 / G_CGS
    M_Msun = M_g / MSUN_G

    frac_err_lag = mean_sigma_lag / mean_lag
    frac_err_mass = np.sqrt(frac_err_lag**2 + (2 * DELTA_V_FRAC_ERR)**2)
    M_err_Msun = M_Msun * frac_err_mass

    results[n_sats] = dict(
        cadence_days=cadence,
        n_epochs=int(BASELINE / cadence),
        mean_recovered_lag=mean_lag,
        mean_lag_sigma=mean_sigma_lag,
        lag_scatter_across_realizations=lag_scatter_across_realizations,
        frac_err_lag=frac_err_lag,
        mass_Msun=M_Msun,
        mass_err_Msun=M_err_Msun,
        frac_err_mass=frac_err_mass,
        n_valid_realizations=len(recovered_lags),
    )

print(f"{'N_sats':>7} {'Cadence(d)':>11} {'N_epochs':>9} {'Lag(d)':>8} {'sig_lag(d)':>11} "
      f"{'frac_err_lag':>13} {'Mass(1e7 Msun)':>15} {'frac_err_mass':>14}")
for n_sats in sorted(results.keys()):
    r = results[n_sats]
    print(f"{n_sats:>7} {r['cadence_days']:>11.3f} {r['n_epochs']:>9} "
          f"{r['mean_recovered_lag']:>8.2f} {r['mean_lag_sigma']:>11.2f} "
          f"{r['frac_err_lag']:>13.3f} {r['mass_Msun']/1e7:>15.3f} {r['frac_err_mass']:>14.3f}")

import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nTrue lag was:", TAU_TRUE, "days")
print("Saved results.json")
