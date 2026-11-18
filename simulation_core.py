"""Shared simulation core: realizations, lag recovery, and mass conversion."""
import numpy as np
from drw import simulate_drw, echo_light_curve
from iccf import iccf_lag, fr_rss_uncertainty

C_LIGHT_DAYS_TO_CM = 2.59e15
G_CGS = 6.674e-8
MSUN_G = 1.989e33
F_VIRIAL = 1.4
DELTA_V_KMS = 3000.0
DELTA_V_FRAC_ERR = 0.10


def make_one_realization(cadence, tau_true=10.0, tophat_width=4.0, tau_drw=50.0,
                          sf_inf=0.20, baseline=120.0, flux_err_frac=0.03,
                          line_scale=0.5, line_scatter=0.05, seed=0):
    """Simulate fine light curves and the noisy observations at a uniform cadence."""
    t_fine = np.arange(0, baseline, 0.1)
    continuum_fine = simulate_drw(t_fine, tau=tau_drw, sf_inf=sf_inf, mean=1.0, seed=seed)
    line_fine = echo_light_curve(t_fine, continuum_fine, lag=tau_true,
                                  tophat_width=tophat_width, scale=line_scale,
                                  extra_scatter=line_scatter, seed=seed + 1)

    t_obs = np.arange(0, baseline, cadence)
    rng = np.random.default_rng(seed + 2)
    t_obs = t_obs + rng.uniform(-cadence * 0.1, cadence * 0.1, len(t_obs))
    t_obs = np.clip(np.sort(t_obs), 0, baseline - 0.01)

    cont_obs = np.interp(t_obs, t_fine, continuum_fine)
    line_obs = np.interp(t_obs, t_fine, line_fine)

    cont_err = flux_err_frac * np.abs(cont_obs)
    line_err = flux_err_frac * np.abs(line_obs)
    cont_obs_noisy = cont_obs + cont_err * rng.standard_normal(len(cont_obs))
    line_obs_noisy = line_obs + line_err * rng.standard_normal(len(line_obs))

    return dict(t_fine=t_fine, continuum_fine=continuum_fine, line_fine=line_fine,
                t_obs=t_obs, cont_obs=cont_obs_noisy, line_obs=line_obs_noisy,
                cont_err=cont_err, line_err=line_err)


def make_one_realization_with_times(t_obs, tau_true=10.0, tophat_width=4.0, tau_drw=50.0,
                                     sf_inf=0.20, baseline=120.0, flux_err_frac=0.03,
                                     line_scale=0.5, line_scatter=0.05, seed=0,
                                     jitter_cadence=None):
    """Same as make_one_realization, but sampled at externally supplied observation times."""
    t_fine = np.arange(0, baseline, 0.1)
    continuum_fine = simulate_drw(t_fine, tau=tau_drw, sf_inf=sf_inf, mean=1.0, seed=seed)
    line_fine = echo_light_curve(t_fine, continuum_fine, lag=tau_true,
                                  tophat_width=tophat_width, scale=line_scale,
                                  extra_scatter=line_scatter, seed=seed + 1)

    t_obs = np.asarray(t_obs, dtype=float)
    rng = np.random.default_rng(seed + 2)

    if jitter_cadence is not None:
        t_obs = t_obs + rng.uniform(-jitter_cadence * 0.1, jitter_cadence * 0.1, len(t_obs))
        t_obs = np.clip(t_obs, 0, baseline - 0.01)

    cont_obs = np.interp(t_obs, t_fine, continuum_fine)
    line_obs = np.interp(t_obs, t_fine, line_fine)

    cont_err = flux_err_frac * np.abs(cont_obs)

    line_err = flux_err_frac * np.abs(line_obs)
    cont_obs_noisy = cont_obs + cont_err * rng.standard_normal(len(cont_obs))
    line_obs_noisy = line_obs + line_err * rng.standard_normal(len(line_obs))

    return dict(t_fine=t_fine, continuum_fine=continuum_fine, line_fine=line_fine,
                t_obs=t_obs, cont_obs=cont_obs_noisy, line_obs=line_obs_noisy,
                cont_err=cont_err, line_err=line_err)


def run_configuration_with_times(n_sats, t_obs, tau_true=10.0, tophat_width=4.0, tau_drw=50.0,
                                  sf_inf=0.20, baseline=120.0, flux_err_frac=0.03,
                                  n_realizations=20, n_frrss=150, seed_base=1000,
                                  jitter_cadence=None):
    """Run all realizations for one configuration using a fixed set of observation times."""
    recovered_lags = []
    lag_uncertainties = []
    example = None

    for real in range(n_realizations):
        seed = seed_base * n_sats + real
        obs = make_one_realization_with_times(t_obs, tau_true, tophat_width, tau_drw,
                                               sf_inf, baseline, flux_err_frac, seed=seed,
                                               jitter_cadence=jitter_cadence)
        t_o, cont_obs_noisy, line_obs_noisy = obs['t_obs'], obs['cont_obs'], obs['line_obs']
        cont_err, line_err = obs['cont_err'], obs['line_err']

        peak, centroid, lag_grid, ccf = iccf_lag(t_o, cont_obs_noisy, t_o, line_obs_noisy,
                                                  lag_min=-5, lag_max=40, lag_step=0.5)
        if np.isnan(centroid):
            continue

        centroids_mc = fr_rss_uncertainty(t_o, cont_obs_noisy, cont_err,
                                           t_o, line_obs_noisy, line_err,
                                           n_sims=n_frrss, lag_min=-5, lag_max=40,
                                           lag_step=0.5, seed=seed + 3)
        if len(centroids_mc) < 10:
            continue

        recovered_lags.append(centroid)
        lag_uncertainties.append(np.std(centroids_mc))

        if example is None:
            example = dict(obs=obs, peak=peak, centroid=centroid,
                            lag_grid=lag_grid, ccf=ccf, centroids_mc=centroids_mc)

    recovered_lags = np.array(recovered_lags)
    lag_uncertainties = np.array(lag_uncertainties)

    mean_lag = np.mean(recovered_lags)
    mean_sigma_lag = np.mean(lag_uncertainties)

    R_cm = mean_lag * C_LIGHT_DAYS_TO_CM
    dV_cms = DELTA_V_KMS * 1e5
    M_g = F_VIRIAL * R_cm * dV_cms**2 / G_CGS
    M_Msun = M_g / MSUN_G

    frac_err_lag = mean_sigma_lag / mean_lag
    frac_err_mass = np.sqrt(frac_err_lag**2 + (2 * DELTA_V_FRAC_ERR)**2)
    M_err_Msun = M_Msun * frac_err_mass

    return dict(
        n_sats=n_sats, n_obs_epochs=len(t_obs),
        mean_recovered_lag=mean_lag, mean_lag_sigma=mean_sigma_lag,
        lag_scatter_across_realizations=float(np.std(recovered_lags)),
        lag_sigma_std_across_realizations=float(np.std(lag_uncertainties)),
        frac_err_lag=frac_err_lag, mass_Msun=M_Msun, mass_err_Msun=M_err_Msun,
        frac_err_mass=frac_err_mass, n_valid_realizations=len(recovered_lags),
        example=example,
    )


def run_configuration(n_sats, cadence, tau_true=10.0, tophat_width=4.0, tau_drw=50.0,
                       sf_inf=0.20, baseline=120.0, flux_err_frac=0.03,
                       n_realizations=20, n_frrss=150, seed_base=1000):
    """Run all realizations for one constellation at a uniform cadence; return lag and mass statistics."""
    recovered_lags = []
    lag_uncertainties = []
    example = None

    for real in range(n_realizations):
        seed = seed_base * n_sats + real
        obs = make_one_realization(cadence, tau_true, tophat_width, tau_drw,
                                    sf_inf, baseline, flux_err_frac, seed=seed)
        t_obs, cont_obs_noisy, line_obs_noisy = obs['t_obs'], obs['cont_obs'], obs['line_obs']
        cont_err, line_err = obs['cont_err'], obs['line_err']

        peak, centroid, lag_grid, ccf = iccf_lag(t_obs, cont_obs_noisy, t_obs, line_obs_noisy,
                                                  lag_min=-5, lag_max=40, lag_step=0.5)
        if np.isnan(centroid):
            continue

        centroids_mc = fr_rss_uncertainty(t_obs, cont_obs_noisy, cont_err,
                                           t_obs, line_obs_noisy, line_err,
                                           n_sims=n_frrss, lag_min=-5, lag_max=40,
                                           lag_step=0.5, seed=seed + 3)
        if len(centroids_mc) < 10:
            continue

        recovered_lags.append(centroid)
        lag_uncertainties.append(np.std(centroids_mc))

        if example is None:
            example = dict(obs=obs, peak=peak, centroid=centroid,
                            lag_grid=lag_grid, ccf=ccf, centroids_mc=centroids_mc)

    recovered_lags = np.array(recovered_lags)
    lag_uncertainties = np.array(lag_uncertainties)

    mean_lag = np.mean(recovered_lags)
    mean_sigma_lag = np.mean(lag_uncertainties)

    R_cm = mean_lag * C_LIGHT_DAYS_TO_CM
    dV_cms = DELTA_V_KMS * 1e5
    M_g = F_VIRIAL * R_cm * dV_cms**2 / G_CGS
    M_Msun = M_g / MSUN_G

    frac_err_lag = mean_sigma_lag / mean_lag
    frac_err_mass = np.sqrt(frac_err_lag**2 + (2 * DELTA_V_FRAC_ERR)**2)
    M_err_Msun = M_Msun * frac_err_mass

    return dict(
        n_sats=n_sats, cadence_days=cadence, n_epochs=int(baseline / cadence),
        mean_recovered_lag=mean_lag, mean_lag_sigma=mean_sigma_lag,
        lag_scatter_across_realizations=float(np.std(recovered_lags)),
        frac_err_lag=frac_err_lag, mass_Msun=M_Msun, mass_err_Msun=M_err_Msun,
        frac_err_mass=frac_err_mass, n_valid_realizations=len(recovered_lags),
        example=example,
    )
