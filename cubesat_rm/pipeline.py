"""End-to-end realization -> lag -> mass-uncertainty pipeline."""
from dataclasses import replace
import numpy as np
from scipy import stats
from .config import SimConfig, OrbitConfig
from .drw import generate_light_curves, sample_light_curves
from .iccf import fr_rss, iccf, peak_and_centroid
from .mass import fractional_mass_error
from .orbit import observation_times


def lag_grid(cfg: SimConfig):
    return np.arange(cfg.lag_min, cfg.lag_max + cfg.lag_step / 2, cfg.lag_step)


def run_realization(seed, obs_times, cadence, cfg: SimConfig):
    """One DRW realization -> sampled curves -> FR/RSS lag distribution.

    Returns dict(lag_mean, lag_sigma, n_epochs). NaNs if the light curve is too sparse.
    """
    rng = np.random.default_rng(seed)
    t, c, l = generate_light_curves(rng, cfg)
    if len(obs_times) < cfg.min_points:
        return dict(lag_mean=np.nan, lag_sigma=np.nan, lag_direct=np.nan, n_epochs=len(obs_times))
    cont, line = sample_light_curves(t, c, l, obs_times, cfg, rng, cadence)
    lags = lag_grid(cfg)
    direct = peak_and_centroid(lags, iccf(cont[0], cont[1], line[0], line[1], lags),
                               cfg.peak_frac, cfg.centroid_contiguous)[1]
    dist = fr_rss(cont, line, lags, cfg.n_mc, rng, cfg.peak_frac, cfg.centroid_contiguous)
    dist = dist[np.isfinite(dist)]
    if dist.size < 10:
        return dict(lag_mean=np.nan, lag_sigma=np.nan, lag_direct=np.nan, n_epochs=len(obs_times))
    return dict(lag_mean=float(dist.mean()), lag_sigma=float(dist.std(ddof=1)),
                lag_direct=float(direct), n_epochs=len(obs_times))


def _worker(args):
    return run_realization(*args)


def run_configuration(n_sats, n_real, cfg: SimConfig, mode="ideal", ocfg: OrbitConfig = None,
                      seed_base=0, n_jobs=1, progress=None, obs=None):
    """Run n_real independent realizations for an N-satellite constellation."""
    cadence = cfg.base_cadence / n_sats
    if obs is None:
        obs = observation_times(n_sats, mode, ocfg, cfg.baseline, cfg.base_cadence)
    # "ideal" and "occultation" share seeds: when every attempt succeeds the observation times are
    # identical, so the two give numerically identical results (as in the paper, Fig. 5a).
    ss = np.random.SeedSequence([seed_base, n_sats, {"ideal": 0, "occultation": 0, "full": 2}[mode]])
    seeds = ss.generate_state(n_real)
    jobs = [(int(s), obs, cadence, cfg) for s in seeds]
    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as p:
            res = p.map(_worker, jobs)
    else:
        res = []
        for k, j in enumerate(jobs):
            res.append(_worker(j))
            if progress:
                progress(k + 1, n_real)
    return summarize(res, cfg, n_sats, cadence, len(obs))


def summarize(results, cfg: SimConfig, n_sats, cadence, n_epochs):
    lag = np.array([r["lag_mean"] for r in results])
    sig = np.array([r["lag_sigma"] for r in results])
    direct = np.array([r.get("lag_direct", np.nan) for r in results])
    ok = np.isfinite(lag) & np.isfinite(sig)
    out = dict(n_sats=n_sats, cadence_days=cadence, n_epochs=int(n_epochs),
               n_real=len(results), n_valid=int(ok.sum()))
    if ok.sum() < 3:
        out.update(lag_mean=np.nan, lag_sigma=np.nan, sem=np.nan, frac_lag_err=np.nan,
                   frac_mass_err=np.nan, sigmas=[], lags=[], directs=[])
        return out
    lag, sig = lag[ok], sig[ok]
    frac = sig.mean() / lag.mean()
    out.update(lag_mean=float(lag.mean()), lag_sigma=float(sig.mean()),
               sem=float(sig.std(ddof=1) / np.sqrt(ok.sum())),
               frac_lag_err=float(frac),
               frac_mass_err=float(fractional_mass_error(frac, cfg.delta_v_frac_err)),
               sigmas=sig.tolist(), lags=lag.tolist(), directs=direct[ok].tolist())
    return out


def welch_pairwise(summaries):
    """Pairwise Welch t-tests on per-realization lag sigma."""
    out = {}
    keys = sorted(summaries)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = summaries[a]["sigmas"], summaries[b]["sigmas"]
            if len(sa) > 2 and len(sb) > 2:
                t, p = stats.ttest_ind(sa, sb, equal_var=False)
                out[f"{a}v{b}"] = dict(t=float(abs(t)), p=float(p))
    return out


def run_realization_obs(seed, t_obs, frac_c, frac_l, cfg: SimConfig):
    """Realization with explicit epoch times and per-epoch FRACTIONAL errors (radiometric model)."""
    rng = np.random.default_rng(seed)
    t, c, l = generate_light_curves(rng, cfg)
    if len(t_obs) < cfg.min_points:
        return dict(lag_mean=np.nan, lag_sigma=np.nan, lag_direct=np.nan, n_epochs=len(t_obs))
    tc = np.asarray(t_obs, float)
    out = []
    for f, fr in ((c, frac_c), (l, frac_l)):
        true = np.interp(tc, t, f); err = np.asarray(fr) * np.abs(true)
        out.append((tc, true + rng.normal(size=tc.size) * err, err))
    cont, line = out
    lags = lag_grid(cfg)
    direct = peak_and_centroid(lags, iccf(cont[0], cont[1], line[0], line[1], lags),
                               cfg.peak_frac, cfg.centroid_contiguous)[1]
    dist = fr_rss(cont, line, lags, cfg.n_mc, rng, cfg.peak_frac, cfg.centroid_contiguous)
    dist = dist[np.isfinite(dist)]
    if dist.size < 10:
        return dict(lag_mean=np.nan, lag_sigma=np.nan, lag_direct=float(direct), n_epochs=len(t_obs))
    return dict(lag_mean=float(dist.mean()), lag_sigma=float(dist.std(ddof=1)),
                lag_direct=float(direct), n_epochs=len(t_obs))


def run_configuration_obs(t_obs, frac_c, frac_l, n_real, cfg: SimConfig, seed_key, n_sats=0):
    seeds = np.random.SeedSequence(list(seed_key)).generate_state(n_real)
    res = [run_realization_obs(int(s), t_obs, frac_c, frac_l, cfg) for s in seeds]
    cad = float(np.median(np.diff(t_obs))) if len(t_obs) > 1 else np.nan
    return summarize(res, cfg, n_sats, cad, len(t_obs))
