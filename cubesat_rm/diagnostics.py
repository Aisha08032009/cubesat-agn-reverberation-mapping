"""Diagnostics a reviewer is likely to ask for.

1. Estimator accuracy: bias, RMSE and 1-sigma coverage of the recovered lag against the KNOWN
   true lag (FR/RSS sigma measures precision, not accuracy), plus a gross-failure rate.
2. Centroid definition: all lags above 0.8 r_peak (paper) vs contiguous region around the peak.
3. Binning test: does the 12-satellite improvement come from finer temporal sampling, or just from
   averaging photometric noise? 12-sat data are averaged in groups of 4 down to the 3-sat cadence.
4. Power-law slope of lag sigma vs N (is "diminishing returns" more than a generic power law?).
"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from .config import SimConfig
from .drw import generate_light_curves, sample_light_curves
from .iccf import fr_rss
from .orbit import observation_times
from .pipeline import run_configuration, lag_grid


def _boot_ci(x, rng, n=2000):
    x = np.asarray(x)
    m = [rng.choice(x, x.size).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def accuracy_stats(summary, true_lag, fail_abs=5.0, seed=0):
    d = np.asarray(summary["directs"], float); s = np.asarray(summary["sigmas"], float)
    ok = np.isfinite(d); d, s = d[ok], s[ok]
    err = d - true_lag
    rng = np.random.default_rng(seed)
    lo, hi = _boot_ci(err, rng)
    return dict(n=int(d.size), bias=float(err.mean()), bias_ci95=[lo, hi],
                rmse=float(np.sqrt(np.mean(err ** 2))), mean_sigma=float(s.mean()),
                coverage_1sigma=float(np.mean(np.abs(err) <= s)),
                gross_failure_rate=float(np.mean(np.abs(err) > fail_abs)))


def binned_test(n_real, n_mc, seed, group=4):
    cfg = SimConfig(n_mc=n_mc)
    sig = {"3 sats": [], "12 sats binned to 3-sat cadence": [], "12 sats": []}
    lags = lag_grid(cfg)
    for k in range(n_real):
        rng = np.random.default_rng([seed, k])
        t, c, l = generate_light_curves(rng, cfg)
        o3 = observation_times(3, "ideal", baseline_days=cfg.baseline)
        o12 = observation_times(12, "ideal", baseline_days=cfg.baseline)
        c3, l3 = sample_light_curves(t, c, l, o3, cfg, rng, cfg.base_cadence / 3)
        c12, l12 = sample_light_curves(t, c, l, o12, cfg, rng, cfg.base_cadence / 12)

        def binned(x):
            tt, ff, ee = x; m = (len(tt) // group) * group
            return (tt[:m].reshape(-1, group).mean(1), ff[:m].reshape(-1, group).mean(1),
                    np.sqrt((ee[:m].reshape(-1, group) ** 2).sum(1)) / group)
        for name, (cc, ll) in {"3 sats": (c3, l3), "12 sats": (c12, l12),
                               "12 sats binned to 3-sat cadence": (binned(c12), binned(l12))}.items():
            d = fr_rss(cc, ll, lags, n_mc, rng, cfg.peak_frac); d = d[np.isfinite(d)]
            if d.size > 10:
                sig[name].append(float(d.std(ddof=1)))
    return {k: dict(mean_sigma=float(np.mean(v)), sem=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v))
            for k, v in sig.items()}


def loglog_slopes(sigmas_by_n):
    n = sorted(sigmas_by_n); s = [sigmas_by_n[k] for k in n]
    out = {f"{a}->{b}": float(-np.log(s[i + 1] / s[i]) / np.log(b / a))
           for i, (a, b) in enumerate(zip(n[:-1], n[1:]))}
    out["overall"] = float(-np.polyfit(np.log(n), np.log(s), 1)[0])
    return out


def run(n_real=40, n_mc=300, seed=42, out="results", sats=(3, 6, 12)):
    res = {"settings": dict(n_real=n_real, n_mc=n_mc, seed=seed)}
    for label, contig in (("centroid_all_lags (paper)", False), ("centroid_contiguous", True)):
        cfg = SimConfig(n_mc=n_mc, centroid_contiguous=contig)
        block = {}
        for n in sats:
            s = run_configuration(n, n_real, cfg, "ideal", seed_base=seed)
            block[n] = dict(frac_lag_err=s["frac_lag_err"],
                            **accuracy_stats(s, cfg.true_lag))
            a = block[n]
            print(f"[{label}] N={n:>2}: frac err {100*s['frac_lag_err']:.1f}%  bias {a['bias']:+.2f} d "
                  f"(95% CI {a['bias_ci95'][0]:+.2f}..{a['bias_ci95'][1]:+.2f})  RMSE {a['rmse']:.2f} d  "
                  f"1-sigma coverage {100*a['coverage_1sigma']:.0f}%  gross failures {100*a['gross_failure_rate']:.0f}%", flush=True)
        block["loglog_slopes_of_sigma"] = loglog_slopes({n: block[n]["mean_sigma"] for n in sats})
        print(f"[{label}] sigma ~ N^-alpha:", {k: round(v, 2) for k, v in block["loglog_slopes_of_sigma"].items()})
        res[label] = block
    res["binning_test"] = binned_test(n_real, n_mc, seed)
    for k, v in res["binning_test"].items():
        print(f"[binning] {k:<34} mean sigma {v['mean_sigma']:.2f} +/- {v['sem']:.2f} d (n={v['n']})")
    Path(out).mkdir(exist_ok=True)
    json.dump(res, open(Path(out) / "diagnostics.json", "w"), indent=2)
    return res
