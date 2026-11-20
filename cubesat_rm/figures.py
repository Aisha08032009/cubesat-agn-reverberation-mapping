"""Generate the figure assets for the manuscript and supplementary analyses."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import SimConfig, OrbitConfig
from .drw import generate_light_curves, sample_light_curves
from .iccf import iccf, peak_and_centroid, fr_rss
from .orbit import compute_visibility, observation_times
from .pipeline import lag_grid

SATS = (3, 6, 12)


def representative_seed(results, n, seed_base=42):
    """Seed of the baseline realization closest to the median in both recovered lag and sigma
    (so the example figures are typical, not best- or worst-case)."""
    b = json.load(open(Path(results) / "baseline.json"))["summaries"][str(n)]
    lag, sig = np.array(b["lags"]), np.array(b["sigmas"])
    assert len(lag) == b["n_real"], "need all realizations valid to map index -> seed"
    seeds = np.random.SeedSequence([seed_base, n, 0]).generate_state(b["n_real"])
    z = lambda x: (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    return int(seeds[int(np.argmin(np.abs(z(lag)) + np.abs(z(sig))))])


def replay(seed, n, cfg=None, n_mc=1200):
    """Re-run one baseline realization exactly as pipeline.run_realization does."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    t, c, l = generate_light_curves(rng, cfg)
    cad = cfg.base_cadence / n
    obs = observation_times(n, "ideal", baseline_days=cfg.baseline, base_cadence=cfg.base_cadence)
    cont, line = sample_light_curves(t, c, l, obs, cfg, rng, cad)
    dist = fr_rss(cont, line, lag_grid(cfg), n_mc, rng, cfg.peak_frac, cfg.centroid_contiguous)
    return t, c, l, cont, line, dist[np.isfinite(dist)]


def fig1_lightcurves(figdir, results="results"):
    cfg = SimConfig()
    seed = representative_seed(results, 12)
    rng = np.random.default_rng(seed)
    t, c, l = generate_light_curves(rng, cfg)          # one underlying realization for all panels
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, n in zip(axes, SATS):
        cad = cfg.base_cadence / n
        obs = np.arange(0, cfg.baseline, cad)
        (tc, fc, ec), (tl, fl, el) = sample_light_curves(t, c, l, obs, cfg, rng, cad)
        ax.plot(t, c, color="C0", lw=1, label="True continuum")
        ax.plot(t, l, color="C1", lw=1, label="True BLR line")
        ax.errorbar(tc, fc, ec, fmt=".", ms=3, color="C0", alpha=.5, label="Observed continuum")
        ax.errorbar(tl, fl, el, fmt=".", ms=3, color="C1", alpha=.5, label="Observed line")
        ax.set_title(f"{n} satellites (cadence = {cad:.3f} days, {len(obs)} epochs)", fontsize=10)
        ax.set_ylabel("Normalized flux"); ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Simulated AGN continuum and BLR line light curves at 3, 6, 12-satellite cadence")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig1_example_light_curves.png", dpi=150); plt.close(fig)
    return dict(seed=seed)


def fig2_iccf(figdir, results="results"):
    cfg = SimConfig()
    seed = representative_seed(results, 12)
    t, c, l, (tc, fc, ec), (tl, fl, el), dist = replay(seed, 12, cfg)
    lags = lag_grid(cfg)
    r = iccf(tc, fc, tl, fl, lags)
    pk, cen = peak_and_centroid(lags, r, cfg.peak_frac)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, r, "k", lw=1)
    ax.axvline(pk, ls="--", color="C0", label=f"Peak lag = {pk:.1f} d")
    ax.axvline(cen, ls="--", color="C1", label=f"Centroid lag = {cen:.1f} d")
    ax.axvline(cfg.true_lag, ls=":", color="k", lw=2, label=f"True lag = {cfg.true_lag:.1f} d")
    ax.set_xlabel("Trial lag (days)"); ax.set_ylabel("Cross-correlation coefficient (r)")
    ax.set_title("Example ICCF (12-satellite configuration)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig2_example_ccf.png", dpi=150); plt.close(fig)
    return dict(seed=seed, peak_lag=float(pk), centroid_lag=float(cen), peak_r=float(np.nanmax(r)))


def fig3_mc(figdir, results="results", n_mc=1200):
    cfg = SimConfig()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    info = {}
    for ax, n in zip(axes, SATS):
        seed = representative_seed(results, n)
        *_, d = replay(seed, n, cfg, n_mc)
        cad = cfg.base_cadence / n
        ax.hist(d, bins=30, color="C0", alpha=.85)
        ax.axvline(cfg.true_lag, color="k", ls=":", label="True lag")
        ax.axvline(d.mean(), color="C1", ls="--", label="Mean recovered")
        ax.set_title(f"{n} satellites\n(cadence={cad:.3f} d)", fontsize=9); ax.set_xlabel("Recovered lag (days)")
        info[n] = dict(seed=seed, mean=float(d.mean()), sigma=float(d.std(ddof=1)))
    axes[0].set_ylabel("FR/RSS trial count"); axes[0].legend(fontsize=7)
    fig.suptitle("FR/RSS Monte Carlo lag distributions by constellation size")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig3_frrss_distributions.png", dpi=150); plt.close(fig)
    return info


def _frac(d, lag=None, noise=None):
    return [100 * d[str(n)]["frac_lag_err"] for n in SATS]


def fig4_sensitivity(results, figdir):
    base = json.load(open(Path(results) / "baseline.json"))["summaries"]
    sens = json.load(open(Path(results) / "sensitivity.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(SATS, [100 * base[str(n)]["frac_lag_err"] for n in SATS], "o-")
    axes[0].set_title("(a) Baseline result\n(true lag = 10 d)")
    for lag, d in sorted(sens["true_lag"].items(), key=lambda kv: float(kv[0])):
        axes[1].plot(SATS, _frac(d), "o-", label=f"True lag = {lag} d")
    axes[1].set_title("(b) Sensitivity to true lag"); axes[1].legend(fontsize=8)
    for nz, d in sorted(sens["noise"].items(), key=lambda kv: float(kv[0])):
        axes[2].plot(SATS, _frac(d), "o-", label=f"Noise = {nz}%")
    axes[2].set_title("(c) Sensitivity to photometric noise"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(SATS); ax.set_xlabel("Number of satellites"); ax.grid(alpha=.3)
    axes[0].set_ylabel("Fractional lag uncertainty (%)")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig4_combined_uncertainty.png", dpi=150); plt.close(fig)


def fig5_validation(results, figdir):
    base = json.load(open(Path(results) / "baseline.json"))["summaries"]
    orb = json.load(open(Path(results) / "orbital.json"))["results"]
    ideal = [100 * base[str(n)]["frac_lag_err"] for n in SATS]
    occ = [100 * orb["occultation"][str(n)]["frac_lag_err"] for n in SATS]
    full = [100 * orb["full"][str(n)]["frac_lag_err"] for n in SATS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(SATS, ideal, "o--", color="gray", label="Idealized uniform cadence")
    axes[0].plot(SATS, occ, "s-", color="C0", label="Earth-occultation-only")
    axes[0].set_title("(a) Idealized vs. Earth-occultation-only")
    axes[1].plot(SATS, ideal, "o--", color="gray", label="Idealized uniform cadence")
    axes[1].plot(SATS, full, "s-", color="C1", label="Full constraints")
    axes[1].set_title("(b) Idealized vs. full mission constraints")
    for ax in axes:
        ax.set_xticks(SATS); ax.set_xlabel("Number of satellites"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    axes[0].set_ylabel("Fractional lag uncertainty (%)")
    fig.suptitle("Validation against realistic orbital visibility")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig5_orbital_validation_combined.png", dpi=150); plt.close(fig)


def fig6_visibility(figdir):
    ocfg = OrbitConfig()
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for ax, n in zip(axes, (1, 12)):
        v = compute_visibility(n, ocfg)
        ax.vlines(v.times[v.full_ok], 0, 1, lw=.6)
        ax.set_ylim(0, 1); ax.set_yticks([0, 1])
        ax.set_ylabel(f"{n} sat(s)\ncan observe")
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Combined observation visibility over the full 120-day campaign (all 4 constraints)")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig6_full_visibility_timeline.png", dpi=150); plt.close(fig)


def make_all(results="results", figdir="figures", seed=None):
    Path(figdir).mkdir(parents=True, exist_ok=True)
    ex = {"fig1": fig1_lightcurves(figdir, results), "fig2": fig2_iccf(figdir, results),
          "fig3": fig3_mc(figdir, results)}
    fig6_visibility(figdir)
    rp = Path(results)
    if (rp / "baseline.json").exists() and (rp / "sensitivity.json").exists():
        fig4_sensitivity(results, figdir)
    if (rp / "baseline.json").exists() and (rp / "orbital.json").exists():
        fig5_validation(results, figdir)
    for fn, files in ((fig7_targets, ["targets.json"]), (fig8_mission, ["targets.json", "mission.json"]), (fig9_sweeps, ["sweeps.json"])):
        if all((rp / f).exists() for f in files):
            fn(results, figdir)
    json.dump(ex, open(rp / "figure_examples.json", "w"), indent=2)
    print("figures written to", figdir)


def fig7_targets(results, figdir):
    """Sun-exclusion by campaign start date and lag uncertainty vs N for the catalogue targets."""
    T = json.load(open(Path(results) / "targets.json"))
    names = sorted(T, key=lambda k: T[k]["ecl_lat"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for n in names:
        st = T[n]["sun"]
        axes[0].plot([s[0] for s in st], [s[1] for s in st], "o-", ms=3, label=f"{n} (β={T[n]['ecl_lat']:+.0f}°)")
    axes[0].set_xlabel("Campaign start (days after 1 Jan 2026)"); axes[0].set_ylabel("Days inside 45° Sun cone (of 120)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3); axes[0].set_title("(a) Solar exclusion by campaign start")
    for n in names:
        r = T[n]["runs"]
        axes[1].plot([1, 3, 6, 12], [100 * r[k]["frac_lag_err"] for k in ("1", "3", "6", "12")], "o-", ms=4, label=n)
    axes[1].set_xscale("log", base=2); axes[1].set_xticks([1, 3, 6, 12]); axes[1].set_xticklabels(["1", "3", "6", "12"])
    axes[1].set_xlabel("Number of satellites"); axes[1].set_ylabel("Fractional lag uncertainty (%)")
    axes[1].set_title("(b) Mission model, 1 Jan 2026 start (V=15)"); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig7_catalogue_targets.png", dpi=150); plt.close(fig)


def fig8_mission(results, figdir):
    M = json.load(open(Path(results) / "mission.json")); T = json.load(open(Path(results) / "targets.json"))
    M["NGC 5548:15:0.04"] = T["NGC 5548"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ls in (("NGC 5548:15:0.04", "-"), ("NGC 5548:16:0.04", "-"), ("NGC 5548:17:0.04", "-")):
        r = M[key]; V = key.split(":")[1]
        axes[0].plot([1, 3, 6, 12], [100 * r[k]["frac_lag_err"] for k in ("1", "3", "6", "12")], "o" + ls, label=f"V = {V}")
    axes[0].set_title("(a) Target brightness (amplitude 4%)")
    for key in ("NGC 5548:15:0.01", "NGC 5548:15:0.04", "NGC 5548:15:0.10"):
        r = M[key]; a = float(key.split(":")[2])
        axes[1].plot([1, 3, 6, 12], [100 * r[k]["frac_lag_err"] for k in ("1", "3", "6", "12")], "o-", label=f"amplitude {100*a:.0f}%")
    axes[1].set_title("(b) Variability amplitude (V = 15)")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks([1, 3, 6, 12]); ax.set_xticklabels(["1", "3", "6", "12"])
        ax.set_xlabel("Number of satellites"); ax.set_ylabel("Fractional lag uncertainty (%)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig8_mission_photon_budget.png", dpi=150); plt.close(fig)


def fig9_sweeps(results, figdir):
    S = json.load(open(Path(results) / "sweeps.json"))
    groups = [("DRW damping timescale", [("tau:20", "20 d"), ("base", "50 d"), ("tau:100", "100 d"), ("tau:200", "200 d")]),
              ("Variability amplitude", [("amp:0.01", "1%"), ("base", "4%"), ("amp:0.10", "10%"), ("amp:0.15", "15%")]),
              ("Transfer function", [("base", "top-hat 4 d"), ("transfer:lognormal:4", "log-normal σ=4 d"), ("transfer:lognormal:8", "log-normal σ=8 d")])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for j, (title, items) in enumerate(groups):
        for i, variant in enumerate(("uniform", "gap")):
            ax = axes[i, j]
            for key, lab in items:
                r = S[key][variant]
                ax.plot([3, 6, 12], [100 * r[k]["frac_lag_err"] for k in ("3", "6", "12")], "o-", ms=4, label=lab)
            ax.set_xscale("log", base=2); ax.set_xticks([3, 6, 12]); ax.set_xticklabels(["3", "6", "12"]); ax.grid(alpha=.3)
            ax.set_title(f"{title}\n({'uniform sampling' if variant == 'uniform' else 'with 36-d Sun gap'})", fontsize=9)
            ax.legend(fontsize=7)
            if i == 1: ax.set_xlabel("Number of satellites")
            if j == 0: ax.set_ylabel("Fractional lag uncertainty (%)")
    fig.tight_layout(); fig.savefig(Path(figdir) / "fig9_parameter_sweeps.png", dpi=150); plt.close(fig)
