"""Command-line interface:  python -m cubesat_rm <command> [options]"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from .config import SimConfig, OrbitConfig
from .pipeline import run_configuration, welch_pairwise


def _progress(label):
    t0 = time.time()
    def cb(k, n):
        if k % max(1, n // 10) == 0 or k == n:
            print(f"  [{label}] {k}/{n} realizations ({time.time()-t0:.0f}s)", flush=True)
    return cb


def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path)


def _table(summaries, title):
    print(f"\n{title}")
    print(f"{'sats':>5} {'cad(d)':>7} {'epochs':>7} {'lag':>7} {'sigma':>7} {'SEM':>6} {'frac lag':>9} {'frac mass':>10}")
    for k in sorted(summaries):
        s = summaries[k]
        print(f"{s['n_sats']:>5} {s['cadence_days']:>7.3f} {s['n_epochs']:>7} {s['lag_mean']:>7.2f} "
              f"{s['lag_sigma']:>7.2f} {s['sem']:>6.3f} {100*s['frac_lag_err']:>8.1f}% {100*s['frac_mass_err']:>9.1f}%")


def cmd_baseline(a):
    cfg = SimConfig(n_mc=a.n_mc, amp_drw=a.amp)
    res = {}
    for n in a.sats:
        res[n] = run_configuration(n, a.n_real, cfg, "ideal", seed_base=a.seed,
                                   n_jobs=a.jobs, progress=_progress(f"{n} sats"))
    _table(res, f"Baseline (true lag {cfg.true_lag} d, {a.n_real} realizations, {a.n_mc} MC trials)")
    tests = welch_pairwise(res)
    for k, v in tests.items():
        print(f"  Welch {k}: t={v['t']:.2f}, p={v['p']:.4g}")
    _dump(dict(summaries=res, welch=tests), Path(a.out) / "baseline.json")


def cmd_sensitivity(a):
    out = {"true_lag": {}, "noise": {}}
    for lag in a.lags:
        cfg = SimConfig(n_mc=a.n_mc, true_lag=float(lag), amp_drw=a.amp)
        out["true_lag"][lag] = {n: run_configuration(n, a.n_real, cfg, "ideal", seed_base=a.seed,
                                n_jobs=a.jobs, progress=_progress(f"lag {lag}d, {n} sats"))
                                for n in a.sats}
        _table(out["true_lag"][lag], f"True lag = {lag} d")
    for nz in a.noise:
        cfg = SimConfig(n_mc=a.n_mc, noise_frac=nz / 100.0, amp_drw=a.amp)
        out["noise"][nz] = {n: run_configuration(n, a.n_real, cfg, "ideal", seed_base=a.seed,
                            n_jobs=a.jobs, progress=_progress(f"noise {nz}%, {n} sats"))
                            for n in a.sats}
        _table(out["noise"][nz], f"Noise = {nz}%")
    _dump(out, Path(a.out) / "sensitivity.json")


def cmd_orbital(a):
    from .orbit import compute_visibility, occulted_fraction, orbital_period_s
    ocfg = OrbitConfig()
    print(f"Orbital period at {ocfg.altitude_km:.0f} km: {orbital_period_s(ocfg.altitude_km)/60:.1f} min")
    print(f"Fraction of orbit occulted (beta={ocfg.beta_deg}): {100*occulted_fraction(ocfg.altitude_km, ocfg.beta_deg):.1f}%")
    print("\nObservation success rates")
    vis_tab = {}
    for n in [1] + list(a.sats):
        v = compute_visibility(n, ocfg)
        vis_tab[n] = dict(attempts=len(v.times), occultation=int(v.occultation_ok.sum()),
                          full=int(v.full_ok.sum()))
        print(f"  N={n:>2}: attempts={len(v.times):>3}  occultation-only={vis_tab[n]['occultation']:>3} "
              f"({100*v.occultation_rate:.1f}%)  full={vis_tab[n]['full']:>3} ({100*v.full_rate:.1f}%)")
    print("\nCoverage across viewing geometries (occultation only):")
    for beta in (10, 30, 55, 70, 90):
        rates = [compute_visibility(n, replace(ocfg, beta_deg=float(beta))).occultation_rate for n in a.sats]
        print(f"  beta={beta:>2}: " + ", ".join(f"N={n}: {100*r:.0f}%" for n, r in zip(a.sats, rates)))
    cfg = SimConfig(n_mc=a.n_mc, amp_drw=a.amp)
    res = {"occultation": {}, "full": {}}
    for mode in ("occultation", "full"):
        for n in a.sats:
            res[mode][n] = run_configuration(n, a.n_real, cfg, mode, ocfg, seed_base=a.seed,
                                             n_jobs=a.jobs, progress=_progress(f"{mode}, {n} sats"))
        _table(res[mode], f"Orbital validation: {mode}")
        for k, v in welch_pairwise(res[mode]).items():
            print(f"  Welch {k}: t={v['t']:.2f}, p={v['p']:.4g}")
    _dump(dict(visibility=vis_tab, results=res), Path(a.out) / "orbital.json")
    single = run_configuration(1, a.n_real, cfg, "occultation", ocfg, seed_base=a.seed, n_jobs=a.jobs,
                               progress=_progress("occultation, 1 sat"))
    _dump(single, Path(a.out) / "orbital_single.json")


def cmd_mc_convergence(a):
    """Stability of the FR/RSS sigma estimate vs number of Monte Carlo trials (footnote 1)."""
    from .drw import generate_light_curves, sample_light_curves
    from .iccf import fr_rss
    from .pipeline import lag_grid
    from .orbit import observation_times
    cfg = SimConfig(amp_drw=a.amp)
    out = {}
    for n in a.sats:
        rng = np.random.default_rng(a.seed + n)
        t, c, l = generate_light_curves(rng, cfg)
        obs = observation_times(n, "ideal", baseline_days=cfg.baseline, base_cadence=cfg.base_cadence)
        cont, line = sample_light_curves(t, c, l, obs, cfg, rng, cfg.base_cadence / n)
        out[n] = {}
        for m in (150, 300, 600, 1200):
            s = [np.nanstd(fr_rss(cont, line, lag_grid(cfg), m, rng)) for _ in range(a.repeats)]
            out[n][m] = dict(mean=float(np.mean(s)), spread=float(np.std(s)))
            print(f"N={n:>2} trials={m:>4}: mean sigma={np.mean(s):.3f}  spread={np.std(s):.3f}")
    _dump(out, Path(a.out) / "mc_convergence.json")


def cmd_diagnostics(a):
    from . import diagnostics
    diagnostics.run(a.n_real, a.n_mc, a.seed, a.out, tuple(a.sats))


def cmd_figures(a):
    from . import figures
    figures.make_all(a.results, a.figdir)


def cmd_all(a):
    cmd_baseline(a); cmd_sensitivity(a); cmd_orbital(a); cmd_diagnostics(a); cmd_figures(a)
    print('Now run: python paper/make_paper.py')


def main(argv=None):
    p = argparse.ArgumentParser(prog="cubesat_rm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, n_real, n_mc):
        sp.add_argument("--sats", type=int, nargs="+", default=[3, 6, 12])
        sp.add_argument("--n-real", type=int, default=n_real)
        sp.add_argument("--n-mc", type=int, default=n_mc)
        sp.add_argument("--jobs", type=int, default=1, help="worker processes")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--amp", type=float, default=SimConfig().amp_drw,
                        help="Stationary DRW fractional RMS amplitude (default: 0.04)")
        sp.add_argument("--out", default="results")
        sp.add_argument("--figdir", default="figures")
        sp.add_argument("--results", default="results")

    sp = sub.add_parser("baseline", help="Sec. 3.1: 3/6/12 satellites, true lag 10 d")
    common(sp, 100, 1200); sp.set_defaults(f=cmd_baseline)
    sp = sub.add_parser("sensitivity", help="Sec. 3.2: true-lag and noise sensitivity")
    common(sp, 30, 1200)
    sp.add_argument("--lags", type=int, nargs="+", default=[5, 10, 20])
    sp.add_argument("--noise", type=int, nargs="+", default=[3, 6, 10])
    sp.set_defaults(f=cmd_sensitivity)
    sp = sub.add_parser("orbital", help="Sec. 3.3: occultation-only and full-constraint validation")
    common(sp, 100, 1200); sp.set_defaults(f=cmd_orbital)
    sp = sub.add_parser("mc-convergence", help="Footnote 1: FR/RSS trial-count stability")
    common(sp, 1, 1200); sp.add_argument("--repeats", type=int, default=15)
    sp.set_defaults(f=cmd_mc_convergence)
    sp = sub.add_parser("diagnostics", help="Bias/RMSE/coverage, centroid definition, binning test, power-law slope")
    common(sp, 40, 300); sp.set_defaults(f=cmd_diagnostics)
    sp = sub.add_parser("figures", help="Regenerate manuscript and supplementary figure assets")
    common(sp, 1, 1); sp.set_defaults(f=cmd_figures)
    sp = sub.add_parser("all", help="Run everything with paper defaults (slow)")
    common(sp, 100, 1200)
    sp.add_argument("--lags", type=int, nargs="+", default=[5, 10, 20])
    sp.add_argument("--noise", type=int, nargs="+", default=[3, 6, 10])
    sp.set_defaults(f=cmd_all)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
