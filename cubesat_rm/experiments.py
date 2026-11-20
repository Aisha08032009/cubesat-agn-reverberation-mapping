"""Experiments added for the mission-realistic analysis (budget, targets, mission, sweeps).

python -m cubesat_rm.experiments budget
python -m cubesat_rm.experiments targets "NGC 5548" "Mrk 509" ...      (V=15, amp=4 %)
python -m cubesat_rm.experiments mission "NGC 5548:17:0.04" ...          (target:V:amp)
python -m cubesat_rm.experiments sweeps tau:20 amp:0.10 transfer:lognormal:4 base
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from .config import SimConfig
from .radiometry import Instrument, frac_noise, exposure_for_noise
from .orbit_dynamic import (TARGETS, DynConfig, compute_dynamic, ecliptic_lat_deg, sun_exclusion_stats,
                            raan_rate_deg_per_day)
from .mission import run_mission, exposures_per_bin
from .orbit import observation_times
from .pipeline import run_configuration

OUT = Path("results")
SATS = (1, 3, 6, 12)


def merge(name, key, val):
    p = OUT / name
    d = json.load(open(p)) if p.exists() else {}
    d[key] = val
    OUT.mkdir(exist_ok=True)
    json.dump(d, open(p, "w"), indent=1)


def budget():
    rows = {}
    for D in (10.0, 15.0):
        inst = Instrument(aperture_cm=D)
        for V in (14, 15, 16, 17):
            rows[f"D{int(D)}_V{V}"] = dict(aperture_cm=D, V=V, noise_cont=frac_noise(V, inst, inst.band_cont_A),
                                          noise_line=frac_noise(V, inst, inst.band_line_A),
                                          t_for_1pct=exposure_for_noise(V, inst, inst.band_cont_A, 0.01))
    inst = Instrument(); dyn = DynConfig(); down = {}
    for n in (1, 12):
        vis = compute_dynamic(n, "NGC 5548", dyn)
        for label, b in (("64 kB cutouts", 64e3), ("1 MB windows", 1e6), ("16 MB full frames", 16e6)):
            ne, dg = exposures_per_bin(vis, inst, 0.5, b, 2e6)
            down[f"N{n}_{label}"] = dict(exposures_per_day=float(ne.sum() / 120), time_limited_per_day=dg["time_limited"] / 120,
                                         bw_limited_per_day=dg["bw_limited"] / 120)
    pass_min = float(compute_dynamic(1, "NGC 5548", dyn).in_pass.mean() * 1440)
    merge("budget.json", "noise", rows); merge("budget.json", "downlink", down)
    merge("budget.json", "meta", dict(pass_minutes_per_day=pass_min, cycle_s=inst.t_exp_s + inst.overhead_s,
                                      raan_rate_deg_day=raan_rate_deg_per_day(500, 51.6), downlink_bps=2e6))
    print(json.dumps(rows, indent=0)[:600]); print(down); print("pass min/day", pass_min)


def targets(names, n_real=30, n_mc=600, V=15.0, amp=0.04):
    cfg = SimConfig(n_mc=n_mc, amp_drw=amp); inst = Instrument()
    for name in names:
        ra, dec = TARGETS[name]
        rec = dict(ecl_lat=ecliptic_lat_deg(ra, dec), sun=sun_exclusion_stats(name), runs={})
        for n in SATS:
            t0 = time.time(); rec["runs"][str(n)] = run_mission(n, name, V, inst, cfg, n_real)
            r = rec["runs"][str(n)]
            print(f"{name:10s} N={n:>2} cov={100*r['coverage']:.0f}% epochs={r['n_epochs']:>3} frac_err={100*r['frac_lag_err']:.1f}% lag={r['lag_mean']:.1f} ({time.time()-t0:.0f}s)", flush=True)
        merge("targets.json", name, rec)


def mission(specs, n_real=30, n_mc=600):
    for sp in specs:
        name, V, amp = sp.split(":"); V = float(V); amp = float(amp)
        cfg = SimConfig(n_mc=n_mc, amp_drw=amp); inst = Instrument()
        runs = {str(n): run_mission(n, name, V, inst, cfg, n_real) for n in SATS}
        for n, r in runs.items():
            print(f"{name} V={V} amp={amp} N={n}: frac_err={100*r['frac_lag_err']:.1f}% lag={r['lag_mean']:.1f} noise={100*r['median_frac_noise_cont']:.2f}%", flush=True)
        merge("mission.json", sp, runs)


def gap_times(n, cfg, lo=41.0, hi=77.0):
    t = observation_times(n, "ideal", baseline_days=cfg.baseline, base_cadence=cfg.base_cadence)
    return t[(t < lo) | (t >= hi)]


def sweeps(specs, n_real=30, n_mc=400):
    for sp in specs:
        parts = sp.split(":"); kind = parts[0]
        cfg = SimConfig(n_mc=n_mc)
        if kind == "tau": cfg = replace(cfg, tau_drw=float(parts[1]))
        elif kind == "amp": cfg = replace(cfg, amp_drw=float(parts[1]))
        elif kind == "transfer": cfg = replace(cfg, transfer=parts[1], transfer_std=float(parts[2]))
        res = {}
        for variant in ("uniform", "gap"):
            res[variant] = {}
            for n in (3, 6, 12):
                obs = gap_times(n, cfg) if variant == "gap" else None
                s = run_configuration(n, n_real, cfg, "ideal", seed_base=42, obs=obs)
                res[variant][str(n)] = {k: v for k, v in s.items() if k not in ("sigmas", "lags", "directs")}
            print(sp, variant, [f"{100*res[variant][str(n)]['frac_lag_err']:.1f}% ({res[variant][str(n)]['lag_mean']:.1f}d)" for n in (3, 6, 12)], flush=True)
        merge("sweeps.json", sp, res)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"budget": lambda: budget(), "targets": lambda: targets(args), "mission": lambda: mission(args),
     "sweeps": lambda: sweeps(args)}[cmd]()
