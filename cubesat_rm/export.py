"""Write simulated light curves as plain text so external lag estimators (JAVELIN, PyROA, ...) can be run.

Format: one file per light curve, columns  time[d]  flux  error.
Usage: python -m cubesat_rm.export --sats 12 --seed 1 --out export/
"""
import argparse
from pathlib import Path
import numpy as np
from .config import SimConfig
from .drw import generate_light_curves, sample_light_curves
from .orbit import observation_times


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sats", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="export")
    a = ap.parse_args(); cfg = SimConfig(); rng = np.random.default_rng(a.seed)
    t, c, l = generate_light_curves(rng, cfg)
    obs = observation_times(a.sats, "ideal", baseline_days=cfg.baseline, base_cadence=cfg.base_cadence)
    cont, line = sample_light_curves(t, c, l, obs, cfg, rng, cfg.base_cadence / a.sats)
    out = Path(a.out); out.mkdir(exist_ok=True)
    for name, (tt, ff, ee) in (("continuum", cont), ("line", line)):
        np.savetxt(out / f"{name}_N{a.sats}_seed{a.seed}.txt", np.column_stack([tt, ff, ee]), header="time_d flux error")
    print("wrote", out)


if __name__ == "__main__":
    main()
