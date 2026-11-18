"""Compares the idealized uniform cadence to Earth-occultation-only visibility."""
import numpy as np
import matplotlib.pyplot as plt
import json
from orbital_visibility import realistic_observation_times, constellation_visibility, orbital_period_days
from simulation_core import run_configuration_with_times

BASELINE = 120.0
BASE_SINGLE_SAT_CADENCE = 2.0
ALTITUDE_KM = 500.0
INCLINATION_DEG = 97.4
BETA_DEG = 55.0

N_REALIZATIONS = 100
N_FRRSS = 1200

CONSTELLATIONS = [1, 3, 6, 12]

print(f"Orbital period at {ALTITUDE_KM} km: {orbital_period_days(ALTITUDE_KM)*24*60:.1f} minutes\n")

results = {}
coverage_info = {}

for n_sats in CONSTELLATIONS:
    cadence = BASE_SINGLE_SAT_CADENCE / n_sats
    t_obs, blocked_frac, n_attempt, n_success = realistic_observation_times(
        n_sats, BASELINE, cadence, altitude_km=ALTITUDE_KM,
        inclination_deg=INCLINATION_DEG, beta_deg=BETA_DEG)

    coverage_info[n_sats] = dict(
        cadence=cadence, n_attempt=n_attempt, n_success=n_success,
        success_rate=n_success / n_attempt, single_sat_blocked_fraction=blocked_frac,
    )
    print(f"{n_sats} satellites: {n_success}/{n_attempt} attempts succeeded "
          f"({100*n_success/n_attempt:.1f}%), single-satellite blocked fraction = {blocked_frac:.1%}")

    if n_success < 20:
        print(f"  Too few successful observations for {n_sats} satellites -- skipping lag recovery.")
        continue

    r = run_configuration_with_times(n_sats, t_obs, n_realizations=N_REALIZATIONS, n_frrss=N_FRRSS,
                                      jitter_cadence=cadence)
    results[n_sats] = r
    print(f"  -> frac_err_lag (REALISTIC) = {r['frac_err_lag']*100:.1f}%")

print("\n=== Robustness check: coverage across different viewing geometries (beta angle) ===")
for beta in [10, 30, 55, 70, 90]:
    line = f"beta={beta:3d} deg: "
    for n_sats in [1, 2, 3]:
        t_fine = np.arange(0, 1.0, 0.0005)
        vis, blocked = constellation_visibility(t_fine, n_sats, ALTITUDE_KM, INCLINATION_DEG, 0.0, beta)
        line += f" N={n_sats} cov={np.mean(vis)*100:5.1f}% "
    print(line)

with open('orbital_comparison_results.json', 'w') as f:
    json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'example'}
               for k, v in results.items()}, f, indent=2)
with open('orbital_coverage_info.json', 'w') as f:
    json.dump({str(k): v for k, v in coverage_info.items()}, f, indent=2)

# Idealized baseline values from run_simulation.py (100 realizations, 1200 MC trials).
# Paper values shown for reference; they do not follow corrected DRW dynamics.
from pathlib import Path
if Path('results.json').exists():
    baseline_data = json.loads(Path('results.json').read_text())
    idealized_frac_err = {int(k): float(v['frac_err_lag']) for k,v in baseline_data.items()}
else:
    print('No freshly computed results.json found; baseline comparison will be omitted.')
    idealized_frac_err = {}

fig, ax = plt.subplots(figsize=(7, 5))
n_sats_list = sorted(n for n in results.keys() if n in (3,6,12))
ideal_sats = [n for n in n_sats_list if n in idealized_frac_err]
ideal_vals = [idealized_frac_err[n] * 100 for n in ideal_sats]
real_vals = [results[n]['frac_err_lag'] * 100 for n in n_sats_list]

if ideal_sats:
    ax.plot(ideal_sats, ideal_vals, 'o--', color='gray', label='Computed idealized uniform cadence')
ax.plot(n_sats_list, real_vals, 's-', color='steelblue', label='Realistic orbital visibility')
ax.set_xticks(n_sats_list)
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional lag uncertainty (%)')
ax.set_title(f'Idealized vs. realistic cadence (beta={BETA_DEG} deg, {ALTITUDE_KM} km orbit)')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig8_idealized_vs_realistic.png', dpi=150)
plt.close(fig)

T = orbital_period_days(ALTITUDE_KM)
t_short = np.arange(0, T * 6, T / 500)
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, n_sats in zip(axes, [1, 3]):
    vis, _ = constellation_visibility(t_short, n_sats, ALTITUDE_KM, INCLINATION_DEG, 0.0, BETA_DEG)
    ax.fill_between(t_short * 24, 0, vis.astype(int), step='post', color='steelblue', alpha=0.6)
    ax.set_ylabel(f'{n_sats} sat(s)\nvisible')
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
axes[-1].set_xlabel('Time (hours)')
fig.suptitle(f'Earth-occultation visibility over 6 orbits (beta={BETA_DEG} deg)')
fig.tight_layout()
fig.savefig('figures/fig9_visibility_pattern.png', dpi=150)
plt.close(fig)

print("\nSaved figures/fig8_idealized_vs_realistic.png and figures/fig9_visibility_pattern.png")
print("Saved orbital_comparison_results.json and orbital_coverage_info.json")
