"""Four-constraint orbital visibility analysis (occultation, Sun-avoidance, eclipse, downlink) vs. the idealized baseline."""
import numpy as np
import matplotlib.pyplot as plt
import json
from orbital_visibility import (full_realistic_observation_times, full_constellation_visibility,
                                 orbital_period_days)
from simulation_core import run_configuration_with_times

BASELINE = 120.0
BASE_SINGLE_SAT_CADENCE = 2.0
N_REALIZATIONS = 100
N_FRRSS = 1200  # Corrected: align with reported main Monte Carlo setting

CONSTELLATIONS = [1, 3, 6, 12]
# Idealized baseline values from run_simulation.py (100 realizations, 1200 MC trials).
# Paper values are comparisons only, NOT corrected output. See results from a fresh baseline.
from pathlib import Path
if Path('results.json').exists():
    baseline_data = json.loads(Path('results.json').read_text())
    IDEALIZED_FRAC_ERR = {int(k): float(v['frac_err_lag']) for k,v in baseline_data.items()}
else:
    print('No freshly computed results.json found; baseline comparison will be omitted.')
    IDEALIZED_FRAC_ERR = {}

results = {}
coverage_info = {}

print("=== Full 4-constraint orbital model: occultation + sun-avoidance + eclipse + downlink ===\n")

for n_sats in CONSTELLATIONS:
    cadence = BASE_SINGLE_SAT_CADENCE / n_sats
    t_obs, diag, n_attempt, n_success = full_realistic_observation_times(n_sats, BASELINE, cadence)

    coverage_info[n_sats] = dict(
        cadence=cadence, n_attempt=n_attempt, n_success=n_success,
        success_rate=n_success / n_attempt, **{k: float(v) for k, v in diag.items()},
    )
    print(f"{n_sats} satellite(s): {n_success}/{n_attempt} attempts succeeded "
          f"({100*n_success/n_attempt:.1f}%)")
    print(f"  single-satellite diagnostics: occulted={diag['occulted_frac']*100:.1f}%, "
          f"eclipsed={diag['eclipse_frac']*100:.1f}%, downlink-blocked={diag['downlink_blocked_frac']*100:.1f}%")

    if n_success < 20:
        print(f"  Too few successful observations ({n_success}) for reliable lag recovery -- skipping.\n")
        continue

    r = run_configuration_with_times(n_sats, t_obs, n_realizations=N_REALIZATIONS, n_frrss=N_FRRSS,
                                      jitter_cadence=cadence)
    results[n_sats] = r
    print(f"  -> frac_err_lag (FULL CONSTRAINTS) = {r['frac_err_lag']*100:.1f}%  "
          f"(idealized model: {IDEALIZED_FRAC_ERR.get(n_sats, float('nan'))*100:.1f}%)\n")

from scipy import stats

def welch_t(m1, s1, n1, m2, s2, n2):
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, df, p

print("=== Significance testing (Welch's t-test on per-realization lag uncertainty) ===")
sig_sats = [n for n in results.keys() if results[n]['n_valid_realizations'] >= 10]
for pair in [(3, 6), (6, 12), (3, 12)]:
    if pair[0] not in sig_sats or pair[1] not in sig_sats:
        continue
    r1, r2 = results[pair[0]], results[pair[1]]
    m1 = r1['mean_lag_sigma']
    m2 = r2['mean_lag_sigma']
    s1 = r1['lag_sigma_std_across_realizations']
    s2 = r2['lag_sigma_std_across_realizations']
    n1, n2 = r1['n_valid_realizations'], r2['n_valid_realizations']
    t, df, p = welch_t(m1, s1, n1, m2, s2, n2)
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{pair[0]} vs {pair[1]} satellites: t={t:.2f}, df={df:.1f}, p={p:.4f}  ({sig})")

with open('full_constraints_results.json', 'w') as f:
    json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'example'}
               for k, v in results.items()}, f, indent=2)
with open('full_constraints_coverage.json', 'w') as f:
    json.dump({str(k): v for k, v in coverage_info.items()}, f, indent=2)
print("Saved full_constraints_results.json and full_constraints_coverage.json")

n_sats_list = sorted(n for n in results.keys() if n in (3,6,12))
ideal_sats = [n for n in n_sats_list if n in IDEALIZED_FRAC_ERR]
ideal_vals = [IDEALIZED_FRAC_ERR[n] * 100 for n in ideal_sats]
full_vals = [results[n]['frac_err_lag'] * 100 for n in n_sats_list]

fig, ax = plt.subplots(figsize=(7, 5))
if ideal_sats:
    ax.plot(ideal_sats, ideal_vals, 'o--', color='gray', label='Computed idealized uniform cadence')
ax.plot(n_sats_list, full_vals, 's-', color='#D55E00',
        label='Full constraints (occultation + sun + eclipse + downlink)')
ax.set_xticks(n_sats_list)
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional lag uncertainty (%)')
ax.set_title('Idealized vs. fully-constrained realistic cadence')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig11_full_constraints_comparison.png', dpi=150)
plt.close(fig)

t_fine = np.arange(0, BASELINE, 0.01)
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for ax, n_sats in zip(axes, [1, 12]):
    vis, diag = full_constellation_visibility(t_fine, n_sats)
    ax.fill_between(t_fine, 0, vis.astype(int), step='post', color='#0072B2', alpha=0.6)
    ax.set_ylabel(f'{n_sats} sat(s)\ncan observe')
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
axes[-1].set_xlabel('Time (days)')
fig.suptitle('Combined visibility over full 120-day campaign (all 4 constraints)')
fig.tight_layout()
fig.savefig('figures/fig12_full_visibility_timeline.png', dpi=150)
plt.close(fig)

print("\nSaved figures/fig11_full_constraints_comparison.png and figures/fig12_full_visibility_timeline.png")
