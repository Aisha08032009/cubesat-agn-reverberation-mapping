"""Sensitivity of the constellation-size result to true lag and photometric noise."""
import numpy as np
import matplotlib.pyplot as plt
import json
from simulation_core import run_configuration

BASE_SINGLE_SAT_CADENCE = 2.0
CONSTELLATIONS = {3: BASE_SINGLE_SAT_CADENCE / 3,
                   6: BASE_SINGLE_SAT_CADENCE / 6,
                   12: BASE_SINGLE_SAT_CADENCE / 12}
N_REALIZATIONS = 30
N_FRRSS = 1200

LAG_VALUES = [5.0, 10.0, 20.0]

lag_sensitivity_results = {}
print("=== TEST 1: Sensitivity to true BLR lag ===")
for tau_true in LAG_VALUES:
    print(f"\nTrue lag = {tau_true} days:")
    lag_sensitivity_results[tau_true] = {}
    for n_sats, cadence in CONSTELLATIONS.items():
        r = run_configuration(n_sats, cadence, tau_true=tau_true,
                               n_realizations=N_REALIZATIONS, n_frrss=N_FRRSS,
                               seed_base=2000 + int(tau_true) * 10)
        lag_sensitivity_results[tau_true][n_sats] = r
        print(f"  {n_sats} sats: frac_err_lag = {r['frac_err_lag']*100:.1f}%")

NOISE_VALUES = [0.03, 0.06, 0.10]

noise_sensitivity_results = {}
print("\n=== TEST 2: Sensitivity to photometric noise level ===")
for noise in NOISE_VALUES:
    print(f"\nNoise level = {noise*100:.0f}%:")
    noise_sensitivity_results[noise] = {}
    for n_sats, cadence in CONSTELLATIONS.items():
        r = run_configuration(n_sats, cadence, flux_err_frac=noise,
                               n_realizations=N_REALIZATIONS, n_frrss=N_FRRSS,
                               seed_base=3000 + int(noise * 1000))
        noise_sensitivity_results[noise][n_sats] = r
        print(f"  {n_sats} sats: frac_err_lag = {r['frac_err_lag']*100:.1f}%")

def strip_examples(results_dict):
    """Drop heavy example light-curve data before saving to JSON."""
    out = {}
    for key, per_config in results_dict.items():
        out[str(key)] = {str(n): {k: v for k, v in r.items() if k != 'example'}
                          for n, r in per_config.items()}
    return out

with open('sensitivity_results.json', 'w') as f:
    json.dump({'lag_sensitivity': strip_examples(lag_sensitivity_results),
                'noise_sensitivity': strip_examples(noise_sensitivity_results)}, f, indent=2)
print("\nSaved sensitivity_results.json")

fig, ax = plt.subplots(figsize=(7, 5))
colors = ['steelblue', 'indianred', 'seagreen']
for color, tau_true in zip(colors, LAG_VALUES):
    n_sats_list = sorted(lag_sensitivity_results[tau_true].keys())
    frac_err = [lag_sensitivity_results[tau_true][n]['frac_err_lag'] * 100 for n in n_sats_list]
    ax.plot(n_sats_list, frac_err, 'o-', color=color, label=f'True lag = {tau_true:.0f} d')
ax.set_xticks(sorted(CONSTELLATIONS.keys()))
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional lag uncertainty (%)')
ax.set_title('Cadence benefit across the "days to weeks" lag range')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig6_lag_sensitivity.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
for color, noise in zip(colors, NOISE_VALUES):
    n_sats_list = sorted(noise_sensitivity_results[noise].keys())
    frac_err = [noise_sensitivity_results[noise][n]['frac_err_lag'] * 100 for n in n_sats_list]
    ax.plot(n_sats_list, frac_err, 's-', color=color, label=f'Noise = {noise*100:.0f}%')
ax.set_xticks(sorted(CONSTELLATIONS.keys()))
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional lag uncertainty (%)')
ax.set_title('Cadence benefit under different photometric noise levels')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig7_noise_sensitivity.png', dpi=150)
plt.close(fig)

print("\nSaved figures/fig6_lag_sensitivity.png and figures/fig7_noise_sensitivity.png")
