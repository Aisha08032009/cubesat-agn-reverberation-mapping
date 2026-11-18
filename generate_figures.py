"""Generates figures 1-5 (example light curves, ICCF, FR/RSS distributions, lag and mass uncertainty)."""
import numpy as np
import matplotlib.pyplot as plt
import os
from simulation_core import run_configuration

os.makedirs('figures', exist_ok=True)

BASE_SINGLE_SAT_CADENCE = 2.0
CONSTELLATIONS = {3: BASE_SINGLE_SAT_CADENCE / 3,
                   6: BASE_SINGLE_SAT_CADENCE / 6,
                   12: BASE_SINGLE_SAT_CADENCE / 12}
N_REALIZATIONS = 100
N_FRRSS = 1200

print("Running configurations for figure generation...")
results = {}
for n_sats, cadence in CONSTELLATIONS.items():
    print(f"  {n_sats} satellites...")
    results[n_sats] = run_configuration(n_sats, cadence, n_realizations=N_REALIZATIONS,
                                         n_frrss=N_FRRSS)

MARKERS = {3: 'o', 6: 's', 12: '^'}
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
for ax, n_sats in zip(axes, sorted(results.keys())):
    obs = results[n_sats]['example']['obs']
    ax.plot(obs['t_fine'], obs['continuum_fine'], color='#0072B2', lw=0.8, alpha=0.4)
    ax.plot(obs['t_fine'], obs['line_fine'], color='#D55E00', lw=0.8, alpha=0.4)
    ax.errorbar(obs['t_obs'], obs['cont_obs'], yerr=obs['cont_err'],
                fmt=MARKERS[n_sats], ms=4, color='#0072B2', alpha=0.8,
                label='Observed continuum', markeredgecolor='black', markeredgewidth=0.3)
    ax.errorbar(obs['t_obs'], obs['line_obs'], yerr=obs['line_err'],
                fmt=MARKERS[n_sats], ms=4, color='#D55E00', alpha=0.8,
                label='Observed line', markeredgecolor='black', markeredgewidth=0.3)
    ax.set_ylabel('Normalized flux')
    ax.set_title(f'{n_sats} satellites (cadence = {CONSTELLATIONS[n_sats]:.3f} days, '
                 f'{int(120/CONSTELLATIONS[n_sats])} epochs)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.2)
axes[-1].set_xlabel('Time (days)')
fig.suptitle('Simulated AGN continuum and BLR light curves at 3, 6, and 12-satellite cadence\n'
             '(true lag = 10 days)', y=1.0)
fig.tight_layout()
fig.savefig('figures/fig1_example_light_curves.png', dpi=150)
plt.close(fig)

ex = results[12]['example']
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ex['lag_grid'], ex['ccf'], color='black', lw=1.5)
ax.axvline(ex['peak'], color='#0072B2', ls='--', label=f"Peak lag = {ex['peak']:.1f} d")
ax.axvline(ex['centroid'], color='#D55E00', ls='--', label=f"Centroid lag = {ex['centroid']:.1f} d")
ax.axvline(10.0, color='gray', ls=':', label='True lag = 10.0 d')
ax.set_xlabel('Trial lag (days)')
ax.set_ylabel('Cross-correlation coefficient (r)')
ax.set_title('Example ICCF (12-satellite configuration)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures/fig2_example_ccf.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, n_sats in zip(axes, sorted(results.keys())):
    centroids_mc = results[n_sats]['example']['centroids_mc']
    ax.hist(centroids_mc, bins=25, color='#0072B2', alpha=0.75)
    ax.axvline(10.0, color='black', ls=':', label='True lag')
    ax.axvline(np.mean(centroids_mc), color='#D55E00', ls='--', label='Mean recovered')
    ax.set_title(f'{n_sats} satellites\n(cadence={CONSTELLATIONS[n_sats]:.3f} d)')
    ax.set_xlabel('Recovered lag (days)')
axes[0].set_ylabel('FR/RSS trial count')
axes[0].legend(fontsize=8)
fig.suptitle('FR/RSS Monte Carlo lag distributions by constellation size')
fig.tight_layout()
fig.savefig('figures/fig3_frrss_distributions.png', dpi=150)
plt.close(fig)

n_sats_list = sorted(results.keys())
frac_lag = [results[n]['frac_err_lag'] * 100 for n in n_sats_list]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(n_sats_list, frac_lag, 'o-', color='#0072B2', ms=8, lw=2)
ax.set_xticks(n_sats_list)
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional lag uncertainty (%)')
ax.set_title('Lag recovery uncertainty vs. constellation size')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig4_lag_uncertainty_vs_nsats.png', dpi=150)
plt.close(fig)

frac_mass = [results[n]['frac_err_mass'] * 100 for n in n_sats_list]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(n_sats_list, frac_mass, 's-', color='#D55E00', ms=8, lw=2)
ax.set_xticks(n_sats_list)
ax.set_xlabel('Number of satellites in constellation')
ax.set_ylabel('Fractional black hole mass uncertainty (%)')
ax.set_title('Mass estimate uncertainty vs. constellation size')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig5_mass_uncertainty_vs_nsats.png', dpi=150)
plt.close(fig)

print("\nAll 5 figures saved to the 'figures' folder:")
for f in sorted(os.listdir('figures')):
    print(" -", f)
