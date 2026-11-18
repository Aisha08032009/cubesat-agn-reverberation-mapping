"""FR/RSS uncertainty convergence versus number of Monte Carlo trials (150-1200)."""
import numpy as np
import matplotlib.pyplot as plt
from simulation_core import make_one_realization
from iccf import iccf_lag, fr_rss_uncertainty

MC_TRIAL_COUNTS = [150, 300, 600, 1200]
N_REPEATS = 15
CONSTELLATIONS = {3: 2.0/3, 6: 2.0/6, 12: 2.0/12}

results = {}

for n_sats, cadence in CONSTELLATIONS.items():
    print(f"\n{n_sats} satellites:")
    results[n_sats] = {}

    obs = make_one_realization(cadence, seed=1000 * n_sats)
    t, c, l = obs['t_obs'], obs['cont_obs'], obs['line_obs']
    cerr, lerr = obs['cont_err'], obs['line_err']

    for n_trials in MC_TRIAL_COUNTS:
        sigma_estimates = []
        for rep in range(N_REPEATS):
            mc = fr_rss_uncertainty(t, c, cerr, t, l, lerr, n_sims=n_trials,
                                     lag_min=-5, lag_max=40, lag_step=0.5,
                                     seed=2000 * n_sats + 100 * rep + n_trials)
            if len(mc) >= 10:
                sigma_estimates.append(np.std(mc))
        sigma_estimates = np.array(sigma_estimates)
        results[n_sats][n_trials] = sigma_estimates
        print(f"  {n_trials} MC trials: sigma_lag = {sigma_estimates.mean():.3f} "
              f"+/- {sigma_estimates.std():.3f} (across {len(sigma_estimates)} repeats)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = {3: '#0072B2', 6: '#D55E00', 12: '#009E73'}
markers = {3: 'o', 6: 's', 12: '^'}

for ax, n_sats in zip(axes, sorted(results.keys())):
    means = [results[n_sats][n].mean() for n in MC_TRIAL_COUNTS]
    p5 = [np.percentile(results[n_sats][n], 5) for n in MC_TRIAL_COUNTS]
    p95 = [np.percentile(results[n_sats][n], 95) for n in MC_TRIAL_COUNTS]

    ax.plot(MC_TRIAL_COUNTS, means, marker=markers[n_sats], color=colors[n_sats],
            ms=9, lw=2, label=f'{n_sats} satellites')
    ax.fill_between(MC_TRIAL_COUNTS, p5, p95, color=colors[n_sats], alpha=0.2,
                     label='5th-95th percentile')
    ax.set_xscale('log')
    ax.set_xticks(MC_TRIAL_COUNTS)
    ax.set_xticklabels(MC_TRIAL_COUNTS)
    ax.set_xlabel('Number of MC trials')
    ax.set_title(f'{n_sats} satellites')
    ax.grid(alpha=0.3)

axes[0].set_ylabel('Estimated lag uncertainty (days)')
axes[0].legend(fontsize=8)
fig.suptitle('FR/RSS uncertainty estimate convergence vs. number of MC trials\n'
             '(shaded band = spread across 15 independent repeats at each trial count)')
fig.tight_layout()
fig.savefig('figures/fig10_mc_convergence.png', dpi=150)
plt.close(fig)

print("\nSaved figures/fig10_mc_convergence.png")
