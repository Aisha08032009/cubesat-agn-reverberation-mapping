"""Reanalysis of the corrected simulation, independent of historical manuscript values.

Examples:
    python run_corrected_experiments.py --mode smoke
    python run_corrected_experiments.py --mode diagnostic
    python run_corrected_experiments.py --mode full

`full` is computationally expensive and uses 100 realizations x 1200 FR/RSS
trials per constellation in both baseline and constrained configurations.
Only data actually computed in the active execution is written as evidence.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from simulation_core import run_configuration, run_configuration_with_times
from orbital_visibility import (orbital_period_days, realistic_observation_times,
                                 full_realistic_observation_times,
                                 target_direction_from_ecliptic)

MODE_SETTINGS = {
    'smoke': dict(n_realizations=4, n_frrss=40),
    'diagnostic': dict(n_realizations=30, n_frrss=300),
    'full': dict(n_realizations=100, n_frrss=1200),
}

def flat(result):
    return {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k,v in result.items() if k != 'example'}

def save_table(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(MODE_SETTINGS), default='smoke')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-orbital', action='store_true', help='skip four-constraint runs')
    a=parser.parse_args()
    config=MODE_SETTINGS[a.mode]
    output=Path(a.output or f'results/{a.mode}')
    output.mkdir(parents=True,exist_ok=True)

    all_rows=[]
    raw={}
    print('DRW convention: SF_inf=0.20 -> stationary sigma=0.20/sqrt(2).',flush=True)
    print('Corrected echo: edge extended; no prehistory burn-in. Full model uses consistent inertial target.',flush=True)
    print('This is REANALYSIS, not an attempt to duplicate the historical manuscript.',flush=True)
    print('Settings:',config,flush=True)
    for n in (3,6,12):
        cad=2.0/n
        print(f'Baseline: N={n}',flush=True)
        r=run_configuration(n,cad,**config)
        raw[f'baseline_{n}']=flat(r)
        all_rows.append(dict(model='idealized', satellites=n, cadence_days=cad, n_attempt=int(120/cad),
                             n_success=int(120/cad), valid_realizations=r['n_valid_realizations'],
                             recovered_lag_days=r['mean_recovered_lag'],
                             mean_lag_sigma_days=r['mean_lag_sigma'],
                             fractional_lag_uncertainty=r['frac_err_lag'],
                             fractional_mass_uncertainty=r['frac_err_mass']))
        print(f"  lag={r['mean_recovered_lag']:.2f} d, lag fraction={r['frac_err_lag']:.2%}",flush=True)

    if not a.no_orbital:
        for n in (3,6,12):
            cad=2.0/n
            print(f'Full four-constraint model: N={n}',flush=True)
            times,diag,n_attempt,n_success=full_realistic_observation_times(n,120,cad,
                                                      consistent_target_geometry=True)
            r=run_configuration_with_times(n,times,jitter_cadence=cad,**config)
            raw[f'full_constraints_{n}']=flat(r)
            raw[f'coverage_{n}']={k:float(v) for k,v in diag.items()} | {'n_attempt':n_attempt,'n_success':n_success}
            all_rows.append(dict(model='four_constraints_consistent_target', satellites=n, cadence_days=cad,
                                 n_attempt=n_attempt,n_success=n_success,
                                 valid_realizations=r['n_valid_realizations'],
                                 recovered_lag_days=r['mean_recovered_lag'],
                                 mean_lag_sigma_days=r['mean_lag_sigma'],
                                 fractional_lag_uncertainty=r['frac_err_lag'],
                                 fractional_mass_uncertainty=r['frac_err_mass']))
            print(f"  coverage={n_success}/{n_attempt} ({n_success/n_attempt:.1%}), lag fraction={r['frac_err_lag']:.2%}",flush=True)

    meta={'status':'new corrected exploratory/reanalysis run; NOT historical manuscript validation',
          'created_utc':datetime.now(timezone.utc).isoformat(),
          'mode':a.mode,'settings':config,
          'original_manuscript_values': 'NOT copied into numerical outputs',
          'model_assumptions':{
              'drw_sf_inf':0.2,'drw_tau_days':50.0,'drw_stationary_sigma':0.2/np.sqrt(2),
              'true_lag_days':10.0,'top_hat_width_days':4.0,
              'noise_fraction':0.03,'campaign_days':120.0,
              'satellite_counts':[3,6,12],
              'cadence_days':'2/N', 'orbit_altitude_km':500,
              'full_target_ecliptic_lon_deg':340,'full_target_ecliptic_lat_deg':42,
              'sun_exclusion_deg':45,'thermal_recovery_minutes':10,
              'buffer_limit_hours':6,'numba_acceleration':'optional; ICCF tested equivalent to NumPy',
          },'important_limitations':['no physically modeled pre-campaign DRW history',
                                   'lag centroid includes all CCF points above threshold',
                                   'mass uncertainty assumes a constant virial factor and assumed BLR line width',
                                   'constellation cadence is an assumed sampling schedule',
                                   'geometry fixed to illustrative position; not an optimized mission'],
          'results':raw}
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    save_table(all_rows,output/'results_table.csv')
    # Presentation-grade figures of calculated outputs only.
    fig,ax=plt.subplots(figsize=(8,5))
    for model,style in [('idealized','o-'),('four_constraints_consistent_target','s--')]:
        rr=[r for r in all_rows if r['model']==model]
        if rr:
            ax.plot([r['satellites'] for r in rr],
                    [100*r['fractional_lag_uncertainty'] for r in rr],style,label=model.replace('_',' '))
    ax.set_xticks([3,6,12]);ax.set_xlabel('Number of satellites')
    ax.set_ylabel('Mean FR/RSS lag uncertainty / mean recovered lag (%)')
    ax.set_title(f'Corrected exploratory simulation ({a.mode}; n={config["n_realizations"]}, MC={config["n_frrss"]})')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/'corrected_lag_uncertainty.png',dpi=160);plt.close(fig)
    if not a.no_orbital:
        fig,ax=plt.subplots(figsize=(7,4.5))
        rr=[r for r in all_rows if r['model']=='four_constraints_consistent_target']
        ax.bar([str(r['satellites']) for r in rr],[100*r['n_success']/r['n_attempt'] for r in rr])
        ax.set_xlabel('Number of satellites');ax.set_ylabel('Successful attempts (%)')
        ax.set_ylim(0,100);ax.set_title('Corrected consistent-target observation coverage')
        fig.tight_layout();fig.savefig(output/'corrected_visibility_coverage.png',dpi=160);plt.close(fig)
    print('Wrote:',output.resolve(),flush=True)

if __name__=='__main__':
    main()
