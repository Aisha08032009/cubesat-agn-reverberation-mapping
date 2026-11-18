"""Low-cost exploratory sensitivity check. Not a reproduction of paper Tables 2-3."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation_core import run_configuration

OUT=Path('results/sensitivity_smoke');OUT.mkdir(parents=True,exist_ok=True)
N_REALIZATIONS=6
N_FRRSS=100
rows=[]
for parameter,vals in [('true_lag_days',[5,10,20]),('photometric_noise',[.03,.06,.10])]:
    for v in vals:
        for nsat in (3,6,12):
            kw={'tau_true':v} if parameter=='true_lag_days' else {'flux_err_frac':v}
            r=run_configuration(nsat,2/nsat,n_realizations=N_REALIZATIONS,n_frrss=N_FRRSS,
                                seed_base=5000+int(100*v),**kw)
            row={'parameter':parameter,'value':v,'satellites':nsat,
                 'n_realizations':N_REALIZATIONS,'fr_rss_trials':N_FRRSS,
                 'lag_uncertainty_percent':100*r['frac_err_lag']}
            rows.append(row)
            print(row,flush=True)
with open(OUT/'sensitivity_results.csv','w',newline='') as f:
    writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
for parameter,filename in [('true_lag_days','lag_sensitivity.png'),('photometric_noise','noise_sensitivity.png')]:
    fig,ax=plt.subplots(figsize=(7,4.5))
    for value in sorted({r['value'] for r in rows if r['parameter']==parameter}):
        rr=[r for r in rows if r['parameter']==parameter and r['value']==value]
        ax.plot([r['satellites'] for r in rr], [r['lag_uncertainty_percent'] for r in rr], 'o-', label=f'{value:g}')
    ax.set_xticks([3,6,12]);ax.set_xlabel('Satellites')
    ax.set_ylabel('Fractional lag uncertainty (%)')
    ax.set_title(f'Sensitivity smoke test: {parameter} (n=6; MC=100)')
    ax.legend(title=parameter);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/filename,dpi=150);plt.close(fig)
