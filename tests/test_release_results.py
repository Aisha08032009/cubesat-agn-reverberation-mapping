"""Guard the saved release artifacts against accidental file/version mixing."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    return json.loads((ROOT / 'results' / name).read_text())


def test_controlled_results_match_final_manuscript():
    summary = _read('baseline.json')['summaries']
    for satellites, expected_lag, expected_mass in [
        ('3', 41.4, 46.0), ('6', 26.4, 33.1), ('12', 18.5, 27.2)
    ]:
        rec = summary[satellites]
        assert rec['n_real'] == 100
        assert round(100 * rec['frac_lag_err'], 1) == expected_lag
        assert round(100 * rec['frac_mass_err'], 1) == expected_mass


def test_mission_example_and_figures_exist():
    runs = _read('mission.json')['NGC 5548:17:0.04']
    for satellites, expected in [('1', 37.7), ('3', 19.6), ('6', 11.5), ('12', 8.0)]:
        assert round(100 * runs[satellites]['frac_lag_err'], 1) == expected
    for fn in ('fig1_example_light_curves.png', 'fig2_example_ccf.png',
               'fig3_frrss_distributions.png', 'fig4_combined_uncertainty.png',
               'fig9_parameter_sweeps.png', 'fig7_catalogue_targets.png',
               'fig8_mission_photon_budget.png'):
        assert (ROOT / 'figures' / fn).is_file()
    assert (ROOT / 'paper' / 'main.pdf').is_file()
