"""Fill paper/main.tex.in from results/*.json so every number in the manuscript comes from the code.

Usage (from the repo root):  python paper/make_paper.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cubesat_rm.orbit import compute_visibility
from cubesat_rm.config import OrbitConfig

R = ROOT / "results"
S = ("3", "6", "12")
B = json.load(open(R / "baseline.json"))
Bs, bw = B["summaries"], B["welch"]
O = json.load(open(R / "orbital.json"))["results"]
O1 = json.load(open(R / "orbital_single.json"))
SENS = json.load(open(R / "sensitivity.json"))
D = json.load(open(R / "diagnostics.json"))
FX = json.load(open(R / "figure_examples.json"))

pct = lambda x, d=1: f"{100 * x:.{d}f}\\%"
pf = lambda p: "p<0.0001" if p < 1e-4 else f"p={p:.4f}"
K = {}

for n in S:
    r = Bs[n]
    K[f"B{n}_LAG"] = f"{r['lag_mean']:.2f}"; K[f"B{n}_SIG"] = f"{r['lag_sigma']:.2f}"
    K[f"B{n}_SEM"] = f"{r['sem']:.3f}"; K[f"B{n}_FL"] = pct(r["frac_lag_err"]); K[f"B{n}_FM"] = pct(r["frac_mass_err"])
    K[f"B{n}_ST"] = pct(r["lag_sigma"] / 10.0); K[f"B{n}_CAD"] = f"{r['cadence_days']:.3f}"
    K[f"B{n}_OFF"] = f"{r['lag_mean'] - 10:+.2f}"
    K[f"B{n}_NEP"] = str(r["n_epochs"])
K["PTS36"] = f"{100 * (Bs['3']['frac_lag_err'] - Bs['6']['frac_lag_err']):.1f}"
K["PTS612"] = f"{100 * (Bs['6']['frac_lag_err'] - Bs['12']['frac_lag_err']):.1f}"
K["PTT36"] = f"{100 * (Bs['3']['lag_sigma'] - Bs['6']['lag_sigma']) / 10:.1f}"
K["PTT612"] = f"{100 * (Bs['6']['lag_sigma'] - Bs['12']['lag_sigma']) / 10:.1f}"
K["RAT36"] = f"{Bs['6']['frac_lag_err'] / Bs['3']['frac_lag_err']:.2f}"
K["RAT612"] = f"{Bs['12']['frac_lag_err'] / Bs['6']['frac_lag_err']:.2f}"
sg = [Bs[n]["lag_sigma"] for n in S]
K["SL36"] = f"{np.log2(sg[0] / sg[1]):.2f}"; K["SL612"] = f"{np.log2(sg[1] / sg[2]):.2f}"
for k, name in (("3v6", "W36"), ("6v12", "W612"), ("3v12", "W312")):
    K[name] = f"$t={bw[k]['t']:.2f}$, ${pf(bw[k]['p'])}$"

K["TABLE1_ROWS"] = "\n".join(
    f"{n} & {K[f'B{n}_CAD']} & {K[f'B{n}_LAG']} & {K[f'B{n}_SIG']} & {K[f'B{n}_SEM']} & {K[f'B{n}_FL']} & {K[f'B{n}_FM']} \\\\" for n in S)

def sens_rows(block, label):
    rows = []
    for key, v in sorted(block.items(), key=lambda kv: float(kv[0])):
        cells = " & ".join(f"{pct(v[n]['frac_lag_err'])} ({v[n]['lag_mean']:.1f}\\,d)" for n in S)
        rows.append(f"{label(key)} & {cells} \\\\")
    return "\n".join(rows)
K["TABLE2_ROWS"] = sens_rows(SENS["true_lag"], lambda k: f"{k} days")
K["TABLE3_ROWS"] = sens_rows(SENS["noise"], lambda k: f"{k}\\%")
sl = {k: {n: v[n]["frac_lag_err"] for n in S} for k, v in SENS["true_lag"].items()}
sn = {k: {n: v[n]["frac_lag_err"] for n in S} for k, v in SENS["noise"].items()}
K["S_N3"] = pct(sn["3"]["3"]); K["S_N3_6"] = pct(sn["6"]["3"]); K["S_N3_10"] = pct(sn["10"]["3"])
K["S_N6_6"] = pct(sn["6"]["6"]); K["S_N12_6"] = pct(sn["6"]["12"])
K["S_N6_10"] = pct(sn["10"]["6"]); K["S_N12_10"] = pct(sn["10"]["12"])
K["S_N_LAG3_10"] = f"{SENS['noise']['10']['3']['lag_mean']:.1f}"; K["S_N_LAG12_10"] = f"{SENS['noise']['10']['12']['lag_mean']:.1f}"
K["S_N_LAG3_6"] = f"{SENS['noise']['6']['3']['lag_mean']:.1f}"
K["S_L5_LAG12"] = f"{SENS['true_lag']['5']['12']['lag_mean']:.1f}"; K["S_L5_LAG3"] = f"{SENS['true_lag']['5']['3']['lag_mean']:.1f}"

vis = {n: compute_visibility(n, OrbitConfig()) for n in (1, 3, 6, 12)}
rows = []
for n in (1, 3, 6, 12):
    v = vis[n]; o = v.occultation_ok.sum(); f = int(v.full_ok.sum())
    if n == 1:
        occ_err = pct(O1["frac_lag_err"]); full_err = "--- (too sparse)"; ideal = "---"
    else:
        occ_err = pct(O["occultation"][str(n)]["frac_lag_err"]); full_err = pct(O["full"][str(n)]["frac_lag_err"]); ideal = pct(Bs[str(n)]["frac_lag_err"])
    rows.append(f"{n} & {len(v.times)} & {int(o)} ({pct(o / len(v.times))}) & {occ_err} & {f} ({pct(f / len(v.times))}) & {full_err} & {ideal} \\\\")
K["TABLE4_ROWS"] = "\n".join(rows)
for n in S:
    K[f"F{n}_FL"] = pct(O["full"][n]["frac_lag_err"]); K[f"F{n}_LAG"] = f"{O['full'][n]['lag_mean']:.1f}"
    K[f"F{n}_RATE"] = pct(vis[int(n)].full_rate)
K["O1_RATE"] = pct(vis[1].occultation_rate); K["O1_FL"] = pct(O1["frac_lag_err"]); K["O1_LAG"] = f"{O1['lag_mean']:.1f}"
K["F1_N"] = str(int(vis[1].full_ok.sum()))
for a, c, name in (("3", "6", "FW36"), ("6", "12", "FW612"), ("3", "12", "FW312")):
    t, p = stats.ttest_ind(O["full"][a]["sigmas"], O["full"][c]["sigmas"], equal_var=False)
    K[name] = f"$t={abs(t):.2f}$, ${pf(p)}$"
K["SUNFRAC"] = pct(float((~vis[12].sun_ok).mean()), 0)

dg = D["centroid_all_lags (paper)"]; dc = D["centroid_contiguous"]; bt = D["binning_test"]
rows = []
for n in S:
    a = dg[n]
    rows.append(f"{n} & ${a['bias']:+.2f}$ (${a['bias_ci95'][0]:+.2f}$, ${a['bias_ci95'][1]:+.2f}$) & {a['rmse']:.2f} & {pct(a['coverage_1sigma'], 0)} & {pct(a['gross_failure_rate'], 0)} & {pct(dc[n]['frac_lag_err'])} \\\\")
K["TABLE5_ROWS"] = "\n".join(rows)
b3, bb, b12 = bt["3 sats"], bt["12 sats binned to 3-sat cadence"], bt["12 sats"]
K["TABLE6_ROWS"] = "\n".join([
    f"3 satellites (180 epochs, 3\\% noise) & {b3['mean_sigma']:.2f} $\\pm$ {b3['sem']:.2f} \\\\",
    f"12 satellites binned 4:1 to the 3-satellite cadence (180 epochs, 1.5\\% noise) & {bb['mean_sigma']:.2f} $\\pm$ {bb['sem']:.2f} \\\\",
    f"12 satellites (720 epochs, 3\\% noise) & {b12['mean_sigma']:.2f} $\\pm$ {b12['sem']:.2f} \\\\"])
K["BIN3"] = f"{b3['mean_sigma']:.2f}"; K["BINB"] = f"{bb['mean_sigma']:.2f}"; K["BIN12"] = f"{b12['mean_sigma']:.2f}"
K["DG_SLOPE"] = f"{dg['loglog_slopes_of_sigma']['overall']:.2f}"
K["D_NREAL"] = str(D["settings"]["n_real"]); K["D_NMC"] = str(D["settings"]["n_mc"])
K["FIG2_PEAK"] = f"{FX['fig2']['peak_lag']:.1f}"; K["FIG2_CEN"] = f"{FX['fig2']['centroid_lag']:.1f}"; K["FIG2_R"] = f"{FX['fig2']['peak_r']:.2f}"
for n in S:
    K[f"FIG3_{n}"] = f"{FX['fig3'][n]['mean']:.1f}"


# ---------------- mission-realistic analysis (budget, targets, mission, sweeps) ----------------
san = lambda x: x.replace(" ", "")
TG = json.load(open(R / "targets.json")); MI = json.load(open(R / "mission.json"))
SW = json.load(open(R / "sweeps.json")); BU = json.load(open(R / "budget.json"))
K["PRE_RATE"] = f"{abs(BU['meta']['raan_rate_deg_day']):.2f}"; K["PRE_DAYS"] = f"{360 / abs(BU['meta']['raan_rate_deg_day']):.0f}"
K["PASS_MIN"] = f"{BU['meta']['pass_minutes_per_day']:.0f}"; K["CYCLE_S"] = f"{BU['meta']['cycle_s']:.0f}"
rows = []
for D in (10, 15):
    for V in (14, 15, 16, 17):
        r = BU["noise"][f"D{D}_V{V}"]
        t1 = "---" if not np.isfinite(r["t_for_1pct"]) else f"{r['t_for_1pct']:.0f}"
        rows.append(f"{D} & {V} & {pct(r['noise_cont'])} & {pct(r['noise_line'])} & {t1} \\\\")
K["TABLE7_ROWS"] = "\n".join(rows)
for V in (14, 15, 16, 17):
    K[f"BU10_V{V}_C"] = pct(BU["noise"][f"D10_V{V}"]["noise_cont"]); K[f"BU10_V{V}_L"] = pct(BU["noise"][f"D10_V{V}"]["noise_line"])
    K[f"BU10_V{V}_T"] = f"{BU['noise'][f'D10_V{V}']['t_for_1pct']:.0f}"
K["BU15_V17_C"] = pct(BU["noise"]["D15_V17"]["noise_cont"])
rows = []
for n in (1, 12):
    for lab in ("64 kB cutouts", "1 MB windows", "16 MB full frames"):
        d = BU["downlink"][f"N{n}_{lab}"]
        rows.append(f"{n} & {lab} & {d['exposures_per_day']:.0f} & {d['time_limited_per_day']:.0f} & {d['bw_limited_per_day']:.0f} \\\\")
K["TABLE8_ROWS"] = "\n".join(rows)
K["DL1_CUT"] = f"{BU['downlink']['N1_64 kB cutouts']['exposures_per_day']:.0f}"; K["DL1_FULL"] = f"{BU['downlink']['N1_16 MB full frames']['exposures_per_day']:.0f}"
K["DL12_CUT"] = f"{BU['downlink']['N12_64 kB cutouts']['exposures_per_day']:.0f}"; K["DL12_FULL"] = f"{BU['downlink']['N12_16 MB full frames']['exposures_per_day']:.0f}"
K["DL1_TIME"] = f"{BU['downlink']['N1_64 kB cutouts']['time_limited_per_day']:.0f}"

rows = []
for name in sorted(TG, key=lambda k: -TG[k]["ecl_lat"]):
    rec = TG[name]; k = san(name); ex = [x[1] for x in rec["sun"]]; gp = [x[2] for x in rec["sun"]]
    r = rec["runs"]; e = {n: pct(r[str(n)]["frac_lag_err"]) for n in (1, 3, 6, 12)}
    cov = pct(r["12"]["coverage"], 0)
    K[f"TG_{k}_LAT"] = f"${rec['ecl_lat']:+.0f}^{{\\circ}}$"; K[f"TG_{k}_EXMAX"] = f"{max(ex):.0f}"; K[f"TG_{k}_GAP"] = f"{max(gp):.0f}"
    K[f"TG_{k}_COV"] = cov; K[f"TG_{k}_LAG1"] = f"{r['1']['lag_mean']:.1f}"; K[f"TG_{k}_LAG12"] = f"{r['12']['lag_mean']:.1f}"
    for n in (1, 3, 6, 12): K[f"TG_{k}_E{n}"] = e[n]
    rows.append(f"{name} & ${rec['ecl_lat']:+.0f}^{{\\circ}}$ & {min(ex):.0f} / {np.mean(ex):.0f} / {max(ex):.0f} & {max(gp):.0f} & {cov} & {e[1]} & {e[3]} & {e[6]} & {e[12]} \\\\")
K["TABLE9_ROWS"] = "\n".join(rows)
Kc = TG["NGC 5548"]["runs"]
conds = [(15, 0.04, Kc), (16, 0.04, MI["NGC 5548:16:0.04"]), (17, 0.04, MI["NGC 5548:17:0.04"]),
         (15, 0.01, MI["NGC 5548:15:0.01"]), (15, 0.10, MI["NGC 5548:15:0.10"]), (17, 0.10, MI["NGC 5548:17:0.10"])]
rows = []
for V, a, r in conds:
    tk = f"V{V}_A{int(round(a * 100)):02d}"
    for n in (1, 3, 6, 12):
        K[f"MI_{tk}_E{n}"] = pct(r[str(n)]["frac_lag_err"]); K[f"MI_{tk}_L{n}"] = f"{r[str(n)]['lag_mean']:.1f}"
    K[f"MI_{tk}_NZ1"] = pct(r["1"]["median_frac_noise_cont"], 2); K[f"MI_{tk}_NZ12"] = pct(r["12"]["median_frac_noise_cont"], 2)
    rows.append(f"{V} & {100 * a:.0f}\\% & {pct(r['1']['median_frac_noise_cont'], 2)} / {pct(r['12']['median_frac_noise_cont'], 2)} & "
                + " & ".join(pct(r[str(n)]["frac_lag_err"]) for n in (1, 3, 6, 12)) + " \\\\")
K["TABLE10_ROWS"] = "\n".join(rows)
K["MI_EXPO1"] = f"{Kc['1']['exposures_total'] / 120:.0f}"; K["MI_EXPO12"] = f"{Kc['12']['exposures_total'] / 120:.0f}"
K["MI_COV1"] = pct(Kc["1"]["coverage"], 0); K["MI_COV12"] = pct(Kc["12"]["coverage"], 0)
K["OCC_ALL"] = pct(min(TG[t]["runs"][str(n)]["occultation_free_any"] for t in TG for n in (3, 6, 12)), 1)
sk = lambda key: key.replace(":", "_").replace(".", "p")
for key, v in SW.items():
    for var in ("uniform", "gap"):
        for n in (3, 6, 12):
            K[f"SW_{sk(key)}_{var}_{n}"] = pct(v[var][str(n)]["frac_lag_err"]); K[f"SWL_{sk(key)}_{var}_{n}"] = f"{v[var][str(n)]['lag_mean']:.1f}"
items = [("DRW damping timescale $\\tau_d$ = 20 d", "tau:20"), ("$\\tau_d$ = 50 d (fiducial)", "base"), ("$\\tau_d$ = 100 d", "tau:100"),
         ("$\\tau_d$ = 200 d", "tau:200"), ("Amplitude 1\\%", "amp:0.01"), ("Amplitude 4\\% (fiducial)", "base"), ("Amplitude 10\\%", "amp:0.10"),
         ("Amplitude 15\\%", "amp:0.15"), ("Top-hat, width 4 d (fiducial)", "base"), ("Log-normal, $\\sigma$ = 4 d", "transfer:lognormal:4"),
         ("Log-normal, $\\sigma$ = 8 d", "transfer:lognormal:8")]
rows = []
for lab, key in items:
    u = " / ".join(f"{100 * SW[key]['uniform'][str(n)]['frac_lag_err']:.1f}" for n in (3, 6, 12))
    g = " / ".join(f"{100 * SW[key]['gap'][str(n)]['frac_lag_err']:.1f}" for n in (3, 6, 12))
    rows.append(f"{lab} & {u} & {g} \\\\")
K["TABLE11_ROWS"] = "\n".join(rows)

tpl = (ROOT / "paper" / "main.tex.in").read_text()
import re
for k, v in K.items():
    tpl = tpl.replace(f"@@{k}@@", v)
left = set(re.findall(r"@@(\w+)@@", tpl))
if left:
    raise SystemExit(f"unfilled tokens: {sorted(left)}")
(ROOT / "paper" / "main.tex").write_text(tpl)
print("wrote paper/main.tex with", len(K), "values")
