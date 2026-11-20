"""Mission-level experiment: photon budget + dynamic orbit -> binned epochs with radiometric noise."""
import numpy as np
from .config import SimConfig
from .radiometry import Instrument, frac_noise
from .orbit_dynamic import DynConfig, compute_dynamic, TARGETS
from .pipeline import run_configuration_obs


def exposures_per_bin(vis, inst: Instrument, bin_days=0.5, bytes_per_exposure=64e3, downlink_bps=2e6):
    """Exposures per time bin from observable time, cycle time and ground-station bandwidth."""
    step = vis.step_s; per = int(round(bin_days * 86400 / step)); nb = vis.can_observe.shape[1] // per
    view = lambda a: a[:, : nb * per].reshape(a.shape[0], nb, per).sum(2) * step
    cycle = inst.t_exp_s + inst.overhead_s
    n_time = np.floor(view(vis.can_observe) / cycle)
    n_bw = np.floor(view(vis.in_pass) * downlink_bps / 8.0 / bytes_per_exposure)
    n = np.minimum(n_time, n_bw)
    return n.sum(0), dict(time_limited=float(n_time.sum()), bw_limited=float(n_bw.sum()), used=float(n.sum()))


def run_mission(n_sats, target, V, inst: Instrument, cfg: SimConfig, n_real=30, bin_days=0.5,
                bytes_per_exposure=64e3, downlink_bps=2e6, dyn: DynConfig = None, seed=42):
    vis = compute_dynamic(n_sats, target, dyn)
    n_exp, diag = exposures_per_bin(vis, inst, bin_days, bytes_per_exposure, downlink_bps)
    nb = len(n_exp); centers = (np.arange(nb) + 0.5) * bin_days
    keep = n_exp >= 1
    fc = frac_noise(V, inst, inst.band_cont_A) / np.sqrt(n_exp[keep])
    fl = frac_noise(V, inst, inst.band_line_A) / np.sqrt(n_exp[keep])
    tid = list(TARGETS).index(target) if isinstance(target, str) else 99
    s = run_configuration_obs(centers[keep], fc, fl, n_real, cfg, (seed, n_sats, tid, int(V * 10), int(cfg.amp_drw * 1000)), n_sats)
    s.update(bins_total=int(nb), bins_with_data=int(keep.sum()), coverage=float(keep.mean()),
             mean_exposures_per_bin=float(n_exp[keep].mean()) if keep.any() else 0.0,
             median_frac_noise_cont=float(np.median(fc)) if keep.any() else np.nan,
             occultation_free_any=float(vis.occ_ok.any(0).mean()),
             sun_free=float(vis.sun_ok.mean()), exposures_total=float(n_exp.sum()), **diag)
    return s
