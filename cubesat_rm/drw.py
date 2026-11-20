"""Damped Random Walk (DRW) light curves and BLR response (Sec. 2.2)."""
import numpy as np
from .config import SimConfig


def simulate_drw(n, dt, tau, amp, rng, mean=1.0):
    """Exact-update DRW on a uniform grid.

    x_{i+1} = a x_i + sqrt(1-a^2) eps,  a = exp(-dt/tau)  (unit stationary variance)
    flux = mean * (1 + amp * x)  -> asymptotic fractional rms = amp.
    """
    a = np.exp(-dt / tau)
    s = np.sqrt(1.0 - a * a)
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * eps[i]
    return mean * (1.0 + amp * x)


def transfer_kernel(cfg: SimConfig):
    """Discretised transfer function on the fine grid: returns (kernel[shift], max_lag_days)."""
    dt, m = cfg.dt_fine, cfg.true_lag
    if cfg.transfer == "tophat":
        tau = np.arange(m - cfg.tophat_width / 2, m + cfg.tophat_width / 2 + dt / 2, dt)
        w = np.ones_like(tau)
    else:
        sd = cfg.transfer_std
        tau = np.arange(dt, m + 8 * sd + dt, dt)
        if cfg.transfer == "lognormal":
            s2 = np.log(1 + (sd / m) ** 2); mu = np.log(m) - s2 / 2
            w = np.exp(-(np.log(tau) - mu) ** 2 / (2 * s2)) / tau
        elif cfg.transfer == "gamma":
            k, th = (m / sd) ** 2, sd ** 2 / m
            w = tau ** (k - 1) * np.exp(-tau / th)
        else:
            raise ValueError(cfg.transfer)
    w = w / w.sum()
    shifts = np.rint(tau / dt).astype(int)
    kern = np.zeros(shifts.max() + 1)
    np.add.at(kern, shifts, w)
    return kern, shifts.max() * dt


def generate_light_curves(rng, cfg: SimConfig):
    """Continuum + BLR line (continuum convolved with the transfer function) on [0, baseline]."""
    kern, max_lag = transfer_kernel(cfg)
    pad = max_lag + 5.0
    t_full = np.arange(-pad, cfg.baseline + cfg.dt_fine, cfg.dt_fine)
    cont_full = simulate_drw(len(t_full), cfg.dt_fine, cfg.tau_drw, cfg.amp_drw, rng)
    line_full = np.convolve(cont_full, kern)[: len(cont_full)] * cfg.line_scale
    keep = t_full >= 0
    return t_full[keep], cont_full[keep], line_full[keep]


def sample_light_curves(t_fine, cont, line, obs_times, cfg: SimConfig, rng, cadence):
    """Sample both curves at (jittered) observation times with Gaussian noise.

    Returns ((t, f, e) continuum, (t, f, e) line); errors are noise_frac * flux.
    """
    obs = np.asarray(obs_times, float)
    jit = rng.uniform(-cfg.jitter_frac, cfg.jitter_frac, size=obs.size) * cadence
    t = np.sort(np.clip(obs + jit, 0.0, cfg.baseline))
    out = []
    for f in (cont, line):
        true = np.interp(t, t_fine, f)
        err = cfg.noise_frac * np.abs(true)
        out.append((t, true + rng.normal(size=t.size) * err, err))
    return out[0], out[1]
