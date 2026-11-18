"""Stationary Ornstein-Uhlenbeck/DRW simulator and top-hat BLR echo.

Convention: ``sf_inf`` is the asymptotic RMS structure-function amplitude,
SF_inf = sqrt(2) * stationary standard deviation, for a stationary OU process.
For a source with 20% long-term *flux RMS*, use ``sf_inf=0.2*sqrt(2)``.
The original uploaded implementation is archived in original_source/.
"""
import numpy as np


def simulate_drw(t_grid, tau=50.0, sf_inf=0.2, mean=1.0, seed=None):
    """Generate a stationary OU realization on a strictly increasing time grid."""
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1 or len(t_grid) < 2 or not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid must be a finite 1-D array of length >= 2")
    if np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be strictly increasing")
    if tau <= 0 or sf_inf < 0:
        raise ValueError("tau must be positive and sf_inf nonnegative")
    rng = np.random.default_rng(seed)
    stationary_sd = sf_inf / np.sqrt(2.0)
    x = np.empty(len(t_grid))
    x[0] = mean + stationary_sd * rng.standard_normal()
    for i, dt in enumerate(np.diff(t_grid), start=1):
        a = np.exp(-dt / tau)
        x[i] = mean + a*(x[i-1]-mean) + stationary_sd*np.sqrt(-np.expm1(-2*dt/tau))*rng.standard_normal()
    return x


def echo_light_curve(t_grid, continuum, lag, tophat_width, scale=1.0, extra_scatter=0.0, seed=None):
    """Top-hat convolution of a delayed continuum; edge-extended rather than zero-padded.

    NOTE: lack of a simulated pre-campaign history can still bias the first
    ~lag+width/2 days. Simulate a warm-up interval for a mission-grade model.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    if len(t_grid) < 2 or continuum.shape != t_grid.shape or tophat_width <= 0:
        raise ValueError("invalid echo input")
    rng = np.random.default_rng(seed)
    dt = np.median(np.diff(t_grid))
    fine_dt = dt/4.0
    t_fine = np.arange(t_grid[0], t_grid[-1] + fine_dt*0.5, fine_dt)
    cont_fine = np.interp(t_fine, t_grid, continuum)
    n_kernel = max(1, int(round(tophat_width/fine_dt)))
    if n_kernel % 2 == 0:
        n_kernel += 1
    kernel = np.ones(n_kernel, dtype=float)/n_kernel
    shifted = np.interp(t_fine - lag, t_fine, cont_fine,
                        left=cont_fine[0], right=cont_fine[-1])
    pad = n_kernel//2
    smoothed = np.convolve(np.pad(shifted, (pad,pad), mode='edge'), kernel, mode='valid')
    line = np.interp(t_grid, t_fine, smoothed)*scale
    if extra_scatter > 0:
        line += extra_scatter*np.nanstd(line)*rng.standard_normal(len(line))
    return line
