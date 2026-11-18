"""ICCF lag recovery (White & Peterson 1994) with FR/RSS uncertainties (Peterson et al. 1998, 2004)."""
import numpy as np
try:
    from fast_iccf import interp_ccf_fast
except ImportError:
    interp_ccf_fast = None


def _interp_ccf(t1, y1, t2, y2, lag_grid):
    """Two-way interpolated cross-correlation over a grid of trial lags (line lags continuum)."""
    ccf = np.zeros_like(lag_grid, dtype=float)
    for i, lag in enumerate(lag_grid):
        t2_shifted = t1 + lag
        mask1 = (t2_shifted >= t2.min()) & (t2_shifted <= t2.max())
        if mask1.sum() < 3:
            r1 = np.nan
        else:
            y2_interp = np.interp(t2_shifted[mask1], t2, y2)
            r1 = np.corrcoef(y1[mask1], y2_interp)[0, 1]

        t1_shifted = t2 - lag
        mask2 = (t1_shifted >= t1.min()) & (t1_shifted <= t1.max())
        if mask2.sum() < 3:
            r2 = np.nan
        else:
            y1_interp = np.interp(t1_shifted[mask2], t1, y1)
            r2 = np.corrcoef(y2[mask2], y1_interp)[0, 1]

        if np.isnan(r1) and np.isnan(r2):
            ccf[i] = np.nan
        else:
            ccf[i] = np.nanmean([r1, r2])
    return ccf


def iccf_lag(t1, y1, t2, y2, lag_min=-5, lag_max=40, lag_step=0.5, r_threshold=0.8):
    """Return peak lag, centroid lag (r >= r_threshold * r_max), lag grid, and CCF."""
    lag_grid = np.arange(lag_min, lag_max, lag_step)
    # The Numba fast path is cross-tested against the original NumPy ICCF below.
    ccf = (interp_ccf_fast(np.asarray(t1, dtype=float), np.asarray(y1, dtype=float),
                            np.asarray(t2, dtype=float), np.asarray(y2, dtype=float), lag_grid)
           if interp_ccf_fast is not None else _interp_ccf(t1, y1, t2, y2, lag_grid))
    valid = ~np.isnan(ccf)
    lag_grid, ccf = lag_grid[valid], ccf[valid]

    if len(ccf) == 0 or np.all(np.isnan(ccf)):
        return np.nan, np.nan, lag_grid, ccf

    r_max = np.nanmax(ccf)
    peak_lag = lag_grid[np.nanargmax(ccf)]

    thresh = r_threshold * r_max
    above = ccf >= thresh
    if above.sum() == 0:
        centroid_lag = peak_lag
    else:
        centroid_lag = np.sum(lag_grid[above] * ccf[above]) / np.sum(ccf[above])

    return peak_lag, centroid_lag, lag_grid, ccf


def fr_rss_uncertainty(t1, y1, err1, t2, y2, err2, n_sims=200,
                        lag_min=-5, lag_max=40, lag_step=0.5, r_threshold=0.8, seed=None):
    """FR/RSS Monte Carlo: returns the centroid lags from n_sims resampled, flux-randomized trials."""
    rng = np.random.default_rng(seed)
    centroids = np.zeros(n_sims)

    for k in range(n_sims):
        idx1 = np.unique(rng.integers(0, len(t1), len(t1)))
        idx2 = np.unique(rng.integers(0, len(t2), len(t2)))
        if len(idx1) < 4 or len(idx2) < 4:
            centroids[k] = np.nan
            continue

        t1s, y1s, e1s = t1[idx1], y1[idx1], err1[idx1]
        t2s, y2s, e2s = t2[idx2], y2[idx2], err2[idx2]

        y1r = y1s + e1s * rng.standard_normal(len(y1s))
        y2r = y2s + e2s * rng.standard_normal(len(y2s))

        order1 = np.argsort(t1s)
        order2 = np.argsort(t2s)
        _, centroid, _, _ = iccf_lag(t1s[order1], y1r[order1], t2s[order2], y2r[order2],
                                      lag_min, lag_max, lag_step, r_threshold)
        centroids[k] = centroid

    centroids = centroids[~np.isnan(centroids)]
    return centroids
