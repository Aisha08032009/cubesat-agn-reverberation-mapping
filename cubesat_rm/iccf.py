"""Interpolated Cross-Correlation Function + FR/RSS uncertainties (Sec. 2.5-2.6).

References: White & Peterson (1994); Peterson et al. (1998, 2004).
"""
import numpy as np


def _pearson_masked(x, y, m):
    """Column-wise Pearson r of x,y (2-D) using only entries where m is True."""
    n = m.sum(axis=0).astype(float)
    ok = n >= 5
    n_safe = np.where(ok, n, 1.0)
    xm = np.where(m, x, 0.0)
    ym = np.where(m, y, 0.0)
    mx = xm.sum(0) / n_safe
    my = ym.sum(0) / n_safe
    dx = np.where(m, x - mx, 0.0)
    dy = np.where(m, y - my, 0.0)
    num = (dx * dy).sum(0)
    den = np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~ok | ~np.isfinite(r)] = np.nan
    return r


def iccf(t1, f1, t2, f2, lags):
    """ICCF of curve 2 lagging curve 1 (f2(t + lag) ~ f1(t)).

    For each trial lag: (i) interpolate curve 2 onto the shifted times of curve 1,
    (ii) interpolate curve 1 onto the shifted times of curve 2, correlate both
    ways and average.
    """
    lags = np.asarray(lags, float)
    # direction 1: f1(t1_i) vs f2(t1_i + lag)
    T1 = t1[:, None] + lags[None, :]
    m1 = (T1 >= t2[0]) & (T1 <= t2[-1])
    F2 = np.interp(T1.ravel(), t2, f2).reshape(T1.shape)
    r1 = _pearson_masked(np.broadcast_to(f1[:, None], T1.shape), F2, m1)
    # direction 2: f2(t2_j) vs f1(t2_j - lag)
    T2 = t2[:, None] - lags[None, :]
    m2 = (T2 >= t1[0]) & (T2 <= t1[-1])
    F1 = np.interp(T2.ravel(), t1, f1).reshape(T2.shape)
    r2 = _pearson_masked(np.broadcast_to(f2[:, None], T2.shape), F1, m2)
    return np.nanmean(np.vstack([r1, r2]), axis=0) if np.isfinite(r1).any() or np.isfinite(r2).any() \
        else np.full(len(lags), np.nan)


def peak_and_centroid(lags, r, peak_frac=0.8, contiguous=False):
    """Peak lag and centroid lag (r-weighted mean of lags with r >= peak_frac * r_max).

    contiguous=False: all lags above threshold (as described in the paper).
    contiguous=True : only the contiguous region around the peak (standard PyCCF practice).
    """
    if not np.isfinite(r).any():
        return np.nan, np.nan
    i = int(np.nanargmax(r))
    rmax = r[i]
    if rmax <= 0:
        return lags[i], np.nan
    sel = np.isfinite(r) & (r >= peak_frac * rmax)
    if contiguous:
        lo = hi = i
        while lo > 0 and sel[lo - 1]:
            lo -= 1
        while hi < len(r) - 1 and sel[hi + 1]:
            hi += 1
        keep = np.zeros_like(sel); keep[lo:hi + 1] = True
        sel = keep
    return lags[i], float(np.sum(lags[sel] * r[sel]) / np.sum(r[sel]))


def centroid_lag(t1, f1, t2, f2, lags, peak_frac=0.8, contiguous=False):
    r = iccf(t1, f1, t2, f2, lags)
    return peak_and_centroid(lags, r, peak_frac, contiguous)[1]


def fr_rss(cont, line, lags, n_mc, rng, peak_frac=0.8, contiguous=False):
    """Flux randomisation / random subset sampling (Peterson et al. 1998).

    cont, line: (t, f, err). Each trial resamples each light curve with replacement,
    discards duplicate points, perturbs the retained fluxes by a Gaussian draw from
    their errors, and recomputes the ICCF centroid. Returns the array of centroids.
    """
    tc, fc, ec = cont
    tl, fl, el = line
    out = np.full(n_mc, np.nan)
    for k in range(n_mc):
        ic = np.unique(rng.integers(0, len(tc), len(tc)))
        il = np.unique(rng.integers(0, len(tl), len(tl)))
        fc_k = fc[ic] + rng.normal(size=ic.size) * ec[ic]
        fl_k = fl[il] + rng.normal(size=il.size) * el[il]
        out[k] = centroid_lag(tc[ic], fc_k, tl[il], fl_k, lags, peak_frac, contiguous)
    return out
