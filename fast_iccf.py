"""Optional Numba-accelerated two-way ICCF kernel; reference implementation in iccf.py.

Returns the same Pearson correlation definition as NumPy's corrcoef at each lag.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _one_direction(t1, y1, t2, y2, lag):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(len(t1)):
        q = t1[i] + lag
        if q < t2[0] or q > t2[-1]:
            continue
        j = np.searchsorted(t2, q, side='right')
        if j == 0:
            yi = y2[0]
        elif j >= len(t2):
            yi = y2[-1]
        else:
            a = (q-t2[j-1])/(t2[j]-t2[j-1])
            yi = y2[j-1] + a*(y2[j]-y2[j-1])
        xi = y1[i]
        sx += xi
        sy += yi
        sxx += xi*xi
        syy += yi*yi
        sxy += xi*yi
        n += 1
    if n < 3:
        return np.nan
    vx = sxx - sx*sx/n
    vy = syy - sy*sy/n
    if vx <= 1.e-16 or vy <= 1.e-16:
        return np.nan
    return (sxy-sx*sy/n)/np.sqrt(vx*vy)


@njit(cache=True)
def interp_ccf_fast(t1, y1, t2, y2, lag_grid):
    result = np.empty(len(lag_grid))
    for k in range(len(lag_grid)):
        lag = lag_grid[k]
        r1 = _one_direction(t1,y1,t2,y2,lag)
        r2 = _one_direction(t2,y2,t1,y1,-lag)
        if np.isnan(r1):
            result[k] = r2
        elif np.isnan(r2):
            result[k] = r1
        else:
            result[k] = (r1+r2)/2
    return result
