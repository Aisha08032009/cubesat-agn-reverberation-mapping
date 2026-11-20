"""Virial black-hole mass and error propagation (Sec. 2.7, Eq. 1)."""
import numpy as np

C_KMS = 299792.458
G_SI = 6.67430e-11
M_SUN = 1.98847e30
DAY_S = 86400.0


def bh_mass(lag_days, delta_v_kms=3000.0, f=1.4):
    """M_BH = f c tau dV^2 / G  [solar masses]."""
    tau = np.asarray(lag_days) * DAY_S
    c = C_KMS * 1e3
    dv = delta_v_kms * 1e3
    return f * c * tau * dv ** 2 / G_SI / M_SUN


def fractional_mass_error(frac_lag_err, frac_dv_err=0.10):
    """M ∝ tau * dV^2, so sigma_M/M = sqrt(frac_lag^2 + (2 * frac_dv)^2)."""
    return np.sqrt(np.asarray(frac_lag_err) ** 2 + (2.0 * frac_dv_err) ** 2)
