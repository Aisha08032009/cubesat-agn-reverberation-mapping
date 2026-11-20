"""Photon budget for a small-aperture CubeSat photometer (replaces the fixed 3 % noise).

Standard CCD-equation model: per-exposure fractional noise
    sigma = sqrt(S + B_sky + B_dark + n_pix * RN^2) / S
with S the target electrons, B_sky the sky electrons in the aperture, etc. All instrument numbers are
ASSUMPTIONS for a representative 6U (10 cm) / 12U (15 cm) photometer, not a specific flight design.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

PH_PER_S_CM2_A_V0 = 1005.0   # photons / s / cm^2 / Angstrom at V = 0 (3.63e-9 erg/s/cm^2/A at 5500 A)


@dataclass
class Instrument:
    aperture_cm: float = 10.0
    obstruction: float = 0.1        # fraction of collecting area lost to obstruction/spiders
    throughput: float = 0.55        # optics + filters + dichroic
    qe: float = 0.80                # detector quantum efficiency
    pixel_arcsec: float = 3.0
    aperture_radius_pix: float = 2.5
    read_noise_e: float = 3.0       # e- / pixel / read
    dark_e_s_pix: float = 0.1       # e- / s / pixel (cooled detector)
    sky_mag_arcsec2: float = 22.0   # zodiacal + stray light, V mag / arcsec^2 (ASSUMPTION)
    band_cont_A: float = 1000.0     # continuum channel width
    band_line_A: float = 150.0      # broad-line (e.g. H-beta) channel width; dichroic -> simultaneous
    t_exp_s: float = 120.0
    overhead_s: float = 60.0        # readout + settle + pointing overhead per epoch

    @property
    def area_cm2(self):
        return np.pi * (self.aperture_cm / 2) ** 2 * (1 - self.obstruction)

    @property
    def n_pix(self):
        return np.pi * self.aperture_radius_pix ** 2


def target_rate(V, inst, band_A):
    return PH_PER_S_CM2_A_V0 * 10 ** (-0.4 * V) * band_A * inst.area_cm2 * inst.throughput * inst.qe


def frac_noise(V, inst, band_A, t_exp=None):
    """Fractional (1-sigma) photometric noise of one exposure."""
    t = inst.t_exp_s if t_exp is None else t_exp
    S = target_rate(V, inst, band_A) * t
    sky_rate = (PH_PER_S_CM2_A_V0 * 10 ** (-0.4 * inst.sky_mag_arcsec2) * band_A * inst.area_cm2
                * inst.throughput * inst.qe * inst.n_pix * inst.pixel_arcsec ** 2)
    var = S + sky_rate * t + inst.dark_e_s_pix * inst.n_pix * t + inst.n_pix * inst.read_noise_e ** 2
    return float(np.sqrt(var) / S)


def exposure_for_noise(V, inst, band_A, target_frac=0.01):
    f = lambda t: frac_noise(V, inst, band_A, t) - target_frac
    if f(1.0) < 0:
        return 1.0
    if f(1e5) > 0:
        return np.inf
    return float(brentq(f, 1.0, 1e5))
