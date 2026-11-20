import numpy as np
from cubesat_rm.config import SimConfig, OrbitConfig
from cubesat_rm.drw import simulate_drw, generate_light_curves
from cubesat_rm.iccf import iccf, peak_and_centroid
from cubesat_rm.mass import bh_mass, fractional_mass_error
from cubesat_rm.orbit import (compute_visibility, occulted_fraction, orbital_period_s,
                              sun_target_separation_deg)


def test_drw_statistics():
    rng = np.random.default_rng(0)
    x = simulate_drw(400000, 0.05, 50.0, 0.2, rng)
    assert abs(x.mean() - 1.0) < 0.05
    assert abs(x.std() - 0.2) < 0.03


def test_line_lags_continuum():
    cfg = SimConfig()
    t, c, l = generate_light_curves(np.random.default_rng(1), cfg)
    lags = np.arange(-5, 30, 0.1)
    r = [np.corrcoef(c[: len(c) - int(s / cfg.dt_fine)], l[int(s / cfg.dt_fine):])[0, 1] for s in lags[lags >= 0]]
    assert abs(lags[lags >= 0][int(np.argmax(r))] - cfg.true_lag) < 1.5


def test_iccf_recovers_known_shift():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 200, 400))
    f = lambda x: np.sin(x / 7.0) + 0.5 * np.sin(x / 3.1)
    lags = np.arange(-5, 40, 0.5)
    r = iccf(t, f(t), t, f(t - 12.0), lags)
    assert abs(peak_and_centroid(lags, r)[0] - 12.0) <= 0.5


def test_orbit_basics():
    assert abs(orbital_period_s(500) / 60 - 94.5) < 0.2
    assert 0.30 < occulted_fraction(500, 55) < 0.40
    assert 0.65 < compute_visibility(1).occultation_rate < 0.72
    for n in (2, 3, 6, 12):
        assert compute_visibility(n).occultation_rate == 1.0


def test_sun_window():
    t = np.arange(0, 120, 0.1)
    blocked = t[sun_target_separation_deg(t, OrbitConfig()) < 45.0]
    assert 33 <= blocked.max() - blocked.min() <= 38
    assert 38 <= blocked.min() <= 44


def test_full_model_not_better_than_occultation():
    for n in (3, 6, 12):
        v = compute_visibility(n)
        assert v.full_rate < v.occultation_rate
        assert v.full_rate < v.sun_ok.mean() + 1e-9


def test_mass_error_propagation():
    assert abs(fractional_mass_error(0.329, 0.10) - 0.385) < 0.002
    assert abs(fractional_mass_error(0.215, 0.10) - 0.293) < 0.002
    assert 1e6 < bh_mass(10.0) < 1e9


def test_transfer_kernels_have_requested_mean_lag():
    import numpy as np
    from cubesat_rm.config import SimConfig
    from cubesat_rm.drw import transfer_kernel
    for tf in ("tophat", "lognormal", "gamma"):
        cfg = SimConfig(transfer=tf, transfer_std=4.0)
        k, _ = transfer_kernel(cfg)
        mean = float(np.sum(np.arange(len(k)) * cfg.dt_fine * k) / k.sum())
        assert abs(mean - 10.0) < 0.2, (tf, mean)


def test_radiometry_monotonic():
    from cubesat_rm.radiometry import Instrument, frac_noise
    inst = Instrument()
    n = [frac_noise(V, inst, 1000.0) for V in (14, 15, 16, 17)]
    assert all(a < b for a, b in zip(n, n[1:]))
    assert frac_noise(15, Instrument(aperture_cm=15), 1000.0) < frac_noise(15, inst, 1000.0)
    assert frac_noise(15, inst, 150.0) > frac_noise(15, inst, 1000.0)


def test_dynamic_orbit():
    from cubesat_rm.orbit_dynamic import raan_rate_deg_per_day, sun_exclusion_stats, compute_dynamic
    assert -5.2 < raan_rate_deg_per_day(500, 51.6) < -4.4
    assert max(x[1] for x in sun_exclusion_stats("Fairall 9")) == 0.0
    assert max(x[1] for x in sun_exclusion_stats("Mrk 509")) > 60
    for n in (2, 3):
        assert compute_dynamic(n, "NGC 5548").occ_ok.any(0).all()
