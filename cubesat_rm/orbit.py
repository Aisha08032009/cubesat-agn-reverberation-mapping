"""Two-body orbital visibility model (Sec. 2.3).

A constellation of N satellites is evenly phased in ONE circular orbital plane. Four
independent constraints are modelled:

1. Earth occultation of the (fixed-direction) target.
2. Sun avoidance (depends on target direction only, so identical for every satellite).
3. Eclipse (cylindrical Earth shadow) + thermal-recovery buffer after eclipse exit.
4. Ground-station downlink: observing allowed only within `buffer_hours` of the last
   successful downlink pass (elevation >= min_elevation at one mid-latitude station).

An attempted observation succeeds only if at least one satellite satisfies all
applicable constraints.
"""
from dataclasses import dataclass
import numpy as np
from .config import OrbitConfig

MU = 3.986004418e14       # m^3 / s^2
R_EARTH = 6371.0e3        # m
OMEGA_EARTH = 7.2921159e-5  # rad / s


@dataclass
class Visibility:
    n_sats: int
    times: np.ndarray            # attempted observation times [days]
    occultation_ok: np.ndarray   # any satellite has a clear view of the target
    sun_ok: np.ndarray           # target outside the Sun-exclusion cone
    eclipse_ok: np.ndarray       # (per-constellation, informational) any sat sunlit & thermally recovered
    downlink_ok: np.ndarray      # (per-constellation, informational) any sat within data buffer
    full_ok: np.ndarray          # all constraints, per-satellite then OR over satellites

    @property
    def occultation_rate(self):
        return float(self.occultation_ok.mean())

    @property
    def full_rate(self):
        return float(self.full_ok.mean())


def orbital_period_s(altitude_km):
    a = R_EARTH + altitude_km * 1e3
    return 2 * np.pi * np.sqrt(a ** 3 / MU)


def _unit(beta_deg, az_deg=0.0):
    b, z = np.radians(beta_deg), np.radians(az_deg)
    return np.array([np.sin(b) * np.cos(z), np.sin(b) * np.sin(z), np.cos(b)])


def blocked_by_earth(r, u):
    """True where Earth blocks the sightline from position r (N,3) along unit vector u."""
    d = r @ u
    perp2 = np.sum(r * r, axis=1) - d ** 2
    return (d < 0) & (perp2 < R_EARTH ** 2)


def occulted_fraction(altitude_km=500.0, beta_deg=55.0, n=20000):
    a = R_EARTH + altitude_km * 1e3
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = a * np.column_stack([np.cos(th), np.sin(th), np.zeros(n)])
    return float(blocked_by_earth(r, _unit(beta_deg)).mean())


def sun_ecliptic_longitude_deg(days_since_j2000):
    """Low-precision solar ephemeris (Vallado 2013, Algorithm 29)."""
    d = days_since_j2000
    M = np.radians(357.5291092 + 0.98560028 * d)
    lam = 280.460 + 0.9856474 * d + 1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M)
    return lam % 360.0


def sun_target_separation_deg(t_days, ocfg: OrbitConfig):
    lam_sun = np.radians(sun_ecliptic_longitude_deg(ocfg.campaign_start_j2000_days + t_days))
    lam_t = np.radians(ocfg.target_ecl_lon_deg)
    b_t = np.radians(ocfg.target_ecl_lat_deg)
    cos_sep = np.cos(b_t) * np.cos(lam_sun - lam_t)   # Sun lies on the ecliptic (lat 0)
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))


def compute_visibility(n_sats, ocfg: OrbitConfig = None, baseline_days=120.0, base_cadence=2.0):
    ocfg = ocfg or OrbitConfig()
    a = R_EARTH + ocfg.altitude_km * 1e3
    period = orbital_period_s(ocfg.altitude_km)
    cadence = base_cadence / n_sats
    t_att = np.arange(0.0, baseline_days, cadence)

    # fine bookkeeping grid
    step = ocfg.grid_step_s
    t_s = np.arange(0.0, baseline_days * 86400.0, step)
    idx = np.clip(np.rint(t_att * 86400.0 / step).astype(int), 0, len(t_s) - 1)

    u_tgt = _unit(ocfg.beta_deg)
    u_sun = _unit(ocfg.sun_beta_deg, ocfg.sun_azimuth_deg)
    inc = np.radians(ocfg.inclination_deg)
    lat = np.radians(ocfg.station_lat_deg)
    lon0 = np.radians(ocfg.station_lon_deg)
    st_hat0 = np.array([np.cos(lat), 0.0, np.sin(lat)])
    st_ang = lon0 + OMEGA_EARTH * t_s
    st_hat = np.column_stack([np.cos(lat) * np.cos(st_ang), np.cos(lat) * np.sin(st_ang),
                              np.full_like(st_ang, np.sin(lat))])
    st_pos = R_EARTH * st_hat
    sin_min_el = np.sin(np.radians(ocfg.min_elevation_deg))
    buffer_s = ocfg.buffer_hours * 3600.0
    thermal_s = ocfg.thermal_recovery_min * 60.0

    occ_any = np.zeros(len(t_att), bool)
    ecl_any = np.zeros(len(t_att), bool)
    dl_any = np.zeros(len(t_att), bool)
    full_any = np.zeros(len(t_att), bool)

    sun_ok = sun_target_separation_deg(t_att, ocfg) >= ocfg.sun_exclusion_deg

    for i in range(n_sats):
        th = 2 * np.pi * t_s / period + 2 * np.pi * i / n_sats
        r_orb = a * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])  # orbit frame
        # 1. target occultation
        occ_ok = ~blocked_by_earth(r_orb, u_tgt)
        # 3. eclipse + thermal recovery
        shadow = blocked_by_earth(r_orb, u_sun)
        exit_mask = np.zeros(len(t_s), bool)
        exit_mask[1:] = shadow[:-1] & ~shadow[1:]
        last_exit = np.maximum.accumulate(np.where(exit_mask, t_s, -np.inf))
        ecl_ok = ~shadow & ((t_s - last_exit) >= thermal_s)
        # 4. downlink (ECI position = orbit frame rotated by inclination about x)
        r_eci = np.column_stack([r_orb[:, 0], r_orb[:, 1] * np.cos(inc), r_orb[:, 1] * np.sin(inc)])
        rel = r_eci - st_pos
        sin_el = np.sum(rel * st_hat, axis=1) / np.linalg.norm(rel, axis=1)
        visible = sin_el >= sin_min_el
        last_dl = np.maximum.accumulate(np.where(visible, t_s, -np.inf))
        last_dl = np.maximum(last_dl, 0.0)           # assume a downlink at campaign start
        dl_ok = (t_s - last_dl) <= buffer_s

        occ_any |= occ_ok[idx]
        ecl_any |= ecl_ok[idx]
        dl_any |= dl_ok[idx]
        full_any |= (occ_ok & ecl_ok & dl_ok)[idx]

    return Visibility(n_sats, t_att, occ_any, sun_ok, ecl_any, dl_any, full_any & sun_ok)


def observation_times(n_sats, mode="ideal", ocfg: OrbitConfig = None, baseline_days=120.0,
                      base_cadence=2.0):
    """Observation times for a constellation.

    mode: 'ideal' (all attempts succeed), 'occultation' (Earth occultation only),
          'full' (all four constraints).
    """
    vis = compute_visibility(n_sats, ocfg, baseline_days, base_cadence)
    if mode == "ideal":
        return vis.times
    if mode == "occultation":
        return vis.times[vis.occultation_ok]
    if mode == "full":
        return vis.times[vis.full_ok]
    raise ValueError(mode)
