"""Dynamic orbital visibility: J2 nodal precession, real Sun vector, catalogued AGN targets.

All geometry is evaluated in an Earth-centred inertial (equatorial) frame on a fixed time grid:
  * satellite plane precesses at the J2 rate (RAAN drift), N satellites evenly phased in one plane;
  * target direction from catalogue RA/Dec; Sun direction from a low-precision ephemeris (Vallado 2013);
  * constraints: Earth occultation, Sun avoidance cone, eclipse (cylindrical shadow) + thermal recovery.
Downlink is modelled as a bandwidth limit on exposures (see mission.py), not as a fixed buffer time.
"""
from dataclasses import dataclass
import numpy as np
from .orbit import R_EARTH, MU, OMEGA_EARTH, orbital_period_s, sun_ecliptic_longitude_deg

J2 = 1.08263e-3
R_EQ = 6378.137e3
EPS = np.radians(23.4392911)

# Approximate J2000 RA/Dec [deg]; checked against SIMBAD at the rounding precision used here.
TARGETS = {
    "NGC 5548": (214.4979, 25.1367),
    "NGC 4151": (182.6358, 39.4058),
    "NGC 7469": (345.8150, 8.8739),
    "Mrk 509": (311.0404, -10.7236),
    "NGC 3783": (174.7575, -37.7386),
    "Fairall 9": (20.9408, -58.8056),
}


@dataclass
class DynConfig:
    altitude_km: float = 500.0
    inclination_deg: float = 51.6
    raan0_deg: float = 0.0
    precession: bool = True
    station_lat_deg: float = 40.0
    station_lon_deg: float = 0.0
    min_elevation_deg: float = 10.0
    sun_exclusion_deg: float = 45.0
    thermal_recovery_min: float = 10.0
    start_j2000_days: float = 9496.5      # 2026-01-01 00:00 UTC
    baseline_days: float = 120.0
    grid_step_s: float = 60.0


def radec_vec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def ecliptic_lat_deg(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return float(np.degrees(np.arcsin(np.sin(dec) * np.cos(EPS) - np.cos(dec) * np.sin(EPS) * np.sin(ra))))


def sun_vec(d_j2000):
    lam = np.radians(sun_ecliptic_longitude_deg(d_j2000))
    return np.column_stack([np.cos(lam), np.sin(lam) * np.cos(EPS), np.sin(lam) * np.sin(EPS)])


def raan_rate_deg_per_day(alt_km, inc_deg):
    a = R_EARTH + alt_km * 1e3
    n = np.sqrt(MU / a ** 3)
    return float(np.degrees(-1.5 * J2 * (R_EQ / a) ** 2 * n * np.cos(np.radians(inc_deg))) * 86400)


def _blocked(r, u):
    """Earth blocks the sightline from r (T,3) along u ((3,) or (T,3))."""
    d = np.sum(r * u, axis=1)
    perp2 = np.sum(r * r, axis=1) - d ** 2
    return (d < 0) & (perp2 < R_EARTH ** 2)


@dataclass
class DynVis:
    n_sats: int
    step_s: float
    can_observe: np.ndarray    # (N, T) bool: all constraints satisfied for that satellite
    in_pass: np.ndarray        # (N, T) bool: ground station in view
    occ_ok: np.ndarray         # (N, T)
    sun_ok: np.ndarray         # (T,)

    @property
    def t_days(self):
        return np.arange(self.can_observe.shape[1]) * self.step_s / 86400.0


def compute_dynamic(n_sats, target, cfg: DynConfig = None) -> DynVis:
    cfg = cfg or DynConfig()
    ra, dec = TARGETS[target] if isinstance(target, str) else target
    u = radec_vec(ra, dec)
    a = R_EARTH + cfg.altitude_km * 1e3
    n_mot = np.sqrt(MU / a ** 3)
    inc = np.radians(cfg.inclination_deg)
    rate = np.radians(raan_rate_deg_per_day(cfg.altitude_km, cfg.inclination_deg)) / 86400.0 if cfg.precession else 0.0
    t_s = np.arange(0.0, cfg.baseline_days * 86400.0, cfg.grid_step_s)
    d_j = cfg.start_j2000_days + t_s / 86400.0
    sun = sun_vec(d_j)
    sun_ok = np.degrees(np.arccos(np.clip(sun @ u, -1, 1))) >= cfg.sun_exclusion_deg
    Om = np.radians(cfg.raan0_deg) + rate * t_s
    cO, sO, ci, si = np.cos(Om), np.sin(Om), np.cos(inc), np.sin(inc)
    lat, lon = np.radians(cfg.station_lat_deg), np.radians(cfg.station_lon_deg)
    gmst = np.radians((280.46061837 + 360.98564736629 * d_j) % 360.0)
    phi = gmst + lon
    st_hat = np.column_stack([np.cos(lat) * np.cos(phi), np.cos(lat) * np.sin(phi), np.full_like(phi, np.sin(lat))])
    st = R_EARTH * st_hat
    sin_min = np.sin(np.radians(cfg.min_elevation_deg))
    thermal = cfg.thermal_recovery_min * 60.0
    can = np.zeros((n_sats, len(t_s)), bool); inpass = np.zeros_like(can); occ = np.zeros_like(can)
    for i in range(n_sats):
        th = n_mot * t_s + 2 * np.pi * i / n_sats
        c, s_ = np.cos(th), np.sin(th)
        r = a * np.column_stack([cO * c - sO * ci * s_, sO * c + cO * ci * s_, si * s_])
        occ_ok = ~_blocked(r, u)
        shadow = _blocked(r, sun)
        ex = np.zeros(len(t_s), bool); ex[1:] = shadow[:-1] & ~shadow[1:]
        last_exit = np.maximum.accumulate(np.where(ex, t_s, -np.inf))
        ecl_ok = ~shadow & ((t_s - last_exit) >= thermal)
        rel = r - st
        inpass[i] = np.sum(rel * st_hat, axis=1) / np.linalg.norm(rel, axis=1) >= sin_min
        occ[i] = occ_ok
        can[i] = occ_ok & ecl_ok & sun_ok
    return DynVis(n_sats, cfg.grid_step_s, can, inpass, occ, sun_ok)


def sun_exclusion_stats(target, cfg: DynConfig = None, starts_days=range(0, 360, 30)):
    """Days excluded by the Sun cone and the longest excluded gap, for 120-d campaigns starting every 30 d."""
    cfg = cfg or DynConfig()
    ra, dec = TARGETS[target] if isinstance(target, str) else target
    u = radec_vec(ra, dec)
    t = np.arange(0, cfg.baseline_days, 0.1)
    out = []
    for s0 in starts_days:
        sep = np.degrees(np.arccos(np.clip(sun_vec(cfg.start_j2000_days + s0 + t) @ u, -1, 1)))
        bad = sep < cfg.sun_exclusion_deg
        longest = cur = 0
        for b in bad:
            cur = cur + 1 if b else 0; longest = max(longest, cur)
        out.append((s0, float(bad.mean() * cfg.baseline_days), longest * 0.1))
    return out
