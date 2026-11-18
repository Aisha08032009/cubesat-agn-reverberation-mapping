"""Orbital visibility model for a CubeSat constellation observing a fixed-direction target from circular LEO."""
import numpy as np

# Modeling assumptions (not derived from a specific spacecraft): 45 deg Sun-avoidance angle,
# 10 min thermal recovery after eclipse, one 40 deg N ground station, 6 h onboard data buffer.
R_EARTH_KM = 6371.0
MU_EARTH = 398600.4418


def orbital_period_days(altitude_km):
    """Circular orbit period (days) via Kepler's third law."""
    a = R_EARTH_KM + altitude_km
    T_sec = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    return T_sec / 86400.0


def orbit_basis_vectors(inclination_deg, raan_deg):
    """Return in-plane basis vectors P, Q and the orbit normal n."""
    i = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    P = np.array([np.cos(raan), np.sin(raan), 0.0])
    Q = np.array([-np.sin(raan) * np.cos(i), np.cos(raan) * np.cos(i), np.sin(i)])
    n = np.cross(P, Q)
    return P, Q, n


def target_direction(beta_deg, P, n):
    """Fixed target unit vector defined by the beta angle from the orbit pole."""
    b = np.radians(beta_deg)
    return np.cos(b) * n + np.sin(b) * P


def satellite_position(true_anomaly_rad, altitude_km, P, Q):
    """Position vectors (km) on a circular orbit at the given true anomalies."""
    R = R_EARTH_KM + altitude_km
    return R * (np.cos(true_anomaly_rad)[:, None] * P
                + np.sin(true_anomaly_rad)[:, None] * Q)


def is_occulted(r, u_target):
    """True where Earth blocks the line of sight from position r to the target direction."""
    dot_ru = r @ u_target
    t_star = -dot_ru
    r_sq = np.sum(r**2, axis=1)
    perp_dist_sq = r_sq - dot_ru**2
    perp_dist_sq = np.clip(perp_dist_sq, 0, None)
    blocked = (t_star > 0) & (perp_dist_sq < R_EARTH_KM**2)
    return blocked


def constellation_visibility(t_days, n_sats, altitude_km=500.0, inclination_deg=97.4,
                              raan_deg=0.0, beta_deg=55.0):
    """Visibility of the target by at least one of n evenly phased satellites (Earth occultation only)."""
    T = orbital_period_days(altitude_km)
    P, Q, n_orbit = orbit_basis_vectors(inclination_deg, raan_deg)
    u_target = target_direction(beta_deg, P, n_orbit)

    phase_offsets = 2 * np.pi * np.arange(n_sats) / n_sats
    base_anomaly = 2 * np.pi * (t_days / T)

    visible_any = np.zeros(len(t_days), dtype=bool)
    single_sat_blocked = None
    for k, phase in enumerate(phase_offsets):
        true_anomaly = base_anomaly + phase
        r = satellite_position(true_anomaly, altitude_km, P, Q)
        blocked = is_occulted(r, u_target)
        if single_sat_blocked is None:
            single_sat_blocked = blocked.copy()
        visible_any |= ~blocked

    blocked_fraction = np.mean(single_sat_blocked)
    return visible_any, blocked_fraction


def sun_ecliptic_longitude_deg(t_days):
    """Low-precision solar ecliptic longitude (deg) as a function of days since campaign start."""
    d = t_days
    L = 280.460 + 0.9856474 * d
    g = np.radians(357.528 + 0.9856003 * d)
    lam = L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    return lam % 360.0


def sun_direction_eci(t_days, obliquity_deg=23.44):
    """Approximate Sun direction unit vector in the equatorial frame."""
    lam = np.radians(sun_ecliptic_longitude_deg(t_days))
    eps = np.radians(obliquity_deg)
    x = np.cos(lam)
    y = np.sin(lam) * np.cos(eps)
    z = np.sin(lam) * np.sin(eps)
    return np.stack([x, y, z], axis=-1)


def is_eclipsed(r, sun_dir):
    """True where the satellite is in Earth's shadow (cylindrical model)."""
    dot_ru = np.sum(r * sun_dir, axis=-1)
    t_star = -dot_ru
    r_sq = np.sum(r**2, axis=-1)
    perp_dist_sq = np.clip(r_sq - dot_ru**2, 0, None)
    return (t_star > 0) & (perp_dist_sq < R_EARTH_KM**2)


def target_direction_from_ecliptic(lon_deg=340.0, lat_deg=42.0, obliquity_deg=23.44):
    """Return fixed target direction in equatorial/inertial frame from ecliptic lon/lat."""
    lon, lat, eps = map(np.radians, (lon_deg, lat_deg, obliquity_deg))
    xe = np.cos(lat)*np.cos(lon)
    ye = np.cos(lat)*np.sin(lon)
    ze = np.sin(lat)
    return np.array([xe, ye*np.cos(eps)-ze*np.sin(eps), ye*np.sin(eps)+ze*np.cos(eps)])


def sun_target_separation_deg(t_days, target_lon_deg=340.0, target_lat_deg=42.0):
    """Sun-target angular separation (deg) in ecliptic coordinates."""
    lam_sun = np.radians(sun_ecliptic_longitude_deg(t_days))
    lam_t = np.radians(target_lon_deg)
    beta_t = np.radians(target_lat_deg)
    cos_sep = np.cos(beta_t) * np.cos(lam_sun - lam_t)
    cos_sep = np.clip(cos_sep, -1, 1)
    return np.degrees(np.arccos(cos_sep))


def sun_avoidance_ok(t_days, exclusion_angle_deg=45.0, target_lon_deg=340.0, target_lat_deg=42.0):
    """True where the Sun-target separation exceeds the exclusion angle (assumed 45 deg)."""
    sep = sun_target_separation_deg(t_days, target_lon_deg, target_lat_deg)
    return sep > exclusion_angle_deg


OMEGA_EARTH_DEG_PER_DAY = 360.9856235

def ground_station_position(t_days, lat_deg=40.0, lon_deg_at_t0=0.0):
    """Inertial position (km) of a fixed ground station, including Earth rotation."""
    phi = np.radians(lat_deg)
    lam0 = np.radians(lon_deg_at_t0)
    omega = np.radians(OMEGA_EARTH_DEG_PER_DAY)
    lam = lam0 + omega * np.asarray(t_days)
    x = R_EARTH_KM * np.cos(phi) * np.cos(lam)
    y = R_EARTH_KM * np.cos(phi) * np.sin(lam)
    z = R_EARTH_KM * np.sin(phi) * np.ones_like(np.asarray(lam, dtype=float))
    return np.stack([x, y, z], axis=-1)


def ground_station_visible(r_sat, t_days, min_elevation_deg=10.0, lat_deg=40.0, lon_deg_at_t0=0.0):
    """True where the satellite is above the minimum elevation for a downlink pass."""
    r_gs = ground_station_position(t_days, lat_deg, lon_deg_at_t0)
    los = r_sat - r_gs
    los_norm = np.linalg.norm(los, axis=-1)
    zenith_hat = r_gs / np.linalg.norm(r_gs, axis=-1, keepdims=True)
    sin_el = np.sum(los * zenith_hat, axis=-1) / los_norm
    elevation = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    return elevation > min_elevation_deg


def _downlink_buffer_ok(t_days, gs_visible, max_buffer_hours=6.0):
    """True while time since the last downlink is under the onboard buffer limit (assumed 6 h)."""
    max_buffer_days = max_buffer_hours / 24.0
    ok = np.zeros(len(t_days), dtype=bool)
    last_downlink_t = t_days[0] - max_buffer_days
    for i in range(len(t_days)):
        if gs_visible[i]:
            last_downlink_t = t_days[i]
        ok[i] = (t_days[i] - last_downlink_t) < max_buffer_days
    return ok


def full_constellation_visibility(t_days, n_sats, altitude_km=500.0, inclination_deg=97.4,
                                   raan_deg=0.0, beta_deg=55.0,
                                   sun_exclusion_deg=45.0, target_lon_deg=340.0, target_lat_deg=42.0,
                                   eclipse_recovery_hours=10.0 / 60.0,
                                   downlink_max_buffer_hours=6.0,
                                   gs_lat_deg=40.0, gs_lon_deg_at_t0=0.0, gs_min_elevation_deg=10.0,
                                   consistent_target_geometry=True):
    """Combine occultation, Sun-avoidance, eclipse/thermal recovery, and downlink-buffer constraints."""
    P, Q, n_orbit = orbit_basis_vectors(inclination_deg, raan_deg)
    # A single inertial target direction must drive occultation and Sun exclusion.
    # Legacy switch reproduces the uploaded code for comparison with paper tables.
    u_target = (target_direction_from_ecliptic(target_lon_deg, target_lat_deg)
                if consistent_target_geometry else target_direction(beta_deg, P, n_orbit))
    T = orbital_period_days(altitude_km)
    sun_dir = sun_direction_eci(t_days)
    sun_ok = sun_avoidance_ok(t_days, sun_exclusion_deg, target_lon_deg, target_lat_deg)

    phase_offsets = 2 * np.pi * np.arange(n_sats) / n_sats
    base_anomaly = 2 * np.pi * (t_days / T)

    combined_visible = np.zeros(len(t_days), dtype=bool)
    diag = {'occulted_frac': None, 'eclipse_frac': None, 'downlink_blocked_frac': None}

    for k, phase in enumerate(phase_offsets):
        true_anomaly = base_anomaly + phase
        r = satellite_position(true_anomaly, altitude_km, P, Q)

        occulted = is_occulted(r, u_target)
        eclipsed = is_eclipsed(r, sun_dir)

        if eclipse_recovery_hours > 0:
            recovery_steps = int(np.ceil(eclipse_recovery_hours / 24.0 / np.median(np.diff(t_days))))
            eclipsed = eclipsed.copy()
            idx_exit = np.where(np.diff(eclipsed.astype(int)) == -1)[0]
            for i in idx_exit:
                eclipsed[i + 1: i + 1 + recovery_steps] = True

        gs_visible = ground_station_visible(r, t_days, gs_min_elevation_deg, gs_lat_deg, gs_lon_deg_at_t0)
        downlink_ok = _downlink_buffer_ok(t_days, gs_visible, downlink_max_buffer_hours)

        sat_k_can_observe = (~occulted) & (~eclipsed) & downlink_ok
        combined_visible |= sat_k_can_observe

        if k == 0:
            diag['occulted_frac'] = np.mean(occulted)
            diag['eclipse_frac'] = np.mean(eclipsed)
            diag['downlink_blocked_frac'] = np.mean(~downlink_ok)

    combined_visible &= sun_ok
    return combined_visible, diag


def full_realistic_observation_times(n_sats, baseline_days, nominal_cadence_days, fine_dt=0.001, **kwargs):
    """Attempted observation times that succeed under all four constraints."""
    t_fine = np.arange(0, baseline_days, fine_dt)
    visible, diag = full_constellation_visibility(t_fine, n_sats, **kwargs)

    attempt_times = np.arange(0, baseline_days, nominal_cadence_days)
    idx = np.clip(np.searchsorted(t_fine, attempt_times), 0, len(t_fine) - 1)
    succeeded = visible[idx]
    actual_times = attempt_times[succeeded]

    return actual_times, diag, len(attempt_times), len(actual_times)


def realistic_observation_times(n_sats, baseline_days, nominal_cadence_days,
                                  altitude_km=500.0, inclination_deg=97.4,
                                  raan_deg=0.0, beta_deg=55.0, fine_dt=0.001):
    """Attempted observation times that succeed under Earth occultation only."""
    t_fine = np.arange(0, baseline_days, fine_dt)
    visible, blocked_fraction = constellation_visibility(
        t_fine, n_sats, altitude_km, inclination_deg, raan_deg, beta_deg)

    attempt_times = np.arange(0, baseline_days, nominal_cadence_days)
    idx = np.clip(np.searchsorted(t_fine, attempt_times), 0, len(t_fine) - 1)
    succeeded = visible[idx]
    actual_times = attempt_times[succeeded]

    return actual_times, blocked_fraction, len(attempt_times), len(actual_times)
