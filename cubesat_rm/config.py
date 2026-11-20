"""All tunable parameters in one place (values taken from the paper, Sec. 2)."""
from dataclasses import dataclass


@dataclass
class SimConfig:
    # --- AGN variability (Sec. 2.2) ---
    tau_drw: float = 50.0          # DRW damping timescale [days]
    # Stationary fractional RMS of the simulated continuum; see manuscript Sec. 2.2.
    amp_drw: float = 0.04
    baseline: float = 120.0        # observing baseline [days]
    true_lag: float = 10.0         # true BLR lag [days]
    tophat_width: float = 4.0      # top-hat transfer-function width [days]
    transfer: str = "tophat"       # 'tophat' | 'lognormal' | 'gamma' (mean lag = true_lag)
    transfer_std: float = 4.0      # std of the lognormal/gamma transfer function [days]
    line_scale: float = 0.5        # mean line flux relative to continuum (cosmetic, cf. Fig. 1)
    dt_fine: float = 0.05          # fine simulation grid step [days]

    # --- Sampling (Sec. 2.3-2.4) ---
    base_cadence: float = 2.0      # single-satellite nominal cadence [days]
    jitter_frac: float = 0.10      # uniform time jitter, as a fraction of the cadence
    noise_frac: float = 0.03       # Gaussian photometric noise, fraction of flux

    # --- Lag recovery (Sec. 2.5-2.6) ---
    lag_min: float = -5.0          # ICCF trial-lag grid [days] (cf. Fig. 2)
    lag_max: float = 40.0
    lag_step: float = 0.5
    peak_frac: float = 0.8         # centroid threshold, fraction of peak r
    centroid_contiguous: bool = False  # True: centroid only over the contiguous region around the peak
    n_mc: int = 1200               # FR/RSS Monte Carlo trials
    min_points: int = 20           # below this the light curve is "too sparse"

    # --- Mass conversion (Sec. 2.7) ---
    virial_f: float = 1.4
    delta_v_kms: float = 3000.0
    delta_v_frac_err: float = 0.10


@dataclass
class OrbitConfig:
    altitude_km: float = 500.0
    beta_deg: float = 55.0         # angle between target direction and orbit-plane pole

    # Sun-avoidance (ASSUMPTIONS flagged in the paper)
    target_ecl_lon_deg: float = 340.0
    target_ecl_lat_deg: float = 42.0
    sun_exclusion_deg: float = 45.0
    campaign_start_j2000_days: float = 9496.5   # days since J2000 at campaign start (~1 Jan 2026)

    # Eclipse / thermal recovery
    thermal_recovery_min: float = 10.0
    sun_beta_deg: float = 60.0     # Sun direction vs orbit pole (not given in paper; ASSUMPTION)
    sun_azimuth_deg: float = 90.0  # Sun azimuth vs target azimuth in orbit frame (ASSUMPTION)

    # Ground-station downlink
    station_lat_deg: float = 40.0
    station_lon_deg: float = 0.0
    min_elevation_deg: float = 10.0
    buffer_hours: float = 6.0
    inclination_deg: float = 51.6  # not given in paper; ASSUMPTION
    grid_step_s: float = 60.0      # time grid for eclipse/downlink bookkeeping
