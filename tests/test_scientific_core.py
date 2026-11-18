"""Regression tests: analytic properties, numerical equivalence, and representative physics."""
import numpy as np
from drw import simulate_drw, echo_light_curve
from iccf import _interp_ccf, iccf_lag, fr_rss_uncertainty
from orbital_visibility import (orbital_period_days, constellation_visibility,
                                 target_direction_from_ecliptic, is_occulted,
                                 sun_target_separation_deg)


def test_ou_stationary_variance_and_timescale():
    # At dt=2 and tau=20, decorrelation is sufficient for long-run estimation.
    t=np.arange(0,60000,2.0)
    x=simulate_drw(t,tau=20,sf_inf=0.2,seed=24)
    assert abs(np.std(x)-0.2/np.sqrt(2)) < 0.015
    lag_corr=np.corrcoef(x[:-1],x[1:])[0,1]
    assert abs(lag_corr-np.exp(-2/20)) < 0.02


def test_drw_repeatability_and_bad_grid_rejection():
    t=np.linspace(0,120,201)
    assert np.array_equal(simulate_drw(t,seed=7),simulate_drw(t,seed=7))
    try:
        simulate_drw([1,1,2],seed=7)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate time must be rejected')


def test_echo_constant_signal_has_no_artificial_end_dimming():
    t=np.arange(0,120,.1)
    y=echo_light_curve(t,np.ones(len(t)),10,4,scale=.5)
    assert np.allclose(y,.5,atol=1e-12)


def test_fast_and_reference_iccf_are_equivalent():
    from fast_iccf import interp_ccf_fast
    t=np.linspace(0,120,101)
    y=np.sin(t*.3)+np.cos(t*.1)
    t2=np.linspace(0,120,121)
    z=np.interp(t2-10,t,y,left=y[0],right=y[-1])
    lag=np.arange(-5,35,.5)
    ref=_interp_ccf(t,y,t2,z,lag)
    opt=interp_ccf_fast(t,y,t2,z,lag)
    assert np.nanmax(np.abs(ref-opt))<1e-9


def test_iccf_approximately_recovers_known_lag_and_mc_repeats():
    t=np.arange(0,100,.5)
    y=np.sin(.2*t)+.3*np.sin(.7*t)
    z=np.interp(t-7,t,y,left=y[0],right=y[-1])
    pk,centroid,_,_=iccf_lag(t,y,t,z,lag_min=0,lag_max=15,lag_step=.1)
    assert abs(pk-7)<.5
    e=np.full(len(t),.02)
    a=fr_rss_uncertainty(t,y,e,t,z,e,n_sims=12,seed=100)
    b=fr_rss_uncertainty(t,y,e,t,z,e,n_sims=12,seed=100)
    assert np.array_equal(a,b)


def test_orbital_period_and_occultation_geometry():
    assert 94.0 < orbital_period_days(500)*24*60 < 95.0
    u=np.array([1.0,0.0,0.0])
    r=np.array([[-6871.0,0,0],[6871.0,0,0]])
    assert is_occulted(r,u).tolist()==[True,False]
    vis,_=constellation_visibility(np.arange(0,1,.001),3,beta_deg=55)
    assert vis.all()


def test_sky_target_is_unit_vector():
    assert np.isclose(np.linalg.norm(target_direction_from_ecliptic()),1)
    sep=sun_target_separation_deg(np.array([0.0]), target_lon_deg=280.460, target_lat_deg=0)[0]
    assert sep < .1
