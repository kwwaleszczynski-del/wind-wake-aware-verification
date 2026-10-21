import numpy as np
import pytest

from conftest import CURVE, analytic_tables
from wake_aware.freestream import (circular_mean, farm_curve, leave_one_out_direction, leave_one_out_median,
                                   nacelle_offsets, power_from_wind, wind_from_power, wrap180)
from wake_aware.wake import WD_GRID, farm_deficit, pair_tables, smooth_direction


def test_pywake_pair_tables_put_the_wake_downstream():
    ws = np.array([6.0, 8.0, 10.0])
    t = pair_tables([0.0, 400.0], [0.0, 0.0], [80.0, 80.0], 92.0, 2050.0, k_values=[0.04], ws=ws)
    assert t.shape == (1, 2, 2, len(WD_GRID), 3)
    assert t[0, 0, 1, 270, 1] > 0.15  # wind from the west: turbine 2 (east) sits in turbine 1's wake
    assert t[0, 0, 1, 90, 1] == 0.0  # wind from the east: turbine 1 is upstream
    assert t[0, 1, 0, 90, 1] == pytest.approx(t[0, 0, 1, 270, 1])  # symmetric layout, symmetric wake
    assert t[0, 0, 1, 250, 1] < 0.2 * t[0, 0, 1, 270, 1]  # 20 deg off axis the wake is mostly missed
    assert np.all(t[0, 0, 0] == 0) and t[0, 0, 1, 270, 2] < t[0, 0, 1, 270, 0]  # lower thrust, shallower wake


def test_direction_smoothing_keeps_the_integral_and_wraps():
    t = np.zeros((1, 2, 2, 360, 1))
    t[0, 0, 1, 1, 0] = 1.0  # a spike at 1 deg spreads across north into 359 deg
    s = smooth_direction(t, 3.0)
    assert s.sum() == pytest.approx(1.0)
    assert s[0, 0, 1, 359, 0] > 0 and s[0, 0, 1, 359, 0] == pytest.approx(s[0, 0, 1, 3, 0])


def test_farm_deficit_superposition_and_stopped_turbines():
    tables = analytic_tables(3)
    theta, ws = np.array([270.0, 270.0]), np.array([8.0, 8.0])
    running = np.ones((2, 3), dtype=bool)
    running[1, 0] = False  # second timestamp: the westernmost turbine is stopped
    d = farm_deficit(tables, theta, ws, 0.04, running)
    expected = np.sqrt(0.3 ** 2 + 0.15 ** 2)  # third turbine: wakes of the first (2 spacings) and second
    assert d[0, 2] == pytest.approx(expected, rel=1e-5)
    assert d[1, 2] == pytest.approx(0.3, rel=1e-5) and d[1, 1] == 0.0
    assert d[0, 0] == 0.0
    assert np.isnan(farm_deficit(tables, np.array([np.nan]), np.array([8.0]), 0.04)).all()


def test_power_inversion_round_trip_in_partial_load():
    ws = np.array([5.5, 6.5, 9.0, 11.0])
    p = power_from_wind(ws, CURVE)
    assert np.allclose(wind_from_power(p, CURVE), ws, atol=1e-3)
    assert np.isnan(wind_from_power(np.array([10.0, 1990.0]), CURVE)).all()  # below 50 kW / above 1850 kW


def test_farm_curve_is_monotone():
    rng = np.random.default_rng(1)
    ws = rng.uniform(3, 20, 20000)
    p = power_from_wind(ws, CURVE) + rng.normal(0, 30, ws.size)
    c = farm_curve(ws, p, np.ones(ws.size, dtype=bool))
    assert (np.diff(c["power_kw"]) >= 0).all()
    assert c["power_kw"].iloc[-1] == pytest.approx(2000.0, abs=15)


def test_circular_mean_across_north_and_wrap():
    a = np.array([[350.0, 10.0, np.nan]])
    assert wrap180(circular_mean(a, ~np.isnan(a))[0]) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(circular_mean(a, np.zeros_like(a, dtype=bool))[0])
    assert wrap180(350.0) == -10.0 and wrap180(-190.0) == 170.0


def test_leave_one_out_direction_and_offsets():
    rng = np.random.default_rng(2)
    wd = rng.uniform(0, 360, 5000)
    true_offsets = np.array([0.0, 4.0, -3.0, 1.0])
    nac = (wd[:, None] + true_offsets[None, :] + rng.normal(0, 2, (5000, 4))) % 360
    valid = np.ones_like(nac, dtype=bool)
    off = nacelle_offsets(nac, valid)
    assert np.allclose(off - off.mean(), true_offsets - true_offsets.mean(), atol=0.3)
    theta = leave_one_out_direction(nac, valid, off)
    assert np.abs(wrap180(theta[:, 1] - wd - (true_offsets - off).mean())).mean() < 2.0
    # the turbine's own nacelle does not enter its reference direction
    moved = nac.copy()
    moved[:, 1] += 30.0
    assert np.allclose(leave_one_out_direction(moved, valid, off)[:, 1], theta[:, 1])


def test_leave_one_out_median():
    v = np.array([[1.0, 2.0, 3.0, 100.0]])
    m = leave_one_out_median(v, np.ones_like(v, dtype=bool))
    assert m.tolist() == [[3.0, 3.0, 2.0, 2.0]]
    assert np.isnan(leave_one_out_median(v, np.array([[True, False, False, False]]))[0, 0])
