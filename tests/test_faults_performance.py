import numpy as np
import pandas as pd
import pytest

from conftest import make_farm
from wake_aware.faults import Fault, inject, yaw_power_factor
from wake_aware.performance import block_bootstrap_ratio, fit_cos2, performance_index, yaw_fit

CUBIC = pd.DataFrame({"ws": np.arange(0.0, 25.001, 0.02)})
CUBIC["power_kw"] = np.minimum(CUBIC["ws"] ** 3, 1728.0)  # cubic up to 12 m/s, then flat


@pytest.fixture(scope="module")
def small():
    return make_farm(days=20, seed=5)


def window(farm):
    start, end = farm.time[1000], farm.time[2000]
    w = np.asarray((farm.time >= start) & (farm.time < end))
    return start, end, w


def test_yaw_factor_is_cos_squared_in_partial_load_and_one_in_strong_wind():
    f = yaw_power_factor(np.zeros(3), np.array([6.0, 8.0, 20.0]), 8.0, CUBIC)
    assert f[:2] == pytest.approx(np.cos(np.deg2rad(8.0)) ** 2, rel=1e-3)
    assert f[2] == pytest.approx(1.0)
    # an existing inflow angle of the same sign makes the loss larger, of the opposite sign smaller
    assert yaw_power_factor(np.array([5.0]), np.array([8.0]), 8.0, CUBIC)[0] < f[0]
    assert yaw_power_factor(np.array([-4.0]), np.array([8.0]), 8.0, CUBIC)[0] > f[0]


def test_yaw_injection(small):
    start, end, w = window(small)
    bad = inject(small, 1, start, end, Fault("yaw", 8.0), CUBIC)
    shift = ((small.nacelle[w, 1] - bad.nacelle[w, 1]) % 360.0)
    assert np.allclose(shift, 8.0)
    # vane readings near -8 deg are brought closer to the wind, so only the total must drop: by about cos^2(8 deg)
    assert bad.power[w, 1].sum() / small.power[w, 1].sum() == pytest.approx(np.cos(np.deg2rad(8.0)) ** 2, abs=0.01)
    assert np.array_equal(bad.power[~w], small.power[~w]) and np.array_equal(bad.power[:, 0], small.power[:, 0])
    with pytest.raises(ValueError):
        inject(small, 1, start, end, Fault("yaw", 4.0))


def test_anemometer_derate_and_loss_injection(small):
    start, end, w = window(small)
    bad = inject(small, 2, start, end, Fault("anemometer", -0.06))
    assert np.allclose(bad.ws[w, 2], 0.94 * small.ws[w, 2]) and np.array_equal(bad.power, small.power)
    bad = inject(small, 2, start, end, Fault("derate", 0.05))
    assert bad.power[w, 2].max() <= 0.95 * small.rated_kw + 1e-9
    assert np.array_equal(bad.ws, small.ws)
    bad = inject(small, 2, start, end, Fault("loss", 0.03))
    assert np.allclose(bad.power[w, 2], 0.97 * small.power[w, 2])
    assert Fault("anemometer", 0.03).label == "anemometer +3 %" and Fault("yaw", -4.0).label == "yaw 4 deg"


def test_cos2_peak_is_unbiased_with_noisy_angles():
    rng = np.random.default_rng(0)
    gamma = rng.normal(0, 6, 40000)  # true inflow angle around the optimum
    phi = 3.0
    delta = gamma + phi + rng.normal(0, 6, gamma.size)  # observed proxy: shifted by phi, plus noise
    ratio = np.cos(np.deg2rad(gamma)) ** 2 + rng.normal(0, 0.03, gamma.size)
    est_phi, a, r = fit_cos2(delta, ratio)
    assert est_phi == pytest.approx(phi, abs=0.7)
    assert r > 0
    days = np.repeat(np.arange(gamma.size // 144 + 1), 144)[: gamma.size]
    res = yaw_fit(delta, ratio, np.ones(gamma.size, dtype=bool), days, n_boot=50)
    assert res["gamma"] == pytest.approx(np.median(delta[np.abs(delta) <= 20]) - res["phi"])
    assert res["p05"] < res["gamma"] < res["p95"]


def test_block_bootstrap_interval_covers_the_ratio():
    rng = np.random.default_rng(0)
    den = rng.uniform(500, 1500, 365)
    num = den * 0.98 * (1 + rng.normal(0, 0.05, 365))
    lo, hi = block_bootstrap_ratio(num, den)
    assert lo < num.sum() / den.sum() < hi and hi - lo < 0.02


def test_performance_index_per_year(small):
    p_hat = small.power * 1.02
    table = performance_index(small.power, p_hat, small.normal, small.time, small.turbines, freq="Y", min_records=100)
    assert set(table.index.get_level_values("turbine")) == set(small.turbines)
    assert np.allclose(table["index"], 1 / 1.02)
    assert (table["ci_low"] <= table["index"]).all() and (table["index"] <= table["ci_high"]).all()
