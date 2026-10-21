import numpy as np
import pytest

from conftest import analytic_tables, make_farm
from wake_aware.model import WarpParams, features, fit_warp, predict, robust_lstsq


@pytest.fixture(scope="module")
def fitted(farm):
    base = np.zeros(len(farm.time), dtype=bool)
    base[: 60 * 144] = True  # first 60 days are the baseline
    model = fit_warp(farm, base, WarpParams(k_ref=0.04, sigma_deg=0.0), analytic_tables(farm.n))
    return model, predict(farm, model), base


def test_free_stream_and_wakes_are_recovered(farm, fitted):
    model, ref, _ = fitted
    true_d = farm.extra["deficit"]
    ok = np.isfinite(ref.deficit)
    assert np.abs(ref.deficit[ok] - true_d[ok]).mean() < 0.02  # direction from the other nacelles, +-3 deg noise
    # the free stream (in the farm curve's wind speed scale) follows the true undisturbed wind
    ws_free = farm.extra["ws_free"]
    sel = np.isfinite(ref.e_free_p[:, 0]) & (ws_free > 5) & (ws_free < 10)
    ratio = ref.e_free_p[sel, 0] / ws_free[sel]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.03) and np.std(ratio) < 0.05


def test_expected_power_matches_healthy_turbines(farm, fitted):
    _, ref, _ = fitted
    for i in range(farm.n):
        ok = ref.p_hat[:, i] >= 100
        assert farm.power[ok, i].sum() / ref.p_hat[ok, i].sum() == pytest.approx(1.0, abs=0.01)


def test_site_factor_is_learned(farm, fitted):
    model, _, _ = fitted
    # turbine 2 sees 3 % more wind, turbine 3 2 % less (relative to the others, power-equivalent scale)
    assert model.coef_p[1, 0] - model.coef_p[0, 0] == pytest.approx(np.log(1.03), abs=0.01)
    assert model.coef_p[2, 0] - model.coef_p[0, 0] == pytest.approx(np.log(0.98), abs=0.01)


def test_reference_ignores_the_turbine_itself():
    farm = make_farm(days=40, seed=3)
    base = np.ones(len(farm.time), dtype=bool)
    model = fit_warp(farm, base, WarpParams(k_ref=0.04, sigma_deg=0.0), analytic_tables(farm.n))
    ref = predict(farm, model)
    broken = farm.copy()
    broken.power[:, 1] *= 0.5
    broken.ws[:, 1] *= 1.2
    broken.nacelle[:, 1] += 20.0
    ref_b = predict(broken, model)
    assert np.allclose(ref.p_hat[:, 1], ref_b.p_hat[:, 1], equal_nan=True)
    assert np.allclose(ref.u_hat[:, 1], ref_b.u_hat[:, 1], equal_nan=True)


def test_robust_lstsq_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = 2.0 + 0.5 * x + rng.normal(0, 0.01, 500)
    y[:20] += 5.0
    coef = robust_lstsq(np.column_stack([np.ones_like(x), x]), y)
    assert coef == pytest.approx([2.0, 0.5], abs=0.01)


def test_features_shape_and_defaults():
    f = features(np.array([0.0, 90.0]), np.array([8.0, 8.0]), np.array([np.nan, 0.12]))
    assert f.shape == (2, 15)  # constant, 6 harmonic pairs, ln E, ln TI
    assert np.allclose(f[:, -2:], 0.0)  # reference wind speed and turbulence (NaN TI falls back to 0.12)
    assert np.allclose(f[1, 1:3], [0.0, 1.0], atol=1e-12)  # cos, sin of 90 deg
