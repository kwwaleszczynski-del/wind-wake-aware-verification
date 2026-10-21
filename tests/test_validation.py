import numpy as np
import pandas as pd
import pytest

from conftest import analytic_tables, make_farm
from wake_aware.faults import Fault
from wake_aware.model import WarpParams, fit_warp, predict
from wake_aware.validation import Setup, campaign, clean_alarms, diagnosis, summarise


@pytest.fixture(scope="module")
def setup():
    farm = make_farm(days=200, seed=11)
    base = np.asarray(farm.time < pd.Timestamp("2019-03-01", tz="UTC"))
    t1, t2 = pd.Timestamp("2019-03-01", tz="UTC"), pd.Timestamp("2019-05-01", tz="UTC")
    cal = np.asarray((farm.time >= t1) & (farm.time < t2))
    test = np.asarray(farm.time >= pd.Timestamp("2019-05-01", tz="UTC"))
    model = fit_warp(farm, base, WarpParams(k_ref=0.04, sigma_deg=0.0), analytic_tables(farm.n))
    rows = cal | test
    ref = predict(farm, model, rows)
    return Setup.build(farm, {"warp": ref}, model.offsets, rows, cal, test, model.curve)


def test_campaign_detects_clear_faults(setup):
    faults = [Fault("yaw", 8.0), Fault("anemometer", 0.06), Fault("derate", 0.05), Fault("loss", 0.05)]
    runs = campaign(setup, faults, n_starts=2, horizon_days=30)
    assert len(runs) == 4 * setup.farm.n * 2 * len(setup.x)
    table = summarise(runs, clean_alarms(setup), setup)
    for fault, detector in [("yaw 8 deg", "yaw"), ("anemometer +/-6 %", "anemo"), ("derating 5 %", "rated"),
                            ("power loss 5 %", "power")]:
        assert table.loc[(fault, detector), "detection_rate"] == 1.0, (fault, detector)
    diag = diagnosis(runs)
    assert (diag["detected"] == 1.0).all() and (diag["correctly_diagnosed"] >= 0.75).all()


def test_signs_alternate_and_estimates_follow_the_fault(setup):
    runs = campaign(setup, [Fault("yaw", 8.0)], n_starts=2, horizon_days=30)
    one = runs[runs["detector"] == "yaw"]
    assert set(one["sign"]) == {1.0, -1.0}
    # nacelle offset estimate: the nacelle turns by -delta
    assert np.allclose(one["est_yaw"] / one["sign"], -8.0, atol=1.0)
