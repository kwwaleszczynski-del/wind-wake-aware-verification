import pandas as pd

from wake_aware.cli import episodes, size_estimates, status_context


def _status():
    t = pd.Timestamp("2021-03-10 08:00", tz="UTC")
    return pd.DataFrame({
        "turbine": [1, 1, 1, 2],
        "start": [t, t + pd.Timedelta(hours=5), t + pd.Timedelta(days=40), t],
        "status": ["Warning", "Warning", "Stop", "Informational"],
        "message": ["Anemometer defect", "Anemometer defect", "Grid loss", "Battery test"],
    })


def test_status_context_counts_messages_near_the_date():
    s = _status()
    day = pd.Timestamp("2021-03-12", tz="UTC")
    assert status_context(s, 1, day) == "Anemometer defect (2x)"  # the grid loss 40 days later is outside
    assert status_context(s, 2, day) == ""  # informational only
    assert status_context(s, 1, day, "anemo") == "Anemometer defect (2x)"
    assert status_context(s, 1, day, "yaw") == ""


def test_episodes_merge_repeated_alarms():
    d = pd.to_datetime(["2021-03-01", "2021-03-15", "2021-06-01"]).tz_localize("UTC")
    alarms = pd.DataFrame({"detector": "anemo", "turbine": 1, "date": d, "side": ["low", "low", "high"]})
    ep = episodes(alarms, _status())
    assert len(ep) == 2
    assert ep.loc[0, "alarms"] == 2 and str(ep.loc[0, "last_alarm"]) == "2021-03-15"
    assert ep.loc[0, "status_log"] == "Anemometer defect (2x)" and ep.loc[1, "side"] == "high"


def test_size_estimates_are_sign_corrected():
    runs = pd.DataFrame({
        "fault": "yaw 4 deg", "kind": "yaw", "size": [4.0, -4.0], "turbine": [1, 1], "start": [1, 2],
        "est_yaw": [-4.2, 3.8], "est_yaw_cos2": [-3.0, 2.0], "est_anemo": 0.0, "est_power": 0.0, "est_rated": 0.0,
    })
    est = size_estimates(runs)
    assert est.loc["yaw 4 deg", "estimate_median"] == 4.0
    assert est.loc["yaw 4 deg", "cos2_fit_median"] == 2.5
