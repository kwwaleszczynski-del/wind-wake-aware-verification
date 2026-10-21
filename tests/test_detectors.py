import numpy as np
import pandas as pd
import pytest

from wake_aware.detectors import DayGrid, calibrate, cusum, daily_log_ratio, daily_median, relative_to_farm, run_charts


def test_day_grid_is_independent_of_datetime_resolution():
    for unit in ("ns", "us", "s"):
        t = pd.date_range("2020-02-28 23:50", periods=3 * 144, freq="10min", tz="UTC").as_unit(unit)
        days = DayGrid(t)
        assert days.n_days == 4
        assert days.code[0] == 0 and days.code[1] == 1 and days.code[-1] == 3
        assert days.day(pd.Timestamp("2020-03-01", tz="UTC")) == 2  # 2020 is a leap year


def test_daily_ratio_and_median():
    t = pd.date_range("2021-01-01", periods=3 * 144, freq="10min", tz="UTC")
    days = DayGrid(t)
    num, den = np.full(len(t), 970.0), np.full(len(t), 1000.0)
    valid = np.ones(len(t), dtype=bool)
    valid[144:144 + 120] = False  # day 2 has only 24 valid records (< 36)
    x = daily_log_ratio(num, den, valid, days)
    assert x[0] == pytest.approx(np.log(0.97)) and np.isnan(x[1])
    med = daily_median(np.arange(len(t), dtype=float) % 144, valid, days)
    assert med[0] == pytest.approx(71.5) and np.isnan(med[1])


def test_cusum_detects_a_step_and_restarts():
    z = np.r_[np.zeros(30), np.full(30, -2.0)]
    alarms, sides, lo, _ = cusum(z, k=0.5, h=5.0, side="low")
    assert alarms[0] == 33 and sides[0] == "low"  # 1.5 per day: 1.5, 3.0, 4.5, 6.0 > 5 on the 4th day
    assert lo[alarms[0]] > 5.0 and len(alarms) > 1  # restarted and alarmed again
    assert cusum(np.zeros(100), 0.5, 5.0, "both")[0] == []
    assert cusum(-z, 0.5, 5.0, "low")[0] == []  # an increase does not trigger the low side


def test_cusum_carries_state_over_missing_days():
    z = np.array([-3.0, np.nan, np.nan, -3.0])
    _, _, lo, _ = cusum(z, k=0.5, h=np.inf, side="low")
    assert lo.tolist() == [2.5, 2.5, 2.5, 5.0]


def test_calibration_gives_no_alarms_on_its_own_data():
    rng = np.random.default_rng(0)
    t = pd.date_range("2018-01-01", periods=400 * 144, freq="10min", tz="UTC")
    days = DayGrid(t)
    x = {"power": rng.normal(0, 0.05, (days.n_days, 3)), "anemo": rng.normal(0, 0.02, (days.n_days, 3))}
    cal_days = np.zeros(days.n_days, dtype=bool)
    cal_days[:365] = True
    cal = calibrate(x, cal_days)
    assert cal.h["power"] >= 5.0
    assert cal.k["power"] == pytest.approx(0.015 / 0.05, rel=0.15)  # half of a 3 % target in sigma units
    assert run_charts(x, cal, cal_days, days).empty


def test_relative_to_farm():
    x = np.array([[0.01, 0.03, -0.01], [np.nan, np.nan, np.nan]])
    r = relative_to_farm(x)
    assert np.allclose(r[0], [0.0, 0.02, -0.02]) and np.isnan(r[1]).all()
