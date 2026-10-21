"""Daily residual statistics and CUSUM charts for fault detection.

Each detector reduces the 10-minute residuals of one turbine to one value per day, standardises it with the
median and robust spread of the calibration year, and runs a CUSUM chart:

    z_d = (x_d - mu) / sigma
    S_d^low  = max(0, S_{d-1}^low  - z_d - k)        S_d^high = max(0, S_{d-1}^high + z_d - k)

An alarm is raised when S exceeds h, after which the chart restarts from zero. Days without enough data leave S
unchanged. Each chart is tuned to the smallest fault it should catch (``TARGET``, 3 % or 3 deg): k is half that
shift in units of the turbine's daily standard deviation. h is set on fault-free calibration data
(see ``calibrate``).

Detectors (WARP, this project):
    power   x = ln(sum P / sum P^)          normal operation, 100 kW <= P^ < 97 % of rated   low side
    rated   x = ln(sum P / sum P^)          normal operation, P^ >= 97 % of rated (full load)  low side
    anemo   x = median ln(u / u^)           normal operation, u^ in 4-11 m/s                 both sides
    yaw     x = median Delta                normal operation, free stream >= 4 m/s           both sides
Baselines (wind_power_curve):
    nb      x = ln(sum P / sum p_ratio)     neighbour index, partial load                    low side
    curve   x = ln(sum P / sum p_curve) minus the farm median of that day (own power curve)   low side

Usage:
    days = DayGrid(farm.time, rows)
    x = daily_statistics(farm, ref, days)              # {"power": (n_days, n), ...}
    cal = calibrate(x, days.mask(farm.period(2018)))   # mu, sigma, h per detector and turbine
    alarms = run_charts(x, cal, days.mask(farm.period(2019, 2020)))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_RECORDS = 36  # 6 hours of 10-minute data per day
MIN_RECORDS_RATED = 6  # full-load hours are rarer; an hour per day is enough as the residual is small there
FULL_LOAD = 0.97
POWER_MIN_KW = 100.0
ANEMO_RANGE = (4.0, 11.0)
YAW_MIN_WS = 4.0
NB_RANGE = (100.0, 1500.0)
H_MIN = 5.0
SIDES = {"power": "low", "rated": "low", "anemo": "both", "yaw": "both", "nb": "low", "curve": "low"}
TARGET = {"power": 0.03, "rated": 0.03, "anemo": 0.03, "yaw": 3.0, "nb": 0.03, "curve": 0.03}
WARP_DETECTORS = ("power", "rated", "anemo", "yaw")
BASELINES = ("nb", "curve")


def side(name: str) -> str:
    """Chart side of a detector; variants such as ``power_nowake`` share the side of ``power``."""
    return SIDES[name.split("_")[0]]


class DayGrid:
    """Maps the 10-minute timestamps of ``rows`` to consecutive day numbers 0..n_days-1."""

    def __init__(self, time: pd.DatetimeIndex, rows=None):
        rows = np.ones(len(time), dtype=bool) if rows is None else rows
        t = time[rows]
        self.dates = pd.date_range(t.min().normalize(), t.max().normalize(), freq="D")
        self.code = ((t.normalize() - self.dates[0]) // pd.Timedelta("1D")).to_numpy().astype(int)
        self.rows = rows

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def mask(self, timestamp_mask) -> np.ndarray:
        """Days that contain any of the selected timestamps."""
        sel = np.asarray(timestamp_mask)[self.rows]
        out = np.zeros(self.n_days, dtype=bool)
        out[np.unique(self.code[sel])] = True
        return out

    def day(self, timestamp) -> int:
        return int((pd.Timestamp(timestamp).normalize() - self.dates[0]) // pd.Timedelta("1D"))


def daily_log_ratio(num, den, valid, days: DayGrid, min_count: int = MIN_RECORDS) -> np.ndarray:
    """ln(sum num / sum den) per day over the valid records (NaN if fewer than ``min_count``)."""
    w = valid & np.isfinite(num) & np.isfinite(den)
    sn = np.bincount(days.code[w], weights=num[w], minlength=days.n_days)
    sd = np.bincount(days.code[w], weights=den[w], minlength=days.n_days)
    cnt = np.bincount(days.code[w], minlength=days.n_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(np.maximum(sn, 1e-3 * sd) / sd)
    x[(cnt < min_count) | (sd <= 0)] = np.nan
    return x


def daily_median(values, valid, days: DayGrid, min_count: int = MIN_RECORDS) -> np.ndarray:
    w = valid & np.isfinite(values)
    s = pd.Series(values[w]).groupby(days.code[w])
    med, cnt = s.median(), s.size()
    out = np.full(days.n_days, np.nan)
    good = cnt.index[cnt >= min_count]
    out[good] = med.loc[good].to_numpy()
    return out


def turbine_statistics(i: int, power, ws, p_curve, delta, normal, ref, p_ratio, days: DayGrid) -> dict:
    """Daily statistics of turbine column ``i``; signal arguments are 1-D arrays over ``days.rows``.

    ``curve`` is returned raw here (before subtracting the farm median of the day).
    """
    p_hat, u_hat, e_free = ref.p_hat[:, i], ref.u_hat[:, i], ref.e_free[:, i]
    full = FULL_LOAD * ref.p_max
    with np.errstate(divide="ignore", invalid="ignore"):
        anemo = np.log(ws / u_hat)
    return {
        "power": daily_log_ratio(power, p_hat, normal & (p_hat >= POWER_MIN_KW) & (p_hat < full), days),
        "rated": daily_log_ratio(power, p_hat, normal & (p_hat >= full), days, MIN_RECORDS_RATED),
        "anemo": daily_median(anemo, normal & (u_hat >= ANEMO_RANGE[0]) & (u_hat <= ANEMO_RANGE[1]), days),
        "yaw": daily_median(delta, normal & (e_free >= YAW_MIN_WS), days),
        "nb": daily_log_ratio(power, p_ratio, normal & ~np.isnan(p_ratio), days),
        "curve": daily_log_ratio(power, p_curve, normal & (p_curve >= NB_RANGE[0]) & (p_curve <= NB_RANGE[1]), days),
    }


def daily_statistics(farm, ref, days: DayGrid) -> dict[str, np.ndarray]:
    """All detectors for all turbines: {detector: (n_days, n_turbines)}; ``ref`` must cover ``days.rows``."""
    r = days.rows
    cols = [turbine_statistics(i, farm.power[r, i], farm.ws[r, i], farm.p_curve[r, i], ref.delta[:, i],
                               farm.normal[r, i], ref, farm.p_ratio[r, i], days) for i in range(farm.n)]
    out = {name: np.column_stack([c[name] for c in cols]) for name in SIDES}
    out["curve"] = relative_to_farm(out["curve"])
    return out


def relative_to_farm(x: np.ndarray) -> np.ndarray:
    """Subtract the farm median of each day (removes seasonal and signal effects shared by all turbines)."""
    med = np.full((x.shape[0], 1), np.nan)
    ok = np.isfinite(x).any(axis=1)
    med[ok, 0] = np.nanmedian(x[ok], axis=1)
    return x - med


def cusum(z, k: float = 0.5, h: float = np.inf, side: str = "low"):
    """CUSUM over daily z-scores. Returns (alarm day indices, alarm sides, S_low, S_high)."""
    s_lo = s_hi = 0.0
    lo, hi = np.zeros(len(z)), np.zeros(len(z))
    alarms, sides = [], []
    for d, v in enumerate(z):
        if np.isfinite(v):
            if side in ("low", "both"):
                s_lo = max(0.0, s_lo - v - k)
            if side in ("high", "both"):
                s_hi = max(0.0, s_hi + v - k)
        lo[d], hi[d] = s_lo, s_hi
        if s_lo > h or s_hi > h:
            alarms.append(d)
            sides.append("low" if s_lo > h else "high")
            s_lo = s_hi = 0.0
    return alarms, sides, lo, hi


@dataclass
class Calibration:
    mu: dict
    sigma: dict
    k: dict
    h: dict


def calibrate(x: dict, cal_days, h_min: float = H_MIN) -> Calibration:
    """Standardisation, reference value and alarm threshold of every detector from fault-free calibration days.

    mu and sigma are the median and 1.4826 x MAD of each turbine's daily values; k = TARGET / 2 / sigma. The
    threshold h is the highest CUSUM value any turbine reached over the calibration days (no alarms there by
    construction), but at least ``h_min``.
    """
    mu, sigma, k, h = {}, {}, {}, {}
    for name, v in x.items():
        c = v[cal_days]
        mu[name] = np.nanmedian(c, axis=0)
        sigma[name] = 1.4826 * np.nanmedian(np.abs(c - mu[name]), axis=0)
        k[name] = TARGET[name.split("_")[0]] / 2.0 / sigma[name]
        z = (c - mu[name]) / sigma[name]
        peak = 0.0
        for i in range(c.shape[1]):
            _, _, lo, hi = cusum(z[:, i], k[name][i], np.inf, side(name))
            peak = max(peak, lo.max(), hi.max())
        h[name] = max(h_min, float(peak) * 1.0001)
    return Calibration(mu, sigma, k, h)


def standardise(x: np.ndarray, cal: Calibration, name: str, i=None) -> np.ndarray:
    mu, sigma = cal.mu[name], cal.sigma[name]
    if i is not None:
        mu, sigma = mu[i], sigma[i]
    return (x - mu) / sigma


def run_charts(x: dict, cal: Calibration, days_mask, days: DayGrid) -> pd.DataFrame:
    """Alarms of every detector and turbine over the selected days (charts start at zero on the first day)."""
    idx = np.flatnonzero(days_mask)
    rows = []
    for name, v in x.items():
        z = standardise(v[idx], cal, name)
        for i in range(v.shape[1]):
            alarms, sides, _, _ = cusum(z[:, i], cal.k[name][i], cal.h[name], side(name))
            rows += [{"detector": name, "turbine": i, "date": days.dates[idx[a]], "side": s}
                     for a, s in zip(alarms, sides)]
    return pd.DataFrame(rows, columns=["detector", "turbine", "date", "side"])
