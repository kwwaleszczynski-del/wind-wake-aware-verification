"""Building blocks for the free-stream reference: farm power curve and its inverse, directions, medians.

The farm power curve F maps a density-normalised nacelle wind speed to power (bin means over all turbines in
normal operation in the baseline year). Its inverse turns any turbine's power into a *power-equivalent wind
speed* e = F^-1(P), independent of that turbine's anemometer. It is only defined in partial load, where power
still responds to the wind (``P_RANGE``).

Usage:
    curve = farm_curve(farm.ws, farm.power, farm.normal & farm.period(2017)[:, None])
    e = wind_from_power(farm.power, curve)
    theta = leave_one_out_direction(farm.nacelle, farm.normal, offsets)   # (time, turbine)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_RANGE = (50.0, 1850.0)  # kW, partial-load range in which power is inverted to a wind speed
CUT_OUT = 25.0


def wrap180(a):
    """Angle difference wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def farm_curve(ws, power, mask, bin_width: float = 0.5, min_count: int = 30) -> pd.DataFrame:
    """Mean power per wind speed bin over the masked records, made non-decreasing; columns ws, power_kw."""
    ws, power = np.asarray(ws, dtype=float)[mask], np.asarray(power, dtype=float)[mask]
    ok = ~(np.isnan(ws) | np.isnan(power))
    df = pd.DataFrame({"ws": ws[ok], "power_kw": power[ok]})
    df["bin"] = np.floor(df["ws"] / bin_width + 0.5) * bin_width
    g = df.groupby("bin")
    curve = pd.DataFrame({"ws": g["ws"].mean(), "power_kw": g["power_kw"].mean(), "n": g.size()})
    curve = curve[curve["n"] >= min_count].reset_index(drop=True)
    curve["power_kw"] = np.maximum.accumulate(curve["power_kw"].clip(lower=0.0).to_numpy())
    return curve


def wind_from_power(power, curve: pd.DataFrame, p_range=P_RANGE) -> np.ndarray:
    """Power-equivalent wind speed F^-1(P); NaN outside ``p_range`` (where the inverse is ill-conditioned)."""
    p = np.asarray(power, dtype=float)
    xs, ys = curve["power_kw"].to_numpy(), curve["ws"].to_numpy()
    xs = xs + np.arange(len(xs)) * 1e-6  # strictly increasing for np.interp on flat parts
    out = np.interp(p, xs, ys)
    out[~((p >= p_range[0]) & (p <= p_range[1]))] = np.nan
    return out


def power_from_wind(ws, curve: pd.DataFrame, cut_out: float = CUT_OUT) -> np.ndarray:
    """Power from the farm curve: linear in between, 0 below the first bin, last value held up to cut-out."""
    ws = np.asarray(ws, dtype=float)
    xs, ys = curve["ws"].to_numpy(), curve["power_kw"].to_numpy()
    p = np.interp(ws, xs, ys, left=0.0, right=ys[-1])
    p[(ws > cut_out) | (ws < xs[0])] = 0.0
    p[np.isnan(ws)] = np.nan
    return p


def circular_mean(angles, mask, axis: int = 1) -> np.ndarray:
    """Circular mean in degrees [0, 360) over ``axis`` of the masked angles; NaN where nothing is valid."""
    a = np.deg2rad(np.where(mask, angles, np.nan))
    s, c = np.nansum(np.sin(a), axis=axis), np.nansum(np.cos(a), axis=axis)
    out = np.rad2deg(np.arctan2(s, c)) % 360.0
    out[np.sum(mask & ~np.isnan(a), axis=axis) == 0] = np.nan
    return out


def leave_one_out_direction(nacelle, valid, offsets=None) -> np.ndarray:
    """(time, turbine) wind direction at each turbine from the *other* turbines' nacelle positions.

    ``offsets`` (one per turbine, degrees) are subtracted first: they are the turbines' static nacelle position
    offsets relative to each other, learned in the baseline year.
    """
    corrected = np.asarray(nacelle, dtype=float) - (0.0 if offsets is None else np.asarray(offsets)[None, :])
    rad = np.deg2rad(np.where(valid, corrected, np.nan))
    s, c = np.nan_to_num(np.sin(rad)), np.nan_to_num(np.cos(rad))
    cnt = (valid & ~np.isnan(rad)).astype(float)
    s_o, c_o, n_o = s.sum(1, keepdims=True) - s, c.sum(1, keepdims=True) - c, cnt.sum(1, keepdims=True) - cnt
    out = np.rad2deg(np.arctan2(s_o, c_o)) % 360.0
    out[n_o == 0] = np.nan
    return out


def nacelle_offsets(nacelle, valid, max_iter: int = 20, tol: float = 0.01) -> np.ndarray:
    """Static nacelle position offset of each turbine relative to the others (median difference, degrees).

    Fixed-point iteration: each turbine's offset is its median difference from the circular mean of the other
    turbines' offset-corrected nacelle positions. The offsets are defined up to a common constant (set to mean 0).
    """
    offsets = np.zeros(nacelle.shape[1])
    for _ in range(max_iter):
        theta = leave_one_out_direction(nacelle, valid, offsets)
        delta = wrap180(nacelle - theta)
        new = np.array([np.nanmedian(np.where(valid[:, i], delta[:, i], np.nan)) for i in range(nacelle.shape[1])])
        new -= new.mean()
        done = np.max(np.abs(new - offsets)) < tol
        offsets = new
        if done:
            break
    return offsets


def leave_one_out_median(values, mask) -> np.ndarray:
    """(time, turbine) median over the *other* turbines of ``values`` where ``mask``; NaN if none."""
    v = np.where(mask, values, np.nan)
    out = np.full(v.shape, np.nan)
    for i in range(v.shape[1]):
        others = np.delete(v, i, axis=1)
        ok = ~np.isnan(others).all(axis=1)
        out[ok, i] = np.nanmedian(others[ok], axis=1)
    return out
