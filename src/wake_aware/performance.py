"""Performance index with an uncertainty interval, and the static yaw misalignment from the cos^n power loss.

Performance index of turbine i over a period T (normal operation, expected power >= 100 kW):

    I_i(T) = sum_t P_i(t) / sum_t P^_i(t)

Its 95 % interval comes from a moving-block bootstrap over days (blocks of 7 days keep the day-to-day
correlation of the weather): the daily sums of P and P^ are resampled in blocks and the ratio recomputed.

Static yaw misalignment. With Delta_i the nacelle position minus the neighbours' wind direction (centred on its
baseline median), the power ratio follows the inflow angle, r = A cos^n(Delta - phi). Delta is a noisy proxy
of the inflow angle; for n = 2 and Gaussian noise eta ~ N(0, s^2) in Delta, cos^2 u = (1 + cos 2u) / 2 gives

    E[r | Delta] = A/2 + (A/2) e^(-2 s^2) cos(2 (Delta - phi)) = A' + R cos(2 (Delta - phi)),

linear in A', R cos 2phi and R sin 2phi. So the peak phi (the nacelle orientation of maximum power) is
estimated without bias by linear least squares, while the noise only flattens the curve. The turbine runs
around median(Delta), so its static misalignment is gamma = median(Delta) - phi and its energy cost
1 - cos^2(gamma). Uncertainty: bootstrap over days.

Usage:
    table = performance_index(farm.power, ref.p_hat, farm.normal, farm.time, freq="Y")
    yaw = yaw_fit(ref.delta[:, i], farm.power[:, i] / ref.p_hat[:, i], mask, days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wake_aware.detectors import POWER_MIN_KW

BLOCK_DAYS = 7


def block_bootstrap_ratio(num_daily, den_daily, n_boot: int = 500, block: int = BLOCK_DAYS, seed: int = 0):
    """2.5 and 97.5 % percentiles of sum(num) / sum(den) under a moving-block bootstrap over days."""
    num, den = np.asarray(num_daily, dtype=float), np.asarray(den_daily, dtype=float)
    n = len(num)
    if n == 0 or den.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def performance_index(power, p_hat, normal, time: pd.DatetimeIndex, turbines, freq: str = "Y",
                      min_records: int = 1000, n_boot: int = 500) -> pd.DataFrame:
    """Index with 95 % bootstrap interval per turbine and period ("Y" years, "Q" quarters, "M" months)."""
    days = time.tz_localize(None).normalize()
    periods = time.tz_localize(None).to_period(freq)
    rows = []
    for i, turbine in enumerate(turbines):
        ok = normal[:, i] & (p_hat[:, i] >= POWER_MIN_KW) & np.isfinite(power[:, i])
        d = pd.DataFrame({"p": power[ok, i], "p_hat": p_hat[ok, i], "day": days[ok], "period": periods[ok]})
        for period, g in d.groupby("period"):
            if len(g) < min_records:
                continue
            daily = g.groupby("day")[["p", "p_hat"]].sum()
            lo, hi = block_bootstrap_ratio(daily["p"], daily["p_hat"], n_boot=n_boot)
            rows.append({"turbine": turbine, "period": str(period), "index": g["p"].sum() / g["p_hat"].sum(),
                         "ci_low": lo, "ci_high": hi, "records": len(g)})
    return pd.DataFrame(rows).set_index(["turbine", "period"])


def fit_cos2(delta_deg, ratio) -> tuple[float, float, float]:
    """Least squares fit of ratio = A + R cos(2 (Delta - phi)); returns (phi [deg], A, R)."""
    x = np.deg2rad(np.asarray(delta_deg, dtype=float))
    design = np.column_stack([np.ones_like(x), np.cos(2 * x), np.sin(2 * x)])
    a, b, c = np.linalg.lstsq(design, np.asarray(ratio, dtype=float), rcond=None)[0]
    return float(np.rad2deg(np.arctan2(c, b) / 2)), float(a), float(np.hypot(b, c))


def yaw_fit(delta, ratio, valid, day_code, delta_lim: float = 20.0, n_boot: int = 200, seed: int = 0) -> dict:
    """Peak phi of the power ratio vs Delta, static misalignment gamma = median(Delta) - phi, energy cost.

    5-95 % interval of gamma from a bootstrap over days.
    """
    ok = valid & np.isfinite(delta) & np.isfinite(ratio) & (np.abs(delta) <= delta_lim) & (ratio > 0.3) & (ratio < 1.7)
    x, y, day = delta[ok], ratio[ok], day_code[ok]
    if len(x) < 500:
        return {"phi": np.nan, "gamma": np.nan, "p05": np.nan, "p95": np.nan, "loss_pct": np.nan, "records": len(x)}

    def gamma(xs, ys):
        phi, _, r = fit_cos2(xs, ys)
        return np.median(xs) - phi if r > 0 else np.nan

    phi = fit_cos2(x, y)[0]
    g = float(np.median(x) - phi)
    if n_boot == 0:
        return {"phi": phi, "gamma": g, "p05": np.nan, "p95": np.nan,
                "loss_pct": 100 * (1 - np.cos(np.deg2rad(g)) ** 2), "records": len(x)}
    groups = list(pd.Series(np.arange(len(x))).groupby(day).indices.values())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        boots.append(gamma(x[sel], y[sel]))
    return {"phi": phi, "gamma": g, "p05": float(np.nanpercentile(boots, 5)), "p95": float(np.nanpercentile(boots, 95)),
            "loss_pct": 100 * (1 - np.cos(np.deg2rad(g)) ** 2), "records": len(x)}
