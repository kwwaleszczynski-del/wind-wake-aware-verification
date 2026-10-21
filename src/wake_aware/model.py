"""WARP - Wake-Adjusted Reference Power: the expected power and anemometer reading of every turbine.

For turbine i at time t (all quantities from the *other* turbines, so a fault on i cannot leak into its own
reference):

1. Direction. theta_-i = circular mean of the other turbines' nacelle positions, each corrected by its static
   offset c_j learned in the baseline year; theta0 is a farm-wide north correction used for the wake lookup.
2. Wakes. Deficits d_j(t) of all turbines for direction theta_-i + theta0 from the single-wake tables (only
   running turbines cast wakes), see ``wake.farm_deficit``. The wake expansion rate follows the turbulence
   intensity measured on the other turbines' nacelles, k(t) = k_ref + k_slope (TI_-i(t) - 0.12): wakes are
   deeper and longer in stable, low-turbulence flow.
3. Free stream. The turbines that are free in this direction (d_j < 2 %, normal operation, j != i) give
       E_-i = median_j  e~_j / (1 - d_j)
   with e~_j = F^-1(P_j), the power-equivalent wind speed, in partial load and kappa_j u_j (anemometer scaled to
   the same scale in the baseline year) at higher power. E^P_-i uses only power (all free turbines in partial
   load) and is the reference for the anemometer check.
4. Expected equivalent wind speed and power:
       e^_i = E_-i (1 - d_i) s_i(theta, E, TI),    P^_i = F(e^_i)
       ln s_i = b0 + sum_{m=1..6} (a_m cos m theta + b_m sin m theta) + c ln(E / 8 m/s) + g ln(TI_-i / 0.12)
   s_i is a smooth site-and-transfer factor fitted per turbine in the baseline year: terrain speed-up, the
   turbine's own small performance offset, and through the TI term the effect of stability on wind shear
   (two turbines have 10 m lower hubs). Six harmonics resolve about 30 deg, too coarse to imitate a wake
   (10-15 deg wide here): the sharp direction structure is left to the wake model.
   The fit is in log wind speed; because the power curve is convex, b0 is then adjusted so that the expected
   baseline energy equals the actual (performance index exactly 1 in the baseline year).
5. Expected anemometer reading: u^_i = E^P_-i (1 - d_i) s^u_i(theta, E, TI), with s^u_i of the same form.

Usage:
    tables = smooth_direction(pair_tables(...), sigma_deg)
    model = fit_warp(farm, farm.period(2017), WarpParams(k=0.04, sigma_deg=6, theta0=0), tables)
    ref = predict(farm, model)          # Reference with p_hat, u_hat, delta ... as (time, turbine) arrays
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wake_aware.freestream import (farm_curve, leave_one_out_direction, leave_one_out_median, nacelle_offsets,
                                   power_from_wind, wind_from_power, wrap180)
from wake_aware.wake import farm_deficit

FREE_DEFICIT = 0.02
TI_REF = 0.12
N_HARMONICS = 6
ENERGY_MIN_KW = 100.0  # expected power above which records count for the performance index
FIT_RANGE = (4.5, 10.0)  # m/s of expected equivalent wind speed used to fit s_i (well inside the invertible range)
E_REF = 8.0
log = logging.getLogger(__name__)


@dataclass
class WarpParams:
    k_ref: float = 0.03  # wake expansion rate at TI = 0.12
    sigma_deg: float = 3.0  # direction smoothing of the wake tables
    theta0: float = 0.0  # farm-wide north correction of the nacelle positions [deg]
    k_slope: float = 0.0  # dk / dTI
    free_deficit: float = FREE_DEFICIT

    def k(self, ti):
        return np.clip(self.k_ref + self.k_slope * (np.nan_to_num(ti, nan=TI_REF) - TI_REF), 0.01, 0.08)


@dataclass
class WarpModel:
    params: WarpParams
    tables: np.ndarray  # direction-smoothed single-wake tables
    curve: pd.DataFrame
    offsets: np.ndarray
    kappa: np.ndarray
    coef_p: np.ndarray  # (turbine, feature)
    coef_u: np.ndarray


@dataclass
class Reference:
    theta: np.ndarray  # leave-one-out direction (offsets removed, no theta0)
    deficit: np.ndarray  # modelled wake deficit at the turbine itself
    e_free: np.ndarray  # free-stream equivalent wind speed, power or scaled anemometer
    e_free_p: np.ndarray  # free-stream equivalent wind speed from power only (NaN if not available)
    n_free: np.ndarray  # number of free reference turbines
    ti: np.ndarray = None  # turbulence intensity from the other turbines
    p_max: float = np.nan  # top of the farm power curve
    e_hat: np.ndarray = None
    p_hat: np.ndarray = None
    u_hat: np.ndarray = None
    delta: np.ndarray = None  # nacelle position minus neighbours' direction, baseline-centred [deg]


FEATURES = (["b0"] + [f"{p}{m}" for m in range(1, N_HARMONICS + 1) for p in ("a", "b")] + ["c_lnE", "g_lnTI"])


def features(theta, e, ti) -> np.ndarray:
    """Design matrix of the site-and-transfer factor: 1, cos/sin(m theta) for m = 1..6, ln(E/8), ln(TI/0.12)."""
    t = np.deg2rad(np.asarray(theta, dtype=float))
    le = np.log(np.clip(np.asarray(e, dtype=float), 3.0, 15.0) / E_REF)
    lt = np.log(np.clip(np.nan_to_num(np.asarray(ti, dtype=float), nan=TI_REF), 0.04, 0.35) / TI_REF)
    cols = [np.ones_like(t)]
    for m in range(1, N_HARMONICS + 1):
        cols += [np.cos(m * t), np.sin(m * t)]
    return np.stack(cols + [le, lt], axis=-1)


def robust_lstsq(design, y, k: float = 4.0, iterations: int = 3) -> np.ndarray:
    """Least squares, refitted after dropping points further than k robust standard deviations."""
    keep = np.ones(len(y), dtype=bool)
    coef = np.zeros(design.shape[1])
    for _ in range(iterations):
        coef = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        r = y - design @ coef
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))
        keep = np.abs(r - med) <= k * max(mad, 1e-6)
    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]


def _row_median(x, mask) -> np.ndarray:
    v = np.where(mask, x, np.nan)
    out = np.full(len(v), np.nan)
    ok = mask.any(axis=1)
    out[ok] = np.nanmedian(v[ok], axis=1)
    return out


def reference_terms(farm, model: WarpModel, rows=None) -> Reference:
    """Steps 1-3 for the selected rows (bool mask over timestamps, default all)."""
    rows = np.ones(len(farm.time), dtype=bool) if rows is None else rows
    p = model.params
    normal, power, ws = farm.normal[rows], farm.power[rows], farm.ws[rows]
    e = wind_from_power(np.where(normal, power, np.nan), model.curve)
    e_mix = np.where(np.isnan(e), model.kappa[None, :] * ws, e)
    theta = leave_one_out_direction(farm.nacelle[rows], normal & ~np.isnan(farm.nacelle[rows]), model.offsets)
    u_look = leave_one_out_median(ws, normal)
    ti = leave_one_out_median(farm.turbulence[rows], normal)
    n_t, n = power.shape
    out = Reference(*(np.full((n_t, n), np.nan) for _ in range(4)), n_free=np.zeros((n_t, n), dtype=int), ti=ti)
    for i in range(n):
        d = farm_deficit(model.tables, theta[:, i] + p.theta0, u_look[:, i], p.k(ti[:, i]), farm.operating[rows])
        free = normal & (d < p.free_deficit)
        free[:, i] = False
        out.deficit[:, i] = d[:, i]
        out.n_free[:, i] = free.sum(axis=1)
        out.e_free[:, i] = _row_median(e_mix / (1 - d), free)
        free_p = free & ~np.isnan(e)
        all_p = (free_p.sum(axis=1) == free.sum(axis=1)) & free.any(axis=1)
        out.e_free_p[:, i] = np.where(all_p, _row_median(e / (1 - d), free_p), np.nan)
    out.theta = theta
    return out


def fit_warp(farm, base, params: WarpParams, tables: np.ndarray) -> WarpModel:
    """Fit the farm curve, nacelle offsets, anemometer scales and site factors on the baseline rows ``base``."""
    norm_b = farm.normal & base[:, None]
    curve = farm_curve(farm.ws, farm.power, norm_b)
    offsets = nacelle_offsets(farm.nacelle[base], norm_b[base] & (farm.ws[base] >= 4.0))
    e = wind_from_power(np.where(norm_b, farm.power, np.nan), curve)
    kappa = np.nanmedian(np.where(np.isnan(e), np.nan, e / farm.ws), axis=0)
    n_feat = features(0.0, E_REF, TI_REF).shape[-1]
    model = WarpModel(params, tables, curve, offsets, kappa, np.zeros((farm.n, n_feat)), np.zeros((farm.n, n_feat)))
    ref = reference_terms(farm, model, base)
    e_b, ws_b, normal_b = e[base], farm.ws[base], farm.normal[base]
    expected = ref.e_free_p * (1 - ref.deficit)
    for i in range(farm.n):
        rows = normal_b[:, i] & (expected[:, i] >= FIT_RANGE[0]) & (expected[:, i] <= FIT_RANGE[1])
        X = features(ref.theta[:, i], ref.e_free_p[:, i], ref.ti[:, i])
        for target, coef in ((e_b[:, i], model.coef_p), (ws_b[:, i], model.coef_u)):
            ok = rows & ~np.isnan(target)
            y = np.log(target[ok]) - np.log(expected[ok, i])
            coef[i] = robust_lstsq(X[ok], y)
    # energy calibration of b0 (the log fit is median-unbiased in wind speed, not in energy)
    e_hat = ref.e_free * (1 - ref.deficit) * np.exp(np.einsum("tif,if->ti", features(ref.theta, ref.e_free, ref.ti),
                                                              model.coef_p))
    power_b = farm.power[base]
    for i in range(farm.n):
        ok = normal_b[:, i] & np.isfinite(power_b[:, i]) & np.isfinite(e_hat[:, i])
        scale = 1.0
        for _ in range(30):
            p_hat = power_from_wind(e_hat[ok, i] * scale, curve)
            sel = p_hat >= ENERGY_MIN_KW
            ratio = power_b[ok, i][sel].sum() / p_hat[sel].sum()
            scale *= ratio ** (1.0 / 3.0)
            if abs(ratio - 1.0) < 1e-5:
                break
        model.coef_p[i, 0] += np.log(scale)
    return model


def predict(farm, model: WarpModel, rows=None) -> Reference:
    """Expected power, expected anemometer reading and the nacelle direction offset (steps 1-5)."""
    rows = np.ones(len(farm.time), dtype=bool) if rows is None else rows
    ref = reference_terms(farm, model, rows)
    s_p = np.exp(np.einsum("tif,if->ti", features(ref.theta, ref.e_free, ref.ti), model.coef_p))
    s_u = np.exp(np.einsum("tif,if->ti", features(ref.theta, ref.e_free_p, ref.ti), model.coef_u))
    ref.e_hat = ref.e_free * (1 - ref.deficit) * s_p
    ref.p_hat = power_from_wind(ref.e_hat, model.curve)
    ref.p_max = float(model.curve["power_kw"].max())
    ref.u_hat = ref.e_free_p * (1 - ref.deficit) * s_u
    ref.delta = wrap180(farm.nacelle[rows] - model.offsets[None, :] - ref.theta)
    return ref


def _scores(farm, model: WarpModel, rows, cap: float = 0.1) -> tuple[float, float]:
    """(daily noise, record-level truncated MSE) of the expected power over ``rows``.

    Daily noise: mean over turbines of the robust spread (1.4826 MAD) of the daily ln(sum P / sum P^) below full
    load - the quantity that limits how fast a power loss can be detected. Record MSE: mean truncated squared
    log residual of the power-equivalent wind speed, 4.5-10 m/s.
    """
    from wake_aware.detectors import FULL_LOAD, POWER_MIN_KW, DayGrid, daily_log_ratio

    ref = predict(farm, model, rows)
    days = DayGrid(farm.time, rows)
    spreads = []
    for i in range(farm.n):
        ok = farm.normal[rows, i] & (ref.p_hat[:, i] >= POWER_MIN_KW) & (ref.p_hat[:, i] < FULL_LOAD * ref.p_max)
        x = daily_log_ratio(farm.power[rows, i], ref.p_hat[:, i], ok, days)
        x = x[np.isfinite(x)]
        spreads.append(1.4826 * np.median(np.abs(x - np.median(x))))
    e = wind_from_power(np.where(farm.normal[rows], farm.power[rows], np.nan), model.curve)
    ok = ~np.isnan(e) & ~np.isnan(ref.e_hat) & (ref.e_hat >= FIT_RANGE[0]) & (ref.e_hat <= FIT_RANGE[1])
    r = np.log(e[ok]) - np.log(ref.e_hat[ok])
    return float(np.mean(spreads)), float(np.mean(np.minimum(r ** 2, cap ** 2)))


def daily_noise(farm, model: WarpModel, rows) -> float:
    return _scores(farm, model, rows)[0]


def residual_score(farm, model: WarpModel, rows) -> float:
    return _scores(farm, model, rows)[1]


def calibrate_wake(farm, base, tables: np.ndarray, grid: list[WarpParams]) -> pd.DataFrame:
    """Fit WARP for every candidate parameter set on the baseline rows; scores sorted best (lowest noise) first."""
    from wake_aware.wake import smooth_direction

    smoothed, rows = {}, []
    for prm in grid:
        if prm.sigma_deg not in smoothed:
            smoothed[prm.sigma_deg] = smooth_direction(tables, prm.sigma_deg)
        noise, mse = _scores(farm, fit_warp(farm, base, prm, smoothed[prm.sigma_deg]), base)
        rows.append({"k_ref": prm.k_ref, "k_slope": prm.k_slope, "sigma_deg": prm.sigma_deg, "theta0": prm.theta0,
                     "daily_noise": noise, "record_mse": mse})
        log.info("%s daily noise %.4f", prm, noise)
    return pd.DataFrame(rows).sort_values("daily_noise").reset_index(drop=True)
