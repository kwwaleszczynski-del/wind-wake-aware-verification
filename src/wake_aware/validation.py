"""Validation by fault injection: sensitivity, time to detection and false alarms, WARP vs the baselines.

Protocol (no information from the test years is used to build or tune anything):

* baseline year (2017): power curve, wake model calibration, site factors, nacelle offsets
* calibration year (2018): standardisation of the daily statistics and CUSUM thresholds, fault-free
* test years (2019-2020): false alarms on the unmodified data, and injection runs: for each fault scenario,
  turbine and start date, the fault is injected into that turbine for ``horizon`` days and every detector's
  chart is run from the start of the test period. A fault counts as detected if the detector alarms between the
  fault start and the end of the horizon; the time to detection is counted in days from the fault start.

The references of the faulty turbine come from the other turbines only, so they are computed once on the clean
data; only the faulty turbine's own signals change between runs. Faults alternate in sign (yaw +/-, anemometer
+/-) across start dates.

Usage:
    setup = Setup.build(farm, {"warp": ref, "nowake": ref0}, model.offsets, test_rows, cal_rows)
    runs = campaign(setup, SCENARIOS, n_starts=8, horizon_days=90)
    table = summarise(runs, setup)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from wake_aware.detectors import (WARP_DETECTORS, Calibration, DayGrid, calibrate, cusum, daily_statistics,
                                  relative_to_farm, run_charts, side, standardise, turbine_statistics)
from wake_aware.faults import SCENARIOS, Fault, inject
from wake_aware.freestream import wrap180
from wake_aware.performance import yaw_fit

MATCHING = {"yaw": {"yaw"}, "anemometer": {"anemo"}, "derate": {"rated", "power"}, "loss": {"power", "rated"}}
# detection systems compared: a fault counts as detected by a system when any of its charts alarms
SYSTEMS = {
    "WARP": ("power", "rated", "anemo", "yaw"),
    "WARP without wake model": ("power_nowake", "rated_nowake", "anemo_nowake", "yaw"),
    "neighbour index": ("nb",),
    "own power curve": ("curve",),
}
log = logging.getLogger(__name__)


def detector_name(name: str, ref_name: str) -> str:
    return name if ref_name == "warp" or name in ("nb", "curve") else f"{name}_{ref_name}"


@dataclass
class Setup:
    farm: object  # Farm restricted to the analysed rows
    refs: dict  # name -> Reference over the same rows
    offsets: np.ndarray
    days: DayGrid
    x: dict  # clean daily statistics, detector -> (n_days, n)
    x_curve_raw: np.ndarray
    cal: Calibration
    cal_days: np.ndarray
    test_days: np.ndarray
    curve: pd.DataFrame

    @classmethod
    def build(cls, farm, refs: dict, offsets, rows, cal_rows, test_rows, curve) -> "Setup":
        """``rows`` covers calibration and test periods; refs must be computed for ``rows``."""
        sub = farm_slice(farm, rows)
        days = DayGrid(sub.time)
        x, x_curve_raw = {}, None
        for ref_name, ref in refs.items():
            stats = daily_statistics(sub, ref, days)
            for name, v in stats.items():
                if ref_name != "warp" and name in ("yaw", "nb", "curve"):
                    continue  # independent of the reference variant
                x[detector_name(name, ref_name)] = v
        x_curve_raw = np.column_stack([
            turbine_statistics(i, sub.power[:, i], sub.ws[:, i], sub.p_curve[:, i], refs["warp"].delta[:, i],
                               sub.normal[:, i], refs["warp"], sub.p_ratio[:, i], days)["curve"] for i in range(sub.n)])
        cal_days, test_days = days.mask(cal_rows[rows]), days.mask(test_rows[rows])
        return cls(sub, refs, offsets, days, x, x_curve_raw, calibrate(x, cal_days), cal_days, test_days, curve)


def farm_slice(farm, rows):
    """Farm restricted to a subset of timestamps."""
    arrays = {k: v[rows] for k, v in vars(farm).items() if isinstance(v, np.ndarray) and v.ndim == 2}
    return replace(farm, time=farm.time[rows], **arrays)


def clean_alarms(setup: Setup) -> pd.DataFrame:
    """Alarms on the unmodified test period, every detector and turbine."""
    return run_charts(setup.x, setup.cal, setup.test_days, setup.days)


def start_dates(setup: Setup, n_starts: int, horizon_days: int, lead_days: int = 14) -> list[pd.Timestamp]:
    """Evenly spaced fault start dates inside the test period, leaving room for the horizon."""
    dates = setup.days.dates[setup.test_days]
    first, last = dates[0] + pd.Timedelta(days=lead_days), dates[-1] - pd.Timedelta(days=horizon_days - 1)
    return list(pd.date_range(first, last, periods=n_starts).normalize())


def _faulty_statistics(setup: Setup, i: int, bad) -> dict:
    """Daily statistics of turbine i after injection, for every detector in ``setup.x``."""
    out = {}
    for ref_name, ref in setup.refs.items():
        delta = wrap180(bad.nacelle[:, i] - setup.offsets[i] - ref.theta[:, i])
        stats = turbine_statistics(i, bad.power[:, i], bad.ws[:, i], bad.p_curve[:, i], delta, bad.normal[:, i],
                                   ref, bad.p_ratio[:, i], setup.days)
        for name, v in stats.items():
            key = detector_name(name, ref_name)
            if key not in setup.x:
                continue
            if name == "curve":
                raw = setup.x_curve_raw.copy()
                raw[:, i] = v
                v = relative_to_farm(raw)[:, i]
            out[key] = v
    return out


def run_injection(setup: Setup, fault: Fault, i: int, start: pd.Timestamp, horizon_days: int) -> list[dict]:
    """One injection: first alarm of every detector after the fault start (within the horizon)."""
    end = start + pd.Timedelta(days=horizon_days)
    bad = inject(setup.farm, i, start, end, fault, setup.curve)
    stats = _faulty_statistics(setup, i, bad)
    d0, d1 = setup.days.day(start), setup.days.day(end)
    test_idx = np.flatnonzero(setup.test_days)
    run_idx = test_idx[test_idx < d1]
    rows = []
    for name, v in stats.items():
        z = standardise(v[run_idx], setup.cal, name, i)
        alarms, sides, _, _ = cusum(z, setup.cal.k[name][i], setup.cal.h[name], side(name))
        after = [(run_idx[a], s) for a, s in zip(alarms, sides) if run_idx[a] >= d0]
        first = after[0] if after else None
        rows.append({"fault": fault.label, "kind": fault.kind, "size": fault.size,
                     "turbine": int(setup.farm.turbines[i]), "start": start, "detector": name,
                     "detected": first is not None,
                     "days_to_detect": (first[0] - d0) if first else np.nan, "side": first[1] if first else None})
    # size estimates over the fault window (detectors' daily statistics relative to their calibration median)
    win = np.arange(d0, d1)
    est = {f"est_{name}": float(np.nanmean(stats[name][win]) - setup.cal.mu[name][i]) for name in WARP_DETECTORS}
    if fault.kind == "yaw":  # the cos^2 fit on the fault window, from scratch
        ref = setup.refs["warp"]
        w = (setup.days.code >= d0) & (setup.days.code < d1)
        delta = wrap180(bad.nacelle[w, i] - setup.offsets[i] - ref.theta[w, i])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = bad.power[w, i] / ref.p_hat[w, i]
        valid = bad.normal[w, i] & (ref.e_hat[w, i] >= 4.5) & (ref.e_hat[w, i] <= 10.0)
        clean_delta = ref.delta[w, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            clean_ratio = setup.farm.power[w, i] / ref.p_hat[w, i]
        est["est_yaw_cos2"] = (yaw_fit(delta, ratio, valid, setup.days.code[w], n_boot=0)["gamma"]
                               - yaw_fit(clean_delta, clean_ratio, valid, setup.days.code[w], n_boot=0)["gamma"])
    for r in rows:
        r.update(est)
    return rows


def campaign(setup: Setup, scenarios=SCENARIOS, n_starts: int = 8, horizon_days: int = 90) -> pd.DataFrame:
    rows = []
    starts = start_dates(setup, n_starts, horizon_days)
    for fault in scenarios:
        for i in range(setup.farm.n):
            for s_idx, start in enumerate(starts):
                sign = -1.0 if (s_idx % 2 and fault.kind in ("yaw", "anemometer")) else 1.0
                f = Fault(fault.kind, sign * fault.size)
                label = fault.label.replace("+", "+/-")
                rows += [dict(r, fault=label, sign=sign) for r in run_injection(setup, f, i, start, horizon_days)]
        log.info("%s: %d runs", fault.label, setup.farm.n * len(starts))
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, clean: pd.DataFrame, setup: Setup, within_days: int = 30) -> pd.DataFrame:
    """Detection rate (within the horizon and within ``within_days``), median days to detection, false alarms."""
    g = runs.groupby(["fault", "detector"])
    table = pd.DataFrame({
        "detection_rate": g["detected"].mean(),
        f"detected_within_{within_days}d": g["days_to_detect"].apply(lambda d: float((d <= within_days).mean())),
        "median_days_to_detect": g["days_to_detect"].median(),
        "runs": g.size(),
    })
    years = setup.test_days.sum() / 365.25
    fa = clean.groupby("detector").size().reindex(list(setup.x), fill_value=0) / (setup.farm.n * years)
    table = table.join(fa.rename("false_alarms_per_turbine_year"), on="detector")
    return table


def diagnosis(runs: pd.DataFrame, detectors=WARP_DETECTORS) -> pd.DataFrame:
    """Share of injections whose first WARP alarm comes from the matching detector (ties count as correct)."""
    rows = []
    sel = runs[runs["detector"].isin(detectors)]
    for (fault, kind, _, _), g in sel.groupby(["fault", "kind", "turbine", "start"]):
        hit = g[g["detected"]]
        if hit.empty:
            rows.append({"fault": fault, "detected": False, "correct": False})
            continue
        first = hit["days_to_detect"].min()
        rows.append({"fault": fault, "detected": True,
                     "correct": bool(MATCHING[kind] & set(hit.loc[hit["days_to_detect"] == first, "detector"]))})
    d = pd.DataFrame(rows)
    return d.groupby("fault").agg(detected=("detected", "mean"), correctly_diagnosed=("correct", "mean"))


def system_summary(runs: pd.DataFrame, clean: pd.DataFrame, setup: Setup, systems=SYSTEMS,
                   within_days: int = 30) -> pd.DataFrame:
    """Per fault and detection system: detection rate, share within ``within_days``, median days, false alarms."""
    years = setup.test_days.sum() / 365.25
    rows = []
    for method, dets in systems.items():
        dets = [d for d in dets if d in setup.x]
        if not dets:
            continue
        sel = runs[runs["detector"].isin(dets)]
        g = sel.groupby(["fault", "turbine", "start"]).agg(detected=("detected", "any"), days=("days_to_detect", "min"))
        fa = clean["detector"].isin(dets).sum() / (setup.farm.n * years)
        for fault, d in g.groupby(level="fault"):
            rows.append({"fault": fault, "method": method, "detection_rate": d["detected"].mean(),
                         f"detected_within_{within_days}d": float((d["days"] <= within_days).mean()),
                         "median_days_to_detect": d["days"].median(), "runs": len(d),
                         "false_alarms_per_turbine_year": fa})
    order = list(dict.fromkeys(runs["fault"]))
    out = pd.DataFrame(rows)
    out["fault"] = pd.Categorical(out["fault"], order, ordered=True)
    return out.sort_values(["fault", "method"]).reset_index(drop=True)


def example(setup: Setup, fault: Fault, i: int, start: pd.Timestamp, horizon_days: int, detector: str,
            margin_days: int = 45) -> dict:
    """Daily statistic of one injection with its first alarm, for plotting."""
    end = start + pd.Timedelta(days=horizon_days)
    stats = _faulty_statistics(setup, i, inject(setup.farm, i, start, end, fault, setup.curve))
    d0, d1 = setup.days.day(start), setup.days.day(end)
    test_idx = np.flatnonzero(setup.test_days)
    run_idx = test_idx[test_idx < d1]
    z = standardise(stats[detector][run_idx], setup.cal, detector, i)
    alarms, _, _, _ = cusum(z, setup.cal.k[detector][i], setup.cal.h[detector], side(detector))
    after = [run_idx[a] for a in alarms if run_idx[a] >= d0]
    show = test_idx[(test_idx >= d0 - margin_days) & (test_idx < d1)]
    scale = 1.0 if detector.startswith("yaw") else 100.0
    return {"dates": setup.days.dates[show], "values": scale * stats[detector][show],
            "mu": scale * setup.cal.mu[detector][i], "start": start, "end": end,
            "alarm": setup.days.dates[after[0]] if after else None, "days": (after[0] - d0) if after else np.nan}
