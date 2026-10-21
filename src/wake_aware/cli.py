"""Command line interface.

Usage:
    python -m wake_aware fetch --data-dir data                 # Kelmarsh SCADA from Zenodo (~1.8 GB), once
    python -m wake_aware run --data-dir data                   # everything: calibration, validation, charts
    python -m wake_aware run --data-dir ../wind-power-curve-kelmarsh/data --quick
    python -m wake_aware run --help

Years: baseline 2017 (model fit), calibration 2018 (chart thresholds, fault-free), test 2019-2020 (false alarms
and injected faults), application 2021 (detectors run on unmodified data and compared with the status logs).
Tables go to ./results, charts to ./docs/img, caches to ./data/processed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wake_aware import plots
from wake_aware.data import load_farm
from wake_aware.detectors import WARP_DETECTORS, DayGrid
from wake_aware.faults import SCENARIOS, Fault
from wake_aware.model import FEATURES, WarpParams, calibrate_wake, fit_warp, predict
from wake_aware.performance import fit_cos2, performance_index, yaw_fit
from wake_aware.validation import (Setup, campaign, clean_alarms, diagnosis, example, summarise,
                                   system_summary)
from wake_aware.wake import pair_tables, smooth_direction

log = logging.getLogger("wake_aware")


def cmd_fetch(args):
    from wind_power_curve.zenodo import download

    for path in download(args.years, args.data_dir):
        log.info("ok %s", path)


def wake_tables(farm, cache_dir: Path) -> np.ndarray:
    path = cache_dir / "wake_tables.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    tables = pair_tables(farm.x, farm.y, farm.hub_height, farm.diameter, farm.rated_kw)
    np.save(path, tables)
    log.info("single-wake tables %s in %.0f s", tables.shape, time.time() - t0)
    return tables


def parameter_grid(quick: bool) -> list[WarpParams]:
    if quick:
        return [WarpParams(0.035, 3.0, 2.0, 0.0)]
    return [WarpParams(k, s, t0, ks) for k, ks, s, t0 in
            itertools.product([0.025, 0.03, 0.035, 0.04], [0.0, 0.3], [1.0, 2.0, 3.0], [0.0, 2.0, 4.0])]


# status messages that relate to what each chart watches
STATUS_KEYWORDS = {
    "anemo": r"[Aa]nemometer|[Ww]ind speed|[Ii]cing",
    "curve": r"[Aa]nemometer|[Ww]ind speed|[Ii]cing|[Pp]ower|reduc",
    "yaw": r"[Yy]aw|[Vv]ane|[Ww]ind direction",
    "power": r"[Pp]ower|reduc|[Tt]emperature|[Pp]itch|[Ii]cing",
    "rated": r"[Pp]ower|reduc|[Tt]emperature|[Pp]itch|[Ii]cing",
    "nb": r"[Pp]ower|reduc|[Tt]emperature|[Pp]itch|[Ii]cing",
}


def status_context(status: pd.DataFrame, turbine: int, date: pd.Timestamp, detector: str | None = None,
                   days: int = 30, top: int = 3) -> str:
    """Most frequent non-informational status messages of a turbine within +-``days`` of a date.

    With ``detector``, only messages related to what that chart watches (``STATUS_KEYWORDS``) are counted.
    """
    lo, hi = date - pd.Timedelta(days=days), date + pd.Timedelta(days=days + 1)
    ev = status[status["turbine"].eq(turbine) & (status["start"] >= lo) & (status["start"] < hi)
                & ~status["status"].isin(["Informational"])]
    pattern = STATUS_KEYWORDS.get(str(detector).split("_")[0]) if detector else None
    if pattern:
        ev = ev[ev["message"].fillna("").str.contains(pattern)]
    if ev.empty:
        return ""
    counts = ev["message"].value_counts().head(top)
    return "; ".join(f"{m} ({n}x)" for m, n in counts.items())


def cmd_run(args):
    t_start = time.time()
    res, img, cache = args.results_dir, args.img_dir, args.cache_dir
    for d in (res, img, cache):
        d.mkdir(parents=True, exist_ok=True)
    years = list(range(args.baseline_year, args.apply_year + 1))
    farm = load_farm(args.data_dir, years, args.baseline_year, cache_dir=cache, refresh=args.refresh)
    tables = wake_tables(farm, cache)
    base = farm.period(args.baseline_year)

    # 1. wake model calibration and the two references (with and without wake model)
    calib = calibrate_wake(farm, base, tables, parameter_grid(args.quick))
    calib.to_csv(res / "wake_calibration.csv", index=False, float_format="%.5f")
    best = calib.iloc[0]
    params = WarpParams(float(best["k_ref"]), float(best["sigma_deg"]), float(best["theta0"]), float(best["k_slope"]))
    log.info("best wake parameters %s", params)
    model = fit_warp(farm, base, params, smooth_direction(tables, params.sigma_deg))
    model0 = fit_warp(farm, base, params, np.zeros_like(tables))
    ref = predict(farm, model)
    ref0 = predict(farm, model0)
    pd.DataFrame(model.coef_p, index=farm.turbines, columns=FEATURES).to_csv(res / "site_factors.csv",
                                                                            float_format="%.4f")

    # 2. performance index with uncertainty, all years
    annual = performance_index(farm.power, ref.p_hat, farm.normal, farm.time, farm.turbines, "Y")
    monthly = performance_index(farm.power, ref.p_hat, farm.normal, farm.time, farm.turbines, "M", min_records=500)
    nb_ok = farm.normal & np.isfinite(farm.p_ratio)
    for table, freq in ((annual, "Y"), (monthly, "M")):
        nb = performance_index(farm.power, np.where(nb_ok, farm.p_ratio, np.nan), nb_ok, farm.time, farm.turbines,
                               freq, min_records=500 if freq == "M" else 1000, n_boot=200)
        table[["index_nb", "nb_ci_low", "nb_ci_high"]] = nb[["index", "ci_low", "ci_high"]].reindex(table.index)
    annual.to_csv(res / "performance_index_annual.csv", float_format="%.4f")
    monthly.to_csv(res / "performance_index_monthly.csv", float_format="%.4f")
    with np.errstate(divide="ignore", invalid="ignore"):
        anemo = np.log(farm.ws / ref.u_hat)
    anemo_ok = farm.normal & (ref.u_hat >= 4.0) & (ref.u_hat <= 11.0)
    anemo_year = pd.DataFrame(100 * np.where(anemo_ok, anemo, np.nan), index=farm.time, columns=farm.turbines)
    anemo_year.groupby(farm.time.year).median().to_csv(res / "anemometer_residual_annual_pct.csv", float_format="%.2f")

    # 3. static yaw misalignment (cos^2 fit) per turbine and year, and the monthly nacelle offset
    day_code = DayGrid(farm.time).code
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = farm.power / ref.p_hat
    fit_ok = farm.normal & (ref.e_hat >= 4.5) & (ref.e_hat <= 10.0)
    yaw_rows = []
    for i, t in enumerate(farm.turbines):
        for y in years:
            r = farm.period(y)
            f = yaw_fit(ref.delta[r, i], ratio[r, i], fit_ok[r, i], day_code[r], n_boot=args.n_boot)
            yaw_rows.append({"turbine": t, "year": y, "median_delta": np.nanmedian(ref.delta[r, i][fit_ok[r, i]]), **f})
    yaw_table = pd.DataFrame(yaw_rows).set_index(["turbine", "year"])
    yaw_table.to_csv(res / "yaw_misalignment.csv", float_format="%.3f")
    delta_ok = farm.normal & (ref.e_free >= 4.0)
    monthly_delta = pd.DataFrame(np.where(delta_ok, ref.delta, np.nan), index=farm.time, columns=farm.turbines)
    monthly_delta = monthly_delta.groupby(farm.time.tz_localize(None).to_period("M")).median()
    monthly_delta.to_csv(res / "nacelle_offset_monthly.csv", float_format="%.2f")

    # 4. validation: calibration year + test years, injected faults
    cal_rows, test_rows = farm.period(args.calibration_year), farm.period(args.test_years[0], args.test_years[-1])
    rows = cal_rows | test_rows
    sub_refs = {"warp": _slice(ref, rows), "nowake": _slice(ref0, rows)}
    setup = Setup.build(farm, sub_refs, model.offsets, rows, cal_rows, test_rows, model.curve)
    noise = pd.DataFrame({k: (1 if k == "yaw" else 100) * v for k, v in setup.cal.sigma.items()}, index=farm.turbines)
    noise.to_csv(res / "daily_noise.csv", float_format="%.2f")  # % (log ratios) and deg (yaw)
    pd.DataFrame({"h": setup.cal.h}).to_csv(res / "cusum_thresholds.csv", float_format="%.2f")
    from wind_power_curve.scada import load

    _, status = load(sorted(set(args.test_years) | {args.apply_year}), args.data_dir)
    clean = clean_alarms(setup)
    clean["turbine"] = farm.turbines[clean["turbine"].to_numpy()] if len(clean) else clean["turbine"]
    episodes(clean, status).to_csv(res / "alarms_test_clean.csv", index=False)
    runs = campaign(setup, SCENARIOS, n_starts=args.n_starts, horizon_days=args.horizon)
    runs.to_csv(res / "injection_runs.csv", index=False, float_format="%.4f")
    by_detector = summarise(runs, clean.assign(turbine=0), setup)
    by_detector.to_csv(res / "detection_by_detector.csv", float_format="%.3f")
    by_method = system_summary(runs, clean, setup)
    by_method.to_csv(res / "detection_by_method.csv", index=False, float_format="%.3f")
    diag = diagnosis(runs)
    diag.to_csv(res / "diagnosis.csv", float_format="%.3f")
    estimates = size_estimates(runs)
    estimates.to_csv(res / "fault_size_estimates.csv", float_format="%.3f")

    # 5. application year: detectors on unmodified data, alarms next to the status log
    app_rows = cal_rows | farm.period(args.apply_year)
    app = Setup.build(farm, {"warp": _slice(ref, app_rows)}, model.offsets, app_rows, cal_rows,
                      farm.period(args.apply_year), model.curve)
    app_alarms = clean_alarms(app)
    app_alarms = app_alarms[app_alarms["detector"].isin(WARP_DETECTORS + ("nb", "curve"))].copy()
    if len(app_alarms):
        app_alarms["turbine"] = farm.turbines[app_alarms["turbine"].to_numpy()]
    episodes(app_alarms, status).to_csv(res / f"alarms_{args.apply_year}.csv", index=False)

    # 6. charts
    plots.layout(farm, img / "layout.png")
    plots.wake_validation(farm, _slice(ref, base), model, base, img / "wake_validation.png")
    plots.detection_summary(by_method[by_method["method"].isin(plots.METHOD_COLORS)], img / "detection_summary.png")
    ex = []
    for fault, det, ylabel in [(Fault("yaw", 4.0), "yaw", "daily median Δ [deg]"),
                               (Fault("anemometer", 0.03), "anemo", "daily median ln(u / û) [%]"),
                               (Fault("derate", 0.05), "rated", "full-load ln(ΣP / ΣP̂) [%]")]:
        turbine, start, sign = typical_run(runs, fault, det)
        i = int(np.flatnonzero(farm.turbines == turbine)[0])
        f = Fault(fault.kind, sign * fault.size)
        e = example(setup, f, i, start, args.horizon, det)
        ex.append(dict(e, fault=f"{f.label} from here", ylabel=ylabel,
                       title=f"{f.label} injected into T{turbine} ({det} chart)"))
    plots.example_runs(ex, img / "example_detections.png")
    plots.performance_index(monthly, img / "performance_index.png")
    best_t = (yaw_table.xs(args.baseline_year, level="year")["p95"]
              - yaw_table.xs(args.baseline_year, level="year")["p05"]).idxmin()
    plots.yaw_tracker(monthly_delta, yaw_example(farm, ref, ratio, fit_ok, base, best_t), img / "yaw_tracker.png")

    summary = {
        "years": years, "baseline_year": args.baseline_year, "calibration_year": args.calibration_year,
        "test_years": args.test_years, "apply_year": args.apply_year, "wake_parameters": vars(params),
        "daily_noise_pct_mean": noise.mean().round(2).to_dict(),
        "injection_runs": int(runs[["fault", "turbine", "start"]].drop_duplicates().shape[0]),
        "clean_false_alarms_per_turbine_year":
            by_method.groupby("method")["false_alarms_per_turbine_year"].first().round(3).to_dict(),
        "runtime_s": round(time.time() - t_start),
    }
    (res / "run.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("done in %.0f s; results in %s, charts in %s", time.time() - t_start, res, img)


def typical_run(runs: pd.DataFrame, fault: Fault, detector: str):
    """(turbine, start, sign) of a detected injection whose detection time is closest to the median."""
    d = runs[(runs["kind"] == fault.kind) & (runs["size"].abs() == fault.size) & (runs["detector"] == detector)
             & runs["detected"]]
    row = d.iloc[(d["days_to_detect"] - d["days_to_detect"].median()).abs().argsort().iloc[0]]
    return int(row["turbine"]), pd.Timestamp(row["start"]), float(np.sign(row["size"]))


def episodes(alarms: pd.DataFrame, status: pd.DataFrame, gap_days: int = 30) -> pd.DataFrame:
    """Merge repeated alarms of one chart and turbine (less than ``gap_days`` apart) into episodes."""
    rows = []
    for (det, turbine), g in alarms.sort_values("date").groupby(["detector", "turbine"]):
        dates = list(g["date"])
        sides = list(g["side"])
        start = 0
        for k in range(1, len(dates) + 1):
            if k == len(dates) or (dates[k] - dates[k - 1]).days >= gap_days:
                rows.append({"detector": det, "turbine": turbine, "first_alarm": dates[start].date(),
                             "last_alarm": dates[k - 1].date(), "alarms": k - start, "side": sides[start],
                             "status_log": status_context(status, turbine, dates[start], det)})
                start = k
    cols = ["detector", "turbine", "first_alarm", "last_alarm", "alarms", "side", "status_log"]
    return pd.DataFrame(rows, columns=cols).sort_values(["first_alarm", "turbine"]).reset_index(drop=True)


def _slice(ref, rows):
    return type(ref)(**{k: (v[rows] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in vars(ref).items()})


def size_estimates(runs: pd.DataFrame) -> pd.DataFrame:
    """Fault size estimated from the detectors' daily statistics over the fault window vs the injected size."""
    one = runs.drop_duplicates(["fault", "turbine", "start"]).copy()
    one["injected"] = one["size"]
    est = {"yaw": ("est_yaw", 1.0, -1.0), "anemometer": ("est_anemo", 100.0, 1.0), "loss": ("est_power", 100.0, -1.0),
           "derate": ("est_rated", 100.0, -1.0)}
    rows = []
    for fault, d in one.groupby("fault"):
        kind = d["kind"].iloc[0]
        col, scale, sign = est[kind]
        injected = d["size"].abs() * (1.0 if kind == "yaw" else 100.0)
        value = sign * scale * d[col] * np.sign(d["size"])
        row = {"fault": fault, "injected": injected.median(), "estimate_median": value.median(),
               "estimate_p10": value.quantile(0.1), "estimate_p90": value.quantile(0.9)}
        if kind == "yaw":
            cos2 = -d["est_yaw_cos2"] * np.sign(d["size"])
            row.update({"cos2_fit_median": cos2.median(), "cos2_fit_p10": cos2.quantile(0.1),
                        "cos2_fit_p90": cos2.quantile(0.9)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("fault")


def yaw_example(farm, ref, ratio, fit_ok, base, turbine) -> dict:
    i = int(np.flatnonzero(farm.turbines == turbine)[0])
    ok = fit_ok[base, i] & np.isfinite(ratio[base, i]) & (np.abs(ref.delta[base, i]) <= 20)
    ok &= (ratio[base, i] > 0.3) & (ratio[base, i] < 1.7)
    x, y = ref.delta[base, i][ok], ratio[base, i][ok]
    phi, a, r = fit_cos2(x, y)
    b = pd.Series(y).groupby(np.round(x / 2) * 2).agg(["mean", "size"])
    binned = b.loc[b["size"] >= 200, "mean"]
    return {"turbine": turbine, "year": int(farm.time[base][0].year), "phi": phi, "A": a, "R": r, "binned": binned}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wake-aware", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the Kelmarsh SCADA archives from Zenodo")
    f.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2022)))
    r = sub.add_parser("run", help="calibrate, validate and chart")
    for p in (f, r):
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("KELMARSH_DATA_DIR", "data")),
                       help="folder with raw/ (Zenodo archives) and processed/ (Parquet cache)")
    r.add_argument("--baseline-year", type=int, default=2017)
    r.add_argument("--calibration-year", type=int, default=2018)
    r.add_argument("--test-years", type=int, nargs=2, default=[2019, 2020])
    r.add_argument("--apply-year", type=int, default=2021)
    r.add_argument("--n-starts", type=int, default=8, help="fault start dates per turbine and scenario")
    r.add_argument("--horizon", type=int, default=90, help="days a fault lasts / may take to be detected")
    r.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the yaw intervals")
    r.add_argument("--quick", action="store_true", help="skip the wake parameter grid, use the documented values")
    r.add_argument("--refresh", action="store_true", help="re-read the archives instead of the Parquet caches")
    r.add_argument("--cache-dir", type=Path, default=Path("data/processed"))
    r.add_argument("--results-dir", type=Path, default=Path("results"))
    r.add_argument("--img-dir", type=Path, default=Path("docs/img"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    {"fetch": cmd_fetch, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
