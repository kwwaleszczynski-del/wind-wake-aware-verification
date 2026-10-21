"""Charts for the README (static PNG, light theme).

Usage:
    plots.wake_validation(farm, ref, rows, path)
    plots.detection_summary(summary, path)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wake_aware.freestream import wind_from_power  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
CRITICAL = "#d03b3b"
METHOD_COLORS = {"WARP": SERIES[0], "neighbour index": SERIES[1], "own power curve": SERIES[2]}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "legend.labelcolor": INK2, "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def wake_validation(farm, ref, model, rows, path: Path, bin_deg: float = 3.0):
    """Measured vs modelled wind at each turbine relative to the free stream, by wind direction."""
    _style()
    e = wind_from_power(np.where(farm.normal[rows], farm.power[rows], np.nan), model.curve)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        e_free = ref.e_free_p[:, i]
        ok = np.isfinite(e[:, i]) & np.isfinite(e_free) & (e_free >= 5) & (e_free <= 9)
        theta = ref.theta[ok, i] % 360.0
        b = np.floor(theta / bin_deg) * bin_deg + bin_deg / 2
        obs = pd.Series(e[ok, i] / ref.e_free_p[ok, i]).groupby(b).median()
        wake = pd.Series(1 - ref.deficit[ok, i]).groupby(b).mean()
        model_ratio = pd.Series(ref.e_hat[ok, i] / ref.e_free[ok, i]).groupby(b).median()
        ax.plot(obs.index, obs.values, color=SERIES[0], lw=1.6, label="measured (power-equivalent)")
        ax.plot(model_ratio.index, model_ratio.values, color=SERIES[1], lw=1.6, label="WARP: wake model x site factor")
        ax.plot(wake.index, wake.values, color=MUTED, lw=1.0, label="wake model alone")
        ax.set_title(f"T{farm.turbines[i]}  (hub {farm.hub_height[i]:.1f} m)", loc="left")
        ax.set_xticks(range(0, 361, 90))
        ax.set_ylim(0.55, 1.12)
    for ax in axes[1]:
        ax.set_xlabel("wind direction [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("wind at turbine / free stream")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Wakes at Kelmarsh, 5-9 m/s: measured from power vs WARP", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def layout(farm, path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(farm.x, farm.y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for t, x, y, h in zip(farm.turbines, farm.x, farm.y, farm.hub_height):
        ax.annotate(f"T{t} ({h:g} m)", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(f"Layout, rotor diameter {farm.diameter:g} m", loc="left")
    ax.margins(0.25)
    _save(fig, path)


def detection_summary(table: pd.DataFrame, path: Path, min_rate: float = 0.25):
    """Detection rate within the horizon and median days to detection, per fault and method."""
    _style()
    faults = [str(f) for f in dict.fromkeys(table["fault"])]
    methods = [m for m in METHOD_COLORS if m in set(table["method"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(faults) * len(methods) / 1.5 + 1.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    y0 = np.arange(len(faults))[::-1] * 1.0
    h = 0.78 / len(methods)
    for k, m in enumerate(methods):
        d = table[table["method"] == m].assign(fault=lambda x: x["fault"].astype(str))
        d = d.set_index("fault").reindex(faults)
        y = y0 + (len(methods) - 1) / 2 * h - k * h
        a1.barh(y, 100 * d["detection_rate"], height=h * 0.8, color=METHOD_COLORS[m], label=m)
        shown = d["detection_rate"] >= min_rate
        a2.scatter(d["median_days_to_detect"][shown], y[shown.to_numpy()], s=42, color=METHOD_COLORS[m],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for yy, v in zip(y, d["detection_rate"]):
            a1.annotate(f"{100 * v:.0f}", (100 * v, yy), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK2)
    a1.set_yticks(y0, faults, color=INK)
    a1.set_xlim(0, 110)
    a1.set_xlabel("detected within 90 days [% of injections]")
    a2.set_xlabel(f"median days to detection (where >= {100 * min_rate:.0f} % detected)")
    a2.set_xlim(left=0)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper left", ncol=len(methods), bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


def example_runs(examples: list[dict], path: Path):
    """Daily statistic and its CUSUM alarm for a few injections (one panel each)."""
    _style()
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.3 * len(examples)))
    for ax, ex in zip(np.atleast_1d(axes), examples):
        dates, x = ex["dates"], ex["values"]
        ax.axvspan(ex["start"], ex["end"], color=SERIES[1], alpha=0.10, lw=0)
        ax.plot(dates, x, color=SERIES[0], lw=1.2, marker="o", ms=2.5)
        ax.axhline(ex["mu"], color=MUTED, lw=1)
        ax.annotate("calibration-year median", (dates[-1], ex["mu"]), xytext=(0, 3), textcoords="offset points",
                    fontsize=7, color=MUTED, ha="right")
        if ex.get("alarm") is not None:
            ax.axvline(ex["alarm"], color=CRITICAL, lw=1.2)
            ax.annotate(f"alarm after {ex['days']:.0f} d", (ex["alarm"], 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate(ex["fault"], (ex["start"], 0), xycoords=("data", "axes fraction"), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(ex["title"], loc="left")
        ax.set_ylabel(ex["ylabel"])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    _save(fig, path)


def performance_index(monthly: pd.DataFrame, path: Path):
    """Monthly WARP index with its 95 % interval and the neighbour index, small multiples per turbine."""
    _style()
    turbines = sorted(set(monthly.index.get_level_values("turbine")))
    fig, axes = plt.subplots(len(turbines), 1, figsize=(10, 1.45 * len(turbines) + 0.6), sharex=True, sharey=True)
    for ax, t in zip(axes, turbines):
        d = monthly.loc[t]
        x = pd.PeriodIndex(d.index, freq="M").to_timestamp()
        ax.fill_between(x, d["ci_low"], d["ci_high"], color=SERIES[0], alpha=0.12, lw=0, step="mid")
        ax.plot(x, d["index"], color=SERIES[0], lw=1.6, drawstyle="steps-mid", label="WARP index, 95 % interval")
        if "index_nb" in d:
            ax.plot(x, d["index_nb"], color=SERIES[1], lw=1.2, drawstyle="steps-mid", label="neighbour index")
        ax.axhline(1.0, color=AXIS, lw=1)
        ax.set_ylabel(f"T{t}", rotation=0, ha="right", va="center", color=INK)
        ax.set_ylim(0.88, 1.12)
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].set_title("Monthly performance index (actual / expected energy)", loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, path)


def yaw_tracker(monthly_delta: pd.DataFrame, fit: dict, path: Path):
    """Left: monthly median nacelle offset vs neighbours per turbine. Right: cos^2 fit for one turbine."""
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    x = monthly_delta.index.to_timestamp()
    for k, t in enumerate(monthly_delta.columns):
        a1.plot(x, monthly_delta[t], color=SERIES[k % len(SERIES)], lw=1.6, label=f"T{t}")
        a1.annotate(f"T{t}", (x[-1], monthly_delta[t].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    a1.axhline(0, color=AXIS, lw=1)
    a1.set_ylabel("nacelle − neighbours' direction, Δ [deg]")
    a1.set_title("Monthly median nacelle offset (baseline 2017 = 0)", loc="left")
    a1.legend(ncol=6, loc="upper left")
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    b = fit["binned"]
    a2.scatter(b.index, b.values, s=22, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="measured (2 deg bins)")
    xs = np.linspace(b.index.min(), b.index.max(), 200)
    a2.plot(xs, fit["A"] + fit["R"] * np.cos(2 * np.deg2rad(xs - fit["phi"])), color=SERIES[1], lw=1.6,
            label="A′ + R cos 2(Δ − φ)")
    a2.axvline(fit["phi"], color=MUTED, lw=1)
    a2.set_xlabel("Δ [deg]")
    a2.set_ylabel("P / P̂")
    a2.set_title(f"T{fit['turbine']}, {fit['year']}: power peaks at Δ = {round(fit['phi'], 1) + 0.0:+.1f}°", loc="left")
    a2.legend(loc="lower center")
    fig.tight_layout()
    _save(fig, path)
