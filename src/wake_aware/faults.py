"""Synthetic faults injected into real SCADA records of one turbine, for validation with known ground truth.

    yaw          static yaw misalignment of delta degrees: the vane is offset, so the controller keeps the
                 rotor delta away from the wind. The nacelle position shifts by delta, and the power follows a
                 cos^n law of the true inflow angle (vane reading + delta), applied as an effective wind speed
                 reduction so that the turbine still reaches rated power in strong wind:
                     r = [cos(v + delta) / cos(v)]^(n/3),   P' = P * F(r u) / F(u)       (n = 2)
    anemometer   calibration error: both nacelle anemometers read (1 + b) times the true value; power unchanged.
    derate       power limited to (1 - s) x rated power (e.g. a temperature or parameter limit).
    loss         power reduced by a factor (1 - s) at all wind speeds (e.g. blade damage, pitch offset).

Only the faulty turbine's own records change, and records outside [start, end) are untouched. Stops and other
non-normal states keep their classification, as the cleaning step would see them unchanged for these faults.

Usage:
    bad = inject(farm, turbine=2, start=pd.Timestamp("2019-05-01", tz="UTC"), end=..., fault=Fault("yaw", 8.0),
                 curve=model.curve)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wake_aware.freestream import power_from_wind

YAW_EXPONENT = 2.0
VANE_CLIP_DEG = 30.0


@dataclass(frozen=True)
class Fault:
    kind: str  # "yaw" [deg], "anemometer", "derate", "loss" [fraction]
    size: float

    @property
    def label(self) -> str:
        if self.kind == "yaw":
            return f"yaw {abs(self.size):g} deg"
        if self.kind == "anemometer":
            return f"anemometer {self.size * 100:+g} %"
        return f"{'derating' if self.kind == 'derate' else 'power loss'} {self.size * 100:g} %"


SCENARIOS = [Fault("yaw", 4.0), Fault("yaw", 8.0), Fault("anemometer", 0.03), Fault("anemometer", 0.06),
             Fault("derate", 0.05), Fault("loss", 0.03)]


def yaw_power_factor(vane, ws, delta_deg: float, curve: pd.DataFrame, n: float = YAW_EXPONENT) -> np.ndarray:
    """Factor P'/P for a static misalignment of ``delta_deg`` given the measured vane angle and wind speed."""
    v = np.deg2rad(np.clip(np.nan_to_num(np.asarray(vane, dtype=float)), -VANE_CLIP_DEG, VANE_CLIP_DEG))
    d = np.deg2rad(delta_deg)
    r = (np.clip(np.cos(v + d), 0.0, None) / np.cos(v)) ** (n / 3.0)
    p0 = power_from_wind(ws, curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = power_from_wind(ws * r, curve) / p0
    direct = r ** 3  # below cut-in of the curve (or no wind reading): plain cos^n of the inflow
    return np.where(np.isfinite(factor) & (p0 > 10.0), factor, direct)


def inject(farm, turbine: int, start: pd.Timestamp, end: pd.Timestamp, fault: Fault, curve: pd.DataFrame | None = None):
    """Copy of ``farm`` with ``fault`` injected into turbine column ``turbine`` for timestamps in [start, end)."""
    out = farm.copy()
    w = np.asarray((farm.time >= start) & (farm.time < end))
    i = turbine
    if fault.kind == "yaw":
        if curve is None:
            raise ValueError("the yaw fault needs the farm power curve")
        out.power[w, i] = farm.power[w, i] * yaw_power_factor(farm.vane[w, i], farm.ws[w, i], fault.size, curve)
        out.nacelle[w, i] = (farm.nacelle[w, i] - fault.size) % 360.0
    elif fault.kind == "anemometer":
        out.ws[w, i] = farm.ws[w, i] * (1.0 + fault.size)
        out.ws_scada[w, i] = farm.ws_scada[w, i] * (1.0 + fault.size)
        own = farm.extra.get("own_curves", {}).get(farm.turbines[i])
        if own is not None:  # the own-curve baseline sees the biased anemometer
            out.p_curve[w, i] = np.interp(out.ws[w, i], own[0], own[1], left=0.0, right=own[1][-1])
    elif fault.kind == "derate":
        out.power[w, i] = np.minimum(farm.power[w, i], (1.0 - fault.size) * farm.rated_kw)
    elif fault.kind == "loss":
        out.power[w, i] = farm.power[w, i] * (1.0 - fault.size)
    else:
        raise ValueError(f"unknown fault kind {fault.kind!r}")
    return out
