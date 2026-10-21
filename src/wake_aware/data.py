"""Kelmarsh SCADA as time x turbine arrays, plus the turbine layout.

Reading, cleaning and the two reference indices of the earlier project are reused from the ``wind_power_curve``
package (github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh): every 10-minute record is classified
(normal operation, stops, curtailment, icing, sensor faults), the wind speed is rebuilt consistently as the mean
of the two nacelle anemometers and normalised to 1.225 kg/m3, and the neighbour reference ``p_ratio`` and the
own-power-curve reference ``p_curve`` are computed against the baseline year. They serve as baselines here.

Usage:
    farm = load_farm(Path("data"), years=[2017, 2018, 2019, 2020, 2021], baseline_year=2017)
    farm.power.shape       # (n_timestamps, 6)
    farm.period(2019)      # boolean mask over timestamps
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from wind_power_curve.performance import reference_curves
from wind_power_curve.pipeline import prepare
from wind_power_curve.scada import load, read_static

RATED_KW = 2050.0
EARTH_RADIUS_M = 6_371_000.0
# columns of the prepared frame that are kept, and their names here
COLUMNS = {
    "power_kw": "power",
    "ws_n": "ws",  # consistent nacelle wind speed (mean of both anemometers), density normalised
    "ws_scada": "ws_scada",  # the raw SCADA wind speed signal (changes definition in April 2018)
    "ws_std": "ws_std",  # 10-minute standard deviation of the SCADA wind speed
    "rho": "rho",
    "nacelle_pos": "nacelle",
    "vane": "vane",
    "p_ratio": "p_ratio",
    "p_curve": "p_curve",
}
log = logging.getLogger(__name__)


@dataclass
class Farm:
    """SCADA signals as (n_timestamps, n_turbines) arrays on a complete 10-minute grid (NaN where missing)."""

    time: pd.DatetimeIndex
    turbines: np.ndarray
    x: np.ndarray  # m east of the farm centre
    y: np.ndarray  # m north of the farm centre
    hub_height: np.ndarray
    diameter: float
    rated_kw: float
    power: np.ndarray
    ws: np.ndarray
    ws_scada: np.ndarray
    ws_std: np.ndarray
    rho: np.ndarray
    nacelle: np.ndarray
    vane: np.ndarray
    normal: np.ndarray  # bool: normal operation (wind_power_curve cleaning)
    operating: np.ndarray  # bool: rotor running, i.e. casting a wake
    p_ratio: np.ndarray  # neighbour reference of wind_power_curve (partial load only)
    p_curve: np.ndarray  # own baseline power curve at the own anemometer (wind_power_curve)
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.turbines)

    def period(self, start, end=None) -> np.ndarray:
        """Mask for a year (``period(2019)``), a range of years (``period(2019, 2020)``) or timestamps."""
        if isinstance(start, (int, np.integer)):
            end = start if end is None else end
            start, end = pd.Timestamp(f"{start}-01-01", tz="UTC"), pd.Timestamp(f"{end + 1}-01-01", tz="UTC")
        return np.asarray((self.time >= start) & (self.time < end))

    def day_index(self) -> np.ndarray:
        """Integer day number (UTC) of every timestamp, counted from 1970-01-01."""
        days = self.time.tz_convert("UTC").normalize()
        return ((days - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1D")).to_numpy()

    @property
    def turbulence(self) -> np.ndarray:
        """Turbulence intensity measured on the nacelle, ws_std / ws (SCADA signal), NaN below 3 m/s."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ti = self.ws_std / self.ws_scada
        return np.where(self.ws_scada >= 3.0, ti, np.nan)

    def copy(self) -> "Farm":
        arrays = {k: getattr(self, k).copy() for k in ("power", "ws", "ws_scada", "nacelle", "vane", "p_curve")}
        return replace(self, **arrays)


def layout(static_csv: Path) -> pd.DataFrame:
    """Turbine positions in metres (x east, y north) around the farm centre, from the static metadata file."""
    df = pd.read_csv(static_csv, encoding="utf-8-sig")
    lat0, lon0 = np.deg2rad(df["Latitude"].mean()), np.deg2rad(df["Longitude"].mean())
    lat, lon = np.deg2rad(df["Latitude"]), np.deg2rad(df["Longitude"])
    return pd.DataFrame({
        "turbine": df["Title"].str.extract(r"(\d+)$", expand=False).astype(int),
        "x_m": EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0),
        "y_m": EARTH_RADIUS_M * (lat - lat0),
        "hub_height_m": df["Hub Height (m)"],
        "rotor_diameter_m": df["Rotor Diameter (m)"],
    }).set_index("turbine").sort_index()


def to_farm(df: pd.DataFrame, lay: pd.DataFrame, rated_kw: float = RATED_KW) -> Farm:
    """Prepared (turbine, timestamp) frame of wind_power_curve -> Farm on a complete 10-minute grid."""
    ts = df.index.get_level_values("timestamp")
    grid = pd.date_range(ts.min(), ts.max(), freq="10min", name="timestamp")
    turbines = np.array(sorted(df.index.get_level_values("turbine").unique()))

    def wide(col):
        return df[col].unstack("turbine").reindex(index=grid, columns=turbines).to_numpy(dtype=float)

    arrays = {dst: wide(src) for src, dst in COLUMNS.items()}
    state = df["state"].astype(str).unstack("turbine").reindex(index=grid, columns=turbines)
    normal = state.eq("normal").to_numpy()
    stopped = state.isin(["stop", "stop_grid"]).to_numpy() | (arrays["power"] <= 0)
    lay = lay.loc[turbines]
    return Farm(time=grid, turbines=turbines, x=lay["x_m"].to_numpy(), y=lay["y_m"].to_numpy(),
                hub_height=lay["hub_height_m"].to_numpy(), diameter=float(lay["rotor_diameter_m"].iloc[0]),
                rated_kw=rated_kw, normal=normal, operating=~stopped, **arrays)


def _year(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")


def load_farm(data_dir: Path, years: list[int], baseline_year: int, cache_dir: Path | None = None,
              refresh: bool = False) -> Farm:
    """Load the Kelmarsh SCADA for ``years`` (wind_power_curve Parquet cache or archives) and prepare it.

    The prepared frame is cached as Parquet in ``cache_dir`` (default ``data_dir/processed``).
    """
    cache_dir = data_dir / "processed" if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"wake_aware_{min(years)}_{max(years)}_b{baseline_year}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
    else:
        scada, status = load(sorted(years), data_dir, refresh=refresh)
        df, _ = prepare(scada, status, read_static(data_dir), _year(baseline_year))
        df = df[list(COLUMNS) + ["state"]]
        df.to_parquet(cache)
        log.info("prepared %d records, cached in %s", len(df), cache)
    farm = to_farm(df, layout(data_dir / "raw" / "Kelmarsh_WT_static.csv"))
    # own baseline curves of wind_power_curve, to recompute p_curve when an anemometer fault is injected
    farm.extra["own_curves"] = {t: (c["ws"].to_numpy(), c["power_kw"].to_numpy())
                                for t, c in reference_curves(df, _year(baseline_year)).items()}
    return farm
