"""Synthetic wind farm for the tests: four turbines on an east-west line, analytic wakes, known site factors."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from wake_aware.data import Farm
from wake_aware.freestream import power_from_wind
from wake_aware.wake import K_GRID, WD_GRID, WS_GRID

RATED = 2000.0
CURVE = pd.DataFrame({"ws": np.arange(3.0, 25.5, 0.5)})
CURVE["power_kw"] = np.clip(RATED * ((CURVE["ws"] - 3.0) / 9.0) ** 2.5, 0.0, RATED)
SPACING = 400.0
WAKE_DEPTH = 0.3
WAKE_WIDTH_DEG = 8.0


def analytic_tables(n: int) -> np.ndarray:
    """Single wakes on an east-west line: j wakes i when the wind blows from j towards i (Gaussian in direction)."""
    out = np.zeros((len(K_GRID), n, n, len(WD_GRID), len(WS_GRID)), dtype=np.float32)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            direction = 270.0 if i > j else 90.0  # wind from the west wakes the turbines further east
            dist = abs(i - j)
            diff = (WD_GRID - direction + 180.0) % 360.0 - 180.0
            depth = WAKE_DEPTH / dist
            out[:, j, i] = (depth * np.exp(-0.5 * (diff / WAKE_WIDTH_DEG) ** 2))[None, :, None]
    return out


def make_farm(days: int = 120, n: int = 4, seed: int = 0, site=(1.0, 1.03, 0.98, 1.0), noise: float = 0.01) -> Farm:
    """Synthetic SCADA: random-walk wind speed and direction, wakes from ``analytic_tables``, power from CURVE."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2019-01-01", periods=days * 144, freq="10min", tz="UTC", name="timestamp")
    t = len(time)
    ws_free = np.clip(8.0 + lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.3, t)), 2.0, 20.0)  # AR(1), std ~3 m/s
    wd = (np.cumsum(rng.normal(0, 3.0, t)) + 200.0) % 360.0
    tables = analytic_tables(n)
    wd_idx = np.round(wd).astype(int) % 360
    deficit = np.zeros((t, n))
    for i in range(n):
        tot = np.zeros(t)
        for j in range(n):
            if j != i:
                tot += tables[0, j, i, wd_idx, 10] ** 2
        deficit[:, i] = np.sqrt(tot)
    ws_local = ws_free[:, None] * (1 - deficit) * np.asarray(site)[None, :]
    power = power_from_wind(ws_local, CURVE) * (1 + rng.normal(0, noise, (t, n)))
    nacelle = (wd[:, None] + rng.normal(0, 3.0, (t, n))) % 360.0
    ones = np.ones((t, n), dtype=bool)
    ws_meas = ws_local * (1 + rng.normal(0, noise, (t, n)))
    return Farm(time=time, turbines=np.arange(1, n + 1), x=np.arange(n) * SPACING, y=np.zeros(n),
                hub_height=np.full(n, 80.0), diameter=92.0, rated_kw=RATED, power=power, ws=ws_meas,
                ws_scada=ws_meas.copy(), ws_std=0.1 * ws_meas, rho=np.full((t, n), 1.225), nacelle=nacelle,
                vane=rng.normal(0, 3.0, (t, n)), normal=ones, operating=ones.copy(),
                p_ratio=power_from_wind(ws_local, CURVE), p_curve=power_from_wind(ws_local, CURVE),
                extra={"wd_true": wd, "ws_free": ws_free, "deficit": deficit, "tables": tables})


@pytest.fixture(scope="session")
def farm():
    return make_farm()
