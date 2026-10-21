"""Single-wake deficit tables from an engineering wake model (PyWake) and their use on SCADA time series.

For every ordered pair of turbines (j upstream, i downstream) the rotor-averaged velocity deficit d_ji that j
alone causes at i is computed with the Gaussian wake of Bastankhah & Porte-Agel (2014) for a grid of wind
directions, free-stream wind speeds and wake expansion rates k. At run time the single wakes of the turbines
that are actually running are combined by root-sum-square, the superposition usually paired with this model:

    d_i(t) = sqrt( sum_{j != i} o_j(t) * d_ji(theta(t), U(t), k)^2 )        o_j = 1 if turbine j is running

Building the farm deficit from single wakes keeps it correct when upstream turbines are stopped. It ignores the
second-order effect that a waked upstream turbine sees less wind and so has a slightly different thrust.

The tables are smoothed over wind direction with a Gaussian kernel of width sigma_theta: a 10-minute mean
direction estimated from nacelle positions is uncertain by several degrees and the wakes meander, so measured
wake losses are wider and shallower than the instantaneous model wake.

Usage:
    tables = pair_tables(farm.x, farm.y, farm.hub_height, farm.diameter, farm.rated_kw)   # (k, j, i, wd, ws)
    smooth = smooth_direction(tables, sigma_deg=6.0)
    d = farm_deficit(smooth, theta, ws, k=0.04, operating=farm.operating)                  # (time, turbine)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

WD_GRID = np.arange(0.0, 360.0, 1.0)
WS_GRID = np.arange(3.0, 25.01, 0.5)
K_GRID = np.round(np.arange(0.01, 0.0801, 0.005), 4)


def pair_tables(x, y, hub_height, diameter: float, rated_kw: float, k_values=K_GRID, wd=WD_GRID,
                ws=WS_GRID) -> np.ndarray:
    """Deficit ``[k, j, i, wd, ws]`` that turbine j alone causes at turbine i (zero on the diagonal).

    The turbine is PyWake's generic pitch-regulated turbine of the given rotor diameter and rated power; only
    its thrust curve matters here. The deficit is averaged over the downstream rotor (4-point CGI quadrature).
    Wind directions follow the meteorological convention (direction the wind
    comes from, clockwise from north) with x pointing east and y north.
    """
    from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
    from py_wake.rotor_avg_models import CGIRotorAvg
    from py_wake.site import UniformSite
    from py_wake.superposition_models import SquaredSum
    from py_wake.wind_farm_models import PropagateDownwind
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

    x, y, h = (np.asarray(a, dtype=float) for a in (x, y, hub_height))
    n = len(x)
    turbine = GenericWindTurbine("generic", diameter, float(np.median(h)), rated_kw)
    site = UniformSite(p_wd=[1.0], ti=0.1)
    out = np.zeros((len(k_values), n, n, len(wd), len(ws)))
    for ik, k in enumerate(k_values):
        deficit = BastankhahGaussianDeficit(k=float(k), rotorAvgModel=CGIRotorAvg(4))
        wfm = PropagateDownwind(site, turbine, wake_deficitModel=deficit, superpositionModel=SquaredSum())
        for j in range(n):
            for i in range(j + 1, n):
                sim = wfm(x[[j, i]], y[[j, i]], h=h[[j, i]], wd=wd, ws=ws)
                ws_eff = np.asarray(sim.WS_eff.values)  # (2, wd, ws)
                out[ik, j, i] = 1.0 - ws_eff[1] / ws[None, :]  # at i, caused by j
                out[ik, i, j] = 1.0 - ws_eff[0] / ws[None, :]  # at j, caused by i
    return np.clip(out, 0.0, 1.0)


def smooth_direction(tables: np.ndarray, sigma_deg: float, wd_step: float = 1.0) -> np.ndarray:
    """Gaussian smoothing over the (periodic) wind direction axis, the second last axis."""
    if sigma_deg <= 0:
        return tables
    return gaussian_filter1d(tables, sigma_deg / wd_step, axis=-2, mode="wrap")


def _weights(values, grid):
    """Lower grid index and linear weight of ``values`` on an evenly spaced ``grid`` (clipped to its ends)."""
    f = np.clip((np.asarray(values, dtype=float) - grid[0]) / (grid[1] - grid[0]), 0.0, len(grid) - 1 - 1e-9)
    lo = np.floor(np.nan_to_num(f)).astype(int)
    return lo, f - lo


def farm_deficit(tables: np.ndarray, theta, ws, k, operating=None, k_grid=K_GRID, ws_grid=WS_GRID,
                 wd_step: float = 1.0) -> np.ndarray:
    """Combined deficit at every turbine, shape (time, turbine).

    ``theta`` and ``ws`` are (time,) arrays of the wind direction and free-stream speed used to look up the
    tables; ``k`` is a scalar or a (time,) array; ``operating`` (time, turbine) marks turbines whose wake counts
    (default: all). NaN direction or speed gives NaN.
    """
    theta, ws = np.asarray(theta, dtype=float), np.asarray(ws, dtype=float)
    n_t, n = len(theta), tables.shape[1]
    k = np.broadcast_to(np.asarray(k, dtype=float), (n_t,))
    kl, kw = _weights(k, k_grid)
    kh = np.minimum(kl + 1, len(k_grid) - 1)
    sl, sw = _weights(ws, ws_grid)
    sh = np.minimum(sl + 1, len(ws_grid) - 1)
    n_wd = tables.shape[-2]
    wd_idx = np.round(np.nan_to_num(theta) / wd_step).astype(int) % n_wd
    on = np.ones((n_t, n)) if operating is None else np.where(np.isnan(operating.astype(float)), 1.0, operating)
    total = np.zeros((n_t, n))
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            t = tables[:, j, i]
            val = ((1 - kw) * ((1 - sw) * t[kl, wd_idx, sl] + sw * t[kl, wd_idx, sh])
                   + kw * ((1 - sw) * t[kh, wd_idx, sl] + sw * t[kh, wd_idx, sh]))
            total[:, i] += on[:, j] * val ** 2
    out = np.sqrt(total)
    out[np.isnan(theta) | np.isnan(ws)] = np.nan
    return out
