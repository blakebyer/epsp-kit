from __future__ import annotations

import numpy as np
import pandas as pd
from epspkit.core.context import RecordingContext

def crop_stim_artifact(
    context: RecordingContext,
    window_ms: tuple[float, float] = (0.0, 1.25)
) -> pd.DataFrame:
    """
    Crop stimulation artifacts from tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    window_ms
        Time window (start_ms, end_ms) in milliseconds to remove after stimulus onset.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with stimulation artifacts removed.
    """
    tidy_df = context.tidy
    t0, t1 = [v / 1000.0 for v in window_ms]

    def crop_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("time")
        x = g["time"].to_numpy()

        # time-based indices (robust to tiny dt rounding)
        start_idx = int(np.searchsorted(x, t0))
        stop_idx = int(np.searchsorted(x, t1))

        cropped = pd.concat([g.iloc[:start_idx], g.iloc[stop_idx:]], ignore_index=True)

        # re-zero time so all traces align perfectly
        cropped["time"] = cropped["time"] - float(cropped["time"].iloc[0])
        return cropped

    context.tidy = tidy_df.groupby(["stim_intensity", "abf_sweep"], group_keys=False)[["stim_intensity", "abf_sweep", "time", "voltage"]].apply(crop_group)
    return context

def template_subtract_stim_artifact(
    context: RecordingContext,
    window_ms: tuple[float, float] = (0.0, 1.25),
) -> pd.DataFrame:
    """
    Template subtract stimulation artifacts from tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    window_ms
        Time window (start_ms, end_ms) in milliseconds to remove after stimulus onset.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with stimulation artifacts removed.
    """
    tidy_df = context.tidy
    t0, t1 = [v / 1000.0 for v in window_ms]

    def subtract_template(g_int: pd.DataFrame) -> pd.DataFrame:
        # g_int = all sweeps for one stim_intensity
        g_int = g_int.sort_values(["abf_sweep", "time"]).copy()

        # Build template using columns only
        template_df = (
            g_int
            .groupby("time", sort=True)["voltage"]
            .mean()
            .reset_index()
        )

        t = template_df["time"].to_numpy()
        T = template_df["voltage"].to_numpy()

        start_idx = np.searchsorted(t, t0)
        stop_idx  = np.searchsorted(t, t1)

        denom = np.dot(T[start_idx:stop_idx], T[start_idx:stop_idx])
        if denom <= 1e-20:
            return g_int

        out = []
        for sw, g in g_int.groupby("abf_sweep", sort=False):
            g = g.sort_values("time").copy()
            x = g["time"].to_numpy()
            y = g["voltage"].to_numpy()

            # sanity: time grids must match
            # (can be removed once you're confident)
            if not np.allclose(x, t):
                raise ValueError("Time grid mismatch between sweep and template")

            num = np.dot(y[start_idx:stop_idx], T[start_idx:stop_idx])
            a = num / denom

            y[start_idx:stop_idx] -= a * T[start_idx:stop_idx]
            g["voltage"] = y
            out.append(g)

        return pd.concat(out, ignore_index=True)

    context.tidy = tidy_df.groupby("stim_intensity", group_keys=False)[["stim_intensity", "abf_sweep", "sweepNumber", "time", "voltage"]].apply(subtract_template)
    return context