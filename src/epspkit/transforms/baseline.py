from __future__ import annotations

import numpy as np
import pandas as pd
from epspkit.core.context import RecordingContext

def baseline_correction(
    context: RecordingContext,
    baseline_window_ms: tuple[float, float] = (0.0, 0.1)
) -> pd.DataFrame:
    """
    Apply baseline correction to tidy DataFrame.

    Parameters
    ----------
    context
        RecordingContext object containing the tidy DataFrame.
    baseline_window_ms
        Time window (start_ms, end_ms) in milliseconds to use for baseline calculation.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with baseline-corrected voltage values.
    """
    tidy_df = context.tidy
    t_start, t_end = [v / 1000.0 for v in baseline_window_ms]

    def correct_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("time")
        x = g["time"].to_numpy()
        y = g["voltage"].to_numpy()

        # time-based indices (robust to tiny dt rounding)
        start_idx = int(np.searchsorted(x, t_start))
        stop_idx = int(np.searchsorted(x, t_end))

        baseline = np.mean(y[start_idx:stop_idx])
        g["voltage"] = g["voltage"] - baseline
        return g

    context.tidy = tidy_df.groupby(["stim_intensity", "sweep"], group_keys=False).apply(correct_group)
    return context