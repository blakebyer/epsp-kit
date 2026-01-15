"""
fEPSP slope feature extraction module.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from typing import Optional
import pandas as pd
import numpy as np

class EPSPFeature(Feature):
    """
    Computes EPSP minima and slopes from averaged traces.
    """
    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.window_ms: float = params.get("window_ms", (1.5,5.0))  # ms
        self.fit_distance = int(params.get("fit_distance", 4))  # points

    def run(self, context: RecordingContext) -> RecordingContext:
        """
        Run EPSP analysis and attach results to the RecordingContext.
        """
        fv_res = context.get_result("fiber_volley")
        fs = context.fs  # Hz
        epsp_df = self.calculate(context.averaged, fv_res, fs=fs)
        context.add_result(self.name, epsp_df)
        return context

    def calculate(
        self,
        abf_df: pd.DataFrame,
        fv_df: Optional[pd.DataFrame],
        fs: float | None = None,
    ) -> pd.DataFrame:
        t0, t1 = [v / 1000.0 for v in self.window_ms]
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            # Smooth the raw mean once; avoid re-smoothing the pre-smoothed column.
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)

            start_idx = int(np.searchsorted(x, t0))
            stop_idx = int(np.searchsorted(x, t1))
            dy_full = emath.gradient(y, x)

            epsp_segment = slice(start_idx, stop_idx)
            slope_center_idx = start_idx + int(np.argmin(dy_full[epsp_segment]))
            epsp_idx = start_idx + int(np.argmin(y[epsp_segment]))

            if fv_df is not None and (fv_df.stim_intensity == stim).any():
                fv_row = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
                fv_amp = fv_row["fv_amp"]
            else:
                fv_amp = np.nan

            epsp_s, epsp_v = x[epsp_idx], y[epsp_idx]
            epsp_amp = abs(epsp_v)

            i0 = max(0, slope_center_idx - self.fit_distance)
            i1 = min(len(y) - 1, slope_center_idx + self.fit_distance)
            t_win = x[i0:i1 + 1] - x[slope_center_idx]
            v_win = y[i0:i1 + 1]
            m, b = np.polyfit(t_win, v_win, 1)
            y_pred = m * t_win + b
            ss_res = np.sum((v_win - y_pred) ** 2)
            ss_tot = np.sum((v_win - v_win.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            results.append({
                "stim_intensity": stim,
                "epsp_s": float(epsp_s),
                "epsp_v": float(epsp_v),
                "slope_mid_s": float(x[slope_center_idx]),
                "slope_mid_v": float(y[slope_center_idx]),
                "epsp_amp": float(epsp_amp),
                "epsp_to_fv": float(abs(m) / fv_amp) if fv_amp and m else np.nan,
                "epsp_slope": float(abs(m)),
                "epsp_slope_ms": float(abs(m) / 1000.0),
                "epsp_r2": float(r2),
            })

        return pd.DataFrame(results)