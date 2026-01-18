"""
Module for detecting population spike features.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
import pandas as pd
import numpy as np

class PopSpikeFeature(Feature):
    """Detect fiber volley extrema and amplitude for each stimulus intensity."""

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.ps_lag = params.get("lag_ms")  # ms
        self.prominence = params.get("prominence")  # mV
        missing = [
            name
            for name, value in {
                "lag_ms": self.ps_lag,
                "prominence": self.prominence
            }.items()
            if value is None
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Missing required PopSpikeFeature parameters: "
                f"{missing_str}."
            )

    def run(self, context: RecordingContext) -> RecordingContext:
        fs = context.fs  # Hz
        epsp_df = context.get_result("epsp")
        if epsp_df is None or epsp_df.empty:
            raise ValueError("EPSPFeature result is required for PopSpikeFeature.")
        ps_df = self.calculate(context.averaged, epsp_df, fs=fs)
        context.add_result(self.name, ps_df)
        return context

    def calculate(
    self,
    abf_df: pd.DataFrame,
    epsp_df: pd.DataFrame,
    fs: float | None = None,
) -> pd.DataFrame:

        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)

            epsp_row = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            t_s = epsp_row["epsp_s"]
            v_s = epsp_row["epsp_v"]

            i0 = np.searchsorted(x, t_s)
            i1 = np.searchsorted(x, t_s + self.ps_lag / 1000.0)

            dy = emath.gradient(y, x)
            y_w, dy_w = y[i0:i1], dy[i0:i1]

            ps_idx = None

            # ---- 1. True peak detection ----
            peaks, props = emath.find_peaks(y_w, prominence=self.prominence)
            if peaks.size:
                ps_rel = peaks[np.argmax(props["prominences"])]
                ps_idx = i0 + ps_rel

            ps_amp = ps_s = ps_v = np.nan

            if ps_idx is not None:
                t_p = x[ps_idx]
                v_p = y[ps_idx]

                # ---- 3. Find post-PS baseline anchor ----
                after = slice(ps_idx + 1, i1)
                if after.start < after.stop:
                    b_rel = np.argmin(y[after])

                    b_idx = after.start + b_rel
                    t_b = x[b_idx]
                    v_b = y[b_idx]

                    # ---- 4. Baseline line ----
                    m = (v_b - v_s) / (t_b - t_s)
                    c = v_s - m * t_s
                    v_base = m * t_p + c

                    ps_amp = abs(v_p - v_base)
                    ps_s, ps_v = t_p, v_p

            results.append({
                "stim_intensity": stim,
                "ps_amp": ps_amp,
                "ps_s": ps_s,
                "ps_v": ps_v,
            })

        return pd.DataFrame(results)

