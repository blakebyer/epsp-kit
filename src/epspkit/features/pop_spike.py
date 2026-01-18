"""
Module for detecting population spike features.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from typing import Optional
import pandas as pd
import numpy as np

class PopSpikeFeature(Feature):
    """Detect fiber volley extrema and amplitude for each stimulus intensity."""

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.ps_lag = params.get("lag_ms")  # ms
        self.prominence = params.get("prominence")  # mV
        self.threshold = params.get("threshold")  # mV/ms
        missing = [
            name
            for name, value in {
                "lag_ms": self.ps_lag,
                "prominence": self.prominence,
                "threshold": self.threshold,
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

    def calculate(self, abf_df: pd.DataFrame, epsp_df: pd.DataFrame, fs: float | None = None) -> pd.DataFrame:
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            # Smooth the raw mean once; avoid re-smoothing the pre-smoothed column.
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)

            epsp_row = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            epsp_s = epsp_row["epsp_s"]
            epsp_v = epsp_row["epsp_v"]

            start_idx = int(np.searchsorted(x, epsp_s))
            stop_idx = int(np.searchsorted(x, epsp_s + self.ps_lag / 1000.0))
            dy = emath.gradient(y, x)
            y_w, dy_w = y[start_idx:stop_idx], dy[start_idx:stop_idx]

            ps_idx = None
            # PS is positive-going: look for peaks in y
            pos_peaks, _ = emath.find_peaks(y_w, prominence=self.prominence)
            if pos_peaks.size:
                ps_rel = pos_peaks[np.argmax(y_w[pos_peaks])]
                ps_idx = start_idx + ps_rel
            else: # if no peaks, try curvature detection
                slope_peaks, _ = emath.find_peaks(dy_w)
                if slope_peaks.size:
                    ps_start = slope_peaks[np.argmax(dy_w[slope_peaks])]  # max positive slope
                    zs = np.where(np.abs(dy_w) < self.threshold)[0]        # near-zero slopes
                    zs = zs[zs > ps_start]
                    if zs.size:
                        ps_end = zs[np.argmin(np.abs(dy_w[zs]))]          # closest to zero
                        j = ps_start + np.argmax(y_w[ps_start:ps_end+1])  # hump top
                        if (y_w[j] - y_w[ps_start]) >= self.prominence:
                            ps_idx = start_idx + j

            ps_amp = ps_amp = ps_s = ps_v = np.nan
            if ps_idx is not None:
                ps_s, ps_v = x[ps_idx], y[ps_idx]
                if np.isfinite(ps_v) and np.isfinite(epsp_v):
                    ps_amp = abs(epsp_v - ps_v)

            results.append({
                "stim_intensity": stim,
                "ps_amp": ps_amp,
                "ps_s": ps_s,
                "ps_v": ps_v
            })

        return pd.DataFrame(results)
