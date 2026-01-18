"""
Fiber volley feature detection implemented as an Analyzer subclass.
"""
from __future__ import annotations

from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
from typing import Optional
import pandas as pd
import numpy as np


class FiberVolleyFeature(Feature):
    """Detect fiber volley extrema and amplitude for each stimulus intensity."""

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.window_ms = params.get("window_ms")  # ms
        if self.window_ms is None:
            raise ValueError("window_ms parameter must be specified for FiberVolleyFeature.")

    def run(self, context: RecordingContext) -> RecordingContext:
        fs = context.fs  # Hz
        fv_df = self.calculate(context.averaged, fs=fs)
        context.add_result(self.name, fv_df)
        return context

    def calculate(self, abf_df: pd.DataFrame, fs: float | None = None) -> pd.DataFrame:
        t0, t1 = [v / 1000.0 for v in self.window_ms]
        results = []

        for stim, g in abf_df.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            # Smooth the raw mean once; avoid re-smoothing the pre-smoothed column.
            y = self.apply_smoothing(g["mean"].to_numpy(), fs=fs)

            start_idx = int(np.searchsorted(x, t0))
            stop_idx = int(np.searchsorted(x, t1))
            y_w = y[start_idx:stop_idx]

            fv_idx = None
            # FV is negative-going: look for troughs (peaks on -y).
            neg_peaks, neg_props = emath.find_peaks(-y_w)
            if neg_peaks.size:
                if "prominences" in neg_props:
                    fv_min_rel = neg_peaks[np.argmax(neg_props["prominences"])]
                else:
                    fv_min_rel = neg_peaks[np.argmin(y_w[neg_peaks])]
                fv_idx = start_idx + fv_min_rel
        

            fv_amp = fv_v = fv_s = np.nan
            if fv_idx is not None:
                fv_s, fv_v = x[fv_idx], y[fv_idx]
            if np.isfinite(fv_v):
                fv_amp = abs(fv_v)

            results.append({
                "stim_intensity": stim,
                "fv_amp": fv_amp,
                "fv_s": fv_s,
                "fv_v": fv_v,
            })

        return pd.DataFrame(results)
