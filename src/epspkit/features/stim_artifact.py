"""
Module for detecting and removing stimulation artifacts.
"""
from __future__ import annotations
from epspkit.features.base import Feature
from epspkit.core.context import RecordingContext
from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core import math as emath
import pandas as pd
import numpy as np
from typing import Optional

class StimArtifactFeature(Feature):
    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        params = self.config.params
        self.window_ms : float = params.get("window_ms", (0.0,1.25))  # ms

    def run(self, context: RecordingContext) -> RecordingContext:
        
        cropped_tidy = self.calculate(context.tidy)
        context.tidy = cropped_tidy

        context.add_result(self.name, cropped_tidy)
        return context
    
    def calculate(self, tidy_df: pd.DataFrame) -> pd.DataFrame:
        
        t0, t1 = [v / 1000.0 for v in self.window_ms]
        
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

        return tidy_df.groupby(["stim_intensity", "sweep"], group_keys=False).apply(crop_group)