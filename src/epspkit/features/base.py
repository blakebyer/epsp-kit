from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from epspkit.core.config import FeatureConfig, SmoothingConfig
from epspkit.core.context import RecordingContext
from epspkit.core import math as emath


class Feature(ABC):
    """
    Base class for all epspkit feature analyzers.

    Subclasses should:
      - implement `run(context)` to compute their metrics
      - read config from `self.config.params`
      - optionally call `self.apply_smoothing(y, fs=context.fs)`
    """

    def __init__(self, config: FeatureConfig, effective_smoothing: SmoothingConfig | None = None):
        """
        Parameters
        ----------
        config
            Per-feature configuration (name, params, optional smoothing).

        effective_smoothing
            Final smoothing policy for this Feature instance, typically decided
            by the pipeline. Common patterns:

            - Use global smoothing:
                  feature = MyFeature(cfg, effective_smoothing=pipeline_cfg.global_smoothing)

            - Let feature override global if desired (in pipeline code):
                  eff = cfg.smoothing if cfg.smoothing.method != "none" else pipeline_cfg.global_smoothing
                  feature = MyFeature(cfg, effective_smoothing=eff)

            If not provided, we fall back to cfg.smoothing.
        """
        self.config = config
        self.name = config.name

        # Final smoothing policy for this feature instance:
        self.smoothing: SmoothingConfig = (
            effective_smoothing
            or config.smoothing
            or SmoothingConfig()
        )

    @abstractmethod
    def run(self, context: RecordingContext) -> RecordingContext:
        """Perform feature extraction on the given recording context."""
        raise NotImplementedError

    def apply_smoothing(self, y, fs: float | None = None):
        """
        Apply the configured smoothing method to a 1D trace y.

        If method == "none", returns y unchanged.

        Parameters
        ----------
        y
            1D array-like trace.
        fs
            Sampling rate in Hz. Required for Butterworth smoothing.
        """
        cfg = self.smoothing

        if cfg.method == "none":
            return y

        y_arr = np.asarray(y)

        if cfg.method == "moving_average":
            return emath.moving_average(y_arr, cfg.window_size)

        if cfg.method == "savgol":
            window = cfg.window_size
            poly = cfg.polyorder

            # enforce odd window and basic validity
            if window % 2 == 0:
                window += 1
            if window <= poly:
                raise ValueError(
                    f"Savgol requires window_size > polyorder "
                    f"(got window_size={window}, polyorder={poly})."
                )

            return emath.savgol(y_arr, window, poly)

        if cfg.method == "butter_lowpass":
            if fs is None:
                raise ValueError("Butterworth smoothing requires a sampling rate fs.")
            return emath.butter_lowpass(y_arr, cfg.cutoff, fs, order=cfg.order)

        raise ValueError(f"Unknown smoothing method: {cfg.method}")