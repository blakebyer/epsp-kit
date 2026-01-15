from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from epspkit.core.context import RecordingContext
from epspkit.core import math as emath
from epspkit.core.config import VizConfig, SmoothingConfig

class Plot(ABC):
    """
    Base class for all epspkit plotters.

    Subclasses should:
      - implement `render(context)` to create their plots
      - read config from `self.config.params`
    """

    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        """
        Parameters
        ----------
        config
            Per-plot configuration (name, params, optional smoothing).
        """
        self.config = config
        self.name = config.name
        self.stim_intensities = list(config.stim_intensities or [])
        
        # Final smoothing policy for this plot instance:
        self.smoothing: SmoothingConfig = (
            effective_smoothing
            or config.smoothing
            or SmoothingConfig()
        )

    @abstractmethod
    def render(self, context: RecordingContext) -> None:
        """Render the plot for the given context."""
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
