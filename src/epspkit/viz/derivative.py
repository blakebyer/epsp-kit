from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot
from epspkit.core import math as emath

class DerivativePlot(Plot):
    """
    Plots averaged sweeps and their derivatives with optional smoothing.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map
        self.smooth = config.smooth

    def render(self, context: RecordingContext) -> None:
        abf_df = context.averaged
        fs = context.fs

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):

                fig, (ax1, ax2) = plt.subplots(
                    2, 1, sharex=True,
                    gridspec_kw={"height_ratios": [2, 1]},
                )

                stim_order = list(self.stim_intensities) or list(pd.unique(abf_df["stim_intensity"]))
                cmap = plt.get_cmap(self.color_map)
                n_colors = max(len(stim_order), 1)

                for idx, stim in enumerate(stim_order):
                    g = abf_df.loc[abf_df["stim_intensity"] == stim]
                    if g.empty:
                        continue

                    color = cmap(idx / (n_colors - 1)) if n_colors > 1 else cmap(0.0)

                    if "mean" not in g.columns:
                        raise ValueError("Expected 'mean' column in averaged data.")

                    x = g["time"].to_numpy()     # seconds
                    y = g["mean"].to_numpy()

                    if self.smooth:
                        y = self.apply_smoothing(y, fs=fs)

                    dy = emath.gradient(y, fs=fs)   # mV/s
                    dy = dy / 1000.0                # convert to mV/ms (if x is seconds)

                    ax1.plot(x, y, label=f"Stim {stim} µA", color=color)
                    ax2.plot(x, dy, label=f"Stim {stim} µA", color=color)

                ax1.set_title("Averaged Sweeps")
                ax1.set_ylabel("Voltage (mV)")
                ax1.legend()
                ax1.grid(True)

                ax2.set_xlabel("Time (ms)")
                ax2.set_ylabel("Derivative (mV/ms)")
                ax2.legend()
                ax2.grid(True)

                # format shared x-axis ticks as ms even though x is seconds
                ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 1000:.0f}"))

                fig.tight_layout()
                plt.show()
