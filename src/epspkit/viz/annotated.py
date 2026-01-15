from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot

class AnnotatedPlot(Plot):
    """
    Plots averaged sweeps with optional smoothing.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map
        self.smooth = config.smooth

    def render(self, context: RecordingContext) -> None:
        """
        Render the sweep plot for the given context.
        """

        abf_df = context.averaged
        fs = context.fs
        plt.style.use(self.style)
        plt.rcParams.update(self.rc_params)
        fig, ax = plt.subplots()

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

            x = g["time"].to_numpy()
            y = g["mean"].to_numpy()

            if self.smooth:
                y = self.apply_smoothing(y, fs=fs)

            ax.plot(x, y, label=f"Stim {stim} µA", color=color)

        ax.set_title('Averaged Sweeps')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.legend()
        ax.grid()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
        fig.tight_layout()
        plt.show()
