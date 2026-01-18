from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot

class SweepPlot(Plot):
    """
    Plots averaged sweeps with optional smoothing.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        abf_df = context.averaged
        fs = context.fs

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):

                fig, ax = plt.subplots()

                stim_order = list(self.stim_intensities) or list(pd.unique(abf_df["stim_intensity"]))
                cmap = plt.get_cmap(self.color_map)
                n_colors = max(len(stim_order), 1)

                for idx, stim in enumerate(stim_order):
                    g = abf_df.loc[abf_df["stim_intensity"] == stim]
                    if g.empty:
                        continue
                    color = cmap(idx / (n_colors - 1)) if n_colors > 1 else 'black'

                    if "mean" not in g.columns:
                        raise ValueError("Expected 'mean' column in averaged data.")

                    x = g["time"].to_numpy()
                    y = g["mean"].to_numpy()
                    y = self.apply_smoothing(y, fs=fs)

                    ax.plot(x, y, label=f"{stim}", color=color)

                ax.set_title('Evoked Field Potentials')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Response (mV)')
                ax.legend(title='Stimulus Intensity (µA)')
                ax.grid()
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:.0f}"))
                fig.tight_layout()
                return fig

    def render(self, context: RecordingContext) -> None:
        """
        Render the sweep plot for the given context.
        """
        self._build_figure(context)
        plt.show()

    def save(
        self,
        context: RecordingContext,
        output_path: Path | str,
        output_stem: str | None = None,
    ) -> Path:
        fig = self._build_figure(context)
        save_path = self._resolve_output_path(
            context,
            output_path,
            output_stem=output_stem,
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
