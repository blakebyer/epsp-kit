from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot
import warnings

class IOPlot(Plot):
    """
    Plots input output curves, including fiber volley amplitude vs stimulus intensity and EPSP slope vs stimulus intensity.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        fv_df = context.get_result("fiber_volley")
        epsp_df = context.get_result("epsp")

        fv_ok = fv_df is not None and not fv_df.empty
        epsp_ok = epsp_df is not None and not epsp_df.empty

        if not fv_ok and not epsp_ok:
            raise ValueError("IOPlot requires fiber_volley or epsp results.")

        if not fv_ok or not epsp_ok:
            warnings.warn(
                "IOPlot: only one result dataframe present; "
                "plotting available relationships only.",
                RuntimeWarning,
                stacklevel=2,
            )

        ncols = (fv_ok and epsp_ok) + fv_ok + epsp_ok

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):
                fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
                if ncols == 1:
                    axes = [axes]

                axes = list(axes)

                # 1) epsp_slope vs fv_amp
                if fv_ok and epsp_ok:
                    ax = axes.pop(0)
                    ax.scatter(fv_df["fv_amp"], epsp_df["epsp_slope"])
                    ax.set_xlabel("Fiber Volley Amplitude (mV)")
                    ax.set_ylabel("fEPSP Slope (mV/ms)")
                    ax.set_title("Synaptic Strength")
                    ax.grid(True)

                # 2) fv_amp vs stim_intensity
                if fv_ok:
                    ax = axes.pop(0)
                    ax.scatter(fv_df["stim_intensity"], fv_df["fv_amp"])
                    ax.set_xlabel("Stimulus Intensity (µA)")
                    ax.set_ylabel("Fiber Volley Amplitude (mV)")
                    ax.set_title("Presynaptic Excitability")
                    ax.grid(True)

                # 3) epsp_slope vs stim_intensity
                if epsp_ok:
                    ax = axes.pop(0)
                    ax.scatter(epsp_df["stim_intensity"], epsp_df["epsp_slope"])
                    ax.set_xlabel("Stimulus Intensity (µA)")
                    ax.set_ylabel("fEPSP Slope (mV/ms)")
                    ax.set_title("Postsynaptic Responsiveness")
                    ax.grid(True)

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
