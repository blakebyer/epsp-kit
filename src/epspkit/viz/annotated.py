from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter

from epspkit.core.config import SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.viz.base import Plot

class AnnotatedPlot(Plot):
    """
    Plots averaged sweeps with feature annotations.
    """
    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        super().__init__(config, effective_smoothing)
        self.rc_params = config.rc_params or {}
        self.style = config.style
        self.color_map = config.color_map

    def _build_figure(self, context: RecordingContext) -> plt.Figure:
        abf_df = context.averaged
        fs = context.fs
        fv_df = context.get_result("fiber_volley")
        epsp_df = context.get_result("epsp")
        ps_df = context.get_result("pop_spike")
        if not any(isinstance(df, pd.DataFrame) and not df.empty for df in [fv_df, epsp_df, ps_df]):
            raise ValueError(
                "AnnotatedPlot requires at least one feature result "
                "(e.g., fiber_volley, epsp, or pop_spike)."
            )

        with plt.style.context(self.style):
            with plt.rc_context(self.rc_params):
                
                fig, ax = plt.subplots()
                used_labels: set[str] = set()

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

                    ax.plot(x, y, label=f"{stim} µA", color=color)
                    self.annotate_features(ax, stim, fv_df, epsp_df, ps_df, used_labels)

                ax.set_title('Evoked Field Potentials with Annotations')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Voltage (mV)')
                ax.legend()
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

    def annotate_features(
        self,
        ax: plt.Axes,
        stim: float,
        fv_df: pd.DataFrame | None,
        epsp_df: pd.DataFrame | None,
        ps_df: pd.DataFrame | None,
        used_labels: set[str],
    ) -> None:
        fv_color = to_rgba("darkviolet", alpha=0.7)
        epsp_slope_color = to_rgba("firebrick", alpha=0.8)
        epsp_min_color = to_rgba("royalblue", alpha=0.8)
        ps_color = to_rgba("darkorange", alpha=0.8)

        if isinstance(fv_df, pd.DataFrame) and not fv_df.empty and (fv_df.stim_intensity == stim).any():
            r = fv_df.loc[fv_df.stim_intensity == stim].iloc[0]
            fv_s, fv_v = r.get("fv_s"), r.get("fv_v")
            if pd.notna(fv_s) and pd.notna(fv_v):
                label = "Fiber Volley" if "Fiber Volley" not in used_labels else None
                ax.scatter(
                    [fv_s],
                    [fv_v],
                    s=70,
                    marker="v",
                    facecolors=fv_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                    label=label,
                )
                used_labels.add("Fiber Volley")
            else:
                fv_s, fv_v = r.get("fv_s"), r.get("fv_v")
                if pd.notna(fv_s) and pd.notna(fv_v):
                    label = "Fiber Volley" if "Fiber Volley" not in used_labels else None
                    ax.scatter(
                        [fv_s],
                        [fv_v],
                        s=70,
                        marker="v",
                        facecolors=fv_color,
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=5,
                        label=label,
                    )
                    used_labels.add("Fiber Volley")

        if isinstance(epsp_df, pd.DataFrame) and not epsp_df.empty and (epsp_df.stim_intensity == stim).any():
            r = epsp_df.loc[epsp_df.stim_intensity == stim].iloc[0]
            slope_s, slope_v = r.get("slope_mid_s"), r.get("slope_mid_v")
            epsp_s, epsp_v = r.get("epsp_s"), r.get("epsp_v")
            if pd.notna(slope_s) and pd.notna(slope_v):
                label = "fEPSP Slope" if "fEPSP Slope" not in used_labels else None
                ax.scatter(
                    [slope_s],
                    [slope_v],
                    s=70,
                    marker="s",
                    facecolors=epsp_slope_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label=label,
                )
                used_labels.add("fEPSP Slope")
            if pd.notna(epsp_s) and pd.notna(epsp_v):
                label = "fEPSP" if "fEPSP" not in used_labels else None
                ax.scatter(
                    [epsp_s],
                    [epsp_v],
                    s=70,
                    marker="o",
                    facecolors=epsp_min_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label=label,
                )
                used_labels.add("fEPSP")

        if isinstance(ps_df, pd.DataFrame) and not ps_df.empty and (ps_df.stim_intensity == stim).any():
            r = ps_df.loc[ps_df.stim_intensity == stim].iloc[0]
            ps_s, ps_v = r.get("ps_s"), r.get("ps_v")
            if pd.notna(ps_s) and pd.notna(ps_v):
                label = "Population Spike" if "Population Spike" not in used_labels else None
                ax.scatter(
                    [ps_s],
                    [ps_v],
                    s=70,
                    marker="d",
                    facecolors=ps_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=7,
                    label=label,
                )
                used_labels.add("Population Spike")
