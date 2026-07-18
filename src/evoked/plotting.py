from __future__ import annotations

from pandera.typing.polars import DataFrame
import polars as pl
import numpy as np
from evoked.base import RecordingResult, IntermediateResult, window_to_indices
from evoked.matched_filter import center_signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import math
import warnings

def plot_io_curve(recording_result: RecordingResult, features: list[str], stimuli: list[str], channel: int = 0, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(ncols=len(features))
        
        if len(features) == 1: 
            axes = [axes]

        cmap = mpl.colormaps["Paired"]
        
        for i, feature in enumerate(features):
            r_result = recording_result.results[feature]
            if r_result is None:
                continue
            
            rdata = r_result.result
            slope_transform = r_result.slope_transform
            rdata = rdata.filter((pl.col("stimulus").is_in(stimuli)) & (pl.col("channel")==channel))

            stats = rdata.group_by("stimulus").agg(
                pl.col("amplitude").mean().alias("mean"),
                (pl.col("amplitude").std() / pl.col("amplitude").count().sqrt()).alias("sem"),
            ).sort("stimulus")
            color_val = cmap(i / max(1, len(features) - 1))

            try:
                stats = stats.with_columns(pl.col("stimulus").cast(pl.Float64))
            except pl.exceptions.InvalidOperationError:
                pass

            stats = stats.sort("stimulus")

            axes[i].errorbar(
                stats['stimulus'].to_numpy(),
                stats['mean'].to_numpy(),
                yerr=stats['sem'].to_numpy(),
                fmt='-o',
                color=color_val,
                capsize=3
            )
            if slope_transform: axes[i].set_ylabel(f'Slope ({rdata.config_meta.get_metadata().get("value_unit")}/{rdata.config_meta.get_metadata().get("time_unit")})') 
            else: axes[i].set_ylabel(f'Amplitude ({rdata.config_meta.get_metadata().get("value_unit")})')
            axes[i].set_xlabel(f'Stimulus ({rdata.config_meta.get_metadata().get("stimulus_unit")})')
            axes[i].set_title(feature)
            axes[i].grid(alpha=0.3)
        
        fig.suptitle('IO Curves')
        plt.tight_layout()

        return fig, axes

def plot_trace(intermediate: DataFrame[IntermediateResult], stimuli: list[str], id_value: str, channel: int = 0, recording_result: RecordingResult | None = None, features: list[str] | None = None, annotated: bool = False, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        intermediate = intermediate.filter(
            (pl.col("id")==id_value) &
            (pl.col("channel")==channel)
            )
        if intermediate.is_empty():
            raise ValueError(f"No data found for id={id_value}, channel={channel}")
        fig, ax = plt.subplots(figsize=(8,6))

        cmap = mpl.colormaps["cividis"]

        for i, stimulus in enumerate(stimuli):
            color_val = cmap(i / max(1, len(stimuli) - 1)) if len(stimuli) > 1 else 'black'
            idata = intermediate.filter(pl.col("stimulus")==stimulus)
            if idata.is_empty():
                raise ValueError(f"No trace data found for stimulus={stimulus} at channel={channel} and id={id_value}")
            time = idata['time'] - idata['time'][0]
            time = time.to_numpy()
            value = idata['value'].to_numpy()
            ax.plot(time, value, color=color_val, label=f"{stimulus}")
            if annotated:
                feature_cmap = mpl.colormaps["Paired"]

                for j, feature in enumerate(features):
                    r_result = recording_result.results[feature]
                    if r_result is None:
                        continue

                    rdata = r_result.result
                    rdata = rdata.filter(
                        (pl.col("id") == id_value) &
                        (pl.col("stimulus") == stimulus) &
                        (pl.col("channel")==channel)
                    )

                    if rdata.is_empty():
                        warnings.warn(f"No detection for {feature} at channel={channel}, id={id_value}, and stimulus={stimulus}")
                        continue

                    feature_color = feature_cmap(j / max(1, len(features) - 1))
                    half_width = (len(r_result.template) // 2) / float(intermediate.config_meta.get_metadata().get("fs"))

                    for mt in rdata["feature_time"].to_numpy():
                        mask = (time >= mt - half_width) & (time <= mt + half_width)
                        y = np.interp(mt, time, value)

                        ax.plot(time[mask], value[mask], color=feature_color, linewidth=2.5, zorder=5)
                        ax.scatter(mt, y, color=feature_color, edgecolors="black", zorder=6)
        
        trace_legend = ax.legend(
            title=f"Stimulus ({intermediate.config_meta.get_metadata().get("stimulus_unit")})",
            loc="lower right"
        )
        ax.add_artist(trace_legend)

        if annotated:
            annotation_handles = [
                Line2D(
                    [0],
                    [0],
                    color=mpl.colormaps["Paired"](i / max(1, len(features) - 1)),
                    linewidth=3,
                    marker="o",
                    markeredgecolor="black",
                    label=feature,
                )
                for i, feature in enumerate(features)
            ]

            ax.legend(
                handles=annotation_handles,
                title="Features",
                loc="lower center"
            )

        fig.suptitle("Evoked Field Potential")
        ax.set_xlabel(f"Time ({intermediate.config_meta.get_metadata().get("time_unit")})")
        ax.set_ylabel(f"Response ({intermediate.config_meta.get_metadata().get("value_unit")})")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig, ax

def plot_fit(
    intermediate: DataFrame[IntermediateResult],
    recording_result: RecordingResult,
    features: list[str],
    stimulus: int,
    id_value: str,
    channel: int = 0,
    rc_params: dict | None = None,
):
    with plt.rc_context(rc_params):
        idata = intermediate.filter(
                (pl.col("id") == id_value) &
                (pl.col("stimulus") == stimulus) &
                (pl.col("channel")==channel)
        )

        if idata.is_empty():
            return None, None

        fig, axes = plt.subplots(
            nrows=len(features),
            ncols=2,
            squeeze=False,
            figsize=(10.5, 3.0 * len(features)),
            gridspec_kw={"width_ratios": [1.35, 1.15]},
        )

        cmap = mpl.colormaps["Paired"]
        
        fs = intermediate.config_meta.get_metadata().get("fs")
        time = idata["time"].to_numpy()
        value = idata["value"].to_numpy()

        for i, feature in enumerate(features):
            r_result = recording_result.results[feature]
            if r_result is None:
                continue

            color_val = cmap(i / max(1, len(features) - 1))

            search_window = r_result.search_window
            slope_transform = r_result.slope_transform

            rdata = r_result.result
            row = rdata.filter(
                        (pl.col("id") == id_value) &
                        (pl.col("stimulus") == stimulus)
            )

            if row.is_empty():
                continue

            row = row.row(0, named=True)

            corr_arr = np.asarray(row["corr_arr"], dtype=float)
            corr = float(row["corr"])
            scale = float(row["scale"])
            r2 = float(row["r2"])
            feature_time_ms = float(row["feature_time"])
            feature_time_s = feature_time_ms / 1000.0

            template = getattr(r_result, "template", None)

            if template is None:
                raise ValueError(
                    f"{feature} has no stored template. "
                    "Store template_arr inside FeatureResult when fitting."
                )

            template = np.asarray(template, dtype=float).ravel()

            signal = value.copy()
            if slope_transform:
                signal = np.gradient(signal, time)

            center_idx = template.size // 2
            left = center_idx
            right = template.size - center_idx - 1

            center_sample = int(np.argmin(np.abs(time - feature_time_s)))
            fit_start = center_sample - left
            fit_stop = center_sample + right + 1

            if fit_start < 0 or fit_stop > signal.size:
                continue

            snippet = signal[fit_start:fit_stop]

            if snippet.size != template.size:
                continue

            rel_time_ms = (time[fit_start:fit_stop] - time[center_sample]) * 1000.0

            template_centered = center_signal(template)
            fitted = scale * template_centered + np.mean(snippet)

            axes[i, 0].axvline(0.0, color="0.75", linestyle="--", linewidth=1.0)

            axes[i, 0].plot(
                rel_time_ms,
                snippet,
                color="black",
                linewidth=1.6,
                label=f"{feature} snippet",
            )

            axes[i, 0].plot(
                rel_time_ms,
                fitted,
                color=color_val,
                linewidth=2.2,
                label=f"{feature} template fit",
            )

            axes[i, 0].text(
                0.02,
                0.98,
                rf"$R^2$={r2:.2f}",
                transform=axes[i, 0].transAxes,
                ha="left",
                va="top",
                color="black",
            )

            y_label = "mV/ms" if slope_transform else "mV"
            axes[i, 0].set_ylabel(y_label)
            axes[i, 0].set_xlabel("Relative time (ms)")
            axes[i, 0].legend(loc="best")
            axes[i, 0].grid(alpha=0.3)

            s_start, s_stop = window_to_indices(time, search_window, fs)

            first_center = s_start + left
            last_center = s_stop - right

            corr_time_ms = time[first_center:last_center] * 1000.0

            # Defensive length match
            n = min(len(corr_time_ms), len(corr_arr))
            corr_time_ms = corr_time_ms[:n]
            corr_arr_plot = corr_arr[:n]

            axes[i, 1].axhline(0.0, color="0.8", linewidth=1.0)

            axes[i, 1].plot(
                corr_time_ms,
                corr_arr_plot,
                color=color_val,
                linewidth=1.8,
            )

            axes[i, 1].axvline(
                feature_time_ms,
                color=color_val,
                linestyle="--",
                linewidth=1.2,
            )

            axes[i, 1].scatter(
                [feature_time_ms],
                [corr],
                facecolor=color_val,
                edgecolor="black",
                linewidth=0.8,
                s=40,
                zorder=5,
            )

            axes[i, 1].set_ylabel("Corr.")
            axes[i, 1].set_xlabel("Time (ms)")
            axes[i, 1].grid(alpha=0.3)

        fig.suptitle(f"{stimulus} µA", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        return fig, axes

def plot_detected(recording_result: RecordingResult, features: list[str], channel: int = 0, rc_params: dict | None = None):
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(8,6))

        cmap = mpl.colormaps["Paired"]
        
        for i, feature in enumerate(features):
            r_result = recording_result.results.get(feature)
            if r_result is None:
                continue

            color_val = cmap(i / max(1, len(features) - 1))
            
            detection = r_result.result.filter(pl.col("channel")==channel)
            stats = (
                detection
                .group_by("stimulus")
                .agg(
                    (pl.col("detected").mean() * 100).alias("percent_detected")
                )
            ).sort("stimulus")

            
            try:
                stats = stats.with_columns(pl.col("stimulus").cast(pl.Float64))
            except pl.exceptions.InvalidOperationError:
                pass

            stats = stats.sort("stimulus")
            
            ax.plot(
                stats["stimulus"].to_numpy(), 
                stats["percent_detected"].to_numpy(),
                marker="o", 
                label=feature,
                color=color_val
            )
            ax.grid(alpha=0.3)
            
        # format shared axis
        ax.set_xlabel(f"Stimulus ({detection.config_meta.get_metadata().get("stimulus_unit")})")
        ax.set_ylabel("Detected (%)")
        ax.set_ylim(-5, 105)
        ax.legend(title="Features")
        fig.suptitle("Feature Detection IO Curves")
        plt.tight_layout()

        return fig, ax


def plot_all_files(
    intermediate,
    stimuli: list[str],
    output_path: str = "all_files.pdf",
    max_per_page: int = 6,
    rc_params: dict | None = None,
):
    """Plot all files and save each figure page to a multipage PDF."""

    with plt.rc_context(rc_params), PdfPages(output_path) as pdf:
        unique_ids = intermediate["id"].unique()
        total_slices = len(unique_ids)
        num_pages = math.ceil(total_slices / max_per_page)

        nrows = 2
        ncols = math.ceil(max_per_page / nrows)

        for page in range(num_pages):
            start_idx = page * max_per_page
            end_idx = min(start_idx + max_per_page, total_slices)
            page_ids = unique_ids[start_idx:end_idx]

            master_fig, master_axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(12, 10),
                squeeze=False,
            )
            axes_flat = master_axes.flatten()

            for idx, id_value in enumerate(page_ids):
                temp_fig, temp_ax = plot_trace(
                    intermediate=intermediate,
                    stimuli=stimuli,
                    id_value=id_value,
                    annotated=False,
                )

                target_ax = axes_flat[idx]

                for line in temp_ax.get_lines():
                    target_ax.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        color=line.get_color(),
                        label=line.get_label(),
                    )

                target_ax.set_title(str(id_value), fontsize=10, fontweight="bold")
                target_ax.legend(title="Stimulus (µA)")
                target_ax.set_xlabel(
                    f'Time ({intermediate.config_meta.get_metadata().get("time_unit")})'
                )
                target_ax.set_ylabel(
                    f'Response ({intermediate.config_meta.get_metadata().get("value_unit")})'
                )
                target_ax.grid(alpha=0.3)

                plt.close(temp_fig)

            for idx in range(len(page_ids), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            master_fig.suptitle(
                f"Evoked Field Potentials - Page {page + 1}",
                fontsize=14,
                fontweight="bold",
            )
            master_fig.tight_layout()

            pdf.savefig(master_fig, bbox_inches="tight")
            plt.close(master_fig)