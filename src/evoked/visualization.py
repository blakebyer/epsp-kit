from __future__ import annotations

from pandera.typing.polars import DataFrame
from typing import Any, Optional
import polars as pl
import numpy as np
import quantities as pq
from pydantic import BaseModel
from abc import ABC, abstractmethod
from evoked.base import ChannelTypes, StimulusTypes, Selector, RecordingResult, RecordingData
from evoked.algorithms.linear import center_signal, window_correlation
from pydantic import model_validator, Field, ConfigDict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import math
import warnings

class BasePlot(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    rc_params: dict[str, Any] | None = None

    figure: Figure | None = Field(
        default=None,
        exclude=True,
    )

    ax: Any = Field(
        default=None,
        exclude=True,
    )

    @abstractmethod
    def plot(self, *args, **kwargs) -> BasePlot:
        ...

PLOT_TYPES = {
    "TracePlot": TracePlot,
    "IOPlot": IOPlot,
    "FitPlot": FitPlot,
    "MultiChannelPlot": MultiChannelPlot,
    "DetectedPlot": DetectedPlot,
    "AllFilesPlot": AllFilesPlot,
}

class TracePlot(BasePlot):
    file_origin: str
    channel: ChannelTypes
    features: Optional[Selector] = None
    stimuli: Selector
    annotated: bool = False
    stimulus_unit: Optional[str] = ""

    @model_validator(mode="after")
    def annotated_check(self) -> TracePlot:
        if self.annotated and self.features is None:
            raise ValueError(
                "If 'annotated' is True, features cannot be empty."
            )
        return self

    def plot(
        self,
        recording: RecordingData,
        results: Optional[RecordingResult] = None,
    ) -> TracePlot:
        features = (
                [self.features]
                if isinstance(self.features, str)
                else list(self.features)
            )

        ch = (
            recording.channel_names.index(self.channel)
            if isinstance(self.channel, str)
            else self.channel
            )

        with plt.rc_context(self.rc_params):
            fig, ax = plt.subplots(layout="constrained")

            selected = recording.select_trials(
                id=self.id,
                stimulus=self.stimuli,
            )

            values = selected.values()      # (trials, samples, channels)
            time = selected.times().rescale(pq.s).magnitude
            colors = plt.get_cmap("cividis", len(selected.trials))
            feature_colors = plt.get_cmap("Accent", len(features))
            for i, trial in enumerate(selected.trials.iter_rows(named=True)):
                trace = values[i, :, ch]

                ax.plot(
                    time,
                    trace,
                    label=str(trial["stimulus"]),
                    color=colors(i)
                )

                if self.annotated:
                    for j, feature in enumerate(features):
                        data = results[feature].result

                        filters = [
                            pl.col(name) == value
                            for name, value in trial.items()
                            if name != "trial_index" and name in data.columns
                        ]

                        data = data.filter(
                            (pl.col("channel") == ch)
                            & pl.all_horizontal(filters)
                        )

                        for row in data.iter_rows(named=True):
                            
                            latency = row["latency"]
                            y = np.interp(latency, time, trace)

                            ax.scatter(
                                latency,
                                y,
                                zorder=5,
                                edgecolors="black",
                                color=feature_colors(j),
                            )

                            algorithm = results[feature].algorithm
                            half_width = (
                                algorithm.template.shape[0]
                                / recording.sampling_rate.rescale(pq.Hz).magnitude
                                / 2
                            )

                            mask = (
                                (time >= latency - half_width)
                                & (time <= latency + half_width)
                            )

                            ax.plot(
                                time[mask],
                                trace[mask],
                                linewidth=2.5,
                                zorder=2,
                                color=feature_colors(j),
                            )

            ax.set_title(f"id={self.id}, channel={ch}", fontweight="bold")
            trace_legend = ax.legend(title=f"Stimulus {self.stimulus_unit}", loc="lower right")
            ax.add_artist(trace_legend)

            if self.annotated:
                feature_handles = [
                    Line2D([0], [0], marker="o", linestyle="-", color=feature_colors(j),
                        markeredgecolor="black", label=feature)
                    for j, feature in enumerate(features)
                ]
                feature_legend = ax.legend(
                    handles=feature_handles, title="Features", loc="lower center"
                )
                ax.add_artist(feature_legend) 
            ax.set_xlabel(f"Time ({recording.time_unit})")
            ax.set_ylabel(f"Response ({recording.value_unit})")
    
            self.figure = fig
            self.ax = ax 

        return self 


class IOPlot(BasePlot):
    channel: ChannelTypes
    features: Selector
    stimuli: Selector
    group_by: Optional[Selector] = None
    stimulus_unit: Optional[str] = ""

    def plot(
        self,
        recording: RecordingData,
        results: RecordingResult,
    ) -> IOPlot:

        features = (
            [self.features]
            if isinstance(self.features, str)
            else list(self.features)
        )

        groups = (
            []
            if self.group_by is None
            else [self.group_by]
            if isinstance(self.group_by, str)
            else list(self.group_by)
        )

        stimuli = (
            [self.stimuli]
            if not isinstance(self.stimuli, (list, tuple))
            else list(self.stimuli)
        )

        with plt.rc_context(self.rc_params):
            fig, axes = plt.subplots(
                ncols=len(features),
                squeeze=False,
                layout="constrained",
            )

            axes = axes.ravel()

            for ax, feature in zip(axes, features):
                data = results[feature].result

                data = data.filter(
                    (pl.col("channel") == self.channel)
                    & pl.col("stimulus").is_in(stimuli)
                )

                if groups:
                    data = data.join(
                        recording.trials.select(
                            ["trial_index", *groups]
                        ),
                        on="trial_index",
                        how="left",
                    )

                    grouped = data.group_by(
                        groups,
                        maintain_order=True,
                    )
                else:
                    grouped = [(None, data)]

                for key, group in grouped:
                    try:
                        group = group.with_columns(
                            pl.col("stimulus").cast(pl.Float64)
                        )
                    except pl.exceptions.InvalidOperationError:
                        pass

                    stats = (
                        group
                        .group_by("stimulus")
                        .agg(
                            pl.col("amplitude")
                            .mean()
                            .alias("mean"),

                            (
                                pl.col("amplitude").std()
                                / pl.len().sqrt()
                            ).alias("sem"),
                        )
                        .sort("stimulus")
                    )

                    label = None
                    if groups:
                        key = key if isinstance(key, tuple) else (key,)
                        label = ", ".join(
                            f"{name}={value}"
                            for name, value in zip(groups, key)
                        )

                    ax.errorbar(
                        stats["stimulus"],
                        stats["mean"],
                        yerr=stats["sem"],
                        marker="o",
                        label=label,
                    )

                ax.set_title(str(feature))
                ax.set_xlabel(f"Stimulus {self.stimulus_unit}")
                ax.set_ylabel("Amplitude")

                if groups:
                    ax.legend()

            self.figure = fig
            self.ax = axes

        return self


class MultiChannelPlot(BasePlot):
    file_origin: str
    channels: Selector
    stimuli: Selector
    trial: int

    def plot(self, recording: RecordingData) -> MultiChannelPlot:
        with plt.rc_context(self.rc_params):
            selected = recording.select_trials(
                        id=self.id,
                        stimulus=self.stimuli,
                    ).select_channels(self.channels).select_trials()

            values = selected.values()      # (trials, samples, channels)
            time = selected.times().rescale(pq.s).magnitude

            for i, trial in enumerate(selected.trials.iter_rows(named=True)):
                    trace = values[trial, :, :]

            
#         intermediate = intermediate.filter(
#             (pl.col("id") == id) &
#             (pl.col("channel").is_in(channels)) &
#             (pl.col("stimulus").is_in(stimuli))
#         ).sort("channel")
#         if intermediate.is_empty():
#             raise ValueError(f"No data found for id={id}, channels={channels}")

#         found_channels = intermediate["channel"].unique().sort().to_list()
#         missing = set(channels) - set(found_channels)
#         if missing:
#             warnings.warn(f"No data for channel(s)={sorted(missing)} at id={id}. Skipping...")

#         fig = plt.figure(figsize=(8, 1 * len(found_channels)), layout="constrained")
#         gs = fig.add_gridspec(len(found_channels), hspace=0)
#         axes = gs.subplots(sharex=True, sharey=True)
#         axes = [axes] if len(found_channels) == 1 else list(axes)

#         cmap = mpl.colormaps["cividis"]

#         for i, ((channel,), cdata) in enumerate(intermediate.group_by("channel", maintain_order=True)):
#             ax = axes[i]
#             time = col_to_2d(cdata, "time")
#             value = col_to_2d(cdata, "value")
#             stimulus_col = cdata["stimulus"].to_list()
#             for j in range(cdata.height):
#                 color_val = cmap(j / max(1, len(stimuli) - 1)) if len(stimuli) > 1 else 'black'
#                 stimulus = stimulus_col[j]
#                 ax.plot(time[j], value[j], color=color_val, label=f"{stimulus}")

#             ax.set_ylabel(f"ch {channel}", rotation=0)
#             ax.tick_params(labelbottom=False)
#             # for spine in ax.spines.values():
#             #     spine.set_visible(False)

#             # ax.set_xticks([])
#             # ax.set_yticks([])

#         axes[-1].tick_params(labelbottom=True)
#         axes[-1].set_xlabel(f"Time ({intermediate.config_meta.get_metadata().get("time_unit")})")
#         axes[-1].legend(title=f"Stimulus ({intermediate.config_meta.get_metadata().get("stimulus_unit")})", loc="lower right")
#         fig.suptitle(f"id={id}", fontweight="bold")
#         fig.tight_layout()

#         return fig, axes


class FitPlot(BasePlot):
    file_origin: str
    channel: ChannelTypes
    features: Selector
    stimulus: StimulusTypes
    stimulus_unit: Optional[str] = ""

    def plot(
        self,
        recording: RecordingData,
        results: RecordingResult,
    ) -> FitPlot:

        features = (
            [self.features]
            if isinstance(self.features, str)
            else list(self.features)
        )

        # Results store channel as integer index
        ch = (
            recording.channel_names.index(self.channel)
            if isinstance(self.channel, str)
            else self.channel
        )

        selected = recording.select_trials(
            file_origin=self.file_origin,
            stimulus=self.stimulus,
        ).select_channels([self.channel])

        if selected.shape[0] != 1:
            raise ValueError(
                "FitPlot requires file_origin + stimulus to identify "
                "exactly one trial. Average the recording first."
            )

        with plt.rc_context(self.rc_params):
            fig, axes = plt.subplots(
                nrows=len(features),
                ncols=2,
                squeeze=False,
                layout="constrained",
            )

            colors = plt.get_cmap("Accent", len(features))

            for i, feature in enumerate(features):
                ax_fit, ax_corr = axes[i]

                result = results[feature]
                algorithm = result.algorithm

                data = result.result.filter(
                    (pl.col("file_origin") == self.file_origin)
                    & (pl.col("channel") == ch)
                    & (pl.col("stimulus") == self.stimulus)
                )

                if data.height != 1:
                    raise ValueError(
                        f"Expected one result for {feature}, found {data.height}."
                    )

                row = data.row(0, named=True)

                # template is (n_samples, n_channels)
                template = algorithm.template[:, ch]

                # Reconstruct the exact search window used by MatchedFilter
                search = (
                    algorithm.search_window
                    if isinstance(algorithm.search_window, tuple)
                    else (
                        algorithm.window[0]
                        - (algorithm.window[1] - algorithm.window[0])
                        * algorithm.search_window,
                        algorithm.window[1]
                        + (algorithm.window[1] - algorithm.window[0])
                        * algorithm.search_window,
                    )
                )

                signal = selected.values(search)[0, :, 0]

                if algorithm.slope_transform:
                    signal = np.gradient(signal)

                time = (
                    selected.times(search)
                    .rescale(pq.s)
                    .magnitude
                )

                # Recompute matching diagnostics
                corr, dot = window_correlation(signal, template)
                best_k = int(np.nanargmax(corr))
                best_corr = float(corr[best_k])

                template_c = center_signal(template)
                template_ss = np.dot(template_c, template_c)
                scale = dot[best_k] / template_ss

                L = len(template)
                left = L // 2

                matched = signal[best_k:best_k + L]
                fitted = scale * template_c + matched.mean()

                fit_time = time[best_k:best_k + L]
                corr_time = time[left:left + len(corr)]

                latency = float(row["latency"])
                r2 = float(row["r2"])

                # Signal + fitted template
                ax_fit.plot(
                    time,
                    signal,
                    color="black",
                    linewidth=1.6,
                    label="Signal",
                )

                ax_fit.plot(
                    fit_time,
                    fitted,
                    color=colors(i),
                    linewidth=2.2,
                    label=f"Template fit ($R^2$={r2:.2f})",
                )

                ax_fit.axvline(
                    latency,
                    color=colors(i),
                    linestyle="--",
                    linewidth=1.0,
                )

                # Correlation trace
                ax_corr.plot(
                    corr_time,
                    corr,
                    color=colors(i),
                    linewidth=1.8,
                )

                ax_corr.scatter(
                    latency,
                    best_corr,
                    facecolor=colors(i),
                    edgecolor="black",
                    linewidth=0.8,
                    s=40,
                    zorder=3,
                )

                value_unit = (
                    f"{recording.value_unit}/{recording.time_unit}"
                    if algorithm.slope_transform
                    else f"{recording.value_unit}"
                )

                ax_fit.set_title(str(feature))
                ax_fit.set_xlabel("Time (s)")
                ax_fit.set_ylabel(value_unit)
                ax_fit.legend()

                ax_corr.set_xlabel("Time (s)")
                ax_corr.set_ylabel("Correlation")

            fig.suptitle(
                f"{self.file_origin}, channel={self.channel}, "
                f"stimulus={self.stimulus} {self.stimulus_unit}"
            )

        self.figure = fig
        self.ax = axes

        return self

                # ax_fit.plot(
#                 time[search_start:search_stop],
#                 signal[search_start:search_stop],
#                 color="black",
#                 linewidth=1.6,
#                 label=f"{feature} snippet"
#             )
#             ax_fit.plot(
#                 fit_time,
#                 fitted,
#                 color=color,
#                 linewidth=2.2,
#                 label=f"{feature} template fit ($R^2$={best_r2:.2f})",
#             )
#             ax_fit.axvline(best_time, color=color, linestyle="--", linewidth=1.0)




                

class DetectedPlot(BasePlot):
    features: Selector
    channel: ChannelTypes
    stimulus_unit: Optional[str] = ""

    def plot(self, results: RecordingResult) -> DetectedPlot:
        features = (
                    [self.features]
                    if isinstance(self.features, str)
                    else self.features
                )

        with plt.rc_context(self.rc_params):
            fig, ax = plt.subplots()

            colors = plt.get_cmap("Accent", len(features))
            
            for i, feature in enumerate(features):
                data = results[feature].result
                
                detection = data.filter(pl.col("channel")==self.channel)
                stats = (
                    detection
                    .group_by("stimulus")
                    .agg(
                        (pl.col("detected").mean() * 100).alias("percent_detected")
                    )
                ).sort("stimulus")
                
                try:
                    stats = stats.with_columns(pl.col("stimulus").cast(pl.Float32))
                except pl.exceptions.InvalidOperationError:
                    pass

                stats = stats.sort("stimulus")
                
                ax.plot(
                    stats["stimulus"].to_numpy(), 
                    stats["percent_detected"].to_numpy(),
                    marker="o", 
                    label=feature,
                    color=colors(i),
                )
                
            ax.set_xlabel(f"Stimulus ({self.stimulus_unit})")
            ax.set_ylabel("Detected (%)")
            ax.set_ylim(-5, 105)
            ax.legend(title="Features")
            ax.set_title("Feature Detection")

            self.figure = fig
            self.ax = ax 

        return self

class AllFilesPlot(BasePlot):
    stimuli: Selector
    output_path: Optional[str] = None
    max_per_page: Optional[int] = None
    rc_params: Optional[dict[str, Any]] = None


# PlotConfig = (
#     TracePlot
#     | MultiChannelPlot
#     | IOPlot
#     | FitPlot
#     | DetectedPlot
#     | AllFilesPlot
# )

# PlotType = Literal["io", "trace", "multichannel", "fit", "detected", "allfiles"]

# class Plotting(BaseModel):
#     plots: Optional[dict[PlotType, PlotConfig | list[PlotConfig]]] = None          
        

# def plot_io_curve(results: RecordingResult, features: list[str], stimuli: list[str], channel: int = 0, rc_params: Optional[dict] = None):
#     with plt.rc_context(rc_params):
#         fig, axes = plt.subplots(ncols=len(features), layout="constrained")
        
#         if len(features) == 1: 
#             axes = [axes]

#         cmap = mpl.colormaps["Accent"]
        
#         for i, feature in enumerate(features):
#             r_result = results.get(feature)
            
#             rdata = r_result.result
#             slope_transform = r_result.slope_transform
#             rdata = rdata.filter((pl.col("stimulus").is_in(stimuli)) & (pl.col("channel")==channel))

#             stats = rdata.group_by("stimulus").agg(
#                 pl.col("amplitude").mean().alias("mean"),
#                 (pl.col("amplitude").std() / pl.col("amplitude").count().sqrt()).alias("sem"),
#             ).sort("stimulus")
#             color_val = cmap(i / max(1, len(features) - 1))

#             try:
#                 stats = stats.with_columns(pl.col("stimulus").cast(pl.Float32))
#             except pl.exceptions.InvalidOperationError:
#                 pass

#             stats = stats.sort("stimulus")

#             axes[i].errorbar(
#                 stats['stimulus'].to_numpy(),
#                 stats['mean'].to_numpy(),
#                 yerr=stats['sem'].to_numpy(),
#                 fmt='-o',
#                 color=color_val,
#                 capsize=3
#             )
#             if slope_transform: axes[i].set_ylabel(f'Slope ({rdata.config_meta.get_metadata().get("value_unit")}/{rdata.config_meta.get_metadata().get("time_unit")})') 
#             else: axes[i].set_ylabel(f'Amplitude ({rdata.config_meta.get_metadata().get("value_unit")})')
#             axes[i].set_xlabel(f'Stimulus ({rdata.config_meta.get_metadata().get("stimulus_unit")})')
#             axes[i].set_title(feature)
#             axes[i].grid(alpha=0.3)
        
#         fig.suptitle('IO Curves', fontweight="bold")
#         fig.tight_layout()

#         return fig, axes

# def plot_trace(intermediate: DataFrame[IntermediateResult], stimuli: list[str], file_origin: str, channel: int = 0, results: Optional[RecordingResult] = None, features: Optional[list[str]] = None, annotated: bool = False, rc_params: Optional[dict] = None):
#     with plt.rc_context(rc_params):
#         intermediate = intermediate.filter(
#             (pl.col("id")==id) &
#             (pl.col("channel")==channel) &
#             (pl.col("stimulus").is_in(stimuli))
#             )
#         if intermediate.is_empty():
#             raise ValueError(f"No data found for id={id}, channel={channel}, stimuli={stimuli}")

#         fig, ax = plt.subplots(layout="constrained")
#         cmap = mpl.colormaps["cividis"]

#         time = col_to_2d(intermediate, "time")
#         value = col_to_2d(intermediate, "value")
#         stimulus_col = intermediate["stimulus"].to_list()  

#         for i in range(intermediate.height):
#             color_val = cmap(i / max(1, len(stimuli) - 1)) if len(stimuli) > 1 else 'black'
#             t = time[i] - time[i, 0]    
#             v = value[i]               
#             stimulus = stimulus_col[i]
#             ax.plot(t, v, color=color_val, label=f"{stimulus}")

#             if annotated:
#                 feature_cmap = mpl.colormaps["Accent"]

#                 for j, feature in enumerate(features):
#                     r_result = results.get(feature)
#                     if r_result is None:
#                         continue

#                     rdata = r_result.result.filter(
#                         (pl.col("id") == id) &
#                         (pl.col("stimulus") == stimulus) &
#                         (pl.col("channel") == channel)
#                     )
#                     if rdata.is_empty():
#                         warnings.warn(f"No detection for {feature} at channel={channel}, id={id}, and stimulus={stimulus}")
#                         continue

#                     feature_color = feature_cmap(j / max(1, len(features) - 1))
#                     half_width = (len(r_result.template) // 2) / float(intermediate.config_meta.get_metadata().get("fs"))

#                     for mt in rdata["feature_time"].to_numpy():
#                         mask = (t >= mt - half_width) & (t <= mt + half_width)
#                         y = np.interp(mt, t, v)
#                         ax.plot(t[mask], v[mask], color=feature_color, linewidth=2.5, zorder=5)
#                         ax.scatter(mt, y, color=feature_color, edgecolors="black", zorder=6)

#         trace_legend = ax.legend(
#             title=f"Stimulus ({intermediate.config_meta.get_metadata().get("stimulus_unit")})",
#             loc="lower right"
#         )
#         ax.add_artist(trace_legend)

#         if annotated:
#             annotation_handles = [
#                 Line2D(
#                     [0], [0],
#                     color=mpl.colormaps["Accent"](i / max(1, len(features) - 1)),
#                     linewidth=3,
#                     marker="o",
#                     markeredgecolor="black",
#                     label=feature,
#                 )
#                 for i, feature in enumerate(features)
#             ]
#             ax.legend(handles=annotation_handles, title="Features", loc="lower center")

#         fig.suptitle(f"id={id}, channel={channel}", fontweight="bold")
#         ax.set_xlabel(f"Time ({intermediate.config_meta.get_metadata().get("time_unit")})")
#         ax.set_ylabel(f"Response ({intermediate.config_meta.get_metadata().get("value_unit")})")
#         ax.grid(alpha=0.3)
#         fig.tight_layout()

#         return fig, ax


# def plot_multichannel(
#     intermediate: DataFrame[IntermediateResult],
#     stimuli: list[str],
#     file_origin: str,
#     channels: list[int],
#     rc_params: Optional[dict] = None,
# ):
#     with plt.rc_context(rc_params):
#         intermediate = intermediate.filter(
#             (pl.col("id") == id) &
#             (pl.col("channel").is_in(channels)) &
#             (pl.col("stimulus").is_in(stimuli))
#         ).sort("channel")
#         if intermediate.is_empty():
#             raise ValueError(f"No data found for id={id}, channels={channels}")

#         found_channels = intermediate["channel"].unique().sort().to_list()
#         missing = set(channels) - set(found_channels)
#         if missing:
#             warnings.warn(f"No data for channel(s)={sorted(missing)} at id={id}. Skipping...")

#         fig = plt.figure(figsize=(8, 1 * len(found_channels)), layout="constrained")
#         gs = fig.add_gridspec(len(found_channels), hspace=0)
#         axes = gs.subplots(sharex=True, sharey=True)
#         axes = [axes] if len(found_channels) == 1 else list(axes)

#         cmap = mpl.colormaps["cividis"]

#         for i, ((channel,), cdata) in enumerate(intermediate.group_by("channel", maintain_order=True)):
#             ax = axes[i]
#             time = col_to_2d(cdata, "time")
#             value = col_to_2d(cdata, "value")
#             stimulus_col = cdata["stimulus"].to_list()
#             for j in range(cdata.height):
#                 color_val = cmap(j / max(1, len(stimuli) - 1)) if len(stimuli) > 1 else 'black'
#                 stimulus = stimulus_col[j]
#                 ax.plot(time[j], value[j], color=color_val, label=f"{stimulus}")

#             ax.set_ylabel(f"ch {channel}", rotation=0)
#             ax.tick_params(labelbottom=False)
#             # for spine in ax.spines.values():
#             #     spine.set_visible(False)

#             # ax.set_xticks([])
#             # ax.set_yticks([])

#         axes[-1].tick_params(labelbottom=True)
#         axes[-1].set_xlabel(f"Time ({intermediate.config_meta.get_metadata().get("time_unit")})")
#         axes[-1].legend(title=f"Stimulus ({intermediate.config_meta.get_metadata().get("stimulus_unit")})", loc="lower right")
#         fig.suptitle(f"id={id}", fontweight="bold")
#         fig.tight_layout()

#         return fig, axes


# def plot_fit(
#     intermediate: DataFrame[IntermediateResult],
#     results: RecordingResult,
#     features: list[str],
#     stimulus: str,
#     file_origin: str,
#     channel: int = 0,
#     rc_params: Optional[dict] = None,
# ):
#     """Plot the best template fit and correlation trace for any number of features.

#     Diagnostic arrays are recomputed from ``intermediate`` and each stored
#     ``FeatureResult`` rather than being retained in ``FitResult``.
#     """
#     if not features:
#         raise ValueError("features must contain at least one feature name.")

#     idata = intermediate.filter(
#         (pl.col("id") == id)
#         & (pl.col("channel") == channel)
#         & (pl.col("stimulus") == stimulus)
#     )
#     if idata.is_empty():
#         raise ValueError(
#             f"No data found for id={id}, channel={channel}, "
#             f"stimulus={stimulus}."
#         )

#     metadata = intermediate.config_meta.get_metadata()
#     fs = metadata.get("fs")
#     time_unit = metadata.get("time_unit")
#     value_unit = metadata.get("value_unit")

#     row = idata.row(0, named=True)
#     time = np.asarray(row["time"])
#     time = time - time[0]
#     raw_signal = np.asarray(row["value"])
#     cmap = mpl.colormaps["Accent"]

#     with plt.rc_context(rc_params):
#         fig, axes = plt.subplots(
#             nrows=len(features),
#             ncols=2,
#             squeeze=False,
#             figsize=(10.5, 3.0 * len(features)),
#             gridspec_kw={"width_ratios": [1.35, 1.15]},
#         )

#         for i, feature in enumerate(features):
#             if feature not in results.results:
#                 raise KeyError(f"Feature {feature} not found in results.results.")

#             feature_result = results.get(feature)
#             color = cmap(i / max(1, len(features) - 1))
#             signal = raw_signal.copy()
#             if feature_result.slope_transform:
#                 signal = np.gradient(signal, time)

#             template = np.asarray(feature_result.template, dtype=float).ravel()
#             L = template.size
#             if L < 3:
#                 raise ValueError(f"Template for {feature} must contain at least 3 samples.")

#             if isinstance(feature_result.search_window, tuple):
#                 search_window_s = feature_result.search_window
#             else:
#                 width = feature_result.window[1] - feature_result.window[0]
#                 pad = width * feature_result.search_window
#                 search_window_s = (
#                     feature_result.window[0] - pad,
#                     feature_result.window[1] + pad,
#                 )

#             search_start, search_stop = window_to_indices(time, search_window_s, fs)
#             search_start = max(0, search_start)
#             search_stop = min(signal.size, search_stop)

#             corr, dot = window_correlation(
#                 signal=signal,
#                 template=template,
#                 search_start=search_start,
#                 search_stop=search_stop,
#             )
#             if not np.any(np.isfinite(corr)):
#                 warnings.warn(
#                     f"Correlation is undefined throughout the search window for "
#                     f"feature {feature}."
#                 )
#                 continue

#             best_k_local = int(np.nanargmax(corr))
#             best_k = search_start + best_k_local
#             best_corr = float(corr[best_k_local])

#             template_c = center_signal(template)
#             template_ss = float(np.dot(template_c, template_c))
#             best_scale = float(dot[best_k_local] / template_ss)

#             snippet = signal[best_k:best_k + L]
#             fitted = best_scale * template_c + np.mean(snippet)
#             fit_time = time[best_k:best_k + L]

#             left = L // 2
#             corr_time = time[search_start + left:search_start + left + corr.size]
#             best_center = best_k + left
#             best_time = time[best_center]
#             best_r2 = best_corr ** 2

#             ax_fit, ax_corr = axes[i]
#             ax_fit.plot(
#                 time[search_start:search_stop],
#                 signal[search_start:search_stop],
#                 color="black",
#                 linewidth=1.6,
#                 label=f"{feature} snippet"
#             )
#             ax_fit.plot(
#                 fit_time,
#                 fitted,
#                 color=color,
#                 linewidth=2.2,
#                 label=f"{feature} template fit ($R^2$={best_r2:.2f})",
#             )
#             ax_fit.axvline(best_time, color=color, linestyle="--", linewidth=1.0)
#             y_unit = (
#                 f"{value_unit}/{time_unit}"
#                 if feature_result.slope_transform
#                 else str(value_unit)
#             )
#             ax_fit.set_title(feature)
#             ax_fit.set_xlabel(f"Time ({time_unit})")
#             ax_fit.set_ylabel(y_unit)
#             ax_fit.legend(loc="best")
#             ax_fit.grid(alpha=0.3)

#             ax_corr.plot(corr_time, corr, color=color, linewidth=1.8)
#             ax_corr.axvline(best_time, color=color, linestyle="--", linewidth=1.0)
#             ax_corr.scatter(
#                 [best_time],
#                 [best_corr],
#                 facecolor=color,
#                 edgecolor="black",
#                 linewidth=0.8,
#                 s=40,
#                 zorder=3,
#             )
#             ax_corr.set_xlabel(f"Time ({time_unit})")
#             ax_corr.set_ylabel("Correlation")
#             ax_corr.grid(alpha=0.3)

#         fig.suptitle(
#             f"id={id}, channel={channel}, stimulus={stimulus} {metadata.get("stimulus_unit")}",
#             fontsize=14,
#             fontweight="bold",
#         )
#         fig.tight_layout()
#         return fig, axes

# def plot_detected(results: RecordingResult, features: list[str], channel: int = 0, rc_params: Optional[dict] = None):
#     with plt.rc_context(rc_params):
#         fig, ax = plt.subplots(figsize=(8,6))

#         cmap = mpl.colormaps["Accent"]
        
#         for i, feature in enumerate(features):
#             r_result = results.results.get(feature)
#             if r_result is None:
#                 continue

#             color_val = cmap(i / max(1, len(features) - 1))
            
#             detection = r_result.result.filter(pl.col("channel")==channel)
#             stats = (
#                 detection
#                 .group_by("stimulus")
#                 .agg(
#                     (pl.col("detected").mean() * 100).alias("percent_detected")
#                 )
#             ).sort("stimulus")

            
#             try:
#                 stats = stats.with_columns(pl.col("stimulus").cast(pl.Float32))
#             except pl.exceptions.InvalidOperationError:
#                 pass

#             stats = stats.sort("stimulus")
            
#             ax.plot(
#                 stats["stimulus"].to_numpy(), 
#                 stats["percent_detected"].to_numpy(),
#                 marker="o", 
#                 label=feature,
#                 color=color_val
#             )
#             ax.grid(alpha=0.3)
            
#         # format shared axis
#         ax.set_xlabel(f"Stimulus ({detection.config_meta.get_metadata().get("stimulus_unit")})")
#         ax.set_ylabel("Detected (%)")
#         ax.set_ylim(-5, 105)
#         ax.legend(title="Features")
#         fig.suptitle("Feature Detection", fontweight="bold")
#         fig.tight_layout()

#         return fig, ax


# def plot_all_files(
#     intermediate,
#     stimuli: list[str],
#     channel: int = 0,
#     output_path: str = "all_files.pdf",
#     max_per_page: int = 6,
#     rc_params: Optional[dict] = None,
# ):
#     """Plot all files and save each figure page to a multipage PDF."""

#     with plt.rc_context(rc_params), PdfPages(output_path) as pdf:
#         unique_ids = intermediate["id"].unique()
#         total_slices = len(unique_ids)
#         num_pages = math.ceil(total_slices / max_per_page)

#         nrows = 2
#         ncols = math.ceil(max_per_page / nrows)

#         for page in range(num_pages):
#             start_idx = page * max_per_page
#             end_idx = min(start_idx + max_per_page, total_slices)
#             page_ids = unique_ids[start_idx:end_idx]

#             master_fig, master_axes = plt.subplots(
#                 nrows=nrows,
#                 ncols=ncols,
#                 figsize=(12, 10),
#                 squeeze=False,
#             )
#             axes_flat = master_axes.flatten()

#             for idx, id in enumerate(page_ids):
#                 temp_fig, temp_ax = plot_trace(
#                     intermediate=intermediate,
#                     stimuli=stimuli,
#                     id=id,
#                     channel=channel,
#                     annotated=False,
#                 )

#                 target_ax = axes_flat[idx]

#                 for line in temp_ax.get_lines():
#                     target_ax.plot(
#                         line.get_xdata(),
#                         line.get_ydata(),
#                         color=line.get_color(),
#                         label=line.get_label(),
#                     )

#                 target_ax.set_title(str(id), fontsize=10, fontweight="bold")
#                 target_ax.legend(title=f"Stimulus ({intermediate.config_meta.get_metadata().get("stimulus_unit")})")
#                 target_ax.set_xlabel(
#                     f'Time ({intermediate.config_meta.get_metadata().get("time_unit")})'
#                 )
#                 target_ax.set_ylabel(
#                     f'Response ({intermediate.config_meta.get_metadata().get("value_unit")})'
#                 )
#                 target_ax.grid(alpha=0.3)

#                 plt.close(temp_fig)

#             for idx in range(len(page_ids), len(axes_flat)):
#                 axes_flat[idx].set_visible(False)

#             master_fig.suptitle(
#                 f"Evoked Field Potentials - Page {page + 1}",
#                 fontsize=14,
#                 fontweight="bold",
#             )
#             master_fig.tight_layout()

#             pdf.savefig(master_fig, bbox_inches="tight")
#             plt.close(master_fig)