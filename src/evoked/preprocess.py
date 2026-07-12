from __future__ import annotations

from evoked.base import RecordingData, IntermediateResult
from evoked.ols import estimate_scale_ols
import numpy as np
import polars as pl
import polars_config_meta
from pandera.typing.polars import DataFrame
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d


def baseline_correct(recording: DataFrame[RecordingData]) -> DataFrame[RecordingData]:
    corrected = []
    fs = recording.config_meta.get_metadata().get("fs").magnitude
    for _, group in recording.group_by(["id","channel","stimulus","sweep_index"]):
        time = group["time"].to_numpy()
        value = group["value"].to_numpy()
        n_samples = int(1e-4 * fs) # 0.1 ms * Hz 
        baseline = np.mean(value[0:n_samples-1])
        group = group.with_columns(
            (pl.col("value") - baseline).alias("value") # subtract first value in epoch
        )
        
        corrected.append(group)
    combined = pl.concat(corrected, how="vertical")
    combined.config_meta.merge(recording)
    return combined

def detect_stim_artifact(
    value: np.ndarray,
    fs: float,
    mad_threshold: float,
    min_gap_s: float,
    max_duration_s: float,
    padding_s: float,
) -> list[tuple[int, int]]:
    """
    Locate stimulus artifacts by detecting extreme, high-frequency
    biphasic slope changes (positive -> negative or vice versa), then
    pad a fixed amount around the detected burst to capture its settling tail.
    """
    dv = np.diff(value, prepend=value[0])

    med = np.median(dv)
    mad = np.median(np.abs(dv - med))
    sigma = 1.4826 * mad
    threshold = mad_threshold * sigma if sigma > 0 else np.finfo(float).eps

    hit = np.flatnonzero(np.abs(dv) > threshold)
    if hit.size == 0:
        return []

    padding = max(0, round(padding_s * fs))
    min_gap = max(1, round(min_gap_s * fs))
    max_duration = max(1, round(max_duration_s * fs))
    n = len(value)

    # merge raw threshold-crossings that are close enough that padding would
    # make their windows overlap anyway, so one burst doesn't fragment into two
    breaks = np.flatnonzero(np.diff(hit) > max(padding, 1))
    events = np.split(hit, breaks + 1)

    windows = []
    last_start = -min_gap - 1

    for event in events:
        event_dv = dv[event]
        has_pos_peak = np.any(event_dv > threshold)
        has_neg_peak = np.any(event_dv < -threshold)
        if not (has_pos_peak and has_neg_peak):
            continue

        start = max(event[0] - 1 - padding, 0)
        if (start - last_start) < min_gap:
            continue

        stop = min(event[-1] + 1 + padding, start + max_duration, n - 1)

        windows.append((start, stop))
        last_start = start

    return windows

def remove_stim_artifact(
    recording: DataFrame[RecordingData],
    artifact: str = "template",
    mad_threshold: float = 6.0,
    min_gap_s: float = 3e-3,
    max_duration_s: float = 2e-3,
    padding_s: float = 5e-4,
    windows: list[tuple[int,int]] | None = None,
    **kwargs
) -> DataFrame[RecordingData]:
    if artifact not in ["none", "zero", "interp", "template"]:
        raise ValueError("Artifact removal method must be one of: none, zero, interp, or template.")

    if artifact == "none":
        return recording

    fs = recording.config_meta.get_metadata().get("fs").magnitude
    removed = []

    if artifact in ["zero", "interp"]:
        for _, group in recording.group_by(["id", "channel", "stimulus", "sweep_index"]):
            time = group["time"].to_numpy()
            value = group["value"].to_numpy().copy()
            
            if windows is None:
                windows = detect_stim_artifact(value, fs, mad_threshold, min_gap_s, max_duration_s, padding_s)

            for start_idx, stop_idx in windows:
                if artifact == "zero":
                    value[start_idx:stop_idx] = 0.0
                else:
                    value[start_idx:stop_idx] = np.interp(
                        time[start_idx:stop_idx],
                        [time[start_idx - 1], time[stop_idx]],
                        [value[start_idx - 1], value[stop_idx]],
                    )

            group = group.with_columns(pl.Series("value", value))
            removed.append(group)
        combined = pl.concat(removed, how="vertical")
        combined.config_meta.merge(recording)
        return combined

    elif artifact == "template":
        for _, group in recording.group_by(["id", "stimulus", "channel"]):
            sweeps = [sweep for _, sweep in group.group_by(["sweep_index"])]
            values = [sweep["value"].to_numpy().copy() for sweep in sweeps]

            # detect once, on the group-average trace, so every sweep/channel
            # shares identical windows - required for OLS
            avg_value = np.mean(values, axis=0)
            windows = detect_stim_artifact(avg_value, fs, mad_threshold, min_gap_s, max_duration_s, padding_s)

            for start_idx, stop_idx in windows:
                snippets = np.array([v[start_idx:stop_idx] for v in values])
                artifact_template = np.mean(snippets, axis=0)

                for i, snippet in enumerate(snippets):
                    scale = estimate_scale_ols(snippet, artifact_template)
                    if not np.isnan(scale):
                        values[i][start_idx:stop_idx] = snippet - scale * artifact_template

            for sweep, value in zip(sweeps, values):
                removed.append(sweep.with_columns(pl.Series("value", value)))

        combined = pl.concat(removed, how="vertical")
        combined.config_meta.merge(recording)
        return combined

def average_traces(
    recording: DataFrame[RecordingData]
) -> DataFrame[IntermediateResult]:
    averaged = []

    for (stimulus, id_value, channel), group in recording.group_by(["stimulus", "id", "channel"]):
        traces = [sweep["value"].to_numpy() for _, sweep in group.group_by("sweep_index")]

        average_value = np.mean(np.vstack(traces), axis=0)
        time = (
            group
            .filter(pl.col("sweep_index") == group["sweep_index"][0])
            ["time"]
            .to_numpy()
        )

        avg_df = pl.DataFrame({
            "id": id_value,
            "time": time,
            "value": average_value,
            "stimulus": stimulus,
            "channel": channel,
        }).with_columns(
            pl.col("id").cast(pl.String),
            pl.col("time").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
            pl.col("stimulus").cast(pl.String),
            pl.col("channel").cast(pl.Int64),
        )

        averaged.append(avg_df)

    combined = pl.concat(averaged, how="vertical")
    combined.config_meta.merge(recording)
    return combined

def apply_smoothing(intermediate: DataFrame[IntermediateResult], smoothing: str = "savgol",
                    smoothing_params: dict = {
                        "size":7,
                        "polyorder":3,
                        "window_length":11,
                        "cutoff":2000.0,
                        "order":2
                    },
                    **kwargs
) -> DataFrame[IntermediateResult]:
    if smoothing == "none":
            return intermediate
    elif smoothing not in ["uniform", "savgol", "butter"]:
        raise ValueError(
            "Smoothing method must be one of: none, uniform, savgol, or butter."
        )
    
    smoothed = []
    for _, group in intermediate.group_by(["stimulus","id","channel"]):
        time = group["time"].to_numpy()
        value = group["value"].to_numpy()

        if smoothing == "uniform":
            size = smoothing_params.get("size")
            smoothed_value = uniform_filter1d(value,size=size,mode="nearest")
        elif smoothing == "savgol":
            polyorder = smoothing_params.get("polyorder")
            window_length = smoothing_params.get("window_length")
            if window_length % 2 == 0:
                window_length += 1
            smoothed_value = savgol_filter(value, window_length=window_length, polyorder=polyorder)
        elif smoothing == "butter":
            def butter_lowpass(y: np.ndarray, cutoff: float, fs: float, order: int):
                b, a = butter(order, cutoff, btype='low', fs=fs)
                return filtfilt(b, a, y)
            cutoff = smoothing_params.get("cutoff")
            order = smoothing_params.get("order")
            fs = intermediate.config_meta.get_metadata().get("fs")
            smoothed_value = butter_lowpass(value, cutoff=cutoff, fs=fs, order=order)
        
        group = group.with_columns(
            pl.Series("value", smoothed_value)
        )
        smoothed.append(group)
    
    combined = pl.concat(smoothed, how="vertical")
    combined.config_meta.merge(intermediate)
    return combined

def preprocess( # super function
        recording: DataFrame[RecordingData], 
        params: dict | None = None
) -> DataFrame[IntermediateResult]:
    params = params or {}
    corrected = baseline_correct(recording)
    removed = remove_stim_artifact(corrected, **params)
    averaged = average_traces(removed)
    smoothed = apply_smoothing(averaged, **params)
    return smoothed
