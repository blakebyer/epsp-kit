from __future__ import annotations

from evoked.base import RecordingData, IntermediateResult, col_to_2d, col_from_2d
from evoked.matched_filter import estimate_scale, window_to_indices
import numpy as np
import polars as pl
import polars_config_meta
from pandera.typing.polars import DataFrame
from typing import Literal
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d


# try to streamline dependencies, such as having fewer xlsx readers/writers, get rid of scikit-learn or make it an extra for calibrate
# fix GLRT, test CCEP recipes, add tests/ folder and tests, expand docs, delete unneeded data, push to GitHub! Github actions deployments/versions

def baseline_correct(recording: RecordingData, baseline_window: tuple = (0.0,1e-3), **kwargs) -> DataFrame[RecordingData]:
    if baseline_window is None:
        return recording

    fs = recording.config_meta.get_metadata().get("fs").magnitude
    time = col_to_2d(recording, "time")
    t0 = time[0] - time[0, 0]  
    start, stop = window_to_indices(t0, baseline_window, fs)

    value = col_to_2d(recording, "value")
    baseline = value[:, start:stop].mean(axis=1, keepdims=True)
    corrected = value - baseline

    return recording.with_columns(col_from_2d(corrected, "value"))


def detect_stim_artifact(
    value: np.ndarray,
    fs: float,
    snr_threshold: float,
    min_gap_s: float,
    max_duration_s: float,
    padding_s: float,
    biphasic: bool,
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
    threshold = snr_threshold * sigma if sigma > 0 else np.finfo(float).eps
 
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
 
    artifact_windows = []
    last_start = -min_gap - 1
 
    for event in events:
        event_dv = dv[event]
        has_pos_peak = np.any(event_dv > threshold)
        has_neg_peak = np.any(event_dv < -threshold)
        is_artifact = (has_pos_peak and has_neg_peak) if biphasic else (has_pos_peak or has_neg_peak)
        if not is_artifact:
            continue

        duration = event[-1] - event[0]
        if duration > max_duration:
            continue

        start = max(event[0] - padding, 0)
        if (start - last_start) < min_gap:
            continue

        stop = min(event[-1] + padding, n)

        artifact_windows.append((start, stop))
        last_start = start
 
    return artifact_windows
 
def remove_stim_artifact(
    recording: DataFrame[RecordingData],
    artifact: Literal["zero", "interp", "template", "none"] = "interp",
    snr_threshold: float = 10.0,
    min_gap_s: float = 3e-3,
    max_duration_s: float = 4e-3,
    padding_s: float = 1e-3,
    artifact_windows: list[tuple[float,float]] | None = None,  # in seconds, relative to trace start
    biphasic: bool = True,
    **kwargs
) -> DataFrame[RecordingData]:
    if artifact == "none":
        return recording

    fs = recording.config_meta.get_metadata().get("fs").magnitude
 
    # convert artifact windows to int
    if artifact_windows is not None:
        artifact_windows = [(int(round(start * fs)), int(round(stop * fs))) for start, stop in artifact_windows]
 
    if artifact in ["zero", "interp"]:
        time = col_to_2d(recording, "time")
        value = col_to_2d(recording, "value").copy()

        for i in range(value.shape[0]):
            windows = artifact_windows or detect_stim_artifact(
                value[i], fs, snr_threshold, min_gap_s, max_duration_s, padding_s, biphasic
            )
            for start_idx, stop_idx in windows:
                if artifact == "zero":
                    value[i, start_idx:stop_idx] = 0.0
                else:
                    value[i, start_idx:stop_idx] = np.interp(
                        time[i, start_idx:stop_idx],
                        [time[i, start_idx - 1], time[i, stop_idx]],
                        [value[i, start_idx - 1], value[i, stop_idx]],
                    )

        return recording.with_columns(col_from_2d(value, "value"))
 
    elif artifact == "template":
        groups = []
        for _, group in recording.group_by(["id", "stimulus", "channel"]):
            values = col_to_2d(group, "value").copy()
            windows = artifact_windows or detect_stim_artifact(
                values.mean(axis=0), fs, snr_threshold, min_gap_s, max_duration_s, padding_s, biphasic
            )
            for start, stop in windows:
                snippets = values[:, start:stop]
                template = snippets.mean(axis=0)
                template_c = template - template.mean()
                for i, snippet in enumerate(snippets):
                    scale = estimate_scale(snippet, template)
                    if np.isfinite(scale):
                        values[i, start:stop] = snippet - scale * template_c
            groups.append(group.with_columns(col_from_2d(values, "value")))

        combined = pl.concat(groups, how="vertical")
        combined.config_meta.merge(recording)
        return combined

def average_traces(recording):
    rows = {"id": [], "channel": [], "stimulus": [], "time": [], "value": []}
    for (stimulus, id_, channel), group in recording.group_by(["stimulus", "id", "channel"]):
        values = col_to_2d(group, "value")
        time = col_to_2d(group, "time")[0]
        rows["id"].append(id_)
        rows["channel"].append(channel)
        rows["stimulus"].append(stimulus)
        rows["time"].append(time)
        rows["value"].append(values.mean(axis=0))

    combined = pl.DataFrame({
        "id": pl.Series(rows["id"], dtype=pl.String),
        "channel": pl.Series(rows["channel"], dtype=pl.Int32),
        "stimulus": pl.Series(rows["stimulus"], dtype=pl.String),
        "time": col_from_2d(np.stack(rows["time"]), "time"),
        "value": col_from_2d(np.stack(rows["value"]), "value"),
    })
    combined.config_meta.merge(recording)
    return combined

def apply_smoothing(intermediate: DataFrame[IntermediateResult], 
                    smoothing: str = "savgol",
                    smoothing_params: dict = {
                        "size":7,
                        "polyorder":3,
                        "window_length":11,
                        "cutoff":2000.0,
                        "order":2
                    },
                    artifact_windows: list[tuple[float, float]] | None = None,
                    **kwargs
) -> DataFrame[IntermediateResult]:
    if smoothing == "none":
            return intermediate
    elif smoothing not in ["uniform", "savgol", "butter"]:
        raise ValueError(
            "Smoothing method must be one of: none, uniform, savgol, or butter."
        )

    time = col_to_2d(intermediate, "time")
    value = col_to_2d(intermediate, "value")

    if artifact_windows:
        window_array = np.asarray(artifact_windows, dtype=float)
        t = time[0]
        excluded = np.any(
            (t[:, None] >= window_array[:, 0])
            & (t[:, None] < window_array[:, 1]),
            axis=1,
        )
    else:
        excluded = np.zeros(value.shape[1], dtype=bool)

    if smoothing == "uniform":
        size = smoothing_params.get("size")
        filtered_value = uniform_filter1d(value,size=size,mode="nearest", axis=-1)
    elif smoothing == "savgol":
        polyorder = smoothing_params.get("polyorder")
        window_length = smoothing_params.get("window_length")
        if window_length % 2 == 0:
            window_length += 1
        filtered_value = savgol_filter(value, window_length=window_length, polyorder=polyorder, axis=-1)
    elif smoothing == "butter":
        def butter_lowpass(y: np.ndarray, cutoff: float, fs: float, order: int):
            b, a = butter(order, cutoff, btype='low', fs=fs)
            return filtfilt(b, a, y, axis=-1)
        cutoff = smoothing_params.get("cutoff")
        order = smoothing_params.get("order")
        fs = intermediate.config_meta.get_metadata().get("fs")
        filtered_value = butter_lowpass(value, cutoff=cutoff, fs=fs, order=order)

    # replace with original values to prevent ringing at artifact edges
    filtered_value[:, excluded] = value[:, excluded]

    combined = intermediate.with_columns(col_from_2d(filtered_value, "value"))
    combined.config_meta.merge(intermediate)
    combined.config_meta.set(artifact_windows=artifact_windows)
    return combined

def preprocess(
        recording: DataFrame[RecordingData], 
        params: dict | None = None
) -> DataFrame[IntermediateResult]:
    params = params or {}
    recording = baseline_correct(recording, **params)
    recording = remove_stim_artifact(recording, **params)
    recording = average_traces(recording)
    recording = apply_smoothing(recording, **params)
    return recording
