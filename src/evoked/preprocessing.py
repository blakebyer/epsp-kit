from __future__ import annotations

from evoked.base import RecordingData, Trials
from evoked.template import estimate_scale
import numpy as np
import polars as pl
import quantities as pq
from typing import Literal, Optional
from scipy.signal import savgol_filter as savgol, butter, filtfilt
from scipy.ndimage import uniform_filter1d


def baseline_correct(
        recording: RecordingData, 
        baseline_window: Optional[tuple[float, float]] = (0.0, 1e-3),
        **kwargs,
    )-> RecordingData:
    if baseline_window is None:
        return recording

    baseline = np.median(
        recording.values(baseline_window),
        axis=1,
        keepdims=True,
    )

    return recording.map_values(
        lambda value: value - baseline
    )

def select_low_noise_channels(
    recording: RecordingData,
    noise_window: tuple[float, float],
    mad_threshold: float = 5.0,
) -> RecordingData:
    noise = recording.values(noise_window)  # (n_trials, n_samples, n_channels)

    center = np.nanmedian(noise, axis=1, keepdims=True)          # (n_trials, 1, n_channels)
    sigma = 1.4826 * np.nanmedian(np.abs(noise - center), axis=1)  # (n_trials, n_channels)

    channel_sigma = np.nanmedian(sigma, axis=0)  # (n_channels,) — typical noise per channel

    ref = np.nanmedian(channel_sigma)
    spread = 1.4826 * np.nanmedian(np.abs(channel_sigma - ref))
    good = channel_sigma <= ref + mad_threshold * spread

    return recording.select_channels([
        name for name, keep in zip(recording.channel_names, good) if keep
    ])

def resample(
    recording: RecordingData,
    target_frequency: float,
) -> RecordingData:
    """Resamples recording to a lower frequency.

    Args:
        recording (RecordingData): 
        target_frequency (float): 

    Raises:
        ValueError: `target_frequency` must be > 0

    Returns:
        RecordingData:
    """
    current_frequency = float(
        recording.sampling_rate.rescale(pq.Hz).magnitude
    )

    if target_frequency <= 0:
        raise ValueError("target_frequency must be > 0.")

    if target_frequency >= current_frequency:
        return recording

    segments = [
        recording._clone_segment(
            seg,
            seg.analogsignals[0].resample(
                sampling_rate=target_frequency * pq.Hz
            ),
        )
        for seg in recording.segments
    ]

    return RecordingData(
        segments=segments,
        trials=recording.trials,
    )

def _detect_stim_artifact(
    value: np.ndarray,
    fs: float,
    snr_threshold: float,
    min_gap_s: float,
    max_duration_s: float,
    padding_s: float,
    biphasic: bool,
    **kwargs,
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
 
def remove_artifacts(
    recording: RecordingData,
    artifact: Literal["zero", "interp", "template", "none"] = "template",
    snr_threshold: float = 10.0,
    min_gap_s: float = 3e-3,
    max_duration_s: float = 2e-3,
    padding_s: float = 1e-3,
    artifact_windows: Optional[list[tuple[float, float]]] = None,
    biphasic: bool = True,
    **kwargs,
) -> RecordingData:
    if artifact not in ["zero", "interp", "template", "none"]:
        raise ValueError(
            "artifact must be one of: none, zero, interp, template"
        )
    
    if artifact == "none":
        return recording

    fs = float(recording.sampling_rate.rescale(pq.Hz).magnitude)
    time = np.asarray(recording.times().rescale(pq.s).magnitude)

    if artifact_windows is not None:
        artifact_windows = [
            (int(round(start * fs)), int(round(stop * fs)))
            for start, stop in artifact_windows
        ]

    if artifact in ["zero", "interp"]:

        def remove(value: np.ndarray) -> np.ndarray:
            value = value.copy()

            for trial in range(value.shape[0]):
                for ch in range(value.shape[2]):
                    trace = value[trial, :, ch]

                    if artifact_windows is None:
                        windows = _detect_stim_artifact(
                            trace,
                            fs,
                            snr_threshold,
                            min_gap_s,
                            max_duration_s,
                            padding_s,
                            biphasic,
                        )
                    else:
                        windows = artifact_windows

                    for start, stop in windows:
                        if artifact == "zero":
                            trace[start:stop] = 0.0
                        else:
                            trace[start:stop] = np.interp(
                                time[start:stop],
                                [time[start - 1], time[stop]],
                                [trace[start - 1], trace[stop]],
                            )

            return value

        return recording.map_values(remove)

    if artifact == "template":

        def remove_template(value: np.ndarray) -> np.ndarray:
            value = value.copy()

            trials = recording.trials.with_row_index("__row")

            for _, group in trials.group_by(
                ["file_origin", "stimulus"],
                maintain_order=True,
            ):
                indices = group["__row"].to_numpy()

                for ch in range(value.shape[2]):
                    traces = value[indices, :, ch]

                    if artifact_windows is None:
                        windows = _detect_stim_artifact(
                            traces.mean(axis=0),
                            fs,
                            snr_threshold,
                            min_gap_s,
                            max_duration_s,
                            padding_s,
                            biphasic,
                        )
                    else:
                        windows = artifact_windows

                    for start, stop in windows:
                        snippets = traces[:, start:stop]
                        template = snippets.mean(axis=0)

                        for i, snippet in enumerate(snippets):
                            scale = estimate_scale(snippet, template)

                            if np.isfinite(scale):
                                traces[i, start:stop] = (
                                    snippet - scale * template
                                )

                    value[indices, :, ch] = traces

            return value

        return recording.map_values(remove_template)

def uniform_filter(recording: RecordingData, size: int = 7):
    return recording.map_values(
            lambda x: uniform_filter1d(
                x, 
                size=size,
                axis=1,
                mode="nearest"
            )
        )

def savgol_filter(recording: RecordingData, polyorder: int = 3, window_length: int = 11):
    if window_length % 2 == 0:
        window_length += 1
    
    return recording.map_values(
        lambda x: savgol(
            x,
            window_length=window_length,
            polyorder=polyorder,
            axis=1
        )
    )

def butter_filter(recording: RecordingData, order: int = 2, cutoff: tuple | float = 2000.0, btype: str = "low"):
    fs = recording.sampling_rate
    b, a = butter(order, cutoff, btype=btype, fs=fs)

    return recording.map_values(
        lambda x: filtfilt(b, a, x, axis=1)
    )

def average_trials(
    recording: RecordingData,
    by: str | list[str],
    **kwargs,
) -> RecordingData:
    by = [by] if isinstance(by, str) else by

    values = recording.values()
    trials = recording.trials.with_row_index("__row")

    segments = []
    rows = []

    for key, group in trials.group_by(by, maintain_order=True):
        positions = group["__row"].to_numpy()
        mean = values[positions].mean(axis=0)

        first = recording.segments[positions[0]]
        signal = first.analogsignals[0].duplicate_with_new_data(mean)

        segments.append(
            recording._clone_segment(first, signal)
        )

        # keep every trial column, not just the groupby keys
        extra_cols = [c for c in trials.columns if c not in by + ["__row"]]
        first_row = group.row(0, named=True)

        rows.append({
            "trial_index": len(rows),
            **{c: first_row[c] for c in extra_cols},
            **dict(zip(by, key if isinstance(key, tuple) else (key,))),
        })

    return RecordingData(
        segments=segments,
        trials=Trials.validate(pl.DataFrame(rows)),
    )

def apply_smoothing(
    recording: RecordingData,
    smoothing: str = "savgol",
    smoothing_params: dict = {
        "size":7,
        "polyorder":3,
        "window_length":11,
        "cutoff":2000.0,
        "order":2
    },
    **kwargs,
) -> RecordingData:
    if smoothing not in ["none", "uniform", "savgol", "butter"]:
        raise ValueError(
            "Smoothing method must be one of: none, uniform, savgol, or butter."
        )

    if smoothing == "none":
        return recording
    

    if smoothing == "uniform":
        size = smoothing_params.get("size")
        return recording.map_values(
            lambda x: uniform_filter1d(
                x, 
                size=size,
                axis=1,
                mode="nearest"
            )
        )
    if smoothing == "savgol":
        polyorder = smoothing_params.get("polyorder")
        window_length = smoothing_params.get("window_length")
        if window_length % 2 == 0:
            window_length += 1

        return recording.map_values(
            lambda x: savgol(
                x,
                window_length=window_length,
                polyorder=polyorder,
                axis=1
            )
        )
    if smoothing == "butter":
        fs = float(recording.sampling_rate.rescale(pq.Hz).magnitude)
        order = smoothing_params.get("order")
        cutoff = smoothing_params.get("cutoff")
        b, a = butter(order, cutoff, btype='low', fs=fs)

        return recording.map_values(
            lambda x: filtfilt(b, a, x, axis=1)
        )

def preprocess(
        recording: RecordingData, 
        params: Optional[dict] = None
) -> RecordingData:
    params = params or {}
    recording = baseline_correct(recording, **params)
    recording = remove_artifacts(recording, **params)
    recording = average_trials(recording, by=["file_origin", "stimulus"])
    recording = apply_smoothing(recording, **params)
    return recording
