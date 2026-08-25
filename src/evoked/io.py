from __future__ import annotations

import os
from pathlib import Path
import yaml
import json
import warnings
import xlsxwriter
import openpyxl
from typing import Any, Optional
import neo
import mne
import mne_bids
import polars as pl
import quantities as pq
import numpy as np
from evoked.base import RecordingData, Trials, TrialCountMismatch, RecordingResult, AlgorithmResult
from evoked.config import RecordingConfig


def resolve_filenames(data_path: str) -> list[str]:
    """Resolve MNE or Neo filenames from a data directory.

    Args:
        data_path (str): Path to data directory.

    Returns:
        list[str]: List of absolute filenames.
    """
    data_path = Path(data_path)

    if (data_path / "dataset_description.json").exists():
        matches = mne_bids.find_matching_paths(
            root=str(data_path),
            datatypes=["ieeg", "eeg", "meg"],
            extensions=[".vhdr", ".edf", ".fif"],
        )
        return [str(p.fpath) for p in matches]

    return [
        str(path)
        for path in data_path.iterdir()
        if path.is_file()
    ]

def downsample_recording(
    segments: list[neo.Segment],
    factor: int,
) -> list[neo.Segment]:
    """From list of `neo.Segment` use `neo.AnalogSignal.downsample()` to downsample the analogsignals.

    Args:
        segments (list[neo.Segment]):
        factor (int): Decimation factor

    Returns:
        list[neo.Segment]:
    """
    if factor <= 1:
        return segments

    out = []

    for seg in segments:
        new_seg = neo.Segment(**seg.annotations)

        signal = seg.analogsignals[0].downsample(factor)
        new_seg.analogsignals.append(signal)

        new_seg.events.extend(seg.events)

        out.append(new_seg)

    return out


def load_bids(path: str) -> mne.io.BaseRaw:
    """Loads `mne.io.BaseRaw` from `path`.

    Args:
        path (str): 

    Returns:
        mne.io.BaseRaw: 
    """
    bids_path = mne_bids.get_bids_path_from_fname(path, check=True)
    root = next(
        parent
        for parent in Path(path).resolve().parents
        if (parent / "dataset_description.json").exists()
    )
    bids_path.update(root=root)
    return mne_bids.read_raw_bids(bids_path, verbose=False)

def load_neo(path: str, block_index: int = 0) -> list[neo.Segment]:
    """Loads list of `neo.Segment` from `path`, defaulting to the first block.

    Args:
        path (str):
        block_index (int, optional): Defaults to 0.

    Returns:
        list[neo.Segment]:
    """
    reader = neo.io.get_io(path)
    block = reader.read_block(block_index=block_index, lazy=True)
    return block.segments

def crop_mne(
    raw: mne.io.BaseRaw,
    epoch: Optional[tuple[float, float]] = None,
) -> neo.Segment:
    """Convert an `mne.io.BaseRaw` object to one `neo.Segment`, optionally cropped with `epoch`.

    Args:
        raw (mne.io.BaseRaw):
        epoch (Optional[tuple[float, float]], optional): Defaults to None.

    Returns:
        neo.Segment:
    """

    if epoch is None:
        start = 0
        stop = len(raw.times)
        t_start = float(raw.first_time)
    else:
        start, stop = raw.time_as_index(epoch)
        t_start = epoch[0]

    chunk = raw.get_data(
        start=start,
        stop=stop,
    )

    signal = neo.AnalogSignal(
        chunk.T,
        units=pq.V,
        sampling_rate=raw.info["sfreq"] * pq.Hz,
        t_start=t_start * pq.s,
        array_annotations={
            "channel_name": np.array(raw.ch_names),
        },
    )

    segment = neo.Segment()
    segment.analogsignals.append(signal)

    times = np.asarray(raw.annotations.onset)
    labels = np.asarray(raw.annotations.description)

    if epoch is not None:
        mask = (
            (times >= epoch[0])
            & (times < epoch[1])
        )
        times = times[mask]
        labels = labels[mask]

    if len(times):
        segment.events.append(
            neo.Event(
                times=times * pq.s,
                labels=labels,
            )
        )

    return segment

def crop_neo(
    segment: neo.Segment,
    epoch: Optional[tuple[float, float]] = None,
) -> neo.Segment:
    """Convert a `neo.Segment` object to one `neo.Segment`, optionally cropped with `epoch`.

    Args:
        segment (neo.Segment): 
        epoch (Optional[tuple[float, float]], optional): Defaults to None.

    Returns:
        neo.Segment:
    """

    proxy = segment.analogsignals[0]

    if epoch is None:
        signal = proxy.load()
    else:
        signal = proxy.load(
            time_slice=(
                epoch[0] * pq.s,
                epoch[1] * pq.s,
            )
        )

    new_segment = neo.Segment(**segment.annotations)
    new_segment.analogsignals.append(signal)

    for events in segment.events:
        if epoch is None:
            new_segment.events.append(events)
            continue

        times = np.asarray(
            events.times.rescale(pq.s).magnitude
        )

        mask = (
            (times >= epoch[0])
            & (times < epoch[1])
        )

        if np.any(mask):
            new_segment.events.append(
                neo.Event(
                    times=events.times[mask],
                    labels=events.labels[mask],
                )
            )

    return new_segment

def epoch_recording(
    data: mne.io.BaseRaw | list[neo.Segment],
    epoch: Optional[tuple[float, float]],
    event_label: Optional[str] = None,
) -> list[neo.Segment]:
    """Epochs an `mne.io.BaseRaw` or list of `neo.Segment` and returns a list of `neo.Segment`.
    If `epoch` is None, returns entire file in one segment.
    If `epoch` is set and `event_label` is None, epochs one segment.
    If `epoch` and `event_label` are set, epochs around `neo.Event.times` and stacks segments into list of length `len(neo.Event.labels)`.

    Args:
        data (mne.io.BaseRaw | list[neo.Segment]): 
        epoch (Optional[tuple[float, float]]): 
        event_label (Optional[str], optional): Defaults to None.

    Raises:
        ValueError: An `epoch` window is required when `event_label` is specified.

    Returns:
        list[neo.Segment]:
    """

    if event_label is None:
        if isinstance(data, mne.io.BaseRaw):
            return [crop_mne(data, epoch)]

        return [
            crop_neo(seg, epoch)
            for seg in data
        ]

    if epoch is None:
        raise ValueError(
            "An epoch window is required when event_label is specified."
        )

    if isinstance(data, mne.io.BaseRaw):
        return epoch_mne(data, epoch, event_label)

    return [
        trial
        for seg in data
        for trial in epoch_neo(seg, epoch, event_label)
    ]


def epoch_mne(raw: mne.io.BaseRaw, 
              epoch: tuple[float, float], 
              event_label: str) -> list[neo.Segment]:
    """From an `mne.io.BaseRaw` object, epochs around events marked with `event_label` to a list of `neo.Segment`.

    Args:
        raw (mne.io.BaseRaw): 
        epoch (tuple[float, float]): 
        event_label (str): 

    Returns:
        list[neo.Segment]: 
    """
    times = np.asarray(raw.annotations.onset)
    labels = np.asarray(raw.annotations.description)
    if event_label is not None:
        mask = labels == event_label
        times = times[mask]
        labels = labels[mask]

    sfreq, t_start = raw.info["sfreq"], float(raw.first_time)
    n_samples = int(round((epoch[1] - epoch[0]) * sfreq))

    trials = []
    for t, label in zip(times, labels):
        start = int(round((t + epoch[0] - t_start) * sfreq))
        chunk = raw.get_data(start=start, stop=start + n_samples)  # (n_channels, n_samples)
        if chunk.shape[1] != n_samples:
            warnings.warn(f"Trial at t={t:.3f}s truncated by recording end, dropping.")
            continue
        signal = neo.AnalogSignal(chunk.T, units=pq.V, sampling_rate=sfreq * pq.Hz,
                                   t_start=epoch[0] * pq.s,
                                   array_annotations={"channel_name": np.array(raw.ch_names)})
        onset = neo.Event(times=np.array([0.0]) * pq.s, labels=np.array([label]))
        trial = neo.Segment()
        trial.analogsignals.append(signal)
        trial.events.append(onset)
        trials.append(trial)
    return trials


def epoch_neo(
    segment: neo.Segment,
    epoch: tuple[float, float],
    event_label: Optional[str] = None,
) -> list[neo.Segment]:
    """From a single `neo.Segment`, epochs around events marked with `event_label` to a list of `neo.Segment`.

    Args:
        segment (neo.Segment): 
        epoch (tuple[float, float]): 
        event_label (Optional[str], optional): Defaults to None.

    Returns:
        list[neo.Segment]:
    """

    events = segment.events[0]
    times = np.asarray(events.times.rescale(pq.s).magnitude)
    labels = np.asarray(events.labels)

    if event_label is not None:
        mask = labels == event_label
        times, labels = times[mask], labels[mask]

    proxy = segment.analogsignals[0]

    trials = []
    for t, label in zip(times, labels):
        signal = proxy.load(
            time_slice=(
                (t + epoch[0]) * pq.s,
                (t + epoch[1]) * pq.s,
            )
        )
        signal.t_start = epoch[0] * pq.s
        onset = neo.Event(times=np.array([0.0]) * pq.s, labels=np.array([label]))
        trial = neo.Segment()
        trial.analogsignals.append(signal)
        trial.events.append(onset)
        trials.append(trial)

    return trials

def load_segments(
    filename: str,
    epoch: Optional[tuple[float, float]] = None,
    event_label: Optional[str] = None,
) -> list[neo.Segment]:
    """
    Loads a MNE or Neo file into list of `neo.Segment`.
    If `epoch` is None, returns entire file in one segment.
    If `epoch` is set and `event_label` is None, epochs one segment.
    If `epoch` and `event_label` are set, epochs around `neo.Event.times` and stacks segments into list of length `len(neo.Event.labels)`.

    Args:
        filename (str):
        epoch (Optional[tuple[float, float]], optional): Defaults to None.
        event_label (Optional[str], optional): Event label in BIDS *_events.tsv. Defaults to None.

    Returns:
        list[neo.Segment]:
    """

    try:
        data = load_bids(filename)
        layout = "continuous"
    except (ValueError, StopIteration):
        data = load_neo(filename)
        layout = "segments" if len(data) > 1 else "continuous"

    segments = epoch_recording(
        data,
        epoch,
        event_label,
    )

    for i, segment in enumerate(segments):
        segment.annotate(
            file_origin=os.path.basename(filename),
            origin_index=i,
        )

    print(
        f"Processed {os.path.basename(filename)} "
        f"as {layout}: {len(segments)} trials."
    )

    return segments
        

def load_config(
        yaml_path: str
    ) -> RecordingConfig:
    """Loads YAML configuration file

    Args:
        yaml_path (str): path to YAML file

    Raises:
        FileNotFoundError: raises if YAML is not found

    Returns:
        RecordingConfig: configuration schema
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Recording metadata not found at {yaml_path}")

    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    
    return RecordingConfig.model_validate(metadata)

def load(
    filenames: list[str],
    epoch: Optional[tuple[float, float]] = None,
    event_label: Optional[str] = None,
) -> list[neo.Segment]:
    """
    Loads MNE or Neo files into a list of `neo.Segment`.
    If `epoch` is None, returns one full-length segment per file.
    If `epoch` is set and `event_label` is None, epochs one segment per file.
    If `epoch` and `event_label` are set, epochs around `neo.Event.times` and stacks segments into list of length `len(neo.Event.labels)` per file.

    Args:
        filenames (list[str]): 
        epoch (Optional[tuple[float, float]], optional): Analysis epoch. Defaults to None.
        event_label (Optional[str], optional): BIDS or Neo event label. Defaults to None.

    Returns:
        list[neo.Segment]:
    """

    return [
        segment
        for filename in filenames
        for segment in load_segments(
            filename,
            epoch,
            event_label,
        )
    ]

def add_trials(
    segments: list[neo.Segment],
    trials: str | pl.DataFrame,
) -> RecordingData:
    """Adds trials to `RecordingData` from path to trials table or from `pl.DataFrame`.

    Args:
        segments (list[neo.Segment]):
        trials (str | pl.DataFrame):

    Raises:
        TrialCountMismatch: Number of trials in `trials` must be the same length as `len(list[neo.Segment])`.

    Returns:
        RecordingData:
    """

    if isinstance(trials, str):
        trials = pl.read_csv(trials, separator="\t")

    if "trial_index" not in trials.columns:
        trials = trials.with_row_index("trial_index")

    if len(trials) != len(segments):
        raise TrialCountMismatch(
            f"{len(segments)} segments but "
            f"{len(trials)} trial rows."
        )

    return RecordingData(
        segments=segments,
        trials=Trials.validate(trials),
    )

def save_results_json(results: RecordingResult, filepath: str) -> None:
    """Saves `results` to machine-readable JSON.

    Args:
        results (RecordingResult): 
        filepath (str): 
    """
    data = {
        name: ar.model_dump(mode="json")
        for name, ar in results.items()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_results_json(filepath: str) -> RecordingResult:
    """Loads a `RecordingResult` from JSON.

    Args:
        filepath (str): 

    Returns:
        RecordingResult: 
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        name: AlgorithmResult.model_validate(value)
        for name, value in data.items()
    }


def save_results_xlsx(
    results: RecordingResult,
    filepath: str,
) -> None:
    """Exports `results` as one sheet per feature plus hidden serialized data in a '_data' sheet.

    Args:
        results (RecordingResult): 
        filepath (str):
    """

    def scalarize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, (tuple, list, dict)):
            return json.dumps(value, default=str)
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    workbook = xlsxwriter.Workbook(filepath)

    for name, ar in results.items():
        worksheet = workbook.add_worksheet(name[:31])

        serialized = ar.model_dump(mode="json")
        params = serialized["algorithm"]

        worksheet.write_row(0, 0, ["parameter", "value"])

        for row, (key, value) in enumerate(params.items(), start=1):
            worksheet.write(row, 0, key)
            worksheet.write(row, 1, scalarize(value))

        df = ar.result

        nested = [
            column
            for column, dtype in zip(df.columns, df.dtypes)
            if dtype.base_type() in (
                pl.List,
                pl.Struct,
                pl.Array,
                pl.Object,
            )
        ]

        if nested:
            df = df.with_columns(
                pl.col(column).map_elements(
                    lambda x: (
                        json.dumps(
                            x,
                            default=lambda o: (
                                o.tolist()
                                if isinstance(o, np.ndarray)
                                else str(o)
                            ),
                        )
                        if x is not None
                        else ""
                    ),
                    return_dtype=pl.String,
                )
                for column in nested
            )

        df.write_excel(
            workbook=workbook,
            worksheet=worksheet,
            position=f"A{len(params) + 4}",
        )

    hidden = workbook.add_worksheet("_data")

    data = {
        name: ar.model_dump(mode="json")
        for name, ar in results.items()
    }

    hidden.write(
        0,
        0,
        json.dumps(data),
    )
    hidden.hide()

    workbook.close()


def load_results_xlsx(
    xlsx_path: str,
) -> RecordingResult:
    """Reload results as `RecordingResult` from the hidden serialized JSON sheet.

    Args:
        xlsx_path (str):

    Returns:
        RecordingResult:
    """
    wb = openpyxl.load_workbook(
        xlsx_path,
        data_only=True,
    )

    raw = wb["_data"]["A1"].value
    data = json.loads(raw)

    return {
        name: AlgorithmResult.model_validate(value)
        for name, value in data.items()
    }