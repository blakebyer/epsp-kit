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


def _load_bids(path: str) -> mne.io.BaseRaw:
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

def _load_neo(path: str, block_index: int = 0) -> list[neo.Segment]:
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

def _crop_mne(
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

def _crop_neo(
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
                proxy.t_start + epoch[0] * pq.s,
                proxy.t_start + epoch[1] * pq.s,
            )
        )
        signal.t_start = epoch[0] * pq.s

    if "channel_name" not in signal.array_annotations:
        signal.array_annotate(
            channel_name=np.array([str(i) for i in range(signal.shape[1])])
        )   

    new_segment = neo.Segment(**segment.annotations)
    new_segment.analogsignals.append(signal)

    for events in segment.events:
        events = events.load()
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

def _epoch_raw(
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
            return [_crop_mne(data, epoch)]

        return [
            _crop_neo(seg, epoch)
            for seg in data
        ]

    if epoch is None:
        raise ValueError(
            "An epoch window is required when event_label is specified."
        )

    if isinstance(data, mne.io.BaseRaw):
        return _epoch_mne(data, epoch, event_label)

    return [
        trial
        for seg in data
        for trial in _epoch_neo(seg, epoch, event_label)
    ]


def _epoch_mne(raw: mne.io.BaseRaw, 
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

    if event_label is not None and not np.any(mask):
        available = sorted(set(labels.tolist()))
        raise ValueError(
            f"No events found with label {event_label!r}. "
            f"Available labels in this file: {available}"
        )

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


def _epoch_neo(
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

    events = segment.events[0].load()
    times = np.asarray(events.times.rescale(pq.s).magnitude)
    labels = np.asarray(events.labels)

    if event_label is not None:
        mask = labels == event_label
        times, labels = times[mask], labels[mask]

    if event_label is not None and not np.any(mask):
        available = sorted(set(labels.tolist()))
        raise ValueError(
            f"No events found with label {event_label!r}. "
            f"Available labels in this file: {available}"
        )

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
        
        if "channel_name" not in signal.array_annotations:
            signal.array_annotate(
                channel_name=np.array([str(i) for i in range(signal.shape[1])])
            )   
        trial = neo.Segment()
        trial.analogsignals.append(signal)
        trial.events.append(onset)
        trials.append(trial)

    return trials

def _load_segments(
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
        data = _load_bids(filename)
        layout = "continuous"
    except (ValueError, StopIteration):
        data = _load_neo(filename)
        layout = "segments" if len(data) > 1 else "continuous"

    segments = _epoch_raw(
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

def _load_files(
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
        for segment in _load_segments(
            filename,
            epoch,
            event_label,
        )
    ]


def load_recording(
    filenames: list[str],
    trials: str | pl.DataFrame,
    epoch: Optional[tuple[float, float]] = None,
    event_label: Optional[str] = None,
) -> RecordingData:
    """Load recording from filenames and attach trial metadata.

    If `epoch` is None, loads full-length files.
    If `epoch` is set and `event_label` is None, crops/epochs segments.
    If `epoch` and `event_label` are set, epochs around matching events.

    Args:
        filenames: Recording files to load.
        trials: Trial metadata as a TSV path or Polars DataFrame.
        epoch: Optional epoch window in seconds.
        event_label: Optional event label used for event-based epoching.

    Returns:
        Loaded recording with validated trial metadata.
    """
    filenames = sorted(filenames, key=os.path.basename)
    segments = _load_files(filenames, epoch=epoch, event_label=event_label)

    if isinstance(trials, str):
        trials = pl.read_csv(trials, separator="\t")

    trials = (
        trials
        .sort("file_origin", maintain_order=True)
        .with_row_index("trial_index")
        .with_columns(pl.col("trial_index").cast(pl.Int64))
    )

    if len(trials) != len(segments):
        raise TrialCountMismatch(
            f"{len(segments)} segments but {len(trials)} trial rows."
        )

    expected = sorted({seg.annotations["file_origin"] for seg in segments})
    actual = sorted(trials["file_origin"].unique().to_list())

    if expected != actual:
        raise TrialCountMismatch(
            f"trials.tsv file_origin doesn't match loaded files segment annotations: "
            f"missing={set(expected) - set(actual)}, extra={set(actual) - set(expected)}"
        )

    return RecordingData(segments=segments, trials=Trials.validate(trials))

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


def _load_results_json(filepath: str) -> RecordingResult:
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

        params = {
            **serialized["algorithm"],
            "template": serialized["template"],
        }

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


def _load_results_xlsx(
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