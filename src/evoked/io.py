from __future__ import annotations
import os
from pathlib import Path
from fractions import Fraction
import yaml
import json
import warnings
import neo
import mne
import mne_bids
import polars as pl
import polars_config_meta
import quantities as pq
import numpy as np
from scipy.signal import resample_poly
from evoked.base import RecordingData, RecordingConfig, RecordingResult
import json
from typing import Any
import xlsxwriter
import openpyxl


def resolve_filenames(data_dir: str) -> list[str]:
    data_path = Path(data_dir)

    if (data_path / "dataset_description.json").exists():
        matches = mne_bids.find_matching_paths(
            root=str(data_path),
            datatypes=["ieeg", "eeg", "meg"],  
            extensions=[".vhdr", ".edf", ".fif"],
        )
        return [str(p.fpath) for p in matches]

    return [
        str(data_path / f)
        for f in os.listdir(data_path)
        if (data_path / f).is_file()
    ]

def resample_chunk(data: np.ndarray, native_fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """(n_channels, n_samples) -> resampled via polyphase filtering, or no-op
    if already at/below target. Replaces resample_block -- now called per
    epoch chunk instead of on the whole trace up front."""
    if target_fs >= native_fs:
        return data, native_fs
    ratio = Fraction(target_fs / native_fs).limit_denominator(1000)
    resampled = resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    new_fs = native_fs * ratio.numerator / ratio.denominator
    return resampled.astype(np.float32), new_fs

def build_channel_dataframe(
    channel_idx: int,
    epoched_channel_data: np.ndarray,  # (n_events, n_samples)
    relative_times: np.ndarray,
    stimuli: list[str],
    file_id: str,
) -> pl.DataFrame:
    """Build one row per sweep, with array-valued time/value columns."""
    n_events = len(epoched_channel_data)

    return pl.DataFrame(
        {
            "id": pl.Series([file_id] * n_events, dtype=pl.String),
            "channel": pl.Series([channel_idx] * n_events, dtype=pl.Int32),
            "sweep_index": pl.Series(np.arange(n_events), dtype=pl.Int32),
            "stimulus": pl.Series(stimuli, dtype=pl.String),
            "time": pl.Series(
                [relative_times.astype(np.float32).tolist()] * n_events,
                dtype=pl.List(pl.Float32),
            ),
            "value": pl.Series(
                epoched_channel_data.astype(np.float32).tolist(),
                dtype=pl.List(pl.Float32),
            ),
        }
    )


def load_single_file(filename: str, block_index: int):
    """Returns either an mne.io.BaseRaw (BIDS, lazy/preload=False) or a lazy
    neo.Block (lazy=True). No forced materialization, no neo.Block wrapping
    for BIDS data."""
    try:
        bids_path = mne_bids.get_bids_path_from_fname(filename, check=True)
        root = next(
            parent
            for parent in Path(filename).resolve().parents
            if (parent / "dataset_description.json").exists()
        )
        bids_path.update(root=root)
    except (ValueError, StopIteration):
        bids_path = None

    if bids_path is None:
        reader = neo.io.get_io(filename)
        return reader.read_block(block_index=block_index)
    else:
        return mne_bids.read_raw_bids(bids_path, verbose=False)


def process_single_file(
    filename: str,
    recordings: dict,
    epoch: tuple[float, float] | None = None,
    target_frequency: float | None = None,
) -> pl.DataFrame:
    base_name = os.path.basename(filename)
    file_meta = recordings[base_name]

    result = load_single_file(filename, block_index=file_meta.block_index)
    is_bids = isinstance(result, mne.io.BaseRaw)

    stimuli = file_meta.expand_stimulus()
    channel_dfs = []

    if file_meta.layout not in ("continuous", "segments"):
        raise ValueError(
            f"'{base_name}': layout must be 'continuous' or 'segments', "
            f"got {file_meta.layout}"
        )

    if file_meta.layout == "continuous":
        if epoch is None:
            raise ValueError(
                f"'{base_name}' is continuous but no analysis epoch was specified."
            )

        if is_bids:
            if len(result.annotations) == 0:
                raise ValueError(f"'{base_name}' is continuous but contains no events.")
            trigger_times = np.asarray(result.annotations.onset)
            labels = np.asarray(result.annotations.description)
            sfreq = result.info["sfreq"]
            t_start = float(result.first_time)
            ch_names = result.ch_names
            value_unit = pq.V.dimensionality
        else:
            if len(result.segments) != 1:
                raise ValueError(
                    f"'{base_name}' is continuous but contains "
                    f"{len(result.segments)} segments; expected exactly one."
                )
            segment = result.segments[0]
            if not segment.events or len(segment.events[0]) == 0:
                raise ValueError(f"'{base_name}' is continuous but contains no events.")
            events = segment.events[0]
            trigger_times = np.asarray(events.times.rescale(pq.s))
            labels = np.asarray(events.labels)
            signal = segment.analogsignals[0]
            sfreq = float(signal.sampling_rate.rescale(pq.Hz))
            t_start = float(signal.t_start.rescale(pq.s))
            ch_names = signal.array_annotations.get("channel_name", np.array([])).tolist()
            value_unit = signal.units.dimensionality

        if file_meta.event_label is not None:
            trigger_times = trigger_times[labels == file_meta.event_label]

        if len(stimuli) != len(trigger_times):
            raise ValueError(
                f"'{base_name}' has {len(trigger_times)} selected events "
                f"but {len(stimuli)} expanded stimulus values."
            )

        start_offset, end_offset = epoch
        fs = sfreq
        n_samples = int(round((end_offset - start_offset) * sfreq))
        epochs = []
        for t in trigger_times:
            if is_bids:
                start = int(round((t + start_offset - t_start) * sfreq))
                stop = start + n_samples
                chunk = result.get_data(start=start, stop=stop)
            else:
                proxy = segment.analogsignals[0]
                chunk = np.asarray(
                    proxy.load(time_slice=((t + start_offset) * pq.s, (t + end_offset) * pq.s))
                ).T
                chunk = chunk[:, :n_samples] # trim to avoid rounding overshoot
            if target_frequency is not None:
                chunk, fs = resample_chunk(chunk, sfreq, target_frequency)
            epochs.append(chunk.astype(np.float32))

        epoched_matrix = np.stack(epochs, axis=1)  # (n_channels, n_events, n_samples)
        relative_times = np.linspace(
            start_offset, end_offset, epoched_matrix.shape[2], endpoint=False
        )

        for ch_idx in range(epoched_matrix.shape[0]):
            channel_dfs.append(
                build_channel_dataframe(
                    channel_idx=ch_idx,
                    epoched_channel_data=epoched_matrix[ch_idx],
                    relative_times=relative_times,
                    stimuli=stimuli,
                    file_id=file_meta.id,
                )
            )

    else:  # "segments"
        if is_bids:
            raise ValueError(f"'{base_name}': 'segments' layout requires Neo; BIDS files are continuous.")

        if len(result.segments) != len(stimuli):
            raise ValueError(
                f"{base_name}: expected {len(stimuli)} segments "
                f"from metadata, but file contains {len(result.segments)} segments. Skipping..."
            )

        n_channels = len(result.segments[0].analogsignals)
        signal = result.segments[0].analogsignals[0]
        fs = float(signal.sampling_rate.rescale(pq.Hz))
        ch_names = signal.array_annotations.get("channel_name", np.array([])).tolist()
        value_unit = signal.units.dimensionality

        for ch_idx in range(n_channels):
            sweep_signals = [seg.analogsignals[ch_idx] for seg in result.segments]
            epoched_channel_data = np.vstack(
                [np.asarray(s, dtype=np.float32).squeeze() for s in sweep_signals]
            )
            ref_sig = sweep_signals[0]
            relative_times = np.asarray(ref_sig.times.rescale(pq.s), dtype=np.float32).squeeze()

            channel_dfs.append(
                build_channel_dataframe(
                    channel_idx=ch_idx,
                    epoched_channel_data=epoched_channel_data,
                    relative_times=relative_times,
                    stimuli=stimuli,
                    file_id=file_meta.id,
                )
            )

    combined = pl.concat(channel_dfs, how="vertical")
    combined.config_meta.set(
        stimulus_unit=file_meta.stimulus_unit.dimensionality if isinstance(file_meta.stimulus_unit, pq.Quantity) else "",
        time_unit=pq.s.dimensionality,
        value_unit=value_unit,
        fs=fs * pq.Hz,
        channel_names=ch_names,
    )

    print(f"Processed {base_name} as {file_meta.layout}: {len(stimuli)} trials.")
    return combined

def load_config(
        yaml_path: str
    ):
    """
    Users should provide a config YAML file for their experiment.


    `recordings` can be given in either of two forms:

    1) Dict form -- a mapping of filename to per-recording overrides.
    Any field omitted here falls back to the corresponding value in
    `default`, and `id` falls back to the filename's basename stem if
    omitted.

        experiment:
            name: experiment 1
            description: this is an experiment

        metadata:
            default:
                stimulus_unit: uA
                order: grouped
                repeats: 3

            recordings:
                2025_03_02_0000.abf:
                    id: drug_group_1
                    stimulus: [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]
                2025_03_02_0002.abf:
                    id: control_group_1
                    stimulus: [0.1, 1, 5, 10]
                    order: interleaved
                    repeats: 4
                    stimulus_unit: V
                2025_03_05_0007.abf:
                    id: slice_01
                    stimulus: [puff1, puff3, puff18, puff8, puff7, puff2]
                    order: explicit
                    repeats: 1
                paired_pulse_01.abf:
                    id: slice_02
                    stimulus: [pp1, pp2, pp3, pp4]
                    order: explicit
                    repeats: 2

        analysis:
            epoch:
                [-0.002,0.025]

            features:
                Fiber volley:
                    window: [0.0020,0.0035]
                fEPSP:
                    window: [0.0035, 0.005]
                    slope_transform: True
                Population spike:
                    window: [0.005, 0.007]
        


    2) List form -- a bare list of filenames, useful when every
    recording shares the same stimulus/order/repeats/stimulus_unit
    from `default`. `id` is derived from each filename's basename
    stem.

        default:
            stimulus: [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]
            repeats: 3
            stimulus_unit: uA
            order: grouped

        recordings:
            - 2025_03_04_0002.abf
            - 2025_03_03_0000.abf
            - 2025_03_05_0004.abf
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Recording metadata not found at {yaml_path}")

    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    
    return RecordingConfig.model_validate(metadata)

def load_bulk(
    filenames: list[str],
    config_path: str,
) -> pl.DataFrame:
    config = load_config(config_path)
    recordings = config.metadata.recordings
    epoch = config.analysis.epoch
    target_frequency = config.analysis.target_frequency

    filenames = [
        fname for fname in filenames
        if os.path.basename(fname) in recordings
    ]
    if not filenames:
        raise ValueError(
            "None of the provided files have metadata under the 'recordings' "
            "section in your YAML."
        )

    dataframes = []
    for filename in filenames:
        try:
            dataframes.append(
                process_single_file(filename, recordings, epoch, target_frequency)
            )
        except ValueError as e:
            if "expected" in str(e):
                warnings.warn(f"{e} This file will be omitted from quantification.")
                continue
            raise

    if not dataframes:
        raise ValueError(f"Loading for {filenames} failed. Data is empty.")

    combined = pl.concat(dataframes, how="vertical")
    combined.config_meta.merge(*dataframes)
    return RecordingData.validate(combined)

def save_results_json(recording_result: RecordingResult, filepath: str):
    """Export recording result to JSON"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(recording_result.model_dump_json(indent=4))

def load_results_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return RecordingResult.model_validate_json(raw) 

def save_results_xlsx(recording_result: RecordingResult, filepath: str) -> None:
    """Export a RecordingResult to .xlsx: one sheet per feature result,
    a parameter block (every scalar field) followed by its dataframe field(s).
    Driven by type, not field names, so it keeps working as fields change."""

    def scalarize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        elif isinstance(value, (tuple, list, dict)):
            return json.dumps(value, default=str)
        elif value is None:
            return ""
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)

    workbook = xlsxwriter.Workbook(filepath)

    for name, fr in recording_result.results.items():
        worksheet = workbook.add_worksheet(name[:31])  # Excel sheet-name limit

        df_fields = {k: v for k, v in fr.__dict__.items() if isinstance(v, pl.DataFrame)}
        param_fields = {k: v for k, v in fr.__dict__.items() if k not in df_fields}

        worksheet.write_row(0, 0, ["parameter", "value"])
        for row, (key, value) in enumerate(param_fields.items(), start=1):
            worksheet.write(row, 0, key)
            worksheet.write(row, 1, scalarize(value))

        row = len(param_fields) + 2  # blank row, then next table
        for df_name, df in df_fields.items():
            nested = [c for c, dt in zip(df.columns, df.dtypes) if dt.base_type() in (pl.List, pl.Struct, pl.Array, pl.Object)]
            if nested:
                df = df.with_columns(pl.col(c).map_elements(lambda x: json.dumps(x, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) if x is not None else "", return_dtype=pl.String) for c in nested)
            df.write_excel(workbook=workbook, worksheet=worksheet, position=f"A{row + 2}")
            row += df.height + 3

    hidden = workbook.add_worksheet("_data")
    hidden.write(0, 0, recording_result.model_dump_json())
    hidden.hide()

    workbook.close()

def load_results_xlsx(xlsx_path: str) -> RecordingResult:
    """Reload a RecordingResult saved by save_results_xlsx, via its hidden JSON sheet."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    raw = wb["_data"]["A1"].value
    return RecordingResult.model_validate_json(raw)
