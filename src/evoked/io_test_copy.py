from __future__ import annotations
import os
import warnings
import neo
import polars as pl
import polars_config_meta
import quantities as pq
import numpy as np
from evoked.base import RecordingData, RecordingConfig
from concurrent.futures import ThreadPoolExecutor
import yaml


def load_single_file(filename: str, block_index: int) -> neo.Block:
    reader = neo.io.get_io(filename)
    return reader.read_block(block_index=block_index)

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
                    measure: amplitude
                fEPSP:
                    window: [0.0035, 0.005]
                    measure: slope
                Population spike:
                    window: [0.005, 0.007]
                    measure: amplitude
        


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

def process_single_file(filename: str, recordings: dict, block_index: int) -> pl.DataFrame:
    base_name = os.path.basename(filename)
    file_meta = recordings.get(base_name)
    
    block = load_single_file(filename, block_index=block_index)

    expanded_stims = file_meta.expand_stimulus()
    expected_sweeps = len(expanded_stims)
    actual_sweeps = len(block.segments)

    if actual_sweeps != expected_sweeps:
        raise ValueError(
            f"\nSkipping {base_name}: expected {expected_sweeps} sweeps from metadata, "
            f"but file contains {actual_sweeps} sweeps. "
        )
    
    file_dataframes = []
    for sweep_idx, segment in enumerate(block.segments):
        current_stim = expanded_stims[sweep_idx]
        for channel_idx, signal in enumerate(segment.analogsignals):
            time = np.asarray(signal.times, dtype=np.float64).squeeze()
            value = np.asarray(signal, dtype=np.float64).squeeze()

            sweep_df = (
                pl.DataFrame({
                "time": time,
                "value": value,
                })
                .with_columns(
                    pl.lit(file_meta.id).cast(pl.String).alias("id"),
                    pl.lit(channel_idx).cast(pl.Int64).alias("channel"),
                    pl.lit(sweep_idx).cast(pl.Int64).alias("sweep_index"),
                    pl.col("time").cast(pl.Float64),
                    pl.col("value").cast(pl.Float64),
                    pl.lit(str(current_stim)).cast(pl.String).alias("stimulus"),
                ))
            fs = signal.sampling_rate
            fs.units = pq.Hz
            sweep_df.config_meta.set(
                stimulus_unit=file_meta.stimulus_unit.dimensionality,
                time_unit=signal.times.units.dimensionality,
                value_unit=signal.units.dimensionality,
                fs=fs,
            )
            file_dataframes.append(sweep_df)
    combined = pl.concat(file_dataframes, how="vertical")
    combined.config_meta.merge(*file_dataframes)
    return combined

def load_bulk(
    filenames: list[str],
    config_path: str,
    block_index: int = 0
) -> pl.DataFrame:
    config = load_config(config_path)
    recordings = config.metadata.recordings

    # fail before starting parallel work
    missing = [
        os.path.basename(fname)
        for fname in filenames
        if os.path.basename(fname) not in recordings
    ]

    if missing:
        raise ValueError(
            "Missing metadata for file(s): "
            + ", ".join(f"'{name}'" for name in missing)
            + ". Each must be added under the 'recordings' section in your YAML."
        )

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_file, filename, recordings, block_index)
            for filename in filenames
        ]

        dataframes = []
        for filename, future in zip(filenames, futures):
            try:
                dataframes.append(future.result())
            except ValueError as e:
                if "sweeps" in str(e):
                    warnings.warn(
                        f"{e}"
                        f"This file will be omitted from quantification."
                    )
                    continue
                else:
                    raise

    if not dataframes:
        raise ValueError("No files matching the metadata were provided.")

    combined = pl.concat(dataframes, how="vertical")
    combined.config_meta.merge(*dataframes)
    return RecordingData.validate(combined)