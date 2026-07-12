import pyabf
from pathlib import Path
from dataclasses import asdict
from evoked.base import RecordingData, RecordingResult
from pandera.typing.polars import DataFrame
import pandera.polars as pa
import xlsxwriter
import polars as pl
import polars_config_meta
import numpy as np
import warnings
import json
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

"""
    Tabular data should look like:
    id  time  value  stimulus  sweep_index
    <str>  <float>  <float>  <int>  <int>  
"""
# TODO:
# Implement other file types like NWB, or make a neo loader. Make sure users can set the channels

@contextmanager
def as_readable_file(file):
    """
    Yield (name, source) where:
    - name is the original filename-like string
    - source is either the original path or a temporary path

    This makes Streamlit UploadedFile and normal paths behave similarly.
    """
    name = getattr(file, "name", str(file))
    suffix = Path(name).suffix.lower()

    # Normal path-like input
    if isinstance(file, (str, Path)):
        yield name, Path(file)
        return

    # File-like input, e.g. Streamlit UploadedFile
    if hasattr(file, "getvalue"):
        with NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            tmp.flush()
            yield name, Path(tmp.name)
        return

    raise TypeError(f"Unsupported file input type: {type(file)}")

@pa.check_types
def load_abf(
    filename,
    stimuli: list[int],
    id_value: str,
    repnum: int,
) -> DataFrame[RecordingData]:
    abf = pyabf.ABF(filename)
    n_stimuli = len(stimuli)
    n_sweeps = len(abf.sweepList)
    expected = n_stimuli * repnum
    if n_sweeps != expected:
        raise ValueError(f"{filename}: expected {expected} sweeps, got {n_sweeps}")
    data = []
    for sweep_i, sweep_index in enumerate(abf.sweepList):
        abf.setSweep(sweep_index)

        stimulus_index = sweep_i // repnum
        stim_stimulus = stimuli[stimulus_index]

        data.append(
            pl.DataFrame({
                "id": id_value,
                "time": np.asarray(abf.sweepX, dtype=np.float64),
                "value": np.asarray(abf.sweepY, dtype=np.float64),
                "stimulus": pl.Series([stim_stimulus] * len(abf.sweepX), dtype=pl.Int64),
                "sweep_index": pl.Series([sweep_index] * len(abf.sweepX), dtype=pl.Int64),
            })
        )
    return pl.concat(data, how="vertical")

@pa.check_types
def load_csv(filename) -> DataFrame[RecordingData]:
    return pl.read_csv(filename)

@pa.check_types
def load_tsv(filename) -> DataFrame[RecordingData]:
    return pl.read_csv(filename, separator='\t')

@pa.check_types
def load_bulk(
    files,
    stimuli: list[int] | None = None,
    id_values: list[str] | None = None,
    repnum: int | None = None,
) -> DataFrame[RecordingData]:

    recordings = []
    seen_ids = set()
    abf_i = 0

    for file in files:
        with as_readable_file(file) as (name, source):
            suffix = Path(name).suffix.lower()

            try:
                if suffix == ".abf":
                    if any(var is None for var in (stimuli, repnum)):
                        raise ValueError("ABF loading requires stimuli and repnum.")

                    if id_values is None:
                        id_value = Path(name).stem
                    else:
                        if abf_i >= len(id_values):
                            raise ValueError("Not enough id_values provided for ABF files.")
                        id_value = id_values[abf_i]

                    df = load_abf(
                        source,
                        stimuli=stimuli,
                        id_value=id_value,
                        repnum=repnum,
                    )
                    abf_i += 1

                elif suffix == ".csv":
                    df = load_csv(source)

                elif suffix == ".tsv":
                    df = load_tsv(source)

                else:
                    raise ValueError(
                        f"Error reading {name}. "
                        "Suffix must be one of: .abf, .csv, or .tsv"
                    )

                base_id = df["id"][0]
                matches = sum(
                    1 for sid in seen_ids
                    if sid == base_id or sid.startswith(f"{base_id}_")
                )

                if matches:
                    df = df.with_columns(
                        pl.lit(f"{base_id}_{matches}").alias("id")
                    )

                seen_ids.add(df["id"][0])
                recordings.append(df)

            except ValueError as e:
                if "sweeps" in str(e):
                    warnings.warn(
                        f"\nSkipping {name}: {str(e)}. "
                        f"This file will be omitted from quantification."
                    )
                    continue
                else:
                    raise e

    if not recordings:
        raise ValueError("No files were loaded.")

    return pl.concat(recordings, how="vertical")

def save_results_json(recording_result: RecordingResult, filepath: str):
    """Export recording result to JSON"""
    # Convert dataclasses to dicts, then handle dataframes and arrays on the fly
    data = asdict(recording_result)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=lambda x: x.to_dict(orient="records") if isinstance(x, pl.DataFrame) else (x.tolist() if isinstance(x, np.ndarray) else str(x)))


def save_results_xlsx(recording_result: RecordingResult, path):
    workbook = xlsxwriter.Workbook(path)

    # ----------------
    # Pipeline sheet
    # ----------------
    pp = asdict(recording_result.preprocess_params)

    worksheet = workbook.add_worksheet("Pipeline")
    worksheet.write(0, 0, "parameter")
    worksheet.write(0, 1, "value")

    for row_i, (key, value) in enumerate(pp.items(), start=1):
        worksheet.write(row_i, 0, key)

        if isinstance(value, np.ndarray):
            worksheet.write(row_i, 1, json.dumps(value.tolist()))
        elif isinstance(value, (tuple, list, dict)):
            worksheet.write(row_i, 1, json.dumps(value))
        elif value is None:
            worksheet.write(row_i, 1, "")
        else:
            worksheet.write(row_i, 1, value)

    # ----------------
    # One sheet per feature
    # ----------------
    for name, fr in recording_result.results.items():
        sheet_name = name[:31]
        worksheet = workbook.add_worksheet(sheet_name)

        template_val = (
            fr.template.tolist()
            if isinstance(fr.template, np.ndarray)
            else fr.template
        )

        header_rows = [
            ("search_window", json.dumps(fr.search_window)),
            ("template_window", json.dumps(fr.template_window)),
            ("slope_transform", fr.slope_transform),
            ("r2_threshold", fr.r2_threshold),
            ("template", json.dumps(template_val)),
        ]

        worksheet.write(0, 0, "parameter")
        worksheet.write(0, 1, "value")

        for row_i, (key, value) in enumerate(header_rows, start=1):
            worksheet.write(row_i, 0, key)

            if isinstance(value, np.ndarray):
                worksheet.write(row_i, 1, json.dumps(value.tolist()))
            elif isinstance(value, (tuple, list, dict)):
                worksheet.write(row_i, 1, json.dumps(value))
            elif value is None:
                worksheet.write(row_i, 1, "")
            else:
                worksheet.write(row_i, 1, value)

        result_df = (
            fr.result.clone()
            if isinstance(fr.result, pl.DataFrame)
            else pl.from_pandas(fr.result)
        )

        if "corr_arr" in result_df.columns:
            result_df = result_df.with_columns(
                pl.col("corr_arr").map_elements(
                    lambda x: (
                        json.dumps(x.tolist())
                        if isinstance(x, np.ndarray)
                        else json.dumps(x)
                        if isinstance(x, (list, tuple, dict))
                        else "" if x is None
                        else str(x)
                    ),
                    return_dtype=pl.String,
                )
            )

        data_start_row = len(header_rows) + 3

        result_df.write_excel(
            workbook=workbook,
            worksheet=worksheet,
            position=f"A{data_start_row}",
        )

    workbook.close()