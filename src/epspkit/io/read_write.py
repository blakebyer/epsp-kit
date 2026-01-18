import pyabf
import pandas as pd
from dataclasses import asdict, is_dataclass
from pathlib import Path
from epspkit.core.context import RecordingContext

# Add other load_*_to_context functions as needed for other filetypes

def load_abf_to_context(
    file_path: str,
    stim_intensities: list[float],
    repnum: int
) -> RecordingContext:
    """
    Load an ABF file and convert it to a RecordingContext.

    Parameters
    ----------
    file_path
        Path to the ABF file.
    stim_intensities
        List of stimulus intensities.
    repnum
        Repetition number.
    Returns
    -------
    RecordingContext
        RecordingContext object containing the data from the ABF file.
    """
    abf = pyabf.ABF(file_path)

    n_intensities = len(stim_intensities)
    n_sweeps = len(abf.sweepList)
    expected = n_intensities * repnum
    if n_sweeps != expected:
        raise ValueError(
            f"{file_path}: expected {expected} sweeps for "
            f"{n_intensities} intensities and repnum={repnum}, got {n_sweeps}"
        )

    data = []
    for sweep_i, sweepNumber in enumerate(abf.sweepList):
        abf.setSweep(sweepNumber)

        intensity_index = sweep_i // repnum
        stim_intensity = stim_intensities[intensity_index]

        sweep_in_rep = (sweep_i % repnum) + 1  # 1..repnum, resets each intensity

        data.append(
            pd.DataFrame({
                "time": abf.sweepX,                 # seconds
                "voltage": abf.sweepY,              # mV
                "stim_intensity": stim_intensity,   # µA
                "abf_sweep": sweepNumber,           # original ABF sweep id/index
                "sweepNumber": sweep_in_rep,        # 1..repnum within each intensity
            })
        )

    tidy_df = pd.concat(data, ignore_index=True)

    context = RecordingContext(
        tidy=tidy_df,
        averaged=pd.DataFrame(),
        fs=abf.sampleRate,  # sampling frequency in Hz
    )

    return context

def save_context_to_xlsx(
    context: RecordingContext,
    file_path: str,
    output_stem: str | None = None,
) -> None:
    """
    Save RecordingContext contents to an Excel workbook.

    Parameters
    ----------
    context
        RecordingContext object containing the results to save.
    file_path
        Path to the output XLSX file.
    output_stem
        Stem used when file_path is a directory.
    """
    def normalize(value):
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return normalize(asdict(value))
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    def safe_sheet_name(name: str, used: set[str]) -> str:
        cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        cleaned = cleaned[:31] if cleaned else "sheet"
        sheet = cleaned
        i = 1
        while sheet in used:
            suffix = f"_{i}"
            sheet = f"{cleaned[:31 - len(suffix)]}{suffix}"
            i += 1
        used.add(sheet)
        return sheet

    output = Path(file_path)
    if output.suffix:
        output = output.with_suffix(".xlsx")
    else:
        stem = output_stem or "recording"
        output = output / f"{stem}_results.xlsx"
    used_sheets: set[str] = set()

    with pd.ExcelWriter(output) as writer:
        if context.tidy is not None and not context.tidy.empty:
            context.tidy.to_excel(
                writer,
                sheet_name=safe_sheet_name("tidy", used_sheets),
                index=False,
            )
        if context.averaged is not None and not context.averaged.empty:
            context.averaged.to_excel(
                writer,
                sheet_name=safe_sheet_name("averaged", used_sheets),
                index=False,
            )

        results = context.results or {}
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sheet_name = safe_sheet_name(f"result_{name}", used_sheets)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        meta = normalize(context.metadata)
        if meta:
            meta_df = pd.json_normalize(meta, sep=".")
            meta_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("metadata", used_sheets),
                index=False,
            )

        cfg = normalize(context.pipeline_cfg) if context.pipeline_cfg is not None else None
        if cfg:
            cfg_df = pd.json_normalize(cfg, sep=".")
            cfg_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pipeline_config", used_sheets),
                index=False,
            )
