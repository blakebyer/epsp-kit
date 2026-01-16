import pyabf
import pandas as pd
from epspkit.core.context import RecordingContext

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
    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        intensity_index = sweepNumber // repnum
        stim_intensity = stim_intensities[intensity_index]

        data.append(
            pd.DataFrame({
                "time": abf.sweepX,  # seconds
                "voltage": abf.sweepY,  # mV
                "stim_intensity": stim_intensity,  # µA
                "sweep": sweepNumber,
                "repnum": repnum,
            })
        )

    tidy_df = pd.concat(data, ignore_index=True)

    context = RecordingContext(
        tidy=tidy_df,
        averaged=pd.DataFrame(),
        fs=abf.sampleRate,  # sampling frequency in Hz
    )

    return context

def save_context_to_json(
    context: RecordingContext,
    file_path: str
) -> None:
    """
    Save RecordingContext results to a JSON file.

    Parameters
    ----------
    context
        RecordingContext object containing the results to save.
    file_path
        Path to the output JSON file.
    """
    import json

    with open(file_path, 'w') as f:
        json.dump(context.results, f, indent=4)
