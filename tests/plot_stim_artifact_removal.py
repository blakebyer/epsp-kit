import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import quantities as pq

from evoked.base import RecordingData, col_to_2d
from evoked.preprocessing import remove_artifacts

FS = 1000.0
N_PTS = 30
TIME_ROW = np.round(np.arange(N_PTS) / FS, 3).tolist()

def make_trace(amp=6.0, biphasic=True):
    v = np.zeros(N_PTS)
    v[6] = amp
    if biphasic:
        v[7] = -amp
    for i in range(9, N_PTS):
        v[i] += 1.5 * np.exp(-(i - 9) / 6.0)
    return np.round(v, 4).tolist()

def build_recording(artifact="interp", biphasic=True):
    frame = pl.DataFrame({
        "id": pl.Series(["S1"], dtype=pl.String),
        "channel": pl.Series([0], dtype=pl.Int32),
        "sweep_index": pl.Series([0], dtype=pl.Int32),
        "time": pl.Series([TIME_ROW], dtype=pl.List(pl.Float32)),
        "value": pl.Series([make_trace(biphasic=biphasic)], dtype=pl.List(pl.Float32)),
        "stimulus": pl.Series(["100"], dtype=pl.String),
    })
    frame.config_meta.set(
        stimulus_unit=pq.uA.dimensionality,
        time_unit=pq.s.dimensionality,
        value_unit=pq.mV.dimensionality,
        fs=FS * pq.Hz,
    )
    recording = RecordingData.validate(frame)
    cleaned = remove_artifacts(recording, artifact=artifact, biphasic=biphasic)
    return recording, cleaned

if __name__ == "__main__":
    ARTIFACT = "template"  # "zero", "interp", or "template"
    BIPHASIC = False
    raw, cleaned = build_recording(artifact=ARTIFACT,biphasic=BIPHASIC)

    t = col_to_2d(raw, "time")[0]
    before = col_to_2d(raw, "value")[0]
    after = col_to_2d(cleaned, "value")[0]

    plt.plot(t, before, label="before", alpha=0.6)
    plt.plot(t, after, label=f"after ({ARTIFACT})")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.legend()
    plt.title("Stim artifact removal")
    plt.show()