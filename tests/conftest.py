import numpy as np
import polars as pl
import polars_config_meta
import pytest
import quantities as pq

from evoked.base import FeatureResult, IntermediateResult, RecordingResult

FS = 1000.0


@pytest.fixture
def intermediate():
    rng = np.random.default_rng(0)
    t = np.arange(40, dtype=np.float32) / FS
    rows = []
    for channel in (0, 1):
        for stimulus, amp in (("100", 1.0), ("200", 2.0)):
            v = rng.normal(0, 0.01, t.size).astype(np.float32)
            v[12:18] += amp * np.hanning(6).astype(np.float32)
            rows.append(("S1", channel, stimulus, t.tolist(), v.tolist()))

    df = pl.DataFrame(
        {
            "id": pl.Series([r[0] for r in rows], dtype=pl.String),
            "channel": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "stimulus": pl.Series([r[2] for r in rows], dtype=pl.String),
            "time": pl.Series([r[3] for r in rows], dtype=pl.List(pl.Float32)),
            "value": pl.Series([r[4] for r in rows], dtype=pl.List(pl.Float32)),
        }
    )
    df.config_meta.set(
        stimulus_unit=pq.uA.dimensionality,
        time_unit=pq.s.dimensionality,
        value_unit=pq.mV.dimensionality,
        fs=FS * pq.Hz,
    )
    return IntermediateResult.validate(df)


@pytest.fixture
def feature_result():
    df = pl.DataFrame(
        {
            "id": ["S1", "S1", "S1", "S1"],
            "channel": [0, 0, 1, 1],
            "stimulus": ["100", "200", "100", "200"],
            "feature_time": [0.015] * 4,
            "amplitude": [1.0, 2.0, 0.8, 1.6],
            "corr": [0.9, 0.95, 0.85, 0.9],
            "r2": [0.81, 0.9025, 0.7225, 0.81],
            "detected": [True, True, True, False],
        }
    )
    df.config_meta.set(
        stimulus_unit=pq.uA.dimensionality,
        time_unit=pq.s.dimensionality,
        value_unit=pq.mV.dimensionality,
        fs=FS * pq.Hz,
    )
    return FeatureResult(
        window=(0.012, 0.018),
        search_window=(0.010, 0.022),
        derivative_transform=False,
        snr_threshold=2.0,
        r2_threshold=0.5,
        template=np.hanning(6),
        template_keys=[],
        result=df,
    )


@pytest.fixture
def recording_result(feature_result):
    out = RecordingResult()
    out.add("fEPSP", feature_result)
    return out
