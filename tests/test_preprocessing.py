from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import quantities as pq
from polars.testing import assert_frame_equal

from evoked.base import RecordingData, IntermediateResult
from evoked.preprocessing import (
    average_traces,
    apply_smoothing,
    baseline_correct,
    detect_stim_artifact,
    preprocess,
    remove_artifacts,
)

FS = 1000.0
N_PTS = 30
TIME_ROW = np.round(np.arange(N_PTS) / FS, 3).tolist()

# channel -> baseline offset, stimulus -> evoked response amplitude/sign
BASELINE_OFFSET = {0: 0.3, 1: -0.2}
STIM_AMP = {"100": 1.5, "200": -1.0}
GROUPS = [(0, "100"), (0, "200"), (1, "100"), (1, "200")]


def make_trace(channel, stimulus, sweep, seed, n_artifacts=1, biphasic=True,
               artifact_start=6, artifact_amp=6.0, artifact_spacing=2, response_gap=1):
    # baseline offset + n_artifacts stim blips (default: 1 biphasic blip at
    # idx 6/7, matching original behavior) + decaying evoked response
    rng = np.random.default_rng(seed)
    v = np.zeros(N_PTS)
    v[:] = BASELINE_OFFSET[channel] + 0.05 * sweep + rng.normal(0, 0.01, N_PTS)

    last_idx = artifact_start
    for k in range(n_artifacts):
        i0 = artifact_start + k * artifact_spacing
        if i0 >= N_PTS:
            break
        v[i0] = artifact_amp
        last_idx = i0
        if biphasic and i0 + 1 < N_PTS:
            v[i0 + 1] = -artifact_amp
            last_idx = i0 + 1

    resp_start = last_idx + 1 + response_gap
    for i in range(resp_start, N_PTS):
        v[i] += STIM_AMP[stimulus] * np.exp(-(i - resp_start) / 6.0)
    return np.round(v, 4).tolist()


def recording_rows():
    rows = []
    seed = 0
    for channel, stimulus in GROUPS:
        for sweep in range(3):
            rows.append(dict(
                id="S1", channel=channel, sweep_index=sweep, stimulus=stimulus,
                time=TIME_ROW, value=make_trace(channel, stimulus, sweep, seed),
            ))
            seed += 1
    return rows


def set_meta(frame):
    frame.config_meta.set(
        stimulus_unit=pq.uA.dimensionality,
        time_unit=pq.s.dimensionality,
        value_unit=pq.mV.dimensionality,
        fs=FS * pq.Hz,
    )
    return frame


@pytest.fixture
def recording():
    rows = recording_rows()
    frame = pl.DataFrame({
        "id": pl.Series([r["id"] for r in rows], dtype=pl.String),
        "channel": pl.Series([r["channel"] for r in rows], dtype=pl.Int32),
        "sweep_index": pl.Series([r["sweep_index"] for r in rows], dtype=pl.Int32),
        "time": pl.Series([r["time"] for r in rows], dtype=pl.List(pl.Float32)),
        "value": pl.Series([r["value"] for r in rows], dtype=pl.List(pl.Float32)),
        "stimulus": pl.Series([r["stimulus"] for r in rows], dtype=pl.String),
    })
    frame = set_meta(frame)
    return RecordingData.validate(frame)


@pytest.fixture
def intermediate():
    # mean-across-sweeps continuation of `recording`: one row per (channel,
    # stimulus), still carrying the un-removed stim artifact
    rows = recording_rows()
    avg_rows = []
    for channel, stimulus in GROUPS:
        group_values = np.array([
            r["value"] for r in rows if r["channel"] == channel and r["stimulus"] == stimulus
        ])
        avg_rows.append(dict(
            id="S1", channel=channel, stimulus=stimulus,
            time=TIME_ROW, value=np.round(group_values.mean(axis=0), 4).tolist(),
        ))

    frame = pl.DataFrame({
        "id": pl.Series([r["id"] for r in avg_rows], dtype=pl.String),
        "channel": pl.Series([r["channel"] for r in avg_rows], dtype=pl.Int32),
        "time": pl.Series([r["time"] for r in avg_rows], dtype=pl.List(pl.Float32)),
        "value": pl.Series([r["value"] for r in avg_rows], dtype=pl.List(pl.Float32)),
        "stimulus": pl.Series([r["stimulus"] for r in avg_rows], dtype=pl.String),
    })
    frame = set_meta(frame)
    return IntermediateResult.validate(frame)


def values(frame) -> np.ndarray:
    return np.stack(frame["value"].to_list()).astype(np.float64)


def test_baseline_correct(recording):
    corrected = baseline_correct(recording)

    raw = values(recording)
    expected = raw - raw[:, :1]  # default window (0, 1e-3) -> sample 0 only
    np.testing.assert_allclose(values(corrected), expected, atol=1e-3)
    assert_frame_equal(corrected.drop("value"), recording.drop("value"))


def test_baseline_correct_none_is_noop(recording):
    assert_frame_equal(baseline_correct(recording, baseline_window=None), recording)


stim_artifact_biphasic = {
    "value": np.array([
         0.02, -0.03,  0.01,  0.04,
         0.02,  5.00, -5.00,  0.01, -0.02,  0.03,
         0.01, -0.04,  0.02,  0.03, -0.01,
         0.02, -0.02,  0.01,  0.03,
        -6.00,  6.00,  0.02, -0.03,  0.01,  0.02,
    ]),
    "fs": 1000.0,
    "snr_threshold": 10.0,
    "min_gap_s": 5e-3,
    "max_duration_s": 2e-3,
    "padding_s": 1e-3,
    "biphasic": True,
}

expected_biphasic = [
    (4, 8),     # artifact at samples 5-6 + 1 sample padding
    (18, 22),   # artifact at samples 19-20 + padding
]

stim_artifact_monophasic = {
    "value": np.array([
         0.02, -0.03,  0.01,  0.04,
         5.00,  0.02, -0.02,  0.03,  0.01,
        -0.04,  0.02,  0.03, -0.01,  0.02,
        -6.00,  0.01,  0.03, -0.02,  0.01,
         0.02, -0.03,  0.01,  0.04, -0.02,
         7.00,  0.02, -0.01,  0.03, -0.02,
    ]),
    "fs": 1000.0,
    "snr_threshold": 10.0,
    "min_gap_s": 5e-3,
    "max_duration_s": 2e-3,
    "padding_s": 1e-3,
    "biphasic": False,
}

expected_monophasic = [
    (3, 6),
    (13, 16),
    (23, 26),
]


@pytest.mark.parametrize(
    "stim_artifact, expected",
    [
        (stim_artifact_biphasic, expected_biphasic),
        (stim_artifact_monophasic, expected_monophasic),
    ],
)
def test_detect_stim_artifact(stim_artifact, expected):
    windows = detect_stim_artifact(**stim_artifact)
    assert windows == expected


def test_detect_stim_artifact_on_recording_rows(recording):
    # every row carries the same biphasic blip at idx 6/7, so with
    # remove_artifacts's default params it should resolve to the same
    # window on every row
    for value in values(recording):
        windows = detect_stim_artifact(
            value, FS, snr_threshold=10.0, min_gap_s=3e-3,
            max_duration_s=4e-3, padding_s=1e-3, biphasic=True,
        )
        assert windows == [(5, 10)]


@pytest.mark.parametrize("artifact", ["zero", "interp"])
def test_remove_artifacts_zero_interp(recording, artifact):
    cleaned = remove_artifacts(recording, artifact=artifact)
    raw = values(recording)
    out = values(cleaned)
    start, stop = 5, 10

    if artifact == "zero":
        np.testing.assert_allclose(out[:, start:stop], 0.0)
    else:
        # interpolated values must lie between the flanking samples
        lo = np.minimum(raw[:, start - 1], raw[:, stop])
        hi = np.maximum(raw[:, start - 1], raw[:, stop])
        interior = out[:, start:stop]
        assert np.all(interior >= lo[:, None] - 1e-6)
        assert np.all(interior <= hi[:, None] + 1e-6)

    outside = np.r_[np.arange(0, start), np.arange(stop, N_PTS)]
    np.testing.assert_allclose(out[:, outside], raw[:, outside])


def test_remove_artifacts_none_is_noop(recording):
    assert_frame_equal(remove_artifacts(recording, artifact="none"), recording)


def test_remove_artifacts_template(recording):
    cleaned = remove_artifacts(recording, artifact="template")
    raw = values(recording)
    out = values(cleaned)
    start, stop = 5, 10

    assert np.abs(out[:, start:stop]).max() < 1.0
    assert np.abs(raw[:, start:stop]).max() == pytest.approx(6.0)


def test_remove_artifacts_explicit_windows(recording):
    cleaned = remove_artifacts(recording, artifact="zero", artifact_windows=[(0.005, 0.010)])
    out = values(cleaned)
    np.testing.assert_allclose(out[:, 5:10], 0.0)


def test_average_traces(recording, intermediate):
    averaged = average_traces(recording).sort(["channel", "stimulus"])
    expected = intermediate.sort(["channel", "stimulus"])

    assert set(averaged.columns) == {"id", "channel", "stimulus", "time", "value"}
    np.testing.assert_allclose(values(averaged), values(expected), atol=1e-3)
    assert_frame_equal(
        averaged.select(["id", "channel", "stimulus"]),
        expected.select(["id", "channel", "stimulus"]),
    )


def test_apply_smoothing_preserves_artifact_window(intermediate):
    # bounds nudged just inside sample 5/9's times so float32 rounding of
    # the time column can't push the edge samples out of the mask
    smoothed = apply_smoothing(intermediate, smoothing="savgol", artifact_windows=[(0.0045, 0.0095)])
    raw = values(intermediate)
    out = values(smoothed)

    np.testing.assert_allclose(out[:, 5:10], raw[:, 5:10])
    assert not np.allclose(out[:, 10:], raw[:, 10:])


def test_apply_smoothing_none_is_noop(intermediate):
    assert_frame_equal(apply_smoothing(intermediate, smoothing="none"), intermediate)


def test_apply_smoothing_invalid_method(intermediate):
    with pytest.raises(ValueError):
        apply_smoothing(intermediate, smoothing="not-a-real-method")


def test_preprocess_end_to_end(recording):
    result = preprocess(recording, params={"artifact": "interp", "smoothing": "savgol"})
    assert set(result.columns) == {"id", "channel", "stimulus", "time", "value"}
    assert result.height == len(GROUPS)
    assert np.isfinite(values(result)).all()