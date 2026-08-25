import neo
import mne
import os
import numpy as np
import polars as pl
import quantities as pq
import pytest
import yaml

from evoked.base import Recording, RecordingConfig
from evoked.io import (
    load_bulk,
    load_config,
    load_results_xlsx,
    resample_chunk,
    resolve_filenames,
    save_results_xlsx,
    build_channel_dataframe,
    load_results_yaml,
    process_single_file,
    save_results_yaml,
    load_single_file,
)


MINIMAL_CONFIG = {
    "experiment": {"name": "exp1"},
    "metadata": {
        "recordings": {
            "file1.abf": {
                "id": "S1",
                "stimulus": ["100", "200"],
                "order": "explicit",
                "layout": "segments",
            }
        }
    },
    "analysis": {
        "features": {
            "fEPSP": {"window": [0.01, 0.02], "noise_window": [0.0, 0.005]}
        }
    },
}


def test_resolve_filenames_plain_directory(tmp_path):
    (tmp_path / "a.abf").write_text("x")
    (tmp_path / "b.abf").write_text("x")
    (tmp_path / "sub").mkdir()  # directories should be excluded
    out = sorted(os.path.basename(f) for f in resolve_filenames(str(tmp_path)))
    assert out == ["a.abf", "b.abf"]


def test_resolve_filenames_bids_dispatches_to_mne_bids(tmp_path, monkeypatch):
    (tmp_path / "dataset_description.json").write_text("{}")
    calls = {}

    class FakePath:
        def __init__(self, fpath):
            self.fpath = fpath

    def fake_find_matching_paths(root, datatypes, extensions):
        calls["root"] = root
        calls["datatypes"] = datatypes
        return [FakePath(str(tmp_path / "sub-01_task-x_eeg.edf"))]

    monkeypatch.setattr("evoked.io.mne_bids.find_matching_paths", fake_find_matching_paths)
    out = resolve_filenames(str(tmp_path))
    assert out == [str(tmp_path / "sub-01_task-x_eeg.edf")]
    assert calls["root"] == str(tmp_path)


def test_resample_chunk_noop_when_target_above_native():
    x = np.arange(20, dtype=float).reshape(1, -1)
    y, fs = resample_chunk(x, native_fs=500.0, target_fs=1000.0)
    np.testing.assert_array_equal(y, x)
    assert fs == 500.0


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.yml")


def test_load_config_parses_valid_yaml(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(MINIMAL_CONFIG))
    config = load_config(str(path))
    assert isinstance(config, RecordingConfig)
    assert config.experiment.name == "exp1"
    assert config.metadata.recordings["file1.abf"].id == "S1"


def test_load_bulk_filters_to_configured_filenames(tmp_path, monkeypatch):
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(MINIMAL_CONFIG))

    df = pl.DataFrame({
        "id": pl.Series(["S1"], dtype=pl.String),
        "channel": pl.Series([0], dtype=pl.Int32),
        "sweep_index": pl.Series([0], dtype=pl.Int32),
        "time": pl.Series([[0.0, 0.001]], dtype=pl.List(pl.Float32)),
        "value": pl.Series([[0.0, 1.0]], dtype=pl.List(pl.Float32)),
        "stimulus": pl.Series(["100"], dtype=pl.String),
    })
    df.config_meta.set(stimulus_unit="", time_unit="", value_unit="", fs=1000.0)

    monkeypatch.setattr("evoked.io.process_single_file", lambda filename, recordings, epoch, target_frequency: df)

    result = load_bulk([str(tmp_path / "file1.abf"), str(tmp_path / "unrelated.abf")], str(path))
    assert result.height == 1
    assert result["id"].to_list() == ["S1"]


def test_load_bulk_raises_when_no_filenames_match(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(MINIMAL_CONFIG))
    with pytest.raises(ValueError, match="None of the provided files"):
        load_bulk([str(tmp_path / "unrelated.abf")], str(path))


def test_save_and_load_results_xlsx_roundtrip(tmp_path, recording_result):
    path = tmp_path / "results.xlsx"
    save_results_xlsx(recording_result, str(path))
    assert path.exists()

    loaded = load_results_xlsx(str(path))
    assert loaded.results.keys() == recording_result.results.keys()
    assert loaded.get("fEPSP").result.shape == recording_result.get("fEPSP").result.shape
    assert loaded.get("fEPSP").window == recording_result.get("fEPSP").window


def test_build_channel_dataframe():
    df = build_channel_dataframe(
        1,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([0.0, 0.001, 0.002]),
        ["100", "200"],
        "S1",
    )
    assert df.shape == (2, 6)
    assert df.schema["channel"] == pl.Int32
    assert df["stimulus"].to_list() == ["100", "200"]


def test_resample_chunk_downsamples():
    x = np.arange(20, dtype=float)[None, :]
    y, fs = resample_chunk(x, native_fs=1000.0, target_fs=500.0)
    assert y.shape[-1] == 10
    assert fs == 500.0

def test_load_single_file_neo(monkeypatch):

    expected = neo.Block()
    calls = {}

    class DummyReader:
        def read_block(self, block_index=0, **kwargs):
            calls["block_index"] = block_index
            return expected

    def fake_get_io(filename):
        calls["filename"] = filename
        return DummyReader()

    def not_bids(*args, **kwargs):
        raise ValueError("not a BIDS filename")

    monkeypatch.setattr(
        "evoked.io.mne_bids.get_bids_path_from_fname",
        not_bids,
    )
    monkeypatch.setattr(
        "evoked.io.neo.io.get_io",
        fake_get_io,
    )

    result = load_single_file("recording.abf", block_index=2)

    assert result is expected
    assert calls["filename"] == "recording.abf"
    assert calls["block_index"] == 2

def test_process_continuous_mne(monkeypatch):
    raw = mne.io.RawArray(
        np.vstack([np.arange(100), np.arange(100) * 2.0]),
        mne.create_info(["ch0", "ch1"], sfreq=1000.0, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations(onset=[0.02, 0.06], duration=[0, 0], description=["stim", "stim"]))
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    rec = Recording(
        id="S1", stimulus=[100, 200], order="explicit", layout="continuous",
        stimulus_unit=pq.uA,
    )
    out = process_single_file("fake.edf", {"fake.edf": rec}, epoch=(-0.002, 0.004))

    assert out.height == 4  # 2 channels x 2 events
    assert out["stimulus"].to_list() == ["100", "200", "100", "200"]
    assert out["time"].list.len().unique().to_list() == [6]


def test_results_yaml_roundtrip(tmp_path, recording_result):
    path = tmp_path / "results.yaml"

    save_results_yaml(recording_result, str(path))
    loaded = load_results_yaml(str(path))

    assert loaded.results.keys() == recording_result.results.keys()
    assert loaded.get("fEPSP").result.shape == recording_result.get("fEPSP").result.shape


def _segments_block(n_sweeps=2, n_channels=2, n_samples=10, fs=1000.0):
    block = neo.Block()
    for sweep in range(n_sweeps):
        seg = neo.Segment()
        for ch in range(n_channels):
            data = (np.arange(n_samples) + sweep * 100 + ch * 1000).reshape(-1, 1)
            sig = neo.AnalogSignal(
                data * pq.mV, sampling_rate=fs * pq.Hz, t_start=0.0 * pq.s,
                array_annotations={"channel_name": np.array([f"ch{ch}"])},
            )
            seg.analogsignals.append(sig)
        block.segments.append(seg)
    return block


def test_process_segments_layout_builds_expected_dataframe(monkeypatch):
    block = _segments_block(n_sweeps=2, n_channels=2, n_samples=10, fs=1000.0)
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: block)

    rec = Recording(id="S1", stimulus=["100", "200"], order="explicit", layout="segments")
    out = process_single_file("fake.smr", {"fake.smr": rec})

    assert out.height == 4  # 2 channels x 2 sweeps
    assert set(out["channel"].to_list()) == {0, 1}
    assert out.filter(out["channel"] == 0)["stimulus"].to_list() == ["100", "200"]
    assert out["time"].list.len().unique().to_list() == [10]
    assert out["value"].list.len().unique().to_list() == [10]


def test_process_segments_layout_raises_on_segment_count_mismatch(monkeypatch):
    block = _segments_block(n_sweeps=2, n_channels=1, n_samples=5, fs=1000.0)
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: block)

    # 3 stimulus values expected but file only has 2 segments
    rec = Recording(id="S1", stimulus=["100", "200", "300"], order="explicit", layout="segments")
    with pytest.raises(ValueError, match="expected 3 segments"):
        process_single_file("fake.smr", {"fake.smr": rec})

def _bids_raw(n_channels=2, n_samples=200, fs=1000.0, onsets=(0.02, 0.06), labels=None):
    data = np.vstack([np.arange(n_samples, dtype=float) + ch * 1000 for ch in range(n_channels)])
    raw = mne.io.RawArray(
        data, mne.create_info([f"ch{ch}" for ch in range(n_channels)], sfreq=fs, ch_types="eeg"),
        verbose=False,
    )
    if onsets:
        labels = list(labels) if labels is not None else ["stim"] * len(onsets)
        raw.set_annotations(mne.Annotations(onset=list(onsets), duration=[0] * len(onsets), description=labels))
    return raw


def test_process_continuous_bids_builds_expected_dataframe(monkeypatch):
    raw = _bids_raw(n_channels=2, onsets=(0.02, 0.06))
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    rec = Recording(id="S1", stimulus=["100", "200"], order="explicit", layout="continuous", stimulus_unit=pq.uA)
    out = process_single_file("sub-01_task-x_eeg.edf", {"sub-01_task-x_eeg.edf": rec}, epoch=(-0.002, 0.004))

    assert out.height == 4  # 2 channels x 2 events
    assert set(out["channel"].to_list()) == {0, 1}
    assert out.filter(out["channel"] == 0)["stimulus"].to_list() == ["100", "200"]
    assert out["time"].list.len().unique().to_list() == [6]


def test_process_continuous_bids_filters_by_event_label(monkeypatch):
    raw = _bids_raw(n_channels=1, onsets=(0.02, 0.04, 0.06), labels=["stim", "other", "stim"])
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    rec = Recording(
        id="S1", stimulus=["100", "200"], order="explicit", layout="continuous",
        event_label="stim", stimulus_unit=pq.uA,
    )
    out = process_single_file("sub-01_task-x_eeg.edf", {"sub-01_task-x_eeg.edf": rec}, epoch=(-0.002, 0.004))

    # only the two "stim"-labeled onsets should survive the event_label filter
    assert out.height == 2
    assert out["stimulus"].to_list() == ["100", "200"]


def test_process_continuous_bids_no_annotations_raises(monkeypatch):
    raw = _bids_raw(n_channels=1, onsets=())
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    rec = Recording(id="S1", stimulus=["100"], order="explicit", layout="continuous", stimulus_unit=pq.uA)
    with pytest.raises(ValueError, match="contains no events"):
        process_single_file("sub-01_task-x_eeg.edf", {"sub-01_task-x_eeg.edf": rec}, epoch=(-0.002, 0.004))


def test_process_continuous_bids_stimulus_count_mismatch_raises(monkeypatch):
    raw = _bids_raw(n_channels=1, onsets=(0.02, 0.06))
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    # 2 events in the file but 3 stimulus values expected
    rec = Recording(id="S1", stimulus=["100", "200", "300"], order="explicit", layout="continuous", stimulus_unit=pq.uA)
    with pytest.raises(ValueError, match="selected events"):
        process_single_file("sub-01_task-x_eeg.edf", {"sub-01_task-x_eeg.edf": rec}, epoch=(-0.002, 0.004))


def test_process_continuous_requires_epoch(monkeypatch):
    raw = _bids_raw(n_channels=1, onsets=(0.02,))
    monkeypatch.setattr("evoked.io.load_single_file", lambda filename, block_index: raw)

    rec = Recording(id="S1", stimulus=["100"], order="explicit", layout="continuous", stimulus_unit=pq.uA)
    with pytest.raises(ValueError, match="no analysis epoch"):
        process_single_file("sub-01_task-x_eeg.edf", {"sub-01_task-x_eeg.edf": rec}, epoch=None)