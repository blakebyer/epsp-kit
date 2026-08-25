import matplotlib.pyplot as plt
import pytest

from evoked.visualization import plot_all_files, plot_detected, plot_fit, plot_multichannel, plot_trace, plot_io_curve


def test_plot_trace_annotated_draws_feature_markers(intermediate, recording_result):
    fig, ax = plot_trace(
        intermediate, stimuli=["100", "200"], id="S1", channel=0,
        recording_result=recording_result, features=["fEPSP"], annotated=True,
    )
    # 2 stimulus lines + at least one highlighted-segment/marker line per trace
    assert len(ax.lines) > 2
    plt.close(fig)


def test_plot_trace_annotated_warns_when_feature_not_detected_for_stimulus(intermediate):
    # build a recording_result whose fEPSP rows omit channel=1/stimulus="200"
    # entirely, so plot_trace's annotated branch hits the "no detection" warn+continue
    import numpy as np
    import polars as pl
    from evoked.base import FeatureResult, RecordingResult

    df = pl.DataFrame({
        "id": ["S1", "S1", "S1"],
        "channel": [0, 0, 1],
        "stimulus": ["100", "200", "100"],
        "feature_time": [0.015] * 3,
        "amplitude": [1.0, 2.0, 0.8],
        "corr": [0.9, 0.95, 0.85],
        "r2": [0.81, 0.9025, 0.7225],
        "detected": [True, True, True],
    })
    feature_result = FeatureResult(
        window=(0.012, 0.018), search_window=(0.010, 0.022), slope_transform=False,
        snr_threshold=2.0, r2_threshold=0.5, template=np.hanning(6),
        template_keys=[], result=df,
    )
    recording_result = RecordingResult()
    recording_result.add("fEPSP", feature_result)

    with pytest.warns(UserWarning, match="No detection"):
        fig, ax = plot_trace(
            intermediate, stimuli=["200"], id="S1", channel=1,
            recording_result=recording_result, features=["fEPSP"], annotated=True,
        )
    plt.close(fig)


def test_plot_multichannel_warns_on_missing_channel(intermediate):
    with pytest.warns(UserWarning, match="No data for channel"):
        fig, axes = plot_multichannel(intermediate, stimuli=["100"], id="S1", channels=[0, 99])
    assert len(axes) == 1  # only channel 0 actually plotted
    plt.close(fig)


def test_plot_multichannel_missing_data_raises(intermediate):
    with pytest.raises(ValueError, match="No data found"):
        plot_multichannel(intermediate, stimuli=["100"], id="nope", channels=[0])


def test_plot_fit_basic(intermediate, recording_result):
    fig, axes = plot_fit(
        intermediate, recording_result, features=["fEPSP"], stimulus="100", id="S1", channel=0,
    )
    assert axes.shape == (1, 2)
    plt.close(fig)


def test_plot_fit_requires_features(intermediate, recording_result):
    with pytest.raises(ValueError, match="at least one feature"):
        plot_fit(intermediate, recording_result, features=[], stimulus="100", id="S1", channel=0)


def test_plot_fit_missing_data_raises(intermediate, recording_result):
    with pytest.raises(ValueError, match="No data found"):
        plot_fit(intermediate, recording_result, features=["fEPSP"], stimulus="nope", id="S1", channel=0)


def test_plot_fit_unknown_feature_raises_keyerror(intermediate, recording_result):
    with pytest.raises(KeyError):
        plot_fit(intermediate, recording_result, features=["not-a-feature"], stimulus="100", id="S1", channel=0)


def test_plot_detected_skips_missing_feature(recording_result):
    fig, ax = plot_detected(recording_result, features=["fEPSP", "not-a-feature"])
    assert len(ax.lines) == 1  # missing feature silently skipped
    plt.close(fig)


def test_plot_all_files_writes_pdf(tmp_path, intermediate):
    out = tmp_path / "all_files.pdf"
    plot_all_files(intermediate, stimuli=["100", "200"], channel=0, output_path=str(out), max_per_page=6)
    assert out.exists()
    assert out.stat().st_size > 0

def test_plot_trace(intermediate):
    fig, ax = plot_trace(intermediate, stimuli=["100", "200"], id="S1", channel=0)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_trace_missing_data(intermediate):
    with pytest.raises(ValueError, match="No data found"):
        plot_trace(intermediate, stimuli=["100"], id="missing", channel=0)


def test_plot_multichannel(intermediate):
    fig, axes = plot_multichannel(intermediate, stimuli=["100"], id="S1", channels=[0, 1])
    assert len(axes) == 2
    assert all(len(ax.lines) == 1 for ax in axes)
    plt.close(fig)


def test_summary_plots(recording_result):
    fig1, axes = plot_io_curve(recording_result, features=["fEPSP"], stimuli=["100", "200"])
    fig2, ax = plot_detected(recording_result, features=["fEPSP"])
    assert len(axes) == 1
    assert len(ax.lines) == 1
    plt.close(fig1)
    plt.close(fig2)
