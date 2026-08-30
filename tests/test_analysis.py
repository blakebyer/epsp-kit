from types import SimpleNamespace

import pytest

from evoked.cli.analyze import clean_config, describe_config, run_analysis, run_plots, run_feature
from evoked.base import Feature, RecordingConfig, TracePlot, MultiChannelPlot, IOPlot, DetectedPlot


def test_describe_config_default_model():
    text = describe_config()
    assert "experiment:" in text
    assert "metadata:" in text
    assert "analysis:" in text
    assert "plots:" in text
    # effectively-required overrides show up even though pydantic marks them optional
    assert "order: <grouped | interleaved | explicit>  # required" in text
    assert "window: <tuple[number, number]>  # required" in text


def test_describe_config_renders_plot_types():
    text = describe_config()
    for plot_type in ("trace:", "multichannel:", "io:", "fit:", "detected:", "allfiles:"):
        assert plot_type in text


def test_run_feature_returns_name_and_result(monkeypatch, feature_result):
    feature = Feature(window=(0.01, 0.02), noise_window=(0.0, 0.005))
    monkeypatch.setattr("evoked.analysis.match_feature", lambda intermediate, **kwargs: feature_result)

    name, result = run_feature(object(), "fEPSP", feature)
    assert name == "fEPSP"
    assert result is feature_result


def test_run_analysis_raises_when_analysis_is_none():
    config = SimpleNamespace(analysis=None)
    with pytest.raises(ValueError, match="No features configured"):
        run_analysis(object(), config)


def test_run_plots_returns_empty_when_plotting_missing():
    config = SimpleNamespace(plotting=None)
    assert run_plots(object(), object(), config) == {}

    config = SimpleNamespace(plotting=SimpleNamespace(plots=None))
    assert run_plots(object(), object(), config) == {}


def test_run_plots_raises_on_none_plot_config():
    config = SimpleNamespace(plotting=SimpleNamespace(plots={"trace": [None]}))
    with pytest.raises(ValueError, match="no configuration"):
        run_plots(object(), object(), config)


def test_run_plots_multiple_entries_get_indexed_keys(monkeypatch, recording_result):
    plots = [
        TracePlot(id="S1", channel=0, stimuli=["100"]),
        TracePlot(id="S1", channel=1, stimuli=["200"]),
    ]
    config = SimpleNamespace(plotting=SimpleNamespace(plots={"trace": plots}))
    monkeypatch.setattr("evoked.analysis.plot_trace", lambda **kwargs: (kwargs["channel"], "fig"))

    out = run_plots(object(), recording_result, config)
    assert set(out.keys()) == {"trace_0", "trace_1"}
    assert out["trace_0"] == (0, "fig")
    assert out["trace_1"] == (1, "fig")


def test_run_plots_dispatches_every_plot_type(monkeypatch, recording_result):
    dispatch_map = {
        "multichannel": (MultiChannelPlot(id="S1", channels=[0, 1], stimuli=["100"]), "evoked.analysis.plot_multichannel"),
        "io": (IOPlot(features=["fEPSP"], stimuli=["100"], channel=0), "evoked.analysis.plot_io_curve"),
        "detected": (DetectedPlot(features=["fEPSP"], channel=0), "evoked.analysis.plot_detected"),
    }
    for plot_type, (plot_config, target) in dispatch_map.items():
        config = SimpleNamespace(plotting=SimpleNamespace(plots={plot_type: [plot_config]}))
        monkeypatch.setattr(target, lambda **kwargs: "sentinel")
        out = run_plots(object(), recording_result, config)
        assert out[plot_type] == "sentinel"


def test_clean_config_only_returns_explicit_fields():
    feature = Feature(window=(0.01, 0.02), noise_window=(0.0, 0.005))
    assert clean_config(feature) == {
        "window": (0.01, 0.02),
        "noise_window": (0.0, 0.005),
    }


def test_run_analysis_dispatches_features(monkeypatch, feature_result):
    feature = Feature(window=(0.01, 0.02), noise_window=(0.0, 0.005))
    config = SimpleNamespace(analysis=SimpleNamespace(features={"fEPSP": feature}))
    monkeypatch.setattr("evoked.analysis.match_feature", lambda intermediate, **kwargs: feature_result)

    out = run_analysis(object(), config)
    assert out.get("fEPSP") is feature_result


def test_run_analysis_requires_features():
    config = SimpleNamespace(analysis=SimpleNamespace(features={}))
    with pytest.raises(ValueError, match="No features configured"):
        run_analysis(object(), config)


def test_run_plots_dispatches_trace(monkeypatch, recording_result):
    plot = TracePlot(id="S1", channel=0, stimuli=["100"])
    config = SimpleNamespace(plotting=SimpleNamespace(plots={"trace": [plot]}))
    expected = (object(), object())
    monkeypatch.setattr("evoked.analysis.plot_trace", lambda **kwargs: expected)

    out = run_plots(object(), recording_result, config)
    assert out == {"trace": expected}
