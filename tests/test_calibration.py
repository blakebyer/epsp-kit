import numpy as np
import polars as pl
import pytest

from evoked.results import RecordingResult
from evoked.cli.calibration import calibrate, calibrate_all, load_truth


def _pred():
    df = pl.DataFrame({
        "id": ["S0", "S1", "S2", "S3"],
        "channel": [0] * 4,
        "stimulus": ["100"] * 4,
        "feature_time": [0.01] * 4,
        "amplitude": [1.0] * 4,
        "corr": [np.sqrt(x) for x in (0.9, 0.8, 0.2, 0.1)],
        "r2": [0.9, 0.8, 0.2, 0.1],
        "detected": [True] * 4,
    })
    result = RecordingResult()
    result.add("fEPSP", FeatureResult(
        window=(0.0, 0.01), search_window=0.25, slope_transform=False,
        snr_threshold=2.0, r2_threshold=0.5, template=np.zeros(5),
        template_keys=[], result=df,
    ))
    return result


def test_load_truth_coerces_numeric_stimulus(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "id,channel,stimulus,feature,detected\n"
        "S0,0,100,fEPSP,true\n"
    )
    truth = load_truth(str(path))
    assert truth["stimulus"].to_list() == ["100"]


def test_calibrate_finds_perfect_separation():
    truth = pl.DataFrame({
        "id": ["S0", "S1", "S2", "S3"],
        "channel": [0] * 4,
        "stimulus": ["100"] * 4,
        "feature": ["fEPSP"] * 4,
        "detected": [True, True, False, False],
    })
    out = calibrate(truth, _pred(), "fEPSP", "f1")
    assert out["f1"].max() == pytest.approx(1.0)


def test_calibrate_rejects_missing_truth():
    truth = pl.DataFrame({
        "id": ["S0"], "channel": [0], "stimulus": ["100"],
        "feature": ["fEPSP"], "detected": [True],
    })
    with pytest.raises(ValueError, match="no matching truth row"):
        calibrate(truth, _pred(), "fEPSP", "f1")


def _pred_with_two_features():
    def make_df():
        return pl.DataFrame({
            "id": ["S0", "S1", "S2", "S3"],
            "channel": [0] * 4,
            "stimulus": ["100"] * 4,
            "feature_time": [0.01] * 4,
            "amplitude": [1.0] * 4,
            "corr": [np.sqrt(x) for x in (0.9, 0.8, 0.2, 0.1)],
            "r2": [0.9, 0.8, 0.2, 0.1],
            "detected": [True] * 4,
        })
    result = RecordingResult()
    for name in ("fEPSP", "Population spike"):
        result.add(name, FeatureResult(
            window=(0.0, 0.01), search_window=0.25, slope_transform=False,
            snr_threshold=2.0, r2_threshold=0.5, template=np.zeros(5),
            template_keys=[], result=make_df(),
        ))
    return result


def _truth_for(feature):
    return pl.DataFrame({
        "id": ["S0", "S1", "S2", "S3"],
        "channel": [0] * 4,
        "stimulus": ["100"] * 4,
        "feature": [feature] * 4,
        "detected": [True, True, False, False],
    })


def test_load_truth_tsv(tmp_path):
    path = tmp_path / "truth.tsv"
    path.write_text("id\tchannel\tstimulus\tfeature\tdetected\nS0\t0\tpulse1\tfEPSP\ttrue\n")
    truth = load_truth(str(path))
    assert truth["id"].to_list() == ["S0"]


def test_load_truth_xlsx(tmp_path):
    # requires `fastexcel` installed (pl.read_excel dependency)
    path = tmp_path / "truth.xlsx"
    pl.DataFrame({
        "id": ["S0"], "channel": [0], "stimulus": ["pulse1"],
        "feature": ["fEPSP"], "detected": [True],
    }).write_excel(str(path))
    truth = load_truth(str(path))
    assert truth["id"].to_list() == ["S0"]


def test_load_truth_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_truth("/no/such/truth.csv")


def test_load_truth_bad_extension_raises(tmp_path):
    path = tmp_path / "truth.parquet"
    path.write_text("not really parquet")
    with pytest.raises(ValueError, match="File extension"):
        load_truth(str(path))


@pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy", "balanced accuracy"])
def test_calibrate_supported_metrics_run_cleanly(metric):
    truth = _truth_for("fEPSP")
    pred = _pred_with_two_features()
    out = calibrate(truth, pred, "fEPSP", metric)
    assert out.height == 50
    assert metric in out.columns
    assert out[metric].max() == pytest.approx(1.0)


def test_calibrate_invalid_metric_raises():
    truth = _truth_for("fEPSP")
    pred = _pred_with_two_features()
    with pytest.raises(ValueError, match="Metric must be one of"):
        calibrate(truth, pred, "fEPSP", "not-a-real-metric")


def test_calibrate_all_runs_every_feature_in_pred():
    pred = _pred_with_two_features()
    truth = pl.concat([_truth_for("fEPSP"), _truth_for("Population spike")])
    out = calibrate_all(truth, pred, "f1")
    assert set(out.keys()) == {"fEPSP", "Population spike"}
    assert all(df.height == 50 for df in out.values())