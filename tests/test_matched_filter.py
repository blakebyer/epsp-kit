from __future__ import annotations

import numpy as np
import pytest

from evoked.algorithms.linear import (
    build_template,
    center_signal,
    estimate_scale,
    estimate_snr,
    fit_template,
    linear_fit,
    match_feature,
    window_correlation,
    window_to_indices,
)

FS = 1000.0
TIME = np.arange(60) / FS
WINDOW = (0.012, 0.018)
NOISE_WINDOW = (0.0, 0.01)


def test_center_signal():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = center_signal(x)
    assert out.mean() == pytest.approx(0.0)
    np.testing.assert_allclose(out, x - x.mean())


def test_window_to_indices_basic():
    start, stop = window_to_indices(TIME, (0.01, 0.02), FS)
    assert start == 10
    assert stop - start == 10


def test_window_to_indices_bad_order():
    with pytest.raises(ValueError):
        window_to_indices(TIME, (0.02, 0.01), FS)


def test_window_to_indices_out_of_range():
    with pytest.raises(ValueError):
        window_to_indices(TIME, (0.05, 1.0), FS)


def test_estimate_scale_recovers_known_gain():
    rng = np.random.default_rng(0)
    template = rng.normal(size=20)
    snippet = 2.5 * template + 3.0
    assert estimate_scale(snippet, template) == pytest.approx(2.5, rel=1e-6)


def test_estimate_scale_mismatched_lengths():
    with pytest.raises(ValueError):
        estimate_scale(np.ones(5), np.ones(6))


def test_window_correlation_finds_known_offset():
    rng = np.random.default_rng(1)
    template = rng.normal(size=15)
    signal = rng.normal(size=200) * 0.01
    true_offset = 50
    signal[true_offset:true_offset + len(template)] += template

    corr, dot = window_correlation(signal, template, 0, len(signal))

    best_k = int(np.nanargmax(corr))
    assert best_k == true_offset
    assert corr[best_k] > 0.99
    assert dot[best_k] > 0


def test_window_correlation_invalid_inputs():
    with pytest.raises(ValueError):
        window_correlation(np.ones(10), np.ones(2), 0, 10)
    with pytest.raises(ValueError):
        window_correlation(np.ones(10), np.arange(3), -1, 10)
    with pytest.raises(ValueError):
        window_correlation(np.ones(10), np.arange(5), 0, 3)


def test_linear_fit_recovers_slope():
    t = np.linspace(0, 1, 50)
    slope, se = linear_fit(t, 3.0 * t + 10.0)
    assert slope == pytest.approx(3.0, rel=1e-6)
    assert se == pytest.approx(0.0, abs=1e-6)


def test_estimate_snr_amplitude_mode():
    signal = np.array([0.0, 5.0, 0.0])
    noise = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
    sigma, snr = estimate_snr(
        TIME[:3], signal, slope_transform=False, noise=noise, return_sigma=True
    )
    mad = np.median(np.abs(noise - np.median(noise)))
    expected_sigma = 1.4826 * mad

    assert sigma == pytest.approx(expected_sigma, rel=1e-6)
    assert snr == pytest.approx(5.0 / expected_sigma, rel=1e-6)


def test_estimate_snr_slope_mode():
    t = np.linspace(0, 1, 20)
    signal = 4.0 * t + np.array([
        0.01, -0.01, 0.00, 0.02, -0.01,
        0.00, 0.01, -0.02, 0.01, 0.00,
        -0.01, 0.02, 0.00, -0.01, 0.01,
        0.00, 0.02, -0.01, 0.00, 0.01,
    ])
    sigma, snr = estimate_snr(t, signal, slope_transform=True, return_sigma=True)
    assert sigma > 0
    assert snr > 10


def test_build_template_keeps_high_snr_rows(intermediate):
    template, keys, snr_threshold, slope = build_template(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        snr_threshold=2.0,
    )

    assert template.shape == (6,)
    assert len(keys) == intermediate.height
    assert snr_threshold == 2.0
    assert slope is False
    assert np.corrcoef(template, np.hanning(6))[0, 1] > 0.98


def test_build_template_raises_when_no_row_passes_snr(intermediate):
    with pytest.raises(ValueError):
        build_template(
            intermediate,
            window=WINDOW,
            noise_window=NOISE_WINDOW,
            snr_threshold=1e12,
        )


def test_fit_template_recovers_feature(intermediate):
    package = build_template(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        snr_threshold=2.0,
    )

    result = fit_template(
        intermediate,
        window=WINDOW,
        search_window=0.5,
        template_package=package,
        r2_threshold=0.5,
    )
    df = result.result

    assert df.height == intermediate.height
    assert df["detected"].all()
    assert (df["r2"] > 0.9).all()
    assert np.all(np.abs(df["feature_time"].to_numpy() - 0.015) <= 0.001)
    np.testing.assert_allclose(result.template, package[0])
    assert result.template_keys == package[1]


def test_fit_template_respects_r2_threshold(intermediate):
    package = build_template(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        snr_threshold=2.0,
    )
    result = fit_template(
        intermediate,
        window=WINDOW,
        search_window=0.5,
        template_package=package,
        r2_threshold=1.1,
    )

    assert not result.result["detected"].any()


def test_fit_template_rejects_too_short_template(intermediate):
    with pytest.raises(ValueError):
        fit_template(
            intermediate,
            window=WINDOW,
            search_window=0.5,
            template_package=(np.ones(2), [], 2.0, False),
            r2_threshold=0.5,
        )


def test_fit_template_rejects_search_window_shorter_than_template(intermediate):
    package = build_template(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        snr_threshold=2.0,
    )

    with pytest.raises(ValueError):
        fit_template(
            intermediate,
            window=WINDOW,
            search_window=(0.014, 0.017),
            template_package=package,
            r2_threshold=0.5,
        )


def test_match_feature_detects_known_bump_location(intermediate):
    result = match_feature(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        search_window=0.5,
        r2_threshold=0.5,
        snr_threshold=2.0,
    )
    df = result.result

    assert df.height == intermediate.height
    assert df["detected"].all()
    assert np.all(np.abs(df["feature_time"].to_numpy() - 0.015) <= 0.001)


def test_match_feature_low_r2_threshold_never_fails_detection(intermediate):
    result = match_feature(
        intermediate,
        window=WINDOW,
        noise_window=NOISE_WINDOW,
        r2_threshold=0.0,
        snr_threshold=2.0,
    )

    assert result.result["detected"].all()