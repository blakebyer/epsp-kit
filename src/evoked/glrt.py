from __future__ import annotations
import numpy as np
from scipy.stats import t as t_dist
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, col_to_2d
from evoked.matched_filter import window_to_indices, center_signal, estimate_snr, window_correlation
from pandera.typing.polars import DataFrame
import warnings


def build_template_glrt(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> tuple[np.ndarray, list, np.ndarray, float, bool]:
    """Builds templates ranked on high SNR and returns them alongside the contributing keys and their slope_transform state."""
    fs = intermediate.config_meta.get_metadata().get("fs")
    time = col_to_2d(intermediate, "time")
    value = col_to_2d(intermediate, "value")
    ids, channels, stimuli = intermediate["id"].to_list(), intermediate["channel"].to_list(), intermediate["stimulus"].to_list()

    t0 = time[0] - time[0,0] # compute once and share across rows

    template_start, template_stop = window_to_indices(t0, window, fs)
    noise_start, noise_stop = window_to_indices(t0, noise_window, fs)

    signal = np.gradient(value, t0, axis=1) if slope_transform else value
    template_time = t0[template_start:template_stop]
    templates = signal[:, template_start:template_stop]
    noises = signal[:, noise_start:noise_stop]

    snr_res = [
        estimate_snr(template_time, templates[i], slope_transform, noises[i], return_sigma=True)
        for i in range(len(templates))
    ]
    sigma_arr = np.array([r[0] for r in snr_res])
    snr_arr = np.array([r[1] for r in snr_res])

    keep = snr_arr >= snr_threshold
    if not np.any(keep):
        raise ValueError(f"No templates with SNR>={snr_threshold}. Median={np.nanmedian(snr_arr):.3f} Max={np.nanmax(snr_arr):.3f}")

    template_array = templates[keep].mean(axis=0)
    noise_variance = sigma_arr[keep] ** 2
    contributing_keys = [(ids[i], channels[i], stimuli[i]) for i in range(len(ids)) if keep[i]]
    return template_array, contributing_keys, noise_variance, snr_threshold, slope_transform


def fit_template_glrt(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    search_window: tuple[float, float] | float,
    template_package: tuple[np.ndarray, list, float, float, bool],
    threshold: float,
) -> FeatureResult:
    """Fits a pre-built template package tuple: (template_array, contributing_keys, slope_transform, snr_threshold).
    Slides the template across the entire trace and keeps the single best-correlated match.
    """
    search_start_t, search_stop_t = search_window if isinstance(search_window, tuple) else (
            window[0] - (window[1]-window[0])*search_window, window[1] + (window[1]-window[0])*search_window)
    template_arr, contributing_keys, noise_variance, snr_threshold, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    L, left = template_arr.size, template_arr.size // 2
    fs = intermediate.config_meta.get_metadata().get("fs")
    template_c = center_signal(template_arr)
    template_ss = np.sum(template_c ** 2)

    time = col_to_2d(intermediate, "time")
    value = col_to_2d(intermediate, "value")
    ids, channels, stimuli = intermediate["id"].to_list(), intermediate["channel"].to_list(), intermediate["stimulus"].to_list()

    t0 = time[0] - time[0, 0]
    signal = np.gradient(value, t0, axis=1) if slope_transform else value
    search_start, search_stop = window_to_indices(t0, (search_start_t, search_stop_t), fs)
    search_start, search_stop = max(0, search_start), min(signal.shape[1], search_stop)

    results = []
    for i in range(signal.shape[0]):
        corr, dot = window_correlation(signal[i], template_arr, search_start, search_stop)
        if not np.any(np.isfinite(corr)):
            warnings.warn(f"Correlation undefined for (id={ids[i]}, channel={channels[i]}, stimulus={stimuli[i]}). Skipping...")
            continue
        
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = corr ** 2
            stat = dot ** 2 / (noise_variance * template_ss)
            F = (r2 / (1 - r2)) * (L - 2)

        n_lags = len(stat)
        best_k = int(np.nanargmax(F))       
        best_center = best_k + left
        best_r2 = float(r2[best_k])
        best_F = float(F[best_k])
        
        best_scale = float(dot[best_k] / template_ss)
        best_amplitude = best_scale * np.ptp(template_c)

        p_naive = float(f_dist.sf(best_F, dfn=1, dfd=L - 2)) 
        p_bonferroni = float(min(1.0, n_lags * p_naive))

        results.append({
            "id": ids[i],
            "channel": channels[i],
            "stimulus": stimuli[i],
            "feature_time": float(t0[best_center]),
            "amplitude": best_amplitude,
            "corr": best_corr, "r2": best_r2, "detected": bool(best_r2 >= threshold)})

    combined = pl.DataFrame(results)
    combined.config_meta.merge(intermediate)

    return FeatureResult(
        method='glrt',
        window=window,
        slope_transform=slope_transform,
        snr_threshold=snr_threshold,
        threshold=threshold,
        template=template_arr,
        template_keys=contributing_keys,
        result=combined,
    )


def match_feature_glrt(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    threshold: float = 0.05,
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> FeatureResult:
    """Builds a template from all high-SNR trials in `intermediate`, then fits it against
    every trial."""
    template_package = build_template_glrt(
        intermediate=intermediate,
        window=window,
        noise_window=noise_window,
        slope_transform=slope_transform,
        snr_threshold=snr_threshold,
    )
    return fit_template_glrt(
        intermediate=intermediate,
        window=window,
        template_package=template_package,
        threshold=threshold,
    )