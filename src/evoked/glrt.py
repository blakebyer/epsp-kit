from __future__ import annotations
import numpy as np
from scipy.signal import correlate
from scipy.stats import chi2
from scipy.stats import f as f_dist
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, window_to_indices
from evoked.matched_filter import center_signal, estimate_snr
from pandera.typing.polars import DataFrame


def build_template_glrt(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> tuple[np.ndarray, list, float, float, bool]:
    """Builds templates ranked on high SNR and returns them alongside the contributing keys and their slope_transform state."""
    fs = intermediate.config_meta.get_metadata().get("fs")
    template_snippets = []
    contributing_keys = []
    noise_sigmas = []
    max_snr = -np.inf
    for (id_value, channel, stimulus), group in intermediate.group_by(["id", "channel", "stimulus"]):
        time = group["time"] - group["time"][0] # start at t=0
        time = time.to_numpy()
        signal = group["value"].to_numpy()

        template_start, template_stop = window_to_indices(time, window, fs)
        noise_start, noise_stop = window_to_indices(time, noise_window, fs)

        signal = np.gradient(signal, time) if slope_transform else signal
        template_time = time[template_start:template_stop]
        template = signal[template_start:template_stop]
        noise = signal[noise_start:noise_stop]
        noise_med = np.median(noise)
        noise_sigma = 1.4826 * np.median(np.abs(noise - noise_med))
        snr = estimate_snr(template_time, template, slope_transform, noise)
        if snr > max_snr:
            max_snr = snr
        if snr < snr_threshold:
            continue
        noise_sigmas.append(noise_sigma)
        template_snippets.append(template)
        contributing_keys.append((id_value, channel, stimulus))
    if len(template_snippets) == 0:
        raise ValueError(f"No templates with SNR>={snr_threshold} were found. Max SNR={max_snr:.3f}. Try lowering the threshold.")
    template_array = np.mean(np.vstack(template_snippets), axis=0)
    noise_sigma = float(np.median(noise_sigmas))       # pooled, robust estimate
    noise_variance = noise_sigma ** 2
    return template_array, contributing_keys, noise_variance, snr_threshold, slope_transform


def fit_template_glrt(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    template_package: tuple[np.ndarray, list, float, float, bool],
    threshold: float,
) -> FeatureResult:
    """Fits a pre-built template package tuple: (template_array, contributing_keys, slope_transform, snr_threshold).
    Slides the template across the entire trace and keeps the single best-correlated match.
    """
    template_arr, contributing_keys, noise_variance, snr_threshold, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    L = template_arr.size
    center_idx = int(L // 2)
    left = center_idx
    template_c = center_signal(template_arr)
    template_ss = np.sum(template_c ** 2)
    template_norm = np.sqrt(template_ss)

    results = []
    for (id_value, channel, stimulus), group in intermediate.group_by(["id", "channel", "stimulus"]):
        # if (id_value, channel, stimulus) in contributing_keys:
        #     continue  # this trial helped build the template; skip to avoid tautological scoring

        time = group["time"] - group["time"][0] # start at t=0
        time = time.to_numpy()
        signal = group["value"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)    

        signal = center_signal(signal)

        D = correlate(signal, template_c, mode="valid", method="fft")
        stat = D ** 2 / (noise_variance * template_ss)

        csum = np.concatenate(([0.0], np.cumsum(signal)))
        csum2 = np.concatenate(([0.0], np.cumsum(signal ** 2)))
        window_sum = csum[L:] - csum[:-L]
        window_sumsq = csum2[L:] - csum2[:-L]
        sum_wc2 = window_sumsq - (window_sum ** 2) / L

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = D / (np.sqrt(sum_wc2) * template_norm)
            r2 = corr ** 2
            F = (r2 / (1 - r2)) * (L - 2)

        n_lags = len(stat)
        best_k = int(np.nanargmax(F))       
        best_center = best_k + left
        best_r2 = float(r2[best_k])
        best_F = float(F[best_k])
        
        best_scale = float(D[best_k] / template_ss)
        best_amplitude = best_scale * np.ptp(template_c)

        p_naive = float(f_dist.sf(best_F, dfn=1, dfd=L - 2)) 
        p_bonferroni = float(min(1.0, n_lags * p_naive))

        results.append({
            "id": id_value,
            "channel": channel,
            "stimulus": stimulus,
            "feature_time": float(time[best_center]),
            "scale": best_scale, 
            "amplitude": best_amplitude, 
            "r2":best_r2,
            "stat": best_F,
            "p_value": p_bonferroni, 
            "detected": bool(np.isfinite(p_bonferroni) and p_bonferroni < threshold),
        })

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
    every trial that didn't contribute to the template."""
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