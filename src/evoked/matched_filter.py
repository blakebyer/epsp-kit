from __future__ import annotations
import numpy as np
from scipy.signal import correlate
from scipy.stats import t as t_dist
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, window_to_indices
from pandera.typing.polars import DataFrame

def center_signal(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    signal_arr = np.asarray(signal, dtype=float)
    return signal_arr - np.mean(signal_arr, axis=axis, keepdims=True)

def linear_fit(time: np.ndarray, signal: np.ndarray):
    time_c = center_signal(time)
    signal_c = center_signal(signal)
    b, _ = np.polyfit(time_c, signal_c, 1)
    sse = np.sum((np.polyval(np.polyfit(time_c, signal_c, 1), time_c) - signal_c)**2)
    b_se = np.sqrt((sse/(len(time_c)-2))/np.sum(time_c**2))
    return b, b_se

def estimate_scale(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size:
        raise ValueError("Snippet and template must have the same length.")
    denom = float(np.dot(template_c, template_c))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(snippet_c, template_c) / denom)

def estimate_snr(
        time: np.ndarray,
        signal: np.ndarray, 
        slope_transform: bool,
        noise: np.ndarray | None = None,
    ):
    if slope_transform:
        b, b_se = linear_fit(time, signal)
        return float(np.abs(b) / b_se)
    else:
        amplitude = np.ptp(signal)
        median_noise = np.median(noise)
        residuals = noise - median_noise
        mad = np.median(np.abs(residuals))
        sigma_noise = 1.4826 * mad
        if sigma_noise <= 1e-20: 
            return np.nan
        return float(amplitude / sigma_noise)

def build_template(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> tuple[np.ndarray, list, float, bool]:
    """Builds templates ranked on high SNR and returns them alongside the contributing keys and their slope_transform state."""
    fs = intermediate.config_meta.get_metadata().get("fs")
    template_snippets = []
    contributing_keys = []
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
        snr = estimate_snr(template_time, template, slope_transform, noise)
        if snr > max_snr:
            max_snr = snr
        if snr < snr_threshold:
            continue
        template_snippets.append(template)
        contributing_keys.append((id_value, channel, stimulus))
    if len(template_snippets) == 0:
        raise ValueError(f"No templates with SNR>={snr_threshold} were found. Max SNR={max_snr:.3f}. Try lowering the threshold.")
    template_array = np.mean(np.vstack(template_snippets), axis=0)
    return template_array, contributing_keys, snr_threshold, slope_transform


def fit_template(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    template_package: tuple[np.ndarray, list, float, bool],
    p_value_threshold: float,
) -> FeatureResult:
    """Fits a pre-built template package tuple: (template_arr, contributing_keys, snr_threshold, slope_transform).
    Slides the template across the entire trace and keeps the single best-correlated match.
    """
    template_arr, contributing_keys, snr_threshold, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    L = template_arr.size
    center_idx = int(L // 2)
    left = center_idx

    template_c = center_signal(template_arr)
    template_ss = np.dot(template_c, template_c)
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

        D = correlate(signal, template_c, mode="valid", method="fft")

        csum = np.concatenate(([0.0], np.cumsum(signal)))
        csum2 = np.concatenate(([0.0], np.cumsum(signal ** 2)))
        window_sum = csum[L:] - csum[:-L]
        window_sumsq = csum2[L:] - csum2[:-L]
        sum_wc2 = window_sumsq - (window_sum ** 2) / L

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = D / (np.sqrt(sum_wc2) * template_norm)
            r2 = corr ** 2
            t_stat = corr * np.sqrt((L - 2) / (1 - r2))

        best_k = int(np.nanargmax(corr))
        n_lags = len(corr)
        best_center = best_k + left
        best_corr = float(corr[best_k])
        best_scale = float(D[best_k] / template_ss)
        best_r2 = best_corr ** 2                          # r2 == corr**2 for single-predictor OLS on centered data
        best_t = t_stat[best_k]
        best_amplitude = best_scale * np.ptp(template_c)

        p_naive = float(t_dist.sf(best_t, df=L - 2))
        p_bonferroni = float(min(1.0, n_lags * p_naive))

        results.append({
            "id": id_value,
            "channel": channel,
            "stimulus": stimulus,
            "feature_time": float(time[best_center]),
            "scale": best_scale,
            "amplitude": best_amplitude, 
            "corr": best_corr, 
            "r2": best_r2,
            "t_stat": best_t,
            "p_value": p_bonferroni,
            "detected": bool(p_bonferroni <= p_value_threshold),
        })

    combined = pl.DataFrame(results)
    combined.config_meta.merge(intermediate)

    return FeatureResult(
        window=window,
        slope_transform=slope_transform,
        snr_threshold=snr_threshold,
        p_value_threshold=p_value_threshold,
        template=template_arr,
        template_keys=contributing_keys,
        result=combined,
    )


def match_feature(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    p_value_threshold: float = 0.05,
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> FeatureResult:
    """Builds a template from all high-SNR trials in `intermediate`, then fits it against
    every trial that didn't contribute to the template."""
    template_package = build_template(
        intermediate=intermediate,
        window=window,
        noise_window=noise_window,
        slope_transform=slope_transform,
        snr_threshold=snr_threshold,
    )
    return fit_template(
        intermediate=intermediate,
        window=window,
        template_package=template_package,
        p_value_threshold=p_value_threshold,
    )