from __future__ import annotations
import numpy as np
from scipy.signal import correlate
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, col_to_2d
from pandera.typing.polars import DataFrame
import warnings

def window_to_indices(
    x: np.ndarray,
    window_s: tuple[float, float],
    fs: float,
) -> tuple[int, int]:
    x = np.asarray(x)  # normalize polars/pandas Series, lists, etc. to ndarray
    t0, t1 = window_s
    if t1 <= t0:
        raise ValueError(f"window_s must satisfy t0 < t1, got {window_s}")

    start = int(np.searchsorted(x, t0))
    n_samples = max(1, int(round((t1 - t0) * fs)))
    stop = start + n_samples

    if stop > x.size:
        raise ValueError(
            f"Window {window_s} requires {n_samples} samples starting at index "
            f"{start}, but this trace only has {x.size} samples total "
            f"({x.size - start} available past the start index)."
        )
    return start, stop

def center_signal(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    signal_arr = np.asarray(signal, dtype=float)
    return signal_arr - np.mean(signal_arr, axis=axis, keepdims=True)

def window_correlation(
    signal: np.ndarray,
    template: np.ndarray,
    search_start: int,
    search_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean-centered Pearson correlation at every valid template
    position inside ``signal[search_start:search_stop]``.

    Returns ``(corr, dot)``, where ``dot`` is the centered template dot
    product used to estimate the OLS scale at each position. Array index ``k``
    corresponds to full-trace template start index ``search_start + k``.
    """
    signal_arr = np.asarray(signal, dtype=float).ravel()
    template_arr = np.asarray(template, dtype=float).ravel()

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")
    if not 0 <= search_start < search_stop <= signal_arr.size:
        raise ValueError(
            f"Invalid search indices ({search_start}, {search_stop}) for "
            f"signal with {signal_arr.size} samples."
        )

    signal_search = signal_arr[search_start:search_stop]
    L = template_arr.size
    if signal_search.size < L:
        raise ValueError(
            f"Search region has {signal_search.size} samples, fewer than "
            f"template length {L}."
        )

    template_c = center_signal(template_arr)
    template_ss = float(np.dot(template_c, template_c))
    if template_ss <= 1e-20:
        raise ValueError("Template has effectively zero variance.")

    dot = correlate(signal_search, template_c, mode="valid", method="fft")

    csum = np.concatenate(([0.0], np.cumsum(signal_search)))
    csum2 = np.concatenate(([0.0], np.cumsum(signal_search ** 2)))
    window_sum = csum[L:] - csum[:-L]
    window_sumsq = csum2[L:] - csum2[:-L]
    window_ss = window_sumsq - (window_sum ** 2) / L
    window_ss = np.maximum(window_ss, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = dot / np.sqrt(window_ss * template_ss)

    corr[window_ss <= 1e-20] = np.nan
    return corr, dot

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
        return_sigma: bool = False,
    ):
    if slope_transform:
        b, b_se = linear_fit(time, signal)
        snr = float(np.abs(b) / b_se)
        return (b_se, snr) if return_sigma else snr
    else:
        amplitude = np.ptp(signal)
        median_noise = np.median(noise)
        residuals = noise - median_noise
        mad = np.median(np.abs(residuals))
        sigma_noise = 1.4826 * mad
        if sigma_noise <= 1e-20:
            return (np.nan, np.nan) if return_sigma else np.nan
        snr = float(amplitude / sigma_noise)
        return (sigma_noise, snr) if return_sigma else snr

    
def build_template(
    intermediate: DataFrame[IntermediateResult], 
    window: tuple[float, float], 
    noise_window: tuple[float, float], 
    slope_transform: bool = False, 
    snr_threshold: float = 10.0) -> tuple[np.ndarray, list, float, bool]:
    fs = intermediate.config_meta.get_metadata().get("fs")
    time = col_to_2d(intermediate, "time")
    value = col_to_2d(intermediate, "value")
    ids, channels, stimuli = intermediate["id"].to_list(), intermediate["channel"].to_list(), intermediate["stimulus"].to_list()

    t0 = time[0] - time[0, 0]                                  # shared across rows -- compute once
    template_start, template_stop = window_to_indices(t0, window, fs)
    noise_start, noise_stop = window_to_indices(t0, noise_window, fs)

    signal = np.gradient(value, t0, axis=1) if slope_transform else value
    template_time = t0[template_start:template_stop]
    templates = signal[:, template_start:template_stop]
    noises = signal[:, noise_start:noise_stop]

    snr_arr = np.array([estimate_snr(template_time, templates[i], slope_transform, noises[i]) for i in range(len(templates))])
    keep = snr_arr >= snr_threshold
    if not np.any(keep):
        raise ValueError(f"No templates with SNR>={snr_threshold}. Median={np.nanmedian(snr_arr):.3f} Max={np.nanmax(snr_arr):.3f}")

    template_array = templates[keep].mean(axis=0)
    contributing_keys = [(ids[i], channels[i], stimuli[i]) for i in range(len(ids)) if keep[i]]
    return template_array, contributing_keys, snr_threshold, slope_transform


def fit_template(
    intermediate: DataFrame[IntermediateResult], 
    window: tuple[float, float], 
    search_window: tuple[float, float] | float, 
    template_package: tuple[np.ndarray, list, float, bool], 
    r2_threshold: float
    ) -> FeatureResult:
    search_start_t, search_stop_t = search_window if isinstance(search_window, tuple) else (
        window[0] - (window[1]-window[0])*search_window, window[1] + (window[1]-window[0])*search_window)
    template_arr, contributing_keys, snr_threshold, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")
    
    L, left = template_arr.size, template_arr.size // 2
    fs = intermediate.config_meta.get_metadata().get("fs")
    template_c = center_signal(template_arr)
    template_ss = np.dot(template_c, template_c)

    time = col_to_2d(intermediate, "time")
    value = col_to_2d(intermediate, "value")
    ids, channels, stimuli = intermediate["id"].to_list(), intermediate["channel"].to_list(), intermediate["stimulus"].to_list()

    t0 = time[0] - time[0, 0]
    signal = np.gradient(value, t0, axis=1) if slope_transform else value
    search_start, search_stop = window_to_indices(
    t0, (search_start_t, search_stop_t), fs)
    search_start, search_stop = max(0, search_start), min(signal.shape[1], search_stop)
    if search_stop - search_start < L:
        raise ValueError(f"search_window {search_window} spans fewer samples than template (L={L}).")

    results = []
    for i in range(signal.shape[0]): # window_correlation is per-trace FFT -- can't batch
        corr, dot = window_correlation(signal[i], template_arr, search_start, search_stop)
        if not np.any(np.isfinite(corr)):
            warnings.warn(f"Correlation undefined for (id={ids[i]}, channel={channels[i]}, stimulus={stimuli[i]}). Skipping...")
            continue
        best_k_local = int(np.nanargmax(corr))
        best_center = search_start + best_k_local + left
        best_corr = float(corr[best_k_local])
        best_r2 = best_corr ** 2 if np.isfinite(best_corr) else np.nan
        best_amplitude = abs(float(dot[best_k_local] / template_ss) * np.ptp(template_c))
        results.append({"id": ids[i], "channel": channels[i], "stimulus": stimuli[i],
                         "feature_time": float(t0[best_center]), "amplitude": best_amplitude,
                         "corr": best_corr, "r2": best_r2, "detected": bool(best_r2 >= r2_threshold)})

    combined = pl.DataFrame(results)
    combined.config_meta.merge(intermediate)
    return FeatureResult(window=window, search_window=search_window, slope_transform=slope_transform,
                          snr_threshold=snr_threshold, r2_threshold=r2_threshold, template=template_arr,
                          template_keys=contributing_keys, result=combined)


def match_feature(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    search_window = 0.25,
    r2_threshold: float = 0.8,
    slope_transform: bool = False,
    snr_threshold: float = 10.0,
) -> FeatureResult:
    """Builds a template from all high-SNR trials in `intermediate`, then fits it against
    every trial."""
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
        search_window=search_window,
        template_package=template_package,
        r2_threshold=r2_threshold,
    )