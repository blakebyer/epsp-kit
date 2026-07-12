"""Linear discriminant analysis"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, window_to_indices
from pandera.typing.polars import DataFrame
from evoked.ols import center_signal, estimate_snr

def estimate_noise_covariance(noise_snippets: list[np.ndarray]) -> np.ndarray:
    """Estimate diagonal noise covariance matrix"""
    X = np.vstack([center_signal(x) for x in noise_snippets])
    variances = np.var(X, axis=0)
    variances[variances <= 1e-20] = 1e-20 
    return np.diag(variances)

def estimate_score(windows: np.ndarray, template: np.ndarray, covariance_matrix: np.ndarray):
    """Linear discriminant analysis score function, assuming equal lag priors."""
    snippet_c = center_signal(windows)
    template_c = center_signal(template)
    
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)
    Cinv_t = np.linalg.solve(covariance_matrix, template_c)

    left = np.dot(template_c, Cinv_s)
    right = 0.5 * np.dot(template_c, Cinv_t)

    return left - right

def estimate_r2_lda(windows: np.ndarray, template: np.ndarray, covariance_matrix: np.ndarray):
    snippet_c = center_signal(windows)
    template_c = center_signal(template)

    scale = estimate_scale_lda(snippet_c, template_c, covariance_matrix)
    pred = scale * template_c
    resid = snippet_c - pred

    Cinv_resid = np.linalg.solve(covariance_matrix, resid)
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)

    sse = float(np.dot(resid, Cinv_resid))
    sst = float(np.dot(snippet_c, Cinv_s))

    if sst <= 1e-20:
        return np.nan
    return 1.0 - sse / sst

def estimate_posterior(score_arr: np.ndarray) -> np.ndarray:
    """Answers: Given the feature exists somewhere in this search window, which lag is most likely?"""
    score_arr = np.asarray(score_arr, dtype=float)
    score_arr = score_arr - np.nanmax(score_arr)  # numerical stability, posterior unchanged
    exp_scores = np.exp(score_arr)
    return exp_scores / np.nansum(exp_scores)

def estimate_scale_lda(
    windows: np.ndarray,
    template: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    if windows.size != template.size:
        raise ValueError("Snippet and template must have the same length to estimate covariance matrix for LDA.")
    snippet_c = center_signal(windows)
    template_c = center_signal(template)
    if covariance_matrix.shape != (snippet_c.size, snippet_c.size):
        raise ValueError("Covariance matrix has wrong shape.")

    # Solve C^{-1} y and C^{-1} t without explicitly inverting
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)
    Cinv_t = np.linalg.solve(covariance_matrix, template_c)

    denom = float(np.dot(template_c, Cinv_t))
    if denom <= 1e-20:
        return np.nan

    return np.dot(template_c, Cinv_s) / denom

def build_template_lda(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    slope_transform: bool = False,
    mad_threshold: float = 10.0,
) -> tuple[np.ndarray, list[tuple], np.ndarray, bool, float]:
    """Builds a template, covariance matrix, and returns it alongside its slope_transform state."""
    fs = intermediate.config_meta.get_metadata().get("fs")

    n_template_samples = int(round((window[1] - window[0]) * fs))
    n_noise_samples = int(round((noise_window[1] - noise_window[0]) * fs))
    if n_template_samples != n_noise_samples:
        raise ValueError(
            f"Template and noise are not the same length, a requirement for LDA. Template is {n_template_samples} and noise is {n_noise_samples}." 
        )

    template_snippets = []
    noise_snippets = []
    contributing_keys = []
    max_snr = -np.inf

    for (id_value, channel, stimulus), group in intermediate.group_by(["id","channel","stimulus"]):
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
        if snr < mad_threshold:
            continue
        template_snippets.append(signal[template_start:template_stop])
        noise_snippets.append(signal[noise_start:noise_stop])
        contributing_keys.append((id_value, channel, stimulus))

    if len(template_snippets) == 0:
        raise ValueError(f"No templates with SNR>={mad_threshold} were found. Max SNR={max_snr:.3f}. Try lowering the threshold.")
    template_array = np.mean(np.vstack(template_snippets), axis=0)
    covariance_matrix = estimate_noise_covariance(noise_snippets)
    
    return template_array, contributing_keys, covariance_matrix, slope_transform, mad_threshold


def fit_template_lda(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    template_package: tuple[np.ndarray, list[tuple], np.ndarray, bool, float],
    r2_threshold: float,
) -> FeatureResult:
    template_arr, contributing_keys, covariance_matrix, slope_transform, mad_threshold = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    L = template_arr.size
    center_idx = int(L // 2)
    left = center_idx
    right = L - center_idx - 1

    template_c = center_signal(template_arr)                    # (L,)
    Cinv_t = np.linalg.solve(covariance_matrix, template_c)      # (L,)
    denom = float(template_c @ Cinv_t)                           # template_c^T C^-1 template_c

    results = []
    for (id_value, channel, stimulus), group in intermediate.group_by(["id", "channel", "stimulus"]):
        time = group["time"] - group["time"][0]
        time = time.to_numpy()
        signal = group["value"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        windows = sliding_window_view(signal, L)                 # (K, L)
        W_c = windows - windows.mean(axis=1, keepdims=True)       # each row centered

        Cinv_W = np.linalg.solve(covariance_matrix, W_c.T)        # (L, K)
        numerator = template_c @ Cinv_W                            # (K,)

        scale = numerator / denom                                  # (K,)
        score = numerator - 0.5 * denom                             # (K,)

        resid = W_c.T - np.outer(template_c, scale)                 # (L, K)
        Cinv_resid = Cinv_W - np.outer(Cinv_t, scale)                # (L, K)
        sse = np.sum(resid * Cinv_resid, axis=0)
        sst = np.sum(W_c.T * Cinv_W, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(sst > 1e-20, 1.0 - sse / sst, np.nan)

        posterior = estimate_posterior(r2)

        best_k = np.argmax(score)
        best_center = best_k + left
        best_r2 = r2[best_k]

        results.append({
            "id": id_value,
            "channel": channel,
            "stimulus": stimulus,
            "feature_time": float(time[best_center]),
            "scale": float(scale[best_k]),
            "score": float(score[best_k]),
            "score_arr": score,
            "r2": float(r2[best_k]),
            "detected": bool(np.isfinite(best_r2) and best_r2 >= r2_threshold),
            "posterior": float(posterior[best_k]),
        })

    combined = pl.DataFrame(results)
    combined.config_meta.merge(intermediate)

    return FeatureResult(
        window=window,
        slope_transform=slope_transform,
        mad_threshold=mad_threshold,
        r2_threshold=r2_threshold,
        template=template_arr,
        template_keys=contributing_keys,
        result=combined,
    )

def match_feature_lda(
    intermediate: DataFrame[IntermediateResult],
    window: tuple[float, float],
    noise_window: tuple[float, float],
    r2_threshold: float,
    slope_transform: bool = False,
    mad_threshold: float = 10.0,
) -> FeatureResult:
    """Builds a template from training data and fits it directly onto testing data."""
    template_package = build_template_lda(
        intermediate=intermediate,
        window=window,
        noise_window=noise_window,
        slope_transform=slope_transform,
        mad_threshold=mad_threshold,
    )
    return fit_template_lda(
        intermediate=intermediate,
        window=window,
        template_package=template_package,
        r2_threshold=r2_threshold,
    )