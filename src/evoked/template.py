from __future__ import annotations

import numpy as np
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer
from scipy.signal import correlate
import quantities as pq
from evoked.base import RecordingData
import warnings

def build_template(
        recording: RecordingData,
        window: tuple[float, float],
    ) -> np.ndarray:
    return np.mean(recording.values(window), axis=0)

def build_template_snr(
    recording: RecordingData,
    window: tuple[float, float],
    noise_window: tuple[float, float] = (0.0, 1e-3),
    snr_threshold: float = 10.0,
) -> np.ndarray:
    value = recording.values(window)  # (n_trials, n_samples, n_channels)
    noise = recording.values(noise_window)

    snr = np.array([
        [
            estimate_snr(
                value[i, :, ch],
                noise[i, :, ch],
            )
            for ch in range(value.shape[2])
        ]
        for i in range(value.shape[0])
    ])

    keep = snr >= snr_threshold

    vmax = np.max(value, axis=1)
    vmin = np.min(value, axis=1)

    trial_sign = np.sign(
        np.where(
            np.abs(vmax) >= np.abs(vmin),
            vmax,
            vmin,
        )
    )

    dominant_sign = np.sign(
        np.sum(
            np.where(keep, trial_sign, 0),
            axis=0,
        )
    )

    keep &= trial_sign == dominant_sign

    valid_channels = keep.any(axis=0)

    for ch in np.flatnonzero(~valid_channels):
        name = (
            recording.channel_names[ch]
            if recording.channel_names
            else str(ch)
        )

        warnings.warn(
            f"No trials with SNR >= {snr_threshold} and dominant polarity "
            f"for channel {name}. "
            f"Max SNR={np.nanmax(snr[:, ch]):.2f}."
        )

    return np.nanmean(
        np.where(
            keep[:, None, :],
            value,
            np.nan,
        ),
        axis=0,
    )

def center_signal(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    return signal - np.mean(signal, axis=axis, keepdims=True)

def window_correlation(
    signal: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized correlation

    Args:
        signal (np.ndarray): 
        template (np.ndarray):

    Returns:
        tuple[np.ndarray, np.ndarray]: corr, dot
    """
    signal = np.asarray(signal, dtype=float).ravel()
    template = np.asarray(template, dtype=float).ravel()

    L = template.size
    if signal.size < L:
        raise ValueError(
            f"Signal has fewer samples ({signal.size}) than "
            f"template ({L})."
        )

    template_c = center_signal(template)
    template_ss = float(np.dot(template_c, template_c))
    if template_ss <= 1e-20 or np.isnan(template_ss):
        raise ValueError("Template has effectively zero variance.")

    dot = correlate(signal, template_c, mode="valid", method="fft")

    csum = np.concatenate(([0.0], np.cumsum(signal)))
    csum2 = np.concatenate(([0.0], np.cumsum(signal ** 2)))
    window_sum = csum[L:] - csum[:-L]
    window_sumsq = csum2[L:] - csum2[:-L]
    window_ss = window_sumsq - (window_sum ** 2) / L
    window_ss = np.maximum(window_ss, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = dot / np.sqrt(window_ss * template_ss)

    corr[window_ss <= 1e-20] = np.nan
    return corr, dot

def estimate_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    return_sigma: bool = False,
):
    amplitude = np.ptp(signal)

    noise_center = np.nanmedian(noise)
    mad = np.nanmedian(np.abs(noise - noise_center))
    sigma_noise = 1.4826 * mad

    if sigma_noise <= 1e-20:
        return (np.nan, np.nan) if return_sigma else np.nan

    snr = float(amplitude / sigma_noise)

    return (sigma_noise, snr) if return_sigma else snr

def estimate_scale(snippet: np.ndarray, template: np.ndarray) -> float:
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if snippet_c.size != template_c.size:
        raise ValueError("Snippet and template must have the same length.")
    denom = float(np.dot(template_c, template_c))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(snippet_c, template_c) / denom)