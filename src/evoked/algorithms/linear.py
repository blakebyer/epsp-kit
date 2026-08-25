from __future__ import annotations

import numpy as np
from scipy.signal import correlate
from typing import Literal, Optional
from pydantic import model_validator
import polars as pl
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
import warnings

class Peak(BaseAlgorithm):
    search_window: tuple[float, float]
    polarity: Literal["positive", "negative", "absolute"] = "absolute"
    threshold: Optional[float] = None

    def match(self, recording: RecordingData) -> AlgorithmResult:
        value = recording.values(self.search_window)
        time = recording.times(self.search_window)

        files = recording.trials["file_origin"].to_list()
        stimuli = recording.trials["stimulus"].to_list()

        results = []

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                x = value[i, :, ch]

                if self.polarity == "positive":
                    k = int(np.nanargmax(x))
                    amplitude = float(x[k])
                elif self.polarity == "negative":
                    k = int(np.nanargmin(x))
                    amplitude = float(x[k])
                else:
                    k = int(np.nanargmax(np.abs(x)))
                    amplitude = float(x[k])

                trial = recording.trials.row(i, named=True)

                results.append({
                    "file_origin": files[i], **trial, "channel": ch, "stimulus": stimuli[i],
                    "latency": float(time[k].rescale(pq.s).magnitude),
                    "amplitude": amplitude,
                    "detected": bool(abs(amplitude) >= self.threshold),
                })

        return AlgorithmResult(
            algorithm=self,
            result=BaseResult.validate(pl.DataFrame(results)),
        )

class MatchedFilter(BaseAlgorithm):
    window: tuple[float, float]
    noise_window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    slope_transform: bool = False
    template_keys: Optional[dict[int, list[tuple[str, str]]]] = None
    snr_threshold: Optional[float] = 10.0
    template: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def check_template_source(self) -> MatchedFilter:
        if self.template_keys is not None:
            return self

        if self.snr_threshold is None:
            raise ValueError(
                "Specify 'template_keys' or a non-None 'snr_threshold'."
            )

        return self

    def match(self, recording: RecordingData) -> AlgorithmResult:
        if self.template_keys is not None:
            template = self._template_from_keys(recording)
        else:
            template = self._template_from_snr(recording)

        self.template = template
        return self._fit(recording, template)

    def _template_from_keys(self, recording: RecordingData) -> np.ndarray:
        value = recording.values(self.window)

        if self.slope_transform:
            value = np.gradient(value, axis=1)

        template = np.empty(
            (value.shape[1], value.shape[2]),
            dtype=value.dtype,
        )

        for ch, keys in self.template_keys.items():
            key_set = set(keys)

            indices = [
                i
                for i, trial in enumerate(recording.trials.iter_rows(named=True))
                if (trial["file_origin"], trial["stimulus"]) in key_set
            ]

            template[:, ch] = value[indices, :, ch].mean(axis=0)

        return template # (n_samples, n_channels)

    def _template_from_snr(self, recording: RecordingData) -> np.ndarray:
        value = recording.values(self.window)                      # (n_trials, n_samples, n_channels)
        noise = recording.values(self.noise_window)

        if self.slope_transform:
            value = np.gradient(value, axis=1)
            noise = np.gradient(noise, axis=1)

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
        keep = snr >= self.snr_threshold
        if not keep.any():
            raise ValueError(f"No trials with SNR>={self.snr_threshold}. Max SNR={np.nanmax(snr)}.")

        self.template_keys = {
            ch: [
                (trial["file_origin"], trial["stimulus"])
                for i, trial in enumerate(recording.trials.iter_rows(named=True))
                if keep[i, ch]
            ]
            for ch in range(value.shape[2])
        }

        template = np.nanmean(
            np.where(
                keep[:, None, :],
                value,
                np.nan,
            ),
            axis=0,
        )
        return template

    def _fit(self, recording: RecordingData, template: np.ndarray) -> AlgorithmResult:
        search = self.search_window if isinstance(self.search_window, tuple) else (
            self.window[0] - (self.window[1] - self.window[0]) * self.search_window,
            self.window[1] + (self.window[1] - self.window[0]) * self.search_window,
        )
        value = recording.values(search)                           # (n_trials, n_samples, n_channels)
        value = np.gradient(value, axis=1) if self.slope_transform else value
        time = recording.times(search)

        files = recording.trials["file_origin"].to_list()
        stimuli = recording.trials["stimulus"].to_list()

        
        L = template.shape[0]
        left = L // 2

        results = []
        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                corr, dot = window_correlation(value[i, :, ch], template[:, ch])
                if not np.any(np.isfinite(corr)):
                    warnings.warn(f"Correlation undefined for (file_origin={files[i]}, channel={ch}, stimulus={stimuli[i]}). Skipping...")
                    continue
                best_k = int(np.nanargmax(corr))
                template_ss = np.dot(center_signal(template[:, ch]), center_signal(template[:, ch]))
                best_corr = float(corr[best_k])
                best_r2 = best_corr ** 2
                trial = recording.trials.row(i, named=True)
                results.append({
                    "file_origin": files[i], **trial, "channel": ch, "stimulus": stimuli[i],
                    "latency": float(time[best_k + left].rescale(pq.s).magnitude),
                    "amplitude": abs(float(dot[best_k] / template_ss) * np.ptp(template[:, ch])),
                    "corr": best_corr, "r2": best_r2,
                })

        return AlgorithmResult(algorithm=self, result=BaseResult.validate(pl.DataFrame(results)))

def center_signal(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    signal_arr = np.asarray(signal, dtype=float)
    return signal_arr - np.mean(signal_arr, axis=axis, keepdims=True)

def window_correlation(
    signal: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized correlation

    Args:
        signal (np.ndarray): _description_
        template (np.ndarray): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
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

    noise_center = np.median(noise)
    mad = np.median(np.abs(noise - noise_center))
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