from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from typing import Optional
from pydantic import model_validator
import polars as pl
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.algorithms.linear import center_signal, estimate_snr, window_correlation
import warnings

class GLRT(BaseAlgorithm):
    window: tuple[float, float]
    noise_window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    slope_transform: bool = False
    template_keys: Optional[dict[int, list[tuple[str, str]]]] = None
    snr_threshold: Optional[float] = 10.0
    alpha: float = 0.05
    template: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def check_template_source(self) -> GLRT:
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
        keys = set(self.template_keys)

        selected = recording.select_trials(
            predicate=pl.struct(["file_origin", "stimulus"]).is_in(keys)
        )

        if selected.n_trials == 0:
            raise ValueError(
                "No trials matched template_keys."
            )

        value = selected.values(self.window)

        if self.slope_transform:
            value = np.gradient(value, axis=1)

        return value.mean(axis=0) # (n_samples, n_channels)

    def _template_from_snr(self, recording: RecordingData) -> np.ndarray:
        value = recording.values(self.window)                      # (n_trials, n_samples, n_channels)
        noise = recording.values(self.noise_window)

        if self.slope_transform:
            value = np.gradient(value, axis=1)
            noise = np.gradient(noise, axis=1)

        sigma = np.empty((value.shape[0], value.shape[2]))
        snr = np.empty_like(sigma)

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                sigma[i, ch], snr[i, ch] = estimate_snr(
                    value[i, :, ch],
                    noise[i, :, ch],
                    return_sigma=True,
                )

        keep = snr >= self.snr_threshold
        if not keep.any():
            raise ValueError(f"No trials with SNR>={self.snr_threshold}. Max SNR={np.nanmax(snr)}.")

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
        noise = recording.values(self.noise_window)

        if self.slope_transform:
            value = np.gradient(value, axis=1)
            noise = np.gradient(noise, axis=1)
        time = recording.times(search)

        files = recording.trials["file_origin"].to_list()
        stimuli = recording.trials["stimulus"].to_list()

        L = template.shape[0]
        n_lags = value.shape[1] - L + 1
        left = L // 2

        results = []
        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                sigma, _ = estimate_snr(
                    value[i, :, ch],
                    noise[i, :, ch],
                    return_sigma=True,
                )
                if not np.isfinite(sigma) or sigma <= 1e-20:
                    continue
                corr, dot = window_correlation(value[i, :, ch], template[:, ch])
                if not np.any(np.isfinite(corr)):
                    warnings.warn(f"Correlation undefined for (file_origin={files[i]}, channel={ch}, stimulus={stimuli[i]}). Skipping...")
                    continue
                
                template_ss = np.dot(center_signal(template[:, ch]), center_signal(template[:, ch]))
                stat = dot**2 / (sigma**2 * template_ss)

                best_k = int(np.nanargmax(stat))

                best_stat = float(stat[best_k])
                best_corr = float(corr[best_k])
                best_r2 = best_corr**2

                p_local = float(chi2.sf(best_stat, df=1))
                p_value = min(1.0, n_lags * p_local) # bonferroni correction

                trial = recording.trials.row(i, named=True)
                results.append({
                    "file_origin": files[i], **trial, "channel": ch, "stimulus": stimuli[i],
                    "latency": float(time[best_k + left].rescale(pq.s).magnitude),
                    "amplitude": abs(float(dot[best_k] / template_ss) * np.ptp(template[:, ch])),
                    "corr": best_corr, "r2": best_r2,
                    "stat": best_stat,
                    "p_value": p_value,
                    "detected": bool(p_value <= self.alpha),
                })

        return AlgorithmResult(algorithm=self, result=BaseResult.validate(pl.DataFrame(results)))