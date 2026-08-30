from __future__ import annotations

import numpy as np
from scipy.signal import correlate
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Literal
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.template import build_template_snr, center_signal, window_correlation
import warnings

class MatchedFilter(BaseAlgorithm):
    method: Literal["matched_filter"] = "matched_filter"
    window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    noise_window: tuple[float, float] = (0.0,1e-3)
    snr_threshold: float = 10.0
    derivative_transform: bool = False

    def match(self, recording: RecordingData) -> AlgorithmResult:
        template_arr = build_template_snr(
            recording=recording,
            window=self.window,
            snr_threshold=self.snr_threshold,
            noise_window=self.noise_window,
        )

        search = self.search_window if isinstance(self.search_window, tuple) else (
            self.window[0] - (self.window[1] - self.window[0]) * self.search_window,
            self.window[1] + (self.window[1] - self.window[0]) * self.search_window,
        )

        time = recording.times(search)
        value = recording.values(search) # (n_trials, n_samples, n_channels)

        dt = 1 / recording.sampling_rate.rescale(pq.Hz).magnitude

        if self.derivative_transform:
            value = np.gradient(value, dt, axis=1)
            template_arr = np.gradient(template_arr, dt, axis=0)        

        L = template_arr.shape[0]
        left = L // 2

        results = []
        trials = recording.trials.to_dicts()
        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                trial = trials[i]
                corr, dot = window_correlation(value[i, :, ch], template_arr[:, ch])
                if not np.any(np.isfinite(corr)):
                    warnings.warn(f"Correlation undefined for (file_origin={trial["file_origin"]}, channel={ch}, stimulus={trial["stimulus"]}). Skipping...")
                    continue
                best_k = int(np.nanargmax(corr))
                template_ss = np.dot(center_signal(template_arr[:, ch]), center_signal(template_arr[:, ch]))
                best_corr = float(corr[best_k])
                best_r2 = best_corr ** 2

                results.append({
                    **trial, 
                    "channel": recording.channel_names[ch],
                    "latency": float(time[best_k + left].rescale(pq.s).magnitude),
                    "amplitude": abs(float(dot[best_k] / template_ss) * np.ptp(template_arr[:, ch])),
                    "corr": best_corr, "r2": best_r2,
                })

        return AlgorithmResult(algorithm=self, template=template_arr, result=BaseResult.validate(pl.DataFrame(results)))

    def detect(self, result: pl.DataFrame, threshold: float) -> DataFrame[BaseResult]:
        return BaseResult.validate(
            result.with_columns(
                detected=pl.col("r2") >= threshold
            )
        )