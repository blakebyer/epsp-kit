from __future__ import annotations

import numpy as np
from scipy.signal import convolve
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Literal
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
import warnings

class RMS(BaseAlgorithm):
    method: Literal["rms"] = "rms"
    search_window: tuple[float, float]
    noise_window: tuple[float, float]
    window_length: int = 7
    derivative_transform: bool = False

    def match(self, recording: RecordingData) -> AlgorithmResult:
        value = recording.values(self.search_window)  # (n_trials, n_samples, n_channels)
        noise = recording.values(self.noise_window)
        time = recording.times().rescale(pq.s).magnitude

        dt = 1 / recording.sampling_rate.rescale(pq.Hz).magnitude

        if self.derivative_transform:
            value = np.gradient(value, dt, axis=1)
            noise = np.gradient(noise, dt, axis=1)

        time = recording.times(self.search_window)

        kernel = np.ones(self.window_length) / self.window_length

        results = []

        trials = recording.trials.to_dicts()

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                trial = trials[i]
                x = value[i, :, ch]
                n = noise[i, :, ch]
                rms = np.sqrt(
                    convolve(x**2, kernel, mode="same")
                )
                noise_rms = np.sqrt(np.nanmean(n**2))

                if not np.any(np.isfinite(rms)):
                    warnings.warn(
                        f"RMS undefined for "
                        f"(file_origin={trial["file_origin"]}, channel={ch}, stimulus={trial["stimulus"]}). "
                        "Skipping..."
                    )
                    continue

                best_k = int(np.nanargmax(rms))
                peak_rms = float(rms[best_k])

                print("trial:", trial)

                results.append({
                    **trial,
                    "channel": recording.channel_names[ch],
                    "latency": float(
                        time[best_k].rescale(pq.s).magnitude
                    ),
                    "rms": peak_rms,
                    "snr": peak_rms / noise_rms,
                })

        return AlgorithmResult(
            algorithm=self,
            result=BaseResult.validate(pl.DataFrame(results)),
        )

    def detect(self, result: pl.DataFrame, threshold: float) -> DataFrame[BaseResult]:
        return BaseResult.validate(
            result.with_columns(
                detected=pl.col("snr") >= threshold
            )
        )