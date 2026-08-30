from __future__ import annotations

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Literal
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.template import estimate_snr

class Peak(BaseAlgorithm):
    method: Literal["peak"] = "peak"
    window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    noise_window: tuple[float, float]
    polarity: Literal["positive", "negative", "absolute"] = "absolute"
    derivative_transform: bool = False

    def match(self, recording: RecordingData) -> AlgorithmResult:
        search = (
            self.search_window
            if isinstance(self.search_window, tuple)
            else (
                self.window[0]
                - (self.window[1] - self.window[0]) * self.search_window,
                self.window[1]
                + (self.window[1] - self.window[0]) * self.search_window,
            )
                        )
        value = recording.values(search)
        noise = recording.values(self.noise_window)
        time = recording.times(search)

        dt = 1 / recording.sampling_rate.rescale(pq.Hz).magnitude

        if self.derivative_transform:
            value = np.gradient(value, dt, axis=1)
            noise = np.gradient(noise, dt, axis=1)

        results = []
        trials = recording.trials.to_dicts()

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                trial = trials[i]
                x = value[i, :, ch]
                n = noise[i, :, ch]

                if self.polarity == "positive":
                    k = int(np.argmax(x))
                    amplitude = float(x[k])
                elif self.polarity == "negative":
                    k = int(np.argmin(x))
                    amplitude = float(x[k])
                else:
                    k = int(np.argmax(np.abs(x)))
                    amplitude = float(x[k])

                sigma, _ = estimate_snr(x, n, return_sigma=True)

                results.append({
                    **trial,
                    "channel": recording.channel_names[ch],
                    "latency": float(time[k].rescale(pq.s).magnitude),
                    "amplitude": amplitude,
                    "snr": abs(amplitude) / sigma,
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

