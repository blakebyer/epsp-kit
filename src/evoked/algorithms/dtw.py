from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from pydantic import Field
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Literal
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.template import build_template_snr
import warnings


class DTW(BaseAlgorithm):
    method: Literal["dtw"] = "dtw"
    window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    noise_window: tuple[float, float] = (0.0,1e-3)
    snr_threshold: float = 10.0
    derivative_transform: bool = False
    paths: dict[
        tuple[str, str, int],
        list[tuple[int, int]]
    ] = Field(default_factory=dict, exclude=True)

    def match(self, recording: RecordingData) -> AlgorithmResult:
        template_arr = build_template_snr(
            recording=recording,
            window=self.window,
            noise_window=self.noise_window,
            snr_threshold=self.snr_threshold,
        )

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

        time = recording.times(search)
        value = recording.values(search)  # (n_trials, n_samples, n_channels)
        

        dt = 1 / recording.sampling_rate.rescale(pq.Hz).magnitude

        if self.derivative_transform:
            value = np.gradient(value, dt, axis=1)
            template_arr = np.gradient(template_arr, dt, axis=0)

        L = template_arr.shape[0]
        n_lags = value.shape[1] - L + 1
        left = L // 2

        if n_lags <= 0:
            raise ValueError(
                "Search window must contain at least as many samples as the template."
            )

        self.paths = {}
        results = []

        trials = recording.trials.to_dicts()

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                trial = trials[i]

                x = np.asarray(
                    value[i, :, ch],
                    dtype=np.double,
                )
                t = np.asarray(
                    template_arr[:, ch],
                    dtype=np.double,
                )

                distances = np.full(n_lags, np.nan)

                for k in range(n_lags):
                    candidate = x[k:k + L]

                    if not np.all(np.isfinite(candidate)):
                        continue

                    distances[k] = dtw.distance_fast(
                        t,
                        candidate,
                        use_pruning=True,
                    )

                if not np.any(np.isfinite(distances)):
                    warnings.warn(
                        f"DTW undefined for "
                        f"(file_origin={trial["file_origin"]}, channel={ch}, "
                        f"stimulus={trial["stimulus"]}). Skipping..."
                    )
                    continue

                best_k = int(np.nanargmin(distances))
                best_distance = float(distances[best_k])

                candidate = x[best_k:best_k + L]

                channel = recording.channel_names[ch]

                # Only calculate the warping path for the winning lag.
                self.paths[(trial["file_origin"], trial["stimulus"], channel)] = dtw.warping_path(
                    t,
                    candidate,
                )

                results.append({
                    **trial,
                    "channel": channel,
                    "latency": float(
                        time[best_k + left].rescale(pq.s).magnitude
                    ),
                    "dtw_distance": best_distance,
                })

        return AlgorithmResult(
            algorithm=self,
            template=template_arr,
            result=BaseResult.validate(
                pl.DataFrame(results)
            ),
        )

    def detect(self, result: pl.DataFrame, threshold: float) -> DataFrame[BaseResult]:
            return BaseResult.validate(
                result.with_columns(
                    detected=pl.col("dtw_distance") <= threshold
                )
            )