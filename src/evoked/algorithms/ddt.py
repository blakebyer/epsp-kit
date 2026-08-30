from __future__ import annotations

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from typing import Literal, Optional
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.template import estimate_snr

def _longest_run_bounds(mask: np.ndarray) -> tuple[int, int] | None:
    """(start, stop) of the longest True run, stop exclusive. None if no run."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    if starts.size == 0:
        return None
    i = int(np.argmax(stops - starts))
    return int(starts[i]), int(stops[i])


def _max_sustained_run(values: np.ndarray, sigma: float, min_run: int,
                        step: float = 0.1) -> tuple[float, int | None, int | None]:
    if sigma <= 0 or min_run <= 0:
        return 0.0, None, None

    k_max = float(np.max(values) / sigma) if np.any(values > 0) else 0.0
    if k_max <= 0:
        return 0.0, None, None

    for k in np.arange(k_max, 0, -step):
        bounds = _longest_run_bounds(values > k * sigma)
        if bounds is not None and (bounds[1] - bounds[0]) >= min_run:
            return float(k), bounds[0], bounds[1]
    return 0.0, None, None


class DDT(BaseAlgorithm):
    method: Literal["ddt"] = "ddt"
    window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    noise_window: tuple[float, float]
    polarity: Literal["positive", "negative", "absolute", "both"] = "absolute"

    duration: Optional[float] = None
    positive_duration: Optional[float] = None
    negative_duration: Optional[float] = None

    def match(self, recording: RecordingData) -> AlgorithmResult:
        dt = 1 / recording.sampling_rate.rescale(pq.Hz).magnitude

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

        raw = recording.values(search)         
        deriv = np.gradient(recording.values(search), dt, axis=1)
        noise = np.gradient(recording.values(self.noise_window), dt, axis=1)
        time = recording.times(search)
           
        results = []
        trials = recording.trials.to_dicts()

        for i in range(deriv.shape[0]):
            for ch in range(deriv.shape[2]):
                trial = trials[i]
                x = deriv[i, :, ch]
                n = noise[i, :, ch]
                sigma, _ = estimate_snr(x, n, return_sigma=True)

                if self.polarity == "both":
                    n_pos = round(self.positive_duration / dt)
                    n_neg = round(self.negative_duration / dt)
                    k_pos, p_start, p_stop = _max_sustained_run(x, sigma, n_pos)
                    k_neg, n_start, n_stop = _max_sustained_run(-x, sigma, n_neg)
                    k = min(k_pos, k_neg)
                    bounds = (min(p_start, n_start), max(p_stop, n_stop)) if p_start is not None and n_start is not None else None
                else:
                    n_min = round(self.duration / dt)
                    if self.polarity == "positive":
                        k, start, stop = _max_sustained_run(x, sigma, n_min)
                    elif self.polarity == "negative":
                        k, start, stop = _max_sustained_run(-x, sigma, n_min)
                    else:  # absolute
                        k, start, stop = _max_sustained_run(np.abs(x), sigma, n_min)
                    bounds = (start, stop) if start is not None else None

                trace = raw[i, :, ch]
                if bounds is not None:
                    start, stop = bounds
                    peak_idx = start + int(np.argmax(np.abs(trace[start:stop])))
                else:
                    peak_idx = int(np.argmax(np.abs(trace)))  # nothing sustained k*sigma; k=0, will fail detect()

                results.append({
                    **trial, 
                    "channel": recording.channel_names[ch], 
                    "latency": float(time[peak_idx].rescale(pq.s).magnitude),
                    "amplitude": float(trace[peak_idx]),
                    "k": k,
                })

        return AlgorithmResult(
            algorithm=self,
            result=BaseResult.validate(pl.DataFrame(results)),
        )

    def detect(self, result: pl.DataFrame, threshold: float) -> DataFrame[BaseResult]:
        return BaseResult.validate(
                result.with_columns(
                    detected=pl.col("k") >= threshold
                )
            )

