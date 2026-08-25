from __future__ import annotations

import numpy as np
from dtaidistance import dtw
from typing import Optional
from pydantic import model_validator, Field
import polars as pl
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
from evoked.algorithms.linear import estimate_snr
import warnings


class DTW(BaseAlgorithm):
    window: tuple[float, float]
    noise_window: tuple[float, float]
    search_window: tuple[float, float] | float = 0.25
    slope_transform: bool = False
    template_keys: Optional[dict[int, list[tuple[str, str]]]] = None
    snr_threshold: Optional[float] = 10.0
    template: Optional[np.ndarray] = None
    paths: dict[
        tuple[str, str, int],
        list[tuple[int, int]]
    ] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def check_template_source(self) -> DTW:
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

    def _fit(
        self,
        recording: RecordingData,
        template: np.ndarray,
    ) -> AlgorithmResult:

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

        value = recording.values(search)  # (n_trials, n_samples, n_channels)
        time = recording.times(search)

        if self.slope_transform:
            value = np.gradient(value, axis=1)

        files = recording.trials["file_origin"].to_list()
        stimuli = recording.trials["stimulus"].to_list()

        L = template.shape[0]
        n_lags = value.shape[1] - L + 1
        left = L // 2

        if n_lags <= 0:
            raise ValueError(
                "Search window must contain at least as many samples as the template."
            )

        self.paths = {}
        results = []

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):

                x = np.asarray(
                    value[i, :, ch],
                    dtype=np.double,
                )
                t = np.asarray(
                    template[:, ch],
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
                        f"(file_origin={files[i]}, channel={ch}, "
                        f"stimulus={stimuli[i]}). Skipping..."
                    )
                    continue

                best_k = int(np.nanargmin(distances))
                best_distance = float(distances[best_k])

                candidate = x[best_k:best_k + L]

                # Only calculate the warping path for the winning lag.
                self.paths[(files[i], stimuli[i], ch)] = dtw.warping_path(
                    t,
                    candidate,
                )

                trial = recording.trials.row(i, named=True)

                results.append({
                    "file_origin": files[i],
                    **trial,
                    "channel": ch,
                    "stimulus": stimuli[i],
                    "latency": float(
                        time[best_k + left].rescale(pq.s).magnitude
                    ),
                    "dtw_distance": best_distance,
                })

        return AlgorithmResult(
            algorithm=self,
            result=BaseResult.validate(
                pl.DataFrame(results)
            ),
        )