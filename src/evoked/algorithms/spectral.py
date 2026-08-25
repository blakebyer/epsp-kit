from __future__ import annotations

import numpy as np
from scipy.signal import convolve
import polars as pl
import quantities as pq
from evoked.base import RecordingData, BaseAlgorithm, BaseResult, AlgorithmResult
import warnings

class RMS(BaseAlgorithm):
    search_window: tuple[float, float]
    noise_window: tuple[float, float]
    window_length: int = 11
    slope_transform: bool = False
    rms_threshold: float = 3.0

    def match(self, recording: RecordingData) -> AlgorithmResult:
        value = recording.values(self.search_window)  # (n_trials, n_samples, n_channels)
        noise = recording.values(self.noise_window)

        if self.slope_transform:

            value = np.gradient(value, axis=1)
            noise = np.gradient(noise, axis=1)

        time = recording.times(self.search_window)

        files = recording.trials["file_origin"].to_list()
        stimuli = recording.trials["stimulus"].to_list()

        kernel = np.ones(self.window_length) / self.window_length

        results = []

        for i in range(value.shape[0]):
            for ch in range(value.shape[2]):
                x = value[i, :, ch]
                n = noise[i, :, ch]
                rms = np.sqrt(
                    convolve(x**2, kernel, mode="same")
                )
                noise_rms = np.sqrt(np.nanmean(n**2))

                if not np.any(np.isfinite(rms)):
                    warnings.warn(
                        f"RMS undefined for "
                        f"(file_origin={files[i]}, channel={ch}, stimulus={stimuli[i]}). "
                        "Skipping..."
                    )
                    continue

                best_k = int(np.nanargmax(rms))
                peak_rms = float(rms[best_k])

                trial = recording.trials.row(i, named=True)

                results.append({
                    "file_origin": files[i],
                    **trial,
                    "channel": ch,
                    "stimulus": stimuli[i],
                    "latency": float(
                        time[best_k].rescale(pq.s).magnitude
                    ),
                    "rms": peak_rms,
                    "snr": peak_rms / noise_rms,
                    "detected": bool(peak_rms >= self.rms_threshold * noise_rms),
                })

        return AlgorithmResult(
            algorithm=self,
            result=BaseResult.validate(pl.DataFrame(results)),
        )