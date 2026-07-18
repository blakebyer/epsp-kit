from __future__ import annotations

import os
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
import pandera.polars as pa
import quantities as pq
import numpy as np
from pandera.typing.polars import Series, DataFrame

def window_to_indices(
    x: np.ndarray,
    window_s: tuple[float, float],
    fs: float,
) -> tuple[int, int]:
    """Convert a time window to a pair of sample indices with a length
    determined by the window width and the sampling rate `fs`.
    """
    t0, t1 = window_s
    if t1 <= t0:
        raise ValueError(f"window_s must satisfy t0 < t1, got {window_s}")

    start = int(np.searchsorted(x, t0))
    n_samples = int(round((t1 - t0) * fs))
    stop = start + n_samples

    if stop > x.size:
        raise ValueError(
            f"Window {window_s} requires {n_samples} samples starting at index "
            f"{start}, but this trace only has {x.size} samples total "
            f"({x.size - start} available past the start index)."
        )
    return start, stop

class EvokedBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

class RecordingData(pa.DataFrameModel):
    id: Series[str]
    channel: Series[int]
    sweep_index: Series[int]
    time: Series[float]
    value: Series[float]
    stimulus: Series[str]

OrderType = Literal["grouped", "interleaved", "explicit"]

class Experiment(EvokedBaseModel):
    name: Optional[str]
    description: Optional[str]

class Recording(EvokedBaseModel):
    id: Optional[str] = None
    stimulus: list[Any] = Field(default_factory=list)
    order: Optional[OrderType] = None
    repeats: Optional[int] = None
    stimulus_unit: Optional[pq.Quantity] = None

    def expand_stimulus(self) -> list[Any]:
        if self.order == "explicit":
            return list(self.stimulus)
        elif self.order == "grouped":
            return [s for s in self.stimulus for _ in range(self.repeats)]
        elif self.order == "interleaved":
            return list(self.stimulus) * self.repeats
        else:
            raise ValueError(f"Unknown order: {self.order}. Must be one of: explicit, grouped, or interleaved")
        
    @field_validator("stimulus_unit", mode="before")
    @classmethod
    def parse_unit(cls, v):
        if isinstance(v, str):
            return getattr(pq, v, pq.dimensionless)   # "uA" -> pq.uA, "kazoo" -> pq.dimensionless
        return v
    
class Metadata(EvokedBaseModel):
    default: dict[str, Any]
    recordings: dict[str, Recording]
    
    @model_validator(mode="before")
    @classmethod
    def normalize_recordings(cls, data: Any) -> Any:
        """
        Allow `recordings` to be provided either as:
 
          - dict form: a mapping of filename -> per-recording overrides
                recordings:
                    file1.abf:
                        id: slice_1
 
          - list form: a bare list of filenames that should rely
            entirely on the `default` block, with `id` derived from
            each filename's basename stem
                recordings:
                    - file1.abf
                    - file2.abf
 
        List form is normalized into dict form here (each filename
        mapped to an empty override dict) so that the rest of the
        model -- including `apply_defaults` below, which fills in `id`
        and pulls remaining fields from `default` -- never has to know
        which form the user provided.
        """
        if isinstance(data, dict):
            recordings = data.get("recordings")
            if isinstance(recordings, list):
                data["recordings"] = {
                    filename: {} for filename in recordings
                }
        return data
 
    @model_validator(mode="after")
    def apply_defaults(self) -> Metadata:
        for filename, recording in self.recordings.items():
 
            if recording.id is None:
                recording.id = os.path.splitext(os.path.basename(filename))[0]
 
            for key, default_value in self.default.items():
                if hasattr(recording, key):
                    field_was_missing = key not in recording.model_fields_set
                    field_is_none = getattr(recording, key) is None
 
                    if field_was_missing or field_is_none:
                        setattr(recording, key, default_value)
 
            if recording.order is None:
                raise ValueError(
                    f"File '{filename}' is missing required field 'order'. "
                    "Specify it under the file entry or in the global 'default' block."
                )
 
            if recording.order != "explicit" and recording.repeats is None:
                raise ValueError(
                    f"File '{filename}' is missing required field 'repeats'. "
                    "Specify it under the file entry or in the global 'default' block."
                )
            if not recording.stimulus:
                raise ValueError(
                    f"File '{filename}' has no stimulus values. "
                    "Specify 'stimulus' under the file entry or in the global 'default' block."
                )
 
        return self
    
class Analysis(EvokedBaseModel):
    epoch: Optional[tuple[float, float]]
    features: dict[str, Any] = Field(default_factory=dict)

class RecordingConfig(EvokedBaseModel):
    experiment: Experiment
    metadata: Metadata
    analysis: Optional[Analysis] = None

class IntermediateResult(pa.DataFrameModel):
    id: Series[str]
    channel: Series[int]
    time: Series[float]
    value: Series[float]
    stimulus: Series[str]

class FitResult(pa.DataFrameModel):
    id: Series[str] 
    channel: Series[int]
    stimulus: Series[str]
    feature_time: Series[float] 
    scale: Series[float]
    amplitude: Series[float]
    corr: Optional[Series[float]]
    r2: Optional[Series[float]]
    t_stat: Optional[Series[float]]
    p_value: Optional[Series[float]]
    detected: Series[bool]

class FeatureResult(EvokedBaseModel):
    window: tuple[float, float]
    slope_transform: bool
    snr_threshold: float
    p_value_threshold: float
    template: np.ndarray 
    template_keys: list[tuple]
    result: DataFrame[FitResult]
    
class RecordingResult(EvokedBaseModel):
    results: dict[str, FeatureResult] = Field(default_factory=dict)
    def add(self, result_key: str, feature_result: FeatureResult) -> None:
        self.results[result_key] = feature_result

