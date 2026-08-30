from __future__ import annotations

import neo
import numpy as np
from typing import Optional, Callable
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer, SerializeAsAny, TypeAdapter
from dataclasses import dataclass
from abc import ABC, abstractmethod
import polars as pl
import pandera.polars as pa
import quantities as pq
from pandera.typing.polars import Series, DataFrame



class Trials(pa.DataFrameModel):
    trial_index: Series[int] 
    file_origin: Series[str]
    stimulus: Series[str] = pa.Field(coerce=True)

    class Config: 
        strict = False # as many groups as the user wants

    @pa.dataframe_check
    def trial_index_is_unique(cls, data) -> bool:
        df = data.lazyframe.collect()
        return df["trial_index"].n_unique() == df.height

class TrialCountMismatch(ValueError):
    """Raised when a file's segment count doesn't match its expanded stimulus list."""

@dataclass
class RecordingData:
    segments: list[neo.Segment]
    trials: pl.DataFrame

    def values(
        self,
        window: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:

        signals = [seg.analogsignals[0] for seg in self.segments]

        if window is not None:
            t0, t1 = window
            signals = [
                sig.time_slice(t0 * pq.s, t1 * pq.s)
                for sig in signals
            ]

        return np.stack([sig.magnitude for sig in signals])

    def times(
        self,
        window: Optional[tuple[float, float]] = None,
    ) -> pq.Quantity:
        signal = self.segments[0].analogsignals[0]

        if window is not None:
            signal = signal.time_slice(
                window[0] * pq.s,
                window[1] * pq.s,
            )

        return signal.times

    @staticmethod
    def _clone_segment(source: neo.Segment, signal) -> neo.Segment:
        signal.array_annotations = {
            key: np.asarray(value).copy()
            for key, value in source.analogsignals[0].array_annotations.items()
        }

        new_seg = neo.Segment(**source.annotations)
        new_seg.analogsignals.append(signal)
        new_seg.events.extend(source.events)
        return new_seg

    def map_values(self, fn: Callable[[np.ndarray], np.ndarray]) -> RecordingData:
        values = fn(self.values())
        segments = [
            self._clone_segment(seg, seg.analogsignals[0].duplicate_with_new_data(values[i]))
            for i, seg in enumerate(self.segments)
        ]
        return RecordingData(segments=segments, trials=self.trials)

    def select_trials(
        self,
        predicate: Optional[pl.Expr] = None,
        **filters,
    ) -> RecordingData:
        selected = self.trials.with_row_index("__row")

        if predicate is not None:
            selected = selected.filter(predicate)

        for column, value in filters.items():
            if column not in self.trials.columns:
                raise KeyError(f"Unknown trial field: {column}")

            if isinstance(value, (list, tuple, set)):
                selected = selected.filter(
                    pl.col(column).is_in(value)
                )
            else:
                selected = selected.filter(
                    pl.col(column) == value
                )

        positions = selected["__row"].to_list()

        segments = [
            self.segments[i]
            for i in positions
        ]

        selected = selected.drop("__row")

        return RecordingData(
            segments=segments,
            trials=Trials.validate(selected),
        )

    def select_channels(self, channels: list[str]) -> RecordingData:
        idx = [self.channel_index(channel) for channel in channels]

        segments = []
        for seg in self.segments:
            sig = seg.analogsignals[0]
            signal = sig.duplicate_with_new_data(np.asarray(sig.magnitude)[:, idx])
            signal.array_annotations = {
                key: np.asarray(value)[idx]
                for key, value in sig.array_annotations.items()
            }
            segments.append(self._clone_segment(seg, signal))

        return RecordingData(segments=segments, trials=self.trials)

    def channel_index(self, channel: str) -> int:
        if channel in self.channel_names:
            return self.channel_names.index(channel)

        raise KeyError(f"Unknown channel: {channel!r}")

    @classmethod
    def concat(cls, items: list[RecordingData]) -> RecordingData:
        segments = [seg for item in items for seg in item.segments]
        trials = pl.concat([item.trials for item in items], how="vertical").with_columns(
            trial_index=pl.int_range(pl.len())
        )
        return cls(segments=segments, trials=Trials.validate(trials))

    @property
    def sampling_rate(self) -> pq.Quantity:
        return self.segments[0].analogsignals[0].sampling_rate

    @property
    def channel_names(self) -> list[str]:
        return self.segments[0].analogsignals[0].array_annotations["channel_name"].tolist()

    @property
    def n_trials(self) -> int:
        return self.shape[0]

    @property
    def n_samples(self) -> int:
        return self.shape[1]

    @property
    def n_channels(self) -> int:
        return self.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values().shape # (n_trials, n_samples, n_channels)

    @property
    def duration(self) -> pq.Quantity:
        signal = self.segments[0].analogsignals[0]
        return signal.t_stop - signal.t_start

    @property
    def value_unit(self) -> pq.Quantity:
        return self.segments[0].analogsignals[0].dimensionality

    @property
    def time_unit(self) -> pq.Quantity:
        return self.segments[0].analogsignals[0].times.dimensionality
    

class BaseResult(pa.DataFrameModel):
    file_origin: Series[str]
    channel: Series[str] = pa.Field(coerce=True)
    stimulus: Series[str] = pa.Field(coerce=True)

    class Config: 
        strict = False

class TruthData(BaseResult):
    feature: Series[str]
    detected: Series[bool]

class BaseAlgorithm(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: str

    @abstractmethod
    def match(self, recording: RecordingData, results: Optional[RecordingResult] = None) -> AlgorithmResult:
        ...

    @abstractmethod
    def detect(self, result: pl.DataFrame, threshold: float) -> DataFrame[BaseResult]:
        ...

class AlgorithmResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    algorithm: SerializeAsAny[BaseAlgorithm]
    template: Optional[np.ndarray] = None
    result: DataFrame[BaseResult]

    def detect(self, threshold: float) -> AlgorithmResult:
        return AlgorithmResult(
            algorithm=self.algorithm,
            result=self.algorithm.detect(self.result, threshold),
        )

    @field_validator("algorithm", mode="before")
    @classmethod
    def deserialize_algorithm(cls, value):
        if isinstance(value, dict):
            from evoked.algorithms.registry import AlgorithmType

            adapter = TypeAdapter(AlgorithmType)
            return adapter.validate_python(value)

        return value

    @field_serializer("result")
    def serialize_result(self, value: pl.DataFrame):
        return value.to_dicts()

    @field_validator("result", mode="before")
    @classmethod
    def deserialize_result(cls, value):
        return pl.DataFrame(value) if isinstance(value, list) else value

    @field_validator("template", mode="before")
    @classmethod
    def _to_array(cls, value):
        return np.asarray(value) if not isinstance(value, np.ndarray) else value
    
    @field_serializer("template")
    def _serialize_template(self, value: np.ndarray) -> list:
        return value.tolist()

RecordingResult = dict[str, AlgorithmResult]