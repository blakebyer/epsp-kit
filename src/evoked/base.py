from __future__ import annotations

import neo
import numpy as np
from typing import Any, Optional, Union, Callable
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer
from dataclasses import dataclass
from abc import ABC, abstractmethod
import polars as pl
import pandera.polars as pa
import quantities as pq
from pandera.typing.polars import Series


ChannelTypes = Union[int, str]
StimulusTypes = Union[str, int, float]
Selector = Union[list, str, tuple]

class Trials(pa.DataFrameModel):
    trial_index: Series[int]
    file_origin: Series[str]
    stimulus: Series[StimulusTypes] = pa.Field(nullable=True, coerce=True)

    class Config: 
        strict = False # as many groups as the user wants
        dtype = "object"

    @pa.dataframe_check
    def trial_index_is_unique(cls, df: pl.DataFrame) -> bool:
        return df["trial_index"].n_unique() == df.height

class TrialCountMismatch(ValueError):
    """Raised when a file's segment count doesn't match its expanded stimulus list."""

@dataclass
class RecordingData:
    segments: list[neo.Segment]
    trials: Trials

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

    def average_by(self, by: str | list[str]) -> RecordingData:
        by = [by] if isinstance(by, str) else by
        values = self.values()

        trials = self.trials.with_row_index("__row")

        segments = []
        rows = []

        for key, group in trials.group_by(by, maintain_order=True):
            positions = group["__row"].to_numpy()
            mean = values[positions].mean(axis=0)

            first = self.segments[positions[0]]
            signal = first.analogsignals[0].duplicate_with_new_data(mean)
            segments.append(self._clone_segment(first, signal))

            key = key if isinstance(key, tuple) else (key,)
            rows.append({
                "trial_index": len(rows),
                **dict(zip(by, key)),
            })

        return RecordingData(
            segments=segments,
            trials=Trials.validate(pl.DataFrame(rows)),
        )

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

    def select_channels(self, channels: list[int] | list[str]) -> RecordingData:
        names = self.channel_names
        idx = (
            [names.index(c) for c in channels]
            if channels and isinstance(channels[0], str)
            else list(channels)
        )

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
        signal = self.segments[0].analogsignals[0]
        names = signal.array_annotations.get("channel_name")
        return names.tolist() if names is not None else []

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
    channel: Series[ChannelTypes]
    stimulus: Series[StimulusTypes]

    class Config: 
        strict = False
        dtype = "object"

class TruthData(BaseResult):
    feature: Series[str]
    detected: Series[bool]

class BaseAlgorithm(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def match(self, recording: RecordingData) -> AlgorithmResult:
        ...

    @abstractmethod
    def detect(self, result: pl.DataFrame, threshold: float) -> pl.DataFrame:
        ...

class AlgorithmResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    algorithm: BaseAlgorithm
    result: pl.DataFrame

    @field_serializer("algorithm")
    def serialize_algorithm(
        self,
        algorithm: BaseAlgorithm,
    ) -> dict[str, Any]:
        data = algorithm.model_dump()

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()

        return {
            "method": type(algorithm).__name__,
            **data,
        }

    @field_validator("algorithm", mode="before")
    @classmethod
    def deserialize_algorithm(
        cls,
        value,
    ):
        if isinstance(value, dict):
            from evoked.algorithms.registry import parse_algorithm

            return parse_algorithm(value)

        return value

    @field_serializer("result")
    def serialize_result(self, value: pl.DataFrame):
        return value.to_dicts()

    @field_validator("result", mode="before")
    @classmethod
    def deserialize_result(cls, value):
        return pl.DataFrame(value) if isinstance(value, list) else value

RecordingResult = dict[str, AlgorithmResult]