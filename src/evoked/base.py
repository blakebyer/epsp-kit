from __future__ import annotations

import os
from typing import Any, Literal, Optional, Annotated
from pydantic import BaseModel, ConfigDict, WithJsonSchema, Field, model_validator, field_validator, field_serializer
import polars as pl
import pandera.polars as pa
import quantities as pq
import numpy as np
from pandera.typing.polars import Series, DataFrame

def col_to_2d(df: pl.DataFrame, col: str) -> np.ndarray:
    """List(Float32) column -> real 2D float32 array, no per-row boxing."""
    lengths = df.get_column(col).list.len()
    if lengths.n_unique() != 1:
        raise ValueError(f"'{col}' has ragged lengths; batch isn't vectorizable.")
    return df.get_column(col).explode().to_numpy().reshape(df.height, int(lengths[0]))

def col_from_2d(arr: np.ndarray, name: str) -> pl.Series:
    """2D float32 array -> List(Float32) column, no .tolist()."""
    return pl.Series(name, arr, dtype=pl.List(pl.Float32))

class RecordingData(pa.DataFrameModel):
    id: Series[str]
    channel: Series[pl.Int32]
    sweep_index: Series[pl.Int32]
    time: Series[Annotated[pl.List, pl.Float32()]]
    value: Series[Annotated[pl.List, pl.Float32()]]
    stimulus: Series[str]

OrderType = Literal["grouped", "interleaved", "explicit"]
LayoutType = Literal["segments", "continuous"]

class Experiment(BaseModel):
    name: str
    description: Optional[str] = None

QuantityConfig = Annotated[
    pq.Quantity,
    WithJsonSchema({"type": "string"}),
]

class Recording(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    block_index: Optional[int] = None
    id: Optional[str] = None
    stimulus: list[str] = Field(default_factory=list)
    order: Optional[OrderType] = None
    repeats: Optional[int] = None
    event_label: str | None = None
    stimulus_unit: Optional[QuantityConfig] = None
    layout: Optional[LayoutType] = None

    def expand_stimulus(self) -> list[str]:
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
    @field_validator("stimulus", mode="before")
    @classmethod
    def coerce_stimulus(cls, s):
        if any(isinstance(i, (int, float)) for i in s):
            return list(map(str, s))
        return s
    
class Metadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    default: Optional[dict[str, Any]] = None
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
 
            for key, default_value in (self.default or {}).items():
                if hasattr(recording, key):
                    field_was_missing = key not in recording.model_fields_set
                    field_is_none = getattr(recording, key) is None
 
                    if field_was_missing or field_is_none:
                        setattr(recording, key, default_value)
            if recording.block_index is None:
                recording.block_index = 0

            if recording.layout == "segments":
                if not recording.stimulus:
                    raise ValueError(
                        f"File '{filename}' has no stimulus values."
                    )

                if recording.order is None:
                    raise ValueError(
                        f"File '{filename}' is missing required field 'order'."
                    )

                if recording.order != "explicit" and recording.repeats is None:
                    raise ValueError(
                        f"File '{filename}' is missing required field 'repeats'."
                    )

            # if recording.stimulus_unit is None:
            #             raise ValueError(
            #                 f"File '{filename}' is missing required field 'stimulus_unit'. "
            #                 "Specify it under the file entry or in the global 'default' block."
            #             )
 
        return self
    
class Feature(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    window: Optional[tuple[float, float]] = None
    noise_window: Optional[tuple[float, float]] = None
    search_window: Optional[tuple[float, float] | float] = None
    slope_transform: Optional[bool] = None
    snr_threshold: Optional[float] = None
    r2_threshold: Optional[float] = None
    
class Analysis(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    default: Optional[dict[str, Any]] = None
    epoch: Optional[tuple[float, float]] = None
    target_frequency: Optional[float] = None
    preprocess: Optional[dict] = None
    features: dict[str, Feature] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_defaults(self) -> Analysis:
        for feature, settings in self.features.items():

            for key, default_value in (self.default or {}).items():
                if hasattr(settings, key):
                    field_was_missing = key not in settings.model_fields_set
                    field_is_none = getattr(settings, key) is None
 
                    if field_was_missing or field_is_none:
                        setattr(settings, key, default_value)
            if settings.window is None:
                raise ValueError(
                    f"Feature '{feature}' is missing required field 'window'. "
                    "Specify it under the feature entry or in the global 'default' block."
                )
            if settings.noise_window is None:
                raise ValueError(
                    f"Feature '{feature}' is missing required field 'noise_window'. "
                    "Specify it under the feature entry or in the global 'default' block."
                )
        return self

class TruthData(pa.DataFrameModel):
    id: Series[str]
    channel: Series[int]
    stimulus: Series[str] = pa.Field(coerce=True)
    feature: Series[str]
    detected: Series[bool]

class TracePlot(BaseModel):
    id: str
    channel: int
    features: Optional[list[str]] = None
    stimuli: list[str]
    annotated: bool = False
    rc_params: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def annotated_check(self) -> TracePlot:
        if self.annotated and self.features is None:
            raise ValueError(
                "If 'annotated' is True, then 'features' cannot be empty."
            )
        return self

class MultiChannelPlot(BaseModel):
    id: str
    channels: list[int]
    stimuli: list[str]
    rc_params: Optional[dict[str, Any]] = None

class IOPlot(BaseModel):
    channel: int
    features: list[str]
    stimuli: list[str]
    rc_params: Optional[dict[str, Any]] = None

class FitPlot(BaseModel):
    id: str
    channel: int
    features: list[str]
    stimulus: str
    rc_params: Optional[dict[str, Any]] = None

class DetectedPlot(BaseModel):
    features: list[str]
    channel: int
    rc_params: Optional[dict[str, Any]] = None

class AllFilesPlot(BaseModel):
    stimuli: list[str]
    output_path: Optional[str] = None
    max_per_page: Optional[int] = None
    rc_params: Optional[dict[str, Any]] = None


PlotConfig = (
    TracePlot
    | MultiChannelPlot
    | IOPlot
    | FitPlot
    | DetectedPlot
    | AllFilesPlot
)

PlotType = Literal["io", "trace", "multichannel", "fit", "detected", "allfiles"]

class Plotting(BaseModel):
    plots: Optional[dict[PlotType, PlotConfig | list[PlotConfig]]] = None

    @staticmethod
    def coerce_stimulus_fields(v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        if "stimulus" in v and isinstance(v["stimulus"], (int, float)):
            v["stimulus"] = str(v["stimulus"])
        if "stimuli" in v and isinstance(v["stimuli"], list):
            if any(isinstance(i, (int, float)) for i in v["stimuli"]):
                v["stimuli"] = list(map(str, v["stimuli"]))
        return v

    @model_validator(mode="before")
    @classmethod
    def dispatch_plot_type(cls, data: Any) -> Any:
        classes = {"trace": TracePlot, "multichannel": MultiChannelPlot, "io": IOPlot, "fit": FitPlot, "detected": DetectedPlot, "allfiles": AllFilesPlot}
        if isinstance(data, dict) and isinstance(data.get("plots"), dict):
            resolved = {}
            for k, v in data["plots"].items():
                entries = v if isinstance(v, list) else [v]
                resolved[k] = [
                    classes[k].model_validate(cls.coerce_stimulus_fields(entry))
                    for entry in entries
                ]
            data["plots"] = resolved
        return data

class RecordingConfig(BaseModel):
    experiment: Experiment
    metadata: Metadata
    analysis: Analysis
    plotting: Optional[Plotting] = None

class IntermediateResult(pa.DataFrameModel):
    id: Series[str]
    channel: Series[pl.Int32]
    time: Series[Annotated[pl.List, pl.Float32()]]
    value: Series[Annotated[pl.List, pl.Float32()]]
    stimulus: Series[str]

class FitResult(pa.DataFrameModel):
    id: Series[str] 
    channel: Series[int]
    stimulus: Series[str]
    feature_time: Series[float] 
    amplitude: Series[float]
    corr: Series[float] = pa.Field(nullable=True)
    r2: Series[float] = pa.Field(nullable=True)
    detected: Series[bool]

class FeatureResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    window: tuple[float, float]
    search_window: tuple[float, float] | float
    slope_transform: bool
    snr_threshold: float
    r2_threshold: float
    template: np.ndarray 
    template_keys: list[tuple]
    result: DataFrame[FitResult]

    @field_validator("template", mode="before")
    @classmethod
    def parse_template(cls, v):
        if isinstance(v, np.ndarray):
            return v
        return np.array(v)

    @field_serializer("template")
    def serialize_template(self, template: np.ndarray, _info):
        return template.tolist()

    @field_serializer("result")
    def serialize_result(self, result: pl.DataFrame, _info):
        return result.to_dicts()
    
class RecordingResult(BaseModel):
    results: dict[str, FeatureResult] = Field(default_factory=dict)
    def add(self, result_key: str, feature_result: FeatureResult) -> None:
        self.results[result_key] = feature_result
    def get(self, result_key: str) -> FeatureResult:
        return self.results[result_key]

