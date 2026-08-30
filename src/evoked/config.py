from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from evoked.algorithms.registry import AlgorithmType
from evoked.visualization import PlotType


class Experiment(BaseModel):
    name: str
    description: Optional[str] = None


class PreprocessConfig(BaseModel):
    baseline_window: Optional[tuple[float, float]] = (0.0, 1e-3)
    artifact: Literal["zero", "interp", "template", "none"] = "template"
    smoothing: Literal["none", "uniform", "savgol", "butter"] = "none"
    smoothing_params: dict[str, Any] = Field(default_factory=dict)


class Analysis(BaseModel):
    epoch: tuple[float, float]
    event_label: Optional[str] = None
    target_frequency: Optional[float] = None
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    features: dict[str, AlgorithmType]


class Plotting(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    plots: list[PlotType] = Field(default_factory=list)


class RecordingConfig(BaseModel):
    experiment: Experiment
    analysis: Analysis
    plotting: Optional[Plotting] = None