from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, SerializeAsAny, Field, model_validator
from evoked.visualization import BasePlot, PLOT_TYPES

class Experiment(BaseModel):
    name: str
    description: Optional[str] = None

class Analysis(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    default: Optional[dict[str, Any]] = None
    epoch: Optional[tuple[float, float]] = None
    event_label: Optional[str] = None
    target_frequency: Optional[float] = None
    preprocess: Optional[dict] = None
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_defaults(self) -> Analysis:
        for feature, settings in self.features.items():

            for key, default_value in (self.default or {}).items():
                if settings.get(key) is None:
                    settings[key] = default_value

        return self

class Plotting(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    plots: dict[str, list[SerializeAsAny[BasePlot]]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_plots(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("plots"), dict):
            parsed = {}
            for plot_type, configs in data["plots"].items():
                if plot_type not in PLOT_TYPES:
                    raise ValueError(f"Unknown plot type '{plot_type}'. Must be one of: {list(PLOT_TYPES)}")
                parsed[plot_type] = [PLOT_TYPES[plot_type].model_validate(c or {}) for c in (configs or [])]
            data = {**data, "plots": parsed}
        return data

class RecordingConfig(BaseModel):
    experiment: Experiment
    analysis: Analysis
    plotting: Optional[Plotting] = None