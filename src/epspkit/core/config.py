from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Any, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from epspkit.core.context import RecordingContext

@dataclass
class SmoothingConfig:
    method: Literal["none", "moving_average", "savgol", "butter_lowpass"] = "moving_average"
    window_size: int | None = None # for moving average and savgol
    polyorder: int | None = None  # for savgol
    cutoff: float | None = None  # Hz, for butter
    fs: float | None = None  # for butter
    order: int = 3  # for butter

@dataclass
class FeatureConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

@dataclass
class IOConfig:
    input_paths: Sequence[str] = field(default_factory=list)
    output_path: Path | None = None
    repnum: int = 3
    stim_intensities: Sequence[float] = field(default_factory=list)
    write_results: bool = True
    write_plots: bool = True

@dataclass
class PipelineConfig:
    io: IOConfig = field(default_factory=IOConfig)
    features: list[FeatureConfig] = field(default_factory=list)
    global_smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

class Feature(ABC):
    """
    Base class for all EPSP-kit analyzers/features (FV, EPSP, PS, etc.)

    Subclasses should:
      - implement `run(context)`
      - use `self.config.params` for their custom settings
      - use `self.get_smoothing(global_cfg)` to determine smoothing policy
    """
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def run(self, context: RecordingContext) -> RecordingContext:
        """Run the feature analysis on the given recording context."""

    def get_smoothing(self, global_smoothing: SmoothingConfig) -> SmoothingConfig:
        """
        Determine effective smoothing:
        - If the feature defines its own smoothing → use that
        - Else → use global smoothing
        """
        if self.config.smoothing is not None:
            return self.config.smoothing
        return global_smoothing