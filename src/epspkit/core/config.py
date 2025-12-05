from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence


@dataclass
class SmoothingConfig:
    """
    Configuration for optional trace smoothing.

    Defaults:
      - method="none" → no smoothing unless explicitly enabled
      - window_size / polyorder chosen to be reasonable for EPSP traces
      - cutoff/order for a generic low-pass if you want Butterworth
    """
    method: Literal["none", "moving_average", "savgol", "butter_lowpass"] = "none"
    window_size: int = 11      # used by moving_average and savgol
    polyorder: int = 3         # only used by savgol
    cutoff: float = 2000.0     # Hz, for butter_lowpass
    order: int = 3             # filter order for butter_lowpass


@dataclass
class FeatureConfig:
    """
    Per-feature configuration.

    name    : identifier for the feature (e.g., "fv", "epsp", "ps")
    params  : arbitrary feature-specific settings
    smoothing : optional smoothing policy for this feature
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass
class IOConfig:
    """
    I/O and basic acquisition configuration for a pipeline run.
    """
    input_paths: Sequence[str] = field(default_factory=list)
    output_path: Path | None = None
    repnum: int = 3
    stim_intensities: Sequence[float] = field(default_factory=list)
    write_results: bool = True
    write_plots: bool = True


@dataclass
class PipelineConfig:
    """
    Top-level configuration for an analysis pipeline.

    io              : input/output and acquisition parameters
    features        : list of feature configs to run
    global_smoothing: default smoothing policy, unless overridden per feature
    """
    io: IOConfig = field(default_factory=IOConfig)
    features: list[FeatureConfig] = field(default_factory=list)
    global_smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)