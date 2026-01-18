from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from epspkit.core.context import RecordingContext
from epspkit.core.config import VizConfig, SmoothingConfig
from epspkit.features.base import Feature
import scienceplots


class Plot(ABC):
    """
    Base class for all epspkit plotters.

    Subclasses should:
      - implement `render(context)` to create their plots
      - read config from `self.config.params`
    """

    def __init__(self, config: VizConfig, effective_smoothing: SmoothingConfig | None = None):
        """
        Parameters
        ----------
        config
            Per-plot configuration (name, params, optional smoothing).
        """
        self.config = config
        self.name = config.name
        self.stim_intensities = list(config.stim_intensities or [])
        
        # Final smoothing policy for this plot instance:
        self.smoothing: SmoothingConfig = (
            effective_smoothing
            or config.smoothing
            or SmoothingConfig()
        )

    @abstractmethod
    def render(self, context: RecordingContext) -> None:
        """Render the plot for the given context."""
        raise NotImplementedError

    apply_smoothing = Feature.apply_smoothing

    def _resolve_output_path(
        self,
        context: RecordingContext,
        output_path: Path | str,
        ext: str = "png",
        output_stem: str | None = None,
    ) -> Path:
        output = Path(output_path)
        stem = output_stem or "recording"

        if output.suffix:
            output_dir = output.parent
            filename = f"{stem}_{self.name}.{ext}"
        else:
            output_dir = output
            filename = f"{stem}_{self.name}.{ext}"

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
