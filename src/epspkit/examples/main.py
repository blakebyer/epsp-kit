from __future__ import annotations

from epspkit.core.config import (
    FeatureConfig,
    IOConfig,
    PipelineConfig,
    SmoothingConfig,
    TransformConfig,
    VizConfig,
)
from epspkit.pipeline.api import run_pipeline


def main() -> None:
    pipeline_config = PipelineConfig(
        io=IOConfig(
            input_paths=[r"C:\\path\\to\\recording.abf"],
            output_path=r"C:\\path\\to\\output",
            repnum=3,
            stim_intensities=[25, 50, 75, 100, 150, 200],
            metadata={"experimenter": "name", "notes": "example run"},
            write_results=True,
            write_plots=True,
            render_plots=False,
        ),
        transforms=[
            TransformConfig(
                name="baseline_correction",
                params={"baseline_window_ms": (0.0, 0.1)},
            ),
            TransformConfig(
                name="crop_stim_artifact",
                params={"window_ms": (0.0, 1.25)},
            ),
            TransformConfig(name="average_sweeps"),
        ],
        features=[
            FeatureConfig(name="fiber_volley", params={"window_ms": (0.0, 1.5)}),
            FeatureConfig(
                name="epsp",
                params={"window_ms": (1.5, 5.0), "fit_distance": 4},
            ),
            FeatureConfig(
                name="pop_spike",
                params={"lag_ms": 3.0, "prominence": 0.2, "threshold": 0.05},
            ),
        ],
        plots=[
            VizConfig(name="sweep"),
            VizConfig(name="annotated"),
        ],
        global_smoothing=SmoothingConfig(method="savgol", window_size=21, polyorder=3),
    )
    run_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
