from __future__ import annotations

from pathlib import Path

from epspkit.core.config import FeatureConfig, IOConfig, PipelineConfig, SmoothingConfig, VizConfig
from epspkit.pipeline.api import run_pipeline
from epspkit.transforms.average import average_sweeps
from epspkit.transforms.baseline import baseline_correction
from epspkit.transforms.stim_artifact import crop_stim_artifact


def main() -> None:
    pipeline_config = PipelineConfig(
        io=IOConfig(
            input_paths=[
                # "path/to/recording.abf",
            ],
            output_path=Path("results"),
            repnum=1,
            stim_intensities=[
                # 10.0, 20.0, 30.0,
            ],
            write_results=True,
            write_plots=True,
        ),
        features=[
            FeatureConfig(name="fiber_volley"),
            FeatureConfig(name="epsp"),
            FeatureConfig(name="pop_spike"),
        ],
        global_smoothing=SmoothingConfig(method="none"),
    )

    transforms = [
        crop_stim_artifact,
        baseline_correction,
        average_sweeps,
    ]

    plots = [
        VizConfig(name="sweep"),
        VizConfig(name="derivative"),
        VizConfig(name="annotated"),
        VizConfig(name="input_output"),
    ]

    run_pipeline(pipeline_config, transforms=transforms, plots=plots)


if __name__ == "__main__":
    main()
