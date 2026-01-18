# epsp-kit
A toolkit for analyzing excitatory post-synaptic potentials (EPSPs) from ABF recordings.

## What it does
- Load ABF recordings into tidy and averaged pandas DataFrames
- Apply transforms (baseline correction, stim artifact removal, averaging)
- Run features (fiber volley, fEPSP slope, population spike)
- Render plots and/or save plot images
- Save results to a single XLSX workbook per input file

## Installation (Poetry)
```bash
git clone https://github.com/blakebyer/epsp-kit.git
cd epsp-kit/epsp-kit
poetry install
```

Run via Poetry:
```bash
poetry run python -m epspkit.cli.main
```

Optional notebook extras:
```bash
poetry install --with notebook
```

## Quick start (pipeline)
```python
from epspkit.core.config import (
    FeatureConfig,
    IOConfig,
    PipelineConfig,
    SmoothingConfig,
    TransformConfig,
    VizConfig,
)
from epspkit.pipeline.api import run_pipeline

pipeline_config = PipelineConfig(
    io=IOConfig(
        input_paths=["/path/to/recording.abf"],
        output_path="/path/to/output/results.xlsx",
        repnum=3,
        stim_intensities=[25, 50, 75, 100, 150, 200],
        metadata={"experimenter": "name", "notes": "pilot run"},
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
        VizConfig(name="derivative"),
        VizConfig(name="annotated"),
    ],
    global_smoothing=SmoothingConfig(method="savgol", window_size=21, polyorder=3),
)

run_pipeline(pipeline_config)
```

## Run Quick Start from the Command-Line 
```cli
python -m epspkit.cli.main
```

## Output
- Results are written to one XLSX per input file named `{stem}_results.xlsx`.
- Each workbook contains sheets: `tidy`, `averaged`, `result_*`, `metadata`, `pipeline_config`.
- Plot images are saved as `{stem}_{plot}.png` when `write_plots=True`.

## Notes
- Feature params are required (no implicit defaults).
- Transform order matters: baseline -> stim artifact removal -> average.
- If multiple `input_paths` are provided, outputs are indexed with `_1`, `_2`, etc.

## Examples
- `examples/main.py`
- `examples/quickstart.ipynb`
