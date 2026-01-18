# epspkit

This document describes the pipeline and input-output configurations, and the knobs/dials for each feature.

## Pipeline overview
The pipeline is configured with a `PipelineConfig` that controls:
- `io`: input/output settings, output path, metadata, and write/render flags
- `transforms`: ordered list of transforms (baseline, stim artifact removal, average)
- `features`: feature analyzers and their parameters
- `plots`: plot selection and optional per-plot smoothing
- `global_smoothing`: default smoothing policy used by features and plots unless overridden

A typical flow is:
1) Load ABF → tidy DataFrame
2) Apply transforms in order
3) Average sweeps → averaged DataFrame
4) Run features → results tables
5) Render/save plots → PNGs
6) Save results → XLSX with tidy/averaged/results/metadata/config

## IO Config
The `IOConfig` section controls data input, output, and sweep organization.

Required fields:
- `input_paths` (list[str]): paths to ABF files.
- `output_path` (str | Path): directory or file path for XLSX/plots when writing.
- `stim_intensities` (list[float]): ordered list matching the acquisition order. Does not necessarily have to be stimulus intensity, but must be specified since it is a grouping variable for averaging traces.
- `repnum` (int): number of sweeps per stimulus intensity.

Important:
- `repnum` must be constant across all stimulus intensities for a given recording.
  The loader assumes `len(stim_intensities) * repnum == number_of_sweeps`.
  If repnum varies by intensity, the loader will raise an error or mis-assign sweeps.
- `{stem}` refers to the input_path filename before the `.abf` extension (e.g., "2025_05_0001.abf" → "2025_05_0001").

Optional fields:
- `metadata` (dict): free-form key/value pairs saved into the XLSX metadata sheet.
- `write_results` (bool): save `{stem}_results.xlsx`.
- `write_plots` (bool): save plot images.
- `render_plots` (bool): show plots on screen.

## Feature knobs and dials

### FiberVolleyFeature (`name="fiber_volley"`)
Required params:
- `window_ms` (tuple[float, float])
  - Time window in milliseconds after stimulus onset to search for the fiber volley.
  - Example: `(0.0, 1.5)`

Outputs:
- `fv_amp`: absolute amplitude of the volley (mV)
- `fv_s`: time (s) of detected volley
- `fv_v`: voltage (mV) at volley

### EPSPFeature (`name="epsp"`)
Required params:
- `window_ms` (tuple[float, float])
  - Time window in milliseconds for searching the EPSP trough and slope.
  - Example: `(1.5, 5.0)`
- `fit_distance` (int)
  - Number of points on either side of the slope minimum used to fit a line.
  - Example: `4`

Outputs:
- `epsp_s`, `epsp_v`: time/voltage at EPSP minimum
- `slope_mid_s`, `slope_mid_v`: time/voltage at slope minimum
- `epsp_amp`: absolute EPSP amplitude
- `epsp_slope`: absolute slope (mV/ms)
- `epsp_r2`: fit quality
- `epsp_to_fv`: slope divided by FV amplitude (requires FV result)

Notes:
- If FiberVolleyFeature is not run, `epsp_to_fv` is `NaN` and a warning is raised.

### PopSpikeFeature (`name="pop_spike"`)
Required params:
- `lag_ms` (float)
  - Search window length after the EPSP minimum (ms).
  - Example: `3.0`
- `prominence` (float)
  - Minimum peak prominence (mV) for population spike detection.
  - Example: `0.2`
- `threshold` (float)
  - Slope threshold (mV/ms) for fallback curvature-based detection.
  - Example: `0.05`

Outputs:
- `ps_amp`: population spike amplitude (mV)
- `ps_s`, `ps_v`: time/voltage at PS peak

Notes:
- Requires EPSPFeature results.

## Smoothing
Smoothing is controlled via `SmoothingConfig`:
- `method`: `none`, `moving_average`, `savgol`, or `butter_lowpass`
- `window_size`, `polyorder`: for `moving_average`/`savgol`
- `cutoff`, `order`: for `butter_lowpass`

Behavior:
- `global_smoothing` sets the default for all features and plots.
- A plot can override smoothing by specifying `VizConfig.smoothing`.

## Transforms
Transforms are configured by name with params via `TransformConfig`:
- `baseline_correction`:
  - `baseline_window_ms` (tuple[float, float])
- `crop_stim_artifact`:
  - `window_ms` (tuple[float, float])
- `template_subtract_stim_artifact`:
  - `window_ms` (tuple[float, float])
- `average_sweeps`: no params

Order matters. Recommended:
1) `baseline_correction`
2) `crop_stim_artifact` or `template_subtract_stim_artifact`
3) `average_sweeps`

## Plots
Available plot names (via `VizConfig.name`):
- `sweep`: averaged sweeps
- `derivative`: sweeps + derivative
- `annotated`: sweeps with feature markers
- `input_output`: plots synaptic strength (fEPSP slope vs fiber volley amplitude), presynaptic excitability (fiber volley amplitude vs stimulus intensity), and postsynaptic responsiveness (fEPSP slope vs stimulus intensity).

Plot output naming uses the input file stem: `{stem}_{plot}.png`.

## Results export
Each input file writes one workbook:
- `{stem}_results.xlsx`
- Sheets: `tidy`, `averaged`, `result_*`, `metadata`, `pipeline_config`

## Example config snippet
```python
PipelineConfig(
    io=IOConfig(
        input_paths=["/path/to/recording.abf"],
        output_path="/path/to/output",
        repnum=3,
        stim_intensities=[25, 50, 75, 100, 150, 200],
        metadata={"experimenter": "name"},
        write_results=True,
        write_plots=True,
        render_plots=False,
    ),
    transforms=[
        TransformConfig(name="baseline_correction", params={"baseline_window_ms": (0.0, 0.1)}),
        TransformConfig(name="crop_stim_artifact", params={"window_ms": (0.0, 1.25)}),
        TransformConfig(name="average_sweeps"),
    ],
    features=[
        FeatureConfig(name="fiber_volley", params={"window_ms": (0.0, 1.5)}),
        FeatureConfig(name="epsp", params={"window_ms": (1.5, 5.0), "fit_distance": 4}),
        FeatureConfig(name="pop_spike", params={"lag_ms": 3.0, "prominence": 0.2, "threshold": 0.05}),
    ],
    plots=[VizConfig(name="sweep")],
    global_smoothing=SmoothingConfig(method="savgol", window_size=21, polyorder=3),
)
```
