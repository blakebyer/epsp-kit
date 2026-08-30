# evoked #
evoked is a package for analyzing evoked local field potentials using template matching.

## What it does
- Load electrophysiology recordings into tidy polars DataFrames
- Preprocesses tidy data (baseline correction, stimulus artifact removal, averaging)
- Matches features (e.g., fiber volley, fEPSP slope, population spike)
- Renders and save common plots (e.g., IO curves)

## Installation
### 1. [Install Python $\geq$ 3.12](https://www.python.org/downloads/)
Check installation:
```bash
python --version
```

### 2. Install Poetry
```bash
python -m pip install poetry
```
### 3. Install evoked
```bash
git clone https://github.com/blakebyer/evoked.git
cd evoked
poetry env use python3.12 # or preferred version
poetry install
```

## Usage
### Command Line (Recommended) ###
1. Create a YAML configuration file adhering to this schema:
```yaml
experiment:  # required
  name: <string>  # required
  description: <string>  # optional
metadata:  # required
  default:  # optional -- mapping
    <name>:
      <any>
  recordings:  # required -- mapping
    <name>:
      block_index: <integer>  # optional
      id: <string>  # optional
      stimulus: <list[string]>  # required
      order: <grouped | interleaved | explicit>  # required
      repeats: <integer>  # optional
      event_label: <string>  # optional
      stimulus_unit: <string>  # optional
      layout: <segments | continuous>  # optional
analysis:  # required
  default:  # optional -- mapping
    <name>:
      <any>
  epoch: <tuple[number, number]>  # optional
  target_frequency: <number>  # optional
  preprocess:  # optional -- mapping
    <name>:
      <any>
  features:  # optional -- mapping
    <name>:
      window: <tuple[number, number]>  # required
      noise_window: <tuple[number, number]>  # required
      search_window: <tuple[number, number] | number>  # optional
      derivative_transform: <boolean>  # optional
      snr_threshold: <number>  # optional
      r2_threshold: <number>  # optional
plotting:  # optional
  plots:  # optional -- mapping of plot type
    trace:
      id: <string>  # required
      channel: <integer>  # required
      features: <list[string]>  # optional
      stimuli: <list[string]>  # required
      annotated: <boolean>  # optional
      rc_params:  # optional -- mapping
        <name>:
          <any>
    multichannel:
      id: <string>  # required
      channels: <list[integer]>  # required
      stimuli: <list[string]>  # required
      rc_params:  # optional -- mapping
        <name>:
          <any>
    io:
      channel: <integer>  # required
      features: <list[string]>  # required
      stimuli: <list[string]>  # required
      rc_params:  # optional -- mapping
        <name>:
          <any>
    fit:
      id: <string>  # required
      channel: <integer>  # required
      features: <list[string]>  # required
      stimulus: <string>  # required
      rc_params:  # optional -- mapping
        <name>:
          <any>
    detected:
      features: <list[string]>  # required
      channel: <integer>  # required
      rc_params:  # optional -- mapping
        <name>:
          <any>
    allfiles:
      stimuli: <list[string]>  # required
      output_path: <string>  # optional
      max_per_page: <integer>  # optional
      rc_params:  # optional -- mapping
        <name>:
          <any>
```
2. Run `evoked` from the CLI:
```bash
> poetry run evoked-analyze --data . --config config.yml
```
Details:
```bash
> poetry run evoked-analyze --help
usage: evoked.analyze [-h] [--data PATH] [--config PATH] [--output PATH] [--describe-config]

Run the evoked analysis pipeline end-to-end from a YAML config: load raw recordings, preprocess, run feature detection, render plots, and save results.

options:
  -h, --help         show this help message and exit
  --data PATH        Directory containing data files
  --config PATH      Path to YAML config file
  --output PATH      Directory to save results into (default: current directory)
  --describe-config  Print the expected config.yml structure and exit
```

### Python API ###
While you can do everything with `evoked.analyze` that you can do with the Python API, you might sometimes prefer a more verbose experience. We have supplied config YAML and Python recipes for common evoked potential types in the `recipes` folder, such as: 

1. Field excitatory postsynaptic potentials (fEPSP)
2. Cortico-cortical evoked potentials (CCEP)
3. Long term potentiation/depression (LTP/LTD)

You might use the Python API if you want to:
- minimally edit the configuration file, where only `config.experiment` and `config.metadata` are required
- have maximum customizability of the preprocessing and template matching parameters
- load data from many directories or inspect intermediate files


### Web App ###
evoked comes with a basic streamlit web app:
```bash
poetry run streamlit run app.py
```

### Tests ###
Tests can be run with

```bash
poetry run pytest --cov=evoked --cov-report=term-missing --cov-report=xml
```

## Advanced Usage ##

### Preprocessing ###
The default behavior is to complete these transformations in order:
1. __Baseline correction__: subtracts the mean value in the first ms from the entire trace
2. __Artifact removal__: interpolates from before and after biphasic peaks in the signal.
3. __Averaging traces__: averages across repeats of the same stimulus.
4. __Smoothing__: Savitzky-Golay filter with polynomial order 2 and window length 11 which does not smooth in artifact windows to prevent ringing.

You can set your preferred arguments in the `config.analysis.preprocess` dictionary.

### Template Matching ###
The default template matching method is a matched filter (in `matched_filter.py`) which is well-suited to electrophysiological signals with variable noise.

For long continuous signals we have provided a generalized likelihood ratio test (GLRT) filter (in `glrt.py`). Warning: this is not well suited to identifying many feature types in a signal. High amplitude spikes can "trick" GLRT because the algorithm is scale invariant. The benefit of GLRT is hypothesis testing.

### Plotting ###
Every plotting function has an `rc_params` argument, a dictionary of `matplotlib.rcParams`, which makes plots highly customizable. 

### Hyperparameter Calibration ###
The default $\mathrm{R^2}$ threshold is 0.8, which may be high depending on the feature variability. You can supply truth labels for a subset of your data to calibrate the detection threshold, for example:

| id | channel | stimulus | feature | detected | 
| --- | --- | --- | --- | --- |
| file1 | 0 | 25 | fEPSP | True | 
| file1 | 35 | 75 | population spike | False | 
| file2 | 120 | puff1 | whisker | False |
| file1 | 97 | puff2 | N1 | False |
| file2 | 6 | puff2 | whisker | True |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |


Get the optimal $\mathrm{R^2}$ threshold for your data in the CLI:
```bash
> poetry run evoked-calibrate --truth ccep_truth.xlsx --pred ccep.xlsx 
Best R^2 threshold for fEPSP=0.915 at balanced accuracy=0.855
Best R^2 threshold for N1=0.865 at balanced accuracy=0.730
Best R^2 threshold for population spike=0.779 at balanced accuracy=0.926
Best R^2 threshold for whisker=0.431 at balanced accuracy=0.671
```
Details:
```bash
> poetry run evoked-calibrate --help
usage: evoked.calibrate [-h] [--truth PATH] [--pred PATH] [--metric str] [--output PATH]         

Compute hyperparameter calibration based on truth labels.

options:
  -h, --help     show this help message and exit
  --truth PATH   Path to truth labels in tabular format
  --pred PATH    Path to recording results in Excel or JSON format
  --metric str   Metric on which to base the calibration (default: balanced accuracy)
  --output PATH  Path to save calibration results into
```
