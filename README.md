# evoked #
evoked is a package for analyzing evoked local field potentials using template matching.

## What it does
- Load electrophysiology recordings into tidy polars DataFrames
- Preprocesses tidy data (baseline correction, stimulus artifact removal, averaging)
- Matches features (e.g., fiber volley, fEPSP slope, population spike)
- Renders and save common plots (e.g., IO curves)

## Installation
### 1. [Install Python >=3.10,<3.15](https://www.python.org/downloads/)
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
### Command Line (Recommended)
Run evoked from the CLI:
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

### Web App ###
evoked comes with a basic nice gui web/desktop app, which can be run by:
```bash
poetry run flet run app.py
```

## Output
- Results are written to one Excel (.xlsx) file with `save_results_xlsx` or JSON file with `save_results_json`.
