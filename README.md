# evoked
evoked is a package for analyzing evoked local field potentials using template matching.

## What it does
- Load ABF/CSV recordings into tidy polars DataFrames
- Preprocesses tidy data (baseline correction, stimulus artifact removal, averaging)
- Matches features (e.g., fiber volley, fEPSP slope, population spike)
- Renders and save common plots (e.g., IO curves)

## Installation
### 1. [Install Python >= 3.13](https://www.python.org/downloads/)
Check installation:
```bash
python3.13 --version
```

### 2. Install Poetry
```bash
python3.13 -m pip install poetry
```
### 3. Install evoked
```bash
git clone https://github.com/blakebyer/evoked.git
cd evoked
poetry env use python3.13
poetry install
```

## Usage
### Python API ###
The Python API is recommended for heavy users.

Edit `main.py` and run:
```bash
poetry run python -m evoked.main
```
### Jupyter Notebook ###
To run the example notebook:

```bash
poetry run jupyter notebook
```
Select [examples/analysis_2026_06_19.ipynb](https://github.com/blakebyer/evoked/blob/main/src/evoked/examples/analysis_2026_06_19.ipynb), follow instructions, and run it. 

### Web App ###
evoked comes with a basic streamlit web app, which can be run by:
```bash
poetry run streamlit run src/evoked/app.py
```

## Output
- Results are written to one Excel (.xlsx) file with `save_results_xlsx` or JSON file with `save_results_json`.
- Each workbook contains sheets: `Pipeline` containing pipeline configuration information and one sheet per feature with feature parameters and results.
