# CCEP
1. Download the [CCEP data](https://openneuro.org/datasets/ds004080/versions/1.0.1) participant 31, 50, and 68 files. 

2. Ensure the data is in the following file tree format for compliance with the [Brain Imaging Data Structure (BIDS)](https://bids-specification.readthedocs.io/en/stable/). In general, it follows the hierarchy `root/sub-*/ses-*/<datatype>/`: 
    ```
    ccep_data/
    ├── README
    ├── dataset_description.json
    ├── events.json
    ├── participants.json
    ├── participants.tsv
    ├── sub-ccepAgeUMCU31/
    │   └── ses-1/
    │       └── ieeg/
    │           ├── sub-ccepAgeUMCU31_ses-1_coordsystem.json
    │           ├── sub-ccepAgeUMCU31_ses-1_electrodes.json
    │           ├── sub-ccepAgeUMCU31_ses-1_electrodes.tsv
    │           ├── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_channels.tsv
    │           ├── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_events.tsv
    │           ├── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.eeg
    │           ├── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.json
    │           ├── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.vhdr
    │           └── sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740_ieeg.vmrk
    ├── sub-ccepAgeUMCU68/
    │   └── ses-1/
    │       └── ieeg/
    │           ├── sub-ccepAgeUMCU68_ses-1_coordsystem.json
    │           ├── sub-ccepAgeUMCU68_ses-1_electrodes.json
    │           ├── sub-ccepAgeUMCU68_ses-1_electrodes.tsv
    │           ├── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_channels.tsv
    │           ├── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_events.tsv
    │           ├── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_ieeg.eeg
    │           ├── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_ieeg.json
    │           ├── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_ieeg.vhdr
    │           └── sub-ccepAgeUMCU68_ses-1_task-SPESclin_run-011603_ieeg.vmrk
    └── sub-ccepAgeUMCU50/
        └── ses-1/
            └── ieeg/
                ├── sub-ccepAgeUMCU50_ses-1_coordsystem.json
                ├── sub-ccepAgeUMCU50_ses-1_electrodes.json
                ├── sub-ccepAgeUMCU50_ses-1_electrodes.tsv
                ├── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_channels.tsv
                ├── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_events.tsv
                ├── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_ieeg.eeg
                ├── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_ieeg.json
                ├── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_ieeg.vhdr
                └── sub-ccepAgeUMCU50_ses-1_task-SPESclin_run-021222_ieeg.vmrk
    ```
3. Run the following commands from the package root `src/evoked`.

    From the CLI (recommended):
    ```bash
    poetry run evoked-analyze --data ccep_data --output . --config recipes/ccep/ccep.yml
    ```
    __OR__

    Using the Python API (for maximum customization):
    ```bash
    poetry run python recipes/ccep/ccep.py
    ```
    _Hint_: if the CLI method raises an error like "file name expected X segments from metadata, but file contains Y segments. Skipping...", inspect the BIDS *_events.tsv. The number of events in *_events.tsv matching `config.metadata.default.event_label` should match `len(config.metadata.recordings.filename.stimulus) * config.metadata.recordings.filename.repeats`.

### Citation
D. van Blooijs, M.A. van den Boom, J.F. van der Aar, G.J.M. Huiskamp, G. Castegnaro, M. Demuru, W.J.E.M. Zweiphenning, P. van Eijsden, K. J. Miller, F.S.S. Leijten, and D. Hermes (2022). CCEP ECoG dataset across age 4-51. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004080.v1.0.1