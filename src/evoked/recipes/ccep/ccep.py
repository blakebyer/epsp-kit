from __future__ import annotations

import os
from evoked.io import (
    resolve_filenames,
    load_recording,
    save_results_xlsx,
)
from evoked.preprocessing import preprocess, select_low_noise_channels
from evoked.template import build_template_snr
from evoked.algorithms.matched_filter import MatchedFilter

data_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\ccep_data"

ccep_files = resolve_filenames(data_path)

recording = load_recording(
    filenames=ccep_files,
    trials="ccep_trials.tsv",
    epoch=(-1.0, 0.5),
    event_label="electrical_stimulation"
)

recording = preprocess(
    recording=recording,
)

recording = select_low_noise_channels(
    recording,
    noise_window=(-1.0, -0.1),
    threshold=5.0,
)

print(recording.channel_names)

# -- N1 --

n1_template = build_template_snr(
    recording=recording,
    window=(9e-3, 60e-3),
    noise_window=(-1.0, -0.1),
    snr_threshold=3.4,
    polarity="negative",
)

# -- N2 --

n2_template = build_template_snr(
    recording=recording,
    window=(40e-3, 0.3),
    noise_window=(-1.0, -0.1),
    snr_threshold=3.4,
    polarity="negative",
)


results = {}

results["N1"] = MatchedFilter().match(
    recording,
    template=n1_template,
)

results["N2"] = MatchedFilter().match(
    recording,
    template=n2_template,
)


output_path = os.path.join(
    os.path.dirname(__file__),
    "CCEP.xlsx",
)

save_results_xlsx(
    results,
    output_path,
)