from __future__ import annotations
import os
from evoked.base import RecordingResult
from evoked.io import resolve_filenames, load_bulk, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.matched_filter import match_feature

data_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\ccep_data"

met_path = os.path.join(os.path.dirname(__file__), "ccep.yml")

test_files = resolve_filenames(data_path)

test = load_bulk(test_files, met_path)

pre = preprocess(test)

results = RecordingResult()

results.add("N1", match_feature(
    pre,
    window=(10e-3, 50e-3),
    noise_window=(0.5, 0.75),
    search_window=(9e-3, 60e-3)
))

results.add("N2", match_feature(
    pre,    
    window=(75e-3, 0.2),
    noise_window=(0.5, 0.75),
    search_window=(40e-3, 0.3)
))

output_path = os.path.join(os.path.dirname(__file__), f"CCEP.xlsx")
save_results_xlsx(results, output_path)
print(f"Saved results to {output_path}")