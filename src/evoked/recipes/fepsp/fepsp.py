from __future__ import annotations
import os
from evoked.base import RecordingResult
from evoked.io import resolve_filenames, load_bulk, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.matched_filter import match_feature

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")

met_path = os.path.join(os.path.dirname(__file__), "fepsp.yml")

test_files = resolve_filenames(data_path)

## Equivalent to: 
# recordings = ["2025_03_02_0000.abf", "2025_03_02_0003.abf", "2025_03_02_0010.abf"]
# test_files = [os.path.join(data_path, f) for f in recordings]

test = load_bulk(test_files, met_path)

pre = preprocess(test)

results = RecordingResult()

results.add("Fiber Volley", match_feature(
    pre,
    window=(0.00125, 0.003),
    noise_window=(0.02324, 0.02499),
))

results.add("fEPSP", match_feature(
    pre,    
    window=(0.0025, 0.004),
    noise_window=(0.02324, 0.02499),
    slope_transform=True,
))

results.add("Population Spike", match_feature(
    pre,
    window=(0.005, 0.0075),
    noise_window=(0.02324, 0.02499),
))

output_path = os.path.join(os.path.dirname(__file__), f"fEPSP experiment.xlsx")
save_results_xlsx(results, output_path)
print(f"Saved results to {output_path}")