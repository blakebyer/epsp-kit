from __future__ import annotations

import os

from evoked.io import resolve_filenames, load_bulk, save_results_xlsx, save_results_json
from evoked.preprocessing import preprocess
from evoked.algorithms.spectral import RMS
from evoked.algorithms.nonlinear import DTW
from evoked.visualization import TracePlot, IOPlot


data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "datasets",
)

met_path = os.path.join(
    os.path.dirname(__file__),
    "fepsp.yml",
)

test_files = resolve_filenames(data_path)

recording = load_bulk(test_files, met_path)

recording = preprocess(
    recording,
    params={"smoothing": "none"},
)

results = {}

results["Fiber Volley"] = DTW(
    window=(0.00125, 0.003),
    noise_window=(0.02324, 0.02499)
).match(recording)

results["fEPSP"] = DTW(
    window=(0.0025, 0.004),
    noise_window=(0.02324, 0.02499),
    slope_transform=True,
    snr_threshold=10.0,
).match(recording)

results["Population Spike"] = DTW(
    window=(0.005, 0.0075),
    noise_window=(0.02324, 0.02499),
    snr_threshold=10.0,
).match(recording)

trace = TracePlot(
    id="2025_03_02_0000",
    channel=0,
    stimuli=["50", "75", "150", "300", "400", "600"],
    features=[
        "Fiber Volley",
        "fEPSP",
        "Population Spike",
    ],
    annotated=True,
)

print(results)

# io = IOPlot(
#     channel=0,
#     features=["Fiber Volley",
#             "fEPSP",
#             "Population Spike",],
#     stimuli=["50", "75", "150", "300", "400", "600"]
# )

trace.plot(
    recording=recording,
    results=results,
)

# io.plot(
#     recording=recording,
#     results=results
# )

trace.figure.savefig(
    "trace_plot.png",
    dpi=600,
    bbox_inches="tight",
)

# io.figure.savefig(
#     "io_plot.png",
#     dpi=600,
#     bbox_inches="tight"
# )

save_results_json(results, "results.json")
save_results_xlsx(results, "results.xlsx")