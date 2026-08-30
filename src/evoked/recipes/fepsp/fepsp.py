from __future__ import annotations

import os

import polars as pl
from evoked.io import resolve_filenames, load_recording, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.template import build_template_snr
from evoked.algorithms.matched_filter import MatchedFilter
from evoked.algorithms.dtw import DTW
from evoked.visualization import TracePlot, IOPlot, DetectedPlot, FitPlot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# import seaborn as sns

datadir = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab"

jcv = pl.read_csv(os.path.join(datadir, 'blind_classifier_jcv.csv'))
adb = pl.read_csv(os.path.join(datadir, 'blind_classifier_ADB.csv'))
sms = pl.read_csv(os.path.join(datadir, 'blind_classifier_sms.csv'))
datafiles = pl.read_excel(
    os.path.join(datadir, r'evoked_user_test\evoked\src\evoked\examples', 'datafiles_table.xlsx'),
    sheet_name="Sheet1",
)
datafiles = datafiles.with_columns(
    pl.col("Data File").str.replace(".abf", "", literal=True).alias("id")
)

FEATURES = ["fiber_volley", "fepsp", "population_spike"]

long_df = pl.concat([jcv, adb, sms]).with_columns(
    [pl.col(feat).replace({"Yes": 1, "No": 0}).cast(pl.Int64) for feat in FEATURES]
)

pooled = long_df.unpivot(
    index=["id", "intensity", "reviewer_alias"],
    on=FEATURES,
    variable_name="feature",
    value_name="rating",
)

wide_all = pooled.pivot(
    index=["id", "intensity", "feature"],
    on="reviewer_alias",
    values="rating",
)

REVIEWER_COLS = ["ADB", "jcv", "sms"]

# majority vote across the 3 reviewer columns (row-wise mode)
wide_all = wide_all.with_columns(
    pl.sum_horizontal(REVIEWER_COLS).alias("_vote_sum")
).with_columns(
    (pl.col("_vote_sum") >= 2).cast(pl.Int64).alias("detected_consensus")
).drop("_vote_sum")

wide_all = wide_all.join(datafiles.select(["id", "Brain Region"]), on="id", how="left")

truth_df = wide_all.rename({
    "intensity": "stimulus",
    "Brain Region": "brain_region",
}).select(["id", "stimulus", "brain_region", "feature", "detected_consensus"])


data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "datasets",
)

fepsp_files = resolve_filenames(data_path)

recording = load_recording(
    filenames=fepsp_files,
    trials="fepsp_trials.tsv",
    epoch=(0.0, 25e-3),
)

recording = preprocess(
    recording=recording,
)

ca1 = recording.select_trials(
    brain_region="CA1"
)

dg = recording.select_trials(
    brain_region="DG",
)

# -- CA1 --
ca1_results = {}

ca1_results["Fiber Volley"] = MatchedFilter(
    window=(0.00125, 0.003),
    noise_window=(0.023, 0.025),
    ).match(ca1)

ca1_results["fEPSP"] = DTW(
    window=(0.0025, 0.00425),
    noise_window=(0.023, 0.025),
    derivative_transform=True,
).match(ca1)

ca1_results["Population Spike"] = MatchedFilter(
    window=(0.00425,0.006),
    noise_window=(0.023, 0.025),
    ).match(ca1)


# -- DG --

dg_fv_template = build_template_snr(
    recording=dg,
    window=(0.00125, 0.003),
    noise_window=(0.023, 0.025)
)

dg_fepsp_template = build_template_snr(
    recording=dg,
    window=(0.003, 0.0045),
    noise_window=(0.023, 0.025),
)

dg_ps_template = build_template_snr(
    recording=dg,
    window=(0.0045, 0.006),
    noise_window=(0.023, 0.025)
)

dg_results = {}

dg_results["Fiber Volley"] = MatchedFilter(
    window=(0.00125, 0.003),
    noise_window=(0.023, 0.025),
).match(dg)

dg_results["fEPSP"] = MatchedFilter(
    window=(0.003, 0.0045),
    noise_window=(0.023, 0.025),
    derivative_transform=True,
).match(dg)

dg_results["Population Spike"] = MatchedFilter(
    window=(0.0045, 0.006),
    noise_window=(0.023, 0.025),
    ).match(dg)

# # -- Truth comparison --

# FEATURE_KEY_MAP = {
#     "Fiber Volley": "fiber_volley",
#     "fEPSP": "fepsp",
#     "Population Spike": "population_spike",
# }

# def r2_separation_density(merged: pl.DataFrame, feature_name: str, brain_region, value_col: str = "r2", bw_adjust: float = 1.0):
#     bounds = (0, 1) if value_col == "r2" else (-1, 1)
#     xlabel = r"$\mathrm{R^2}$" if value_col == "r2" else "Correlation"

#     detected = merged.filter(pl.col("detected_consensus") == 1)[value_col].to_numpy()
#     not_detected = merged.filter(pl.col("detected_consensus") == 0)[value_col].to_numpy()

#     fig, ax = plt.subplots(figsize=(4,3))
#     sns.kdeplot(detected, ax=ax, bw_adjust=bw_adjust, linewidth=1.6, color="#087e8b", label="Detected") # fill=True, alpha=0.5, label="Detected", 
#     sns.kdeplot(not_detected, ax=ax, bw_adjust=bw_adjust, linewidth=1.6, color="#ff5a5f", label="Not Detected") # fill=True, alpha=0.5, label="Not Detected", 

#     # for b in bounds:
#     #     ax.axvline(b, color="0.3", linestyle="--", linewidth=1)

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(None)
#     ax.set_yticks([])
#     ax.set_xticks([-0.5,0.0,0.5,1.0])
#     #ax.legend(frameon=False)
#     ax.spines[['top', 'right', 'left']].set_visible(False)
#     #ax.set_title(f"{feature_name} {xlabel} Distribution")
#     fig.savefig(f"{feature_name}_{value_col}_density_{brain_region}.png", dpi=600, bbox_inches="tight")
#     return fig, ax


# def merge_with_consensus(result, truth_df: pl.DataFrame, feature_key: str, brain_region: str) -> pl.DataFrame:
#     result_df = result.result.with_columns(
#         pl.col("stimulus").cast(pl.Int64),
#         pl.col("file_origin").str.replace(".abf", "", literal=True).alias("id"),
#     )

#     ground_truth = truth_df.filter(
#         (pl.col("brain_region") == brain_region) & (pl.col("feature") == feature_key)
#     ).select(["id", "stimulus", "detected_consensus"]).with_columns(
#         pl.col("stimulus").cast(pl.Int64)
#     )

#     return result_df.join(ground_truth, on=["id", "stimulus"], how="inner")


# for region, results in [("CA1", ca1_results), ("DG", dg_results)]:
#     for feature_name, result in results.items():
#         merged = merge_with_consensus(result, truth_df, FEATURE_KEY_MAP[feature_name], region)
#         r2_separation_density(merged, feature_name, region, value_col="corr")

# -- IO plots --

# stimuli = [
#     25, 50, 75, 100, 150, 200,
#     250, 300, 400, 500, 600,
# ]

# io = IOPlot(
#     channel=0,
#     features=[
#         "Fiber Volley",
#         "fEPSP",
#         "Population Spike",
#     ],
#     stimuli=stimuli,
#     stimulus_unit="μA",
# )

# io.plot(
#     recording=recording,
#     results=ca1_results,
# )

# io.figure.savefig(
#     "ca1_io_plot.png",
#     dpi=600,
#     bbox_inches="tight",
# )

# io.plot(
#     recording=recording,
#     results=dg_results,
# )

# io.figure.savefig(
#     "dg_io_plot.png",
#     dpi=600,
#     bbox_inches="tight",
# )


save_results_xlsx(
    ca1_results,
    "ca1_results.xlsx",
)
print("Saved ca1_results.xlsx")

save_results_xlsx(
    dg_results,
    "dg_results.xlsx",
)
print("Saved dg_results.xlsx")