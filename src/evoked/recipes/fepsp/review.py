"""Automated vs. manual fEPSP feature agreement and detection performance."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_curve,
)
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from evoked.io import load_recording, resolve_filenames, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.template import build_template_snr
from evoked.algorithms.matched_filter import MatchedFilter

DATADIR = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab"
REGIONS = ["CA1", "DG"]
REGION_COLORS = {"CA1": "#1A85FF", "DG": "#D41159"}
FEATURE_STYLES = {"Fiber Volley": "-", "fEPSP": "--", "Population Spike": ":"}
CLASSIFIER_FEATURES = ["fiber_volley", "fepsp", "population_spike"]
FEATURE_KEY_MAP = {"Fiber Volley": "fiber_volley", "fEPSP": "fepsp", "Population Spike": "population_spike"}


def _minimal(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------- #
# Blind-reviewer concordance (Fleiss' kappa)
# --------------------------------------------------------------------- #

def load_classifications(datadir: str) -> pl.DataFrame:
    frames = [
        pl.read_csv(os.path.join(datadir, f"blind_classifier_{name}.csv"))
        for name in ("jcv", "ADB", "sms")
    ]
    long_df = pl.concat(frames).with_columns(
        [pl.col(f).replace({"Yes": 1, "No": 0}).cast(pl.Int64) for f in CLASSIFIER_FEATURES]
    )
    return long_df.with_columns(
        (pl.col("id").cast(pl.Utf8) + "_" + pl.col("intensity").cast(pl.Utf8)).alias("subject")
    )


def fleiss_kappas(long_df: pl.DataFrame) -> dict[str, float]:
    kappas = {}
    for feat in CLASSIFIER_FEATURES:
        wide = long_df.pivot(index="subject", on="reviewer_alias", values=feat).drop("subject")
        counts, _ = aggregate_raters(wide.to_numpy())
        kappas[feat] = fleiss_kappa(counts, method="fleiss")
        print(f"Fleiss Kappa ({feat}) = {kappas[feat]:.3f}")

    pooled = long_df.unpivot(
        index=["subject", "reviewer_alias"], on=CLASSIFIER_FEATURES,
        variable_name="feature", value_name="rating",
    ).with_columns((pl.col("subject") + "_" + pl.col("feature")).alias("item"))
    wide_pooled = pooled.pivot(index="item", on="reviewer_alias", values="rating").drop("item")
    print(f"ADB-jcv = {fleiss_kappa(aggregate_raters(wide_pooled[['ADB', 'jcv']].to_numpy())[0]):.3f}")
    print(f"ADB-sms = {fleiss_kappa(aggregate_raters(wide_pooled[['ADB', 'sms']].to_numpy())[0]):.3f}")
    print(f"jcv-sms = {fleiss_kappa(aggregate_raters(wide_pooled[['jcv', 'sms']].to_numpy())[0]):.3f}")
    counts_pooled, _ = aggregate_raters(wide_pooled.to_numpy())
    kappas["pooled"] = fleiss_kappa(counts_pooled, method="fleiss")
    print(f"Fleiss Kappa (pooled) = {kappas['pooled']:.3f}")
    return kappas


def consensus_truth(long_df: pl.DataFrame, datafiles: pl.DataFrame) -> pl.DataFrame:
    pooled = long_df.unpivot(
        index=["id", "intensity", "reviewer_alias"], on=CLASSIFIER_FEATURES,
        variable_name="feature", value_name="rating",
    )
    wide = pooled.pivot(index=["id", "intensity", "feature"], on="reviewer_alias", values="rating")
    wide = wide.with_columns(
        (pl.sum_horizontal(["ADB", "jcv", "sms"]) >= 2).cast(pl.Int64).alias("detected_consensus")
    ).join(datafiles.select(["id", "Brain Region"]), on="id", how="left")
    return wide.rename({"intensity": "stimulus", "Brain Region": "brain_region"}).select(
        ["id", "stimulus", "brain_region", "feature", "detected_consensus"]
    )


# --------------------------------------------------------------------- #
# Manual quantitative amplitude/slope lookup
# --------------------------------------------------------------------- #

def load_manual_amplitudes(datadir: str, datafiles: pl.DataFrame) -> pl.DataFrame:
    """The manual sheet keys on (animal_id, hippo_reg, slice#) rather than
    recording id, so slices are matched to ids in acquisition order within
    each animal/region. That ordering assumption is unavoidable given the
    source data, but it's isolated here rather than inline at module level."""
    b6 = pl.read_excel(
        os.path.join(datadir, r"evoked_user_test\evoked\src\evoked\data", "b6_timecourse_data_all.xlsx"),
        sheet_name="fv_epsp",
    ).rename({"stim_intensity": "stimulus"})

    id_map = (
        datafiles.select(["id", "Mouse ID", "Brain Region"])
        .rename({"Mouse ID": "animal_id", "Brain Region": "hippo_reg"})
        .drop_nulls(subset=["id"])
        .unique()
        .with_columns(pl.col("id").str.split("_").list.last().cast(pl.Int64).alias("_suffix"))
        .sort(["animal_id", "_suffix"])
        .drop("_suffix")
        .with_columns(pl.col("id").cum_count().over("animal_id").alias("slice#"))
    )

    return b6.join(id_map, on=["animal_id", "hippo_reg", "slice#"], how="inner")


def comparison_table(manual: pl.DataFrame, result: pl.DataFrame, manual_col: str) -> pl.DataFrame:
    result_df = result.with_columns(
        pl.col("stimulus").cast(pl.Int64),
        pl.col("file_origin").str.replace(".abf", "", literal=True).alias("id"),
    )
    manual_df = manual.select(["id", "stimulus", "hippo_reg", manual_col]).with_columns(
        pl.col("stimulus").cast(pl.Int64)
    )
    return result_df.join(manual_df, on=["id", "stimulus"], how="inner", validate="1:1")


# --------------------------------------------------------------------- #
# Automated vs manual amplitude agreement plots
# --------------------------------------------------------------------- #

def plot_spaghetti(comp: pl.DataFrame, manual_col: str, xlab: str, ylab: str, title: str, path: str):
    df = comp.select(["id", "stimulus", "amplitude", manual_col]).drop_nulls()
    ids = df["id"].unique(maintain_order=True).sort().to_list()

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab20", max(len(ids), 1))
    for i, id_ in enumerate(ids):
        group = df.filter(pl.col("id") == id_).sort("stimulus")
        ax.plot(
            group[manual_col].abs().to_numpy(), group["amplitude"].to_numpy(),
            color=cmap(i), linewidth=1.2, alpha=0.6,
        )

    # lims = [
    #     min(df[manual_col].abs().min(), df["amplitude"].min()),
    #     max(df[manual_col].abs().max(), df["amplitude"].max()),
    # ]
    # ax.plot(lims, lims, "--", color="0.5")
    #ax.set_xlim(lims); ax.set_ylim(lims)
    x = df[manual_col].abs().to_numpy()
    y = df["amplitude"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, "--", color="black", linewidth=1.3)
    ax.annotate(
        f"slope={slope:.2f}", xy=(0.03, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
    )

    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    _minimal(ax)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _region_mean_sem(comp: pl.DataFrame, manual_col: str) -> pl.DataFrame:
    df = comp.select(["stimulus", "amplitude", manual_col]).drop_nulls().with_columns(
        pl.col(manual_col).abs()
    )
    return (
        df.group_by("stimulus")
        .agg(
            pl.col("amplitude").mean().alias("auto_mean"),
            (pl.col("amplitude").std() / pl.col("amplitude").count().sqrt()).alias("auto_sem"),
            pl.col(manual_col).mean().alias("man_mean"),
            (pl.col(manual_col).std() / pl.col(manual_col).count().sqrt()).alias("man_sem"),
        )
        .sort("stimulus")
    )


def plot_amplitude_agreement(comparisons: dict[str, dict[str, pl.DataFrame]], manual_cols: dict[str, str],
                              labels: dict[str, tuple[str, str]], path: str):
    fig, axes = plt.subplots(ncols=len(comparisons), figsize=(5 * len(comparisons), 5))
    axes = np.atleast_1d(axes)
    for ax, (feature, per_region) in zip(axes, comparisons.items()):
        for region, comp in per_region.items():
            summary = _region_mean_sem(comp, manual_cols[feature])
            ax.errorbar(
                summary["man_mean"].to_numpy(), summary["auto_mean"].to_numpy(),
                xerr=summary["man_sem"].to_numpy(), yerr=summary["auto_sem"].to_numpy(),
                fmt="o", color=REGION_COLORS[region], label=region,
            )
            # x = summary["man_mean"].to_numpy()
            # y = summary["auto_mean"].to_numpy()
            # slope, intercept = np.polyfit(x, y, 1)
            # xs = np.array([x.min(), x.max()])
            # ax.plot(xs, slope * xs + intercept, color=REGION_COLORS[region], linestyle="--", linewidth=1)
            # ax.annotate(
            #     f"{region}: m={slope:.2f}",
            #     xy=(1, 0.95 - 0.08 * list(per_region).index(region)),
            #     xycoords="axes fraction", ha="right",
            #     color=REGION_COLORS[region], fontsize=9,
            # )
        xlab, ylab = labels[feature]
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(feature)
        ax.legend(frameon=False,loc="upper left")
        _minimal(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Automated detection pipeline
# --------------------------------------------------------------------- #

def run_region(recording, brain_region: str, windows: dict) -> dict:
    sub = recording.select_trials(brain_region=brain_region)
    results = {}
    for feature, (window, extra) in windows.items():
        template = build_template_snr(recording=sub, window=window, noise_window=(0.023, 0.025))
        results[feature] = MatchedFilter(**extra).match(sub, template=template)
    return results


def plot_io_stacked(results_by_region: dict[str, dict], features: list[str], path: str):
    fig, axes = plt.subplots(ncols=len(features), figsize=(5 * len(features), 4))
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, features):
        for region, results in results_by_region.items():
            df = results[feature].result.with_columns(pl.col("stimulus").cast(pl.Int64))
            summary = (
                df.group_by("stimulus")
                .agg(
                    pl.col("amplitude").mean().alias("mean"),
                    (pl.col("amplitude").std() / pl.col("amplitude").count().sqrt()).alias("sem"),
                )
                .sort("stimulus")
            )
            ax.errorbar(
                summary["stimulus"].to_numpy(), summary["mean"].to_numpy(), yerr=summary["sem"].to_numpy(),
                color=REGION_COLORS[region], marker="o", linewidth=1.5, label=region,
            )
        ax.set_xlabel("Stimulus (\u03bcA)"); ax.set_ylabel("Amplitude"); ax.set_title(feature)
        ax.legend(frameon=False)
        _minimal(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Detection performance: ROC / decision-threshold / confusion matrices
# NOTE: r2-separation histograms omitted here per request — reuse the
# previous version's histogram/density functions unchanged if needed again.
# --------------------------------------------------------------------- #

def merge_with_consensus(result, truth_df: pl.DataFrame, feature_key: str, brain_region: str) -> pl.DataFrame:
    result_df = result.result.with_columns(
        pl.col("stimulus").cast(pl.Int64),
        pl.col("file_origin").str.replace(".abf", "", literal=True).alias("id"),
    )
    ground_truth = truth_df.filter(
        (pl.col("brain_region") == brain_region) & (pl.col("feature") == feature_key)
    ).select(["id", "stimulus", "detected_consensus"]).with_columns(pl.col("stimulus").cast(pl.Int64))
    return result_df.join(ground_truth, on=["id", "stimulus"], how="inner")


def best_r2_threshold(result, y_true: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds via AlgorithmResult.detect() (not a hand-rolled
    r2 >= t line) and return the balanced-accuracy-maximizing cutoff."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0, 1, 50):
        y_pred = result.detect(float(t)).result["detected"].to_numpy()
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t, best_score


def decision_threshold_plot(result, y_true: np.ndarray, ax, title: str):
    """6 panels x 5 metrics is a lot of near-identical curves. To pare down:
    keep only Balanced Accuracy + F1 (the two that actually drive picking a
    cutoff) and overlay CA1/DG on one axis per feature using REGION_COLORS,
    giving 3 panels total instead of 6."""
    thresholds = np.linspace(0, 1, 50)
    metrics = {"Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    for t in thresholds:
        y_pred = result.detect(float(t)).result["detected"].to_numpy()
        metrics["Accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["Balanced Accuracy"].append(balanced_accuracy_score(y_true, y_pred))
        metrics["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics["F1"].append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(metrics["Balanced Accuracy"]))
    cmap = plt.get_cmap("Dark2")
    for i, (name, vals) in enumerate(metrics.items()):
        ax.plot(thresholds, vals, label=name, color=cmap(i / (len(metrics) - 1)))
    ax.axvline(thresholds[best_idx], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\mathrm{R^2}$ threshold"); ax.set_title(title)
    _minimal(ax)


def confusion_matrices(entries: list[tuple[str, str, object, np.ndarray]], path: str):
    fig, axes = plt.subplots(ncols=len(entries), figsize=(3 * len(entries), 3), sharey=True)
    for ax, (feature, region, result, y_true) in zip(axes, entries):
        t, _ = best_r2_threshold(result, y_true)
        y_pred = result.detect(t).result["detected"].to_numpy()
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, colorbar=False,
            display_labels=["Not Detected", "Detected"], cmap="Blues",
        )
        ax.set_title(f"{feature}\n{region} (t={t:.2f})")
        ax.grid(False); ax.set_xlabel(None)
        _minimal(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def roc_overlay(entries: list[tuple[str, str, np.ndarray, np.ndarray]], path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    for feature, region, y_true, r2_vals in entries:
        fpr, tpr, _ = roc_curve(y_true, r2_vals)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=REGION_COLORS[region], linestyle=FEATURE_STYLES[feature],
                linewidth=1.5, label=f"{feature} {region} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _minimal(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ======================================================================= #
# Run
# ======================================================================= #

long_df = load_classifications(DATADIR)
kappas = fleiss_kappas(long_df)

datafiles = pl.read_excel(
    os.path.join(DATADIR, r"evoked_user_test\evoked\src\evoked\examples", "datafiles_table.xlsx"),
    sheet_name="Sheet1",
).with_columns(pl.col("Data File").str.replace(".abf", "", literal=True).alias("id"))

truth_df = consensus_truth(long_df, datafiles)
manual = load_manual_amplitudes(DATADIR, datafiles)

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
recording = preprocess(load_recording(
    filenames=resolve_filenames(data_path),
    trials="fepsp_trials.tsv",
    epoch=(0.0, 25e-3),
))

WINDOWS = {
    "CA1": {
        "Fiber Volley": ((0.00125, 0.003), {}),
        "fEPSP": ((0.0025, 0.00425), {"derivative_transform": True}),
        "Population Spike": ((0.0045, 0.006), {}),
    },
    "DG": {
        "Fiber Volley": ((0.00125, 0.003), {}),
        "fEPSP": ((0.003, 0.0045), {"derivative_transform": True}),
        "Population Spike": ((0.0045, 0.006), {}),
    },
}

results = {region: run_region(recording, region, WINDOWS[region]) for region in REGIONS}

# -- amplitude agreement (automated vs manual), FV/fEPSP only --
MANUAL_COLS = {"Fiber Volley": "fv_amp", "fEPSP": "epsp_slope_mv/ms"}
LABELS = {
    "Fiber Volley": ("Manual amplitude (mV)", "Automated amplitude (mV)"),
    "fEPSP": ("Manual slope (mV/s)", "Automated slope (mV/s)"),
}

comparisons = {
    feature: {
        region: comparison_table(manual, results[region][feature].result, MANUAL_COLS[feature])
        for region in REGIONS
    }
    for feature in MANUAL_COLS
}

plot_amplitude_agreement(comparisons, MANUAL_COLS, LABELS, "amplitude_agreement_stacked.png")

for feature in MANUAL_COLS:
    for region in REGIONS:
        plot_spaghetti(
            comparisons[feature][region], MANUAL_COLS[feature],
            *LABELS[feature], f"{feature}", f"spaghetti_{feature}_{region}.png",
        )

# -- IO curves, stacked CA1/DG, FV & fEPSP only --
plot_io_stacked(results, ["Fiber Volley", "fEPSP"], "io_curves_stacked.png")

# -- detection performance --
merged = {
    (feature, region): merge_with_consensus(results[region][feature], truth_df, FEATURE_KEY_MAP[feature], region)
    for region in REGIONS for feature in FEATURE_KEY_MAP
}

roc_entries = [
    (feature, region, merged[(feature, region)]["detected_consensus"].to_numpy(),
     merged[(feature, region)]["r2"].to_numpy())
    for region in REGIONS for feature in FEATURE_KEY_MAP
]
roc_overlay(roc_entries, "roc_curves.png")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharey=True)
for row, region in enumerate(REGIONS):
    for col, feature in enumerate(FEATURE_KEY_MAP):
        y_true = merged[(feature, region)]["detected_consensus"].to_numpy()
        decision_threshold_plot(results[region][feature], y_true, axes[row, col], f"{feature} ({region})")
axes[0, 0].set_ylabel("Value"); axes[1, 0].set_ylabel("Value")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()
fig.savefig("decision_threshold_plots.png", dpi=600, bbox_inches="tight")
plt.close(fig)

cm_entries = [
    (feature, region, results[region][feature], merged[(feature, region)]["detected_consensus"].to_numpy())
    for region in REGIONS for feature in FEATURE_KEY_MAP
]
confusion_matrices(cm_entries, "confusion_matrices.png")

save_results_xlsx(results["CA1"], "ca1_results.xlsx")
save_results_xlsx(results["DG"], "dg_results.xlsx")

print("Fleiss kappas:", kappas)