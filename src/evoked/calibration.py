from __future__ import annotations
import argparse
import xlsxwriter
import os
import polars as pl
import numpy as np
from evoked.base import TruthData, RecordingResult
from evoked.io import load_results_json, load_results_xlsx
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from typing import Literal
from pandera.typing.polars import DataFrame

def load_truth(truth_path: str):
    if not os.path.exists(truth_path):
        raise FileNotFoundError(
            f"Truth data not found at {truth_path}"
        )
    ext = os.path.splitext(truth_path)[1].lower()
    if ext == ".csv":
        truth_data = pl.read_csv(truth_path)
    elif ext == ".tsv":
        truth_data = pl.read_csv(truth_path, separator="\t")
    elif ext == ".xlsx":
        truth_data = pl.read_excel(truth_path)  # defaults to loading first sheet
    else:
        raise ValueError(f"File extension must be one of: .csv, .tsv, or .xlsx (got {ext})")
        
    return TruthData.validate(truth_data)

MetricType = Literal["f1", "accuracy", "precision", "recall", "balanced accuracy"]
def calibrate(
        truth: DataFrame[TruthData], 
        pred: RecordingResult, 
        feature: str,
        metric: MetricType):
    key_cols = ["id", "channel", "stimulus"]
    pred = pred.get(feature).result.sort(key_cols).drop("detected")
    truth = truth.filter(pl.col("feature") == feature).sort(key_cols)
    
    joined = pred.join(truth, how="left",on=key_cols)

    missing = joined.filter(pl.col("detected").is_null())
    if missing.height > 0:
        raise ValueError(
            f"{missing.height} pred row(s) for feature={feature} have no matching "
            f"truth row on {key_cols}:\n{missing.select(key_cols)}"
        )
    
    y_true = joined['detected'].cast(pl.Int32).to_numpy()
    metrics = []
    r2_thresholds = np.linspace(0, 1, 50)
    for t in r2_thresholds:
        y_pred = (joined['r2'] >= t).cast(pl.Int32).to_numpy()
        if metric == "f1":
            metrics.append(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "accuracy":
            metrics.append(accuracy_score(y_true, y_pred))
        elif metric == "precision":
            metrics.append(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            metrics.append(recall_score(y_true, y_pred, zero_division=0))
        elif metric == "balanced accuracy":
            metrics.append(balanced_accuracy_score(y_true, y_pred))
        else: 
            raise ValueError(f"Metric must be one of: f1, accuracy, precision, recall, balanced accuracy (got {metric})")

    best_k = np.nanargmax(metrics)
    best_r2 = r2_thresholds[best_k]
    best_metric = metrics[best_k]

    print(f"Best R^2 threshold for {feature}={best_r2:.3f} at {metric}={best_metric:.3f}")
    return pl.DataFrame({
        "r2_thresholds":r2_thresholds,
        f"{metric}":metrics,
    })

def calibrate_all(
        truth: DataFrame[TruthData],
        pred: RecordingResult,
        metric: MetricType) -> dict[str, pl.DataFrame]:
    results = {}
    for feature in pred.results:
        results[feature] = calibrate(truth, pred, feature, metric)
    return results

def main():
    parser = argparse.ArgumentParser(
        prog="evoked.calibrate",
        description="Compute hyperparameter calibration based on truth labels.",
    )
    parser.add_argument("--truth", metavar="PATH", help="Path to truth labels in tabular format")
    parser.add_argument("--pred", metavar="PATH", help="Path to recording results in Excel or JSON format")
    parser.add_argument("--metric", metavar="str", default="balanced accuracy", help="Metric on which to base the calibration (default: balanced accuracy)")
    parser.add_argument("--output", metavar="PATH", help="Path to save calibration results into")
    args = parser.parse_args()

    if not args.truth or not args.pred:
        parser.error("--truth and --pred are required")

    truth = load_truth(args.truth)
    pred = load_results_xlsx(args.pred) if os.path.splitext(args.pred)[1] == ".xlsx" else load_results_json(args.pred)

    if truth is None or pred is None:
        raise ValueError("truth/pred do not exist")
    
    safe_metric = str(args.metric).lower() # metric is case-insensitive, feature is not
    res = calibrate_all(truth, pred, safe_metric)

    if args.output:
        workbook = xlsxwriter.Workbook(args.output)
        for feature, df in res.items():
            df.write_excel(workbook=workbook, worksheet=feature[:31])  # Excel sheet-name limit
        workbook.close()


if __name__ == "__main__":
    main()


