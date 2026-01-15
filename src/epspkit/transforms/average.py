from __future__ import annotations

import numpy as np
import pandas as pd

from epspkit.core.context import RecordingContext


def average_sweeps(context: RecordingContext) -> RecordingContext:
    tidy = context.tidy
    if tidy is None or tidy.empty:
        context.averaged = tidy.copy()
        return context

    group_cols = ["stim_intensity", "time"]
    missing = [col for col in group_cols if col not in tidy.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns for averaging: {missing_str}")

    if "mean" in tidy.columns:
        context.averaged = (
            tidy.sort_values(group_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        return context

    if "value" not in tidy.columns:
        raise ValueError("Expected 'value' column in tidy data.")

    averaged = (
        tidy.groupby(group_cols, sort=False)
        .agg(
            mean=("value", "mean"),
            sem=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
        .sort_values(group_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    context.averaged = averaged
    return context
