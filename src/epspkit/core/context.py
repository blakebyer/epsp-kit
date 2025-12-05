from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from epspkit.core.config import PipelineConfig


@dataclass
class RecordingContext:
    """
    Container for all information associated with one recording.

    Attributes
    ----------
    tidy : pd.DataFrame
        Full tidy time-series (columns: time, voltage, sweep, intensity, etc.).
    averaged : pd.DataFrame
        Sweep-averaged signal, usually one row per timepoint.
    fs : float
        Sampling rate in Hz.
    meta : dict[str, Any]
        Origin metadata (file path, experiment ID, stim times, etc.).
    results : dict[str, dict[str, Any]]
        Per-feature outputs: {'epsp': {...}, 'fiber_volley': {...}}.
    pipeline_cfg : PipelineConfig | None
        Optionally attached by the pipeline to make config discoverable
        to features/utilities that only receive a RecordingContext.
    """

    tidy: pd.DataFrame
    averaged: pd.DataFrame
    fs: float
    meta: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    pipeline_cfg: PipelineConfig | None = None  # attached by pipeline

    def add_result(self, feature_name: str, result: dict[str, Any]) -> None:
        """Add or update results from a feature analyzer."""
        self.results[feature_name] = result

    def get_result(self, feature_name: str) -> dict[str, Any] | None:
        """Retrieve results for a given feature, or None if missing."""
        return self.results.get(feature_name, None)