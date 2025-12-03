# epspkit/core/context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import pandas as pd


@dataclass
class RecordingContext:
    """
    Container for all information associated with one recording.

    Attributes
    ----------
    tidy : pd.DataFrame
        Full tidy time-series (columns: time, voltage, sweep, intensity, etc.)
    averaged : pd.DataFrame
        Sweep-averaged signal, usually one row per timepoint.
    fs : float
        Sampling rate in Hz.
    meta : dict[str, Any]
        Origin metadata (file path, experiment ID, stim times, etc.)
    results : dict[str, dict]
        Per-feature outputs: {'epsp': {...}, 'fiber_volley': {...}}
    pipeline_cfg : optional
        Filled in by the pipeline—you can attach PipelineConfig here.
    """

    tidy: pd.DataFrame
    averaged: pd.DataFrame
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_cfg: Any | None = None  # attached by pipeline

    def add_result(self, feature_name: str, result: Dict[str, Any]):
        """Add or update results from a feature analyzer."""
        self.results[feature_name] = result

    def get_result(self, feature_name: str):
        return self.results.get(feature_name, None)
