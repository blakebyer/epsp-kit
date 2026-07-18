import os
from evoked.base import RecordingResult
import polars_config_meta
from evoked.io_test_copy import load_bulk
from evoked.preprocess import preprocess
from evoked.matched_filter import match_feature
from evoked.lda import match_feature_lda
from evoked.plotting import plot_trace, plot_io_curve, plot_detected
import polars as pl
import matplotlib.pyplot as plt
import numpy as np