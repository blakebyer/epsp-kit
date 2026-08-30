from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from evoked.base import RecordingData, RecordingResult, BaseAlgorithm
from evoked.config import RecordingConfig, PreprocessConfig
from evoked.io import resolve_filenames, load_recording, load_config, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.visualization import TracePlot, MultiChannelPlot, IOPlot, FitPlot, DetectedPlot, AllFilesPlot


def run_feature(
    recording: RecordingData,
    feature_name: str,
    algorithm: BaseAlgorithm
    ):
    return feature_name, algorithm.match(recording)

def run_analysis(recording: RecordingData, config: RecordingConfig) -> RecordingResult:
    if not config.analysis.features:
        raise ValueError("No features configured under 'analysis.features'")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_feature, recording, name, algorithm)
            for name, algorithm in config.analysis.features.items()
        ]

    return dict(future.result() for future in futures)


def run_plot(plot, recording: RecordingData, results: RecordingResult) -> dict:
    if isinstance(
        plot,
        (TracePlot, IOPlot, FitPlot),
    ):
        return plot.plot(
            recording=recording,
            results=results,
        )

    if isinstance(
        plot,
        (DetectedPlot),
    ):
        return plot.plot(
            results=results,
        )

    if isinstance(
        plot, 
        (MultiChannelPlot, AllFilesPlot),
    ):
        return plot.plot(
            recording=recording,    
        )
    
    raise TypeError(
        f"Unsupported plot type: "
        f"{type(plot).__name__}"
    )

def run_plots(
        recording: RecordingData,
        results: RecordingResult,
        config: RecordingConfig,
    ):
    if (
        config.plotting is None
        or not config.plotting.plots
    ):
        return {}

    figures = {}

    for i, plot in enumerate(
        config.plotting.plots
    ):
        plotted = run_plot(
            plot,
            recording,
            results,
        )

        figures[f"{plot.type}_{i}"] = plotted.figure

    return figures


def analyze(data_path: str, config_path: str, trials_path: str) -> tuple[RecordingResult, dict, RecordingConfig]:
    config = load_config(config_path)
    filenames = resolve_filenames(data_path)

    recording = load_recording(filenames, trials_path, config.analysis.epoch, config.analysis.event_label)

    preprocess_config = (
        PreprocessConfig.model_validate(
            config.analysis.preprocess
            )
        )

    recording = preprocess(
        recording=recording,
        params=preprocess_config.model_dump(),
    )

    results = run_analysis(
        recording=recording, 
        config=config,
    )

    figures = run_plots(
        recording=recording, 
        results=results, 
        config=config,
    )

    return results, figures, config


def main():
    parser = argparse.ArgumentParser(
        prog="evoked-analyze",
        description="Run the evoked analysis pipeline end-to-end from YAML config and Trials TSV.",
    )
    parser.add_argument("--data", metavar="PATH", help="Directory containing data files")
    parser.add_argument("--config", metavar="PATH", help="Path to YAML config file")
    parser.add_argument("--trials", metavar="PATH", help="Path to Trials TSV")
    parser.add_argument("--output", default=".", metavar="PATH", help="Directory to save results (default: current directory)")
    args = parser.parse_args()


    if not args.data or not args.config:
        parser.error("--data and --config are required")

    os.makedirs(args.output, exist_ok=True)

    results, figures, config = analyze(args.data, args.config, args.trials)
    experiment_name = re.sub(r'[<>:"/\\|?*]', "_", config.experiment.name)

    for name, fig in figures.items():
        fig_path = os.path.join(args.output, f"{experiment_name}_{name}.png")
        fig.savefig(fig_path, dpi=600, bbox_inches="tight")
        print(f"Saved {name} plot to {fig_path}")
    
    
    results_path = os.path.join(args.output, f"{experiment_name}.xlsx")
    save_results_xlsx(results, results_path)
    print(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
