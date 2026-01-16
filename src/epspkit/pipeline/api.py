from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from epspkit.core.config import FeatureConfig, PipelineConfig, SmoothingConfig, VizConfig
from epspkit.core.context import RecordingContext
from epspkit.features.epsp import EPSPFeature
from epspkit.features.fiber_volley import FiberVolleyFeature
from epspkit.features.pop_spike import PopSpikeFeature
from epspkit.io.read_write import load_abf_to_context, save_context_to_json
from epspkit.transforms.average import average_sweeps
from epspkit.viz.annotated import AnnotatedPlot
from epspkit.viz.derivative import DerivativePlot
from epspkit.viz.input_output import IOPlot
from epspkit.viz.sweep import SweepPlot

FEATURE_CLASSES: dict[str, type] = {
    "fiber_volley": FiberVolleyFeature,
    "epsp": EPSPFeature,
    "pop_spike": PopSpikeFeature,
}

PLOT_CLASSES: dict[str, type] = {
    "sweep": SweepPlot,
    "derivative": DerivativePlot,
    "annotated": AnnotatedPlot,
    "input_output": IOPlot,
}


def _effective_smoothing(
    config: FeatureConfig | VizConfig,
    global_smoothing: SmoothingConfig,
) -> SmoothingConfig:
    if config.smoothing and config.smoothing.method != "none":
        return config.smoothing
    return global_smoothing


def _build_feature(config: FeatureConfig, global_smoothing: SmoothingConfig):
    feature_cls = FEATURE_CLASSES.get(config.name)
    if feature_cls is None:
        options = ", ".join(sorted(FEATURE_CLASSES))
        raise ValueError(f"Unknown feature '{config.name}'. Available: {options}")
    smoothing = _effective_smoothing(config, global_smoothing)
    return feature_cls(config, smoothing)


def _build_plot(config: VizConfig, global_smoothing: SmoothingConfig):
    plot_cls = PLOT_CLASSES.get(config.name)
    if plot_cls is None:
        options = ", ".join(sorted(PLOT_CLASSES))
        raise ValueError(f"Unknown plot '{config.name}'. Available: {options}")
    smoothing = _effective_smoothing(config, global_smoothing)
    return plot_cls(config, smoothing)


def run_context(
    context: RecordingContext,
    pipeline_config: PipelineConfig,
    transforms: Sequence[Callable[[RecordingContext], RecordingContext]] | None = None,
    plots: Sequence[VizConfig] | None = None,
) -> RecordingContext:
    context.pipeline_cfg = pipeline_config

    for transform in transforms or []:
        context = transform(context)

    if context.averaged is None or context.averaged.empty:
        average_sweeps(context)

    for feature_cfg in pipeline_config.features:
        feature = _build_feature(feature_cfg, pipeline_config.global_smoothing)
        context = feature.run(context)

    if pipeline_config.io.write_plots and plots:
        for plot_cfg in plots:
            plot = _build_plot(plot_cfg, pipeline_config.global_smoothing)
            plot.render(context)

    return context


def run_pipeline(
    pipeline_config: PipelineConfig,
    transforms: Sequence[Callable[[RecordingContext], RecordingContext]] | None = None,
    plots: Sequence[VizConfig] | None = None,
) -> list[RecordingContext]:
    if not pipeline_config.io.input_paths:
        raise ValueError("PipelineConfig.io.input_paths is required.")
    if not pipeline_config.io.stim_intensities:
        raise ValueError("PipelineConfig.io.stim_intensities is required.")

    contexts = []
    for path in pipeline_config.io.input_paths:
        context = load_abf_to_context(
            file_path=path,
            stim_intensities=list(pipeline_config.io.stim_intensities),
            repnum=pipeline_config.io.repnum,
        )
        context.meta["input_path"] = path
        contexts.append(run_context(context, pipeline_config, transforms, plots))

    if pipeline_config.io.write_results and pipeline_config.io.output_path:
        _write_results(contexts, pipeline_config.io.output_path)

    return contexts


def _write_results(contexts: Sequence[RecordingContext], output_path: Path | str) -> None:
    output = Path(output_path)
    if len(contexts) == 1 and output.suffix:
        save_context_to_json(contexts[0], str(output))
        return

    output.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        input_path = context.meta.get("input_path", "recording")
        stem = Path(input_path).stem
        save_context_to_json(context, str(output / f"{stem}_results.json"))
