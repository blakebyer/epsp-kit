from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from epspkit.core.config import (
    FeatureConfig,
    PipelineConfig,
    SmoothingConfig,
    TransformConfig,
    VizConfig,
)
from epspkit.core.context import RecordingContext
from epspkit.features.epsp import EPSPFeature
from epspkit.features.fiber_volley import FiberVolleyFeature
from epspkit.features.pop_spike import PopSpikeFeature
from epspkit.io.read_write import load_abf_to_context, save_context_to_xlsx
from epspkit.transforms.average import average_sweeps
from epspkit.transforms.baseline import baseline_correction
from epspkit.transforms.stim_artifact import (
    crop_stim_artifact,
    template_subtract_stim_artifact,
)
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

TRANSFORM_FUNCS: dict[str, Callable[..., RecordingContext]] = {
    "baseline_correction": baseline_correction,
    "crop_stim_artifact": crop_stim_artifact,
    "template_subtract_stim_artifact": template_subtract_stim_artifact,
    "average_sweeps": average_sweeps,
}

def effective_smoothing(
    config: FeatureConfig | VizConfig,
    global_smoothing: SmoothingConfig,
) -> SmoothingConfig:
    return config.smoothing if config.smoothing is not None else global_smoothing


def _build_component(
    config: FeatureConfig | VizConfig,
    registry: dict[str, type],
    kind: str,
    global_smoothing: SmoothingConfig,
):
    component_cls = registry.get(config.name)
    if component_cls is None:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} '{config.name}'. Available: {options}")
    smoothing = effective_smoothing(config, global_smoothing)
    return component_cls(config, smoothing)


def build_feature(config: FeatureConfig, global_smoothing: SmoothingConfig):
    return _build_component(config, FEATURE_CLASSES, "feature", global_smoothing)


def build_plot(config: VizConfig, global_smoothing: SmoothingConfig):
    return _build_component(config, PLOT_CLASSES, "plot", global_smoothing)

def resolve_plot_smoothing(pipeline_config: PipelineConfig) -> None:
    if not pipeline_config.plots:
        return
    global_cfg = pipeline_config.global_smoothing
    for plot_cfg in pipeline_config.plots:
        if plot_cfg.smoothing is None:
            plot_cfg.smoothing = SmoothingConfig(
                method=global_cfg.method,
                window_size=global_cfg.window_size,
                polyorder=global_cfg.polyorder,
                cutoff=global_cfg.cutoff,
                order=global_cfg.order,
            )
def resolve_feature_smoothing(pipeline_config: PipelineConfig) -> None:
    if not pipeline_config.features:
        return
    global_cfg = pipeline_config.global_smoothing
    for feature_cfg in pipeline_config.features:
        if feature_cfg.smoothing is None:
            feature_cfg.smoothing = SmoothingConfig(
                method=global_cfg.method,
                window_size=global_cfg.window_size,
                polyorder=global_cfg.polyorder,
                cutoff=global_cfg.cutoff,
                order=global_cfg.order,
            )

def build_transform(
    config: TransformConfig,
) -> tuple[Callable[..., RecordingContext], dict[str, Any]]:
    transform_fn = TRANSFORM_FUNCS.get(config.name)
    if transform_fn is None:
        options = ", ".join(sorted(TRANSFORM_FUNCS))
        raise ValueError(f"Unknown transform '{config.name}'. Available: {options}")
    return transform_fn, config.params or {}


def run_context(
    context: RecordingContext,
    pipeline_config: PipelineConfig,
    output_stem: str | None = None,
) -> RecordingContext:
    if pipeline_config.io.write_plots and not pipeline_config.io.output_path:
        raise ValueError(
            "PipelineConfig.io.output_path is required when write_plots is True."
        )

    context.pipeline_cfg = pipeline_config
    resolve_plot_smoothing(pipeline_config)
    resolve_feature_smoothing(pipeline_config)

    for transform_cfg in pipeline_config.transforms:
        transform_fn, params = build_transform(transform_cfg)
        result = transform_fn(context, **params)
        if result is not None and not isinstance(result, RecordingContext):
            raise TypeError("Transforms must return RecordingContext or None.")
        if isinstance(result, RecordingContext):
            context = result

    if context.averaged is None or context.averaged.empty:
        average_sweeps(context)

    for feature_cfg in pipeline_config.features:
        feature = build_feature(feature_cfg, pipeline_config.global_smoothing)
        context = feature.run(context)

    plots = pipeline_config.plots
    if plots and (pipeline_config.io.render_plots or pipeline_config.io.write_plots):
        for plot_cfg in plots:
            plot = build_plot(plot_cfg, pipeline_config.global_smoothing)
            if pipeline_config.io.render_plots:
                plot.render(context)
            if pipeline_config.io.write_plots:
                plot.save(context, pipeline_config.io.output_path, output_stem=output_stem)

    return context


def run_pipeline(
    pipeline_config: PipelineConfig,
) -> list[RecordingContext]:
    if not pipeline_config.io.input_paths:
        raise ValueError("PipelineConfig.io.input_paths is required.")
    if not pipeline_config.io.stim_intensities:
        raise ValueError("PipelineConfig.io.stim_intensities is required.")
    if (
        pipeline_config.io.write_results
        or pipeline_config.io.write_plots
    ) and not pipeline_config.io.output_path:
        raise ValueError(
            "PipelineConfig.io.output_path is required when write_results or "
            "write_plots is True."
        )

    def build_output_stems(input_paths: Sequence[str]) -> list[str]:
        stems = [Path(path).stem for path in input_paths]
        if len(stems) > 1:
            stems = [f"{stem}_{idx + 1}" for idx, stem in enumerate(stems)]
        return stems

    output_stems = build_output_stems(pipeline_config.io.input_paths)
    contexts = []
    for idx, path in enumerate(pipeline_config.io.input_paths):
        context = load_abf_to_context(
            file_path=path,
            stim_intensities=list(pipeline_config.io.stim_intensities),
            repnum=pipeline_config.io.repnum,
        )
        context.metadata.update(pipeline_config.io.metadata or {})
        output_stem = output_stems[idx] if idx < len(output_stems) else None
        contexts.append(run_context(context, pipeline_config, output_stem=output_stem))

    if pipeline_config.io.write_results and pipeline_config.io.output_path:
        write_results(contexts, pipeline_config.io.output_path, output_stems)

    return contexts


def write_results(
    contexts: Sequence[RecordingContext],
    output_path: Path | str,
    output_stems: Sequence[str] | None = None,
) -> None:
    output = Path(output_path)
    stems = list(output_stems or [])
    if not stems:
        stems = [f"recording_{idx + 1}" for idx in range(len(contexts))]

    if output.suffix and len(contexts) == 1:
        save_context_to_xlsx(contexts[0], str(output), output_stem=stems[0])
        return

    output_dir = output if not output.suffix else output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, context in enumerate(contexts):
        stem = stems[idx] if idx < len(stems) else f"recording_{idx + 1}"
        save_context_to_xlsx(context, str(output_dir), output_stem=stem)
