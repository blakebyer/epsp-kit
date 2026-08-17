from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from evoked.base import RecordingResult, RecordingConfig, TracePlot, MultiChannelPlot, IOPlot, FitPlot, DetectedPlot, AllFilesPlot
from evoked.io import resolve_filenames, load_bulk, load_config, save_results_xlsx
from evoked.preprocessing import preprocess
from evoked.matched_filter import match_feature
from evoked.plotting import plot_trace, plot_multichannel, plot_io_curve, plot_fit, plot_detected, plot_all_files


def clean_config(model: BaseModel) -> dict:
    """Return only the fields the user actually set on `model`, so spreading
    the result into a function call lets that function's own default apply
    to anything left unset, instead of an explicit None overriding it."""
    return {name: getattr(model, name) for name in model.model_fields_set}

def describe_config(model_cls: type[BaseModel] = RecordingConfig) -> str:
    _EFFECTIVELY_REQUIRED = {
        "Recording": {"order", "stimulus"},
        "Feature": {"window", "noise_window"},
    }
    _PLOT_TYPE_CLASSES = {
        "trace": TracePlot, "multichannel": MultiChannelPlot, "io": IOPlot, "fit": FitPlot,
        "detected": DetectedPlot, "allfiles": AllFilesPlot,
    }

    def make_resolver(defs: dict):
        def resolve(node: dict) -> dict:
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            if "anyOf" in node:
                non_null = [b for b in node["anyOf"] if b.get("type") != "null"]
                if len(non_null) == 1:
                    return resolve(non_null[0])
                return node  # genuine union (e.g. tuple[float,float] | float) -- keep intact
            return node
        return resolve

    def type_str(node: dict, resolve) -> str:
        node = resolve(node)
        if "anyOf" in node:
            parts = []
            for branch in node["anyOf"]:
                branch = resolve(branch)
                if branch.get("type") == "null":
                    continue
                s = type_str(branch, resolve)
                if s not in parts:
                    parts.append(s)
            return " | ".join(parts)
        if "enum" in node:
            return " | ".join(str(v) for v in node["enum"])
        if node.get("type") == "array":
            if "prefixItems" in node:  # fixed-length tuple, e.g. tuple[float, float]
                items = [type_str(item, resolve) for item in node["prefixItems"]]
                return f"tuple[{', '.join(items)}]"
            if "items" in node:  # variable-length list, e.g. list[str]
                return f"list[{type_str(node['items'], resolve)}]"
            return "list"
        return node.get("type", "string")

    def render(node: dict, resolve, indent: int = 1) -> list[str]:
        pad = "  " * indent
        required = set(node.get("required", []))
        overrides = _EFFECTIVELY_REQUIRED.get(node.get("title"), set())
        lines = []

        for name, field_schema in node.get("properties", {}).items():
            resolved = resolve(field_schema)
            tag = "required" if (name in required or name in overrides) else "optional"

            if name == "plots":
                lines.append(f"{pad}{name}:  # {tag} -- mapping of plot type")
                for plot_type, plot_cls in _PLOT_TYPE_CLASSES.items():
                    plot_schema = plot_cls.model_json_schema()
                    plot_resolve = make_resolver(plot_schema.get("$defs", {}))
                    lines.append(f"{pad}  {plot_type}:")
                    lines.extend(render(plot_schema, plot_resolve, indent + 2))
                continue

            if resolved.get("type") == "object" and "properties" in resolved:
                lines.append(f"{pad}{name}:  # {tag}")
                lines.extend(render(resolved, resolve, indent + 1))
            elif resolved.get("type") == "object" and "additionalProperties" in resolved:
                raw_value_schema = resolved["additionalProperties"]
                lines.append(f"{pad}{name}:  # {tag} -- mapping")
                lines.append(f"{pad}  <name>:")
                if isinstance(raw_value_schema, dict):
                    value_schema = resolve(raw_value_schema)
                    if "properties" in value_schema:
                        lines.extend(render(value_schema, resolve, indent + 2))
                    else:
                        lines.append(f"{pad}    <{type_str(value_schema, resolve)}>")
                else:
                    lines.append(f"{pad}    <any>")
            else:
                lines.append(f"{pad}{name}: <{type_str(resolved, resolve)}>  # {tag}")

        return lines

    schema = model_cls.model_json_schema()
    resolve = make_resolver(schema.get("$defs", {}))
    return "\n".join(render(resolve(schema), resolve, indent=0))


def run_feature(intermediate, feature_name, feature_config):
    kwargs = clean_config(feature_config)
    return feature_name, match_feature(intermediate=intermediate, **kwargs)


def run_analysis(intermediate, config) -> RecordingResult:
    if config.analysis is None or not config.analysis.features:
        raise ValueError("No features configured under 'analysis.features' -- nothing to analyze.")

    recording_result = RecordingResult()
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_feature, intermediate, name, feature_config)
            for name, feature_config in config.analysis.features.items()
        ]
        for future in futures:
            name, feature_result = future.result()
            recording_result.add(name, feature_result)

    return recording_result


def run_plots(intermediate, recording_result, config) -> dict:
    if config.plotting is None or not config.plotting.plots:
        return {}

    figures = {}
    for plot_type, plot_configs in config.plotting.plots.items():
        for i, plot_config in enumerate(plot_configs):
            if plot_config is None:
                raise ValueError(f"Plot '{plot_type}' has no configuration.")

            kwargs = clean_config(plot_config)
            key = plot_type if len(plot_configs) == 1 else f"{plot_type}_{i}"

            if plot_type == "trace":
                figures[key] = plot_trace(intermediate=intermediate, recording_result=recording_result, **kwargs)
            elif plot_type == "multichannel":
                figures[key] = plot_multichannel(intermediate=intermediate, **kwargs)
            elif plot_type == "io":
                figures[key] = plot_io_curve(recording_result=recording_result, **kwargs)
            elif plot_type == "fit":
                figures[key] = plot_fit(intermediate=intermediate, recording_result=recording_result, **kwargs)
            elif plot_type == "detected":
                figures[key] = plot_detected(recording_result=recording_result, **kwargs)
            elif plot_type == "allfiles":
                figures[key] = plot_all_files(intermediate=intermediate, **kwargs)

    return figures


def analyze(data_dir: str, config_path: str) -> tuple[RecordingResult, dict, RecordingConfig]:
    config = load_config(config_path)
    filenames = resolve_filenames(data_dir)

    recording = load_bulk(filenames, config_path)
    intermediate = preprocess(recording, config.analysis.preprocess)

    recording_result = run_analysis(intermediate, config)
    figures = run_plots(intermediate, recording_result, config)

    return recording_result, figures, config


def main():
    parser = argparse.ArgumentParser(
        prog="evoked.analyze",
        description="Run the evoked analysis pipeline end-to-end from a YAML config: "
                     "load raw recordings, preprocess, run feature detection, render plots, and save results.",
    )
    parser.add_argument("--data", metavar="PATH", help="Directory containing data files")
    parser.add_argument("--config", metavar="PATH", help="Path to YAML config file")
    parser.add_argument("--output", default=".", metavar="PATH", help="Directory to save results into (default: current directory)")
    parser.add_argument("--describe-config", action="store_true", help="Print the expected config.yml structure and exit")
    args = parser.parse_args()

    if args.describe_config:
        print(describe_config())
        return

    if not args.data or not args.config:
        parser.error("--data and --config are required unless --describe-config is used")

    os.makedirs(args.output, exist_ok=True)

    recording_result, figures, config = analyze(args.data, args.config)
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", config.experiment.name)

    for name, feature_result in recording_result.results.items():
        n_detected = feature_result.result["detected"].sum()
        n_total = feature_result.result.height
        print(f"{name}: {n_detected}/{n_total} trials detected")

    for plot_type, fig_ax in figures.items():
        fig, _ = fig_ax
        out_path = os.path.join(args.output, f"{safe_name}_{plot_type}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved {plot_type} plot to {out_path}")
    
    
    results_path = os.path.join(args.output, f"{safe_name}.xlsx")
    save_results_xlsx(recording_result, results_path)
    print(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
