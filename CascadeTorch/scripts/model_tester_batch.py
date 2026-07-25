#!/usr/bin/env python3
"""Batch-run pretrained CascadeTorch models and compare outputs on selected ROIs.

Arguments:
- --input-csv: Path to the input test trace CSV (CSV must include 'Frame', 'Time', and
    'Trace_ROI{n}' columns). Default: Dataset/TestTrace_...csv
- --model-yaml: Path to the YAML manifest listing pretrained model names. Default:
    Pretrained_models/available_models_CascadeTorch.yaml
- --output-dir: Directory where plot and CSV outputs will be saved. Default: ./model_comparison_output
- --device: PyTorch device to use ('cuda' or 'cpu'). Defaults to auto-detect if not provided.
- --rois: Comma-separated ROI numbers to plot and compare (e.g., "1,2,3").
- --prefixes: Comma-separated model name prefixes to select models from the YAML manifest.
- --all-downloaded: If set, evaluate all pretrained models that are already downloaded locally.

Capabilities:
- Loads a CSV of fluorescence traces and automatically identifies columns matching
    the pattern 'Trace_ROI{n}'.
- Reads a YAML manifest of pretrained models (with a simple-text fallback parser
    if PyYAML is unavailable).
- Filters model names by provided prefixes or enumerates locally downloaded model
    folders when requested.
- Runs inference for each selected pretrained model via the helper function
    `run_inference(traces, model_name, device)` imported from `model_tester`.
- Saves spike-probability CSV files per model and creates a combined ROI comparison
    plot (`model_comparison_rois.png`) using Matplotlib.
- Validates inputs and raises informative errors for missing files, missing YAML
    entries, or missing ROI columns.

Examples:
- Run default example:
        python scripts/model_tester_batch.py
- Run with custom CSV and specific ROIs:
        python scripts/model_tester_batch.py --input-csv Dataset/my.csv --rois 1,5,7

"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

# Ensure this script can import the local model_tester helper and the cascade2p package.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from model_tester import run_inference

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required to run this comparison script. Install it with pip install matplotlib"
    ) from exc

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_PREFIXES = [
    "Global_EXC_10Hz",
    "Global_EXC_15Hz",
    "GC8_EXC_10Hz",
    "Interneurons_GC8+_7.5Hz",
]
DEFAULT_ROIS = [1, 2, 3, 5, 7, 21]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a batch of pretrained CascadeTorch models on a test trace CSV, "
            "then plot and compare model outputs for selected ROIs."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=pathlib.Path,
        default=pathlib.Path("Dataset/TestTrace_LeftDCN_FOV1_40Hz450um1s_FOOT_5s10isi.csv"),
        help="Path to the input test trace CSV.",
    )
    parser.add_argument(
        "--model-yaml",
        type=pathlib.Path,
        default=pathlib.Path("Pretrained_models/available_models_CascadeTorch.yaml"),
        help="Path to the YAML manifest listing pretrained model names.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("./model_comparison_output"),
        help="Directory where plot and CSV outputs will be saved.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device to use (cuda or cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--rois",
        default=','.join(str(r) for r in DEFAULT_ROIS),
        help="Comma-separated ROI numbers to plot and compare.",
    )
    parser.add_argument(
        "--prefixes",
        default=','.join(DEFAULT_PREFIXES),
        help="Comma-separated model name prefixes to include from the YAML manifest.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        required=True,
        help="Frame rate (Hz) of the input calcium recording (e.g. 40.0). Used to resample traces to each model's trained frame rate before inference.",
    )
    parser.add_argument(
        "--all-downloaded",
        action="store_true",
        help="Evaluate all pretrained models that have already been downloaded locally.",
    )
    return parser.parse_args()


def load_model_names(yaml_path: pathlib.Path) -> list[str]:
    raw = yaml_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected YAML structure in {yaml_path}")
        return list(data.keys())

    model_names = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":"):
            model_names.append(stripped[:-1])
    return model_names


def filter_model_names(model_names: list[str], prefixes: list[str]) -> list[str]:
    selected = [name for name in model_names if any(name.startswith(prefix) for prefix in prefixes)]
    # Preserve order while removing any accidental duplicates from the manifest
    return list(dict.fromkeys(selected))


def load_downloaded_model_names(model_folder: pathlib.Path) -> list[str]:
    if not model_folder.is_dir():
        raise ValueError(f"Model folder does not exist: {model_folder}")

    model_names = [item.name for item in sorted(model_folder.iterdir()) if item.is_dir() and item.name != "__pycache__"]
    if not model_names:
        raise ValueError(f"No downloaded pretrained model folders found in {model_folder}")
    return model_names


def load_trace_data(input_csv: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, list[str], list[int]]:
    df = pd.read_csv(input_csv)
    trace_columns = [col for col in df.columns if re.fullmatch(r"Trace_ROI\d+", col)]
    if not trace_columns:
        raise ValueError(f"No raw Trace_ROI columns found in {input_csv}")

    roi_labels = [int(col.replace("Trace_ROI", "")) for col in trace_columns]
    traces = df[trace_columns].values.T
    return df, traces, trace_columns, roi_labels


def build_roi_index_map(roi_labels: list[int]) -> dict[int, int]:
    return {roi: idx for idx, roi in enumerate(roi_labels)}


def save_predictions_csv(
    output_dir: pathlib.Path,
    model_name: str,
    meta_df: pd.DataFrame,
    trace_columns: list[str],
    spike_data: np.ndarray,
) -> pathlib.Path:
    assert spike_data.shape[0] == len(trace_columns)
    output_df = pd.concat(
        [meta_df.reset_index(drop=True), pd.DataFrame(spike_data.T, columns=[f"Spike_{col.replace('Trace_', '')}" for col in trace_columns])],
        axis=1,
    )
    output_path = output_dir / f"{model_name}_spikes.csv"
    output_df.to_csv(output_path, index=False)
    return output_path


def plot_roi_comparison(
    output_dir: pathlib.Path,
    time: np.ndarray,
    selected_model_outputs: dict[str, np.ndarray],
    roi_indices: dict[int, int],
    rois: list[int],
) -> pathlib.Path:
    num_rois = len(rois)
    fig, axes = plt.subplots(num_rois, 1, sharex=True, figsize=(14, 2.5 * num_rois))
    if num_rois == 1:
        axes = [axes]

    palette = plt.get_cmap("tab10")
    model_names = list(selected_model_outputs.keys())

    for row_idx, roi in enumerate(rois):
        ax = axes[row_idx]
        if roi not in roi_indices:
            ax.text(0.5, 0.5, f"ROI {roi} not found in trace columns", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"ROI {roi}")
            continue

        trace_idx = roi_indices[roi]
        for model_idx, model_name in enumerate(model_names):
            output = selected_model_outputs[model_name]
            ax.plot(
                time,
                output[trace_idx],
                label=model_name,
                color=palette(model_idx % 10),
                linewidth=1.5,
            )

        ax.set_title(f"ROI {roi}")
        ax.set_ylabel("Spike probability")
        ax.grid(True, linestyle="--", alpha=0.35)
        if row_idx == 0:
            ax.legend(loc="upper right", fontsize="small", ncol=1)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    plot_path = output_dir / "model_comparison_rois.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model manifest from {args.model_yaml}")
    model_names = load_model_names(args.model_yaml)

    prefixes = [prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip()]
    if args.all_downloaded:
        selected_model_names = load_downloaded_model_names(args.model_yaml.parent)
    else:
        selected_model_names = filter_model_names(model_names, prefixes)
        if not selected_model_names:
            raise RuntimeError(
                f"No models found using prefixes {prefixes}. Check {args.model_yaml}."
            )

    print(f"Selected {len(selected_model_names)} models:")
    for model_name in selected_model_names:
        print(f"  - {model_name}")

    trace_df, traces, trace_columns, roi_labels = load_trace_data(args.input_csv)
    roi_indices = build_roi_index_map(roi_labels)
    requested_rois = [int(item.strip()) for item in args.rois.split(",") if item.strip()]

    print(f"Loaded traces with {len(trace_columns)} ROIs.")
    print(f"Plotting ROIs: {requested_rois}")

    selected_model_outputs: dict[str, np.ndarray] = {}
    for model_name in selected_model_names:
        print(f"\nRunning inference for {model_name}")
        spike_prob = run_inference(traces, model_name, args.input_fps, args.device)
        selected_model_outputs[model_name] = spike_prob
        save_predictions_csv(args.output_dir, model_name, trace_df[["Frame", "Time"]], trace_columns, spike_prob)
        print(f"Saved spike probabilities for {model_name}")

    plot_path = plot_roi_comparison(
        args.output_dir,
        trace_df["Time"].to_numpy(),
        selected_model_outputs,
        roi_indices,
        requested_rois,
    )

    print(f"Saved comparison plot to {plot_path}")
    print("Batch model comparison complete.")


if __name__ == "__main__":
    main()
