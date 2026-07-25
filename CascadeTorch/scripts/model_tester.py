#!/usr/bin/env python3
"""Test pretrained CascadeTorch models on CSV trace data.

Loads CSV with Frame, Time, Trace_ROI* columns, runs spike inference using
a specified pretrained model, and outputs CSV with Frame, Time, Spike_ROI* columns.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import os

from scipy.signal import resample

# Ensure project root is on sys.path so we can import the local `cascade2p` package
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cascade2p import checks
checks.check_packages()

import numpy as np
import pandas as pd
import torch
from cascade2p import cascade


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spike inference on CSV trace data using pretrained CascadeTorch model."
    )
    parser.add_argument(
        "input_csv",
        type=pathlib.Path,
        help="Path to input file: CSV with Frame/Time/Trace_ROI* or .npy array (n_rois x n_frames).",
    )
    parser.add_argument(
        "--npy",
        action="store_true",
        help="Treat the input file as a .npy NumPy array of shape (n_rois, n_frames).",
    )
    parser.add_argument(
        "model_name",
        help="Name of pretrained model to use (e.g., 'Global_EXC_30Hz_smoothing25ms').",
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=None,
        help="Path to output CSV file. Defaults to input_csv with '_spikes' suffix.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        required=True,
        help="Frame rate (Hz) of the input calcium recording (e.g. 40.0). Used to resample traces to the model's trained frame rate before inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for PyTorch (cuda or cpu). Defaults to auto-detect.",
    )
    return parser.parse_args()


def load_traces_from_csv(csv_path: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load CSV and extract traces as (n_rois, n_frames) array."""
    df = pd.read_csv(csv_path)
    
    # Find trace columns
    trace_cols = [col for col in df.columns if col.startswith('Trace_ROI')]
    if not trace_cols:
        raise ValueError("No Trace_ROI columns found in CSV.")
    
    # Extract traces: shape (n_frames, n_rois)
    traces = df[trace_cols].values.T  # Transpose to (n_rois, n_frames)
    
    return df[['Frame', 'Time']], traces


def load_traces_from_npy(npy_path: pathlib.Path) -> tuple[None, np.ndarray]:
    """Load traces from a .npy file. Returns (None, traces).

    Expects an array of shape (n_rois, n_frames) but will accept other shapes
    and leave them as-is (the downstream code will log the shape).
    """
    arr = np.load(npy_path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Loaded object from {npy_path} is not a NumPy array")
    return None, arr


def _get_model_fps(config_file: pathlib.Path) -> float:
    """Read the trained sampling rate from a model's config.yaml."""
    from cascade2p import config as cascade_config
    cfg = cascade_config.read_config(str(config_file))
    return float(cfg["sampling_rate"])


def run_inference(
    traces: np.ndarray,
    model_name: str,
    input_fps: float,
    selected_device: str | None = None,
    debug_dir: pathlib.Path | None = None,
) -> np.ndarray:
    """Run spike inference using CascadeTorch.

    Traces are resampled from input_fps to the model's trained frame rate before
    inference, then resampled back to the original frame count on return so the
    output shape always matches the input shape.

    If debug_dir is provided, the resampled input actually fed into CASCADE is
    saved as '<model_name>_input_trace.csv' inside that directory.
    """

    # device selection (only cpu available on mac)
    if selected_device is None:
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = str(selected_device).lower()

    try:
        device = torch.device(selected_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"Invalid device '{selected_device}'. Use 'cuda' or 'cpu'."
        ) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use '--device cpu' or install CUDA-enabled PyTorch."
        )

    print(f"Using device: {device}")

    # start-of-process log
    print(f"Model: {model_name}")
    print(f"Input shape: {traces.shape} (neurons x timepoints)")

    # only download model if model does not exist locally
    model_folder = "Pretrained_models"
    model_path = pathlib.Path(model_folder) / model_name
    config_file = model_path / "config.yaml"
    if config_file.is_file():
        print(f"Using cached pretrained model at {model_path}")
    else:
        cascade.download_model(model_name, verbose=1)

    # Resample traces to model frame rate if needed
    model_fps = _get_model_fps(config_file)
    n_original_frames = traces.shape[1]
    if input_fps != model_fps:
        n_resampled = round(n_original_frames * model_fps / input_fps)
        print(f"Resampling traces: {input_fps} Hz → {model_fps} Hz ({n_original_frames} → {n_resampled} frames)")
        traces_for_inference = resample(traces, n_resampled, axis=1)
    else:
        traces_for_inference = traces

    # Debug export: save the resampled trace that is actually fed into CASCADE
    if debug_dir is not None:
        debug_dir = pathlib.Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        n_rois = traces_for_inference.shape[0]
        debug_df = pd.DataFrame(
            traces_for_inference.T,
            columns=[f"Trace_ROI{i+1}" for i in range(n_rois)],
        )
        debug_df.insert(0, "Frame", np.arange(traces_for_inference.shape[1]))
        debug_df.insert(1, "Time", debug_df["Frame"] / model_fps)
        debug_path = debug_dir / f"{model_name}_input_trace.csv"
        debug_df.to_csv(debug_path, index=False)
        print(f"Debug trace saved to: {debug_path}")

    # Run prediction
    spike_prob = cascade.predict(model_name, traces_for_inference, device=device)

    # Resample output back to original frame count.
    # Replace NaN edge-padding with 0 first: scipy.signal.resample is FFT-based
    # and any NaN in the input propagates to the entire output.
    if input_fps != model_fps:
        spike_prob = np.nan_to_num(spike_prob, nan=0.0)
        print(f"Resampling output back to {input_fps} Hz ({spike_prob.shape[1]} → {n_original_frames} frames)")
        spike_prob = resample(spike_prob, n_original_frames, axis=1)
        spike_prob = np.clip(spike_prob, 0, None)

    print(f"Output shape: {spike_prob.shape}")
    return spike_prob


def save_spikes_to_csv(meta_df: pd.DataFrame, spike_prob: np.ndarray, output_path: pathlib.Path) -> None:
    """Save spike probabilities to CSV with same structure as input."""
    # Transpose back to (n_frames, n_rois)
    spikes_df = pd.DataFrame(spike_prob.T)
    # Try to infer column names from meta_df trace columns when available
    trace_cols = [col for col in meta_df.columns if col.startswith('Trace_')]
    if trace_cols and spikes_df.shape[1] == len(trace_cols):
        spikes_df.columns = [f"Spike_{col.replace('Trace_', '')}" for col in trace_cols]
    
    # Combine with Frame and Time
    output_df = pd.concat([meta_df, spikes_df], axis=1)
    output_df.to_csv(output_path, index=False)


def save_spikes_to_npy(spike_prob: np.ndarray, output_path: pathlib.Path) -> None:
    """Save spike probability array to a .npy file."""
    # Ensure parent exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use .npy extension if not provided
    if output_path.suffix != ".npy":
        output_path = output_path.with_suffix(".npy")
    np.save(output_path, spike_prob, allow_pickle=False)


def main() -> None:
    args = parse_arguments()
    input_path = args.input_csv.resolve()

    # Detect .npy inputs either via flag or file extension
    is_npy = bool(args.npy) or input_path.suffix.lower() == ".npy"

    if args.output_csv is None:
        # Always produce CSV output regardless of input type
        output_path = input_path.parent / f"{input_path.stem}_spikes.csv"
    else:
        output_path = args.output_csv.resolve()

    print(f"Loading traces from: {input_path}")
    if is_npy:
        meta_df, traces = load_traces_from_npy(input_path)
        # Ensure traces shape is (n_rois, n_frames)
        if traces.ndim == 1:
            traces = traces.reshape((1, -1))
        if traces.ndim != 2:
            raise ValueError(f"Unsupported array shape for traces: {traces.shape}")
        n_rois, n_frames = traces.shape
        # If loader didn't provide meta_df, synthesize one with Frame, Time, and Trace_ROI columns
        if meta_df is None:
            meta_df = pd.DataFrame({
                'Frame': np.arange(n_frames),
                'Time': np.arange(n_frames),
            })
            for i in range(n_rois):
                meta_df[f'Trace_ROI{i+1}'] = np.nan
    else:
        meta_df, traces = load_traces_from_csv(input_path)

    spike_prob = run_inference(traces, args.model_name, args.input_fps, args.device, debug_dir=output_path.parent)

    print(f"Saving spikes to: {output_path}")
    # Always save CSV output
    save_spikes_to_csv(meta_df, spike_prob, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()