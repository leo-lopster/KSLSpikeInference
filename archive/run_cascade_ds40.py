#!/usr/bin/env python3
"""Run the official DS40 CASCADE model from the dedicated CASCADE environment.

The calcium-imaging notebook uses the Cajal kernel because its image-processing
dependencies are installed there.  This small bridge is launched with the Python
executable from the CASCADE conda environment, where TensorFlow/Keras is available.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "Spinal_cord_excitatory_30Hz_smoothing50ms"


def _native_bin_edges(time_s: np.ndarray) -> np.ndarray:
    """Midpoint bin edges for samples acquired at possibly jittered timestamps."""
    edges = np.empty(time_s.size + 1, dtype=np.float64)
    edges[1:-1] = (time_s[:-1] + time_s[1:]) / 2
    edges[0] = time_s[0] - (time_s[1] - time_s[0]) / 2
    edges[-1] = time_s[-1] + (time_s[-1] - time_s[-2]) / 2
    return edges


def _aggregate_spike_bins(
    model_time_s: np.ndarray,
    spikes_per_model_bin: np.ndarray,
    native_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum model-bin spike counts into the original acquisition bins.

    A native bin is left NaN if any of its constituent model predictions is NaN,
    which preserves CASCADE's receptive-field padding at the recording edges.
    """
    edges = _native_bin_edges(native_time_s)
    native_width_s = np.diff(edges)
    bin_index = np.searchsorted(edges, model_time_s, side="right") - 1
    in_range = (bin_index >= 0) & (bin_index < native_time_s.size)
    bin_index = bin_index[in_range]

    expected_per_bin = np.bincount(bin_index, minlength=native_time_s.size)
    counts = np.full(
        (spikes_per_model_bin.shape[0], native_time_s.size), np.nan, dtype=np.float64
    )

    for roi_i, prediction in enumerate(spikes_per_model_bin):
        values = prediction[in_range]
        finite = np.isfinite(values)
        finite_per_bin = np.bincount(bin_index[finite], minlength=native_time_s.size)
        sums = np.bincount(
            bin_index[finite], weights=values[finite], minlength=native_time_s.size
        )
        complete = (expected_per_bin > 0) & (finite_per_bin == expected_per_bin)
        counts[roi_i, complete] = sums[complete]

    return counts, counts / native_width_s[None, :]


def run_inference(
    input_path: Path,
    output_path: Path,
    cascade_root: Path,
    model_name: str = DEFAULT_MODEL,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        traces = np.asarray(data["traces"], dtype=np.float64)
        time_s = np.asarray(data["time_s"], dtype=np.float64)

    if traces.ndim != 2:
        raise ValueError(f"traces must have shape (ROIs, timepoints), got {traces.shape}")
    if time_s.ndim != 1 or traces.shape[1] != time_s.size:
        raise ValueError(
            f"time_s length {time_s.size} does not match trace length {traces.shape[1]}"
        )
    if time_s.size < 65:
        raise ValueError("CASCADE requires more than its 64-sample receptive window")
    if not np.isfinite(traces).all() or not np.isfinite(time_s).all():
        raise ValueError("CASCADE input traces and timestamps must be finite")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")

    model_folder = cascade_root / "Pretrained_models"
    model_path = model_folder / model_name
    config_path = model_path / "config.yaml"
    model_files = sorted(model_path.glob("Model_NoiseLevel_*_Ensemble_*.h5"))
    if not (cascade_root / "cascade2p" / "cascade.py").is_file():
        raise FileNotFoundError(f"CASCADE source not found under {cascade_root}")
    if not config_path.is_file() or not model_files:
        raise FileNotFoundError(f"CASCADE model is incomplete: {model_path}")

    sys.path.insert(0, str(cascade_root))
    from cascade2p import cascade, config, utils  # noqa: PLC0415

    cfg = config.read_config(str(config_path))
    model_rate_hz = float(cfg["sampling_rate"])
    native_rate_hz = float(1 / np.median(np.diff(time_s)))
    if abs(native_rate_hz - model_rate_hz) / model_rate_hz <= 0.05:
        model_time_s = time_s.copy()
        model_traces = traces.copy()
    else:
        native_edges = _native_bin_edges(time_s)
        model_time_s = np.arange(
            native_edges[0] + 0.5 / model_rate_hz,
            native_edges[-1],
            1 / model_rate_hz,
        )
        model_traces = np.vstack(
            [np.interp(model_time_s, time_s, trace) for trace in traces]
        )

    # Estimate noise before interpolation.  Recomputing it on an upsampled trace
    # would make adjacent differences artificially small and choose the wrong model.
    trace_noise_levels = utils.calculate_noise_levels(traces, native_rate_hz)
    available_noise_levels = np.asarray(cfg["noise_levels"], dtype=np.float64)
    selected_noise_levels = available_noise_levels[
        np.argmin(
            np.abs(trace_noise_levels[:, None] - available_noise_levels[None, :]), axis=1
        )
    ]

    # Keras 3 can read these legacy HDF5 networks for inference, but restoring the
    # serialized optimizer is unnecessary and is less compatible across versions.
    import tensorflow.keras.models as keras_models  # noqa: PLC0415

    original_load_model = keras_models.load_model

    def load_for_inference(*args, **kwargs):
        kwargs["compile"] = False
        return original_load_model(*args, **kwargs)

    keras_models.load_model = load_for_inference
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Argument `decay` is no longer supported")
            spikes_per_model_bin = cascade.predict(
                model_name,
                model_traces,
                model_folder=str(model_folder),
                threshold=0,
                padding=np.nan,
                trace_noise_levels=trace_noise_levels,
                verbosity=0,
            )
    finally:
        keras_models.load_model = original_load_model

    native_spikes_per_bin, native_spike_rate_hz = _aggregate_spike_bins(
        model_time_s, spikes_per_model_bin, time_s
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        model_name=np.asarray(model_name),
        model_rate_hz=np.asarray(model_rate_hz),
        native_rate_hz=np.asarray(native_rate_hz),
        time_model_s=model_time_s,
        spike_prob_model=spikes_per_model_bin,
        spike_rate_model_hz=spikes_per_model_bin * model_rate_hz,
        time_native_s=time_s,
        spikes_per_native_bin=native_spikes_per_bin,
        spike_rate_native_hz=native_spike_rate_hz,
        trace_noise_levels=trace_noise_levels,
        selected_noise_levels=selected_noise_levels,
        available_noise_levels=available_noise_levels,
    )

    print(
        f"CASCADE complete: {traces.shape[0]} ROIs, {native_rate_hz:.3f} Hz -> "
        f"{model_rate_hz:g} Hz -> native bins; saved {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cascade-root", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_inference(args.input, args.output, args.cascade_root, args.model)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
