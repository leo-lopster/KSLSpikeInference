"""CASCADE spike inference, including frame-rate resampling to match a pretrained model.

The model's training rate is read from its NAME (e.g. 'Spinal_cord_excitatory_30Hz_...'
-> 30 Hz), which is the project convention; `describe_models` also surfaces the rate in
each config.yaml so the two can be compared, because at least one shipped folder is
misnamed (GC8_EXC_15Hz_smoothing50ms_high_noise carries sampling_rate 10.0).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly


# --------------------------------------------------------------------------- models

def model_config(model_dir, name):
    """Parsed config.yaml for one pretrained model folder."""
    with open(Path(model_dir) / name / "config.yaml") as f:
        return yaml.safe_load(f)


def model_rate_from_name(name):
    """Training sampling rate parsed out of the model name itself.

    'Spinal_cord_excitatory_30Hz_smoothing50ms' -> 30.0
    'GC8_EXC_7.5Hz_smoothing100ms_high_noise'   -> 7.5
    """
    match = re.search(r"[_-](\d+(?:\.\d+)?)Hz", name)
    if not match:
        raise ValueError(f"No '<rate>Hz' token in model name {name!r}; cannot infer its "
                         "training rate. Rename the folder or pass the rate explicitly.")
    return float(match.group(1))


def available_models(model_dir):
    return sorted(p.name for p in Path(model_dir).iterdir() if p.is_dir())


def describe_models(model_dir):
    """Print every local model with its name-rate, config-rate and training datasets."""
    print(f"{'model':<52}{'name':>6}{'cfg':>7}   training datasets")
    for name in available_models(model_dir):
        cfg = model_config(model_dir, name)
        name_rate, cfg_rate = model_rate_from_name(name), float(cfg["sampling_rate"])
        flag = "" if np.isclose(name_rate, cfg_rate) else "   <-- MISMATCH"
        print(f"{name:<52}{name_rate:>6}{cfg_rate:>7}   "
              f"{', '.join(cfg['training_datasets'])}{flag}")


# ----------------------------------------------------------------------- resampling

def resample_traces(traces, src_rate, dst_rate, max_denominator=1000):
    """Resample a (n_rois, n_frames) array along time from src_rate to dst_rate (Hz).

    Polyphase (`resample_poly`) rather than FFT resampling: the ratio is rational and
    small, and polyphase avoids the circular-convolution edge ringing that
    `scipy.signal.resample` produces on non-periodic calcium traces.

    NaN-safe -- NaNs are interpolated before filtering (an FIR would otherwise smear
    them across the whole trace) and the invalid mask is re-applied afterwards, so
    CASCADE's unpredictable edge frames stay marked as unpredictable.
    """
    x = np.asarray(traces, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if np.isclose(src_rate, dst_rate):
        return x.copy()

    ratio = Fraction(dst_rate / src_rate).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator

    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        idx = np.arange(x.shape[1])
        for i in range(x.shape[0]):
            m = bad[i]
            if m.all():
                x[i] = 0.0
            elif m.any():
                x[i, m] = np.interp(idx[m], idx[~m], x[i, ~m])

    y = resample_poly(x, up, down, axis=1)

    if bad.any():
        # nearest source frame for each output frame, so the invalid mask carries over
        src_idx = np.clip((np.arange(y.shape[1]) * down) // up, 0, x.shape[1] - 1)
        y[bad[:, src_idx]] = np.nan
    return y


# ------------------------------------------------------------------------ inference

def _import_cascade(cascade_dir):
    """Import CascadeTorch, applying the duplicate-OpenMP workaround first.

    `import torch` aborts in this env with 'OMP: Error #15' (two copies of
    libomp.dylib) unless KMP_DUPLICATE_LIB_OK is set before the import.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cascade_dir = str(Path(cascade_dir).resolve())
    if cascade_dir not in sys.path:
        sys.path.insert(0, cascade_dir)
    import torch
    from cascade2p import cascade
    from cascade2p.utils import calculate_noise_levels
    return torch, cascade, calculate_noise_levels


@dataclass
class CascadeResult:
    """Inference output plus everything needed to reproduce and audit it."""
    spike_rate: np.ndarray          # (n_rois, n_frames) spikes/s on the ACQUISITION grid
    spike_rate_model: np.ndarray    # same, on the model's grid (identical if no resample)
    pad: int                        # unpredictable frames per end, acquisition grid
    noise_levels: np.ndarray        # measured at the acquisition rate
    model_name: str
    model_rate: float
    model_cfg: dict = field(repr=False)
    frame_rate: float = 0.0
    resampled: bool = False
    ratio: float = 1.0


def check_model(model_dir, model_name, frame_rate, verbose=True):
    """Validate a model choice against the recording and report the resampling plan.

    Returns (model_cfg, model_rate, ratio, needs_resample).
    """
    cfg = model_config(model_dir, model_name)
    model_rate = model_rate_from_name(model_name)
    ratio = model_rate / frame_rate
    needs_resample = not np.isclose(ratio, 1.0)

    if not np.isclose(model_rate, float(cfg["sampling_rate"])) and verbose:
        print(f"! NAME/CONFIG MISMATCH for {model_name}: the name says {model_rate} Hz, "
              f"config.yaml says {cfg['sampling_rate']} Hz. Resampling follows the NAME, "
              "but CASCADE reads the CONFIG internally -- resolve before trusting output.")

    if verbose:
        print(f"> Selected: {model_name}")
        print(f"> Trained at {model_rate:.4g} Hz on {', '.join(cfg['training_datasets'])}; "
              f"smoothing {1000 * cfg['smoothing']:.0f} ms, window {cfg['windowsize']} frames")
        if needs_resample:
            print(f"> Recording is {frame_rate:.2f} Hz -> traces are resampled {ratio:.4g}x to "
                  f"{model_rate:.4g} Hz for inference, then the inferred rate is resampled back "
                  f"to {frame_rate:.2f} Hz so every window stays on the real acquisition grid.")
            if ratio > 1:
                print("! Upsampling adds NO information. The interpolated trace is smoother "
                      "than anything the model saw in training and its frame-to-frame noise "
                      "is artificially low (noise levels are therefore measured at "
                      f"{frame_rate:.2f} Hz and passed in). True resolution stays "
                      f"{frame_rate:.2f} Hz / {frame_rate / 2:.2f} Hz Nyquist regardless of "
                      f"the {model_rate:.4g} Hz label.")
        else:
            print("> Recording rate already matches the model; no resampling needed.")
        if "GCaMP6" in " ".join(cfg["training_datasets"]):
            print("! Indicator mismatch: model trained on GCaMP6, recording is GCaMP8. "
                  "GCaMP6-tuned models misestimate rate at both ends on GCaMP8 "
                  "(CLAUDE.md App.A 4C) -- closer tissue traded for wrong indicator.")
    return cfg, model_rate, ratio, needs_resample


def run(dff_matrix, frame_rate, model_name, cascade_dir, model_dir=None, verbose=True,
        announce_model=True):
    """Infer spike rate (spikes/s) from dF/F, resampling to the model rate and back.

    CASCADE's raw output is spikes per FRAME (its training target is a per-frame binned
    spike count, `cascade2p/utils.py` histogram + normalised Gaussian smoothing), so it
    is multiplied by the rate of the grid it was produced on. That conversion happens
    BEFORE resampling back: a rate is an intensity and survives resampling unchanged,
    whereas spikes-per-frame would need rescaling by the frame-duration ratio.
    """
    model_dir = Path(model_dir or Path(cascade_dir) / "Pretrained_models")
    torch, cascade, calculate_noise_levels = _import_cascade(cascade_dir)

    # announce_model=False when the notebook already reported the plan in its own cell
    cfg, model_rate, ratio, needs_resample = check_model(
        model_dir, model_name, frame_rate, verbose=verbose and announce_model)

    # Noise level decides which ensemble each ROI is routed to and MUST be measured on
    # the native-rate traces: the metric is a median frame-to-frame difference, and
    # interpolated frames are correlated by construction, so measuring it after
    # upsampling reports a far-too-clean trace.
    noise_levels = calculate_noise_levels(dff_matrix, frame_rate)
    model_noise = np.asarray(cfg["noise_levels"], dtype=float)
    if verbose:
        print(f"\n> Noise levels at the native {frame_rate:.2f} Hz (percent units): "
              f"median {np.median(noise_levels):.2f}, "
              f"range [{noise_levels.min():.2f}, {noise_levels.max():.2f}]")
        print(f"> Model covers {model_noise.min():.0f}-{model_noise.max():.0f}; "
              f"{int((noise_levels < model_noise.min()).sum())} ROI(s) below, "
              f"{int((noise_levels > model_noise.max()).sum())} above -> nearest ensemble.")

    if needs_resample:
        traces_for_model = resample_traces(dff_matrix, frame_rate, model_rate)
        if verbose:
            naive = calculate_noise_levels(traces_for_model, model_rate)
            print(f"> Resampled for inference: {np.shape(dff_matrix)} @ {frame_rate:.2f} Hz "
                  f"-> {traces_for_model.shape} @ {model_rate:.4g} Hz")
            print(f"> (noise measured AFTER resampling would read {np.median(naive):.2f} "
                  f"instead of {np.median(noise_levels):.2f} -- native value passed instead)")
    else:
        traces_for_model = dff_matrix

    spike_prob = cascade.predict(
        model_name,
        traces_for_model,
        model_folder=str(model_dir),
        padding=np.nan,
        trace_noise_levels=noise_levels,
        device=torch.device("cpu"),
        verbosity=1 if verbose else 0,
    )
    spike_rate_model = spike_prob * model_rate  # spikes/frame -> spikes/s

    if needs_resample:
        spike_rate = resample_traces(spike_rate_model, model_rate, frame_rate)
        n_target = np.shape(dff_matrix)[1]
        if spike_rate.shape[1] != n_target:
            n = min(spike_rate.shape[1], n_target)
            print(f"! Resampled length {spike_rate.shape[1]} != {n_target}; trimming to {n}.")
            spike_rate = spike_rate[:, :n]
        if verbose:
            print(f"> Inferred at {model_rate:.4g} Hz {spike_rate_model.shape}, "
                  f"resampled back to {frame_rate:.2f} Hz {spike_rate.shape}")
    else:
        spike_rate = spike_rate_model

    pad = int(np.isnan(spike_rate[0]).argmin())
    if verbose:
        print(f"\n> Inferred rate {spike_rate.shape}; {pad} NaN frame(s) padded per end "
              f"(= {pad / frame_rate:.1f} s)")
        print(f"> Rate range [{np.nanmin(spike_rate):.3f}, {np.nanmax(spike_rate):.3f}] spikes/s")

    return CascadeResult(
        spike_rate=spike_rate, spike_rate_model=spike_rate_model, pad=pad,
        noise_levels=noise_levels, model_name=model_name, model_rate=model_rate,
        model_cfg=cfg, frame_rate=frame_rate, resampled=needs_resample, ratio=ratio,
    )


def save(result, out_dir, extra=None):
    """Persist the inferred rate plus the parameters that produced it (CLAUDE.md 4)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "spike_rate.npy", result.spike_rate)

    params = {
        "algorithm": "CASCADE (Rupprecht et al. 2021), CascadeTorch",
        "model_name": result.model_name,
        "model_rate_hz_from_name": result.model_rate,
        "model_rate_hz_from_config": float(result.model_cfg["sampling_rate"]),
        "model_training_datasets": result.model_cfg["training_datasets"],
        "model_smoothing_s": result.model_cfg["smoothing"],
        "recording_frame_rate_hz": result.frame_rate,
        "output_units": "spikes/s (CASCADE spikes-per-frame x grid rate)",
        "resampled": bool(result.resampled),
        "resample_ratio": result.ratio,
        "resample_method": "scipy.signal.resample_poly (polyphase), NaN-preserving",
        "noise_levels_measured_at_hz": result.frame_rate,
        "padding": "nan",
        "pad_frames_per_end": result.pad,
        "noise_levels": [float(x) for x in result.noise_levels],
    }
    params.update(extra or {})
    with open(out_dir / "spike_inference_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"> Saved spike_rate.npy + spike_inference_params.json -> {out_dir}")
    return out_dir


def load(out_dir):
    """Read back a saved inference. Returns (spike_rate, params).

    The counterpart to `save`, so a session can pick up at the feature/PCA stage without
    re-running the model. The params sidecar is not optional here: an inferred rate whose
    model, frame rate and padding are unknown cannot be interpreted, only plotted.
    """
    out_dir = Path(out_dir)
    rate_path = out_dir / "spike_rate.npy"
    params_path = out_dir / "spike_inference_params.json"
    missing = [p.name for p in (rate_path, params_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not in {out_dir}; run CASCADE and save it first.")

    spike_rate = np.load(rate_path)
    params = json.loads(params_path.read_text())
    print(f"> Loaded spike_rate {spike_rate.shape} ({params.get('model_name', '?')}, "
          f"{params.get('recording_frame_rate_hz', float('nan')):.2f} Hz) <- {out_dir}")
    return spike_rate, params
