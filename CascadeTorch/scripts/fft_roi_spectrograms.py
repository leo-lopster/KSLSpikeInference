#!/usr/bin/env python3
"""Run per-ROI FFT analysis and optional spectrogram plots for trace CSVs.

Example:
    python3 CascadeTorch/scripts/fft_roi_spectrograms.py \
        "D3Red_FOV1 Data/LeftDCN_FOV1_400Hz450um1s_FOOT_5s10isi_roi_traces_trace_only.csv" \
        --spectrogram-rois 1,5,10
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROI_RE = re.compile(r"^Trace_ROI(\d+)(?P<suffix>_normalized|_average)?$")
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cascadetorch_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FFTs for each ROI trace and optionally save individual ROI spectrograms."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path(
            "D3Red_FOV1 Data/"
            "LeftDCN_FOV1_400Hz450um1s_FOOT_5s10isi_roi_traces_trace_only.csv"
        ),
        help="Trace-only CSV containing Frame, Time, and Trace_ROI columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for FFT CSVs and figures. Defaults to <input_dir>/FFT_Output/<input_stem>.",
    )
    parser.add_argument(
        "--use-normalized",
        action="store_true",
        help="Use Trace_ROI*_normalized columns instead of raw Trace_ROI* columns.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sampling rate in Hz. Defaults to 1 / median Time step.",
    )
    parser.add_argument(
        "--time-column",
        default="Time",
        help="Name of the time column used to infer sample rate. Default: Time.",
    )
    parser.add_argument(
        "--window",
        choices=("hann", "none"),
        default="hann",
        help="Window applied before FFT. Default: hann.",
    )
    parser.add_argument(
        "--detrend",
        choices=("mean", "linear", "none"),
        default="mean",
        help="Detrend each ROI before FFT/spectrogram. Default: mean.",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=None,
        help="Optional maximum frequency in Hz for saved FFT rows and plots.",
    )
    parser.add_argument(
        "--spectrogram-rois",
        default=None,
        help=(
            "Comma-separated ROI numbers to plot as spectrograms, for example '1,5,10'. "
            "Use 'all' to plot every ROI. Omit to skip spectrograms."
        ),
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=128,
        help="Spectrogram window length in samples. Default: 128.",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=None,
        help=(
            "Spectrogram overlap in samples. Defaults to nperseg // 2 unless "
            "--spectrogram-time-step is set."
        ),
    )
    parser.add_argument(
        "--spectrogram-time-step",
        type=float,
        default=None,
        help=(
            "Desired time increment between adjacent spectrogram columns in seconds, "
            "for example 1.0. Overrides --noverlap."
        ),
    )
    parser.add_argument(
        "--plot-fft",
        action="store_true",
        help="Save a summary FFT amplitude plot for all ROIs.",
    )
    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=5,
        help="Number of non-DC FFT peaks to report per ROI. Default: 5.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser.parse_args()


def roi_number(column_name: str) -> int:
    match = ROI_RE.match(column_name)
    if not match:
        return 0
    return int(match.group(1))


def trace_columns(df: pd.DataFrame, use_normalized: bool) -> list[str]:
    wanted_suffixes = {"_normalized"} if use_normalized else {"", "_average"}
    columns = []
    for column in df.columns:
        match = ROI_RE.match(column)
        if not match:
            continue
        suffix = match.group("suffix") or ""
        if suffix in wanted_suffixes:
            columns.append(column)

    columns = sorted(columns, key=roi_number)
    if not columns:
        signal_type = "normalized" if use_normalized else "raw or averaged"
        raise ValueError(f"No {signal_type} Trace_ROI columns found in input CSV.")
    return columns


def infer_sample_rate(df: pd.DataFrame, time_column: str) -> float:
    if time_column not in df.columns:
        raise ValueError(f"Time column not found: {time_column}")
    time = pd.to_numeric(df[time_column], errors="coerce")
    diffs = time.diff().iloc[1:]
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        raise ValueError(f"Could not infer sampling rate from {time_column}.")
    return 1.0 / float(positive_diffs.median())


def detrend_signal(values: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)

    cleaned = values.astype(float, copy=True)
    if not finite.all():
        x = np.arange(len(cleaned))
        cleaned[~finite] = np.interp(x[~finite], x[finite], cleaned[finite])

    if mode == "mean":
        return cleaned - np.mean(cleaned)
    if mode == "linear":
        x = np.arange(len(cleaned), dtype=float)
        slope, intercept = np.polyfit(x, cleaned, 1)
        return cleaned - (slope * x + intercept)
    return cleaned


def fft_amplitude(
    values: np.ndarray,
    sample_rate: float,
    detrend: str,
    window: str,
) -> tuple[np.ndarray, np.ndarray]:
    signal = detrend_signal(values, detrend)
    n = len(signal)
    if n < 2:
        raise ValueError("Need at least two samples for FFT.")

    if window == "hann":
        weights = np.hanning(n)
        coherent_gain = float(np.mean(weights))
        signal = signal * weights
    else:
        coherent_gain = 1.0

    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(signal)
    amplitude = np.abs(spectrum) * 2.0 / (n * coherent_gain)
    amplitude[0] = amplitude[0] / 2.0
    return frequencies, amplitude


def build_fft_table(
    df: pd.DataFrame,
    columns: list[str],
    sample_rate: float,
    detrend: str,
    window: str,
    max_frequency: float | None,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows = []
    spectra = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        frequencies, amplitude = fft_amplitude(values, sample_rate, detrend, window)
        if max_frequency is not None:
            keep = frequencies <= max_frequency
            frequencies = frequencies[keep]
            amplitude = amplitude[keep]
        spectra[column] = (frequencies, amplitude)
        roi = roi_number(column)
        rows.extend(
            {
                "roi": roi,
                "trace_column": column,
                "frequency_hz": float(freq),
                "amplitude": float(amp),
            }
            for freq, amp in zip(frequencies, amplitude)
        )
    return pd.DataFrame(rows), spectra


def build_peak_table(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for column, (frequencies, amplitude) in spectra.items():
        non_dc = frequencies > 0
        freqs = frequencies[non_dc]
        amps = amplitude[non_dc]
        if len(amps) == 0:
            continue
        order = np.argsort(amps)[::-1][:top_n]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "roi": roi_number(column),
                    "trace_column": column,
                    "rank": rank,
                    "frequency_hz": float(freqs[idx]),
                    "amplitude": float(amps[idx]),
                }
            )
    return pd.DataFrame(rows)


def get_pyplot(interactive: bool = False):
    import matplotlib

    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def parse_roi_selection(selection: str | None, columns: list[str]) -> list[str]:
    if selection is None:
        return []
    if selection.strip().lower() == "all":
        return columns

    requested = {
        int(item.strip())
        for item in selection.split(",")
        if item.strip()
    }
    by_roi = {roi_number(column): column for column in columns}
    missing = sorted(requested - set(by_roi))
    if missing:
        raise ValueError(f"Requested ROI(s) not found: {missing}")
    return [by_roi[roi] for roi in sorted(requested)]


def resolve_spectrogram_overlap(
    nperseg: int,
    noverlap: int | None,
    sample_rate: float,
    time_step: float | None,
) -> tuple[int, float]:
    if nperseg <= 1:
        raise ValueError("--nperseg must be greater than 1.")

    if time_step is not None:
        if time_step <= 0:
            raise ValueError("--spectrogram-time-step must be greater than 0.")
        hop_samples = max(1, int(round(time_step * sample_rate)))
        if hop_samples > nperseg:
            max_step = nperseg / sample_rate
            raise ValueError(
                "--spectrogram-time-step is larger than the spectrogram window. "
                f"Use --nperseg >= {hop_samples}, or choose a step <= {max_step:.4g} s."
            )
        return nperseg - hop_samples, hop_samples / sample_rate

    resolved_overlap = noverlap if noverlap is not None else nperseg // 2
    if resolved_overlap < 0:
        raise ValueError("--noverlap cannot be negative.")
    if resolved_overlap >= nperseg:
        raise ValueError("--noverlap must be smaller than --nperseg.")

    hop_samples = nperseg - resolved_overlap
    return resolved_overlap, hop_samples / sample_rate


def plot_fft_summary(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str,
    show: bool = False,
) -> None:
    plt = get_pyplot(interactive=show)
    fig, ax = plt.subplots(figsize=(12, 6))
    for column, (frequencies, amplitude) in spectra.items():
        ax.plot(frequencies, amplitude, linewidth=0.9, alpha=0.65, label=f"ROI {roi_number(column)}")
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    if len(spectra) <= 15:
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_spectrogram(
    time: np.ndarray,
    values: np.ndarray,
    sample_rate: float,
    column: str,
    output_path: Path,
    detrend: str,
    nperseg: int,
    noverlap: int,
    time_step: float,
    max_frequency: float | None,
    show: bool = False,
) -> None:
    plt = get_pyplot(interactive=show)
    signal = detrend_signal(values, detrend)
    nfft = max(8, min(nperseg, len(signal)))
    overlap = min(noverlap, nfft - 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    power, freqs, bins, image = ax.specgram(
        signal,
        NFFT=nfft,
        Fs=sample_rate,
        noverlap=overlap,
        cmap="magma",
        scale="dB",
    )
    del power, freqs, bins
    if max_frequency is not None:
        ax.set_ylim(0, max_frequency)
    ax.set_title(f"ROI {roi_number(column)} spectrogram (dt={time_step:.4g} s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if len(time) > 0:
        ax.set_xlim(float(time[0]), float(time[-1]))
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Power (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    input_csv = args.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_csv.parent / "FFT_Output" / input_csv.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    columns = trace_columns(df, args.use_normalized)
    sample_rate = args.sample_rate or infer_sample_rate(df, args.time_column)
    has_averaged_columns = all(column.endswith("_average") for column in columns)
    signal_label = "averaged" if has_averaged_columns else ("normalized" if args.use_normalized else "raw")

    fft_table, spectra = build_fft_table(
        df=df,
        columns=columns,
        sample_rate=sample_rate,
        detrend=args.detrend,
        window=args.window,
        max_frequency=args.max_frequency,
    )
    peaks = build_peak_table(spectra, args.top_n_peaks)

    suffix = signal_label
    fft_path = output_dir / f"{input_csv.stem}_{suffix}_fft_amplitudes.csv"
    peaks_path = output_dir / f"{input_csv.stem}_{suffix}_fft_top_peaks.csv"
    fft_table.to_csv(fft_path, index=False)
    peaks.to_csv(peaks_path, index=False)

    if args.plot_fft:
        fft_png = output_dir / f"{input_csv.stem}_{suffix}_fft_summary.png"
        plot_fft_summary(
            spectra,
            fft_png,
            title=f"{input_csv.stem} ({signal_label}, sample rate {sample_rate:.4g} Hz)",
            show=args.show,
        )

    selected_spectrograms = parse_roi_selection(args.spectrogram_rois, columns)
    if selected_spectrograms:
        spec_dir = output_dir / "spectrograms"
        spec_dir.mkdir(exist_ok=True)
        time = pd.to_numeric(df[args.time_column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(time).any():
            time = np.arange(len(df), dtype=float) / sample_rate
        noverlap, spectrogram_time_step = resolve_spectrogram_overlap(
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            sample_rate=sample_rate,
            time_step=args.spectrogram_time_step,
        )
        for column in selected_spectrograms:
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            spec_png = spec_dir / f"{input_csv.stem}_{suffix}_ROI{roi_number(column)}_spectrogram.png"
            plot_spectrogram(
                time=time,
                values=values,
                sample_rate=sample_rate,
                column=column,
                output_path=spec_png,
                detrend=args.detrend,
                nperseg=args.nperseg,
                noverlap=noverlap,
                time_step=spectrogram_time_step,
                max_frequency=args.max_frequency,
                show=args.show,
            )

    print(f"Input: {input_csv}")
    print(f"ROIs analysed: {len(columns)}")
    print(f"Sample rate: {sample_rate:.6g} Hz")
    print(f"FFT amplitudes: {fft_path}")
    print(f"Top peaks: {peaks_path}")
    if selected_spectrograms:
        print(f"Spectrograms: {output_dir / 'spectrograms'}")
        print(f"Spectrogram time step: {spectrogram_time_step:.6g} s")


if __name__ == "__main__":
    main()
