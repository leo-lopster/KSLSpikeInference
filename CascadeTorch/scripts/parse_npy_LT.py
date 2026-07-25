import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read .npy files from input directory and write outputs to output directory"
    )
    parser.add_argument("input_dir", type=Path, help="Input directory or input .npy file")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: input file folder or input directory)",
    )
    parser.add_argument(
        "--elapsed-times",
        type=Path,
        default=None,
        help="Path to ElapsedTimes.yaml (default: LT_Data/ElapsedTimes.yaml relative to project)",
    )
    return parser.parse_args()


def check_input(input_path: Path) -> list[Path]:
    """Validate the input path and return a list of .npy files to process."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise SystemExit(f"Input file is not a .npy file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*.npy"))
        if not files:
            raise SystemExit(f"No .npy files found in directory: {input_path.resolve()}")
        return files

    raise SystemExit(f"Input path does not exist: {input_path}")


def _save_head_file(fpath: Path, arr: np.ndarray, out_dir: Path) -> None:
    """Save a small head preview for the array to the output directory."""
    if arr.ndim == 0:
        head_text = repr(arr)[:200]
        with open(out_dir / f"{fpath.stem}_head.txt", "w", encoding="utf-8") as fh:
            fh.write(head_text + "\n")
    elif arr.ndim == 1:
        pd.DataFrame(arr[:5], columns=["value"]).to_csv(
            out_dir / f"{fpath.stem}_head.csv", index=False
        )
    else:
        n_slices = min(5, arr.shape[0])
        flattened = arr[:n_slices].reshape(n_slices, -1)
        pd.DataFrame(flattened).to_csv(out_dir / f"{fpath.stem}_head.csv", index=False)


def _format_head(arr: np.ndarray) -> str:
    """Return a human-readable preview string for the first few elements of the array."""
    if arr.ndim == 0:
        return repr(arr)[:200]
    if arr.ndim == 1:
        return f"Head (first 5 elements): {arr[:5]}"
    if arr.ndim == 2:
        return pd.DataFrame(arr[:5]).to_string(index=False)

    n_slices = min(5, arr.shape[0])
    return pd.DataFrame(arr[:n_slices].reshape(n_slices, -1)).to_string(index=False)


def _build_summary_record(fpath: Path, arr: np.ndarray) -> dict[str, object]:
    """Build a summary record for a single .npy file."""
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    size = int(arr.size)

    if np.issubdtype(arr.dtype, np.number):
        arr_flat = arr.ravel()
        vmin = float(np.min(arr_flat))
        vmax = float(np.max(arr_flat))
        vmean = float(np.mean(arr_flat))
        vstd = float(np.std(arr_flat))
    else:
        vmin = vmax = vmean = vstd = None

    return {
        "filename": fpath.name,
        "path": str(fpath.resolve()),
        "shape": str(shape),
        "dtype": dtype,
        "size": size,
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "std": vstd,
    }


def make_summary(files: list[Path], out_dir: Path) -> pd.DataFrame:
    """Process files, save head previews, and return a summary DataFrame."""
    summaries = []

    for fpath in files:
        print(f"\nProcessing: {fpath.name}")
        try:
            arr = np.load(fpath, allow_pickle=True)
        except Exception as exc:
            print(f"  Failed to load {fpath.name}: {exc}")
            continue

        try:
            head_text = _format_head(arr)
            print(f"  {head_text}")
            _save_head_file(fpath, arr, out_dir)
        except Exception as exc:
            print(f"  Failed to build head for {fpath.name}: {exc}")

        try:
            summaries.append(_build_summary_record(fpath, arr))
        except Exception as exc:
            print(f"  Failed to summarize {fpath.name}: {exc}")

    return pd.DataFrame(summaries)


def parse_elapsed_times(elapsed_path: Path) -> list[float]:
    if not elapsed_path.exists():
        raise SystemExit(f"Elapsed times file not found: {elapsed_path}")
    text = elapsed_path.read_text(encoding="utf-8")
    m = re.search(r"theElapsedTimes\s*:\s*\[(.*?)\]", text, re.S)
    if not m:
        raise SystemExit(f"Could not parse elapsed times from {elapsed_path}")
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",") if p.strip()]

    if not parts:
        raise SystemExit(f"ElapsedTimes list is empty in: {elapsed_path}")

    # Detect optional leading count (some YAML variants include the count as first element)
    has_leading_count = False
    try:
        first_val = float(parts[0])
        if first_val.is_integer() and int(first_val) == len(parts) - 1:
            has_leading_count = True
    except Exception:
        has_leading_count = False

    try:
        if has_leading_count:
            values = [float(p) for p in parts[1:]]
        else:
            values = [float(p) for p in parts]
    except ValueError:
        raise SystemExit(f"ElapsedTimes file contains non-numeric entries: {elapsed_path}")

    # Heuristic: LT_Data elapsed times are sometimes in milliseconds. Convert to seconds
    # when values are large (e.g., > 1000). This avoids mixing ms and s across pipelines.
    if values:
        max_val = max(values)
        if max_val > 1000:
            values = [v / 1000.0 for v in values]
            print(f"Converted elapsed times from milliseconds to seconds (max was {max_val}).")

    return values


def save_traces_with_times(fpath: Path, arr: np.ndarray, elapsed_times: list[float], out_dir: Path) -> Path:
    if arr.ndim != 2:
        raise ValueError(f"Array is not 2D (ndim={arr.ndim}) for file {fpath}")

    n0, n1 = arr.shape

    def try_get_times_for(n_frames: int) -> tuple[bool, list[float]]:
        if len(elapsed_times) == n_frames:
            return True, elapsed_times
        if len(elapsed_times) >= n_frames + 1 and int(elapsed_times[0]) == len(elapsed_times) - 1:
            vals = elapsed_times[1:1 + n_frames]
            if len(vals) == n_frames:
                return True, vals
        if len(elapsed_times) >= n_frames:
            return True, elapsed_times[:n_frames]
        return False, []

    ok, times = try_get_times_for(n1)
    transpose_needed = False
    if not ok:
        ok, times = try_get_times_for(n0)
        transpose_needed = True

    if not ok:
        raise ValueError(
            f"Could not match elapsed times (len={len(elapsed_times)}) to array shape {arr.shape} for {fpath.name}"
        )

    if transpose_needed:
        data = arr
        n_frames = n0
        n_rois = n1
    else:
        data = arr.T
        n_frames = n1
        n_rois = n0

    trace_cols = [f"Trace_ROI{i+1}" for i in range(n_rois)]
    df = pd.DataFrame(data, columns=trace_cols)
    df.insert(0, "Frame", list(range(n_frames)))
    df.insert(1, "Time", times)

    out_path = out_dir / f"{fpath.stem}_trace.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_summary(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    """Save the summary DataFrame to a CSV file and return the saved path."""
    summary_path = out_dir / "npy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        if args.input_dir.is_file():
            output_dir = args.input_dir.parent
        else:
            output_dir = args.input_dir

    print(f"Input directory: {args.input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    files = check_input(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = make_summary(files, output_dir)
    summary_path = save_summary(summary_df, output_dir)
    print(f"\nSaved summary for {len(summary_df)} file(s) to {summary_path.resolve()}")

    # Determine elapsed times file
    if args.elapsed_times is None:
        default_elapsed = Path(__file__).resolve().parents[1] / "LT_Data" / "ElapsedTimes.yaml"
        elapsed_path = default_elapsed
    else:
        elapsed_path = args.elapsed_times

    print(f"Using elapsed times from: {elapsed_path}")
    elapsed_times = parse_elapsed_times(elapsed_path)

    # For each file, if it's 2D write a trace CSV with Frame and Time columns
    for fpath in files:
        try:
            arr = np.load(fpath, allow_pickle=True)
        except Exception as exc:
            print(f"Skipping {fpath.name}: failed to reload array: {exc}")
            continue

        if arr.ndim != 2:
            print(f"WARNING: {fpath.name} is not 2D (ndim={arr.ndim}); skipping trace CSV generation.")
            continue

        try:
            out_csv = save_traces_with_times(fpath, arr, elapsed_times, output_dir)
            print(f"Wrote trace CSV: {out_csv}")
        except Exception as exc:
            print(f"Failed to write trace CSV for {fpath.name}: {exc}")


if __name__ == "__main__":
    main()
