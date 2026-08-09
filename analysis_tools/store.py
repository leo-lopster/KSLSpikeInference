"""Every file the pipeline reads or writes, in one module.

Split out of the notebook so a GUI (or any other driver) can enter the pipeline at
whatever stage it has data for, instead of re-running §1-§3 to reach §4. The five I/O
points this covers:

  1. import  .npy frames from an .imgdir          -> load_imgdir / load_timebase
  2. import + export ROI labels as .tiff          -> load_roi_labels / save_roi_labels
  3. import  dF/F traces for multiple ROIs        -> load_traces / import_dff
  4. export  dF/F traces                          -> save_traces
  5. export  PCA analysis + summary (.csv/.tiff)  -> save_pca / save_features / save_figure

Point 3 is the "traces-only" entry: given a dF/F table from anywhere, everything upstream
of it (images, preprocessing, ROI labels) is unnecessary and should be treated as
unavailable rather than stale. `available()` reports which artifacts exist so a caller can
gate its stages on that instead of guessing.

Conventions kept from the rest of the package: parameters are always explicit (nothing
reads notebook globals), arrays go to .npz, tables to .csv, and every writer drops a JSON
sidecar recording what produced the data (CLAUDE.md §4 "always log method + parameters").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Names accepted as the time column of an imported dF/F table.
_TIME_NAMES = {"time", "time_s", "times", "times_s", "t", "t_s", "sec", "secs", "seconds"}

# Stripped from column/index labels when recovering integer ROI ids ("ROI3" -> 3).
_ROI_PREFIXES = ("roi_", "roi", "cell_", "cell")

# Column names that hold ROI identity, never a time axis.
_ROI_LABEL_NAMES = {"roi", "roi_id", "rois", "cell", "cell_id", "id", "label"}


# --------------------------------------------------------------------------- naming

def dataset_tag(data_dir):
    """'Debi_DRG-Streamtodisk-1769166699-997.imgdir' -> 'Debi_DRG'.

    The tag every downstream filename keys off. Centralised because it is duplicated
    across four notebook cells, and a GUI that derives it differently would silently
    read and write a different dataset's files.
    """
    return Path(data_dir).name.split("-")[0]


def method_tag(denoise_method):
    """'gaussian,nlm' -> 'gaussian-nlm'. Part of the preprocessed-stack filename."""
    return str(denoise_method).replace(",", "-").replace(" ", "")


def analysis_dir(root, tag):
    """`root`/`tag`, created if absent. Home of every §4 artifact for one dataset."""
    out = Path(root) / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------- 1. image frames

def list_frames(data_dir, channel=0):
    """Sorted per-timepoint .npy frame paths for one channel of an .imgdir."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"ImageData_Ch{channel}_TP*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No 'ImageData_Ch{channel}_TP*.npy' frames in {data_dir}. Check the path and "
            "the channel number (channel is not always 0 in multi-channel acquisitions)."
        )
    return files


def load_frame(path):
    """One frame as (H, W) -- the stored arrays carry a singleton plane axis."""
    return np.load(path)[0]


def load_imgdir(data_dir, channel=0):
    """Lazy (n_timepoints, H, W) dask array over an .imgdir, one frame per chunk.

    Nothing is read beyond the first frame (for shape/dtype) until something computes.
    """
    import dask.array as da
    from dask import delayed

    files = list_frames(data_dir, channel)
    probe = load_frame(files[0])
    lazy = delayed(load_frame)
    stack = da.stack([
        da.from_delayed(lazy(f), shape=probe.shape, dtype=probe.dtype) for f in files
    ], axis=0)
    print(f"> {len(files)} timepoints, frame {probe.shape} {probe.dtype} <- {Path(data_dir).name}")
    return stack


def load_timebase(data_dir, n_frames, fallback_hz=10.0):
    """Per-frame acquisition times (s) from ElapsedTimes.yaml.

    Returns (t_s, frame_rate_hz, source), where source is "ElapsedTimes.yaml" or
    "fallback". The YAML stores a list whose first entry is the frame count, followed by
    one elapsed time (ms) per frame; a length disagreement with the stack means the
    metadata does not describe these frames, so a nominal axis is used instead and said
    so out loud -- FRAME_RATE feeds CASCADE resampling and the Nyquist guard on the
    frequency bands, so a silently wrong one corrupts everything downstream.
    """
    import yaml

    path = Path(data_dir) / "ElapsedTimes.yaml"
    source = "ElapsedTimes.yaml"
    if path.exists():
        with open(path) as f:
            elapsed = yaml.safe_load(f)["theElapsedTimes"]
        times_ms = np.asarray(elapsed[1:], dtype=float)  # drop the leading count entry
        if len(times_ms) != n_frames:
            print(f"! ElapsedTimes has {len(times_ms)} entries but the stack has {n_frames} "
                  f"frames; falling back to a nominal {fallback_hz} Hz time axis.")
            times_s, source = np.arange(n_frames, dtype=float) / fallback_hz, "fallback"
        else:
            times_s = (times_ms - times_ms[0]) / 1000.0
    else:
        print(f"! No ElapsedTimes.yaml in {data_dir}; using a nominal {fallback_hz} Hz axis.")
        times_s, source = np.arange(n_frames, dtype=float) / fallback_hz, "fallback"

    rate = frame_rate(times_s)
    print(f"> Time axis: {n_frames} frames, {times_s[-1]:.1f} s total, ~{rate:.1f} Hz ({source})")
    return times_s, rate, source


def frame_rate(t):
    """Acquisition rate (Hz) from a time axis, via the median inter-frame interval."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("Need at least two timepoints to derive a frame rate.")
    return float(1.0 / np.median(np.diff(t)))


# --------------------------------------------------------------------- 2. ROI labels

def load_roi_labels(path, expected_shape=None):
    """ROI label image as uint16, optionally checked against a reference image shape."""
    labels = tifffile.imread(str(path))
    if expected_shape is not None and tuple(labels.shape) != tuple(expected_shape[-2:]):
        raise ValueError(
            f"ROI label shape {labels.shape} != expected {tuple(expected_shape[-2:])} ({path}). "
            "The labels were drawn on a different image -- check the path and the reference layer."
        )
    n_roi = len(np.unique(labels)) - 1
    print(f"> Loaded ROI labels {labels.shape} {labels.dtype}, {n_roi} ROI(s) <- {path}")
    return labels.astype(np.uint16)


def save_roi_labels(path, labels):
    """Write an ROI label image as a uint16 .tiff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    tifffile.imwrite(str(path), labels.astype(np.uint16))
    print(f"> Saved ROI labels {labels.shape}, {len(np.unique(labels)) - 1} ROI(s) -> {path}")
    return path


# ------------------------------------------------------------- preprocessed stack

def save_preprocessed(stack, out_dir, tag, m_tag, params):
    """Stream a lazy preprocessed stack to a chunked Zarr store + .params.json sidecar.

    The full (n_timepoints, H, W) float32 stack is never held in RAM at once, but this
    does trigger the whole preprocessing compute -- it is the slow cell.
    """
    from dask.diagnostics import ProgressBar

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / f"{tag}_{m_tag}.zarr"

    with ProgressBar():
        stack.to_zarr(str(zarr_path), overwrite=True)

    sidecar = dict(params)
    sidecar.setdefault("n_timepoints", int(stack.shape[0]))
    sidecar.setdefault("frame_shape", list(stack.shape[1:]))
    with open(zarr_path.with_suffix(".params.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"> Saved preprocessed stack -> {zarr_path}")
    print(f"> Params logged -> {zarr_path.with_suffix('.params.json')}")
    return zarr_path


def load_preprocessed(out_dir, tag, m_tag):
    """Rebuild the lazy preprocessed stack from Zarr. Returns (stack, params)."""
    import dask.array as da

    zarr_path = Path(out_dir) / f"{tag}_{m_tag}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"No preprocessed stack at {zarr_path}. Either run preprocessing first, or "
            "check DENOISE_METHOD -- the method is part of the filename, so changing it "
            "points at a different store."
        )
    stack = da.from_zarr(str(zarr_path))
    params_path = zarr_path.with_suffix(".params.json")
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    print(f"> Loaded preprocessed stack {stack.shape} {stack.dtype} <- {zarr_path}")
    return stack, params


# ------------------------------------------------------------- 3 + 4. dF/F traces

def save_traces(out_dir, roi_ids, dff, t, raw=None, params=None):
    """Persist dF/F traces three ways: .npz (exact), .csv (portable), .json (provenance).

    The CSV is the interchange copy -- `import_dff` reads it back, and so can anything
    else. The npz is what `load_traces` uses, because it round-trips dtypes and the raw
    (pre-dF/F) traces without a parsing step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_ids = np.asarray(roi_ids)
    dff = np.asarray(dff, dtype=np.float32)
    t = np.asarray(t, dtype=float)
    if dff.shape[0] != roi_ids.size:
        raise ValueError(f"dff has {dff.shape[0]} rows but {roi_ids.size} roi_ids were given; "
                         "dff must be (n_rois, n_frames).")
    if dff.shape[1] != t.size:
        raise ValueError(f"dff has {dff.shape[1]} frames but the time axis has {t.size}.")

    arrays = {"roi_ids": roi_ids, "dff": dff, "t": t}
    if raw is not None:
        arrays["raw"] = np.asarray(raw, dtype=np.float32)
    np.savez_compressed(out_dir / "traces.npz", **arrays)

    frame = pd.DataFrame(dff.T, columns=[f"ROI{r}" for r in roi_ids])
    frame.insert(0, "time_s", t)
    frame.to_csv(out_dir / "traces.csv", index=False)

    sidecar = {
        "n_rois": int(roi_ids.size),
        "n_frames": int(dff.shape[1]),
        "frame_rate_hz": frame_rate(t),
        "duration_s": float(t[-1] - t[0]),
        "units": "dF/F0 (dimensionless)",
        "has_raw_traces": raw is not None,
    }
    sidecar.update(params or {})
    with open(out_dir / "traces_params.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"> Saved traces.npz + traces.csv + traces_params.json ({roi_ids.size} ROIs x "
          f"{dff.shape[1]} frames) -> {out_dir}")
    return out_dir


def load_traces(out_dir):
    """Read back what `save_traces` wrote. Returns dict(roi_ids, dff, t, raw, params)."""
    out_dir = Path(out_dir)
    npz_path = out_dir / "traces.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No traces.npz in {out_dir}. Run the dF/F cell and save it, or use "
            "import_dff() to bring in traces from elsewhere."
        )
    with np.load(npz_path) as data:
        out = {k: data[k] for k in data.files}
    out.setdefault("raw", None)
    params_path = out_dir / "traces_params.json"
    out["params"] = json.loads(params_path.read_text()) if params_path.exists() else {}
    print(f"> Loaded {out['dff'].shape[0]} ROI(s) x {out['dff'].shape[1]} frames <- {npz_path}")
    return out


def import_dff(path, orientation="auto", time_column="auto", frame_rate_hz=None):
    """Import dF/F traces produced anywhere -- the traces-only entry point.

    Accepts .csv, .npz (keys `dff`, optionally `t`/`roi_ids`) or a bare 2-D .npy.

    `orientation` -- "roi_rows", "roi_cols", or "auto". Auto reads a table with a time
    column as roi_cols (one column per ROI); otherwise it assumes the longer axis is
    time, which is right for any realistic recording and wrong for a 3-frame test file.
    Say which you mean when it matters.

    `time_column` -- a column name/position, "auto" (a column named time/t/seconds, or a
    strictly increasing numeric first column), or None to ignore any and synthesise the
    axis from `frame_rate_hz`.

    Raises if the timebase cannot be established. It is not guessable, and every
    downstream stage -- CASCADE resampling, phase windows, the Nyquist ceiling on the
    frequency bands -- is wrong in a quiet, plausible-looking way if it is off.
    """
    path = Path(path)
    resolved = {"source": str(path), "orientation": orientation, "time_column": time_column}

    if path.suffix == ".csv":
        table = pd.read_csv(path)
        t, table, how = _split_time_column(table, time_column, orientation)
        resolved["time_source"] = how
        orient = _resolve_orientation(orientation, table.shape, t is not None)
        if orient == "roi_rows":
            table = _index_by_roi(table)
            values = table.to_numpy(dtype=float)
            roi_ids = _roi_ids(list(table.index), values.shape[0])
        else:
            values = table.to_numpy(dtype=float).T
            roi_ids = _roi_ids(list(table.columns), values.shape[0])
    elif path.suffix in (".npz", ".npy"):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "dff" not in data.files:
                    raise KeyError(f"{path} has keys {data.files} but no 'dff' array.")
                values = data["dff"]
                t = data["t"] if "t" in data.files else None
                stored_ids = data["roi_ids"] if "roi_ids" in data.files else None
        else:
            values, t, stored_ids = np.load(path), None, None
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D dF/F array, got shape {values.shape}.")
        resolved["time_source"] = "stored 't' array" if t is not None else "none"
        # Arrays are ROI-major by this package's convention; auto only has the axis
        # lengths to go on, so it takes the longer axis as time.
        orient = orientation
        if orient == "auto":
            orient = "roi_rows" if values.shape[1] >= values.shape[0] else "roi_cols"
        if orient == "roi_cols":
            values = values.T
        roi_ids = (np.asarray(stored_ids) if stored_ids is not None
                   else np.arange(1, values.shape[0] + 1))
    else:
        raise ValueError(f"Unsupported dF/F file type {path.suffix!r}; expected .csv/.npz/.npy.")

    resolved["orientation_resolved"] = orient

    if t is None:
        if frame_rate_hz is None:
            raise ValueError(
                f"{path} carries no time axis and no frame_rate_hz was given. Supply the "
                "acquisition rate in Hz -- it cannot be inferred from the traces, and "
                "spike inference, phase windows and the frequency bands all depend on it."
            )
        t = np.arange(values.shape[1], dtype=float) / float(frame_rate_hz)
        resolved["time_source"] = f"synthesised from frame_rate_hz={frame_rate_hz}"
    t = np.asarray(t, dtype=float)

    if t.size != values.shape[1]:
        raise ValueError(f"Time axis has {t.size} points but the traces have "
                         f"{values.shape[1]} frames; check `orientation`.")

    rate = frame_rate(t)
    resolved.update({"n_rois": int(len(roi_ids)), "n_frames": int(values.shape[1]),
                     "frame_rate_hz": rate, "mode": "traces-only"})
    print(f"> Imported {len(roi_ids)} ROI(s) x {values.shape[1]} frames "
          f"(~{rate:.2f} Hz, {resolved['time_source']}) <- {path}")
    print("> Traces-only mode: image, preprocessing and ROI-label stages do not apply to "
          "these traces.")
    return {"roi_ids": np.asarray(roi_ids), "dff": values.astype(np.float32), "t": t,
            "params": resolved}


def _split_time_column(table, time_column, orientation="auto"):
    """Pull the time axis out of an imported table. Returns (t | None, rest, how).

    The unnamed case is the delicate one: a roi-per-row table leads with a column of ROI
    ids, which is just as numeric and just as increasing as a time axis. Two guards keep
    it from being eaten -- a roi-per-row table has no time column to find at all, and in
    auto mode a leading column is only time if the table is taller than it is wide (one
    row per frame). Otherwise the ids get read as 7 timepoints for 300 frames of data.
    """
    if time_column is None:
        return None, table, "ignored"

    if time_column != "auto":
        col = table.columns[time_column] if isinstance(time_column, int) else time_column
        if col not in table.columns:
            raise KeyError(f"No column {col!r} in the imported table; columns are "
                           f"{list(table.columns)[:8]}...")
        return table[col].to_numpy(dtype=float), table.drop(columns=[col]), f"column {col!r}"

    named = [c for c in table.columns if str(c).strip().lower() in _TIME_NAMES]
    if named:
        col = named[0]
        return table[col].to_numpy(dtype=float), table.drop(columns=[col]), f"column {col!r}"

    if orientation == "roi_rows":
        return None, table, "none (roi-per-row table has no time column)"

    first = table.columns[0]
    if str(first).strip().lower() in _ROI_LABEL_NAMES:
        return None, table, "none"
    values = pd.to_numeric(table[first], errors="coerce").to_numpy(dtype=float)
    looks_like_time = (values.size > 1 and np.all(np.isfinite(values))
                       and np.all(np.diff(values) > 0)
                       and table.shape[0] >= table.shape[1])
    if looks_like_time:
        return values, table.drop(columns=[first]), f"increasing first column {first!r}"
    return None, table, "none"


def _resolve_orientation(orientation, shape, has_time):
    """Decide whether a table's rows are ROIs or frames.

    A table carrying a time column is frames-per-row by construction. Without one, the
    longer axis is taken as time -- right for any realistic recording, wrong for a
    handful of frames, which is why `orientation` can be stated outright.
    """
    if orientation in ("roi_rows", "roi_cols"):
        return orientation
    if orientation != "auto":
        raise ValueError(
            f"orientation must be 'auto', 'roi_rows' or 'roi_cols', got {orientation!r}.")
    return "roi_cols" if (has_time or shape[0] >= shape[1]) else "roi_rows"


def _index_by_roi(table):
    """Move a leading ROI-label column into the index of a roi-per-row table."""
    first = table.columns[0]
    if str(first).strip().lower() in _ROI_LABEL_NAMES or table[first].dtype == object:
        return table.set_index(first)
    return table


def _roi_ids(labels, n):
    """Integer ROI ids from column/index labels, falling back to 1..n.

    Recovers the ids written by `save_traces` ('ROI7' -> 7) so a round-trip through CSV
    keeps ROI identity, which is what every group membership listing is keyed on.
    """
    if len(labels) != n:
        return np.arange(1, n + 1)
    out = []
    for label in labels:
        text = str(label).strip()
        for prefix in _ROI_PREFIXES:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        try:
            out.append(int(float(text)))
        except ValueError:
            return np.arange(1, n + 1)
    return np.asarray(out)


# -------------------------------------------------------- 5. features, PCA, figures

def save_features(out_dir, name, frame):
    """Write a per-ROI feature/summary table as <name>.csv, indexed by ROI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index_label="roi")
    print(f"> Saved {name}.csv ({frame.shape[0]} ROIs x {frame.shape[1]} cols) -> {path}")
    return path


def load_features(out_dir, name):
    """Read back a table written by `save_features`."""
    path = Path(out_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No {name}.csv in {out_dir}.")
    return pd.read_csv(path, index_col="roi")


def save_groups(out_dir, frame):
    """Write the roi -> group assignment table (roi_groups.csv), one row per ROI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "roi_groups.csv"
    frame.to_csv(path, index=False)
    print(f"> Saved roi_groups.csv ({frame.shape[0]} ROIs) -> {path}")
    return path


def load_groups(out_dir):
    """Read back roi_groups.csv."""
    path = Path(out_dir) / "roi_groups.csv"
    if not path.exists():
        raise FileNotFoundError(f"No roi_groups.csv in {out_dir}.")
    return pd.read_csv(path)


def save_pca(out_dir, label, pca, scores, n_pc, Z, feature_names,
             cut_height=None, groups=None, roi_ids=None):
    """Persist one PCA + linkage so the tree can be re-cut without re-fitting.

    Stores what is needed to re-plot and re-cut -- scores, loadings, explained variance,
    the linkage matrix -- but NOT the StandardScaler statistics, so this cannot project
    new ROIs into an old PCA space. That is deliberate: `grouping.pca_and_linkage` fits
    the scaler internally, and the PCA is per-dataset, so there is nothing to project.

    `label` is "rate" (Index A) or "freq" (Index B); the two are never pooled.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores)

    np.savez_compressed(
        out_dir / f"pca_{label}.npz",
        scores=scores,
        components=np.asarray(pca.components_),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
        Z=np.asarray(Z),
    )

    sidecar = {
        "index": label,
        "n_pc": int(n_pc),
        "n_rois": int(scores.shape[0]),
        "cumulative_variance_at_n_pc": float(
            np.cumsum(pca.explained_variance_ratio_)[int(n_pc) - 1]),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "feature_names": [str(f) for f in feature_names],
        "cut_height": None if cut_height is None else float(cut_height),
        "linkage_method": "ward",
        "roi_ids": None if roi_ids is None else [int(r) for r in roi_ids],
        "groups": None if groups is None else [int(g) for g in groups],
    }
    with open(out_dir / f"pca_{label}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    # Human-readable companion: the PC coordinates each grouping decision was made on.
    table = pd.DataFrame(
        scores[:, :int(n_pc)],
        columns=[f"PC{i + 1}" for i in range(int(n_pc))],
        index=pd.Index(roi_ids if roi_ids is not None else range(1, scores.shape[0] + 1),
                       name="roi"),
    )
    if groups is not None:
        table.insert(0, "group", np.asarray(groups))
    table.to_csv(out_dir / f"pca_{label}_scores.csv", index_label="roi")

    print(f"> Saved pca_{label}.npz + .json + _scores.csv ({n_pc} PC(s), "
          f"{scores.shape[0]} ROIs) -> {out_dir}")
    return out_dir


def load_pca(out_dir, label):
    """Read back one saved PCA. Returns dict(scores, components, ..., Z, params)."""
    out_dir = Path(out_dir)
    npz_path = out_dir / f"pca_{label}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No pca_{label}.npz in {out_dir}.")
    with np.load(npz_path) as data:
        out = {k: data[k] for k in data.files}
    json_path = out_dir / f"pca_{label}.json"
    out["params"] = json.loads(json_path.read_text()) if json_path.exists() else {}
    print(f"> Loaded PCA '{label}' ({out['scores'].shape[0]} ROIs) <- {npz_path}")
    return out


def save_figure(fig, out_dir, name, dpi=200):
    """Write a matplotlib figure to <out_dir>/figures/<name>.tiff.

    TIFF because that is what the rest of the imaging pipeline speaks and it stays
    lossless; the plot functions in `plots` return their figure for exactly this.
    """
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.tiff"
    fig.savefig(path, format="tiff", dpi=dpi, bbox_inches="tight")
    print(f"> Saved figure -> {path}")
    return path


# ------------------------------------------------------------------------- gating

def available(out_dir):
    """Which artifacts exist in an analysis dir -- drives stage gating in a GUI.

    Lets a caller enable "load" over "recompute" without catching FileNotFoundError,
    and lets it tell an empty directory apart from a half-finished run.
    """
    out_dir = Path(out_dir)
    return {
        "traces": (out_dir / "traces.npz").exists(),
        "traces_csv": (out_dir / "traces.csv").exists(),
        "spike_rate": (out_dir / "spike_rate.npy").exists(),
        "rate_features": (out_dir / "rate_features.csv").exists(),
        "freq_features": (out_dir / "freq_features.csv").exists(),
        "roi_groups": (out_dir / "roi_groups.csv").exists(),
        "pca_rate": (out_dir / "pca_rate.npz").exists(),
        "pca_freq": (out_dir / "pca_freq.npz").exists(),
        "figures": (out_dir / "figures").is_dir(),
    }
