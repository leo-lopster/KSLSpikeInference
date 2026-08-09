"""Per-ROI brightness traces and dF/F.

Lifted out of the notebook so a GUI can run the same extraction. Frames are streamed one
at a time from the (lazy) stack, so the whole float32 movie is never in RAM at once.

Fixed while lifting: the notebook's loop allocated a trace of length `n_t` but wrote at
the absolute frame index `t`, which only lines up when `start_t == 0`. Any other start
wrote past the end of the array or shifted the trace. Here the write index is relative to
the window, so `start_t` behaves as advertised.
"""

from __future__ import annotations

import numpy as np


def roi_ids_in(roi_labels):
    """Sorted non-zero label ids in an ROI label image."""
    return [int(r) for r in np.unique(roi_labels) if r != 0]


def extract_roi_traces(stack, roi_labels, start_t=0, n_timepoints=None, progress=None):
    """Mean intensity per ROI per frame over a window. Returns {roi_id: (n,) float32}.

    `progress` is an optional callback taking (frames_done, frames_total) -- used by the
    GUI to drive a progress bar and stay cancellable; pass None in scripts.
    """
    start_t = int(max(0, start_t))
    total = int(stack.shape[0])
    if start_t >= total:
        raise ValueError(f"start_t={start_t} is past the end of the stack ({total} frames).")

    want = total - start_t if n_timepoints is None else int(max(0, n_timepoints))
    n_frames = min(want, total - start_t)
    if n_frames == 0:
        raise ValueError("Extraction window is empty; check start_t and n_timepoints.")

    ids = roi_ids_in(roi_labels)
    if not ids:
        raise ValueError("The ROI label image is empty -- paint or load labels with at least one ROI.")

    # Pixel indices per ROI, precomputed once rather than re-masking every frame.
    indices = {roi: np.where(roi_labels == roi) for roi in ids}
    traces = {roi: np.zeros(n_frames, dtype=np.float32) for roi in ids}

    for i in range(n_frames):
        frame = np.asarray(stack[start_t + i])  # materialise exactly one frame
        for roi in ids:
            yy, xx = indices[roi]
            traces[roi][i] = frame[yy, xx].mean()
        if progress is not None and (i % 10 == 0 or i == n_frames - 1):
            progress(i + 1, n_frames)

    print(f"> Extracted {len(ids)} ROI trace(s) x {n_frames} frames "
          f"(frames {start_t}..{start_t + n_frames - 1})")
    return traces


def compute_dff(traces, baseline_slides):
    """(F - F0) / F0 per ROI, with F0 the mean of the first `baseline_slides` frames.

    A fixed leading baseline, not a rolling percentile: it assumes the recording starts
    quiet. Frames 0..baseline_slides therefore have dF/F ~ 0 by construction, which is
    why the `pre` phase in the analysis starts after them rather than at 0.
    """
    baseline_slides = int(baseline_slides)
    if baseline_slides < 1:
        raise ValueError("baseline_slides must be >= 1.")

    out = {}
    for roi, trace in traces.items():
        if baseline_slides > trace.size:
            raise ValueError(f"baseline_slides={baseline_slides} exceeds the {trace.size}-frame "
                             f"trace for ROI {roi}.")
        f0 = float(np.mean(trace[:baseline_slides]))
        if f0 == 0:
            raise ZeroDivisionError(f"ROI {roi} has a zero baseline (F0=0); dF/F is undefined. "
                                    "Check the ROI labels against the preprocessed stack.")
        out[roi] = (trace - f0) / f0
    return out


def as_matrix(dff):
    """{roi: trace} -> (roi_ids, (n_rois, n_frames) matrix), row order fixed by roi id.

    Going via an explicit sorted list matters: np.array on dict_values gives a 0-d object
    array, and the row order is what every downstream plot and group listing is keyed on.
    """
    roi_ids = sorted(dff)
    return roi_ids, np.vstack([dff[roi] for roi in roi_ids])


def window_times(times_s, start_t, n_frames):
    """The slice of the acquisition time axis matching an extraction window."""
    return np.asarray(times_s)[start_t:start_t + n_frames]
