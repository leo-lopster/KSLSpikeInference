"""Per-phase feature tables: Index A (inferred rate) and Index B (event frequency).

The two indices are kept in separate tables on purpose -- they feed separate PCAs.
Pooling rate and frequency features into one decomposition yields PCs that are a blend
of both and can be read as neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def phase_masks(t, phases):
    """Boolean mask per phase over the time axis `t` (seconds)."""
    masks = {name: (t >= t0) & (t < t1) for name, (t0, t1) in phases.items()}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Phase {name!r} {phases[name]} contains no frames of t.")
    return masks


def describe_phases(t, phases, masks, frame_rate):
    for name, mask in masks.items():
        print(f"> {name:<5} requested {phases[name][0]:>5.1f}-{phases[name][1]:<5.1f} s | "
              f"realised {t[mask][0]:>5.1f}-{t[mask][-1]:<5.1f} s, {mask.sum():>3} frames")
    total = sum(int(m.sum()) for m in masks.values())
    print(f"> {total} frames assigned across {len(phases)} disjoint phases; "
          f"recording ends at {t[-1]:.1f} s, frame rate {frame_rate:.2f} Hz")
    last = max(hi for _, hi in phases.values())
    if last > t[-1]:
        print(f"! A phase runs to {last:.0f} s but the extraction stops at {t[-1]:.1f} s -- "
              f"report that window as ending at {t[-1]:.1f} s.")


def auto_active_thresh(spike_rate, baseline_mask, n_sigma=3.0):
    """Robust activity threshold from the baseline phase: median + n_sigma * MAD-sigma.

    A fixed absolute value does not transfer between models -- the GC8 models floor at
    0 spikes/s while the spinal-cord model never drops below ~0.3, so any constant is
    either above every peak or below every trough for one of them. There is no field
    consensus on principled thresholding (CLAUDE.md 6); this is a stated convention.
    """
    base = spike_rate[:, baseline_mask]
    med = np.nanmedian(base)
    mad = np.nanmedian(np.abs(base - med))
    return float(med + n_sigma * 1.4826 * mad), float(med), float(1.4826 * mad)


def rate_features(spike_rate, roi_ids, masks, active_thresh):
    """Index A: mean / peak / active-fraction of the inferred rate, per phase.

    Integrated spike count is deliberately excluded: it equals mean x duration, so it
    would simply double the weight of the mean in the PCA.
    """
    out = pd.DataFrame(index=pd.Index(roi_ids, name="roi"))
    for name, mask in masks.items():
        seg = spike_rate[:, mask]
        finite = np.isfinite(seg)
        # NaN-padded frames must not count as "inactive" -- mask them out rather than
        # letting (NaN > thresh) -> False silently deflate the fraction.
        active = np.where(finite, seg > active_thresh, np.nan)
        out[f"rate_mean_{name}"] = np.nanmean(seg, axis=1)
        out[f"rate_peak_{name}"] = np.nanmax(seg, axis=1)
        out[f"rate_active_frac_{name}"] = np.nanmean(active, axis=1)
    return out


def dff_summary(dff_matrix, roi_ids, masks):
    """Descriptive dF/F amplitude per phase. Fed to NEITHER PCA.

    dF/F amplitude is not a rate proxy (CLAUDE.md App.A 11), and including it would
    also break the one-index-per-PCA rule this section is built on.
    """
    out = pd.DataFrame(index=pd.Index(roi_ids, name="roi"))
    for name, mask in masks.items():
        seg = dff_matrix[:, mask]
        out[f"dff_mean_{name}"] = seg.mean(axis=1)
        out[f"dff_peak_{name}"] = seg.max(axis=1)
    return out


def report_threshold(features, masks, active_thresh):
    """Is the threshold discriminating, or above/below everything?"""
    for name in masks:
        frac = features[f"rate_active_frac_{name}"]
        print(f"> {name:<5} ACTIVE_THRESH={active_thresh:.3f}: "
              f"{int((frac > 0).sum()):>2}/{len(frac)} ROIs ever active, "
              f"{int((frac == 1).sum())} always active")
    constant = [c for c in features.columns if features[c].nunique() <= 1]
    if constant:
        print(f"! Constant feature(s) {constant}: identical for every ROI, so they add "
              "nothing to the PCA (z-scoring leaves them at 0). Adjust ACTIVE_THRESH if "
              "the active-fraction columns are the ones flat-lining.")


def group_summary(rate_feats, dff_feats, groups, phase_order):
    """Per-group phase profile: the answer to 'which group fires under which condition'.

    `dominant_phase` is an argmax over three means -- with a single trial it ranks, it
    does not test.
    """
    rows = []
    for g in range(1, int(groups.max()) + 1):
        sel = groups == g
        row = {"group": f"G{g}", "n_rois": int(sel.sum())}
        means = {name: rate_feats[sel][f"rate_mean_{name}"].mean() for name in phase_order}
        for name in phase_order:
            row[f"rate_mean_{name}"] = means[name]
            row[f"rate_peak_{name}"] = rate_feats[sel][f"rate_peak_{name}"].mean()
            row[f"dff_peak_{name}"] = dff_feats[sel][f"dff_peak_{name}"].mean()
        row["dominant_phase"] = max(means, key=means.get)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group").round(3)


def freq_group_summary(freq_feats, groups, phase_order, bands):
    """Per-group frequency profile: mean dominant frequency + highest-power band."""
    rows = []
    for g in range(1, int(groups.max()) + 1):
        sel = freq_feats[groups == g]
        row = {"group": f"F{g}", "n_rois": int(len(sel))}
        for name in phase_order:
            row[f"domfreq_{name}"] = sel[f"domfreq_{name}"].mean()
            band_means = {f"{lo}-{hi}": sel[f"bp{lo}-{hi}_{name}"].mean() for lo, hi in bands}
            row[f"topband_{name}"] = max(band_means, key=band_means.get) + " Hz"
        rows.append(row)
    return pd.DataFrame(rows).set_index("group").round(3)


def _load_fft_amplitude(cascade_dir):
    """Reuse the project's existing FFT helper rather than a second implementation.

    CascadeTorch/scripts/fft_roi_spectrograms.py guards main() behind __name__, so
    importing it runs nothing.
    """
    scripts_dir = str((Path(cascade_dir) / "scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from fft_roi_spectrograms import fft_amplitude
    return fft_amplitude


def describe_bands(masks, frame_rate, bands):
    for name, mask in masks.items():
        n = int(mask.sum())
        print(f"> {name:<5} {n:>3} frames = {n / frame_rate:.1f} s -> FFT bin width "
              f"{frame_rate / n:.3f} Hz")
    print(f"> Bands {bands} Hz. The lower edge is set by the SHORTEST phase (its bin "
          f"width), the upper by the {frame_rate / 2:.1f} Hz Nyquist limit.")


def freq_features(spike_rate, roi_ids, masks, frame_rate, bands, cascade_dir):
    """Index B: relative band powers + dominant frequency of the inferred rate, per phase.

    Band powers are RELATIVE (fraction of in-band power), which is what makes phases of
    different length comparable despite their different FFT bin widths. Anything slower
    than the lowest band edge is excluded by design -- stated, not hidden.
    """
    fft_amplitude = _load_fft_amplitude(cascade_dir)
    fmin, fmax = bands[0][0], bands[-1][1]

    n_silent = 0
    rows = {}
    for i, roi in enumerate(roi_ids):
        row = {}
        for name, mask in masks.items():
            seg = spike_rate[i, mask]
            # fft_amplitude() detrends (linear) and Hann-windows internally; its detrend
            # step also interpolates over the NaN padding at the trace ends.
            freqs, amps = fft_amplitude(seg, frame_rate, "linear", "hann")
            power = amps ** 2
            band = (freqs >= fmin) & (freqs <= fmax)
            total = power[band].sum()

            if not np.isfinite(total) or total <= 0:
                n_silent += 1
                for lo, hi in bands:
                    row[f"bp{lo}-{hi}_{name}"] = 0.0
                row[f"domfreq_{name}"] = 0.0
                continue

            for j, (lo, hi) in enumerate(bands):
                last = j == len(bands) - 1
                sel = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
                row[f"bp{lo}-{hi}_{name}"] = float(power[sel].sum() / total)
            row[f"domfreq_{name}"] = float(freqs[band][np.argmax(power[band])])
        rows[roi] = row

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "roi"
    if n_silent:
        print(f"! {n_silent} ROI x phase cell(s) had no inferred activity; band powers are "
              "0 and dominant frequency reads 0 Hz (= undefined, not slow).")
    return out
