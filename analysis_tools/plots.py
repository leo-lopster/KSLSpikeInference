"""Figures for §5.

Conventions (CLAUDE.md 5): perceptually-uniform colormaps only (viridis/magma, never
jet); any panel showing inferred spikes names the algorithm and its parameters in the
same panel.

Every function builds its figure and returns it, so a caller can hand it to
`store.save_figure` for export without rebuilding the plot.

Figures are built with `matplotlib.figure.Figure` directly, never through pyplot. That
keeps them out of pyplot's global figure manager -- which matters because the GUI embeds
these same figures in Qt canvases (`FigureCanvasQTAgg`) for its live preview, and a
pyplot-owned figure would leak between the two lifecycles and drag pyplot's thread
affinity into the Qt event loop.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram

# Qualitative colours for group overlays: Tableau 10, matplotlib's `tab10`.
#
# Pinned to the colormap rather than written as the 'C0'..'C9' property-cycle shorthand.
# The default cycle IS tab10, so the two agree today -- but the cycle follows rcParams,
# and a style sheet, a seaborn import or a matplotlib default change would repaint every
# figure while the exported group mask, whose colours are resolved from this same list,
# kept the old ones. The point of the mask is that its colours match the figures, so the
# palette has to be a fixed fact rather than an ambient setting.
#
# Cycled, so k > 10 repeats rather than failing; beyond 10 groups the dendrogram, the
# scatter, the group means and the mask still agree with each other, but two different
# groups share a colour.
GROUP_COLORS = [to_hex(c) for c in colormaps["tab10"].colors]


def group_color(g):
    """Colour for group `g` (1-based), matching across every panel that shows groups."""
    return GROUP_COLORS[(int(g) - 1) % len(GROUP_COLORS)]


def group_rgb(g):
    """`group_color` as an 8-bit (R, G, B) triple, for painting colour into an image.

    The single source of truth for group colour is `GROUP_COLORS`, and everything --
    figures and exported mask alike -- resolves through here or `group_color`, which is
    what keeps a group the same colour in a TIFF as it is in the trace panels.
    """
    return tuple(int(round(255 * v)) for v in to_rgb(group_color(g)))


def pca_summary(pca, scores, features, roi_ids, title, var_target=0.80, groups=None):
    """Scree, PC1/PC2 loadings, and the ROIs in PC space.

    `groups` optionally colours the PC-space scatter by cluster assignment.
    """
    var = pca.explained_variance_ratio_ * 100
    fig = Figure(figsize=(16, 4.4))
    axes = fig.subplots(1, 3)

    axes[0].bar(np.arange(1, len(var) + 1), var, color="0.5")
    axes[0].plot(np.arange(1, len(var) + 1), np.cumsum(var), marker="o", ms=3, lw=1, color="C1")
    axes[0].axhline(var_target * 100, ls="--", lw=0.8, color="C3")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Scree (bars) + cumulative (line)")

    y = np.arange(len(features.columns))
    axes[1].barh(y - 0.2, pca.components_[0], height=0.4, label=f"PC1 ({var[0]:.0f}%)")
    axes[1].barh(y + 0.2, pca.components_[1], height=0.4, label=f"PC2 ({var[1]:.0f}%)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(features.columns, fontsize=7)
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Loading")
    axes[1].set_title("What each PC is made of")
    axes[1].legend(fontsize=8)

    _scatter_pcs(axes[2], scores, roi_ids, var, groups)
    axes[2].set_title("ROIs in PC space")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def _scatter_pcs(ax, scores, roi_ids, var, groups=None):
    """PC1/PC2 scatter with every ROI labelled, optionally coloured by group."""
    if groups is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=18, color="0.3")
    else:
        groups = np.asarray(groups)
        for g in range(1, int(groups.max()) + 1):
            sel = groups == g
            ax.scatter(scores[sel, 0], scores[sel, 1], s=26, color=group_color(g),
                       label=f"G{g} (n={int(sel.sum())})")
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
    for i, roi in enumerate(roi_ids):
        ax.annotate(str(roi), (scores[i, 0], scores[i, 1]), fontsize=5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")


def pca_scatter(pca, scores, roi_ids, groups, title):
    """Just the PC-space scatter, coloured by group -- the live-preview panel.

    Separate from `pca_summary` because the preview redraws this on every k change and
    the scree/loading panels do not depend on the cut at all.
    """
    var = pca.explained_variance_ratio_ * 100
    fig = Figure(figsize=(6.4, 4.6))
    ax = fig.subplots()
    _scatter_pcs(ax, scores, roi_ids, var, groups)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def dendrogram_plot(Z, roi_ids, title, cut):
    fig = Figure(figsize=(13, 4.6))
    ax = fig.subplots()
    dendrogram(Z, labels=[str(r) for r in roi_ids], leaf_font_size=6,
               color_threshold=cut, above_threshold_color="0.65", ax=ax)
    ax.axhline(cut, ls="--", lw=1, color="C3")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Ward merge cost")
    ax.set_title(f"{title}  (cut at h={cut:.2f})")
    fig.tight_layout()
    return fig


def rate_heatmap(spike_rate, t, roi_ids, groups, stim_window, title):
    """Inferred rate per ROI, rows sorted by group, group boundaries drawn in white.

    `stim_window` may be None -- a region analysis has no stimulus phase to mark out.
    """
    order = np.argsort(groups, kind="stable")
    fig = Figure(figsize=(12, 0.18 * len(roi_ids) + 2.2))
    ax = fig.subplots()
    im = ax.imshow(spike_rate[order], aspect="auto", cmap="magma", interpolation="nearest",
                   vmin=0, vmax=np.nanpercentile(spike_rate, 99),
                   extent=[t[0], t[-1], len(roi_ids) - 0.5, -0.5])
    for boundary in np.flatnonzero(np.diff(groups[order])) + 0.5:
        ax.axhline(boundary, color="w", lw=1.2)
    for edge in stim_window or ():
        ax.axvline(edge, color="w", ls="--", lw=0.9)
    ax.set_yticks(range(len(roi_ids)))
    ax.set_yticklabels([f"{roi_ids[i]} (G{groups[i]})" for i in order], fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI (grouped)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Inferred rate (spikes/s)")
    fig.tight_layout()
    return fig


def group_means(spike_rate, t, groups, roi_ids, stim_window, pad, title,
                show_rois=True, roi_alpha=0.3):
    """Group-mean inferred rate ± SEM, over each member ROI's own trace in grey.

    The individual traces are the point of the panel: a mean ± SEM alone hides whether a
    group is coherent or is one loud ROI dragging a quiet majority. Restricted to the
    predictable span -- the padded ends are all-NaN and carry no mean at all.

    y-axes are independent: a shared scale would flatten every low-rate group to a
    straight line once a high-rate group is present.

    `stim_window` may be None, and is for a region analysis: the window IS the region, so
    shading it would grey out every panel edge to edge and mark nothing off against
    anything.
    """
    n_groups = int(groups.max())
    valid = slice(pad, len(t) - pad) if pad else slice(None)
    t_valid = t[valid]

    fig = Figure(figsize=(12, 1.9 * n_groups + 1.2))
    axes = fig.subplots(n_groups, 1, sharex=True)
    for ax, g in zip(np.atleast_1d(axes), range(1, n_groups + 1)):
        members = np.flatnonzero(groups == g)
        sel = spike_rate[members][:, valid]
        color = group_color(g)
        if stim_window is not None:
            ax.axvspan(*stim_window, color="0.88", zorder=0)

        if show_rois:
            for row in sel:
                ax.plot(t_valid, row, lw=0.5, color="0.45", alpha=roi_alpha, zorder=1)

        mean = np.nanmean(sel, axis=0)
        sem = np.nanstd(sel, axis=0) / np.sqrt(max(sel.shape[0], 1))
        ax.fill_between(t_valid, mean - sem, mean + sem, alpha=0.35, color=color, lw=0, zorder=2)
        ax.plot(t_valid, mean, lw=1.2, color=color, zorder=3)

        ax.set_ylabel(f"G{g}\n(n={len(members)})", rotation=0, ha="right", va="center",
                      fontsize=8, color=color)
        ax.margins(x=0)

    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    subtitle = ("grey = individual ROIs, coloured = group mean ± SEM (colour matches the "
                "PC-space scatter)")
    if stim_window is not None:
        subtitle += ", shaded band = stimulus"
    fig.suptitle(f"{title}\n{subtitle}", y=0.995)
    fig.tight_layout()
    return fig
