"""Figures for §5.

Conventions (CLAUDE.md 5): perceptually-uniform colormaps only (viridis/magma, never
jet); any panel showing inferred spikes names the algorithm and its parameters in the
same panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def pca_summary(pca, scores, features, roi_ids, title, var_target=0.80):
    """Scree, PC1/PC2 loadings, and the ROIs in PC space."""
    var = pca.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

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

    axes[2].scatter(scores[:, 0], scores[:, 1], s=18, color="0.3")
    for i, roi in enumerate(roi_ids):
        axes[2].annotate(str(roi), (scores[i, 0], scores[i, 1]), fontsize=5,
                         xytext=(2, 2), textcoords="offset points")
    axes[2].set_xlabel(f"PC1 ({var[0]:.0f}%)")
    axes[2].set_ylabel(f"PC2 ({var[1]:.0f}%)")
    axes[2].set_title("ROIs in PC space")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def dendrogram_plot(Z, roi_ids, title, cut):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    dendrogram(Z, labels=[str(r) for r in roi_ids], leaf_font_size=6,
               color_threshold=cut, above_threshold_color="0.65", ax=ax)
    ax.axhline(cut, ls="--", lw=1, color="C3")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Ward merge cost")
    ax.set_title(f"{title}  (cut at h={cut:.2f})")
    plt.tight_layout()
    plt.show()


def rate_heatmap(spike_rate, t, roi_ids, groups, stim_window, title):
    """Inferred rate per ROI, rows sorted by group, group boundaries drawn in white."""
    order = np.argsort(groups, kind="stable")
    fig, ax = plt.subplots(figsize=(12, 0.18 * len(roi_ids) + 2.2))
    im = ax.imshow(spike_rate[order], aspect="auto", cmap="magma", interpolation="nearest",
                   vmin=0, vmax=np.nanpercentile(spike_rate, 99),
                   extent=[t[0], t[-1], len(roi_ids) - 0.5, -0.5])
    for boundary in np.flatnonzero(np.diff(groups[order])) + 0.5:
        ax.axhline(boundary, color="w", lw=1.2)
    for edge in stim_window:
        ax.axvline(edge, color="w", ls="--", lw=0.9)
    ax.set_yticks(range(len(roi_ids)))
    ax.set_yticklabels([f"{roi_ids[i]} (G{groups[i]})" for i in order], fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI (grouped)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Inferred rate (spikes/s)")
    plt.tight_layout()
    plt.show()


def group_means(spike_rate, t, groups, roi_ids, stim_window, pad, title,
                show_rois=True, roi_alpha=0.3):
    """Group-mean inferred rate ± SEM, over each member ROI's own trace in grey.

    The individual traces are the point of the panel: a mean ± SEM alone hides whether a
    group is coherent or is one loud ROI dragging a quiet majority. Restricted to the
    predictable span -- the padded ends are all-NaN and carry no mean at all.

    y-axes are independent: a shared scale would flatten every low-rate group to a
    straight line once a high-rate group is present.
    """
    n_groups = int(groups.max())
    valid = slice(pad, len(t) - pad) if pad else slice(None)
    t_valid = t[valid]

    fig, axes = plt.subplots(n_groups, 1, sharex=True, figsize=(12, 1.9 * n_groups + 1.2))
    for ax, g in zip(np.atleast_1d(axes), range(1, n_groups + 1)):
        members = np.flatnonzero(groups == g)
        sel = spike_rate[members][:, valid]
        ax.axvspan(*stim_window, color="0.88", zorder=0)

        if show_rois:
            for row in sel:
                ax.plot(t_valid, row, lw=0.5, color="0.45", alpha=roi_alpha, zorder=1)

        mean = np.nanmean(sel, axis=0)
        sem = np.nanstd(sel, axis=0) / np.sqrt(max(sel.shape[0], 1))
        ax.fill_between(t_valid, mean - sem, mean + sem, alpha=0.35, color="C0", lw=0, zorder=2)
        ax.plot(t_valid, mean, lw=1.2, color="C0", zorder=3)

        ax.set_ylabel(f"G{g}\n(n={len(members)})", rotation=0, ha="right", va="center", fontsize=8)
        ax.margins(x=0)

    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    subtitle = "grey = individual ROIs, blue = group mean ± SEM, shaded band = stimulus"
    fig.suptitle(f"{title}\n{subtitle}", y=0.995)
    plt.tight_layout()
    plt.show()
