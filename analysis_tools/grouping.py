"""PCA + Ward hierarchical clustering, shared by Index A and Index B.

Same pipeline, different feature table, separate fit -- the two indices are never
pooled into one decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_and_linkage(features, var_target=0.80):
    """z-score -> PCA -> Ward linkage on the leading PCs.

    Returns (pca, scores, n_pc, Z).
    """
    X = StandardScaler().fit_transform(features.to_numpy(dtype=float))
    pca = PCA().fit(X)
    scores = pca.transform(X)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
    return pca, scores, n_pc, linkage(scores[:, :n_pc], method="ward")


def cut_candidates(Z, max_k=8):
    """Every cut from k=2..max_k with its height, merge gap and resulting group sizes.

    Printed so that "cut by inspection" is an informed choice rather than a guess.
    """
    rows = []
    for k in range(2, max_k + 1):
        lo, hi = Z[-k, 2], Z[-(k - 1), 2]
        height = 0.5 * (lo + hi)
        sizes = np.bincount(fcluster(Z, height, criterion="distance"))[1:]
        rows.append({"k": k, "height": round(float(height), 2), "gap": round(float(hi - lo), 2),
                     "smallest": int(sizes.min()), "largest": int(sizes.max()),
                     "sizes": list(map(int, sorted(sizes)[::-1]))})
    return pd.DataFrame(rows).set_index("k")


def suggest_cut(Z, max_k=8, min_group_size=3, max_group_frac=0.9):
    """Largest merge gap near the top of the tree, among cuts that actually partition.

    Two guards, because the plain largest-gap rule reliably returns a non-answer: it
    tends to slice off one extreme ROI and call the remaining 95% a single group. A cut
    is eligible only if every group holds >= `min_group_size` ROIs AND no group holds
    more than `max_group_frac` of them -- the same criterion used to declare a partition
    degenerate, applied up front instead of as a post-hoc complaint.
    """
    cand = cut_candidates(Z, max_k)
    n_rois = int(sum(cand.iloc[0]["sizes"]))
    usable = cand[(cand["smallest"] >= min_group_size)
                  & (cand["largest"] <= max_group_frac * n_rois)]
    if usable.empty:
        print(f"! No cut in k=2..{max_k} gives groups of >= {min_group_size} ROIs with none "
              f"exceeding {max_group_frac:.0%} of the population; falling back to the plain "
              "largest-gap cut (expect a degenerate partition).")
        usable = cand
    return float(usable.loc[usable["gap"].idxmax(), "height"])


def cut_tree(Z, cut_height, roi_ids, max_group_frac=0.9, label="rate", prefix="G"):
    """Apply a cut and flag the degenerate case, where one group holds nearly everything."""
    groups = fcluster(Z, cut_height, criterion="distance")
    sizes = np.bincount(groups)[1:]
    print(f"> Cut at h={cut_height:.2f} -> {len(sizes)} group(s): "
          + ", ".join(f"{prefix}{g + 1} n={int(n)}" for g, n in enumerate(sizes)))

    if sizes.max() / len(roi_ids) > max_group_frac:
        outliers = [str(roi_ids[i]) for i in np.flatnonzero(groups != (sizes.argmax() + 1))]
        print(f"! DEGENERATE PARTITION: {sizes.max()}/{len(roi_ids)} ROIs fall in one group; "
              f"the tree only separates outlier ROI(s) {', '.join(outliers)}.")
        print(f"! Read this as 'the {label} features carry no multi-group structure in this "
              "recording', NOT as 'there is one big responsive group'.")
    return groups


def print_membership(groups, roi_ids, prefix="G"):
    for g in range(1, int(groups.max()) + 1):
        members = [str(roi_ids[i]) for i in np.flatnonzero(groups == g)]
        print(f"  {prefix}{g}: ROIs {', '.join(members)}")


def compare_partitions(groups_a, groups_b, name_a="rate group", name_b="freq group"):
    """Contingency table + adjusted Rand index between two groupings of the same ROIs."""
    from sklearn.metrics import adjusted_rand_score

    table = pd.crosstab(
        pd.Series([f"G{g}" for g in groups_a], name=name_a),
        pd.Series([f"F{g}" for g in groups_b], name=name_b),
    )
    ari = adjusted_rand_score(groups_a, groups_b)
    print(table.to_string())
    print(f"\n> Adjusted Rand index = {ari:.3f}  "
          "(1 = identical partitions, ~0 = no more agreement than chance)")
    print("> High agreement: when a neuron responds also predicts how fast it fluctuates.")
    print("> Low agreement: the two indices carry independent information.")
    return table, ari
