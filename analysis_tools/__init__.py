"""Analysis helpers for the §5 phase-conditioned grouping in process_using_napari.ipynb.

Split out of the notebook so the heavy blocks (CASCADE inference, feature building,
PCA/clustering, plotting) can be version-controlled, diffed and reused, leaving the
notebook as a thin driver over one block of constants.

Every function takes its parameters explicitly -- nothing reads notebook globals -- so
the constants stay in one place (the notebook's config cell) rather than being spread
across module defaults.
"""

from . import cascade_runner, features, grouping, plots, preprocess, store, traces

__all__ = ["cascade_runner", "features", "grouping", "plots", "preprocess", "store", "traces"]
