"""PyQt front-end for the DRG/DCN spike-inference pipeline.

Runs the same workflow as process_using_napari.ipynb -- open an .imgdir, preprocess,
paint/load ROIs, extract dF/F, run CASCADE, then features/PCA/grouping and export -- with
napari opening alongside for the image and ROI work.

    python spike_inference_gui.py

Every option is documented in CascadeTorch/README.md (the "Local fork" section, which
replaced GUI_OPTIONS_SPEC.md), and every file operation goes through
analysis_tools/, so the GUI and the notebook share one implementation. Long jobs run on a
worker thread and stream their stdout into the log pane at the bottom.

Not included, deliberately: max/STD projection rendering. The ROI layer is therefore
sized against the preprocessed stack itself rather than a projection layer.
"""

from __future__ import annotations

import io
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")  # figures are exported to .tiff, never drawn into the Qt event loop

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import napari  # noqa: E402
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

import analysis_tools as at  # noqa: E402

REPO = Path(__file__).resolve().parent
LAYER_RAW = "raw_stack"
LAYER_PREPROCESSED = "preprocessed"
LAYER_LABELS = "ROI labels"


# =============================================================================
# Session state
# =============================================================================

@dataclass
class Session:
    """Everything the pipeline carries between stages.

    `mode` is the gate: "full" enters at the images, "traces" enters at an imported dF/F
    table and has no pixel data behind it, so the image/preprocessing/ROI stages are
    absent rather than merely stale.
    """

    mode: str = "full"

    # stage 1-2: images
    data_dir: Path | None = None
    tag: str = ""                     # set when a .zarr is loaded without its source .imgdir
    image_files: list = field(default_factory=list)
    frame_shape: tuple | None = None
    times_s: np.ndarray | None = None
    acq_rate: float | None = None
    time_source: str = ""
    # Two stacks, kept apart on purpose: `raw_stack` is what came off disk, `stack` is the
    # preprocessed one every later stage uses. Collapsing them let ROI extraction run on
    # raw frames whenever preprocessing had not been built yet.
    raw_stack: object = None          # dask array, lazy
    stack: object = None              # dask array, lazy, preprocessed
    stack_source: Path | None = None  # the .zarr `stack` reads from, if any

    # stage 3-4: ROIs and traces
    roi_labels: np.ndarray | None = None
    raw_traces: dict | None = None
    roi_ids: list | None = None
    dff_matrix: np.ndarray | None = None
    t_plot: np.ndarray | None = None
    trace_params: dict = field(default_factory=dict)

    # stage 5: inference
    spike_rate: np.ndarray | None = None
    pad: int = 0

    @property
    def dataset_tag(self):
        if self.data_dir:
            return at.store.dataset_tag(self.data_dir)
        return self.tag  # a .zarr loaded on its own carries the tag in its filename

    @property
    def frame_rate(self):
        if self.t_plot is not None and len(self.t_plot) > 1:
            return at.store.frame_rate(self.t_plot)
        return self.acq_rate


# =============================================================================
# Worker plumbing -- one job at a time, stdout mirrored into the log
# =============================================================================

class JobSignals(QObject):
    message = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)


class _LogStream(io.TextIOBase):
    """Feeds everything the analysis_tools functions print into the log pane."""

    def __init__(self, emit):
        self._emit = emit

    def write(self, text):
        if text.strip():
            self._emit(text.rstrip("\n"))
        return len(text)

    def flush(self):
        pass


class Job(QRunnable):
    """Runs one callable off the UI thread, capturing its stdout."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.signals = JobSignals()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    @Slot()
    def run(self):
        import contextlib

        stream = _LogStream(self.signals.message.emit)
        try:
            with contextlib.redirect_stdout(stream):
                result = self._fn(*self._args, **self._kwargs)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)


# =============================================================================
# Small widget helpers
# =============================================================================

def _spin(lo, hi, value, step=1, decimals=None):
    box = QSpinBox() if decimals is None else QDoubleSpinBox()
    if decimals is not None:
        box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _path_row(line_edit, on_browse, button_text="Browse…"):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit, 1)
    button = QPushButton(button_text)
    button.clicked.connect(on_browse)
    layout.addWidget(button)
    return row


def _info(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _caption(text):
    label = _info(text)
    label.setStyleSheet("color: palette(mid); font-size: 11px;")
    return label


# =============================================================================
# Main window
# =============================================================================

class PipelineWindow(QMainWindow):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.session = Session()
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(1)  # one job at a time; stdout capture is global
        self._busy = False
        self.roi_labels_path = None  # set by the load/save dialogs in tab 3
        # The last CASCADE run, kept so it can be written to a folder chosen after the
        # fact. Cleared when a rate is loaded from disk instead of inferred -- that one
        # is already saved somewhere, and re-saving it would only duplicate it.
        self._cascade_result = None
        self._cascade_extra = None

        self.setWindowTitle("Spike Inference Pipeline")
        self.resize(760, 940)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dataset_tab(), "1 · Dataset")
        self.tabs.addTab(self._build_preprocess_tab(), "2 · Preprocess")
        self.tabs.addTab(self._build_roi_tab(), "3 · ROIs")
        self.tabs.addTab(self._build_traces_tab(), "4 · Traces")
        self.tabs.addTab(self._build_inference_tab(), "5 · Spike inference")
        self.tabs.addTab(self._build_analysis_tab(), "6 · Analysis")

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: monospace; font-size: 11px;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("Ready.")

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 0, 6, 6)
        bottom_layout.addWidget(QLabel("Log"))
        bottom_layout.addWidget(self.log)
        bottom_layout.addWidget(self.progress)
        bottom_layout.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._refresh_gating()
        self._log("Ready. \n"
                  "Available processing options: \n"
                  "1) Open an .imgdir in tab 1 to start analysis from scratch \n"
                  "2) Open a preprocessed .zarr stack in tab 2 and add labels in tab 3 \n"
                  "3) Import dF/F traces in tab 4 \n")

    # ------------------------------------------------------------------ tab 1
    def _build_dataset_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("Dataset (point 1 — .npy frames from an .imgdir)")
        form = QFormLayout(box)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("(none selected) — e.g. ./datasets/<name>.imgdir")
        self.data_dir_edit.textChanged.connect(self._refresh_gating)
        form.addRow("Data Directory", _path_row(self.data_dir_edit, self._browse_data_dir))
        self.channel_spin = _spin(0, 8, 0)
        form.addRow("Calcium Indicator Channel", self.channel_spin)
        self.fallback_hz_spin = _spin(0.01, 10000.0, 10.0, 0.5, decimals=2)
        form.addRow("Fallback Freq. (Hz)", self.fallback_hz_spin)
        form.addRow(_caption("fallback_hz is used only when ElapsedTimes.yaml is missing or "
                             "its length disagrees with the stack. The warning appears in "
                             "the log — do not ignore it."))
        layout.addWidget(box)

        self.load_dataset_btn = QPushButton("Load dataset")
        self.load_dataset_btn.clicked.connect(self._on_load_dataset)
        layout.addWidget(self.load_dataset_btn)

        self.dataset_info = _info("No dataset loaded.")
        layout.addWidget(self.dataset_info)
        layout.addStretch(1)
        return page

    def _browse_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select an .imgdir",
                                                self._dialog_start(self.data_dir_edit))
        if path:
            self.data_dir_edit.setText(path)

    def _on_load_dataset(self):
        data_dir = Path(self.data_dir_edit.text()).expanduser()
        channel = self.channel_spin.value()
        fallback = self.fallback_hz_spin.value()

        def work():
            files = at.store.list_frames(data_dir, channel)
            stack = at.store.load_imgdir(data_dir, channel)
            times, rate, source = at.store.load_timebase(data_dir, stack.shape[0], fallback)
            return files, stack, times, rate, source

        def done(result):
            files, stack, times, rate, source = result
            s = self.session
            s.data_dir, s.image_files, s.raw_stack = data_dir, files, stack
            s.tag = ""
            s.frame_shape = tuple(stack.shape[1:])
            s.times_s, s.acq_rate, s.time_source = times, rate, source
            self._add_layer(LAYER_RAW, stack)
            self.dataset_info.setText(
                f"<b>{data_dir.name}</b><br>{len(files)} timepoints · frame {s.frame_shape} "
                f"{stack.dtype}<br>time axis: {source} · ~{rate:.2f} Hz<br>"
                f"dataset_tag: <b>{s.dataset_tag}</b>")
            self._sync_derived_paths()
            self._refresh_gating()

        self._run(work, done, "Loading dataset…")

    # ------------------------------------------------------------------ tab 2
    def _build_preprocess_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("Denoise chain")
        form = QFormLayout(box)
        self.denoise_checks = {}
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for name in at.preprocess.DENOISE_METHODS:
            check = QCheckBox(name)
            check.setChecked(name == "nlm")
            # connect after setChecked: the signal would otherwise fire into
            # _sync_derived_paths before the widgets it reads exist
            check.stateChanged.connect(self._sync_derived_paths)
            self.denoise_checks[name] = check
            row_layout.addWidget(check)
        row_layout.addStretch(1)
        form.addRow("Denoise Method", row)
        # form.addRow(_caption("Applied in sequence, in the fixed order gaussian → nlm → dct.")) ## DCT is Deprecated
        form.addRow(_caption("Applied in sequence, in the fixed order gaussian → nlm"))

        self.sigma_spin = _spin(0.0, 50.0, 3.0, 0.1, decimals=2)
        form.addRow("[Gaussian] Sigma", self.sigma_spin)
        self.nlm_patch_spin = _spin(1, 51, 5)
        form.addRow("[NLM] Patch Size", self.nlm_patch_spin)
        self.nlm_dist_spin = _spin(1, 51, 6)
        form.addRow("[NLM] Patch Distance", self.nlm_dist_spin)
        self.nlm_h_spin = _spin(0.0, 20.0, 1.25, 0.05, decimals=2)
        form.addRow("[NLM] H-Factor", self.nlm_h_spin)
        # self.dct_spin = _spin(0.0, 1.0, 0.05, 0.01, decimals=3)
        # form.addRow("[DCT] Threshold Fraction", self.dct_spin)
        # self.bg_spin = _spin(0.0, 100.0, 4.0, 0.5, decimals=1)
        # form.addRow("Background Percentile", self.bg_spin)
        # self.upsample_spin = _spin(1, 100, 10)
        # form.addRow("Upsample Factor", self.upsample_spin)
        layout.addWidget(box)

        io_box = QGroupBox("Zarr cache")
        io_form = QFormLayout(io_box)
        self.preprocessed_dir_edit = QLineEdit()
        self.preprocessed_dir_edit.setPlaceholderText("(none selected) — e.g. ./preprocessed")
        self.preprocessed_dir_edit.textChanged.connect(self._refresh_gating)
        io_form.addRow("PREPROCESSED_DIR",
                       _path_row(self.preprocessed_dir_edit, self._browse_preprocessed_dir))
        self.zarr_info = _info()
        io_form.addRow("resolved path", self.zarr_info)
        io_form.addRow(_caption("The denoise chain is part of the filename. Changing it "
                                "retargets which store is saved AND loaded — check the "
                                "resolved path before pressing anything."))
        layout.addWidget(io_box)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build lazy stack")
        self.build_btn.clicked.connect(self._on_build_stack)
        self.save_zarr_btn = QPushButton("Save to Zarr…")
        self.save_zarr_btn.clicked.connect(self._on_save_zarr)
        self.load_zarr_btn = QPushButton("Load from Zarr")
        self.load_zarr_btn.clicked.connect(self._on_load_zarr)
        for button in (self.build_btn, self.save_zarr_btn, self.load_zarr_btn):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        layout.addWidget(_caption("‘Build lazy stack’ only wires the chain up and shows it in "
                                  "napari; nothing is computed until napari draws a frame or "
                                  "you save. Saving computes every frame and is slow."))
        layout.addStretch(1)
        return page

    def _browse_preprocessed_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Preprocessed directory",
                                                self._dialog_start(self.preprocessed_dir_edit))
        if path:
            self.preprocessed_dir_edit.setText(path)

    def _denoise_method(self):
        chosen = [n for n in at.preprocess.DENOISE_METHODS if self.denoise_checks[n].isChecked()]
        return ",".join(chosen)

    def _preprocess_params(self):
        return at.preprocess.PreprocessParams(
            denoise_method=self._denoise_method(),
            denoise_sigma=self.sigma_spin.value(),
            nlm_patch_size=self.nlm_patch_spin.value(),
            nlm_patch_distance=self.nlm_dist_spin.value(),
            nlm_h_factor=self.nlm_h_spin.value(),
            # dct_threshold_fraction=self.dct_spin.value(),
            # background_percentile=self.bg_spin.value(),
            # upsample_factor=self.upsample_spin.value(),
        )

    def _zarr_path(self):
        method = self._denoise_method()
        if not (self.session.data_dir and method):
            return None
        out_dir = self._dir(self.preprocessed_dir_edit)
        if out_dir is None:
            return None
        return out_dir / f"{self.session.dataset_tag}_{at.store.method_tag(method)}.zarr"

    def _on_build_stack(self):
        params = self._preprocess_params()
        files = self.session.image_files

        def work():
            return at.preprocess.build_stack(files, params)

        def done(stack):
            self.session.stack = stack
            self.session.stack_source = None  # a fresh chain, not backed by any store
            self._add_layer(f"{LAYER_PREPROCESSED}_{params.denoise_method}", stack)
            self._refresh_gating()
            self._prompt_save_zarr()

        try:
            params.methods()
        except ValueError as exc:
            return self._warn(str(exc))
        self._run(work, done, "Building lazy preprocessing chain…")

    def _prompt_save_zarr(self):
        """Offer to write a freshly built chain to disk.

        Asked rather than assumed because the write computes every frame and is slow. But
        it is asked at all because a built chain is lazy: nothing has been computed yet,
        trace extraction reads from a saved store, and leaving it unsaved means the whole
        preprocessing cost gets paid again later, per frame.
        """
        path = self._zarr_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self, "Save preprocessed stack?",
            f"A new preprocessing chain is built but nothing is computed or stored yet.\n\n"
            f"Trace extraction (tab 4) reads from a saved .zarr store, so this chain has "
            f"to be written before it can be used downstream.\n\n"
            f"Save it now to {path.name}?\n\n"
            f"This computes every frame and can take a long time.")
        if answer == QMessageBox.StandardButton.Yes:
            self._on_save_zarr()
        else:
            self._log("> Not saved. Build again or press 'Save to Zarr…' when ready; "
                      "extraction needs the store on disk.")

    def _on_save_zarr(self):
        path = self._zarr_path()
        if path is None:
            return self._warn("Select a dataset and at least one denoise method first.")
        if path.exists():
            answer = QMessageBox.question(
                self, "Overwrite?",
                f"{path.name} already exists and will be overwritten.\n\n"
                "This recomputes every frame and can take a long time. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                return

        if self.session.stack_source is not None and self.session.stack_source == path:
            # Saving adopts the store, so the session now READS the target. Writing it
            # again would compute the source from a store that overwrite=True has just
            # deleted -- the file survives but its contents come out zeroed.
            self._log(f"> {path.name} is already saved and is what this session reads; "
                      "nothing to write. Build again to make a new chain.")
            return
        params = self._preprocess_params()
        sidecar = {"dataset": self.session.data_dir.name if self.session.data_dir
                   else self.session.dataset_tag, **params.to_sidecar()}
        stack, tag = self.session.stack, self.session.dataset_tag
        out_dir = self._dir(self.preprocessed_dir_edit)
        method_tag = at.store.method_tag(params.denoise_method)

        def work():
            at.store.save_preprocessed(stack, out_dir, tag, method_tag, sidecar)
            # Adopt the store we just wrote. Left alone, the session would keep the lazy
            # delayed chain, and every later frame read -- all 500 of them during ROI
            # extraction -- would silently re-run denoising and motion correction instead
            # of reading the cache. Saving therefore also loads.
            return at.store.load_preprocessed(out_dir, tag, method_tag)

        def done(result):
            cached, _params = result
            self.session.stack = cached
            self.session.stack_source = out_dir / f"{tag}_{method_tag}.zarr"
            self._add_layer(f"{LAYER_PREPROCESSED}_{method_tag}", cached)
            self.extract_zarr_edit.setText(str(out_dir / f"{tag}_{method_tag}.zarr"))
            self._log("> Session now reads the saved store; preprocessing will not re-run.")
            self._sync_derived_paths()
            self._refresh_gating()

        self._run(work, done, "Writing Zarr (computes every frame)…")

    def _on_load_zarr(self):
        """Load a cached preprocessed stack, choosing the store in a dialog.

        Available from a cold start: a .zarr is a complete entry point on its own, so this
        does not require a dataset to have been opened first. The tag and denoise chain are
        read back out of the store's name (`<dataset_tag>_<method_tag>.zarr`) rather than
        assumed from the current controls, which is what keeps the loaded stack and the UI
        describing the same thing.
        """
        suggested = self._zarr_path()
        start = str(suggested) if suggested else self._dialog_start(self.preprocessed_dir_edit)
        chosen = QFileDialog.getExistingDirectory(self, "Load preprocessed .zarr store", start)
        if not chosen:
            return  # cancelled
        store = Path(chosen)
        if store.suffix != ".zarr":
            return self._warn(f"{store.name} is not a .zarr store. Pick the "
                              "<dataset_tag>_<method>.zarr folder itself.")
        if "_" not in store.stem:
            return self._warn(f"Cannot read a dataset tag and denoise chain out of "
                              f"{store.name}; expected <dataset_tag>_<method>.zarr.")
        tag, method_tag = store.stem.rsplit("_", 1)
        out_dir = store.parent

        def work():
            return at.store.load_preprocessed(out_dir, tag, method_tag)

        def done(result):
            stack, params = result
            s = self.session
            s.stack = stack
            s.stack_source = store
            if not s.data_dir:
                s.tag = tag
                s.frame_shape = tuple(stack.shape[1:])
            elif tag != s.dataset_tag:
                self._log(f"! Loaded store is tagged '{tag}' but the open dataset is "
                          f"'{s.dataset_tag}'. Outputs will be written under "
                          f"'{s.dataset_tag}'.")
            self.preprocessed_dir_edit.setText(str(out_dir))
            self.extract_zarr_edit.setText(str(store))
            self._apply_denoise_method(params.get("DENOISE_METHOD", method_tag.replace("-", ",")))
            self._add_layer(f"{LAYER_PREPROCESSED}_{method_tag}", stack)
            self._refresh_gating()

        self._run(work, done, "Loading Zarr…")

    def _apply_denoise_method(self, method):
        """Point the checkboxes at the chain a loaded store was actually built with."""
        chosen = {m.strip() for m in str(method).replace("-", ",").split(",") if m.strip()}
        for name, check in self.denoise_checks.items():
            check.blockSignals(True)
            check.setChecked(name in chosen)
            check.blockSignals(False)

    # ------------------------------------------------------------------ tab 3
    def _build_roi_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("ROI labels (point 2 — uint16 .tiff label image)")
        form = QFormLayout(box)
        self.labels_dir_edit = QLineEdit()
        self.labels_dir_edit.setPlaceholderText("(none selected) — e.g. ./labels")
        self.labels_dir_edit.textChanged.connect(self._refresh_gating)
        form.addRow("LABELS_DIR", _path_row(self.labels_dir_edit, self._browse_labels_dir))
        form.addRow(_caption("LABELS_DIR only sets where the load/save dialogs open. The labels "
                             "path itself is chosen in those dialogs and recorded with the "
                             "traces, so it always reflects the file actually used."))
        layout.addWidget(box)

        buttons = QHBoxLayout()
        self.new_labels_btn = QPushButton("New blank ROI labels")
        self.new_labels_btn.clicked.connect(self._on_new_labels)
        self.save_labels_btn = QPushButton("Save ROI labels")
        self.save_labels_btn.clicked.connect(self._on_save_labels)
        self.load_labels_btn = QPushButton("Load ROI labels")
        self.load_labels_btn.clicked.connect(self._on_load_labels)
        for button in (self.new_labels_btn, self.save_labels_btn, self.load_labels_btn):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.roi_info = _info("No ROI labels loaded.")
        layout.addWidget(self.roi_info)
        layout.addWidget(_caption(
            "Paint ROIs on the 'ROI labels' layer in napari with the brush tool; press M for a "
            "fresh label id per ROI. The labels are sized against the preprocessed stack — this "
            "build renders no max/STD projection."))
        layout.addStretch(1)
        return page

    def _browse_labels_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Labels directory",
                                                self._dialog_start(self.labels_dir_edit))
        if path:
            self.labels_dir_edit.setText(path)

    def _labels_dialog_start(self):
        """Where the ROI-label dialogs open: last file used, else LABELS_DIR, else the repo.

        The suggested filename follows the `<dataset_tag>_ROI.tiff` convention, but it is
        only a suggestion -- what the dialog returns is what gets used.
        """
        if self.roi_labels_path is not None:
            return str(self.roi_labels_path)
        labels_dir = self._dir(self.labels_dir_edit) or REPO
        tag = self.session.dataset_tag or "ROI"
        return str(labels_dir / f"{tag}_ROI.tiff")

    def _frame_shape(self):
        if self.session.stack is not None:
            return tuple(self.session.stack.shape[-2:])
        return self.session.frame_shape

    def _on_new_labels(self):
        shape = self._frame_shape()
        if shape is None:
            return self._warn("Load a dataset or a preprocessed stack first.")
        self.session.roi_labels = np.zeros(shape, dtype=np.uint16)
        self._add_labels_layer(self.session.roi_labels)
        self.roi_info.setText(f"Blank ROI labels {shape} — paint ROIs in napari, then save.")
        self._refresh_gating()

    def _on_load_labels(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "Load ROI labels",
                                                self._labels_dialog_start(),
                                                "TIFF (*.tiff *.tif)")
        if not chosen:
            return  # cancelled
        shape = self._frame_shape()
        try:
            labels = at.store.load_roi_labels(Path(chosen), expected_shape=shape)
        except (FileNotFoundError, ValueError) as exc:
            return self._warn(str(exc))
        self.roi_labels_path = Path(chosen)
        self.session.roi_labels = labels
        self._add_labels_layer(labels)
        self._report_labels()
        self._refresh_gating()

    def _on_save_labels(self):
        layer = self.viewer.layers[LAYER_LABELS] if LAYER_LABELS in self.viewer.layers else None
        if layer is None:
            return self._warn("No 'ROI labels' layer in napari. Create or load one first.")
        chosen, _ = QFileDialog.getSaveFileName(self, "Save ROI labels",
                                                self._labels_dialog_start(),
                                                "TIFF (*.tiff *.tif)")
        if not chosen:
            return  # cancelled
        labels = np.asarray(layer.data).astype(np.uint16)
        try:
            at.store.save_roi_labels(Path(chosen), labels)
        except OSError as exc:
            return self._warn(str(exc))
        # The saved labels are the ones the session already holds (it came off the napari
        # layer), so saving leaves it live -- no reload needed to extract traces.
        self.roi_labels_path = Path(chosen)
        self.session.roi_labels = labels
        self._report_labels()
        self._refresh_gating()

    def _report_labels(self):
        labels = self._current_labels()
        if labels is None:
            return self.roi_info.setText("No ROI labels loaded.")
        # Counted with the same rule extraction uses, not len(unique) - 1: that assumes a
        # background pixel exists and undercounts by one on a label image where every
        # pixel belongs to an ROI.
        n = len(at.traces.roi_ids_in(labels))
        where = f"<br><code>{self.roi_labels_path}</code>" if self.roi_labels_path else ""
        self.roi_info.setText(f"ROI labels {labels.shape} · <b>{n}</b> ROI(s).{where}")

    def _current_labels(self):
        """The live napari layer wins over the last loaded array -- it may have been painted."""
        if LAYER_LABELS in self.viewer.layers:
            return np.asarray(self.viewer.layers[LAYER_LABELS].data).astype(np.uint16)
        return self.session.roi_labels

    def _add_labels_layer(self, labels):
        if LAYER_LABELS in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[LAYER_LABELS])
        self.viewer.add_labels(labels.astype(np.uint16), name=LAYER_LABELS, opacity=0.6)

    # ------------------------------------------------------------------ tab 4
    def _build_traces_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.extract_box = QGroupBox("Extract dF/F from a preprocessed .zarr store")
        form = QFormLayout(self.extract_box)
        self.extract_zarr_edit = QLineEdit()
        self.extract_zarr_edit.setPlaceholderText("(none selected) — <dataset_tag>_<method>.zarr")
        self.extract_zarr_edit.textChanged.connect(self._refresh_gating)
        form.addRow("source .zarr", _path_row(self.extract_zarr_edit, self._browse_extract_zarr))
        form.addRow(_caption("Traces are read from this store on disk, not from whatever is "
                             "currently in the viewer. Filled in automatically when a store is "
                             "loaded or saved in tab 2; override it here to extract from a "
                             "different preprocessing run."))
        self.start_spin = _spin(0, 10_000_000, 0)
        form.addRow("START_TIMEPOINT", self.start_spin)
        self.extract_spin = _spin(1, 10_000_000, 500)
        form.addRow("EXTRACT_TIMEPOINTS", self.extract_spin)
        self.baseline_spin = _spin(1, 10_000_000, 20)
        form.addRow("BASELINE_SLIDES", self.baseline_spin)
        form.addRow(_caption("F0 is the mean of the first BASELINE_SLIDES frames, so dF/F is ~0 "
                             "there by construction. Keep the analysis 'pre' phase after that "
                             "window (tab 6)."))
        self.extract_btn = QPushButton("Extract traces")
        self.extract_btn.clicked.connect(self._on_extract)
        form.addRow(self.extract_btn)
        layout.addWidget(self.extract_box)

        export_box = QGroupBox("Export / reload (points 3 and 4)")
        export_layout = QVBoxLayout(export_box)
        self.analysis_dir_edit = QLineEdit()
        self.analysis_dir_edit.setPlaceholderText("(none selected) — e.g. ./analysis")
        self.analysis_dir_edit.textChanged.connect(self._refresh_gating)
        export_layout.addWidget(QLabel("Analysis root (outputs go to <root>/<dataset_tag>/)"))
        export_layout.addWidget(_path_row(self.analysis_dir_edit, self._browse_analysis_dir))
        buttons = QHBoxLayout()
        self.save_traces_btn = QPushButton("Save traces")
        self.save_traces_btn.clicked.connect(self._on_save_traces)
        self.load_traces_btn = QPushButton("Load saved traces")
        self.load_traces_btn.clicked.connect(self._on_load_traces)
        buttons.addWidget(self.save_traces_btn)
        buttons.addWidget(self.load_traces_btn)
        export_layout.addLayout(buttons)
        layout.addWidget(export_box)

        import_box = QGroupBox("Traces-only entry — import dF/F from anywhere (point 3)")
        import_form = QFormLayout(import_box)
        self.import_path_edit = QLineEdit()
        self.import_path_edit.textChanged.connect(self._refresh_gating)
        import_form.addRow("file", _path_row(self.import_path_edit, self._browse_import))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["auto", "roi_rows", "roi_cols"])
        import_form.addRow("orientation", self.orientation_combo)
        self.time_column_edit = QLineEdit("auto")
        import_form.addRow("time_column", self.time_column_edit)
        self.import_rate_spin = _spin(0.0, 100000.0, 0.0, 0.5, decimals=3)
        import_form.addRow("frame_rate_hz", self.import_rate_spin)
        import_form.addRow(_caption(
            "Required (non-zero) when the file carries no time axis — it cannot be inferred, "
            "and CASCADE resampling, the phase windows and the frequency bands all depend on "
            "it. Importing disables tabs 1–3: there is no pixel data behind these traces."))
        self.import_btn = QPushButton("Import dF/F (switches to traces-only mode)")
        self.import_btn.clicked.connect(self._on_import_dff)
        import_form.addRow(self.import_btn)
        layout.addWidget(import_box)

        self.traces_info = _info("No traces.")
        layout.addWidget(self.traces_info)
        layout.addStretch(1)
        return page

    def _browse_analysis_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Analysis root",
                                                self._dialog_start(self.analysis_dir_edit))
        if path:
            self.analysis_dir_edit.setText(path)

    def _browse_extract_zarr(self):
        start = self._dialog_start(self.extract_zarr_edit)
        if self._dir(self.extract_zarr_edit) is None:
            suggested = self._zarr_path()
            start = str(suggested) if suggested else self._dialog_start(self.preprocessed_dir_edit)
        path = QFileDialog.getExistingDirectory(self, "Preprocessed .zarr store to extract from",
                                                start)
        if path:
            self.extract_zarr_edit.setText(path)

    def _browse_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import dF/F", self._dialog_start(),
                                              "Traces (*.csv *.npz *.npy)")
        if path:
            self.import_path_edit.setText(path)

    def _analysis_dir(self):
        root = self._dir(self.analysis_dir_edit)
        if root is None:
            raise ValueError("Set the analysis root directory (tab 4) before saving or "
                             "loading analysis outputs.")
        return at.store.analysis_dir(root, self.session.dataset_tag or "imported_traces")

    def _suggested_output_dir(self):
        """Where an output dialog should open — <root>/<tag>, without creating it."""
        root = self._dir(self.analysis_dir_edit)
        if root is None:
            return str(REPO)
        return str(root / (self.session.dataset_tag or "imported_traces"))

    def _pick_output_dir(self, title):
        """Ask for a destination folder. Returns a Path, or None if cancelled.

        Outputs go where the user says at the moment of writing rather than into a
        derived <root>/<tag>, because that derived path is the same for every run of a
        dataset: a second inference or analysis pass would overwrite the first without
        warning. The dialog's 'new folder' button is the escape hatch.
        """
        chosen = QFileDialog.getExistingDirectory(self, title, self._suggested_output_dir())
        if not chosen:
            return None
        out_dir = Path(chosen)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _on_extract(self):
        """Extract per-ROI traces from the .zarr store named in the panel.

        Reads the store rather than `session.stack` on purpose. The session may hold the
        lazy delayed preprocessing chain, and extracting from that re-runs denoising and
        motion correction for every frame in the window -- work already paid for and
        cached. Naming the store also makes the provenance exact: the traces record which
        preprocessing run they came from, not just "whatever was loaded".
        """
        labels = self._current_labels()
        if labels is None:
            return self._warn("Create or load ROI labels first (tab 3).")
        store_path = self._dir(self.extract_zarr_edit)
        if store_path is None:
            return self._warn("Select the preprocessed .zarr store to extract from.")
        if not store_path.exists():
            return self._warn(f"No such store: {store_path}")
        if store_path.suffix != ".zarr" or "_" not in store_path.stem:
            return self._warn(f"{store_path.name} is not a <dataset_tag>_<method>.zarr store.")

        tag, method_tag = store_path.stem.rsplit("_", 1)
        start, count, baseline = (self.start_spin.value(), self.extract_spin.value(),
                                  self.baseline_spin.value())
        times = self.session.times_s
        labels_path = self.roi_labels_path
        fallback_hz = self.fallback_hz_spin.value()

        def work(progress):
            stack, _params = at.store.load_preprocessed(store_path.parent, tag, method_tag)
            if tuple(stack.shape[-2:]) != tuple(labels.shape):
                raise ValueError(
                    f"ROI labels are {labels.shape} but {store_path.name} holds "
                    f"{tuple(stack.shape[-2:])} frames -- they describe different "
                    "recordings, so every trace would be nonsense.")
            raw = at.traces.extract_roi_traces(stack, labels, start, count, progress=progress)
            print(f"> Extracted frames {start}..{start + len(next(iter(raw.values()))) - 1} "
                  f"from {store_path.name}, labels: {labels_path}")
            dff = at.traces.compute_dff(raw, baseline)
            roi_ids, matrix = at.traces.as_matrix(dff)
            if times is None:
                # Extracting from a store opened without its .imgdir: no ElapsedTimes.yaml
                # exists, so the time axis is nominal. Say so -- the frame rate feeds
                # CASCADE resampling and the frequency bands downstream.
                print(f"! No acquisition timebase for this store; using a nominal "
                      f"{fallback_hz} Hz axis. Load the .imgdir in tab 1 for real times.")
                t_plot = np.arange(matrix.shape[1], dtype=float) / fallback_hz
            else:
                t_plot = at.traces.window_times(times, start, matrix.shape[1])
            return raw, roi_ids, matrix, t_plot

        def done(result):
            raw, roi_ids, matrix, t_plot = result
            s = self.session
            s.raw_traces, s.roi_ids, s.dff_matrix, s.t_plot = raw, roi_ids, matrix, t_plot
            s.trace_params = {"mode": "full",
                              "dataset": s.data_dir.name if s.data_dir else s.dataset_tag,
                              "preprocessed_zarr": str(store_path),
                              "roi_labels": str(labels_path) if labels_path else None,
                              "START_TIMEPOINT": start, "EXTRACT_TIMEPOINTS": count,
                              "BASELINE_SLIDES": baseline}
            self._report_traces()
            self._refresh_gating()

        self._run(work, done, "Extracting ROI traces…", with_progress=True)

    def _on_save_traces(self):
        s = self.session
        out_dir = self._analysis_dir()
        raw = (np.vstack([s.raw_traces[r] for r in s.roi_ids])
               if s.raw_traces is not None else None)
        params = dict(s.trace_params)

        def work():
            return at.store.save_traces(out_dir, s.roi_ids, s.dff_matrix, s.t_plot,
                                        raw=raw, params=params)

        self._run(work, lambda _: self._refresh_gating(), "Saving traces…")

    def _on_load_traces(self):
        out_dir = self._analysis_dir()

        def work():
            return at.store.load_traces(out_dir)

        self._run(work, lambda result: self._adopt_traces(result, "full"), "Loading traces…")

    def _on_import_dff(self):
        path = Path(self.import_path_edit.text())
        if not path.exists():
            return self._warn(f"No such file: {path}")
        orientation = self.orientation_combo.currentText()
        column = self.time_column_edit.text().strip()
        time_column = None if column.lower() in ("", "none") else column
        rate = self.import_rate_spin.value() or None

        def work():
            return at.store.import_dff(path, orientation=orientation,
                                       time_column=time_column, frame_rate_hz=rate)

        self._run(work, lambda result: self._adopt_traces(result, "traces"),
                  "Importing dF/F…")

    def _adopt_traces(self, result, mode):
        s = self.session
        s.mode = mode
        s.roi_ids = [int(r) for r in result["roi_ids"]]
        s.dff_matrix = np.asarray(result["dff"], dtype=np.float32)
        s.t_plot = np.asarray(result["t"], dtype=float)
        s.raw_traces = None
        params = result.get("params", {})
        s.trace_params = dict(params)

        if mode == "traces":
            # No pixel data behind these traces: forget the image stages rather than
            # leaving them pointing at an unrelated recording.
            s.stack, s.raw_stack, s.roi_labels = None, None, None
            s.stack_source = None
            s.image_files, s.times_s, s.tag = [], None, ""
            self.roi_labels_path = None
            source = params.get("dataset") or Path(params.get("source", "imported")).stem
            s.data_dir = Path(source)
            self.viewer.layers.clear()
            self._log("> Traces-only mode: image, preprocessing and ROI stages disabled.")
        self._report_traces()
        self._sync_derived_paths()
        self._refresh_gating()

    def _report_traces(self):
        s = self.session
        if s.dff_matrix is None:
            self.traces_info.setText("No traces.")
            return
        self.traces_info.setText(
            f"<b>{len(s.roi_ids)}</b> ROI(s) × <b>{s.dff_matrix.shape[1]}</b> frames · "
            f"{s.t_plot[0]:.1f}–{s.t_plot[-1]:.1f} s · ~{s.frame_rate:.2f} Hz · "
            f"mode: <b>{s.mode}</b>")

    # ------------------------------------------------------------------ tab 5
    def _build_inference_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("CASCADE")
        form = QFormLayout(box)
        self.cascade_dir_edit = QLineEdit()
        self.cascade_dir_edit.setPlaceholderText("(none selected) — e.g. ./CascadeTorch")
        self.cascade_dir_edit.textChanged.connect(self._on_cascade_dir_changed)
        form.addRow("CASCADE_DIR", _path_row(self.cascade_dir_edit, self._browse_cascade_dir))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._describe_model)
        form.addRow("MODEL_NAME", self.model_combo)
        layout.addWidget(box)

        self.model_info = _info()
        layout.addWidget(self.model_info)
        layout.addWidget(_caption(
            "Model choice is a scientific claim, not a preference. The default is a spinal "
            "dorsal-horn model applied to DRG — the nearest available analogue, and an "
            "untested transfer (CLAUDE.md §4C, App. A §8). No DRG-specific ground truth "
            "exists; state this wherever the output is quoted."))

        buttons = QHBoxLayout()
        self.run_cascade_btn = QPushButton("Run CASCADE")
        self.run_cascade_btn.clicked.connect(self._on_run_cascade)
        self.save_spike_btn = QPushButton("Save inferred spikes…")
        self.save_spike_btn.clicked.connect(self._on_save_spike_rate)
        self.load_spike_btn = QPushButton("Load saved inference…")
        self.load_spike_btn.clicked.connect(self._on_load_spike_rate)
        buttons.addWidget(self.run_cascade_btn)
        buttons.addWidget(self.save_spike_btn)
        buttons.addWidget(self.load_spike_btn)
        layout.addLayout(buttons)
        layout.addWidget(_caption(
            "Saving asks for the destination folder each time rather than writing to the "
            "analysis root automatically, so a second run cannot overwrite the "
            "spike_rate.npy of an earlier one for the same dataset."))

        self.inference_info = _info("No inferred rate.")
        layout.addWidget(self.inference_info)
        layout.addStretch(1)
        self._reload_models()
        return page

    def _browse_cascade_dir(self):
        path = QFileDialog.getExistingDirectory(self, "CascadeTorch directory",
                                                self._dialog_start(self.cascade_dir_edit))
        if path:
            self.cascade_dir_edit.setText(path)

    def _model_dir(self):
        cascade_dir = self._dir(self.cascade_dir_edit)
        return None if cascade_dir is None else cascade_dir / "Pretrained_models"

    def _on_cascade_dir_changed(self):
        self._reload_models()
        self._refresh_gating()

    def _reload_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        model_dir = self._model_dir()
        models = []
        if model_dir is not None:
            try:
                models = at.cascade_runner.available_models(model_dir)
            except (FileNotFoundError, NotADirectoryError, OSError):
                models = []
        self.model_combo.addItems(models)
        default = "Spinal_cord_excitatory_30Hz_smoothing50ms"
        if default in models:
            self.model_combo.setCurrentText(default)
        self.model_combo.blockSignals(False)
        self._describe_model()

    def _describe_model(self):
        name = self.model_combo.currentText()
        rate = self.session.frame_rate
        if self._model_dir() is None:
            return self.model_info.setText("Set CASCADE_DIR to list the pretrained models.")
        if not name:
            return self.model_info.setText("No models found under Pretrained_models/.")
        if rate is None:
            return self.model_info.setText("Load traces to see the resampling plan.")
        stream = io.StringIO()
        import contextlib
        try:
            with contextlib.redirect_stdout(stream):
                at.cascade_runner.check_model(self._model_dir(), name, rate, verbose=True)
        except Exception as exc:  # a malformed config should not kill the panel
            return self.model_info.setText(f"Could not read model config: {exc}")
        self.model_info.setText("<pre style='white-space:pre-wrap'>"
                                + stream.getvalue().strip() + "</pre>")

    def _on_run_cascade(self):
        s = self.session
        name, cascade_dir, model_dir = (self.model_combo.currentText(),
                                        self._dir(self.cascade_dir_edit), self._model_dir())
        rate, matrix = s.frame_rate, s.dff_matrix
        extra = {"dataset": s.data_dir.name if s.data_dir else "imported",
                 "mode": s.mode,
                 "roi_ids": [int(r) for r in s.roi_ids],
                 "BASELINE_SLIDES": s.trace_params.get("BASELINE_SLIDES"),
                 "START_TIMEPOINT": s.trace_params.get("START_TIMEPOINT"),
                 "EXTRACT_TIMEPOINTS": s.trace_params.get("EXTRACT_TIMEPOINTS")}

        def work():
            return at.cascade_runner.run(matrix, rate, name, cascade_dir,
                                         model_dir=model_dir, announce_model=False)

        def done(result):
            self._cascade_result, self._cascade_extra = result, extra
            self._adopt_spike_rate((result.spike_rate, result.pad))
            self._prompt_save_spike_rate()

        self._run(work, done, "Running CASCADE…")

    def _prompt_save_spike_rate(self):
        """Offer to write a fresh inference to disk, in a folder chosen now.

        Asked rather than assumed: the destination is per-run, so writing it without
        asking would silently overwrite the previous run's spike_rate.npy for the same
        dataset. Nothing is lost by declining -- the rate stays in the session and the
        Save button stays live -- until the window closes.
        """
        answer = QMessageBox.question(
            self, "Save inferred spikes?",
            "CASCADE finished. The inferred rate is in memory but not on disk.\n\n"
            "Save spike_rate.npy + spike_inference_params.json to a folder now?")
        if answer == QMessageBox.StandardButton.Yes:
            self._on_save_spike_rate()
        else:
            self._log("> Not saved. Press 'Save inferred spikes…' when ready; the rate is "
                      "lost when this window closes.")

    def _on_save_spike_rate(self):
        if self._cascade_result is None:
            return self._warn("Run CASCADE first — a rate loaded from disk is already saved.")
        out_dir = self._pick_output_dir("Folder for the inferred spikes")
        if out_dir is None:
            return  # cancelled
        result, extra = self._cascade_result, self._cascade_extra
        roi_ids, t_plot = self.session.roi_ids, self.session.t_plot
        if (out_dir / "spike_rate.npy").exists():
            answer = QMessageBox.question(
                self, "Overwrite?",
                f"{out_dir.name} already holds a spike_rate.npy from an earlier run.\n\n"
                "Overwrite it?")
            if answer != QMessageBox.StandardButton.Yes:
                return

        def work():
            path = at.cascade_runner.save(result, out_dir, extra=extra)
            _save_spike_rate_csv(out_dir, result.spike_rate, roi_ids, t_plot)
            return path

        self._run(work, lambda path: self._log(f"> Inferred spikes -> {path}"),
                  "Saving inferred spikes…")

    def _on_load_spike_rate(self):
        out_dir = self._pick_output_dir("Folder holding spike_rate.npy")
        if out_dir is None:
            return  # cancelled

        def work():
            spike_rate, params = at.cascade_runner.load(out_dir)
            return spike_rate, int(params.get("pad_frames_per_end", 0))

        def done(result):
            # Loaded, not inferred: there is no CascadeResult to re-save, and the copy on
            # disk in `out_dir` is the authoritative one.
            self._cascade_result = self._cascade_extra = None
            self._adopt_spike_rate(result)

        self._run(work, done, "Loading inferred rate…")

    def _adopt_spike_rate(self, result):
        spike_rate, pad = result
        s = self.session
        if s.dff_matrix is not None and spike_rate.shape[0] != len(s.roi_ids):
            return self._warn(
                f"The saved rate has {spike_rate.shape[0]} ROIs but the current traces have "
                f"{len(s.roi_ids)}. They are from different runs — reload matching traces.")
        s.spike_rate, s.pad = spike_rate, pad
        self.inference_info.setText(
            f"Inferred rate <b>{spike_rate.shape[0]}</b> ROI(s) × "
            f"<b>{spike_rate.shape[1]}</b> frames · {pad} NaN pad frame(s) per end · "
            "units spikes/s")
        self._refresh_gating()

    # ------------------------------------------------------------------ tab 6
    def _build_analysis_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        phase_box = QGroupBox("Stimulus phases (s)")
        phase_layout = QVBoxLayout(phase_box)
        self.phase_table = QTableWidget(3, 3)
        self.phase_table.setHorizontalHeaderLabels(["name", "start", "end"])
        for row, (name, start, end) in enumerate(
                [("pre", 2.0, 10.0), ("stim", 10.0, 15.0), ("post", 15.0, 50.0)]):
            for col, value in enumerate((name, start, end)):
                self.phase_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.phase_table.setMaximumHeight(140)
        phase_layout.addWidget(self.phase_table)
        phase_layout.addWidget(_caption(
            "'pre' starts after the F0 window, not at 0: frames 0..BASELINE_SLIDES defined F0, "
            "so dF/F there is ~0 by construction and would fake a silent baseline."))
        layout.addWidget(phase_box)

        band_box = QGroupBox("Index B frequency bands (Hz)")
        band_layout = QVBoxLayout(band_box)
        self.band_table = QTableWidget(4, 2)
        self.band_table.setHorizontalHeaderLabels(["low", "high"])
        for row, (lo, hi) in enumerate([(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]):
            self.band_table.setItem(row, 0, QTableWidgetItem(str(lo)))
            self.band_table.setItem(row, 1, QTableWidgetItem(str(hi)))
        self.band_table.setMaximumHeight(160)
        band_layout.addWidget(self.band_table)
        self.band_caption = _caption("")
        band_layout.addWidget(self.band_caption)
        layout.addWidget(band_box)

        opts_box = QGroupBox("Thresholds, PCA and clustering")
        form = QFormLayout(opts_box)
        self.auto_thresh_check = QCheckBox("auto (baseline median + n·σ)")
        self.auto_thresh_check.setChecked(True)
        self.auto_thresh_check.stateChanged.connect(
            lambda: self.active_thresh_spin.setEnabled(not self.auto_thresh_check.isChecked()))
        form.addRow("ACTIVE_THRESH", self.auto_thresh_check)
        self.active_thresh_spin = _spin(0.0, 1000.0, 0.0, 0.01, decimals=4)
        self.active_thresh_spin.setEnabled(False)
        form.addRow("  manual value (spikes/s)", self.active_thresh_spin)
        self.n_sigma_spin = _spin(0.0, 20.0, 3.0, 0.1, decimals=2)
        form.addRow("ACTIVE_N_SIGMA", self.n_sigma_spin)
        self.var_target_spin = _spin(0.05, 1.0, 0.95, 0.01, decimals=2)
        form.addRow("PCA_VAR_TARGET", self.var_target_spin)
        self.max_k_spin = _spin(2, 50, 8)
        form.addRow("MAX_K", self.max_k_spin)
        self.min_group_spin = _spin(1, 100, 3)
        form.addRow("MIN_GROUP_SIZE", self.min_group_spin)
        self.max_frac_spin = _spin(0.1, 1.0, 0.9, 0.05, decimals=2)
        form.addRow("MAX_GROUP_FRAC", self.max_frac_spin)
        self.rate_cut_edit = QLineEdit()
        self.rate_cut_edit.setPlaceholderText("blank = auto (suggest_cut)")
        form.addRow("RATE_CUT_HEIGHT", self.rate_cut_edit)
        self.freq_cut_edit = QLineEdit()
        self.freq_cut_edit.setPlaceholderText("blank = auto (suggest_cut)")
        form.addRow("FREQ_CUT_HEIGHT", self.freq_cut_edit)
        layout.addWidget(opts_box)

        self.run_analysis_btn = QPushButton("Run analysis and export (CSV + TIFF)")
        self.run_analysis_btn.clicked.connect(self._on_run_analysis)
        layout.addWidget(self.run_analysis_btn)
        self.analysis_info = _info("Not run.")
        layout.addWidget(self.analysis_info)
        layout.addWidget(_caption(
            "Running asks for an output folder first; figures are written as .tiff into "
            "<that folder>/figures/ rather than drawn here. Group labels are an arbitrary "
            "integer labelling, not a cell type."))
        layout.addStretch(1)
        return page

    def _phases(self):
        phases = {}
        for row in range(self.phase_table.rowCount()):
            name = self.phase_table.item(row, 0)
            start = self.phase_table.item(row, 1)
            end = self.phase_table.item(row, 2)
            if not (name and name.text().strip()):
                continue
            phases[name.text().strip()] = (float(start.text()), float(end.text()))
        if not phases:
            raise ValueError("Define at least one phase.")
        return phases

    def _bands(self):
        bands = []
        for row in range(self.band_table.rowCount()):
            lo, hi = self.band_table.item(row, 0), self.band_table.item(row, 1)
            if lo and hi and lo.text().strip() and hi.text().strip():
                bands.append((float(lo.text()), float(hi.text())))
        if not bands:
            raise ValueError("Define at least one frequency band.")
        return bands

    def _on_run_analysis(self):
        s = self.session
        try:
            phases, bands = self._phases(), self._bands()
        except ValueError as exc:
            return self._warn(str(exc))

        # The top edge is *meant* to sit at Nyquist, so compare with tolerance: a
        # measured 9.999999 Hz puts Nyquist at 4.9999995 and would otherwise reject the
        # documented default band of 5.0 Hz on floating-point margin alone.
        nyquist = s.frame_rate / 2.0
        top = max(hi for _, hi in bands)
        if top > nyquist and not np.isclose(top, nyquist, rtol=1e-6):
            return self._warn(f"The top band edge ({top:g} Hz) exceeds Nyquist "
                              f"({nyquist:.3f} Hz) for a {s.frame_rate:.3f} Hz recording. "
                              "Lower it — there is no information above Nyquist to recover.")

        # Index B reads the FFT helper out of CascadeTorch/scripts, so a missing
        # CASCADE_DIR fails halfway through -- after the Index A CSVs are already on
        # disk. Check it before anything is written.
        if self._dir(self.cascade_dir_edit) is None:
            return self._warn("Set CASCADE_DIR (tab 5) before running the analysis: the "
                              "frequency features (Index B) load their FFT helper from "
                              "CascadeTorch/scripts.")

        # Asked here, not derived from the analysis root: every artifact below
        # (features, PCA, groups, figures) is written under this folder, and re-running
        # with different phases or cut heights into the derived <root>/<tag> would
        # overwrite the previous pass for the same dataset with no trace of it.
        out_dir = self._pick_output_dir("Folder for this analysis run's outputs")
        if out_dir is None:
            return  # cancelled
        self._log(f"> Analysis outputs -> {out_dir}")

        cut_rate = self.rate_cut_edit.text().strip()
        cut_freq = self.freq_cut_edit.text().strip()
        cfg = dict(
            phases=phases, bands=bands, out_dir=out_dir,
            cascade_dir=self._dir(self.cascade_dir_edit),
            var_target=self.var_target_spin.value(), max_k=self.max_k_spin.value(),
            min_group=self.min_group_spin.value(), max_frac=self.max_frac_spin.value(),
            n_sigma=self.n_sigma_spin.value(),
            active_thresh=None if self.auto_thresh_check.isChecked()
            else self.active_thresh_spin.value(),
            cut_rate=float(cut_rate) if cut_rate else None,
            cut_freq=float(cut_freq) if cut_freq else None,
            model_name=self.model_combo.currentText(),
        )
        spike_rate, matrix, roi_ids, t_plot, pad, rate = (
            s.spike_rate, s.dff_matrix, s.roi_ids, s.t_plot, s.pad, s.frame_rate)

        def work():
            return _compute_analysis(spike_rate, matrix, roi_ids, t_plot, pad, rate, cfg)

        self._run(work, self._finish_analysis, "Running features, PCA and clustering…")

    def _finish_analysis(self, bundle):
        """Draw and export the figures here, on the UI thread.

        pyplot is not thread-safe: building figures inside the worker deadlocks against
        the Qt event loop (0% CPU, never returns). The heavy part -- features, PCA,
        linkage, the CSV/npz writes -- already ran off-thread; only plotting is left, and
        it takes a couple of seconds.
        """
        self.status.setText("Exporting figures…")
        QApplication.processEvents()
        paths = _export_figures(bundle, bundle["cfg"])
        for path in paths:
            self._log(f"> Saved figure -> {path}")
        self.analysis_info.setText(bundle["summary"])
        self.status.setText("Done.")
        self._refresh_gating()

    # ------------------------------------------------------------- infrastructure
    def _add_layer(self, name, data):
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])
        self.viewer.add_image(data, name=name)

    @staticmethod
    def _dir(line_edit):
        """A directory field's value, or None when blank.

        Every path field starts empty, so callers must never turn one into Path("") --
        that silently becomes the current working directory and reads or writes the
        wrong place.
        """
        text = line_edit.text().strip()
        return Path(text).expanduser() if text else None

    def _dialog_start(self, line_edit=None):
        """Where a file dialog should open: the field's value if set, else the repo."""
        current = self._dir(line_edit) if line_edit is not None else None
        return str(current) if current else str(REPO)

    def _sync_derived_paths(self):
        path = self._zarr_path()
        preprocessed_dir = self._dir(self.preprocessed_dir_edit)
        if path is None:
            self.zarr_info.setText("— set PREPROCESSED_DIR and load a dataset")
        else:
            exists = "on disk" if path.exists() else "not yet saved"
            cached = sorted(p.stem.split("_", 1)[-1]
                            for p in preprocessed_dir.glob(
                                f"{self.session.dataset_tag}_*.zarr")) if self.session.data_dir else []
            self.zarr_info.setText(f"<code>{path}</code><br><i>{exists}</i>"
                                   + (f" · cached chains: {', '.join(cached)}" if cached else ""))
        if self.session.frame_rate:
            nyquist = self.session.frame_rate / 2.0
            self.band_caption.setText(
                f"Upper edge is capped by Nyquist = {nyquist:.2f} Hz at "
                f"{self.session.frame_rate:.2f} Hz acquisition.")

    def _refresh_gating(self):
        s = self.session
        full = s.mode == "full"
        idle = not self._busy
        has_stack = s.stack is not None
        has_labels = LAYER_LABELS in self.viewer.layers or s.roi_labels is not None
        has_traces = s.dff_matrix is not None
        has_rate = s.spike_rate is not None

        # Tab availability tracks pipeline state only, never busy-ness. Disabling the
        # current tab makes Qt jump focus to the first enabled one, which dragged the
        # user to "4 · Traces" after every job; running jobs are blocked by disabling
        # the action buttons instead.
        current = self.tabs.currentIndex()
        for index, enabled in enumerate([full, full, full, True, has_traces, has_rate]):
            self.tabs.setTabEnabled(index, enabled)
        if self.tabs.isTabEnabled(current):
            self.tabs.setCurrentIndex(current)

        self.load_dataset_btn.setEnabled(full and idle and self._dir(self.data_dir_edit) is not None)
        # Build needs somewhere to write to and frames to read; Save needs a stack to have
        # been built (or loaded) first -- offering Save before that just queues a failure.
        has_preprocessed_dir = self._dir(self.preprocessed_dir_edit) is not None
        self.build_btn.setEnabled(full and idle and has_preprocessed_dir and bool(s.image_files))
        self.save_zarr_btn.setEnabled(full and idle and has_preprocessed_dir
                                      and s.stack is not None)
        # A .zarr store is a complete entry point, so loading one is available immediately.
        self.load_zarr_btn.setEnabled(full and idle)
        for widget in (self.new_labels_btn, self.load_labels_btn):
            widget.setEnabled(full and idle and (has_stack or s.frame_shape is not None))
        self.save_labels_btn.setEnabled(full and idle and has_labels)
        self.extract_box.setEnabled(
            full and idle and has_labels
            and self._dir(self.extract_zarr_edit) is not None)
        has_analysis_root = self._dir(self.analysis_dir_edit) is not None
        self.save_traces_btn.setEnabled(idle and has_traces and has_analysis_root)
        self.load_traces_btn.setEnabled(idle and bool(s.dataset_tag) and has_analysis_root)
        self.import_btn.setEnabled(idle and bool(self.import_path_edit.text().strip()))
        # Inference and analysis no longer need the analysis root: each write asks for
        # its own destination folder, so the root is only a starting point for those
        # dialogs. Saving/loading traces still uses the derived <root>/<tag>.
        self.run_cascade_btn.setEnabled(idle and has_traces
                                        and bool(self.model_combo.currentText()))
        self.save_spike_btn.setEnabled(idle and self._cascade_result is not None)
        self.load_spike_btn.setEnabled(idle)
        self.run_analysis_btn.setEnabled(idle and has_rate)
        self._sync_derived_paths()

    def _run(self, fn, on_done=None, status="Working…", with_progress=False):
        if self._busy:
            return self._warn("A job is already running; wait for it to finish.")
        self._busy = True
        self.status.setText(status)
        self.progress.setVisible(with_progress)
        self.progress.setValue(0)
        self._refresh_gating()
        self._log(f"--- {status}")

        job = Job((lambda: fn(self._emit_progress)) if with_progress else fn)
        job.signals.message.connect(self._log)
        job.signals.progress.connect(self._on_progress)
        job.signals.failed.connect(self._on_failed)
        job.signals.finished.connect(lambda result: self._on_finished(result, on_done))
        self._current_signals = job.signals
        self.pool.start(job)

    def _emit_progress(self, done, total):
        self._current_signals.progress.emit(done, total)

    def _on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, result, on_done):
        self._busy = False
        self.progress.setVisible(False)
        self.status.setText("Done.")
        if on_done is not None:
            try:
                on_done(result)
            except Exception:
                self._on_failed(traceback.format_exc())
                return
        self._refresh_gating()

    def _on_failed(self, message):
        self._busy = False
        self.progress.setVisible(False)
        self.status.setText("Failed — see log.")
        self._log(message)
        self._refresh_gating()
        QMessageBox.critical(self, "Job failed", message.strip().splitlines()[-1])

    def _log(self, text):
        self.log.appendPlainText(text)

    def _warn(self, message):
        self._log(f"! {message}")
        QMessageBox.warning(self, "Cannot continue", message)


def _save_spike_rate_csv(out_dir, spike_rate, roi_ids, t):
    """`spike_rate.csv` — the same numbers as spike_rate.npy, in a portable layout.

    Column layout follows `traces.csv` (`time_s` then one column per ROI) so the inferred
    rate lines up row-for-row with the dF/F it came from and both open in the same
    spreadsheet. That means the CSV is the transpose of the .npy, which is stored
    (n_rois, n_frames).

    The pad frames CASCADE leaves at each end are NaN; they are written as empty cells,
    which is what `pd.read_csv` reads back as NaN. Do not fill them -- the model produced
    no estimate there.
    """
    spike_rate = np.asarray(spike_rate)
    columns = ([f"ROI{r}" for r in roi_ids]
               if roi_ids is not None and len(roi_ids) == spike_rate.shape[0]
               else [f"ROI{i + 1}" for i in range(spike_rate.shape[0])])
    frame = pd.DataFrame(spike_rate.T, columns=columns)
    # The time axis comes from the traces, so a mismatch means the two are not from the
    # same run (CASCADE resampling can trim a frame). Fall back to frame index rather
    # than write a time column that silently misaligns the rate with the recording.
    if t is not None and len(t) == spike_rate.shape[1]:
        frame.insert(0, "time_s", np.asarray(t, dtype=float))
    else:
        frame.insert(0, "frame", np.arange(spike_rate.shape[1]))
        print(f"! Time axis has {0 if t is None else len(t)} points but the rate has "
              f"{spike_rate.shape[1]} frames; wrote a frame index instead of time_s.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spike_rate.csv"
    frame.to_csv(path, index=False)
    print(f"> Saved spike_rate.csv ({spike_rate.shape[0]} ROIs x {spike_rate.shape[1]} "
          f"frames, spikes/s) -> {path}")
    return path


# =============================================================================
# The analysis stage
#
# Split in two on purpose. Everything numeric runs in the worker; the figures are
# built afterwards on the UI thread, because pyplot is not thread-safe -- creating
# figures inside the worker while the Qt event loop runs deadlocks at 0% CPU and
# never returns.
# =============================================================================

def _compute_analysis(spike_rate, dff_matrix, roi_ids, t_plot, pad, frame_rate, cfg):
    """Features -> PCA -> Ward cut -> summaries -> CSV/npz, for both indices.

    No plotting: returns everything `_export_figures` needs to draw afterwards.
    """
    out_dir, phases, bands = cfg["out_dir"], cfg["phases"], cfg["bands"]
    phase_order = list(phases)
    masks = at.features.phase_masks(t_plot, phases)
    at.features.describe_phases(t_plot, phases, masks, frame_rate)

    active_thresh = cfg["active_thresh"]
    if active_thresh is None:
        active_thresh, _, _ = at.features.auto_active_thresh(
            spike_rate, masks[phase_order[0]], cfg["n_sigma"])

    rate_features = at.features.rate_features(spike_rate, roi_ids, masks, active_thresh)
    dff_summary = at.features.dff_summary(dff_matrix, roi_ids, masks)
    at.features.report_threshold(rate_features, masks, active_thresh)

    # --- Index A (Inferred Spike Rate) -------------------------------------------------------------
    pca_rate, scores_rate, npc_rate, z_rate = at.grouping.pca_and_linkage(
        rate_features, cfg["var_target"])
    print(at.grouping.cut_candidates(z_rate, cfg["max_k"]).to_string())
    cut_rate = cfg["cut_rate"] if cfg["cut_rate"] is not None else at.grouping.suggest_cut(
        z_rate, cfg["max_k"], cfg["min_group"], cfg["max_frac"])
    rate_groups = at.grouping.cut_tree(z_rate, cut_rate, roi_ids, cfg["max_frac"],
                                       label="inferred-rate")
    print(at.features.group_summary(rate_features, dff_summary, rate_groups,
                                    phase_order).to_string())
    print("\n(dff_peak_* is descriptive only -- dF/F amplitude is not a rate proxy, "
          "CLAUDE.md App.A 11)")
    at.grouping.print_membership(rate_groups, roi_ids, prefix="G")
    at.store.save_features(out_dir, "rate_features",
                           rate_features.join(dff_summary).assign(rate_group=rate_groups))
    at.store.save_pca(out_dir, "rate", pca_rate, scores_rate, npc_rate, z_rate,
                      list(rate_features.columns), cut_height=cut_rate,
                      groups=rate_groups, roi_ids=roi_ids)

    # --- Index B (Frequency Domain) -------------------------------------------------------------
    at.features.describe_bands(masks, frame_rate, bands)
    freq_features = at.features.freq_features(spike_rate, roi_ids, masks, frame_rate, bands,
                                              cfg["cascade_dir"])
    pca_freq, scores_freq, npc_freq, z_freq = at.grouping.pca_and_linkage(
        freq_features, cfg["var_target"])
    print(at.grouping.cut_candidates(z_freq, cfg["max_k"]).to_string())
    cut_freq = cfg["cut_freq"] if cfg["cut_freq"] is not None else at.grouping.suggest_cut(
        z_freq, cfg["max_k"], cfg["min_group"], cfg["max_frac"])
    freq_groups = at.grouping.cut_tree(z_freq, cut_freq, roi_ids, cfg["max_frac"],
                                       label="frequency", prefix="F")
    print(at.features.freq_group_summary(freq_features, freq_groups, phase_order,
                                         bands).to_string())
    at.grouping.print_membership(freq_groups, roi_ids, prefix="F")
    _table, ari = at.grouping.compare_partitions(rate_groups, freq_groups)
    at.store.save_features(out_dir, "freq_features", freq_features)
    at.store.save_pca(out_dir, "freq", pca_freq, scores_freq, npc_freq, z_freq,
                      list(freq_features.columns), cut_height=cut_freq,
                      groups=freq_groups, roi_ids=roi_ids)
    at.store.save_groups(out_dir, pd.DataFrame({"roi": roi_ids, "rate_group": rate_groups,
                                                "freq_group": freq_groups}))

    summary = (f"Index A: {int(rate_groups.max())} group(s) at h={cut_rate:.2f} · "
               f"Index B: {int(freq_groups.max())} group(s) at h={cut_freq:.2f} · "
               f"ARI={ari:.3f}<br>ACTIVE_THRESH={active_thresh:.3f} spikes/s<br>"
               f"Exported to <code>{out_dir}</code> (CSV + figures/*.tiff)")

    return {
        "cfg": cfg, "summary": summary, "spike_rate": spike_rate, "t_plot": t_plot,
        "roi_ids": roi_ids, "pad": pad, "active_thresh": active_thresh,
        "stim": phases.get("stim", tuple(list(phases.values())[0])),
        "rate_features": rate_features, "freq_features": freq_features,
        "pca_rate": pca_rate, "scores_rate": scores_rate, "z_rate": z_rate,
        "cut_rate": cut_rate, "rate_groups": rate_groups,
        "pca_freq": pca_freq, "scores_freq": scores_freq, "z_freq": z_freq,
        "cut_freq": cut_freq, "freq_groups": freq_groups,
    }


def _export_figures(bundle, cfg):
    """Draw and save the six figures. UI thread only -- see the note above."""
    import matplotlib.pyplot as plt

    out_dir = cfg["out_dir"]
    b = bundle
    label = (f"CASCADE | {cfg['model_name']} | "
             f"ACTIVE_THRESH={b['active_thresh']:.3f} spikes/s")
    paths = []

    def save(fig, name):
        paths.append(at.store.save_figure(fig, out_dir, name))
        plt.close(fig)  # the analysis can be re-run many times in one session

    save(at.plots.pca_summary(b["pca_rate"], b["scores_rate"], b["rate_features"],
                              b["roi_ids"], "PCA-A: per-phase inferred-rate features",
                              cfg["var_target"]), "pca_rate_summary")
    save(at.plots.dendrogram_plot(b["z_rate"], b["roi_ids"], "Index A (inferred rate)",
                                  b["cut_rate"]), "dendrogram_rate")
    save(at.plots.rate_heatmap(b["spike_rate"], b["t_plot"], b["roi_ids"], b["rate_groups"],
                               b["stim"], f"Inferred spike rate by Index A\n{label}"),
         "rate_heatmap")
    save(at.plots.group_means(b["spike_rate"], b["t_plot"], b["rate_groups"], b["roi_ids"],
                              b["stim"], b["pad"],
                              f"Group-mean inferred rate ± SEM\n{label}"),
         "rate_group_means")
    save(at.plots.pca_summary(b["pca_freq"], b["scores_freq"], b["freq_features"],
                              b["roi_ids"], "PCA-B: per-phase frequency features",
                              cfg["var_target"]), "pca_freq_summary")
    save(at.plots.dendrogram_plot(b["z_freq"], b["roi_ids"], "Index B (frequency)",
                                  b["cut_freq"]), "dendrogram_freq")
    return paths


# =============================================================================

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = napari.Viewer(title="Spike Inference — images and ROIs")
    window = PipelineWindow(viewer)
    window.show()
    napari.run()
    return app


if __name__ == "__main__":
    main()

