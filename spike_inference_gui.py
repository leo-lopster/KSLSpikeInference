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

TEMPORARILY DISABLED: Index B (the frequency-domain PCA -- band powers and dominant
frequency of the inferred rate). Tab 6 runs Index A only. Every disabled block is
commented out behind the marker `[Index B disabled]`, so `grep -n "\\[Index B disabled\\]"`
lists everything that has to be un-commented to bring it back. The library functions it
used (`features.freq_features`, `features.describe_bands`, `features.freq_group_summary`,
`grouping.compare_partitions`) are untouched and still work -- only the call sites here
are switched off.
"""

from __future__ import annotations

import io
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")  # no pyplot GUI backend; the live canvases are built explicitly
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import napari  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from qtpy.QtCore import (  # noqa: E402
    QEvent, QObject, QPoint, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot,
)
from qtpy.QtGui import QColor, QFontDatabase, QFontMetrics  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from superqt import QLabeledRangeSlider  # noqa: E402

import analysis_tools as at  # noqa: E402

REPO = Path(__file__).resolve().parent
LAYER_RAW = "raw_stack"
LAYER_PREPROCESSED = "preprocessed"
LAYER_LABELS = "ROI labels"

# Width floor for the tab-5 model dropdowns, in characters. The longest name in
# Pretrained_models/available_models_CascadeTorch.yaml is 57 characters; below ~48 the
# longest entries start to elide at the current font.
MODEL_COMBO_CHARS = 58


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


class _Canvas(FigureCanvasQTAgg):
    """A matplotlib canvas sized in inches, for the live preview.

    Height is pinned rather than left to the layout because these live inside a scroll
    area, which gives its child whatever height it asks for -- an unpinned canvas
    collapses to nothing. `show_figure` swaps in a new figure and keeps the widget height
    in step with it, so a group-traces figure that grows with k stays fully drawn instead
    of being squeezed.

    The background handling is not cosmetic. FigureCanvasQTAgg sets WA_OpaquePaintEvent,
    promising Qt that it paints every pixel of the widget, and then paints its Agg buffer
    from the top-left corner. The promise breaks the moment a figure smaller than the
    widget is swapped in -- fewer groups means a shorter group-traces figure -- because
    nothing repaints the margin around it and the PREVIOUS, larger figure stays on
    screen behind the new one. Dropping that attribute and filling the background
    instead makes Qt clear the whole widget before every paint.
    """

    DPI = 100

    def __init__(self, height_inches):
        super().__init__(Figure(figsize=(11, height_inches), dpi=self.DPI))
        self.setMinimumHeight(int(height_inches * self.DPI))
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("white"))  # matches figure facecolor
        self.setPalette(palette)

    def show_figure(self, fig):
        fig.set_dpi(self.DPI)
        # Re-run the layout at draw time: the figure is about to be shown at the widget's
        # width, which is not the figsize the plot function chose for export.
        fig.set_layout_engine("tight")
        self.figure = fig
        fig.set_canvas(self)
        # setFixedHeight, not setMinimumHeight: a minimum only lets the widget shrink if
        # the layout chooses to, so going from k=10 to k=3 left a tall widget holding a
        # short figure -- the empty band below it is where the old figure lingered.
        self.setFixedHeight(max(1, int(round(fig.get_figheight() * self.DPI))))
        self.draw_idle()

    def clear(self):
        self.figure.clear()
        self.draw_idle()


def _combo(min_chars=MODEL_COMBO_CHARS):
    """A dropdown that will not shrink below `min_chars` characters of text.

    Two settings, because they do different things: the size-adjust policy fixes the
    PREFERRED width (what QFormLayout grants when there is room), while the minimum
    width is the actual floor -- without it Qt still squeezes the box back to ~103 px
    and elides the names as soon as the window is narrowed.
    """
    box = QComboBox()
    box.setMinimumContentsLength(min_chars)
    box.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumWidth(box.fontMetrics().horizontalAdvance("0" * min_chars) + 40)
    return box


def _form(parent):
    """A QFormLayout whose fields fill the column instead of sitting at their sizeHint.

    macOS is the only platform whose style defaults QFormLayout to
    `FieldsStayAtSizeHint`, which hands each field exactly the width it asks for. For a
    word-wrapped QLabel that width is a narrow heuristic -- the Zarr-cache status was
    given 318 px for a message needing two lines, and rendered only the first, cutting it
    mid-sentence. It also pinned the path fields to their minimum so they never grew with
    the window. `AllNonFixedFieldsGrow` is what every other platform already does.
    """
    form = QFormLayout(parent)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def _path_edit(placeholder):
    """A path field wide enough to show its own placeholder.

    A QLineEdit's default minimum width is a fixed character count with no relation to
    what it holds, so these fields opened narrow enough to elide the example path they
    exist to demonstrate. Measuring the placeholder and making that the floor keeps the
    hint readable at any window width; the field still grows with the window, since
    `_path_row` gives it the stretch.
    """
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    # + room for the frame and the text margins Qt puts either side of the content.
    edit.setMinimumWidth(edit.fontMetrics().horizontalAdvance(placeholder) + 24)
    return edit


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


class _HelpIcon(QLabel):
    """An 'i' badge that floats its explanation beside itself while hovered.

    The explanatory notes used to sit permanently under the control they describe, which
    made every panel mostly prose. They are still worth keeping verbatim -- several record
    why a default is what it is -- so they move here rather than being cut.

    A plain `setToolTip` would be less code, but Qt decides the wrap width itself and
    hides the tip on a timer; these notes are whole paragraphs and are read while
    comparing them against the control. This popup is a fixed width, wraps, and stays up
    for exactly as long as the pointer is over the icon.
    """

    POPUP_WIDTH = 460

    def __init__(self, text, parent=None):
        super().__init__("ⓘ", parent)
        self._text = " ".join(text.split())
        self._popup = None
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        # palette(text), not palette(mid): the badge is the only thing advertising that an
        # explanation exists, so it takes the palette's full-contrast foreground -- black
        # on a light theme, white on a dark one -- rather than the muted grey the captions
        # used when they were always on screen.
        self.setStyleSheet("color: palette(text); font-size: 13px; font-weight: bold;")
        # The text is still SET so assistive tech and anything querying toolTip() can read
        # it, but `event` below swallows the render -- see there.
        self.setToolTip(self._text)
        self.setAccessibleDescription(self._text)

    def event(self, event):
        # Qt would raise its own tooltip on the usual delay, on top of the popup already
        # showing the same words twice over. Consume the request: the popup IS this
        # widget's tooltip, it just renders it wider and holds it for as long as hovered.
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().event(event)

    def enterEvent(self, event):
        self._show_popup()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._popup is not None:
            self._popup.hide()
        super().leaveEvent(event)

    def _show_popup(self):
        if self._popup is None:
            popup = QLabel(self._text, self, Qt.WindowType.ToolTip)
            popup.setWordWrap(True)
            popup.setMargin(8)
            popup.setFixedWidth(self.POPUP_WIDTH)
            # Transparent to the mouse: the popup opens directly under the pointer's
            # path, and if it took hover events it would trigger this icon's leaveEvent
            # and flicker itself out of existence.
            popup.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            popup.setStyleSheet(
                "background: palette(base); color: palette(text);"
                "border: 1px solid palette(mid); font-size: 11px;")
            self._popup = popup
        self._popup.adjustSize()
        self._popup.move(self._popup_position())
        self._popup.show()

    def _popup_position(self):
        """Below the icon, pulled back inside the screen when it would overhang."""
        size = self._popup.size()
        point = self.mapToGlobal(QPoint(0, self.height() + 4))
        x, y = point.x(), point.y()
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            x = max(area.left() + 8, min(x, area.right() - size.width() - 8))
            if y + size.height() > area.bottom():
                y = self.mapToGlobal(QPoint(0, 0)).y() - size.height() - 4
        return QPoint(x, y)


def _help_label(name, help_text):
    """A form-row label carrying an information icon: `name` + hover explanation."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(QLabel(name))
    layout.addWidget(_HelpIcon(help_text))
    layout.addStretch(1)
    return row


class _ModelPickerDialog(QDialog):
    """Pick one pretrained model out of the download index.

    Chooses only -- the caller runs the download, so the job keeps the main window's
    progress bar, log and error handling. Replaces a 156-entry combo box: the names alone
    do not say which model suits a recording, and the property that decides it (the
    training rate) is buried mid-string, so the table breaks it out into a sortable column.

    Everything shown is parsed from the index; nothing is fetched. Verified against all
    156 entries: each name carries a `<rate>Hz` and a `smoothing<N>ms` token.
    """

    COLUMNS = ["Model", "Installed", "Family", "Rate (Hz)", "Smoothing (ms)", "Noise"]

    def __init__(self, model_dir, index, installed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pretrained models")
        self._names = sorted(index)
        self._installed = set(installed)

        layout = QVBoxLayout(self)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter by name or family — e.g. spinal, 30Hz, GC8s")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.hide_installed_check = QCheckBox("hide installed")
        self.hide_installed_check.stateChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Filter"))
        filter_row.addWidget(self.filter_edit, 1)
        filter_row.addWidget(self.hide_installed_check)
        layout.addLayout(filter_row)

        self.table = QTableWidget(len(self._names), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self._fill_table()
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.SortOrder.AscendingOrder)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._accept_if_selected())
        layout.addWidget(self.table, 1)
        self.resize(self._preferred_width(), 580)

        self.status = _info()
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.download_button = QPushButton("Download")
        self.download_button.setDefault(True)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._accept_if_selected)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(self.download_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        # Models on disk but absent from the index -- hand-placed or retrained folders --
        # are usable but cannot be re-downloaded, so they are named rather than left to
        # make the two counts silently disagree.
        self._unlisted = sorted(self._installed - set(index))
        self._apply_filter()

    def _preferred_width(self):
        """Wide enough for the longest model name, capped to the screen.

        Sized from the content rather than a round number: the Model column stretches
        into whatever the fixed columns leave behind, and at a hardcoded 880 px that came
        to 393 px against the 424 px the longest name needs — so the longest names, the
        ones hardest to tell apart, were the ones elided.
        """
        # Each column is as wide as the wider of its contents and its header: the values
        # under "Smoothing (ms)" are all narrower than that title, and counting only the
        # contents under-measured the fixed columns by ~190 px, which came straight out of
        # the stretching Model column.
        header = self.table.horizontalHeader()
        def column_width(col):
            return max(self.table.sizeHintForColumn(col), header.sectionSizeHint(col))

        width = sum(column_width(c) for c in range(len(self.COLUMNS)))
        width += 72  # scrollbar, frame, cell padding
        screen = self.screen()
        if screen is not None:
            width = min(width, int(screen.availableGeometry().width() * 0.9))
        return max(720, width)

    # --- table -----------------------------------------------------------------------
    def _fill_table(self):
        for row, name in enumerate(self._names):
            here = name in self._installed
            first = QTableWidgetItem(name)
            # The name travels in UserRole, never read back out of the cell text, so an
            # elided or decorated cell can never become a filesystem path.
            first.setData(Qt.ItemDataRole.UserRole, name)
            self.table.setItem(row, 0, first)
            self.table.setItem(row, 1, QTableWidgetItem("✓" if here else ""))
            self.table.setItem(row, 2, QTableWidgetItem(re.split(r"[_-]", name)[0]))
            self.table.setItem(row, 3, _numeric_item(_model_rate(name)))
            self.table.setItem(row, 4, _numeric_item(_model_smoothing_ms(name)))
            self.table.setItem(row, 5,
                               QTableWidgetItem("high" if name.endswith("_high_noise")
                                                else "standard"))

    def _apply_filter(self):
        needle = self.filter_edit.text().strip().lower()
        hide_installed = self.hide_installed_check.isChecked()
        shown = 0
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            hidden = (needle and needle not in name.lower()) or \
                     (hide_installed and name in self._installed)
            self.table.setRowHidden(row, bool(hidden))
            shown += not hidden
        note = (f"<br>{len(self._unlisted)} local model(s) not in the index: "
                f"{', '.join(self._unlisted)}" if self._unlisted else "")
        self.status.setText(
            f"<b>{len(self._installed & set(self._names))}</b> of <b>{len(self._names)}</b> "
            f"listed model(s) installed · <b>{shown}</b> shown{note}")

    # --- selection -------------------------------------------------------------------
    def _on_selection_changed(self):
        self.download_button.setEnabled(self.selected_model() is not None)

    def selected_model(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def _accept_if_selected(self):
        if self.selected_model() is not None:
            self.accept()


def _model_rate(name):
    """Training rate in Hz parsed out of a model name, or 0.0 when it carries none."""
    try:
        return at.cascade_runner.model_rate_from_name(name)
    except ValueError:
        return 0.0


def _model_smoothing_ms(name):
    match = re.search(r"smoothing_?(\d+)ms", name)
    return float(match.group(1)) if match else 0.0


def _numeric_item(value):
    """A cell that sorts by magnitude, not alphabetically ('10' before '7.5')."""
    item = QTableWidgetItem()
    item.setData(Qt.ItemDataRole.EditRole, float(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


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
        # The last previewed feature/PCA/tree bundle. Held so the k sliders can re-cut
        # without recomputing, and dropped the moment any setting behind it changes.
        self._preview = None
        self._analysis_parent_dir = None  # asked once, reused for every run subfolder
        self._region_touched = False      # until dragged, the region is the whole recording
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._redraw_preview)

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
        # Ask Qt for the platform's fixed-width font (Menlo here) rather than naming a
        # "monospace" family in a stylesheet. No system ships a family by that literal
        # name, so Qt falls back to scanning every installed font to build its alias
        # table -- a ~60 ms startup cost that prints
        # 'Populating font family aliases took N ms ... missing font family "Monospace"'.
        log_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        log_font.setPointSize(10)
        self.log.setFont(log_font)

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

        box = QGroupBox("Dataset (.npy frames from an .imgdir)")
        form = _form(box)
        self.data_dir_edit = _path_edit("e.g. ./datasets/<name>.imgdir")
        self.data_dir_edit.textChanged.connect(self._refresh_gating)
        form.addRow("Raw Data Directory", _path_row(self.data_dir_edit, self._browse_data_dir))
        self.channel_spin = _spin(0, 8, 0)
        form.addRow("Use Channel...", self.channel_spin)
        self.fallback_hz_spin = _spin(0.01, 10000.0, 10.0, 0.5, decimals=2)
        form.addRow(_help_label(
            "Fallback Freq. (Hz)",
            "fallback_hz is used only when ElapsedTimes.yaml is missing or its length "
            "disagrees with the stack. The warning appears in the log — do not ignore "
            "it."), self.fallback_hz_spin)
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
        form = _form(box)
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
        form.addRow(_help_label(
            "Denoise Method",
            # "Applied in sequence, in the fixed order gaussian → nlm → dct."  ## DCT is Deprecated
            "Applied in sequence, in the fixed order gaussian → nlm."), row)

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
        io_form = _form(io_box)
        self.preprocessed_dir_edit = _path_edit("e.g. ./preprocessed")
        self.preprocessed_dir_edit.textChanged.connect(self._refresh_gating)
        io_form.addRow("Preprocessed Data Directory",
                       _path_row(self.preprocessed_dir_edit, self._browse_preprocessed_dir))
        self.zarr_info = _info()
        io_form.addRow(_help_label(
            "resolved path",
            "Preprocess (denoise) methods are indicated in the filename. Changing it retargets which store "
            "is saved AND loaded — check the resolved path before pressing anything."),
            self.zarr_info)
        layout.addWidget(io_box)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build lazy stack")
        self.build_btn.clicked.connect(self._on_build_stack)
        self.save_zarr_btn = QPushButton("Save to Zarr…")
        self.save_zarr_btn.clicked.connect(self._on_save_zarr)
        self.load_zarr_btn = QPushButton("Load from Zarr")
        self.load_zarr_btn.clicked.connect(self._on_load_zarr)

        # Build new .zarr
        buttons.addWidget(self.build_btn)
        buttons.addWidget(_HelpIcon(
            "‘Build lazy stack’ only wires the chain up and shows it in napari; nothing is "
            "computed until napari draws a frame or you save. Saving computes every frame "
            "and is slow."))

        # Save/Load .zarr
        for button in (self.save_zarr_btn, self.load_zarr_btn):
            buttons.addWidget(button)
        
        buttons.addStretch(1)
        layout.addLayout(buttons)
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
        """Where the current dataset + denoise chain would be cached, or None.

        Gated on `dataset_tag`, not on `data_dir`: a .zarr loaded on its own is a complete
        entry point and carries its tag in its filename, with no .imgdir behind it.
        Requiring data_dir here left the resolved path blank for exactly that route --
        the one case where the store is already known.
        """
        method = self._denoise_method()
        tag = self.session.dataset_tag
        if not (tag and method):
            return None
        out_dir = self._dir(self.preprocessed_dir_edit)
        if out_dir is None:
            return None
        return out_dir / f"{tag}_{at.store.method_tag(method)}.zarr"

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

        box = QGroupBox("ROI labels (uint16 .tiff label image)")
        form = _form(box)
        self.labels_dir_edit = _path_edit("e.g. ./labels")
        self.labels_dir_edit.textChanged.connect(self._refresh_gating)
        form.addRow(_help_label(
            "Labels Directory",
            "Labels Directory (LABELS_DIR) only sets where the load/save dialogs open. The labels path itself "
            "is chosen in those dialogs and recorded with the traces, so it always reflects "
            "the file actually used."),
            _path_row(self.labels_dir_edit, self._browse_labels_dir))
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
        buttons.addWidget(_HelpIcon(
            "Paint ROIs on the 'ROI labels' layer in napari with the brush tool; press M for "
            "a fresh label id per ROI. The labels are sized against the preprocessed stack — "
            "this build renders no max/STD projection."))
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.roi_info = _info("No ROI labels loaded.")
        layout.addWidget(self.roi_info)
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
        self._check_labels_against_traces(labels)
        self._refresh_gating()

    def _check_labels_against_traces(self, labels):
        """Report whether a label image and the loaded traces describe the same ROIs.

        In full mode the shape check already rejects labels drawn on another recording.
        Traces-only mode has no stack to check against, so the ids are the only thing the
        two artifacts share -- and a group mask painted from a mismatched label image is
        silently blank for every ROI that does not line up, which looks identical to an
        unpainted one.
        """
        s = self.session
        if s.roi_ids is None:
            return
        painted = set(at.traces.roi_ids_in(labels))
        analysed = {int(r) for r in s.roi_ids}
        missing, extra = sorted(analysed - painted), sorted(painted - analysed)
        if not (missing or extra):
            self._log(f"> Labels and traces agree: the same {len(analysed)} ROI id(s) in both.")
            return
        if missing:
            self._log(f"! {len(missing)} ROI(s) in the traces have no label of that id and "
                      f"would be blank in a group mask: {_id_list(missing)}.")
        if extra:
            self._log(f"! {len(extra)} label(s) have no matching trace: {_id_list(extra)}.")

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
        form = _form(self.extract_box)
        self.extract_zarr_edit = _path_edit("(none selected) — <dataset_tag>_<method>.zarr")
        self.extract_zarr_edit.textChanged.connect(self._refresh_gating)
        form.addRow(_help_label(
            "source .zarr",
            "Traces are read from this store on disk, not from whatever is currently in the "
            "viewer. Filled in automatically when a store is loaded or saved in tab 2; "
            "override it here to extract from a different preprocessing run."),
            _path_row(self.extract_zarr_edit, self._browse_extract_zarr))
        self.start_spin = _spin(0, 10_000_000, 0)
        form.addRow("START_TIMEPOINT", self.start_spin)
        self.extract_spin = _spin(1, 10_000_000, 500)
        form.addRow("EXTRACT_TIMEPOINTS", self.extract_spin)
        self.baseline_spin = _spin(1, 10_000_000, 20)
        form.addRow(_help_label(
            "BASELINE_SLIDES",
            "F0 is the mean of the first BASELINE_SLIDES frames, so dF/F is ~0 there by "
            "construction. Keep the analysis 'pre' phase after that window (tab 6)."),
            self.baseline_spin)
        self.extract_btn = QPushButton("Extract traces")
        self.extract_btn.clicked.connect(self._on_extract)
        form.addRow(self.extract_btn)
        layout.addWidget(self.extract_box)

        export_box = QGroupBox("Export traces")
        export_layout = QVBoxLayout(export_box)
        self.analysis_dir_edit = _path_edit("(none selected) — e.g. ./analysis")
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

        import_box = QGroupBox("Traces-only entry — import dF/F from anywhere")
        import_form = _form(import_box)
        self.import_path_edit = QLineEdit()
        self.import_path_edit.textChanged.connect(self._refresh_gating)
        import_form.addRow("file", _path_row(self.import_path_edit, self._browse_import))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["auto", "roi_rows", "roi_cols"])
        import_form.addRow("orientation", self.orientation_combo)
        self.time_column_edit = QLineEdit("auto")
        import_form.addRow("time_column", self.time_column_edit)
        self.import_rate_spin = _spin(0.0, 100000.0, 0.0, 0.5, decimals=3)
        import_form.addRow(_help_label(
            "frame_rate_hz",
            "Required (non-zero) when the file carries no time axis — it cannot be inferred, "
            "and CASCADE resampling and the phase windows both depend on "
            # [Index B disabled] "the frequency bands" also depended on it
            "it. Importing disables tabs 1–3: there is no pixel data behind these traces."),
            self.import_rate_spin)
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
        form = _form(box)
        self.cascade_dir_edit = _path_edit("(none selected) — e.g. ./CascadeTorch")
        self.cascade_dir_edit.textChanged.connect(self._on_cascade_dir_changed)
        form.addRow("CascadeTorch Directory", _path_row(self.cascade_dir_edit, self._browse_cascade_dir))
        self.model_combo = _combo()
        self.model_combo.currentTextChanged.connect(self._describe_model)
        form.addRow(_help_label(
            "MODEL_NAME",
            "Model choice is a scientific claim, not a preference. The default is a spinal "
            "dorsal-horn model applied to DRG — the nearest available analogue, and an "
            "untested transfer (CLAUDE.md §4C, App. A §8). No DRG-specific ground truth "
            "exists; state this wherever the output is quoted."), self.model_combo)
        layout.addWidget(box)

        self.model_info = _info()
        layout.addWidget(self.model_info)

        download_row = QHBoxLayout()
        self.download_btn = QPushButton("Download pretrained models…")
        self.download_btn.clicked.connect(self._on_download_model)
        download_row.addWidget(self.download_btn)
        download_row.addWidget(_HelpIcon(
            "The repository ships no model weights — only the index of download links, "
            "Pretrained_models/available_models_CascadeTorch.yaml. Downloaded models land "
            "in CASCADE_DIR/Pretrained_models/<name>/ and appear in MODEL_NAME above. A "
            "model is a few MB to tens of MB; the list is whatever that index file says, "
            "nothing is fetched to build it."))
        download_row.addStretch(1)
        layout.addLayout(download_row)

        buttons = QHBoxLayout()
        self.run_cascade_btn = QPushButton("Run CASCADE")
        self.run_cascade_btn.clicked.connect(self._on_run_cascade)
        self.save_spike_btn = QPushButton("Save inferred spikes…")
        self.save_spike_btn.clicked.connect(self._on_save_spike_rate)
        self.load_spike_btn = QPushButton("Load saved inference…")
        self.load_spike_btn.clicked.connect(self._on_load_spike_rate)
        buttons.addWidget(self.run_cascade_btn)
        buttons.addWidget(self.save_spike_btn)
        buttons.addWidget(_HelpIcon(
            "Saving asks for the destination folder each time rather than writing to the "
            "analysis root automatically, so a second run cannot overwrite the "
            "spike_rate.npy of an earlier one for the same dataset."))
        buttons.addWidget(self.load_spike_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

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

    def _reload_models(self, select=None):
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
        # `select` is the model just downloaded: pick it, so pressing Download and then
        # Run CASCADE does what it looks like it does.
        if select in models:
            self.model_combo.setCurrentText(select)
        elif default in models:
            self.model_combo.setCurrentText(default)
        self.model_combo.blockSignals(False)
        self._describe_model()

    def _on_download_model(self):
        """Choose a model in the picker, then run the same download job as before."""
        model_dir = self._model_dir()
        if model_dir is None:
            return self._warn("Set CASCADE_DIR before downloading a model.")
        try:
            index = at.cascade_runner.model_index(model_dir)
            installed = at.cascade_runner.available_models(model_dir)
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
            return self._warn(f"Cannot list the downloadable models: {exc}")

        dialog = _ModelPickerDialog(model_dir, index, installed, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  # cancelled
        name = dialog.selected_model()
        if not name:
            return
        if at.cascade_runner.is_installed(model_dir, name):
            answer = QMessageBox.question(
                self, "Already installed",
                f"{name} is already in {model_dir.name}/.\n\n"
                "Download it again and replace the local copy?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        model_dir.mkdir(parents=True, exist_ok=True)

        def work(progress):
            return at.cascade_runner.download_model(model_dir, name, progress=progress)

        def done(path):
            self._log(f"> Model ready -> {path}")
            self._reload_models(select=name)
            self._refresh_gating()

        self._run(work, done, f"Downloading {name}…", with_progress=True)

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
        self._sync_region_bounds()
        self._invalidate_preview()
        self.inference_info.setText(
            f"Inferred rate <b>{spike_rate.shape[0]}</b> ROI(s) × "
            f"<b>{spike_rate.shape[1]}</b> frames · {pad} NaN pad frame(s) per end · "
            "units spikes/s")
        self._refresh_gating()

    # ------------------------------------------------------------------ tab 6
    def _build_analysis_tab(self):
        """Tab 6 is itself split: 'Setup' holds the controls, 'Live preview' the canvases."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.addTab(self._build_analysis_setup(), "Setup")
        self.analysis_tabs.addTab(self._build_preview_page(), "Live preview")
        page_layout.addWidget(self.analysis_tabs)
        return page

    def _build_analysis_setup(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # --- analysis window: phases OR one region, never both on screen at once ------
        window_box = QGroupBox("Analysis window")
        window_layout = QVBoxLayout(window_box)
        mode_row = QHBoxLayout()
        self.phase_mode_radio = QRadioButton("Chop up into phases")
        self.region_mode_radio = QRadioButton("Select a region")
        self.phase_mode_radio.setChecked(True)
        self.window_mode_group = QButtonGroup(self)
        self.window_mode_group.addButton(self.phase_mode_radio, 0)
        self.window_mode_group.addButton(self.region_mode_radio, 1)
        mode_row.addWidget(self.phase_mode_radio)
        mode_row.addWidget(_HelpIcon(
            "Phases are cut from the whole recording. The first phase doubles as the "
            "ACTIVE_THRESH baseline, so start it after the F0 window, not at 0: frames "
            "0..BASELINE_SLIDES defined F0, so dF/F there is ~0 by construction and would "
            "fake a silent baseline."))
        mode_row.addSpacing(16)
        mode_row.addWidget(self.region_mode_radio)
        mode_row.addWidget(_HelpIcon(
            "The analysis runs on this window only, as a single phase. ACTIVE_THRESH is "
            "measured BEFORE the region rather than inside it — a threshold taken from the "
            "window being tested is set by the response it is meant to detect."))
        mode_row.addStretch(1)
        window_layout.addLayout(mode_row)

        self.window_stack = QStackedWidget()
        self.window_stack.addWidget(self._build_phase_page())
        self.window_stack.addWidget(self._build_region_page())
        self.window_mode_group.idToggled.connect(self._on_window_mode_changed)
        window_layout.addWidget(self.window_stack)
        layout.addWidget(window_box)

        # [Index B disabled] the frequency-band table -- the only consumer of the bands
        # was freq_features, so with Index B off there is nothing to configure here.
        # band_box = QGroupBox("Index B frequency bands (Hz)")
        # band_layout = QVBoxLayout(band_box)
        # self.band_table = QTableWidget(4, 2)
        # self.band_table.setHorizontalHeaderLabels(["low", "high"])
        # for row, (lo, hi) in enumerate([(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]):
        #     self.band_table.setItem(row, 0, QTableWidgetItem(str(lo)))
        #     self.band_table.setItem(row, 1, QTableWidgetItem(str(hi)))
        # self.band_table.setMaximumHeight(160)
        # band_layout.addWidget(self.band_table)
        # self.band_caption = _caption("")
        # band_layout.addWidget(self.band_caption)
        # layout.addWidget(band_box)

        opts_box = QGroupBox("Thresholds, PCA and clustering")
        form = _form(opts_box)
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
        self.max_k_spin = _spin(2, 50, 12)
        self.max_k_spin.valueChanged.connect(self._on_max_k_changed)
        form.addRow("MAX_K", self.max_k_spin)
        self.min_group_spin = _spin(1, 100, 3)
        form.addRow("MIN_GROUP_SIZE", self.min_group_spin)
        self.max_frac_spin = _spin(0.1, 1.0, 0.9, 0.05, decimals=2)
        form.addRow("MAX_GROUP_FRAC", self.max_frac_spin)
        layout.addWidget(opts_box)

        cluster_box = QGroupBox("Cluster count")
        cluster_form = _form(cluster_box)
        self.rate_k_slider, self.rate_k_label = self._k_slider("rate")
        cluster_form.addRow(_help_label(
            "Index A groups (k)",
            "k cuts the tree with fcluster(maxclust); the equivalent height is shown beside "
            "the slider and written to analysis_params.json, so k-specified runs stay "
            "comparable with height-specified ones. Fill an override to pin a height "
            "instead — it then takes precedence over k."),
            self._k_row(self.rate_k_slider, self.rate_k_label))
        # [Index B disabled] second k slider and its height override
        # self.freq_k_slider, self.freq_k_label = self._k_slider("freq")
        # cluster_form.addRow("Index B groups (k)", self._k_row(self.freq_k_slider,
        #                                                       self.freq_k_label))
        self.rate_cut_edit = QLineEdit()
        self.rate_cut_edit.setPlaceholderText("blank = use k above")
        cluster_form.addRow("RATE_CUT_HEIGHT override", self.rate_cut_edit)
        # [Index B disabled]
        # self.freq_cut_edit = QLineEdit()
        # self.freq_cut_edit.setPlaceholderText("blank = use k above")
        # cluster_form.addRow("FREQ_CUT_HEIGHT override", self.freq_cut_edit)
        layout.addWidget(cluster_box)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview (no files written)")
        self.preview_btn.clicked.connect(self._on_preview_analysis)
        self.run_analysis_btn = QPushButton("Run analysis and export (CSV + TIFF)")
        self.run_analysis_btn.clicked.connect(self._on_run_analysis)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(_HelpIcon(
            "Preview computes the features, PCA and tree and draws them in the Live preview "
            "tab without writing anything; k can then be dragged with no recomputation."))
        buttons.addWidget(self.run_analysis_btn)
        buttons.addWidget(_HelpIcon(
            "Exporting asks once for a parent folder and writes an auto-named subfolder per "
            "run, so configurations never overwrite each other. Group labels are an "
            "arbitrary integer labelling, not a cell type."))
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.analysis_info = _info("Not run.")
        layout.addWidget(self.analysis_info)
        layout.addStretch(1)
        return page

    def _build_phase_page(self):
        page = QWidget()
        phase_layout = QVBoxLayout(page)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        self.phase_table = QTableWidget(3, 3)
        self.phase_table.setHorizontalHeaderLabels(["name", "start", "end"])
        for row, (name, start, end) in enumerate(
                [("pre", 2.0, 10.0), ("injury", 10.0, 15.0), ("post", 15.0, 50.0)]):
            for col, value in enumerate((name, start, end)):
                self.phase_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.phase_table.setMaximumHeight(160)
        self.phase_table.itemChanged.connect(self._invalidate_preview)
        phase_layout.addWidget(self.phase_table)

        row_buttons = QHBoxLayout()
        add_btn = QPushButton("Add phase")
        add_btn.clicked.connect(self._on_add_phase_row)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_phase_row)
        row_buttons.addWidget(add_btn)
        row_buttons.addWidget(remove_btn)
        row_buttons.addStretch(1)
        phase_layout.addLayout(row_buttons)
        return page

    def _build_region_page(self):
        page = QWidget()
        region_layout = QVBoxLayout(page)
        region_layout.setContentsMargins(0, 0, 0, 0)

        self.region_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.region_slider.setRange(0, 1)
        self.region_slider.setValue((0, 1))
        self.region_slider.valueChanged.connect(self._on_region_changed)
        region_layout.addWidget(self.region_slider)

        spin_row = QHBoxLayout()
        self.region_start_spin = _spin(0, 10_000_000, 0)
        self.region_end_spin = _spin(0, 10_000_000, 1)
        for label, box in (("start frame", self.region_start_spin),
                           ("end frame", self.region_end_spin)):
            spin_row.addWidget(QLabel(label))
            spin_row.addWidget(box)
            box.valueChanged.connect(self._on_region_spin_changed)
        spin_row.addStretch(1)
        region_layout.addLayout(spin_row)

        self.region_info = _info("Load an inferred rate to set the region.")
        region_layout.addWidget(self.region_info)
        # [Index B disabled] the region note (now on the mode radio) also warned that a
        # band narrower than the FFT bin width (frame rate / frames) is refused.
        return page

    def _k_slider(self, which):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(2, self.max_k_spin.value() if hasattr(self, "max_k_spin") else 12)
        slider.setValue(4)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        label = QLabel("k=4")
        slider.valueChanged.connect(lambda _v, w=which: self._on_k_changed(w))
        return slider, label

    @staticmethod
    def _k_row(slider, label):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider, 1)
        label.setMinimumWidth(150)
        row_layout.addWidget(label)
        return row

    def _build_preview_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        self.preview_index_combo = QComboBox()
        self.preview_index_combo.addItem("Index A — inferred rate", "rate")
        # [Index B disabled] with one entry left the combo has nothing to switch between,
        # so it is hidden rather than shown as a dropdown that cannot drop.
        # self.preview_index_combo.addItem("Index B — frequency", "freq")
        self.preview_index_combo.currentIndexChanged.connect(lambda _i: self._redraw_preview())
        self.preview_index_combo.setVisible(False)
        header.addWidget(QLabel("Showing <b>Index A</b> — inferred rate "
                                "(Index B is disabled in this build)"))
        header.addStretch(1)
        layout.addLayout(header)

        self.preview_status = _info("Press ‘Preview’ in the Setup tab.")
        layout.addWidget(self.preview_status)

        self.preview_dendrogram = _Canvas(4.6)
        self.preview_scatter = _Canvas(4.6)
        self.preview_traces = _Canvas(6.0)
        self.preview_explainer = _caption("")

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.addWidget(self.preview_dendrogram)
        inner_layout.addWidget(self.preview_scatter)
        inner_layout.addWidget(self.preview_explainer)
        inner_layout.addWidget(self.preview_traces)
        inner_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        return page

    # --- tab 6 window mode ----------------------------------------------------------
    def _on_window_mode_changed(self, index, checked):
        if not checked:
            return
        self.window_stack.setCurrentIndex(index)
        self._sync_region_bounds()
        self._invalidate_preview()

    def _region_mode(self):
        return self.region_mode_radio.isChecked()

    def _on_add_phase_row(self):
        row = self.phase_table.rowCount()
        self.phase_table.insertRow(row)
        for col, value in enumerate((f"phase{row + 1}", 0.0, 0.0)):
            self.phase_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _on_remove_phase_row(self):
        rows = sorted({i.row() for i in self.phase_table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self.phase_table.rowCount() - 1]
        for row in rows:
            if self.phase_table.rowCount() > 1:
                self.phase_table.removeRow(row)
        self._invalidate_preview()

    def _sync_region_bounds(self):
        """Point the region controls at the loaded recording's frame count.

        Until the handles are dragged the region tracks the whole recording, so switching
        to region mode analyses everything rather than the two-frame sliver a freshly
        constructed slider would otherwise report.
        """
        t = self.session.t_plot
        if t is None or len(t) < 2:
            return
        last = len(t) - 1
        for widget in (self.region_slider, self.region_start_spin, self.region_end_spin):
            widget.blockSignals(True)
        self.region_slider.setRange(0, last)
        self.region_start_spin.setRange(0, last)
        self.region_end_spin.setRange(0, last)
        start, end = self.region_slider.value()
        if not self._region_touched or end <= start or end > last:
            start, end = 0, last
            self.region_slider.setValue((start, end))
        self.region_start_spin.setValue(start)
        self.region_end_spin.setValue(end)
        for widget in (self.region_slider, self.region_start_spin, self.region_end_spin):
            widget.blockSignals(False)
        self._describe_region()

    def _on_region_changed(self, value):
        start, end = value
        self._region_touched = True
        self.region_start_spin.blockSignals(True)
        self.region_end_spin.blockSignals(True)
        self.region_start_spin.setValue(start)
        self.region_end_spin.setValue(end)
        self.region_start_spin.blockSignals(False)
        self.region_end_spin.blockSignals(False)
        self._describe_region()
        self._invalidate_preview()

    def _on_region_spin_changed(self):
        start, end = self.region_start_spin.value(), self.region_end_spin.value()
        if end <= start:
            return  # half-typed range; wait for the other box
        self._region_touched = True
        self.region_slider.blockSignals(True)
        self.region_slider.setValue((start, end))
        self.region_slider.blockSignals(False)
        self._describe_region()
        self._invalidate_preview()

    def _region_frames(self):
        """(start, end) frame indices of the region, inclusive of both ends."""
        start, end = self.region_slider.value()
        return int(start), int(end)

    def _describe_region(self):
        t = self.session.t_plot
        if t is None:
            return
        start, end = self._region_frames()
        end = min(end, len(t) - 1)
        n = end - start + 1
        rate = self.session.frame_rate or 0.0
        bin_width = rate / n if n else float("inf")
        self.region_info.setText(
            f"frames <b>{start}–{end}</b> of 0–{len(t) - 1} · "
            f"<b>{t[start]:.2f}–{t[end]:.2f} s</b> · {n} frames = {n / rate:.2f} s"
            # [Index B disabled] the FFT resolution of the region, which only mattered
            # for the frequency bands:
            # f"<br>Index B resolution here: FFT bin width {bin_width:.3f} Hz — bands "
            # f"below that are refused."
        )

    def _phases(self):
        """The phase dict for the current mode: the table, or the region as one phase."""
        if self._region_mode():
            t = self.session.t_plot
            if t is None:
                raise ValueError("Load traces and an inferred rate before selecting a region.")
            start, end = self._region_frames()
            end = min(end, len(t) - 1)
            if end <= start:
                raise ValueError("The region is empty; drag the two handles apart.")
            # Named for the window it covers, so the export folder and every figure legend
            # carry the region rather than a generic label.
            return {f"region_{t[start]:.1f}-{t[end]:.1f}s": (float(t[start]),
                                                             float(t[end]) + 1e-9)}
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

    # [Index B disabled] reader for the frequency-band table
    # def _bands(self):
    #     bands = []
    #     for row in range(self.band_table.rowCount()):
    #         lo, hi = self.band_table.item(row, 0), self.band_table.item(row, 1)
    #         if lo and hi and lo.text().strip() and hi.text().strip():
    #             bands.append((float(lo.text()), float(hi.text())))
    #     if not bands:
    #         raise ValueError("Define at least one frequency band.")
    #     return bands

    def _analysis_cfg(self, out_dir=None):
        """Validate the current settings and build the config both paths share.

        Returns None after warning the user when anything is unusable -- the checks that
        used to sit inline in the run handler, now also guarding the preview so an empty
        region or a phase with no frames is caught before any work starts.
        """
        s = self.session
        try:
            phases = self._phases()
            # [Index B disabled] bands = self._bands()
        except ValueError as exc:
            self._warn(str(exc))
            return None

        # [Index B disabled] the Nyquist guard on the top band edge. It only constrained
        # the frequency bands, and nothing reads them now.
        # The top edge is *meant* to sit at Nyquist, so compare with tolerance: a
        # measured 9.999999 Hz puts Nyquist at 4.9999995 and would otherwise reject the
        # documented default band of 5.0 Hz on floating-point margin alone.
        # nyquist = s.frame_rate / 2.0
        # top = max(hi for _, hi in bands)
        # if top > nyquist and not np.isclose(top, nyquist, rtol=1e-6):
        #     self._warn(f"The top band edge ({top:g} Hz) exceeds Nyquist "
        #                f"({nyquist:.3f} Hz) for a {s.frame_rate:.3f} Hz recording. "
        #                "Lower it — there is no information above Nyquist to recover.")
        #     return None

        # [Index B disabled] the CASCADE_DIR requirement. It existed only because
        # freq_features loads its FFT helper from CascadeTorch/scripts; Index A never
        # needed it, so demanding it now would block the analysis for no reason.
        # if self._dir(self.cascade_dir_edit) is None:
        #     self._warn("Set CASCADE_DIR (tab 5) before running the analysis: the "
        #                "frequency features (Index B) load their FFT helper from "
        #                "CascadeTorch/scripts.")
        #     return None

        cut_rate = self.rate_cut_edit.text().strip()
        # [Index B disabled] cut_freq = self.freq_cut_edit.text().strip()
        return dict(
            phases=phases, out_dir=out_dir,
            # [Index B disabled] bands=bands,
            region=self._region_frames() if self._region_mode() else None,
            cascade_dir=self._dir(self.cascade_dir_edit),
            var_target=self.var_target_spin.value(), max_k=self.max_k_spin.value(),
            min_group=self.min_group_spin.value(), max_frac=self.max_frac_spin.value(),
            n_sigma=self.n_sigma_spin.value(),
            active_thresh=None if self.auto_thresh_check.isChecked()
            else self.active_thresh_spin.value(),
            k_rate=self.rate_k_slider.value(),
            # [Index B disabled] k_freq=self.freq_k_slider.value(),
            cut_rate=float(cut_rate) if cut_rate else None,
            # [Index B disabled] cut_freq=float(cut_freq) if cut_freq else None,
            model_name=self.model_combo.currentText(),
        )

    def _analysis_inputs(self, cfg):
        """(spike_rate, dff, roi_ids, t, pad) for the configured window.

        In region mode everything is cropped to the selected frames and `pad` is
        RECOUNTED on the crop: session.pad describes CASCADE's NaN frames at the ends of
        the full trace, and a crop may exclude them entirely or land inside them. Reusing
        the old number would blank real frames in the group-mean panel, or expose NaN
        ones.
        """
        s = self.session
        spike_rate, dff, t, pad = s.spike_rate, s.dff_matrix, s.t_plot, s.pad
        if cfg["region"] is None:
            return spike_rate, dff, s.roi_ids, t, pad

        start, end = cfg["region"]
        end = min(end, len(t) - 1)
        sl = slice(start, end + 1)
        spike_rate, dff, t = spike_rate[:, sl], dff[:, sl], t[sl]
        finite = np.isfinite(spike_rate[0])
        pad = int(np.argmax(finite)) if not finite.all() else 0
        self._log(f"> Region: frames {start}–{end} ({t[0]:.2f}–{t[-1]:.2f} s), "
                  f"{spike_rate.shape[1]} frames; NaN pad recounted as {pad} per end.")
        return spike_rate, dff, s.roi_ids, t, pad

    def _baseline_mask(self, cfg):
        """Baseline frames for ACTIVE_THRESH, as a mask over the FULL time axis.

        Always the full axis, never the crop, so the threshold can be measured outside
        the window being analysed. In phase mode that is the first phase, as before. In
        region mode it is everything BEFORE the region: a threshold measured inside the
        window under test is set by the very response it is meant to detect, so a region
        covering the stimulus would report almost nothing as active.
        """
        full_t = self.session.t_plot
        if cfg["region"] is None:
            first = list(cfg["phases"])[0]
            return at.features.phase_masks(full_t, cfg["phases"])[first], f"phase {first!r}"

        start, end = cfg["region"]
        if start == 0:
            self._log("! The region starts at frame 0, so nothing precedes it: "
                      "ACTIVE_THRESH falls back to the region itself. That is circular — "
                      "set a manual threshold, or start the region later.")
            mask = np.zeros(len(full_t), dtype=bool)
            mask[start:min(end, len(full_t) - 1) + 1] = True
            return mask, "the region itself (no earlier frames exist)"
        mask = at.features.baseline_mask_before(full_t, float(full_t[start]))
        return mask, f"frames 0–{start - 1} ({full_t[0]:.2f}–{full_t[start - 1]:.2f} s)"

    def _resolve_active_thresh(self, cfg):
        """The threshold and a one-line provenance string, ready for the params JSON."""
        if cfg["active_thresh"] is not None:
            return float(cfg["active_thresh"]), "set manually"
        mask, where = self._baseline_mask(cfg)
        thresh, med, sigma = at.features.auto_active_thresh(
            self.session.spike_rate, mask, cfg["n_sigma"])
        return thresh, (f"auto: median + {cfg['n_sigma']:g}·σ over {where} "
                        f"(median {med:.3f}, σ {sigma:.3f})")

    # --- preview --------------------------------------------------------------------
    def _on_preview_analysis(self):
        cfg = self._analysis_cfg()
        if cfg is None:
            return
        spike_rate, dff, roi_ids, t, pad = self._analysis_inputs(cfg)
        try:
            thresh, provenance = self._resolve_active_thresh(cfg)
        except ValueError as exc:
            return self._warn(str(exc))
        cfg["active_thresh"], cfg["thresh_provenance"] = thresh, provenance

        def work():
            return _compute_features(spike_rate, dff, roi_ids, t, pad, self.session.frame_rate,
                                     cfg)

        def done(bundle):
            self._preview = bundle
            self._sync_k_ranges()
            self._redraw_preview()
            self.analysis_tabs.setCurrentIndex(1)
            self._refresh_gating()

        self._run(work, done, "Computing features, PCA and tree…")

    def _invalidate_preview(self, *_):
        """Drop a preview whose configuration no longer matches the controls.

        Anything that changes the feature table invalidates it. The alternative -- leaving
        the old figures up -- means the panel shows a grouping of a window the user is no
        longer looking at, which is exactly the mistake the preview exists to prevent.
        """
        if getattr(self, "_preview", None) is None:
            return
        self._preview = None
        for canvas in (self.preview_dendrogram, self.preview_scatter, self.preview_traces):
            canvas.clear()
        self.preview_explainer.setText("")
        self.preview_status.setText(
            "Settings changed — press ‘Preview’ in the Setup tab to recompute.")
        self._refresh_gating()

    def _sync_k_ranges(self):
        """Cap both k sliders at what the trees can actually deliver (n_rois groups)."""
        bundle = getattr(self, "_preview", None)
        if bundle is None:
            return
        n_rois = len(bundle["roi_ids"])
        # [Index B disabled] the freq entry, ("freq", self.freq_k_slider), is dropped here
        for which, slider in (("rate", self.rate_k_slider),):
            top = max(2, min(self.max_k_spin.value(), n_rois))
            slider.blockSignals(True)
            slider.setRange(2, top)
            slider.setValue(min(slider.value(), top))
            slider.blockSignals(False)
            self._update_k_label(which)

    def _on_max_k_changed(self):
        self._sync_k_ranges()
        self._invalidate_preview()  # MAX_K also drives cut_candidates in the export

    def _on_k_changed(self, which):
        self._update_k_label(which)
        if getattr(self, "_preview", None) is None:
            return
        # Debounced: dragging the slider fires valueChanged per pixel, and each redraw
        # costs ~100-200 ms. Coalescing to one redraw per idle moment keeps the drag
        # smooth instead of queueing a backlog of stale frames.
        self._preview_timer.start(120)

    def _update_k_label(self, which):
        # [Index B disabled] both lookups collapse to the rate widgets; restore the
        # `if which == "rate" else self.freq_k_slider / self.freq_k_label` forms with it.
        slider, label = self.rate_k_slider, self.rate_k_label
        bundle = getattr(self, "_preview", None)
        text = f"k={slider.value()}"
        if bundle is not None:
            try:
                height = at.grouping.height_for_k(bundle[which]["Z"], slider.value())
                text += f"  (h={height:.2f})"
            except ValueError:
                text += "  (beyond the tree)"
        label.setText(text)

    def _redraw_preview(self):
        """Re-cut the stored tree at the current k and redraw. No recomputation."""
        bundle = getattr(self, "_preview", None)
        if bundle is None:
            return
        # [Index B disabled] the combo carries a single hidden entry, so fall back to
        # "rate" rather than KeyError-ing on an empty or cleared selection.
        which = self.preview_index_combo.currentData() or "rate"
        part = bundle[which]
        # [Index B disabled] `which` can only be "rate", so the freq branches collapse:
        # k = (self.rate_k_slider if which == "rate" else self.freq_k_slider).value()
        # prefix = "G" if which == "rate" else "F"
        k = self.rate_k_slider.value()
        prefix = "G"

        groups = at.grouping.cut_tree_k(part["Z"], k, bundle["roi_ids"],
                                        self.max_frac_spin.value(), verbose=False)
        height = at.grouping.height_for_k(part["Z"], k)
        label = "Index A (inferred rate)" if which == "rate" else "Index B (frequency)"

        self.preview_dendrogram.show_figure(
            at.plots.dendrogram_plot(part["Z"], bundle["roi_ids"], label, height))
        self.preview_scatter.show_figure(
            at.plots.pca_scatter(part["pca"], part["scores"], bundle["roi_ids"], groups,
                                 f"{label} — ROIs in PC space, coloured by group"))
        self.preview_traces.show_figure(
            at.plots.group_means(bundle["spike_rate"], bundle["t"], groups,
                                 bundle["roi_ids"], bundle["stim"], bundle["pad"],
                                 f"{label} — inferred rate per {prefix}-group at k={k}"))
        self._update_k_label(which)

        sizes = np.bincount(groups)[1:]
        self.preview_status.setText(
            f"<b>{int(groups.max())}</b> group(s) at k={k} (h={height:.2f}) · sizes "
            f"{', '.join(str(int(n)) for n in sizes)} · {len(bundle['roi_ids'])} ROIs · "
            f"window: {bundle['window_label']}")
        self.preview_explainer.setText(_pca_explainer(part, bundle, which))

    def _on_run_analysis(self):
        cfg = self._analysis_cfg()
        if cfg is None:
            return
        spike_rate, dff, roi_ids, t, pad = self._analysis_inputs(cfg)
        try:
            thresh, provenance = self._resolve_active_thresh(cfg)
        except ValueError as exc:
            return self._warn(str(exc))
        cfg["active_thresh"], cfg["thresh_provenance"] = thresh, provenance

        # A parent folder, asked once per session; each run gets its own auto-named
        # subfolder under it. Deriving the whole path from <root>/<tag> instead would give
        # every run of a dataset the same destination, so a second region or a second k
        # would overwrite the first with no trace of it.
        parent = self._analysis_parent()
        if parent is None:
            return  # cancelled
        out_dir = parent / _run_folder_name(cfg)
        if out_dir.exists() and any(out_dir.iterdir()):
            answer = QMessageBox.question(
                self, "Overwrite?",
                f"{out_dir.name} already holds output from an identical configuration.\n\n"
                "Overwrite it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg["out_dir"] = out_dir
        self._log(f"> Analysis outputs -> {out_dir}")

        rate = self.session.frame_rate
        # Read here, on the UI thread: `_current_labels` prefers the live napari layer,
        # and a worker must not touch it. None in traces-only mode.
        labels = self._current_labels()

        def work():
            return _compute_analysis(spike_rate, dff, roi_ids, t, pad, rate, cfg)

        def done(bundle):
            self._export_group_mask(bundle, labels)
            self._finish_analysis(bundle)

        self._run(work, done, "Running features, PCA and clustering…")

    def _export_group_mask(self, bundle, labels):
        """Write this run's grouping back into image space, as two TIFFs.

        `roi_groups.csv` says which group an ROI landed in; these say WHERE that group is.

        `roi_group_mask.tiff` is the data copy: uint16, each ROI's pixels carrying its
        Index A group number (1..k), 0 elsewhere. Values, not colours -- read it to count
        pixels or to re-colour it however you like.

        `roi_group_overlay.tiff` is the picture: RGBA, each ROI painted in the tab10 colour
        its group has in the trace panels and the PC-space scatter, background fully
        transparent. Lay it over a max projection of the recording and a group is the same
        colour there as it is in every figure of the same run. Both files come from the
        same `plots.group_color`, so they cannot drift apart.

        Written per run, next to the CSVs, because the grouping is a property of the run:
        a different window or a different k paints a different mask, and both are correct
        for the settings that produced them.

        `labels` is passed in rather than read here -- see the caller. None in traces-only
        mode until an ROI label image is loaded in tab 3.
        """
        if labels is None:
            # Not recoverable from anything else in the session: the mask needs the pixels
            # each ROI occupies, and traces alone carry no geometry. Load or paint a label
            # image in tab 3 -- available in traces-only mode for exactly this -- and run
            # the analysis again.
            self._log("> No ROI label image in this session, so no group mask was written. "
                      "Load the matching ROI labels in tab 3 and re-run to get one.")
            return None

        labels = np.asarray(labels)
        roi_ids, groups = bundle["roi_ids"], np.asarray(bundle["rate_groups"])
        out_dir = Path(bundle["cfg"]["out_dir"])

        # label id -> group, as a lookup indexed by label value: one vectorised pass over
        # the image rather than a full-frame comparison per ROI. Anything not analysed
        # keeps the 0 it was initialised with, which is also the background value.
        painted = set(at.traces.roi_ids_in(labels))
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.uint16)
        missing = [int(r) for r in roi_ids if int(r) not in painted]
        for roi, group in zip(roi_ids, groups):
            if int(roi) in painted:
                lookup[int(roi)] = int(group)

        mask = lookup[labels]
        n_groups = int(groups.max())
        path = out_dir / "roi_group_mask.tiff"
        at.store.save_roi_labels(path, mask)
        self._log(f"> Group mask -> {path} (pixel value = Index A group "
                  f"1..{n_groups}, 0 = background)")

        # The colour copy. Painted group by group rather than through a lookup table:
        # there are at most MAX_K of them, and going via `plots.group_rgb` per group is
        # what ties the file to the figures instead of to a palette copied out by hand.
        # Background keeps alpha 0 -- an opaque black frame would hide the projection this
        # is meant to sit on top of.
        rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
        for g in range(1, n_groups + 1):
            sel = mask == g
            rgba[sel, :3] = at.plots.group_rgb(g)
            rgba[sel, 3] = 255
        overlay_path = out_dir / "roi_group_overlay.tiff"
        at.store.save_rgba_overlay(overlay_path, rgba)
        swatches = " ".join(f"G{g}={at.plots.group_color(g)}"
                            for g in range(1, min(n_groups, 10) + 1))
        self._log(f"> Group overlay -> {overlay_path} (RGBA, tab10, transparent "
                  f"background) · {swatches}"
                  + ("  … colours repeat past G10" if n_groups > 10 else ""))

        # Both directions of disagreement are worth naming: the label image can be
        # repainted after extraction without invalidating the traces (a known gap), and a
        # silently blank ROI in the mask looks identical to an unpainted one.
        if missing:
            self._log(f"! {len(missing)} analysed ROI(s) are absent from the current label "
                      f"image and are blank in the mask: {_id_list(missing)}. The labels "
                      "were changed after these traces were extracted.")
        extra = sorted(painted - {int(r) for r in roi_ids})
        if extra:
            self._log(f"! {len(extra)} ROI(s) in the label image took no part in this "
                      f"analysis and are blank in the mask: {_id_list(extra)}.")
        return path

    def _analysis_parent(self):
        """The parent folder for auto-named run subfolders, asked once and remembered."""
        if self._analysis_parent_dir is not None and self._analysis_parent_dir.exists():
            return self._analysis_parent_dir
        chosen = QFileDialog.getExistingDirectory(
            self, "Parent folder for analysis runs (subfolders are named automatically)",
            self._suggested_output_dir())
        if not chosen:
            return None
        self._analysis_parent_dir = Path(chosen)
        return self._analysis_parent_dir

    def _finish_analysis(self, bundle):
        """Draw and export the figures here, on the UI thread.

        The figures themselves are pyplot-free now (`analysis_tools.plots` builds
        `Figure` objects directly), so this no longer has to be here for thread safety --
        it stays because drawing is the slow part and the status line should update
        between the numbers and the export.
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
            # Name what is actually missing. The old wording named one cause ("load a
            # dataset") out of three, and named a field that has since been renamed, so
            # it misdirected whenever the blocker was an unticked denoise method.
            missing = []
            if not self.session.dataset_tag:
                missing.append("load a dataset or a .zarr store")
            if not self._denoise_method():
                missing.append("tick at least one denoise method")
            if preprocessed_dir is None:
                missing.append("set the preprocessed data directory")
            self.zarr_info.setText("— " + "; ".join(missing))
        else:
            exists = "on disk" if path.exists() else "not yet saved"
            # Also keyed on dataset_tag rather than data_dir, so a session started from a
            # .zarr still lists the other chains cached beside it.
            cached = sorted(p.stem.split("_", 1)[-1]
                            for p in preprocessed_dir.glob(
                                f"{self.session.dataset_tag}_*.zarr"))
            self.zarr_info.setText(f"<code>{path}</code><br><i>{exists}</i>"
                                   + (f" · cached chains: {', '.join(cached)}" if cached else ""))
        # [Index B disabled] Nyquist note under the band table
        # if self.session.frame_rate:
        #     nyquist = self.session.frame_rate / 2.0
        #     self.band_caption.setText(
        #         f"Upper edge is capped by Nyquist = {nyquist:.2f} Hz at "
        #         f"{self.session.frame_rate:.2f} Hz acquisition.")

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
        # Tab 5 is always available, unlike the stages around it: an empty
        # Pretrained_models/ has to be fillable before there are any traces, and the
        # tab holds the model downloader. A disabled tab disables its children, so
        # gating it on has_traces made downloading impossible on a fresh clone. The
        # actions inside it carry their own gates instead.
        # Tab 3 is available in BOTH modes, unlike tabs 1-2. Imported traces arrive with
        # no pixel data behind them, so an ROI label image is the only way the pixel
        # location of each ROI ever re-enters the session -- and without it tab 6 cannot
        # paint its group mask. Extraction stays disabled there; only load/save do work.
        current = self.tabs.currentIndex()
        for index, enabled in enumerate([full, full, True, True, True, has_rate]):
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
        # A blank canvas has to be sized against something, so it stays a full-mode
        # action. Loading does not: in traces-only mode there is no stack to check the
        # shape against, and the ids are cross-checked against the traces instead.
        self.new_labels_btn.setEnabled(
            full and idle and (has_stack or s.frame_shape is not None))
        self.load_labels_btn.setEnabled(
            idle and (not full or has_stack or s.frame_shape is not None))
        self.save_labels_btn.setEnabled(idle and has_labels)
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
        # Downloading needs no traces and no ROIs -- it is how an empty Pretrained_models/
        # gets its first model, so it stays available whatever else the session lacks.
        self.download_btn.setEnabled(idle and self._model_dir() is not None)
        self.save_spike_btn.setEnabled(idle and self._cascade_result is not None)
        # Needs traces for the same reason the tab used to: a rate loaded with no dF/F
        # behind it unlocks tab 6, whose analysis reads dff_matrix and would crash on it.
        self.load_spike_btn.setEnabled(idle and has_traces)
        self.preview_btn.setEnabled(idle and has_rate)
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


def _id_list(ids, limit=10):
    """A comma-separated id list, truncated so one stray ROI set cannot flood the log."""
    head = ", ".join(str(i) for i in ids[:limit])
    return head if len(ids) <= limit else f"{head}, … (+{len(ids) - limit} more)"


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
# Three layers, because the live preview and the export need different amounts of it:
#   _compute_features  features -> PCA -> linkage. No cut, no files. This is what the
#                      preview holds on to, so changing k costs a cut and a redraw.
#   _compute_analysis  the above plus the cut, the summaries and every CSV/npz write.
#   _export_figures    drawing, run on the UI thread after the numbers are in.
# =============================================================================

def _run_folder_name(cfg):
    """Auto-name for one run's output subfolder: window + cut, so runs never collide."""
    if cfg.get("region") is not None:
        window = next(iter(cfg["phases"]))          # already 'region_<t0>-<t1>s'
    else:
        window = f"phases_{'-'.join(cfg['phases'])}"
    cut = (f"h{cfg['cut_rate']:g}" if cfg.get("cut_rate") is not None
           else f"k{cfg.get('k_rate') or 4}")
    return f"{window}_{cut}".replace(" ", "").replace("/", "_")


def _pca_explainer(part, bundle, which):
    """What the PCA is actually measuring, in the panel where the groups are judged.

    Written out because the grouping is easy to over-read: the distances are between
    per-phase summary statistics, not between traces, and which features dominate is a
    consequence of choices made elsewhere in this tab (ACTIVE_THRESH above all).
    """
    pca, features = part["pca"], part["features"]
    var = pca.explained_variance_ratio_ * 100
    n_pc = part["n_pc"]
    names = list(features.columns)

    def top(component, n=3):
        order = np.argsort(np.abs(component))[::-1][:n]
        return ", ".join(f"{names[i]} ({component[i]:+.2f})" for i in order)

    # [Index B disabled] `which` is always "rate" now; the else-branch below is kept so
    # the wording comes back with the index rather than having to be rewritten.
    if which == "rate":
        what = ("three numbers per ROI per phase — mean rate, peak rate and active "
                "fraction (the share of frames above ACTIVE_THRESH)")
        levers = (
            "Because z-scoring gives every feature equal weight, a feature that barely "
            "varies across ROIs contributes almost nothing to the distance. ACTIVE_THRESH "
            "enters only through active fraction: raise it and quiet ROIs collapse "
            "together, lower it and the feature saturates at 1 and separates nothing. "
            "Peak rate is a single-frame statistic, so a bursty ROI sits far from a steady "
            "one at identical mean rate.")
    else:
        what = ("relative band powers plus the dominant frequency, per ROI per phase — "
                "how the inferred rate fluctuates, not how strongly it fires")
        levers = (
            "Band powers are relative (shares of in-band power), so an ROI's overall rate "
            "cancels out and only the SHAPE of its spectrum matters — a quiet ROI and a "
            "loud one with the same rhythm land together. Silent ROI×phase cells score 0 "
            "across every band, which clusters them as a group of their own.")

    return (
        f"<b>What this PCA measures.</b> It runs on the z-scored feature table, not on the "
        f"traces: for {'Index A' if which == 'rate' else 'Index B'} that is {what}. "
        f"Distances between ROIs are Euclidean in the leading {n_pc} PC(s) — those covering "
        f"PCA_VAR_TARGET of the variance — and Ward linkage merges whichever pair adds "
        f"least within-cluster variance.<br>"
        f"<b>How spiking features shape it.</b> {levers}<br>"
        f"<b>This fit:</b> {n_pc} PC(s) cover {np.cumsum(var)[n_pc - 1]:.0f}% of variance "
        f"across {len(names)} features · PC1 ({var[0]:.0f}%) loads most on {top(pca.components_[0])} "
        f"· PC2 ({var[1]:.0f}%) on {top(pca.components_[1])} · ACTIVE_THRESH = "
        f"{bundle['active_thresh']:.3f} spikes/s, {bundle['thresh_provenance']}.<br>"
        f"Groups are a partition of this feature space, not cell types.")


def _compute_features(spike_rate, dff_matrix, roi_ids, t_plot, pad, frame_rate, cfg):
    """Everything up to (but not including) the cut, for both indices.

    Deliberately stops before `cut_tree`: the cut is the one thing the live preview
    changes, and keeping it out of here is what makes dragging k free.
    """
    phases = cfg["phases"]
    masks = at.features.phase_masks(t_plot, phases)
    at.features.describe_phases(t_plot, phases, masks, frame_rate)

    # The GUI resolves the threshold before calling, because only it can reach the
    # UNCROPPED recording that a region's baseline has to come from. Falling back to the
    # first phase here keeps this function usable on its own, which is how it behaved
    # before the region mode existed.
    active_thresh = cfg.get("active_thresh")
    provenance = cfg.get("thresh_provenance")
    if active_thresh is None:
        first = list(phases)[0]
        active_thresh, med, sigma = at.features.auto_active_thresh(
            spike_rate, masks[first], cfg["n_sigma"])
        provenance = (f"auto: median + {cfg['n_sigma']:g}·σ over phase {first!r} of this "
                      f"window (median {med:.3f}, σ {sigma:.3f})")
    print(f"> ACTIVE_THRESH = {active_thresh:.3f} spikes/s ({provenance})")

    rate_feats = at.features.rate_features(spike_rate, roi_ids, masks, active_thresh)
    dff_summary = at.features.dff_summary(dff_matrix, roi_ids, masks)
    at.features.report_threshold(rate_feats, masks, active_thresh)

    # [Index B disabled] band description + frequency features
    # at.features.describe_bands(masks, frame_rate, cfg["bands"])
    # freq_feats = at.features.freq_features(spike_rate, roi_ids, masks, frame_rate,
    #                                        cfg["bands"], cfg["cascade_dir"])

    out = {"cfg": cfg, "roi_ids": roi_ids, "spike_rate": spike_rate, "dff": dff_matrix,
           "t": t_plot, "pad": pad, "frame_rate": frame_rate, "masks": masks,
           "phase_order": list(phases), "dff_summary": dff_summary,
           "active_thresh": active_thresh, "thresh_provenance": provenance,
           "stim": _stim_window(phases, cfg.get("region")),
           "window_label": (next(iter(phases)) if cfg.get("region") is not None
                            else f"{len(phases)} phases over {t_plot[0]:.1f}–{t_plot[-1]:.1f} s")}
    # [Index B disabled] the freq entry, ("freq", freq_feats), is dropped from this loop
    for label, feats in (("rate", rate_feats),):
        pca, scores, n_pc, Z = at.grouping.pca_and_linkage(feats, cfg["var_target"])
        out[label] = {"features": feats, "pca": pca, "scores": scores, "n_pc": n_pc, "Z": Z}
    return out


def _stim_window(phases, region=None):
    """The band shaded in the trace panels: the phase named 'stim', else the second one.

    None when there is no such band to mark. A region analysis is a single window, and a
    lone phase covers the whole axis -- shading it greys out every subplot and says only
    "this is the axis". The plot functions read None as "draw no band".
    """
    if region is not None:
        return None
    if "stim" in phases:
        return phases["stim"]
    values = list(phases.values())
    return values[1] if len(values) > 1 else None


def _resolve_cut(part, cfg, roi_ids, which, label, prefix):
    """Groups + the height that produced them, honouring an explicit height over k."""
    Z = part["Z"]
    override = cfg.get(f"cut_{which}")
    if override is not None:
        return at.grouping.cut_tree(Z, override, roi_ids, cfg["max_frac"], label=label,
                                    prefix=prefix), float(override)
    k = int(cfg.get(f"k_{which}") or 4)
    groups = at.grouping.cut_tree_k(Z, k, roi_ids, cfg["max_frac"], label=label,
                                    prefix=prefix)
    return groups, at.grouping.height_for_k(Z, k)


def _compute_analysis(spike_rate, dff_matrix, roi_ids, t_plot, pad, frame_rate, cfg):
    """Features -> PCA -> Ward cut -> summaries -> CSV/npz, for both indices.

    No plotting: returns everything `_export_figures` needs to draw afterwards.
    """
    out_dir, phases = cfg["out_dir"], cfg["phases"]
    # [Index B disabled] bands = cfg["bands"]
    bundle = _compute_features(spike_rate, dff_matrix, roi_ids, t_plot, pad, frame_rate, cfg)
    phase_order, masks = bundle["phase_order"], bundle["masks"]
    active_thresh, dff_summary = bundle["active_thresh"], bundle["dff_summary"]
    rate_features = bundle["rate"]["features"]
    # [Index B disabled] freq_features = bundle["freq"]["features"]

    # --- Index A (Inferred Spike Rate) -------------------------------------------------------------
    part_rate = bundle["rate"]
    pca_rate, scores_rate, npc_rate, z_rate = (part_rate["pca"], part_rate["scores"],
                                               part_rate["n_pc"], part_rate["Z"])
    print(at.grouping.cut_candidates(z_rate, cfg["max_k"]).to_string())
    rate_groups, cut_rate = _resolve_cut(part_rate, cfg, roi_ids, "rate",
                                         "inferred-rate", "G")
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
    # [Index B disabled] the whole second index: its cut, summary, membership, the
    # rate-vs-frequency agreement (ARI) and both of its exported artifacts.
    # part_freq = bundle["freq"]
    # pca_freq, scores_freq, npc_freq, z_freq = (part_freq["pca"], part_freq["scores"],
    #                                            part_freq["n_pc"], part_freq["Z"])
    # print(at.grouping.cut_candidates(z_freq, cfg["max_k"]).to_string())
    # freq_groups, cut_freq = _resolve_cut(part_freq, cfg, roi_ids, "freq", "frequency", "F")
    # print(at.features.freq_group_summary(freq_features, freq_groups, phase_order,
    #                                      bands).to_string())
    # at.grouping.print_membership(freq_groups, roi_ids, prefix="F")
    # _table, ari = at.grouping.compare_partitions(rate_groups, freq_groups)
    # at.store.save_features(out_dir, "freq_features", freq_features)
    # at.store.save_pca(out_dir, "freq", pca_freq, scores_freq, npc_freq, z_freq,
    #                   list(freq_features.columns), cut_height=cut_freq,
    #                   groups=freq_groups, roi_ids=roi_ids)
    # at.store.save_groups(out_dir, pd.DataFrame({"roi": roi_ids, "rate_group": rate_groups,
    #                                             "freq_group": freq_groups}))
    at.store.save_groups(out_dir, pd.DataFrame({"roi": roi_ids, "rate_group": rate_groups}))
    _save_analysis_params(out_dir, cfg, bundle, rate_groups, cut_rate)

    summary = (f"Index A: {int(rate_groups.max())} group(s) at h={cut_rate:.2f}<br>"
               # [Index B disabled] the Index B group count and the ARI line
               # f"Index B: {int(freq_groups.max())} group(s) at h={cut_freq:.2f} · "
               # f"ARI={ari:.3f}<br>"
               f"ACTIVE_THRESH={active_thresh:.3f} spikes/s "
               f"({bundle['thresh_provenance']})<br>"
               f"Window: {bundle['window_label']}<br>"
               f"Exported to <code>{out_dir}</code> (CSV + figures/*.tiff)")

    return {
        "cfg": cfg, "summary": summary, "spike_rate": spike_rate, "t_plot": t_plot,
        "roi_ids": roi_ids, "pad": pad, "active_thresh": active_thresh,
        "stim": bundle["stim"],
        "rate_features": rate_features,
        "pca_rate": pca_rate, "scores_rate": scores_rate, "z_rate": z_rate,
        "cut_rate": cut_rate, "rate_groups": rate_groups,
        # [Index B disabled]
        # "freq_features": freq_features,
        # "pca_freq": pca_freq, "scores_freq": scores_freq, "z_freq": z_freq,
        # "cut_freq": cut_freq, "freq_groups": freq_groups,
    }


def _save_analysis_params(out_dir, cfg, bundle, rate_groups, cut_rate):
    # [Index B disabled] the signature dropped `freq_groups`, `cut_freq` and `ari`:
    # def _save_analysis_params(out_dir, cfg, bundle, rate_groups, freq_groups,
    #                           cut_rate, cut_freq, ari):
    """Everything needed to reproduce this run, next to its outputs (CLAUDE.md 4).

    The window, the threshold AND WHERE IT CAME FROM, and both cuts -- an exported
    grouping whose threshold provenance is unknown cannot be compared with another run,
    only looked at.
    """
    region = cfg.get("region")
    t = bundle["t"]
    params = {
        "window_mode": "region" if region is not None else "phases",
        "region_frames": list(region) if region is not None else None,
        "window_seconds": [float(t[0]), float(t[-1])],
        "n_frames": int(len(t)),
        "phases": {k: list(v) for k, v in cfg["phases"].items()},
        # [Index B disabled] "bands_hz": [list(b) for b in cfg["bands"]],
        "index_b_frequency_pca": "disabled in this build",
        "frame_rate_hz": bundle["frame_rate"],
        "pad_frames_per_end": int(bundle["pad"]),
        "n_rois": len(bundle["roi_ids"]),
        "active_thresh": float(bundle["active_thresh"]),
        "active_thresh_provenance": bundle["thresh_provenance"],
        "active_n_sigma": cfg["n_sigma"],
        "pca_var_target": cfg["var_target"],
        "pca_n_components_rate": int(bundle["rate"]["n_pc"]),
        # [Index B disabled] "pca_n_components_freq": int(bundle["freq"]["n_pc"]),
        "max_k": cfg["max_k"], "min_group_size": cfg["min_group"],
        "max_group_frac": cfg["max_frac"],
        "cut_rate": {"requested_k": cfg.get("k_rate"), "height_override": cfg.get("cut_rate"),
                     "height_used": float(cut_rate), "n_groups": int(rate_groups.max())},
        # [Index B disabled]
        # "cut_freq": {"requested_k": cfg.get("k_freq"), "height_override": cfg.get("cut_freq"),
        #              "height_used": float(cut_freq), "n_groups": int(freq_groups.max())},
        # "adjusted_rand_index": float(ari),
        "spike_inference_model": cfg["model_name"],
        "clustering": "z-score -> PCA -> Ward linkage -> fcluster",
    }
    path = Path(out_dir) / "analysis_params.json"
    path.write_text(json.dumps(params, indent=2))
    print(f"> Saved analysis_params.json -> {path}")
    return path


def _export_figures(bundle, cfg):
    """Draw and save the six figures, on the UI thread.

    `analysis_tools.plots` builds bare `Figure` objects, so nothing here touches pyplot
    and there is no global figure registry to close against -- the figures are released
    when this function returns.
    """
    out_dir = cfg["out_dir"]
    b = bundle
    label = (f"CASCADE | {cfg['model_name']} | "
             f"ACTIVE_THRESH={b['active_thresh']:.3f} spikes/s")
    paths = []

    def save(fig, name):
        paths.append(at.store.save_figure(fig, out_dir, name))

    save(at.plots.pca_summary(b["pca_rate"], b["scores_rate"], b["rate_features"],
                              b["roi_ids"], "PCA-A: per-phase inferred-rate features",
                              cfg["var_target"], groups=b["rate_groups"]),
         "pca_rate_summary")
    save(at.plots.dendrogram_plot(b["z_rate"], b["roi_ids"], "Index A (inferred rate)",
                                  b["cut_rate"]), "dendrogram_rate")
    save(at.plots.rate_heatmap(b["spike_rate"], b["t_plot"], b["roi_ids"], b["rate_groups"],
                               b["stim"], f"Inferred spike rate by Index A\n{label}"),
         "rate_heatmap")
    save(at.plots.group_means(b["spike_rate"], b["t_plot"], b["rate_groups"], b["roi_ids"],
                              b["stim"], b["pad"],
                              f"Group-mean inferred rate ± SEM\n{label}"),
         "rate_group_means")
    # [Index B disabled] its two figures; the export drops from six to four
    # save(at.plots.pca_summary(b["pca_freq"], b["scores_freq"], b["freq_features"],
    #                           b["roi_ids"], "PCA-B: per-phase frequency features",
    #                           cfg["var_target"], groups=b["freq_groups"]),
    #      "pca_freq_summary")
    # save(at.plots.dendrogram_plot(b["z_freq"], b["roi_ids"], "Index B (frequency)",
    #                               b["cut_freq"]), "dendrogram_freq")
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

