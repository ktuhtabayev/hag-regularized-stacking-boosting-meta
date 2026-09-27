from __future__ import annotations

import re
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from hag_regularized_stacking_boosting_meta.algorithms.hag.majorizing_functions import (
    get_majorizing_function,
)
from hag_regularized_stacking_boosting_meta.domain.params import MajorizingConfig
from hag_regularized_stacking_boosting_meta.gui.plots import (
    draw_criterion_history,
    draw_margin_strip,
    draw_margin_widths,
)
from hag_regularized_stacking_boosting_meta.gui.theme import THEME, build_stylesheet
from hag_regularized_stacking_boosting_meta.io.configs import (
    RunConfig,
    load_dataset_catalog,
    load_default_config,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_pipeline_outputs
from hag_regularized_stacking_boosting_meta.services import report_builder as rb
from hag_regularized_stacking_boosting_meta.services.runner import (
    NewObjectClassification,
    PipelineResult,
    classify_new_object,
    random_new_object,
    run_pipeline,
)


try:
    from PyQt6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QSettings,
        QThread,
        QTimer,
        QUrl,
        Qt,
        pyqtSignal,
    )
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTableView,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on local GUI install
    raise RuntimeError("PyQt6 is required to launch the desktop GUI.") from exc


# Default parameters per majorizing function (the snippets documented in configs/default.yaml)
MAJORIZER_PRESETS: Dict[str, Dict[str, Any]] = {
    "identity": {},
    "sigmoid": {"k": 1.0, "x0": 0.0},
    "logistic": {"k": 2.0, "x0": 0.0},
    "exponential": {"k": 1.0, "x0": 0.0, "scale": 1.0},
    "quadratic": {"a": 1.0, "b": 0.0, "c": 0.0},
    "piecewise_linear": {"x": [-10.0, -1.0, 0.0, 1.0, 10.0], "y": [0.0, 0.2, 0.5, 0.8, 1.0]},
}

MAX_SET_IDS_SHOWN = 40
RESIZE_SAMPLE_ROWS = 100
MARGIN_ANIMATION_MS = 900
ORGANIZER_AUTO = -1


def format_majorizer_params(params: Dict[str, Any]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in params.items())


def parse_majorizer_params(text: str) -> Dict[str, Any]:
    """'k: 1.0, x0: 0.0' -> {'k': 1.0, 'x0': 0.0} (YAML flow mapping without braces)."""
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = yaml.safe_load("{" + text + "}")
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Majorizing parameters must look like 'k: 1.0, x0: 0.0' (got: {text})"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Majorizing parameters must be name: value pairs (got: {text})")
    missing = [str(key) for key, value in parsed.items() if value is None]
    if missing:
        raise ValueError(
            f"Majorizing parameter without a value: {', '.join(missing)} (use 'name: value')"
        )
    return {str(key): value for key, value in parsed.items()}


def _display_value(value: object, decimals: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return str(value)


def _natural_sort_key(value: object) -> tuple[object, ...]:
    if pd.isna(value):
        return (2, ())
    if isinstance(value, Real):
        return (0, float(value))

    parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
        if part
    )
    return (1, parts)


class DataFrameTableModel(QAbstractTableModel):
    """Expose a DataFrame lazily so large research tables remain responsive."""

    def __init__(self, frame: pd.DataFrame, *, decimals: int = 6) -> None:
        super().__init__()
        self._source_frame = frame.reset_index(drop=True).copy()
        self.frame = self._source_frame.copy()
        self.decimals = decimals

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(
                self.frame.iat[index.row(), index.column()],
                self.decimals,
            )
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if column == -1:
            sorted_frame = self._source_frame.copy()
        elif 0 <= column < self.columnCount():
            values = self.frame.iloc[:, column].tolist()
            positions = sorted(
                range(len(values)),
                key=lambda index: _natural_sort_key(values[index]),
                reverse=order == Qt.SortOrder.DescendingOrder,
            )
            sorted_frame = self.frame.iloc[positions].reset_index(drop=True)
        else:
            return

        self.layoutAboutToBeChanged.emit()
        self.frame = sorted_frame
        self.layoutChanged.emit()


def frame_to_table(frame: pd.DataFrame, *, decimals: int = 6) -> QTableView:
    model = DataFrameTableModel(frame, decimals=decimals)
    table = QTableView()
    table.setModel(model)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setSortingEnabled(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    horizontal = table.horizontalHeader()
    horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    horizontal.setStretchLastSection(False)
    # Size columns from the header + first rows instead of measuring every cell
    horizontal.setResizeContentsPrecision(RESIZE_SAMPLE_ROWS)
    vertical = table.verticalHeader()
    vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vertical.setDefaultSectionSize(26)
    if model.rowCount() * model.columnCount() <= 50_000:
        table.resizeColumnsToContents()
    else:
        for col_idx, column in enumerate(model.frame.columns):
            table.setColumnWidth(col_idx, min(180, max(72, len(str(column)) * 9 + 24)))
    horizontal.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    table.setSortingEnabled(True)
    return table


def _tab_group(tabs: Dict[str, QWidget]) -> QTabWidget:
    group = QTabWidget()
    group.setDocumentMode(True)
    for title, widget in tabs.items():
        group.addTab(widget, title)
    return group


def _message_widget(message: str) -> QWidget:
    text = QTextEdit()
    text.setReadOnly(True)
    text.setPlainText(message)
    widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(text)
    widget.setLayout(layout)
    return widget


class PipelineWorker(QThread):
    """Run the full pipeline off the GUI thread so the window stays responsive."""

    stage_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # (PipelineResult, output folder)
    failed = pyqtSignal(str)

    def __init__(self, config: RunConfig, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._project_root = project_root

    def run(self) -> None:  # pragma: no cover - thread entry point
        try:
            result = run_pipeline(
                self._config,
                project_root=self._project_root,
                on_stage=self.stage_changed.emit,
            )
            self.stage_changed.emit("Writing outputs")
            output_dir = write_pipeline_outputs(result, runs_root(self._config, self._project_root))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit((result, output_dir))


def runs_root(config: RunConfig, project_root: Path) -> Path:
    return Path(project_root) / config.output.root_dir / config.output.runs_dir


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, restore_settings: bool = True) -> None:
        super().__init__()
        self.project_root = project_root
        self.config_path = project_root / "configs" / "default.yaml"
        self.last_result: PipelineResult | None = None
        self.last_output_dir: Path | None = None
        self.pipeline_worker: PipelineWorker | None = None
        self.figures: dict[str, Figure] = {}
        self.restore_settings = restore_settings
        self.settings = QSettings("hag-regularized-stacking-boosting-meta", "DesktopApp")
        self.dataset_catalog = load_dataset_catalog(self.config_path)

        # Widgets of the current result tabs (rebuilt by populate_tabs)
        self.snew_table: QTableWidget | None = None
        self.snew_results: QTabWidget | None = None
        self.snew_results_layout: QVBoxLayout | None = None
        self.prediction_label: QLabel | None = None
        self.seed_spin: QSpinBox | None = None
        self.margin_slider: QSlider | None = None
        self.margin_step_label: QLabel | None = None
        self.margin_play_button: QPushButton | None = None
        self.margin_canvas: FigureCanvasQTAgg | None = None
        self.margin_timer = QTimer(self)
        self.margin_timer.setInterval(MARGIN_ANIMATION_MS)
        self.margin_timer.timeout.connect(self._advance_margin_animation)

        self.setWindowTitle("HAG Regularized Stacking Boosting META")
        self.resize(1440, 880)

        # ---- Row 1: dataset + actions ----
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Dataset path from config")
        self.dataset_path.textEdited.connect(self._on_dataset_path_edited)

        self.dataset_preset = QComboBox()
        self.dataset_preset.addItem("Custom...")
        for preset_name in self.dataset_catalog:
            self.dataset_preset.addItem(preset_name)
        self.dataset_preset.activated.connect(self._on_dataset_preset_selected)

        browse_button = QPushButton("Dataset...")
        browse_button.clicked.connect(self.choose_dataset)

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_current_pipeline)

        export_button = QPushButton("Export")
        export_button.clicked.connect(self.export_last_result)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # busy indicator
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setVisible(False)

        # ---- Row 2: HAG parameters ----
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 10.0)
        self.alpha.setDecimals(4)
        self.alpha.setSingleStep(0.05)
        self.alpha.setToolTip("α: regularization strength in R + sign(y)·α·f(−R)")

        self.delta = QDoubleSpinBox()
        self.delta.setRange(0.0, 10.0)
        self.delta.setDecimals(4)
        self.delta.setSingleStep(0.01)
        self.delta.setToolTip("δ: HAG stops when the best θ/γ ≤ δ")

        self.kappa = QSpinBox()
        self.kappa.setRange(1, 999)
        self.kappa.setToolTip("κ: HAG stops when |TUPLAM| reaches κ")

        self.organizer = QSpinBox()
        self.organizer.setRange(ORGANIZER_AUTO, 9999)
        self.organizer.setSpecialValueText("Auto (max ω)")
        self.organizer.setPrefix("x")
        self.organizer.setToolTip(
            "Organizer u (0-based feature index). Auto = feature with the highest weight ω."
        )

        self.majorizer = QComboBox()
        self.majorizer.addItems(list(MAJORIZER_PRESETS))
        self.majorizer.activated.connect(self._on_majorizer_selected)
        self.majorizer.setToolTip("Majorizing function f used by the regularizer")

        self.majorizer_params = QLineEdit()
        self.majorizer_params.setPlaceholderText("k: 1.0, x0: 0.0")
        self.majorizer_params.setToolTip("Parameters of f as name: value pairs")

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(140)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset"))
        controls.addWidget(self.dataset_preset)
        controls.addWidget(self.dataset_path, 1)
        controls.addWidget(browse_button)
        controls.addWidget(self.run_button)
        controls.addWidget(export_button)
        controls.addWidget(self.open_output_button)
        controls.addWidget(self.progress_bar)

        params = QHBoxLayout()
        for label, widget in (
            ("α", self.alpha),
            ("δ", self.delta),
            ("κ", self.kappa),
            ("Organizer", self.organizer),
            ("Majorizer f", self.majorizer),
        ):
            params.addWidget(QLabel(label))
            params.addWidget(widget)
        params.addWidget(self.majorizer_params, 1)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(params)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_text)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.load_defaults()
        if self.restore_settings:
            self._restore_saved_state()

    # ------------------------------------------------------------------
    # Defaults, settings, dataset selection
    # ------------------------------------------------------------------

    def load_defaults(self) -> None:
        cfg = load_default_config(self.config_path)
        self.dataset_path.setText(str(self.project_root / cfg.dataset.path))
        self.alpha.setValue(float(cfg.hag.alpha))
        self.delta.setValue(float(cfg.hag.delta))
        self.kappa.setValue(int(cfg.hag.kappa))
        organizer = cfg.hag.organizer_index
        self.organizer.setValue(ORGANIZER_AUTO if organizer is None else int(organizer))
        name = str(cfg.hag.majorizing.name)
        if self.majorizer.findText(name) < 0:
            self.majorizer.addItem(name)
        self.majorizer.setCurrentText(name)
        self.majorizer_params.setText(format_majorizer_params(dict(cfg.hag.majorizing.params or {})))
        self._sync_preset_to_path()

    def _restore_saved_state(self) -> None:
        # Keep dataset and HAG parameters at their configured defaults.
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.pipeline_worker is not None and self.pipeline_worker.isRunning():
            QMessageBox.information(self, "Run in progress", "Wait for the current run to finish.")
            event.ignore()
            return
        self.margin_timer.stop()
        if self.restore_settings:
            self._save_state()
        super().closeEvent(event)

    def _on_dataset_preset_selected(self, index: int) -> None:
        preset_name = self.dataset_preset.itemText(index)
        entry = self.dataset_catalog.get(preset_name)
        if entry is None:
            return
        self.dataset_path.setText(str(self.project_root / str(entry["path"])))

    def _on_dataset_path_edited(self, _text: str) -> None:
        self._sync_preset_to_path()

    def _sync_preset_to_path(self) -> None:
        """Show the matching preset name for the current path, else Custom."""
        current = self.dataset_path.text().strip()
        for row, (_preset_name, entry) in enumerate(self.dataset_catalog.items(), start=1):
            if current == str(self.project_root / str(entry["path"])):
                self.dataset_preset.setCurrentIndex(row)
                return
        self.dataset_preset.setCurrentIndex(0)

    def _on_majorizer_selected(self, index: int) -> None:
        preset = MAJORIZER_PRESETS.get(self.majorizer.itemText(index), {})
        self.majorizer_params.setText(format_majorizer_params(preset))

    def choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose dataset",
            str(self.project_root / "datasets"),
            "Datasets (*.csv *.dat);;All files (*.*)",
        )
        if path:
            self.dataset_path.setText(str(Path(path)))
            self._sync_preset_to_path()

    def _config_from_controls(self) -> RunConfig:
        cfg = load_default_config(self.config_path)
        dataset = cfg.dataset
        dataset_text = self.dataset_path.text().strip()
        if dataset_text:
            path = Path(dataset_text)
            if not path.exists():
                raise ValueError(f"Dataset file not found: {path}")
            try:
                path = path.resolve().relative_to(self.project_root.resolve())
            except ValueError:
                pass
            # format stays as configured ("auto" detects csv/dat from the extension)
            dataset = replace(dataset, path=str(path))

        majorizing = MajorizingConfig(
            name=self.majorizer.currentText(),
            params=parse_majorizer_params(self.majorizer_params.text()),
        )
        try:
            probe = get_majorizing_function(majorizing)(np.array([-1.0, 0.0, 1.0]))
        except Exception as exc:
            raise ValueError(f"Invalid majorizing function: {exc}") from exc
        if not np.all(np.isfinite(probe)):
            raise ValueError("Majorizing function returns non-finite values for x in [-1, 1].")

        organizer = self.organizer.value()
        hag = replace(
            cfg.hag,
            alpha=float(self.alpha.value()),
            delta=float(self.delta.value()),
            kappa=int(self.kappa.value()),
            organizer_index=None if organizer == ORGANIZER_AUTO else int(organizer),
            majorizing=majorizing,
        )
        return replace(cfg, dataset=dataset, hag=hag)

    # ------------------------------------------------------------------
    # Run / export
    # ------------------------------------------------------------------

    def run_current_pipeline(self) -> None:
        if self.pipeline_worker is not None and self.pipeline_worker.isRunning():
            return
        try:
            cfg = self._config_from_controls()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            self.status_text.setPlainText(str(exc))
            return

        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_text.setPlainText("Running HAG → META pipeline...")

        self.pipeline_worker = PipelineWorker(cfg, self.project_root, parent=self)
        self.pipeline_worker.stage_changed.connect(self._on_stage_changed)
        self.pipeline_worker.finished_ok.connect(self._on_pipeline_finished)
        self.pipeline_worker.failed.connect(self._on_pipeline_failed)
        self.pipeline_worker.finished.connect(self._on_pipeline_done)
        self.pipeline_worker.start()

    def _on_stage_changed(self, stage: str) -> None:
        self.status_text.setPlainText(f"Running: {stage}...")

    def _on_pipeline_finished(self, payload: tuple[PipelineResult, Path]) -> None:
        result, output_dir = payload
        self.last_result = result
        self.last_output_dir = output_dir
        self.populate_tabs(result)
        self.open_output_button.setEnabled(True)
        self.status_text.setPlainText(self._status_summary(result, output_dir))

    def _on_pipeline_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Run failed", message)
        self.status_text.setPlainText(message)

    def _on_pipeline_done(self) -> None:
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def export_last_result(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "Nothing to export", "Run the pipeline first.")
            return
        try:
            self.last_output_dir = write_pipeline_outputs(
                self.last_result,
                runs_root(self.last_result.config, self.project_root),
            )
            self.open_output_button.setEnabled(True)
            self.status_text.setPlainText(self._status_summary(self.last_result, self.last_output_dir))
            QMessageBox.information(self, "Export complete", str(self.last_output_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def open_output_folder(self) -> None:
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Run or export first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))

    def save_figure(self, key: str, file_name: str) -> None:
        figure = self.figures.get(key)
        if figure is None:
            QMessageBox.information(self, "No plot", "This plot is not available.")
            return
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Run or export first.")
            return
        target_dir = self.last_output_dir / "figures"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        figure.savefig(target_path, dpi=160, bbox_inches="tight")
        QMessageBox.information(self, "Plot saved", str(target_path))

    # ------------------------------------------------------------------
    # Result tabs
    # ------------------------------------------------------------------

    def populate_tabs(self, result: PipelineResult) -> None:
        self.margin_timer.stop()
        self.last_result = result
        self.tabs.clear()
        self.figures = {}

        self.tabs.addTab(
            _tab_group(
                {
                    "Objects": frame_to_table(rb.dataset_frame(result)),
                    "Features": frame_to_table(rb.feature_summary_frame(result)),
                }
            ),
            "Dataset",
        )
        self.tabs.addTab(self._quantitative_tab(result), "Quantitative Weights")
        self.tabs.addTab(self._nominal_tab(result), "Nominal Weights")
        self.tabs.addTab(
            _tab_group(
                {
                    "Merged Contributions η": frame_to_table(rb.merged_contrib_frame(result)),
                    "Weight Ranking": frame_to_table(rb.weight_ranking_frame(result)),
                }
            ),
            "HAG Prep",
        )
        self.tabs.addTab(self._hag_training_tab(result), "HAG Training")
        self.tabs.addTab(frame_to_table(rb.meta_training_frame(result)), "META Training Set")
        self.tabs.addTab(self._classification_tab(result), "META Classification")
        self.tabs.addTab(self._margin_tab(result), "Margin Analysis")

    def _quantitative_tab(self, result: PipelineResult) -> QWidget:
        if result.quantitative is None:
            return _message_widget("This dataset has no quantitative features (feature-type flag 1).")
        return _tab_group(
            {
                "Criterion-1": frame_to_table(rb.criterion1_frame(result)),
                "Binary {1,2}": frame_to_table(rb.quantitative_binary_frame(result)),
                "Contributions η": frame_to_table(rb.quantitative_contrib_frame(result)),
            }
        )

    def _nominal_tab(self, result: PipelineResult) -> QWidget:
        if result.nominal is None:
            return _message_widget("This dataset has no nominal features (feature-type flag 0).")
        return _tab_group(
            {
                "λ, β, ω": frame_to_table(rb.nominal_weights_frame(result)),
                "Gradations": frame_to_table(rb.nominal_gradations_frame(result)),
                "Contributions η": frame_to_table(rb.nominal_contrib_frame(result)),
            }
        )

    def _hag_training_tab(self, result: PipelineResult) -> QWidget:
        save_button = QPushButton("Save Criterion Plot PNG")
        save_button.clicked.connect(lambda: self.save_figure("criterion", "hag_criterion.png"))

        figure = Figure(figsize=(8, 4.6), facecolor=THEME["figure_bg"])
        self.figures["criterion"] = figure
        draw_criterion_history(figure.add_subplot(111), result)
        figure.tight_layout()
        canvas = FigureCanvasQTAgg(figure)

        plot = QWidget()
        plot_layout = QVBoxLayout()
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(save_button)
        plot_controls.addStretch(1)
        plot_layout.addLayout(plot_controls)
        plot_layout.addWidget(canvas, 1)
        plot.setLayout(plot_layout)
        canvas.draw_idle()

        return _tab_group(
            {
                "Iterations": frame_to_table(rb.hag_iterations_frame(result)),
                "Step 3 Scan θ/γ": frame_to_table(rb.step3_scan_frame(result)),
                "Latent Features dij": frame_to_table(rb.latent_frame(result)),
                "Criterion Plot": plot,
            }
        )

    # ---- META classification of a new object ----

    def _classification_tab(self, result: PipelineResult) -> QWidget:
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(int(result.config.seed))
        self.seed_spin.setToolTip("Seed for a reproducible random Snew")

        random_button = QPushButton("Random Snew")
        random_button.clicked.connect(self.generate_random_new_object)

        classify_button = QPushButton("Classify")
        classify_button.setObjectName("primaryButton")
        classify_button.setToolTip("Classify the initial values in the table with META")
        classify_button.clicked.connect(self.classify_edited_new_object)

        self.prediction_label = QLabel()
        self.prediction_label.setObjectName("predictionLabel")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Seed"))
        controls.addWidget(self.seed_spin)
        controls.addWidget(random_button)
        controls.addWidget(classify_button)
        controls.addStretch(1)
        controls.addWidget(self.prediction_label)

        self.snew_table = QTableWidget(4, len(result.hag.tuplam))
        self.snew_table.setVerticalHeaderLabels(["Type", "Γc", "Initial value", "Binary value"])
        self.snew_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.snew_table.verticalHeader().setDefaultSectionSize(26)
        self.snew_table.setFixedHeight(4 * 26 + 60)

        self.snew_results_layout = QVBoxLayout()
        self.snew_results = None

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.snew_table)
        layout.addLayout(self.snew_results_layout, 1)
        widget.setLayout(layout)
        self._show_new_object(result.new_object)
        return widget

    def _show_new_object(self, snew: NewObjectClassification) -> None:
        result = self.last_result
        assert result is not None
        result.new_object = snew
        table = self.snew_table
        assert table is not None and self.snew_results_layout is not None

        table.setHorizontalHeaderLabels(snew.headers)
        snew_frame = rb.new_object_frame(result, snew)
        for column, row in snew_frame.iterrows():
            cells = [
                (row["Type"], False),
                (_display_value(row["Γc"], 6), False),
                (_display_value(row["Initial value"], 6), True),
                (_display_value(row["Binary value"], 6), False),
            ]
            for row_index, (text, editable) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if editable:
                    flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                table.setItem(row_index, int(column), item)
        table.resizeColumnsToContents()

        if self.snew_results is not None:
            self.snew_results_layout.removeWidget(self.snew_results)
            self.snew_results.deleteLater()
        self.snew_results = _tab_group(
            {
                "Filtering B1/B2": frame_to_table(
                    rb.filtering_frame(result, snew, max_ids=MAX_SET_IDS_SHOWN)
                ),
                "Decision": frame_to_table(rb.decision_frame(result, snew)),
            }
        )
        self.snew_results_layout.addWidget(self.snew_results)

        label = rb.class_name(snew.prediction.predicted_label, result)
        if self.prediction_label is not None:
            self.prediction_label.setText(f"Snew ∈ {label}   ·   {snew.source}")

    def _read_snew_initial_values(self) -> list[float]:
        assert self.snew_table is not None
        values = []
        for column in range(self.snew_table.columnCount()):
            header = self.snew_table.horizontalHeaderItem(column).text()
            item = self.snew_table.item(2, column)
            text = item.text().strip().replace(",", ".") if item is not None else ""
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(f"Initial value of {header} must be a number (got '{text}').") from exc
        return values

    def classify_edited_new_object(self) -> None:
        if self.last_result is None:
            return
        try:
            values = self._read_snew_initial_values()
            snew = classify_new_object(self.last_result, values)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot classify Snew", str(exc))
            return
        self._show_new_object(snew)
        self._refresh_status()

    def generate_random_new_object(self) -> None:
        if self.last_result is None or self.seed_spin is None:
            return
        self._show_new_object(random_new_object(self.last_result, self.seed_spin.value()))
        self._refresh_status()

    # ---- Margin analysis ----

    def _margin_tab(self, result: PipelineResult) -> QWidget:
        if not result.margins:
            return _message_widget(
                "No latent features (p = 0): HAG selected a single feature, so there is "
                "no margin to analyse."
            )

        p = len(result.margins)
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, p - 1)
        self.margin_slider.setMaximumWidth(360)
        self.margin_slider.valueChanged.connect(self.refresh_margin_plot)

        self.margin_step_label = QLabel()
        self.margin_play_button = QPushButton("Play")
        self.margin_play_button.setToolTip("Step through r1..rp to watch the margin evolve")
        self.margin_play_button.clicked.connect(self.toggle_margin_animation)

        save_button = QPushButton("Save Margin Plot PNG")
        save_button.clicked.connect(self._save_margin_figure)

        figure = Figure(figsize=(8, 6.2), facecolor=THEME["figure_bg"])
        self.figures["margin"] = figure
        self.margin_canvas = FigureCanvasQTAgg(figure)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Latent feature"))
        controls.addWidget(self.margin_slider)
        controls.addWidget(self.margin_step_label)
        controls.addWidget(self.margin_play_button)
        controls.addWidget(save_button)
        controls.addStretch(1)

        plot = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addLayout(controls)
        plot_layout.addWidget(self.margin_canvas, 1)
        plot.setLayout(plot_layout)
        self.refresh_margin_plot()

        object_tabs = {
            f"r{j + 1}": frame_to_table(rb.object_margins_frame(result, j)) for j in range(p)
        }
        return _tab_group(
            {
                "Report": frame_to_table(rb.margin_report_frame(result)),
                "Object Margins": _tab_group(object_tabs),
                "Margin Plot": plot,
            }
        )

    def refresh_margin_plot(self) -> None:
        figure = self.figures.get("margin")
        result = self.last_result
        if figure is None or result is None or self.margin_slider is None:
            return
        index = self.margin_slider.value()
        if self.margin_step_label is not None:
            self.margin_step_label.setText(f"r{index + 1} / r{len(result.margins)}")
        figure.clear()
        grid = figure.add_gridspec(2, 1, height_ratios=[3, 2])
        draw_margin_strip(figure.add_subplot(grid[0]), result, index)
        draw_margin_widths(figure.add_subplot(grid[1]), result, index)
        figure.tight_layout()
        if self.margin_canvas is not None:
            self.margin_canvas.draw_idle()

    def toggle_margin_animation(self) -> None:
        if self.margin_slider is None or self.margin_play_button is None:
            return
        if self.margin_timer.isActive():
            self.margin_timer.stop()
            self.margin_play_button.setText("Play")
            return
        if self.margin_slider.value() >= self.margin_slider.maximum():
            self.margin_slider.setValue(0)
        self.margin_play_button.setText("Pause")
        self.margin_timer.start()

    def _advance_margin_animation(self) -> None:
        if self.margin_slider is None:
            self.margin_timer.stop()
            return
        if self.margin_slider.value() >= self.margin_slider.maximum():
            self.margin_timer.stop()
            if self.margin_play_button is not None:
                self.margin_play_button.setText("Play")
            return
        self.margin_slider.setValue(self.margin_slider.value() + 1)

    def _save_margin_figure(self) -> None:
        index = self.margin_slider.value() if self.margin_slider is not None else 0
        self.save_figure("margin", f"margin_r{index + 1}.png")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        if self.last_result is not None:
            self.status_text.setPlainText(self._status_summary(self.last_result, self.last_output_dir))

    def _status_summary(self, result: PipelineResult, output_dir: Path | None) -> str:
        ds = result.dataset
        params = result.config.hag
        snew = result.new_object
        k1 = int(np.sum(ds.y == result.k1_label))
        k2 = int(np.sum(ds.y == result.k2_label))
        organizer = "auto (max ω)" if params.organizer_index is None else f"x{params.organizer_index}"
        tuplam = ", ".join(f"x{i}" for i in result.hag.tuplam)
        output_text = str(output_dir) if output_dir is not None else "Not exported"
        return (
            f"Run: {result.run_id}   ({result.elapsed_seconds:.3f} s)\n"
            f"Dataset: {result.dataset_path}\n"
            f"Objects: {ds.X.shape[0]} (|K1| = {k1}, |K2| = {k2})   Features: {ds.X.shape[1]} "
            f"({len(ds.quantitative_idx)} quantitative, {len(ds.nominal_idx)} nominal)\n"
            f"HAG: α = {params.alpha:g}, δ = {params.delta:g}, κ = {params.kappa}, "
            f"organizer = {organizer} → u = x{result.hag.organizer}, "
            f"f = {params.majorizing.name}({format_majorizer_params(dict(params.majorizing.params or {}))})\n"
            f"TUPLAM: [{tuplam}]   (p = {result.hag.p} latent features)\n"
            f"Snew [{snew.source}] ∈ {rb.class_name(snew.prediction.predicted_label, result)}   "
            f"score1 = {snew.prediction.decision.score1:.4f}, score2 = {snew.prediction.decision.score2:.4f}\n"
            f"Output folder: {output_text}"
        )


def main() -> None:  # pragma: no cover - GUI entrypoint
    project_root = Path(__file__).resolve().parents[3]
    app = QApplication([])
    app.setOrganizationName("hag-regularized-stacking-boosting-meta")
    app.setApplicationName("DesktopApp")
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    window = MainWindow(project_root)
    window.show()
    app.exec()
