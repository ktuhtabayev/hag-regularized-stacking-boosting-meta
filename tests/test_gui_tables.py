from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QHeaderView  # noqa: E402

from hag_regularized_stacking_boosting_meta.gui.app import (  # noqa: E402
    ORGANIZER_AUTO,
    MainWindow,
    frame_to_table,
    parse_majorizer_params,
)
from hag_regularized_stacking_boosting_meta.gui.plots import (  # noqa: E402
    draw_criterion_history,
    draw_margin_strip,
    draw_margin_widths,
)
from hag_regularized_stacking_boosting_meta.gui.theme import THEME, build_stylesheet  # noqa: E402
from hag_regularized_stacking_boosting_meta.io.configs import load_default_config  # noqa: E402
from hag_regularized_stacking_boosting_meta.services.runner import run_pipeline  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def default_result():
    return run_pipeline(load_default_config(ROOT / "configs" / "default.yaml"), project_root=ROOT)


def test_gui_tables_do_not_stretch_last_column_and_remain_resizable(app) -> None:
    table = frame_to_table(
        pd.DataFrame(
            {
                "Object": ["S0", "S1"],
                "x0": [0.1234567, 1.0],
                "Class": [1, 2],
            }
        )
    )
    header = table.horizontalHeader()
    model = table.model()

    assert header.stretchLastSection() is False
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(model.columnCount() - 1) == QHeaderView.ResizeMode.Interactive


def test_gui_table_preserves_source_order_and_sorts_object_names_naturally(app) -> None:
    source_order = ["S1", "S2", "S10", "S3"]
    table = frame_to_table(pd.DataFrame({"Object": source_order, "Class": [1, 1, 2, 1]}))
    model = table.model()

    assert table.horizontalHeader().sortIndicatorSection() == -1
    assert model.frame["Object"].tolist() == source_order

    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert model.frame["Object"].tolist() == ["S1", "S2", "S3", "S10"]

    table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert model.frame["Object"].tolist() == ["S10", "S3", "S2", "S1"]

    model.sort(-1)
    assert model.frame["Object"].tolist() == source_order


def test_gui_table_values_are_center_aligned_and_floats_trimmed(app) -> None:
    table = frame_to_table(pd.DataFrame({"Feature": ["x0", "x3"], "Weight ω": [0.5, 2.0]}))
    model = table.model()

    assert model.data(model.index(0, 1)) == "0.5"
    assert model.data(model.index(1, 1)) == "2"
    alignment = model.data(model.index(0, 0), Qt.ItemDataRole.TextAlignmentRole)
    assert int(alignment) == int(Qt.AlignmentFlag.AlignCenter)


def test_gui_stylesheet_is_built_from_theme_colors() -> None:
    stylesheet = build_stylesheet()
    assert THEME["window_bg"] in stylesheet
    assert THEME["accent"] in stylesheet
    assert f"color: {THEME['text']}" in stylesheet
    assert "#primaryButton" in stylesheet
    assert "QDoubleSpinBox" in stylesheet


def test_parameter_controls_initialize_from_config(app) -> None:
    window = MainWindow(ROOT, restore_settings=False)

    assert window.alpha.value() == pytest.approx(0.3)
    assert window.delta.value() == pytest.approx(0.1)
    assert window.kappa.value() == 15
    assert window.organizer.value() == ORGANIZER_AUTO
    assert window.majorizer.currentText() == "sigmoid"
    assert parse_majorizer_params(window.majorizer_params.text()) == {"k": 1.0, "x0": 0.0}
    assert window.dataset_preset.currentText() == "default_csv"


def test_controls_build_the_run_config(app) -> None:
    window = MainWindow(ROOT, restore_settings=False)
    window.alpha.setValue(0.25)
    window.kappa.setValue(5)
    window.organizer.setValue(ORGANIZER_AUTO)
    window.majorizer.setCurrentText("quadratic")
    window._on_majorizer_selected(window.majorizer.currentIndex())

    cfg = window._config_from_controls()

    assert window.organizer.text() == "Auto (max ω)"
    assert cfg.hag.alpha == pytest.approx(0.25)
    assert cfg.hag.kappa == 5
    assert cfg.hag.organizer_index is None
    assert cfg.hag.majorizing.name == "quadratic"
    assert cfg.hag.majorizing.params == {"a": 1.0, "b": 0.0, "c": 0.0}
    assert cfg.dataset.path == str(Path("datasets") / "raw" / "default.csv")


def test_invalid_majorizer_params_are_rejected(app) -> None:
    window = MainWindow(ROOT, restore_settings=False)
    window.majorizer.setCurrentText("piecewise_linear")
    window.majorizer_params.setText("x: [0, 1], y: [0]")

    with pytest.raises(ValueError, match="Invalid majorizing function"):
        window._config_from_controls()
    with pytest.raises(ValueError):
        parse_majorizer_params("k 1.0")


def test_dataset_preset_dropdown_fills_path_and_tracks_custom_edits(app) -> None:
    window = MainWindow(ROOT, restore_settings=False)
    preset_names = [window.dataset_preset.itemText(i) for i in range(window.dataset_preset.count())]
    assert preset_names[0] == "Custom..."
    assert "heart_disease_270_csv" in preset_names

    # a forced organizer index belongs to the previous dataset
    window.organizer.setValue(2)
    row = preset_names.index("cancer_nominal_dat")
    window.dataset_preset.setCurrentIndex(row)
    window._on_dataset_preset_selected(row)
    assert window.dataset_path.text().endswith("Cancer-N (589, 44, 2).dat")
    assert window.organizer.value() == ORGANIZER_AUTO

    window.organizer.setValue(2)
    window._on_dataset_preset_selected(row)  # same dataset again keeps the choice
    assert window.organizer.value() == 2
    window._on_dataset_path_edited("")
    assert window.organizer.value() == ORGANIZER_AUTO

    window.dataset_path.setText(str(ROOT / "datasets" / "nonexistent.csv"))
    window._sync_preset_to_path()
    assert window.dataset_preset.currentText() == "Custom..."
    with pytest.raises(ValueError, match="not found"):
        window._config_from_controls()


def test_result_tabs_and_new_object_classification(app, default_result) -> None:
    window = MainWindow(ROOT, restore_settings=False)
    window.populate_tabs(default_result)

    tab_names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert tab_names == [
        "Dataset",
        "Quantitative Weights",
        "Nominal Weights",
        "HAG Prep",
        "HAG Training",
        "META Training Set",
        "META Classification",
        "Margin Analysis",
    ]
    hag_tabs = window.tabs.widget(tab_names.index("HAG Training"))
    assert [hag_tabs.tabText(i) for i in range(hag_tabs.count())] == [
        "Iterations",
        "Step 3 Scan θ/γ",
        "Latent Features dij",
        "Criterion Plot",
    ]
    assert "criterion" in window.figures and "margin" in window.figures

    # Type the initial values of training object S1 (class 1) and classify them
    result = default_result
    for column, fidx in enumerate(result.hag.tuplam):
        window.snew_table.item(2, column).setText(str(result.dataset.X[1, fidx]))
    window.classify_edited_new_object()
    assert "Class 1 (K1)" in window.prediction_label.text()
    assert result.new_object.source == "edited"

    window.seed_spin.setValue(7)
    window.generate_random_new_object()
    assert "seed=7" in window.prediction_label.text()


def test_margin_animation_steps_through_latent_features(app, default_result) -> None:
    window = MainWindow(ROOT, restore_settings=False)
    window.populate_tabs(default_result)

    assert window.margin_step_label.text() == f"r1 / r{default_result.hag.p}"
    window.toggle_margin_animation()
    assert window.margin_timer.isActive()
    window._advance_margin_animation()
    assert window.margin_slider.value() == 1
    assert window.margin_step_label.text().startswith("r2 /")
    window.toggle_margin_animation()
    assert not window.margin_timer.isActive()


def test_plots_draw_for_every_latent_feature(default_result) -> None:
    figure = Figure()
    draw_criterion_history(figure.add_subplot(111), default_result)
    for j in range(default_result.hag.p):
        figure.clear()
        strip, widths = figure.subplots(2, 1)
        draw_margin_strip(strip, default_result, j)
        draw_margin_widths(widths, default_result, j)
        assert strip.get_title().startswith(f"r{j + 1}:")
