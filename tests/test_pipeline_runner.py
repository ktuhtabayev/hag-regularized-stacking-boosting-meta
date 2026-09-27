"""services.runner: the GUI pipeline reproduces the Excel experiment and the scripts' results."""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import greedy_hag_grouping
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.algorithms.meta.training_set import (
    prepare_meta_training_dataset,
)
from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.io.writers import write_pipeline_outputs
from hag_regularized_stacking_boosting_meta.services import report_builder as rb
from hag_regularized_stacking_boosting_meta.services.runner import (
    classify_new_object,
    random_new_object,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


def _config(**hag_overrides):
    cfg = load_default_config(CONFIG)
    return replace(cfg, hag=replace(cfg.hag, **hag_overrides)) if hag_overrides else cfg


@pytest.fixture(scope="module")
def default_result():
    return run_pipeline(_config(), project_root=ROOT)


def test_default_run_reproduces_the_excel_experiment(default_result) -> None:
    r = default_result
    criterion1 = rb.criterion1_frame(r).set_index("Feature")
    assert criterion1.loc["x0", ["π1 (min)", "π2 (split)", "π3 (max)"]].tolist() == [56, 63, 74]
    assert criterion1.loc["x0", "Weight ω"] == pytest.approx(0.634921, abs=1e-6)

    # Excel: organizer x3 (index 2), first winner x6 (index 5) with cr1 = 0.355663879
    assert r.hag.organizer == 2
    assert r.hag.tuplam[:2] == [2, 5]
    assert r.hag.crit_history[0] == pytest.approx(0.355663879, abs=1e-9)

    # Excel latent table r1: max over K2 = -0.149427417, min over K1 = 0.23923915
    assert r.margins[0].left_boundary == pytest.approx(-0.149427417, abs=1e-9)
    assert r.margins[0].right_boundary == pytest.approx(0.23923915, abs=1e-8)


@pytest.mark.parametrize(
    ("dataset", "organizer"),
    [
        ("datasets/raw/Cancer/Cancer (589, 44, 2).dat", 0),
        ("datasets/raw/Heart-Disease/Heart-Disease (270, 13, 2).csv", 12),
    ],
)
def test_default_organizer_is_the_top_weight_feature_of_the_chosen_dataset(
    dataset: str, organizer: int
) -> None:
    cfg = load_default_config(CONFIG)
    r = run_pipeline(replace(cfg, dataset=replace(cfg.dataset, path=dataset)), project_root=ROOT)
    assert r.hag.organizer == organizer == int(np.argmax(r.prep.w_full))
    assert r.hag.tuplam[0] == organizer


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_diverging_majorizer_fails_with_a_clear_error() -> None:
    majorizing = replace(_config().hag.majorizing, name="exponential", params={"k": 1.0, "x0": 0.0, "scale": 1.0})
    with pytest.raises(ValueError, match=r"HAG diverged at step r4"):
        run_pipeline(_config(majorizing=majorizing), project_root=ROOT)


@pytest.mark.parametrize(
    "dataset",
    ["datasets/raw/default.dat", "datasets/raw/Heart-Disease/Heart-Disease (270, 13, 2).csv"],
)
def test_service_matches_the_script_call_sequence(dataset: str) -> None:
    cfg = load_default_config(CONFIG)
    cfg = replace(cfg, dataset=replace(cfg.dataset, path=dataset))
    service = run_pipeline(cfg, project_root=ROOT)

    ds = load_dataset_bundle(replace(cfg.dataset, path=str(ROOT / dataset)))
    with contextlib.redirect_stdout(io.StringIO()):
        prep = prepare_hag_inputs(
            X=ds.X,
            y=ds.y,
            feature_types=ds.feature_types,
            quantitative_idx=ds.quantitative_idx,
            nominal_idx=ds.nominal_idx,
        )
        hag = greedy_hag_grouping(X=prep.X_contrib_full, y=ds.y, weights=prep.w_full, params=cfg.hag)
    meta = prepare_meta_training_dataset(
        X=ds.X, y=ds.y, feature_types=ds.feature_types, tuplam=hag.tuplam, dij=hag.dij
    )

    np.testing.assert_array_equal(service.prep.X_contrib_full, prep.X_contrib_full)
    np.testing.assert_array_equal(service.prep.w_full, prep.w_full)
    assert service.hag.tuplam == hag.tuplam
    assert service.hag.crit_history == hag.crit_history
    np.testing.assert_array_equal(service.hag.dij, hag.dij)
    np.testing.assert_array_equal(service.meta.S, meta.S)
    assert service.meta.headers == meta.headers


def test_step3_scan_winner_is_the_minimum_ratio(default_result) -> None:
    r = default_result
    for t, scan in enumerate(r.hag.candidate_history):
        winner = r.hag.tuplam[t + 1]
        assert scan[winner] == min(scan.values())
        assert scan[winner] == r.hag.crit_history[t]
        assert set(scan).isdisjoint(r.hag.tuplam[: t + 1])


def test_kappa_5_stops_at_five_features_like_the_excel_experiment() -> None:
    r = run_pipeline(_config(kappa=5), project_root=ROOT)
    assert r.hag.tuplam == [2, 5, 12, 3, 8]
    assert r.hag.p == 4
    assert rb.hag_iterations_frame(r)["Next"].iloc[-1] == "stop: |TUPLAM| = κ = 5"


def test_iterations_report_when_candidates_run_out(default_result) -> None:
    # default.csv has 13 features < κ = 15, so HAG ends when the pool is empty
    frame = rb.hag_iterations_frame(default_result)
    assert frame["Next"].iloc[-1] == "stop: no candidates left"
    assert (frame["Next"].iloc[1:-1] == "continue").all()


def test_training_objects_classify_as_their_own_class(default_result) -> None:
    r = default_result
    for i in range(r.dataset.X.shape[0]):
        snew = classify_new_object(r, [r.dataset.X[i, f] for f in r.hag.tuplam])
        assert snew.prediction.predicted_label == r.dataset.y[i]


def test_random_new_object_is_reproducible_and_binarized_by_gamma(default_result) -> None:
    r = default_result
    first = random_new_object(r, 123)
    second = random_new_object(r, 123)
    np.testing.assert_array_equal(first.a_init, second.a_init)
    assert first.source == "random (seed=123)"

    for value, binary, fidx in zip(first.a_init, first.a_bin, r.hag.tuplam):
        if r.dataset.feature_types[fidx] == 1:
            assert binary == (1 if value <= r.gamma_map[fidx] else 2)
        else:
            assert binary == value


def test_organizer_auto_uses_the_highest_weight() -> None:
    r = run_pipeline(_config(organizer_index=None), project_root=ROOT)
    assert r.hag.organizer == r.prep.weight_sorted_feature_idx[0]
    assert r.hag.tuplam[0] == r.hag.organizer


def test_write_pipeline_outputs(tmp_path: Path, default_result) -> None:
    run_dir = write_pipeline_outputs(default_result, tmp_path)

    assert run_dir == tmp_path / "gui" / default_result.run_id
    for relative in (
        "run_info.json",
        "quantitative/criterion1.csv",
        "nominal/weights.csv",
        "hag/iterations.csv",
        "hag/step3_scan.csv",
        "hag/dij.csv",
        "meta/meta_train.csv",
        "meta/filtering.csv",
        "margin/margin_report.csv",
        f"margin/latent_r{default_result.hag.p}_object_margins.csv",
    ):
        assert (run_dir / relative).is_file(), relative

    header = (run_dir / "meta" / "meta_train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["Object"] + default_result.meta.headers
