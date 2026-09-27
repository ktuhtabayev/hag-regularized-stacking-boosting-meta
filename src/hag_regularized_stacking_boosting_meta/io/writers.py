from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hag_regularized_stacking_boosting_meta.services.runner import PipelineResult


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_to_jsonable)
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv_matrix(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray | None = None,
    delimiter: str = ",",
    header: list[str] | None = None,
) -> Path:
    """
    Writes numeric matrix to CSV.
    If y is provided, appends y as last column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y).reshape(-1, 1)
        out = np.concatenate([X, y], axis=1)
    else:
        out = X

    with path.open("w", encoding="utf-8", newline="") as f:
        if header:
            f.write(delimiter.join(header) + "\n")
        for row in out:
            f.write(delimiter.join(str(v) for v in row) + "\n")
    return path


GUI_RUNS_TASK = "gui"


def write_pipeline_outputs(result: "PipelineResult", runs_root: str | Path) -> Path:
    """
    Write every table of one GUI run to <runs_root>/gui/<run_id>/.
    Re-writing the same run (Export after classifying another Snew) updates it in place.
    """
    from hag_regularized_stacking_boosting_meta.services import report_builder as rb
    from hag_regularized_stacking_boosting_meta.utils.run_manager import create_run_dir

    run_dir = create_run_dir(GUI_RUNS_TASK, runs_root, run_id=result.run_id).run_dir

    def table(relative: str, frame) -> None:
        if frame is None or frame.empty:
            return
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")

    write_json(run_dir / "run_info.json", rb.run_summary(result))
    write_text(run_dir / "dataset_path.txt", str(result.dataset_path))
    write_json(run_dir / "dataset_config.json", result.config.dataset)

    table("dataset/dataset.csv", rb.dataset_frame(result))
    table("dataset/features.csv", rb.feature_summary_frame(result))

    table("quantitative/criterion1.csv", rb.criterion1_frame(result))
    table("quantitative/binary.csv", rb.quantitative_binary_frame(result))
    table("quantitative/contributions.csv", rb.quantitative_contrib_frame(result))

    table("nominal/weights.csv", rb.nominal_weights_frame(result))
    table("nominal/gradations.csv", rb.nominal_gradations_frame(result))
    table("nominal/contributions.csv", rb.nominal_contrib_frame(result))

    table("hag/merged_contributions.csv", rb.merged_contrib_frame(result))
    table("hag/weight_rank.csv", rb.weight_ranking_frame(result))
    table("hag/iterations.csv", rb.hag_iterations_frame(result))
    table("hag/step3_scan.csv", rb.step3_scan_frame(result))
    table("hag/dij.csv", rb.latent_frame(result))

    snew = result.new_object
    table("meta/meta_train.csv", rb.meta_training_frame(result))
    table("meta/snew.csv", rb.new_object_frame(result, snew))
    table("meta/filtering.csv", rb.filtering_frame(result, snew))
    table("meta/decision.csv", rb.decision_frame(result, snew))

    table("margin/margin_report.csv", rb.margin_report_frame(result))
    for j in range(len(result.margins)):
        table(f"margin/latent_r{j + 1}_object_margins.csv", rb.object_margins_frame(result, j))

    return run_dir

