from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs


def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv_float(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(float), delimiter=",", header=header, comments="", fmt="%.9f")


def _jsonable_dataset_config(cfg_dataset: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for k, v in vars(cfg_dataset).items():
        payload[k] = str(v) if isinstance(v, Path) else v
    return payload


def main() -> None:
    cfg = load_default_config("configs/default.yaml")
    ds = load_dataset_bundle(cfg.dataset)

    quantitative_idx = np.asarray(getattr(ds, "quantitative_idx", []), dtype=int)
    nominal_idx = np.asarray(getattr(ds, "nominal_idx", []), dtype=int)

    # ------------------------------------------------------------
    # Graceful messages instead of hard failure when one stage is absent
    # ------------------------------------------------------------
    if quantitative_idx.size == 0:
        print("============================================================")
        print("HAG PREP DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Note: no quantitative features found (feature_types = 1).")
        print("Quantitative stage will be skipped in the merge/prep step.")

    if nominal_idx.size == 0:
        print("============================================================")
        print("HAG PREP DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Note: no nominal features found (feature_types = 0).")
        print("Nominal stage will be skipped in the merge/prep step.")

    if quantitative_idx.size == 0 and nominal_idx.size == 0:
        print("============================================================")
        print("HAG PREP DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("HAG prep skipped.")
        print("Reason: no quantitative and no nominal features were found.")
        print("Check your dataset feature-sign row:")
        print("  - quantitative feature  -> 1")
        print("  - nominal feature       -> 0")
        print("No output files were created for run_hag_prep.py.")
        return

    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,  # 0/1 signs row (0=nominal,1=quant)
        quantitative_idx=getattr(ds, "quantitative_idx", None),
        nominal_idx=getattr(ds, "nominal_idx", None),
    )

    # outputs/runs/hag_prep/<run_id>/
    run_name = "hag_prep"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path("outputs") / "runs" / run_name / run_id)

    # reproducibility snapshot (same policy as other demos)
    dataset_path_txt = run_dir / "dataset_path.txt"
    dataset_cfg_json = run_dir / "dataset_config.json"
    dataset_path_txt.write_text(str(cfg.dataset.path), encoding="utf-8")
    dataset_cfg_json.write_text(
        json.dumps(_jsonable_dataset_config(cfg.dataset), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # requested artifacts
    merged_contrib_path = run_dir / "merged_contrib.csv"
    weights_path = run_dir / "weights.json"
    weight_rank_path = run_dir / "weight_rank.json"

    n = prep.X_contrib_full.shape[1]
    header = ",".join([f"f_{i}" for i in range(n)])  # f_0 aligns to x1, f_1 to x2, ...
    _save_csv_float(merged_contrib_path, prep.X_contrib_full, header=header)

    # weights.json
    weights_payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "name": getattr(ds, "name", None),
            "path": str(cfg.dataset.path),
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "quantitative_idx": getattr(ds, "quantitative_idx", []),
            "nominal_idx": getattr(ds, "nominal_idx", []),
        },
        "weights": {
            "w_full": prep.w_full.tolist(),  # length n; aligned to original feature positions
            "by_feature_index": {str(i): float(prep.w_full[i]) for i in range(n)},
        },
        "files": {
            "dataset_path_txt": str(dataset_path_txt.as_posix()),
            "dataset_config_json": str(dataset_cfg_json.as_posix()),
            "merged_contrib_csv": str(merged_contrib_path.as_posix()),
            "weights_json": str(weights_path.as_posix()),
            "weight_rank_json": str(weight_rank_path.as_posix()),
        },
    }
    weights_path.write_text(json.dumps(weights_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # weight_rank.json (robust to naming differences)
    sorted_idx = getattr(prep, "weight_sorted_feature_idx", getattr(prep, "weight_sorted_idx"))
    rank = getattr(prep, "weight_rank_per_feature", getattr(prep, "weight_rank_per_feature_index"))

    sorted_table = [
        {"rank": int(pos), "feature_index": int(fi), "weight": float(prep.w_full[fi])}
        for pos, fi in enumerate(sorted_idx)
    ]

    weight_rank_payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "ranking_rule": "primary: weight desc; tie: smaller feature_index first (left-to-right)",
        "sorted_feature_indices": sorted_idx,
        "rank_per_feature_index": rank,
        "sorted_table": sorted_table,  # easiest to compare with Excel
        "top_20": sorted_table[:20],
    }
    weight_rank_path.write_text(json.dumps(weight_rank_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console print (clear + matches Excel logic)
    print("HAG prep demo completed")
    print("Run folder:", run_dir)
    print("Merged contrib CSV:", merged_contrib_path)
    print("Weights JSON:", weights_path)
    print("Weight rank JSON:", weight_rank_path)
    print("Merged contrib shape:", prep.X_contrib_full.shape)
    print("Weights length:", prep.w_full.shape[0])

    print("\n(A) FEATURE INDICES sorted by weight DESC:")
    print("Top-10 sorted feature indices:", sorted_idx[:10])

    print("\n(B) RANK per FEATURE INDEX (Excel 'rank row', but 0-based):")
    show_n = min(13, len(rank))
    print(f"Rank-per-feature (0..{show_n-1}):", rank[:show_n])

    print("\nTop-10 table (rank, feature_index, weight):")
    for row in sorted_table[:10]:
        print(row)


if __name__ == "__main__":
    main()