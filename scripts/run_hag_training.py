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
from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import (
    greedy_hag_grouping,
)


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
    # ============================================================
    # 1) Load config + dataset
    # ============================================================
    cfg = load_default_config("configs/default.yaml")
    ds = load_dataset_bundle(cfg.dataset)

    # ============================================================
    # 2) Prep merged contributions + merged weights (aligned to original indices)
    # ============================================================
    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        quantitative_idx=getattr(ds, "quantitative_idx", None),
        nominal_idx=getattr(ds, "nominal_idx", None),
    )

    # ============================================================
    # 3) Run Algorithm-1 (HAG greedy grouping)
    # IMPORTANT: X must already be η-contribution-values dataset
    # ============================================================
    res = greedy_hag_grouping(
        X=prep.X_contrib_full,
        y=ds.y,
        weights=prep.w_full,
        params=cfg.hag,   # NEW: use domain.params.HAGParams from YAML (includes majorizing.name+params)
    )

    # ============================================================
    # 4) Prepare run folder
    # outputs/runs/train/<run_id>/
    # ============================================================
    run_name = "train"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path(cfg.output.root_dir) / cfg.output.runs_dir / run_name / run_id)

    # ============================================================
    # 5) Reproducibility snapshot (same policy)
    # ============================================================
    dataset_path_txt = run_dir / "dataset_path.txt"
    dataset_cfg_json = run_dir / "dataset_config.json"
    run_cfg_json = run_dir / "run_config.json"

    dataset_path_txt.write_text(str(cfg.dataset.path), encoding="utf-8")
    dataset_cfg_json.write_text(
        json.dumps(_jsonable_dataset_config(cfg.dataset), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # NEW: write HAG config including majorizing.name + majorizing.params
    run_cfg_payload: Dict[str, Any] = {
        "seed": int(cfg.seed),
        "hag": {
            "alpha": float(cfg.hag.alpha),
            "delta": float(cfg.hag.delta),
            "kappa": int(cfg.hag.kappa),
            "cr1": float(cfg.hag.cr1),
            "k1_label": int(cfg.hag.k1_label),
            "k2_label": int(cfg.hag.k2_label),
            "majorizing": {
                "name": str(cfg.hag.majorizing.name),
                "params": dict(cfg.hag.majorizing.params or {}),
            },
        },
    }
    run_cfg_json.write_text(json.dumps(run_cfg_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ============================================================
    # 6) Artifacts: tuplam.json + dij.csv
    # ============================================================
    tuplam_path = run_dir / "tuplam.json"
    dij_path = run_dir / "dij.csv"

    tuplam_payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir.as_posix()),
        "tuplam": res.tuplam,                 # 0-based feature indices (selection order)
        "size": len(res.tuplam),
        "p_latent": res.p,                    # p = |TUPLAM| - 1
        "crit_history": res.crit_history,     # crit per Step-4 iteration
        "files": {
            "dataset_path_txt": str(dataset_path_txt.as_posix()),
            "dataset_config_json": str(dataset_cfg_json.as_posix()),
            "run_config_json": str(run_cfg_json.as_posix()),
            "dij_csv": str(dij_path.as_posix()),
        },
    }
    tuplam_path.write_text(json.dumps(tuplam_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # dij.csv (latent feature matrix), columns: r1..rp
    if res.dij.size == 0:
        dij_path.write_text("r1\n", encoding="utf-8")
    else:
        header = ",".join([f"r{i+1}" for i in range(res.dij.shape[1])])
        _save_csv_float(dij_path, res.dij, header=header)

    # ============================================================
    # 7) Console summary
    # ============================================================
    print("HAG train demo completed")
    print("Run folder:", run_dir)
    print("TUPLAM JSON:", tuplam_path)
    print("dij CSV:", dij_path)
    print("TUPLAM:", res.tuplam)
    print("|TUPLAM|:", len(res.tuplam), " => p =", res.p)
    print("dij shape:", res.dij.shape)
    print("majorizing:", cfg.hag.majorizing.name, "params:", cfg.hag.majorizing.params)


if __name__ == "__main__":
    main()
