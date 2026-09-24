from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.algorithms.hag.prep import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import greedy_hag_grouping
from hag_regularized_stacking_boosting_meta.algorithms.meta.prep import prepare_meta_training_dataset


def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv_meta(path: Path, headers: list[str], S: np.ndarray, y: np.ndarray) -> None:
    """
    Save META training dataset as:
      ai0..aip, di1..dip, Class
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1, 1)
    out = np.hstack([S, y.astype(float)])

    header = ",".join(headers)
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.9f")


def _jsonable_dataset_config(cfg_dataset: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for k, v in vars(cfg_dataset).items():
        payload[k] = str(v) if isinstance(v, Path) else v
    return payload


def main() -> None:
    # 1) Load config + dataset
    cfg = load_default_config("configs/default.yaml")
    ds = load_dataset_bundle(cfg.dataset)

    # 2) Run HAG prep -> contributions + weights (for organizer only)
    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        quantitative_idx=getattr(ds, "quantitative_idx", None),
        nominal_idx=getattr(ds, "nominal_idx", None),
    )

    # 3) Run HAG (Algorithm-1)
    hag_res = greedy_hag_grouping(
        X=prep.X_contrib_full,
        y=ds.y,
        weights=prep.w_full,
        params=cfg.hag,
    )

    # 4) Build META training dataset using:
    #    - original ds.X for ai columns
    #    - ds.feature_types to choose binary for quantitative
    #    - hag_res.tuplam as ordered SET features
    #    - hag_res.dij as latent features
    meta = prepare_meta_training_dataset(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        tuplam=hag_res.tuplam,
        dij=hag_res.dij,
    )

    # 5) Save outputs
    run_name = "meta_prep"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path("outputs") / "runs" / run_name / run_id)

    dataset_path_txt = run_dir / "dataset_path.txt"
    dataset_cfg_json = run_dir / "dataset_config.json"
    run_cfg_json = run_dir / "run_config.json"

    dataset_path_txt.write_text(str(cfg.dataset.path), encoding="utf-8")
    dataset_cfg_json.write_text(
        json.dumps(_jsonable_dataset_config(cfg.dataset), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_cfg_payload: Dict[str, Any] = {
        "seed": int(cfg.seed),
        "hag": {
            "alpha": float(cfg.hag.alpha),
            "delta": float(cfg.hag.delta),
            "kappa": int(cfg.hag.kappa),
            "cr1": float(cfg.hag.cr1),
            "k1_label": int(cfg.hag.k1_label),
            "k2_label": int(cfg.hag.k2_label),
            "organizer_index": getattr(cfg.hag, "organizer_index", None),
            "majorizing": {
                "name": str(cfg.hag.majorizing.name),
                "params": dict(cfg.hag.majorizing.params or {}),
            },
        },
    }
    run_cfg_json.write_text(json.dumps(run_cfg_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Artifacts
    tuplam_path = run_dir / "tuplam.json"
    dij_path = run_dir / "dij.csv"
    meta_train_path = run_dir / "meta_train.csv"

    # Save tuplam.json
    tuplam_payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir.as_posix()),
        "tuplam": hag_res.tuplam,              # 0-based ordered
        "size": len(hag_res.tuplam),
        "p_latent": hag_res.p,
        "crit_history": hag_res.crit_history,
    }
    tuplam_path.write_text(json.dumps(tuplam_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save dij.csv
    if hag_res.dij.size == 0:
        dij_path.write_text("r1\n", encoding="utf-8")
    else:
        header = ",".join([f"r{i+1}" for i in range(hag_res.dij.shape[1])])
        np.savetxt(dij_path, hag_res.dij.astype(float), delimiter=",", header=header, comments="", fmt="%.9f")

    # Save meta_train.csv
    _save_csv_meta(meta_train_path, meta.headers, meta.S, meta.y)

    # Console summary
    print("META preparation completed")
    print("Run folder:", run_dir)
    print("TUPLAM JSON:", tuplam_path)
    print("dij CSV:", dij_path)
    print("META train CSV:", meta_train_path)
    print("TUPLAM (0-based):", hag_res.tuplam)
    print("META matrix shape (S):", meta.S.shape, " + Class =>", (meta.S.shape[0], meta.S.shape[1] + 1))


if __name__ == "__main__":
    main()