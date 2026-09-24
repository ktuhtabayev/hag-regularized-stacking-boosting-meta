from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle

# STABLE FACADE import (your project rule)
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import build_nominal_contributions


def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv_float(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(float), delimiter=",", header=header, comments="", fmt="%.9f")


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def main() -> None:
    cfg = load_default_config("configs/default.yaml")
    ds = load_dataset_bundle(cfg.dataset)

    nominal_idx = np.asarray(getattr(ds, "nominal_idx", []), dtype=int)

    if nominal_idx.size == 0:
        print("============================================================")
        print("NOMINAL DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Nominal demo skipped.")
        print("Reason: no nominal features found (feature_types = 0).")
        print("Check your dataset feature-sign row:")
        print("  - nominal feature       -> 0")
        print("  - quantitative feature  -> 1")
        print("No output files were created for nominal_demo.")
        return

    res = build_nominal_contributions(ds.X, ds.y, ds.nominal_idx)

    run_name = "nominal_demo"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path("outputs") / "runs" / run_name / run_id)

    dataset_path_txt = run_dir / "dataset_path.txt"
    dataset_cfg_json = run_dir / "dataset_config.json"

    dataset_path_txt.write_text(str(cfg.dataset.path), encoding="utf-8")

    dataset_cfg_payload: Dict[str, Any] = {}
    for k, v in vars(cfg.dataset).items():
        dataset_cfg_payload[k] = str(v) if isinstance(v, Path) else v

    dataset_cfg_json.write_text(
        json.dumps(dataset_cfg_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    n_headers = ",".join([f"nf_{idx}" for idx in res.nominal_idx])

    contrib_path = run_dir / "nominal_contrib.csv"
    table_path = run_dir / "lambda_beta_weight_table.json"

    _save_csv_float(contrib_path, res.contribution_X, header=n_headers)

    # Canonical names (with fallbacks just in case)
    weight_wc = getattr(res, "weight_wc", getattr(res, "w_c", None))
    D1_c = getattr(res, "D1_c", getattr(res, "d1_c", None))
    D2_c = getattr(res, "D2_c", getattr(res, "d2_c", None))
    contributions_eta = getattr(res, "contributions_eta", getattr(res, "eta_c", None))

    payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "name": getattr(ds, "name", None),
            "path": cfg.dataset.path,
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "nominal_idx": res.nominal_idx,
            "quantitative_idx": getattr(ds, "quantitative_idx", None),
        },
        "hag_defaults": {
            "alpha": cfg.hag.alpha,
            "delta": cfg.hag.delta,
            "kappa": cfg.hag.kappa,
            "majorizing_function": getattr(cfg.hag, "majorizing_function", None),
        },
        "nominal_tables": _to_jsonable(
            {
                "lambda_c": getattr(res, "lambda_c", None),
                "beta_c": getattr(res, "beta_c", None),
                "weight_wc": weight_wc,
                "p_c": getattr(res, "p_c", None),
                "l1_c": getattr(res, "l1_c", None),
                "l2_c": getattr(res, "l2_c", None),
                "D1_c": D1_c,
                "D2_c": D2_c,
                "gradations": getattr(res, "gradations", None),
                "contributions_eta": contributions_eta,
            }
        ),
        "files": {
            "dataset_path_txt": str(dataset_path_txt.as_posix()),
            "dataset_config_json": str(dataset_cfg_json.as_posix()),
            "nominal_contrib_csv": str(contrib_path.as_posix()),
            "lambda_beta_weight_table_json": str(table_path.as_posix()),
        },
    }

    table_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Nominal demo completed")
    print("Run folder:", run_dir)
    print("Contrib CSV:", contrib_path)
    print("Lambda/Beta/Weight JSON:", table_path)
    print("Nominal idx:", res.nominal_idx)
    print("Contrib shape:", res.contribution_X.shape)


if __name__ == "__main__":
    main()