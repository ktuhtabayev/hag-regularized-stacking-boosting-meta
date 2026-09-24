from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import build_quantitative_nominalization


def _make_run_id() -> str:
    # Example: 20260205_142240_0f347ac5
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv_int(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(int), delimiter=",", header=header, comments="", fmt="%d")


def _save_csv_float(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(float), delimiter=",", header=header, comments="", fmt="%.9f")


def main() -> None:
    # 1) Load YAML -> DatasetConfig (+ HAG defaults in cfg.hag)
    cfg = load_default_config("configs/default.yaml")

    # 2) Load dataset
    ds = load_dataset_bundle(cfg.dataset)

    quantitative_idx = np.asarray(getattr(ds, "quantitative_idx", []), dtype=int)

    if quantitative_idx.size == 0:
        print("============================================================")
        print("QUANTITATIVE DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Quantitative demo skipped.")
        print("Reason: no quantitative features found (feature_types = 1).")
        print("Check your dataset feature-sign row:")
        print("  - quantitative feature  -> 1")
        print("  - nominal feature       -> 0")
        print("No output files were created for quantitative_demo.")
        return

    # 3) Run quantitative pipeline
    res = build_quantitative_nominalization(ds.X, ds.y, ds.quantitative_idx)

    # 4) Prepare run folder EXACTLY as requested:
    # outputs/runs/quantitative_demo/<run_id>/
    run_name = "quantitative_demo"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path("outputs") / "runs" / run_name / run_id)

    # 5) Save dataset path + dataset config (GUI-friendly + reproducible)
    dataset_path_txt = run_dir / "dataset_path.txt"
    dataset_cfg_json = run_dir / "dataset_config.json"

    dataset_path_txt.write_text(str(cfg.dataset.path), encoding="utf-8")

    # Make DatasetConfig JSON-friendly
    dataset_cfg_payload: Dict[str, Any] = {}
    for k, v in vars(cfg.dataset).items():
        # Path -> str, everything else ok
        dataset_cfg_payload[k] = str(v) if isinstance(v, Path) else v

    dataset_cfg_json.write_text(
        json.dumps(dataset_cfg_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 6) Create column headers (keep original feature indices!)
    q_headers = ",".join([f"qf_{idx}" for idx in res.quantitative_idx])

    # 7) Save outputs
    binary_path = run_dir / "quantitative_binary.csv"
    contrib_path = run_dir / "quantitative_contrib.csv"
    crit_path = run_dir / "criterion1_table.json"

    _save_csv_int(binary_path, res.binary_X, header=q_headers)
    _save_csv_float(contrib_path, res.contribution_X, header=q_headers)

    # 8) Save Criterion-1 table + Γc + η contributions as JSON
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
            "quantitative_idx": res.quantitative_idx,
            "nominal_idx": getattr(ds, "nominal_idx", None),
        },
        "hag_defaults": {
            "alpha": cfg.hag.alpha,
            "delta": cfg.hag.delta,
            "kappa": cfg.hag.kappa,
            "majorizing_function": getattr(cfg.hag, "majorizing_function", None),
        },
        "criterion1": {
            # feature_idx -> {pi1,pi2,pi3,wc,gamma,eta}
            str(fidx): {
                "pi1": res.pi_table[fidx][0],
                "pi2": res.pi_table[fidx][1],
                "pi3": res.pi_table[fidx][2],
                "wc": res.weights_wc[fidx],
                "gamma": res.gamma[fidx],
                "eta": res.contributions_eta[fidx],  # {1:...,2:...}
            }
            for fidx in res.quantitative_idx
        },
        "files": {
            "dataset_path_txt": str(dataset_path_txt.as_posix()),
            "dataset_config_json": str(dataset_cfg_json.as_posix()),
            "quantitative_binary_csv": str(binary_path.as_posix()),
            "quantitative_contrib_csv": str(contrib_path.as_posix()),
            "criterion1_table_json": str(crit_path.as_posix()),
        },
    }

    crit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 9) Print summary (CMD-friendly)
    print("Quantitative demo completed")
    print("Run folder:", run_dir)
    print("Binary CSV:", binary_path)
    print("Contrib CSV:", contrib_path)
    print("Criterion JSON:", crit_path)
    print("Quantitative idx:", res.quantitative_idx)
    print("Binary shape:", res.binary_X.shape)
    print("Contrib shape:", res.contribution_X.shape)


if __name__ == "__main__":
    main()