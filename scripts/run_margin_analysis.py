from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.evaluation.margin_analysis import (
    margin_analysis_latent_matrix,
    build_margin_report_rows,
)


def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Margin analysis demo on latent features (dij).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--train-run", type=str, default=None, help="Optional explicit train run folder.")
    return p.parse_args()


def _load_dij_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)


def main() -> None:
    args = _parse_args()
    cfg = load_default_config(args.config)

    train_root = Path(cfg.output.root_dir) / cfg.output.runs_dir / "train"

    # 1) Locate train run (dij.csv)
    if args.train_run:
        train_run_dir = Path(args.train_run)
    else:
        train_run_dir = _latest_run_dir(train_root)

    if train_run_dir is None or not train_run_dir.exists():
        raise SystemExit(f"No train runs found at: {train_root}. Run: python scripts/run_hag_training.py")

    dij_path = train_run_dir / "dij.csv"
    if not dij_path.exists():
        raise SystemExit(f"Missing dij.csv in train run: {train_run_dir}")

    # 2) Load dataset to get y
    ds = load_dataset_bundle(cfg.dataset)
    y = ds.y

    # 3) Load dij and compute margins per latent feature
    dij = _load_dij_csv(dij_path)
    if dij.ndim == 1:
        dij = dij.reshape(-1, 1)

    if dij.shape[0] != y.shape[0]:
        raise SystemExit(f"dij rows ({dij.shape[0]}) != y length ({y.shape[0]}).")

    results = margin_analysis_latent_matrix(
        dij=dij,
        y=y,
        k1_label=int(cfg.hag.k1_label),
        k2_label=int(cfg.hag.k2_label),
    )

    # 4) Prepare run folder
    run_name = "margin_analysis"
    run_id = _make_run_id()
    run_dir = _ensure_dir(Path(cfg.output.root_dir) / cfg.output.runs_dir / run_name / run_id)

    # 5) Save per-feature scalar report
    report_rows = build_margin_report_rows(results)
    report_json = run_dir / "margin_report.json"
    report_csv = run_dir / "margin_report.csv"

    report_json.write_text(json.dumps(report_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    header = "latent_index_1based,left_boundary,right_boundary,midpoint,margin_width,left_argmax_idx_0based,right_argmin_idx_0based"
    lines = [header]
    for r in report_rows:
        lines.append(
            f"{int(r['latent_index_1based'])},"
            f"{r['left_boundary']:.9f},"
            f"{r['right_boundary']:.9f},"
            f"{r['midpoint']:.9f},"
            f"{r['margin_width']:.9f},"
            f"{int(r['left_argmax_idx_0based'])},"
            f"{int(r['right_argmin_idx_0based'])}"
        )
    report_csv.write_text("\n".join(lines), encoding="utf-8")

    # 6) Save per-object tables per latent feature
    for j, r in enumerate(results, start=1):
        out = np.column_stack(
            [
                np.arange(y.shape[0], dtype=int),
                dij[:, j - 1],
                y.astype(int),
                r.object_margin.astype(float),
                r.yhat.astype(int),
            ]
        )
        out_path = run_dir / f"latent_r{j}_object_margins.csv"
        np.savetxt(
            out_path,
            out,
            delimiter=",",
            header="Sid,d,Class,ObjectMargin,yhat",
            comments="",
            fmt=["%d", "%.9f", "%d", "%.9f", "%d"],
        )

    print("✅ Margin analysis completed")
    print("Train run:", train_run_dir)
    print("dij.csv:", dij_path)
    print("Run folder:", run_dir)
    print("margin_report.csv:", report_csv)
    print("margin_report.json:", report_json)
    print("Saved per-latent-feature object tables:", dij.shape[1])


if __name__ == "__main__":
    main()