from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hag_regularized_stacking_boosting_meta.cli import (
    latest_run_dir,
    new_stage_run,
    parse_stage_args,
    save_csv,
    stage_root,
)
from hag_regularized_stacking_boosting_meta.evaluation.margin_analysis import (
    margin_analysis_latent_matrix,
    build_margin_report_rows,
)
from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.io.writers import write_json, write_text


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-run", type=str, default=None, help="Optional explicit train run folder.")


def _load_dij_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)


def main() -> None:
    args = parse_stage_args("Margin analysis demo on latent features (dij).", _add_arguments)
    cfg = load_default_config(args.config)

    train_root = stage_root(cfg, "train")

    # 1) Locate train run (dij.csv)
    train_run_dir = Path(args.train_run) if args.train_run else latest_run_dir(train_root)

    if train_run_dir is None or not train_run_dir.exists():
        raise SystemExit(f"No train runs found at: {train_root}. Run: python scripts/run_hag_training.py")

    dij_path = train_run_dir / "dij.csv"
    if not dij_path.exists():
        raise SystemExit(f"Missing dij.csv in train run: {train_run_dir}")

    # 2) Load dataset to get y
    y = load_dataset_bundle(cfg.dataset).y

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
    run_dir = new_stage_run(cfg, "margin_analysis").run_dir

    # 5) Save per-feature scalar report
    report_rows = build_margin_report_rows(results)
    report_json = run_dir / "margin_report.json"
    report_csv = run_dir / "margin_report.csv"

    write_json(report_json, report_rows)

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
    write_text(report_csv, "\n".join(lines))

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
        save_csv(
            run_dir / f"latent_r{j}_object_margins.csv",
            out,
            "Sid,d,Class,ObjectMargin,yhat",
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
