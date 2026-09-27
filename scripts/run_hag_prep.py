from __future__ import annotations

from typing import Any, Dict

from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.cli import (
    dataset_name,
    load_stage_inputs,
    new_stage_run,
    parse_stage_args,
    save_csv,
    write_dataset_snapshot,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json


def _print_banner(dataset_path: str) -> None:
    print("============================================================")
    print("HAG PREP DEMO")
    print("============================================================")
    print(f"Dataset path: {dataset_path}")


def main() -> None:
    args = parse_stage_args("HAG preparation: merged contribution table and weight ranking.")
    cfg, ds = load_stage_inputs(args.config)

    # ------------------------------------------------------------
    # Graceful messages instead of hard failure when one stage is absent
    # ------------------------------------------------------------
    if not ds.quantitative_idx:
        _print_banner(cfg.dataset.path)
        print("Note: no quantitative features found (feature_types = 1).")
        print("Quantitative stage will be skipped in the merge/prep step.")

    if not ds.nominal_idx:
        _print_banner(cfg.dataset.path)
        print("Note: no nominal features found (feature_types = 0).")
        print("Nominal stage will be skipped in the merge/prep step.")

    if not ds.quantitative_idx and not ds.nominal_idx:
        _print_banner(cfg.dataset.path)
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
        quantitative_idx=ds.quantitative_idx,
        nominal_idx=ds.nominal_idx,
    )

    # outputs/runs/hag_prep/<run_id>/ + reproducibility snapshot (same policy as other stages)
    run = new_stage_run(cfg, "hag_prep")
    run_dir = run.run_dir
    dataset_path_txt, dataset_cfg_json = write_dataset_snapshot(run_dir, cfg)

    # requested artifacts
    merged_contrib_path = run_dir / "merged_contrib.csv"
    weights_path = run_dir / "weights.json"
    weight_rank_path = run_dir / "weight_rank.json"

    n = prep.X_contrib_full.shape[1]
    header = ",".join(f"f_{i}" for i in range(n))  # f_0 aligns to x1, f_1 to x2, ...
    save_csv(merged_contrib_path, prep.X_contrib_full.astype(float), header)

    # weights.json
    weights_payload: Dict[str, Any] = {
        "run_name": run.task,
        "run_id": run.run_id,
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "name": dataset_name(cfg.dataset),
            "path": str(cfg.dataset.path),
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "quantitative_idx": ds.quantitative_idx,
            "nominal_idx": ds.nominal_idx,
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
    write_json(weights_path, weights_payload)

    # weight_rank.json
    sorted_idx = prep.weight_sorted_feature_idx
    rank = prep.weight_rank_per_feature
    sorted_table = [
        {"rank": int(pos), "feature_index": int(fi), "weight": float(prep.w_full[fi])}
        for pos, fi in enumerate(sorted_idx)
    ]

    weight_rank_payload: Dict[str, Any] = {
        "run_name": run.task,
        "run_id": run.run_id,
        "ranking_rule": "primary: weight desc; tie: smaller feature_index first (left-to-right)",
        "sorted_feature_indices": sorted_idx,
        "rank_per_feature_index": rank,
        "sorted_table": sorted_table,  # easiest to compare with Excel
        "top_20": sorted_table[:20],
    }
    write_json(weight_rank_path, weight_rank_payload)

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
