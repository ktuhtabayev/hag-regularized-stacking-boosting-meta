from __future__ import annotations

from typing import Any, Dict

from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import build_quantitative_nominalization
from hag_regularized_stacking_boosting_meta.cli import (
    dataset_name,
    load_stage_inputs,
    new_stage_run,
    parse_stage_args,
    save_csv,
    write_dataset_snapshot,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json


def main() -> None:
    args = parse_stage_args("Quantitative weights: Criterion-1, Γc, binary {1,2} and η contributions.")

    # 1) Load YAML -> DatasetConfig (+ HAG defaults in cfg.hag), then the dataset
    cfg, ds = load_stage_inputs(args.config)

    if not ds.quantitative_idx:
        print("============================================================")
        print("QUANTITATIVE DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Quantitative demo skipped.")
        print("Reason: no quantitative features found (feature_types = 1).")
        print("Check your dataset feature-sign row:")
        print("  - quantitative feature  -> 1")
        print("  - nominal feature       -> 0")
        print("No output files were created for run_quantitative_weights.py.")
        return

    # 2) Run quantitative pipeline
    res = build_quantitative_nominalization(ds.X, ds.y, ds.quantitative_idx)

    # 3) Run folder: outputs/runs/quantitative_weights/<run_id>/
    run = new_stage_run(cfg, "quantitative_weights")
    run_dir = run.run_dir

    # 4) Save dataset path + dataset config (GUI-friendly + reproducible)
    dataset_path_txt, dataset_cfg_json = write_dataset_snapshot(run_dir, cfg)

    # 5) Save outputs; column headers keep the original feature indices
    q_headers = ",".join(f"qf_{idx}" for idx in res.quantitative_idx)
    binary_path = run_dir / "quantitative_binary.csv"
    contrib_path = run_dir / "quantitative_contrib.csv"
    crit_path = run_dir / "criterion1_table.json"

    save_csv(binary_path, res.binary_X.astype(int), q_headers, fmt="%d")
    save_csv(contrib_path, res.contribution_X.astype(float), q_headers)

    # 6) Save Criterion-1 table + Γc + η contributions as JSON
    payload: Dict[str, Any] = {
        "run_name": run.task,
        "run_id": run.run_id,
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "name": dataset_name(cfg.dataset),
            "path": cfg.dataset.path,
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "quantitative_idx": res.quantitative_idx,
            "nominal_idx": ds.nominal_idx,
        },
        "hag_defaults": {
            "alpha": cfg.hag.alpha,
            "delta": cfg.hag.delta,
            "kappa": cfg.hag.kappa,
            "majorizing_function": str(cfg.hag.majorizing.name),
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
    write_json(crit_path, payload)

    # 7) Print summary (CMD-friendly)
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
