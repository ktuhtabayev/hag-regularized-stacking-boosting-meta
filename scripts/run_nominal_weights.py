from __future__ import annotations

from typing import Any, Dict

# STABLE FACADE import (your project rule)
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import build_nominal_contributions
from hag_regularized_stacking_boosting_meta.cli import (
    dataset_name,
    load_stage_inputs,
    new_stage_run,
    parse_stage_args,
    save_csv,
    to_jsonable,
    write_dataset_snapshot,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json


def main() -> None:
    args = parse_stage_args("Nominal weights: λ, β, ω and η contributions.")
    cfg, ds = load_stage_inputs(args.config)

    if not ds.nominal_idx:
        print("============================================================")
        print("NOMINAL DEMO")
        print("============================================================")
        print(f"Dataset path: {cfg.dataset.path}")
        print("Nominal demo skipped.")
        print("Reason: no nominal features found (feature_types = 0).")
        print("Check your dataset feature-sign row:")
        print("  - nominal feature       -> 0")
        print("  - quantitative feature  -> 1")
        print("No output files were created for run_nominal_weights.py.")
        return

    res = build_nominal_contributions(ds.X, ds.y, ds.nominal_idx)

    run = new_stage_run(cfg, "nominal_weights")
    run_dir = run.run_dir
    dataset_path_txt, dataset_cfg_json = write_dataset_snapshot(run_dir, cfg)

    n_headers = ",".join(f"nf_{idx}" for idx in res.nominal_idx)
    contrib_path = run_dir / "nominal_contrib.csv"
    table_path = run_dir / "lambda_beta_weight_table.json"

    save_csv(contrib_path, res.contribution_X.astype(float), n_headers)

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
            "nominal_idx": res.nominal_idx,
            "quantitative_idx": ds.quantitative_idx,
        },
        "hag_defaults": {
            "alpha": cfg.hag.alpha,
            "delta": cfg.hag.delta,
            "kappa": cfg.hag.kappa,
            "majorizing_function": str(cfg.hag.majorizing.name),
        },
        "nominal_tables": to_jsonable(
            {
                "lambda_c": res.lambda_c,
                "beta_c": res.beta_c,
                "weight_wc": res.weight_wc,
                "p_c": res.p_c,
                "l1_c": res.l1_c,
                "l2_c": res.l2_c,
                "D1_c": res.D1_c,
                "D2_c": res.D2_c,
                "gradations": res.gradations,
                "contributions_eta": res.contributions_eta,
            }
        ),
        "files": {
            "dataset_path_txt": str(dataset_path_txt.as_posix()),
            "dataset_config_json": str(dataset_cfg_json.as_posix()),
            "nominal_contrib_csv": str(contrib_path.as_posix()),
            "lambda_beta_weight_table_json": str(table_path.as_posix()),
        },
    }
    write_json(table_path, payload)

    print("Nominal demo completed")
    print("Run folder:", run_dir)
    print("Contrib CSV:", contrib_path)
    print("Lambda/Beta/Weight JSON:", table_path)
    print("Nominal idx:", res.nominal_idx)
    print("Contrib shape:", res.contribution_X.shape)


if __name__ == "__main__":
    main()
