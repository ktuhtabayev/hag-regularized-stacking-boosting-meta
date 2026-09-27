from __future__ import annotations

from typing import Any, Dict

from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import (
    greedy_hag_grouping,
)
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.cli import (
    hag_run_config_payload,
    load_stage_inputs,
    new_stage_run,
    parse_stage_args,
    write_dataset_snapshot,
    write_dij_csv,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json


def main() -> None:
    args = parse_stage_args("HAG training (Algorithm-1): TUPLAM and latent features dij.")

    # ============================================================
    # 1) Load config + dataset
    # ============================================================
    cfg, ds = load_stage_inputs(args.config)

    # ============================================================
    # 2) Prep merged contributions + merged weights (aligned to original indices)
    # ============================================================
    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        quantitative_idx=ds.quantitative_idx,
        nominal_idx=ds.nominal_idx,
    )

    # ============================================================
    # 3) Run Algorithm-1 (HAG greedy grouping); prints the Excel Step-3 columns
    # IMPORTANT: X must already be η-contribution-values dataset
    # ============================================================
    res = greedy_hag_grouping(
        X=prep.X_contrib_full,
        y=ds.y,
        weights=prep.w_full,
        params=cfg.hag,  # domain.params.HAGParams from YAML (includes majorizing.name+params)
    )

    # ============================================================
    # 4) Run folder outputs/runs/train/<run_id>/ + reproducibility snapshot
    # ============================================================
    run = new_stage_run(cfg, "train")
    run_dir = run.run_dir

    dataset_path_txt, dataset_cfg_json = write_dataset_snapshot(run_dir, cfg)
    # HAG config including majorizing.name + majorizing.params
    run_cfg_json = write_json(run_dir / "run_config.json", hag_run_config_payload(cfg))

    # ============================================================
    # 5) Artifacts: tuplam.json + dij.csv
    # ============================================================
    tuplam_path = run_dir / "tuplam.json"
    dij_path = run_dir / "dij.csv"

    tuplam_payload: Dict[str, Any] = {
        "run_name": run.task,
        "run_id": run.run_id,
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
    write_json(tuplam_path, tuplam_payload)

    # dij.csv (latent feature matrix), columns: r1..rp
    write_dij_csv(dij_path, res.dij)

    # ============================================================
    # 6) Console summary
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
