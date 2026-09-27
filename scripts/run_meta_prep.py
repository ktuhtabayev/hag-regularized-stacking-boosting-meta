from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import greedy_hag_grouping
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.algorithms.meta.training_set import prepare_meta_training_dataset
from hag_regularized_stacking_boosting_meta.cli import (
    hag_run_config_payload,
    load_stage_inputs,
    new_stage_run,
    parse_stage_args,
    save_csv,
    write_dataset_snapshot,
    write_dij_csv,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json


def _save_csv_meta(path: Path, headers: list[str], S: np.ndarray, y: np.ndarray) -> None:
    """
    Save META training dataset as:
      ai0..aip, di1..dip, Class
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1, 1)
    save_csv(path, np.hstack([S, y.astype(float)]), ",".join(headers))


def main() -> None:
    args = parse_stage_args("META preparation: training set (ai*, di*, Class) from a fresh HAG run.")

    # 1) Load config + dataset
    cfg, ds = load_stage_inputs(args.config)

    # 2) Run HAG prep -> contributions + weights (for organizer only)
    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        quantitative_idx=ds.quantitative_idx,
        nominal_idx=ds.nominal_idx,
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
    run = new_stage_run(cfg, "meta_prep")
    run_dir = run.run_dir

    write_dataset_snapshot(run_dir, cfg)
    write_json(run_dir / "run_config.json", hag_run_config_payload(cfg))

    # Artifacts
    tuplam_path = run_dir / "tuplam.json"
    dij_path = run_dir / "dij.csv"
    meta_train_path = run_dir / "meta_train.csv"

    tuplam_payload: Dict[str, Any] = {
        "run_name": run.task,
        "run_id": run.run_id,
        "run_dir": str(run_dir.as_posix()),
        "tuplam": hag_res.tuplam,              # 0-based ordered
        "size": len(hag_res.tuplam),
        "p_latent": hag_res.p,
        "crit_history": hag_res.crit_history,
    }
    write_json(tuplam_path, tuplam_payload)
    write_dij_csv(dij_path, hag_res.dij)
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
