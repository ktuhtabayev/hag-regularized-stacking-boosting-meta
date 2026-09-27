from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import greedy_hag_grouping
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.algorithms.meta.new_object import (
    form_meta_new_object,
    format_snew,
)
from hag_regularized_stacking_boosting_meta.cli import (
    latest_run_dir,
    new_stage_run,
    parse_stage_args,
    save_csv,
    stage_root,
)
from hag_regularized_stacking_boosting_meta.io.configs import RunConfig, load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import LoadedDataset, load_dataset_bundle
from hag_regularized_stacking_boosting_meta.io.writers import write_json, write_text


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset path from YAML (optional).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for a reproducible Snew (default: a fresh random Snew on every run).",
    )
    p.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help=(
            "Optional explicit HAG train run folder to reuse. "
            "If omitted, script auto-detects latest valid run under outputs/runs/train/ "
            "or runs HAG if none exist."
        ),
    )


def _load_tuplam_from_run_folder(run_dir: Path) -> List[int]:
    tuplam_path = run_dir / "tuplam.json"
    if not tuplam_path.exists():
        raise FileNotFoundError(f"tuplam.json not found: {tuplam_path}")

    payload = json.loads(tuplam_path.read_text(encoding="utf-8"))
    tuplam = payload.get("tuplam", None)
    if tuplam is None or not isinstance(tuplam, list) or len(tuplam) == 0:
        raise ValueError(f"Invalid tuplam.json in {run_dir}: missing/empty 'tuplam'")
    return [int(x) for x in tuplam]


def _load_dataset_path_from_run_folder(run_dir: Path) -> Optional[Path]:
    p = run_dir / "dataset_path.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8").strip()
    return Path(txt) if txt else None


def _is_valid_train_run(run_dir: Path) -> bool:
    return (run_dir / "tuplam.json").exists()


@dataclass(frozen=True)
class TuplamSource:
    tuplam: List[int]
    source: str
    train_run_dir: Optional[Path]


def _get_tuplam_auto(
    *,
    cfg: RunConfig,
    ds: LoadedDataset,
    explicit_run_folder: Optional[str],
) -> TuplamSource:
    # 1) Explicit run folder
    if explicit_run_folder:
        run_dir = Path(explicit_run_folder)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-folder does not exist: {run_dir}")
        tuplam = _load_tuplam_from_run_folder(run_dir)
        return TuplamSource(
            tuplam=tuplam,
            source=f"reused explicit run folder: {run_dir}",
            train_run_dir=run_dir,
        )

    # 2) Auto-detect latest train run
    latest = latest_run_dir(stage_root(cfg, "train"), _is_valid_train_run)
    if latest is not None:
        tuplam = _load_tuplam_from_run_folder(latest)
        return TuplamSource(
            tuplam=tuplam,
            source=f"auto-reused latest train run: {latest}",
            train_run_dir=latest,
        )

    # 3) No train run exists -> run HAG now
    prep = prepare_hag_inputs(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        quantitative_idx=ds.quantitative_idx,
        nominal_idx=ds.nominal_idx,
    )
    hag_res = greedy_hag_grouping(
        X=prep.X_contrib_full,
        y=ds.y,
        weights=prep.w_full,
        params=cfg.hag,
    )
    return TuplamSource(
        tuplam=hag_res.tuplam,
        source="computed by running HAG (no prior train run found)",
        train_run_dir=None,
    )


def _load_dataset(cfg: RunConfig, dataset_path: Path) -> LoadedDataset:
    return load_dataset_bundle(replace(cfg.dataset, path=str(dataset_path)))


def main() -> None:
    args = parse_stage_args(
        "Create META new object Snew=(a0..ap) in initial + binary formats.", _add_arguments
    )

    # 1) Load config
    cfg = load_default_config(args.config)

    # Effective dataset path: an explicit --dataset-path wins over the YAML path
    effective_dataset_path = Path(args.dataset_path or cfg.dataset.path)
    ds = _load_dataset(cfg, effective_dataset_path)

    # 2) Obtain tuplam (auto reuse or auto run)
    tuplam_src = _get_tuplam_auto(cfg=cfg, ds=ds, explicit_run_folder=args.run_folder)

    # If reused train run has dataset_path.txt, prefer it unless user forced dataset-path
    if (args.dataset_path is None) and (tuplam_src.train_run_dir is not None):
        run_ds_path = _load_dataset_path_from_run_folder(tuplam_src.train_run_dir)
        if run_ds_path is not None:
            effective_dataset_path = run_ds_path
            ds = _load_dataset(cfg, effective_dataset_path)

    # 3) Form Snew
    obj = form_meta_new_object(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        tuplam=tuplam_src.tuplam,
        seed=args.seed,
    )

    print("\n==============================")
    print("META NEW OBJECT (Snew)")
    print("==============================")
    print("Source:", tuplam_src.source)
    print("Config:", args.config)
    print("Dataset path:", effective_dataset_path)
    print("TUPLAM (0-based):", obj.tuplam)

    print("\nInitial format (nominal original, quantitative raw):")
    print(format_snew(obj.a_headers, obj.a_init))

    print("\nBinary format (nominal same, quantitative -> {1,2}):")
    print(format_snew(obj.a_headers, obj.a_bin))

    # 4) Save for GUI / reuse
    out_dir = new_stage_run(cfg, "meta_new_object").run_dir
    write_text(out_dir / "source.txt", tuplam_src.source)
    write_text(out_dir / "config_path.txt", str(Path(args.config)))
    write_text(out_dir / "dataset_path.txt", str(effective_dataset_path))
    if tuplam_src.train_run_dir is not None:
        write_text(out_dir / "train_run_dir.txt", str(tuplam_src.train_run_dir))

    write_text(out_dir / "tuplam.txt", str(obj.tuplam))
    write_text(out_dir / "snew_headers.txt", ",".join(obj.a_headers))
    save_csv(out_dir / "snew_init.csv", obj.a_init.reshape(1, -1), header="")
    save_csv(out_dir / "snew_binary.csv", obj.a_bin.reshape(1, -1), header="")

    # JSON for GUI / future animation / visualization
    payload: Dict[str, Any] = {
        "source": tuplam_src.source,
        "config_path": str(Path(args.config)),
        "dataset_path": str(effective_dataset_path),
        "train_run_dir": str(tuplam_src.train_run_dir) if tuplam_src.train_run_dir is not None else None,
        "tuplam": [int(x) for x in obj.tuplam],
        "headers": list(obj.a_headers),
        "snew_init": [float(x) for x in obj.a_init.tolist()],
        "snew_binary": [int(round(float(x))) for x in obj.a_bin.tolist()],
        "gamma_map": {str(k): float(v) for k, v in obj.gamma_map.items()},
    }
    write_json(out_dir / "snew.json", payload)

    print("\nSaved:")
    print("Run folder:", out_dir)
    print("snew_init.csv:", out_dir / "snew_init.csv")
    print("snew_binary.csv:", out_dir / "snew_binary.csv")
    print("snew.json:", out_dir / "snew.json")


if __name__ == "__main__":
    main()
