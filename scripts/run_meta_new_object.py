from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import prepare_hag_inputs
from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import greedy_hag_grouping
from hag_regularized_stacking_boosting_meta.algorithms.meta.new_object import (
    form_meta_new_object,
    format_snew,
)


# -----------------------------
# Helpers
# -----------------------------

def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create META new object Snew=(a0..ap) in initial + binary formats."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config path (default: configs/default.yaml).",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset path from YAML (optional).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260222,
        help="Random seed for reproducible Snew (default: 20260222).",
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
    return p.parse_args()


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


def _find_latest_train_run(train_root: Path) -> Optional[Path]:
    if not train_root.exists():
        return None

    candidates = [p for p in train_root.iterdir() if p.is_dir() and _is_valid_train_run(p)]
    if not candidates:
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda p: (p.name, p.stat().st_mtime),
        reverse=True,
    )
    return candidates_sorted[0]


@dataclass(frozen=True)
class TuplamSource:
    tuplam: List[int]
    source: str
    train_run_dir: Optional[Path]


def _get_tuplam_auto(
    *,
    cfg,
    ds,
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
    train_root = Path("outputs") / "runs" / "train"
    latest = _find_latest_train_run(train_root)
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
        quantitative_idx=getattr(ds, "quantitative_idx", None),
        nominal_idx=getattr(ds, "nominal_idx", None),
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


def _dataset_cfg_with_path(dataset_cfg: Any, dataset_path: Path):
    """Return a copy of frozen/unfrozen dataset config with updated path."""
    return replace(dataset_cfg, path=Path(dataset_path))


def main() -> None:
    args = _parse_args()

    # 1) Load config
    cfg = load_default_config(args.config)

    # Decide effective dataset path WITHOUT mutating frozen config
    effective_dataset_path = Path(cfg.dataset.path)

    # Optional explicit dataset override wins first
    if args.dataset_path:
        effective_dataset_path = Path(args.dataset_path)

    # First load using current effective path
    dataset_cfg = _dataset_cfg_with_path(cfg.dataset, effective_dataset_path)
    ds = load_dataset_bundle(dataset_cfg)

    # 2) Obtain tuplam (auto reuse or auto run)
    tuplam_src = _get_tuplam_auto(cfg=cfg, ds=ds, explicit_run_folder=args.run_folder)

    # If reused train run has dataset_path.txt, prefer it unless user forced dataset-path
    if (args.dataset_path is None) and (tuplam_src.train_run_dir is not None):
        run_ds_path = _load_dataset_path_from_run_folder(tuplam_src.train_run_dir)
        if run_ds_path is not None:
            effective_dataset_path = run_ds_path
            dataset_cfg = _dataset_cfg_with_path(cfg.dataset, effective_dataset_path)
            ds = load_dataset_bundle(dataset_cfg)

    # 3) Form Snew
    obj = form_meta_new_object(
        X=ds.X,
        y=ds.y,
        feature_types=ds.feature_types,
        tuplam=tuplam_src.tuplam,
        # seed=args.seed,
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
    out_dir = _ensure_dir(Path("outputs") / "runs" / "meta_new_object" / _make_run_id())
    (out_dir / "source.txt").write_text(tuplam_src.source, encoding="utf-8")
    (out_dir / "config_path.txt").write_text(str(Path(args.config)), encoding="utf-8")
    (out_dir / "dataset_path.txt").write_text(str(effective_dataset_path), encoding="utf-8")
    if tuplam_src.train_run_dir is not None:
        (out_dir / "train_run_dir.txt").write_text(str(tuplam_src.train_run_dir), encoding="utf-8")

    (out_dir / "tuplam.txt").write_text(str(obj.tuplam), encoding="utf-8")
    (out_dir / "snew_headers.txt").write_text(",".join(obj.a_headers), encoding="utf-8")
    np.savetxt(out_dir / "snew_init.csv", obj.a_init.reshape(1, -1), delimiter=",", fmt="%.9f")
    np.savetxt(out_dir / "snew_binary.csv", obj.a_bin.reshape(1, -1), delimiter=",", fmt="%.9f")

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
    (out_dir / "snew.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved:")
    print("Run folder:", out_dir)
    print("snew_init.csv:", out_dir / "snew_init.csv")
    print("snew_binary.csv:", out_dir / "snew_binary.csv")
    print("snew.json:", out_dir / "snew.json")


if __name__ == "__main__":
    main()