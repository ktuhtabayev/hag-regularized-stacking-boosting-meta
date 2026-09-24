from __future__ import annotations

import argparse
import itertools
import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from hag_regularized_stacking_boosting_meta.io.loaders import DatasetConfig, load_dataset_bundle
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import (
    build_quantitative_nominalization,
    build_nominal_contributions,
)


# ============================================================
# Utilities
# ============================================================

def _now_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Any:
    """Convert common objects to JSON-serializable forms."""
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict override into base and return a new dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _find_grid_specs(d: Dict[str, Any], path_prefix: str = "") -> List[Tuple[str, List[Any]]]:
    """
    Find all keys ending with '_grid' anywhere in nested dict.
    Returns list of (path_to_basekey, values).
    Example: {"hag":{"alpha_grid":[0.1,0.3]}} -> [("hag.alpha",[0.1,0.3])]
    """
    specs: List[Tuple[str, List[Any]]] = []
    for k, v in d.items():
        cur_path = f"{path_prefix}.{k}" if path_prefix else k
        if isinstance(v, dict):
            specs.extend(_find_grid_specs(v, cur_path))
        else:
            if k.endswith("_grid") and isinstance(v, list):
                base_key = k[:-5]  # remove "_grid"
                base_path = f"{path_prefix}.{base_key}" if path_prefix else base_key
                specs.append((base_path, v))
    return specs


def _set_by_path(d: Dict[str, Any], dot_path: str, value: Any) -> None:
    """Set d[a][b][c] = value given dot path 'a.b.c' creating dicts as needed."""
    parts = dot_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _remove_grid_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of d with any '*_grid' keys removed (recursively)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k.endswith("_grid"):
            continue
        if isinstance(v, dict):
            out[k] = _remove_grid_keys(v)
        else:
            out[k] = v
    return out


def _expand_grids(merged_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand config into multiple configs by cartesian product of all *_grid specs.
    If no grids found -> single config returned.
    """
    grid_specs = _find_grid_specs(merged_cfg)
    if not grid_specs:
        return [_remove_grid_keys(merged_cfg)]

    base_cfg = _remove_grid_keys(merged_cfg)

    # cartesian product across all grid lists
    keys = [k for k, _ in grid_specs]
    values_lists = [v for _, v in grid_specs]

    expanded: List[Dict[str, Any]] = []
    for combo in itertools.product(*values_lists):
        cfg_i = dict(base_cfg)
        # deep copy nested dicts safely by yaml roundtrip (simple + reliable)
        cfg_i = yaml.safe_load(yaml.safe_dump(cfg_i))
        for path, val in zip(keys, combo):
            _set_by_path(cfg_i, path, val)
        expanded.append(cfg_i)

    return expanded


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _save_csv_int(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(int), delimiter=",", header=header, comments="", fmt="%d")


def _save_csv_float(path: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(path, data.astype(float), delimiter=",", header=header, comments="", fmt="%.9f")


# ============================================================
# Stage runners (ONLY qdemo + ndemo for now)
# ============================================================

def _dataset_config_from_cfg(cfg: Dict[str, Any]) -> DatasetConfig:
    ds = cfg.get("dataset", {}) or {}
    # Keep defaults safe if missing keys
    return DatasetConfig(
        path=ds.get("path", "datasets/raw/default.csv"),
        # path=ds.get("path", "datasets/raw/default.dat"),
        format=ds.get("format", None),
        delimiter=ds.get("delimiter", ","),
        has_metadata_header=bool(ds.get("has_metadata_header", True)),
        has_feature_type_row=bool(ds.get("has_feature_type_row", True)),
        label_col=int(ds.get("label_col", -1)),
        label_mapping=ds.get("label_mapping", {"1": 1, "2": 2}),
    )


def _write_repro_files(run_dir: Path, dataset_cfg: DatasetConfig, merged_cfg: Dict[str, Any]) -> None:
    # dataset_path.txt
    (run_dir / "dataset_path.txt").write_text(str(dataset_cfg.path), encoding="utf-8")

    # dataset_config.json
    ds_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(dataset_cfg).items()}
    (run_dir / "dataset_config.json").write_text(
        json.dumps(ds_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # run_config.yaml (the exact merged config used for this run)
    _save_yaml(run_dir / "run_config.yaml", merged_cfg)


def run_quantitative_demo(run_dir: Path, merged_cfg: Dict[str, Any]) -> None:
    dataset_cfg = _dataset_config_from_cfg(merged_cfg)
    ds = load_dataset_bundle(dataset_cfg)

    if not ds.quantitative_idx:
        raise RuntimeError(
            "No quantitative features found (feature_types=1). "
            "Check dataset last row (0/1 flags)."
        )

    _write_repro_files(run_dir, dataset_cfg, merged_cfg)

    res = build_quantitative_nominalization(ds.X, ds.y, ds.quantitative_idx)

    q_headers = ",".join([f"qf_{idx}" for idx in res.quantitative_idx])

    binary_path = run_dir / "quantitative_binary.csv"
    contrib_path = run_dir / "quantitative_contrib.csv"
    crit_path = run_dir / "criterion1_table.json"

    _save_csv_int(binary_path, res.binary_X, header=q_headers)
    _save_csv_float(contrib_path, res.contribution_X, header=q_headers)

    payload: Dict[str, Any] = {
        "stage": "quantitative_demo",
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "path": str(dataset_cfg.path),
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "quantitative_idx": res.quantitative_idx,
            "nominal_idx": getattr(ds, "nominal_idx", None),
        },
        "hag_defaults": _to_jsonable(merged_cfg.get("hag", {})),
        "criterion1": {
            str(fidx): {
                "pi1": res.pi_table[fidx][0],
                "pi2": res.pi_table[fidx][1],
                "pi3": res.pi_table[fidx][2],
                "wc": res.weights_wc[fidx],
                "gamma": res.gamma[fidx],
                "eta": res.contributions_eta[fidx],
            }
            for fidx in res.quantitative_idx
        },
        "files": {
            "dataset_path_txt": "dataset_path.txt",
            "dataset_config_json": "dataset_config.json",
            "run_config_yaml": "run_config.yaml",
            "quantitative_binary_csv": "quantitative_binary.csv",
            "quantitative_contrib_csv": "quantitative_contrib.csv",
            "criterion1_table_json": "criterion1_table.json",
        },
    }

    crit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_nominal_demo(run_dir: Path, merged_cfg: Dict[str, Any]) -> None:
    dataset_cfg = _dataset_config_from_cfg(merged_cfg)
    ds = load_dataset_bundle(dataset_cfg)

    if not getattr(ds, "nominal_idx", None):
        raise RuntimeError(
            "No nominal features found (feature_types=0). "
            "Check dataset last row (0/1 flags): nominal must be 0."
        )

    _write_repro_files(run_dir, dataset_cfg, merged_cfg)

    res = build_nominal_contributions(ds.X, ds.y, ds.nominal_idx)

    n_headers = ",".join([f"nf_{idx}" for idx in res.nominal_idx])

    contrib_path = run_dir / "nominal_contrib.csv"
    table_path = run_dir / "lambda_beta_weight_table.json"

    _save_csv_float(contrib_path, res.contribution_X, header=n_headers)

    # IMPORTANT: use the correct attribute names from NominalWeightsResult
    # (lambda_c, beta_c, w_c, p_c, l1_c, l2_c, d1_c, d2_c, eta_c)
    payload: Dict[str, Any] = {
        "stage": "nominal_demo",
        "run_dir": str(run_dir.as_posix()),
        "dataset": {
            "path": str(dataset_cfg.path),
            "shape_X": list(ds.X.shape),
            "shape_y": list(ds.y.shape),
            "classes": sorted(set(ds.y.tolist())),
            "feature_types": ds.feature_types.tolist(),
            "nominal_idx": res.nominal_idx,
            "quantitative_idx": getattr(ds, "quantitative_idx", None),
        },
        "hag_defaults": _to_jsonable(merged_cfg.get("hag", {})),
        "nominal_tables": _to_jsonable(
            {
                "lambda_c": res.lambda_c,
                "beta_c": res.beta_c,
                "weight_wc": res.w_c,            # correct name
                "p_c": res.p_c,
                "l1_c": res.l1_c,
                "l2_c": res.l2_c,
                "D1_c": res.d1_c,                # correct name (camel in json)
                "D2_c": res.d2_c,                # correct name (camel in json)
                "gradations": res.gradations,
                "contributions_eta": res.eta_c,  # correct name (eta_c)
            }
        ),
        "files": {
            "dataset_path_txt": "dataset_path.txt",
            "dataset_config_json": "dataset_config.json",
            "run_config_yaml": "run_config.yaml",
            "nominal_contrib_csv": "nominal_contrib.csv",
            "lambda_beta_weight_table_json": "lambda_beta_weight_table.json",
        },
    }

    table_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# Reading experiments.yaml and executing
# ============================================================

def _load_experiments_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"experiments.yaml not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _iter_selected_experiments(cfg: Dict[str, Any], selected_id: str | None) -> List[Dict[str, Any]]:
    exps = cfg.get("experiments", []) or []
    if selected_id is None:
        return exps
    out = [e for e in exps if e.get("id") == selected_id]
    if not out:
        known = [e.get("id") for e in exps]
        raise ValueError(f"Unknown experiment id '{selected_id}'. Known: {known}")
    return out


def _run_one(stage: str, run_name: str, merged_cfg: Dict[str, Any], root_runs_dir: str = "outputs/runs") -> Path:
    run_id = _now_run_id()
    run_dir = _ensure_dir(Path(root_runs_dir) / run_name / run_id)

    if stage == "quantitative_demo":
        run_quantitative_demo(run_dir, merged_cfg)
    elif stage == "nominal_demo":
        run_nominal_demo(run_dir, merged_cfg)
    else:
        raise ValueError(
            f"Stage '{stage}' not supported in run_experiments.py yet. "
            "Only: quantitative_demo, nominal_demo."
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments from configs/experiments.yaml (only qdemo + ndemo for now)."
    )
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml")
    parser.add_argument("--id", default=None, help="Run only one experiment by id (otherwise run all)")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded runs without executing")
    args = parser.parse_args()

    cfg = _load_experiments_yaml(args.config)

    defaults = cfg.get("defaults", {}) or {}
    outputs_cfg = defaults.get("outputs", {}) or {}
    runs_dir = outputs_cfg.get("runs_dir", "outputs/runs")

    selected = _iter_selected_experiments(cfg, args.id)

    total_runs = 0
    for exp in selected:
        exp_id = exp.get("id", "<no-id>")
        stage = exp.get("stage")
        run_name = exp.get("run_name") or stage

        if stage not in ("quantitative_demo", "nominal_demo"):
            # skip future stages safely
            print(f"⏭️  Skipping '{exp_id}' (stage={stage}) - not supported yet in run_experiments.py")
            continue

        overrides = exp.get("overrides", {}) or {}

        merged = _deep_merge(defaults, overrides)
        expanded_cfgs = _expand_grids(merged)

        print(f"\n=== Experiment: {exp_id} | stage={stage} | run_name={run_name} ===")
        print(f"Grid expanded runs: {len(expanded_cfgs)}")

        for i, cfg_i in enumerate(expanded_cfgs, start=1):
            total_runs += 1
            if args.dry_run:
                print(f"  [DRY] run {i}/{len(expanded_cfgs)} -> would write to {runs_dir}/{run_name}/<run_id>/")
                continue

            run_dir = _run_one(stage=stage, run_name=run_name, merged_cfg=cfg_i, root_runs_dir=runs_dir)
            print(f"  ✅ run {i}/{len(expanded_cfgs)} completed -> {run_dir.as_posix()}")

    if args.dry_run:
        print(f"\n(DRY RUN) Total planned runs: {total_runs}")
    else:
        print(f"\n✅ Done. Total executed runs: {total_runs}")


if __name__ == "__main__":
    main()
