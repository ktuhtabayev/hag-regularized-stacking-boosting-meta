from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from hag_regularized_stacking_boosting_meta.domain.params import HAGParams, MajorizingConfig
from hag_regularized_stacking_boosting_meta.io.loaders import DatasetConfig


@dataclass(frozen=True)
class OutputConfig:
    """
    Output settings that GUI + scripts can share.
    """
    root_dir: str = "outputs"
    runs_dir: str = "runs"


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetConfig
    hag: HAGParams
    output: OutputConfig = OutputConfig()
    seed: int = 42


def _as_dict(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def load_default_config(path: str | Path) -> RunConfig:
    """
    Load YAML config -> RunConfig.
    This is the ONE place where users/GUI control majorizing functions and params.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _as_dict(raw)

    ds_raw = _as_dict(raw.get("dataset", {}))
    hag_raw = _as_dict(raw.get("hag", {}))
    out_raw = _as_dict(raw.get("output", {}))

    # --------------------------
    # DatasetConfig (already used across project)
    # --------------------------
    dataset = DatasetConfig(
        path=str(_get(ds_raw, "path", "datasets/raw/default.csv")),
        format=_get(ds_raw, "format", None),
        delimiter=_get(ds_raw, "delimiter", ","),
        has_metadata_header=bool(_get(ds_raw, "has_metadata_header", True)),
        has_feature_type_row=bool(_get(ds_raw, "has_feature_type_row", True)),
        label_col=int(_get(ds_raw, "label_col", -1)),
        label_mapping=_get(ds_raw, "label_mapping", {"1": 1, "2": 2}),
    )

    # --------------------------
    # Majorizing config
    # --------------------------
    maj_raw = _as_dict(hag_raw.get("majorizing", {}))
    # Backward compatibility:
    # older configs might have majorizing_function: "sigmoid"
    legacy_name = hag_raw.get("majorizing_function", None)

    majorizing = MajorizingConfig(
        name=str(maj_raw.get("name", legacy_name or "identity")),
        params=_as_dict(maj_raw.get("params", {})),
    )

    # --------------------------
    # HAGParams
    # --------------------------
    hag = HAGParams(
        alpha=float(_get(hag_raw, "alpha", 0.3)),
        delta=float(_get(hag_raw, "delta", 0.1)),
        kappa=int(_get(hag_raw, "kappa", 5)),
        cr1=float(_get(hag_raw, "cr1", 10.0)),
        k1_label=int(_get(hag_raw, "k1_label", 1)),
        k2_label=int(_get(hag_raw, "k2_label", 2)),
        majorizing=majorizing,
    )

    # --------------------------
    # Output
    # --------------------------
    output = OutputConfig(
        root_dir=str(_get(out_raw, "root_dir", "outputs")),
        runs_dir=str(_get(out_raw, "runs_dir", "runs")),
    )

    seed = int(_get(raw, "seed", 42))

    return RunConfig(dataset=dataset, hag=hag, output=output, seed=seed)


def load_dataset_catalog(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the optional `dataset_catalog` block (preset name -> {"path": ...}).
    Used by the GUI dataset dropdown; entries without a path are skipped.
    """
    raw = _as_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    catalog: Dict[str, Dict[str, Any]] = {}
    for name, entry in _as_dict(raw.get("dataset_catalog", {})).items():
        entry = _as_dict(entry)
        if entry.get("path"):
            catalog[str(name)] = entry
    return catalog
