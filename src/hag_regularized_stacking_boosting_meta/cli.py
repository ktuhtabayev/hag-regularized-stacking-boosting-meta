"""
Shared plumbing for the stage scripts in scripts/: arguments, config, run folders
and the artifact formats that more than one stage writes.

Every stage writes outputs/runs/<stage>/<run_id>/ (root and runs dir from the
config's `output` block); paths in the config are relative to the working directory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from hag_regularized_stacking_boosting_meta.io.configs import RunConfig, load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import (
    DatasetConfig,
    LoadedDataset,
    load_dataset_bundle,
)
from hag_regularized_stacking_boosting_meta.io.writers import write_json, write_text
from hag_regularized_stacking_boosting_meta.utils.run_manager import (
    RUN_ID_PATTERN,
    RunContext,
    create_run_dir,
)


DEFAULT_CONFIG = "configs/default.yaml"


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

def configure_console() -> None:
    """UTF-8 stdout/stderr, so Unicode messages survive redirection on Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def stage_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the --config option every stage accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG}).",
    )
    return parser


def parse_stage_args(
    description: str,
    add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    configure_console()
    parser = stage_parser(description)
    if add_arguments is not None:
        add_arguments(parser)
    return parser.parse_args(argv)


def load_stage_inputs(config_path: str) -> Tuple[RunConfig, LoadedDataset]:
    cfg = load_default_config(config_path)
    return cfg, load_dataset_bundle(cfg.dataset)


# ------------------------------------------------------------------
# Run folders
# ------------------------------------------------------------------

def runs_root(cfg: RunConfig) -> Path:
    return Path(cfg.output.root_dir) / cfg.output.runs_dir


def stage_root(cfg: RunConfig, stage: str) -> Path:
    return runs_root(cfg) / stage


def new_stage_run(cfg: RunConfig, stage: str) -> RunContext:
    """Create outputs/runs/<stage>/<new run_id>/."""
    return create_run_dir(stage, runs_root(cfg))


def _run_recency_key(run_dir: Path) -> Tuple[str, int, str]:
    """
    Run ids start with their creation second (YYYYMMDD_HHMMSS); ids from the same second
    differ only by a random suffix, so those are ordered by modification time instead.
    """
    match = RUN_ID_PATTERN.match(run_dir.name)
    if match is None:
        return (run_dir.name, 0, run_dir.name)
    return (match.group(1), run_dir.stat().st_mtime_ns, run_dir.name)


def latest_run_dir(root: Path, is_valid: Optional[Callable[[Path], bool]] = None) -> Optional[Path]:
    """Newest run folder under `root`, optionally among valid ones only."""
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir() and (is_valid is None or is_valid(p))]
    return max(dirs, key=_run_recency_key) if dirs else None


# ------------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, dict keys, numpy values and paths to JSON types."""
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_csv(path: Path, data: np.ndarray, header: str, fmt: str | Sequence[str] = "%.9f") -> None:
    """Comma-separated numeric table with a single header line (no comment prefix)."""
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)


def dataset_name(dataset: DatasetConfig) -> str:
    """Dataset file name without its extension, e.g. 'Heart-Disease (270, 13, 2)'."""
    return Path(dataset.path).stem


def dataset_config_payload(dataset: DatasetConfig) -> Dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(dataset).items()}


def write_dataset_snapshot(run_dir: Path, cfg: RunConfig) -> Tuple[Path, Path]:
    """dataset_path.txt + dataset_config.json: which data (and how it was read) produced the run."""
    path_txt = write_text(run_dir / "dataset_path.txt", str(cfg.dataset.path))
    config_json = write_json(run_dir / "dataset_config.json", dataset_config_payload(cfg.dataset))
    return path_txt, config_json


def hag_run_config_payload(cfg: RunConfig) -> Dict[str, Any]:
    """run_config.json of the HAG stages: seed + every HAG parameter incl. the majorizing function."""
    return {
        "seed": int(cfg.seed),
        "hag": {
            "alpha": float(cfg.hag.alpha),
            "delta": float(cfg.hag.delta),
            "kappa": int(cfg.hag.kappa),
            "cr1": float(cfg.hag.cr1),
            "k1_label": int(cfg.hag.k1_label),
            "k2_label": int(cfg.hag.k2_label),
            "majorizing": {
                "name": str(cfg.hag.majorizing.name),
                "params": dict(cfg.hag.majorizing.params or {}),
            },
        },
    }


def write_dij_csv(path: Path, dij: np.ndarray) -> None:
    """Latent features r1..rp (one column each); only a header when HAG produced none."""
    if dij.size == 0:
        write_text(path, "r1\n")
    else:
        header = ",".join(f"r{i + 1}" for i in range(dij.shape[1]))
        save_csv(path, dij.astype(float), header)
