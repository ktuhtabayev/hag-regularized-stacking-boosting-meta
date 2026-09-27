"""META prediction demo (Algorithm-2).

This script:
1) Finds or auto-creates every required upstream artifact:
   - train run
   - meta_prep run
   - meta_new_object run
2) Loads the latest valid run for each stage if multiple exist
3) Runs META Steps 1..4
4) Saves GUI-friendly debug JSON with full B1/B2 history per j

Usage:
  python scripts/run_meta_prediction.py
  python scripts/run_meta_prediction.py --reuse-new-object
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hag_regularized_stacking_boosting_meta.algorithms.meta import MetaClassifier
from hag_regularized_stacking_boosting_meta.io.configs import load_default_config


_AI_RE = re.compile(r"^ai(\d+)\b")
_DI_RE = re.compile(r"^di(\d+)\b")


def _make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run META prediction demo with full auto-dependency creation.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--train-run", type=str, default=None, help="Optional explicit train run folder.")
    p.add_argument("--meta-prep-run", type=str, default=None, help="Optional explicit meta_prep run folder.")
    p.add_argument("--new-object-run", type=str, default=None, help="Optional explicit meta_new_object run folder.")
    p.add_argument(
        "--reuse-new-object",
        action="store_true",
        help="Reuse the latest valid meta_new_object run instead of generating a fresh one.",
    )
    return p.parse_args()


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _sorted_cols_by_prefix(headers: List[str], rx: re.Pattern[str]) -> List[int]:
    pairs: List[Tuple[int, int]] = []
    for idx, h in enumerate(headers):
        m = rx.match(h.strip())
        if m:
            pairs.append((int(m.group(1)), idx))
    pairs.sort(key=lambda t: t[0])
    return [idx for _, idx in pairs]


def _load_meta_train_csv(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray | None]:
    """Loads meta_train.csv into (A, D, y, headers, row_ids) with correct ai/di ordering.

    Supports optional 'Sid' column for stable object IDs. If absent, row order 0..n-1 is used.
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows: List[List[str]] = [r for r in reader if r]

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2:
        raise ValueError("meta_train.csv must be a 2-D table")

    if "Class" not in headers:
        raise ValueError("meta_train.csv must have 'Class' column")
    y_col = headers.index("Class")

    a_cols = _sorted_cols_by_prefix(headers, _AI_RE)
    d_cols = _sorted_cols_by_prefix(headers, _DI_RE)

    if not a_cols:
        raise ValueError("No ai* columns found in meta_train.csv headers")
    if not d_cols:
        raise ValueError("No di* columns found in meta_train.csv headers")

    A = data[:, a_cols].astype(int)
    D = data[:, d_cols].astype(float)
    y = data[:, y_col].astype(int)

    if D.shape[1] != (A.shape[1] - 1):
        raise ValueError(
            f"Invalid meta_train.csv: expected D columns={A.shape[1]-1}, got {D.shape[1]}."
        )

    row_ids: np.ndarray | None = None
    if "Sid" in headers:
        row_ids = data[:, headers.index("Sid")].astype(int)

    return A, D, y, headers, row_ids


def _is_valid_train_run(run_dir: Path) -> bool:
    return (run_dir / "tuplam.json").exists() and (run_dir / "dij.csv").exists()


def _is_valid_meta_prep_run(run_dir: Path) -> bool:
    return (run_dir / "meta_train.csv").exists()


def _is_valid_new_object_run(run_dir: Path) -> bool:
    return (run_dir / "snew_binary.csv").exists() and (run_dir / "snew_headers.txt").exists()


def _load_snew_binary(run_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any] | None]:
    """Load Snew binary vector from JSON if present, otherwise CSV fallback."""
    snew_json = run_dir / "snew.json"
    if snew_json.exists():
        payload = json.loads(snew_json.read_text(encoding="utf-8"))
        headers = [str(x) for x in payload.get("headers", [])]
        a_new = np.asarray(payload.get("snew_binary", []), dtype=int)
        if a_new.ndim != 1:
            raise ValueError(f"Invalid snew.json in {run_dir}")
        return a_new, headers, payload

    csv_path = run_dir / "snew_binary.csv"
    headers_path = run_dir / "snew_headers.txt"
    if not csv_path.exists() or not headers_path.exists():
        raise FileNotFoundError(f"Missing Snew artifacts in {run_dir}")

    arr = np.loadtxt(csv_path, delimiter=",", dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr], dtype=float)
    elif arr.ndim == 2:
        arr = arr.reshape(-1)
    headers = [h.strip() for h in headers_path.read_text(encoding="utf-8").split(",") if h.strip()]
    return arr.astype(int), headers, None


def _run_script(project_root: Path, script_name: str, config_path: str) -> None:
    script_path = project_root / "scripts" / script_name
    cmd = [sys.executable, str(script_path), "--config", config_path]
    subprocess.run(cmd, cwd=str(project_root), check=True)


def _get_or_create_train_run(
    *,
    project_root: Path,
    explicit_run: Optional[str],
    config_path: str,
    train_root: Path,
) -> Path:
    if explicit_run is not None:
        run_dir = Path(explicit_run)
        if not run_dir.exists():
            raise FileNotFoundError(f"--train-run does not exist: {run_dir}")
        if not _is_valid_train_run(run_dir):
            raise FileNotFoundError(f"Invalid train run: {run_dir}")
        return run_dir

    latest = _latest_run_dir(train_root)
    if latest is not None and _is_valid_train_run(latest):
        return latest

    _run_script(project_root, "run_hag_training.py", config_path)
    latest = _latest_run_dir(train_root)
    if latest is None or not _is_valid_train_run(latest):
        raise RuntimeError("run_hag_training.py was executed, but no valid train run was created.")
    return latest


def _get_or_create_meta_prep_run(
    *,
    project_root: Path,
    explicit_run: Optional[str],
    config_path: str,
    train_root: Path,
    meta_root: Path,
) -> Path:
    if explicit_run is not None:
        run_dir = Path(explicit_run)
        if not run_dir.exists():
            raise FileNotFoundError(f"--meta-prep-run does not exist: {run_dir}")
        if not _is_valid_meta_prep_run(run_dir):
            raise FileNotFoundError(f"Invalid meta_prep run: {run_dir}")
        return run_dir

    latest = _latest_run_dir(meta_root)
    if latest is not None and _is_valid_meta_prep_run(latest):
        return latest

    # Ensure upstream train exists first
    _get_or_create_train_run(
        project_root=project_root,
        explicit_run=None,
        config_path=config_path,
        train_root=train_root,
    )

    _run_script(project_root, "run_meta_prep.py", config_path)
    latest = _latest_run_dir(meta_root)
    if latest is None or not _is_valid_meta_prep_run(latest):
        raise RuntimeError("run_meta_prep.py was executed, but no valid meta_prep run was created.")
    return latest


def _get_or_create_new_object_run(
    *,
    project_root: Path,
    explicit_run: Optional[str],
    config_path: str,
    train_root: Path,
    meta_new_root: Path,
    reuse_new_object: bool,
) -> Path:
    """
    Behavior:
    - If --new-object-run is explicitly provided, use it.
    - Else if --reuse-new-object is provided, reuse the latest valid meta_new_object run.
    - Otherwise ALWAYS run run_meta_new_object.py to generate a fresh random Snew,
      then take the latest valid meta_new_object run.
    """
    if explicit_run is not None:
        run_dir = Path(explicit_run)
        if not run_dir.exists():
            raise FileNotFoundError(f"--new-object-run does not exist: {run_dir}")
        if not _is_valid_new_object_run(run_dir):
            raise FileNotFoundError(f"Invalid meta_new_object run: {run_dir}")
        return run_dir

    if reuse_new_object:
        latest = _latest_run_dir(meta_new_root)
        if latest is not None and _is_valid_new_object_run(latest):
            return latest

    # Ensure upstream train exists first
    _get_or_create_train_run(
        project_root=project_root,
        explicit_run=None,
        config_path=config_path,
        train_root=train_root,
    )

    # Create a fresh new object
    _run_script(project_root, "run_meta_new_object.py", config_path)

    latest = _latest_run_dir(meta_new_root)
    if latest is None or not _is_valid_new_object_run(latest):
        raise RuntimeError("run_meta_new_object.py was executed, but no valid new-object run was created.")
    return latest


def main() -> None:
    args = _parse_args()
    cfg = load_default_config(args.config)

    project_root = Path(__file__).resolve().parent.parent

    train_root = Path(cfg.output.root_dir) / cfg.output.runs_dir / "train"
    meta_root = Path(cfg.output.root_dir) / cfg.output.runs_dir / "meta_prep"
    meta_new_root = Path(cfg.output.root_dir) / cfg.output.runs_dir / "meta_new_object"

    # -----------------------------
    # 1) Load or auto-create every needed stage
    # -----------------------------
    train_run_dir = _get_or_create_train_run(
        project_root=project_root,
        explicit_run=args.train_run,
        config_path=args.config,
        train_root=train_root,
    )

    meta_prep_run_dir = _get_or_create_meta_prep_run(
        project_root=project_root,
        explicit_run=args.meta_prep_run,
        config_path=args.config,
        train_root=train_root,
        meta_root=meta_root,
    )

    meta_new_run_dir = _get_or_create_new_object_run(
        project_root=project_root,
        explicit_run=args.new_object_run,
        config_path=args.config,
        train_root=train_root,
        meta_new_root=meta_new_root,
        reuse_new_object=args.reuse_new_object,
    )

    # -----------------------------
    # 2) Load training + Snew
    # -----------------------------
    meta_train_path = meta_prep_run_dir / "meta_train.csv"
    if not meta_train_path.exists():
        raise SystemExit(f"Missing file: {meta_train_path}")

    A, D, y, headers, row_ids = _load_meta_train_csv(meta_train_path)
    a_new, snew_headers, snew_payload = _load_snew_binary(meta_new_run_dir)

    if a_new.shape[0] != A.shape[1]:
        raise SystemExit(
            f"Snew length mismatch: got {a_new.shape[0]}, expected {A.shape[1]} based on meta_train.csv"
        )

    # -----------------------------
    # 3) Run META prediction
    # -----------------------------
    clf = MetaClassifier(
        k1_label=int(cfg.hag.k1_label),
        k2_label=int(cfg.hag.k2_label),
    ).fit(A=A, D=D, y=y, row_ids=row_ids)

    res = clf.predict(a_new, return_debug=True)

    print("META predict demo")
    print("Train run:", train_run_dir)
    print("META prep run:", meta_prep_run_dir)
    print("meta_train.csv:", meta_train_path)
    print("META new-object run:", meta_new_run_dir)
    print("a_new (binary):", a_new.tolist())

    if res.debug is not None:
        for j in sorted(res.debug.b1_history.keys()):
            b1 = res.debug.b1_history[j]
            b2 = res.debug.b2_history[j]
            if j == 0:
                print(f"Step 1 (j=0): |B1|={len(b1)}  |B2|={len(b2)}  B1={b1}  B2={b2}")
            else:
                print(f"Step 2 (j={j}): |B1|={len(b1)}  |B2|={len(b2)}  B1={b1}  B2={b2}")

    d = res.decision
    print("\nStep 4 decision")
    print("|K1|=", d.k1_size, "|K2|=", d.k2_size)
    print("|B1|=", d.b1_size, "|B2|=", d.b2_size)
    print("score1=|B1|/|K1|=", d.score1)
    print("score2=|B2|/|K2|=", d.score2)
    print("predicted_label:", d.predicted_label)

    # -----------------------------
    # 4) Save GUI-friendly predict artifacts
    # -----------------------------
    predict_root = Path(cfg.output.root_dir) / cfg.output.runs_dir / "predict"
    predict_run_dir = _ensure_dir(predict_root / _make_run_id())

    artifacts_ref = {
        "train_run_dir": str(train_run_dir),
        "meta_prep_run_dir": str(meta_prep_run_dir),
        "meta_train_csv": str(meta_train_path),
        "meta_new_object_run_dir": str(meta_new_run_dir),
        "snew_json": str(meta_new_run_dir / "snew.json") if (meta_new_run_dir / "snew.json").exists() else None,
        "snew_binary_csv": str(meta_new_run_dir / "snew_binary.csv"),
        "reuse_new_object": bool(args.reuse_new_object),
    }
    (predict_run_dir / "artifacts_ref.json").write_text(
        json.dumps(artifacts_ref, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    debug_steps: List[Dict[str, Any]] = []
    if res.debug is not None:
        for j in sorted(res.debug.b1_history.keys()):
            debug_steps.append(
                {
                    "j": int(j),
                    "b1_ids": [int(x) for x in res.debug.b1_history[j]],
                    "b2_ids": [int(x) for x in res.debug.b2_history[j]],
                    "b1_size": int(len(res.debug.b1_history[j])),
                    "b2_size": int(len(res.debug.b2_history[j])),
                }
            )

    meta_debug = {
        "a_new_headers": snew_headers,
        "a_new_binary": [int(x) for x in a_new.tolist()],
        "steps": debug_steps,
        "decision": {
            "predicted_label": int(d.predicted_label),
            "score1": float(d.score1),
            "score2": float(d.score2),
            "b1_size": int(d.b1_size),
            "b2_size": int(d.b2_size),
            "k1_size": int(d.k1_size),
            "k2_size": int(d.k2_size),
        },
    }
    (predict_run_dir / "meta_debug.json").write_text(
        json.dumps(meta_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prediction_payload = {
        "predicted_label": int(d.predicted_label),
        "a_new_binary": [int(x) for x in a_new.tolist()],
    }
    (predict_run_dir / "prediction.json").write_text(
        json.dumps(prediction_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved predict artifacts:")
    print("Predict run folder:", predict_run_dir)
    print("artifacts_ref.json:", predict_run_dir / "artifacts_ref.json")
    print("meta_debug.json:", predict_run_dir / "meta_debug.json")
    print("prediction.json:", predict_run_dir / "prediction.json")
    print("\nReuse the latest S(new) [FLAG]: 'python scripts/run_meta_prediction.py --reuse-new-object'")


if __name__ == "__main__":
    main()