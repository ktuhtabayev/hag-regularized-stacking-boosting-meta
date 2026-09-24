from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_to_jsonable)
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv_matrix(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray | None = None,
    delimiter: str = ",",
    header: list[str] | None = None,
) -> Path:
    """
    Writes numeric matrix to CSV.
    If y is provided, appends y as last column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y).reshape(-1, 1)
        out = np.concatenate([X, y], axis=1)
    else:
        out = X

    with path.open("w", encoding="utf-8", newline="") as f:
        if header:
            f.write(delimiter.join(header) + "\n")
        for row in out:
            f.write(delimiter.join(str(v) for v in row) + "\n")
    return path
