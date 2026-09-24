from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import csv
import numpy as np


# =========================
# GUI-friendly exceptions
# =========================

class DatasetLoadError(Exception):
    """
    A clean error type that GUI can catch and display nicely.
    Use this in PyQt controllers:
        try:
            ds = load_dataset_bundle(cfg)
        except DatasetLoadError as e:
            QMessageBox.critical(self, "Dataset load failed", str(e))
    """
    pass


# =========================
# Public dataclasses
# =========================

@dataclass(frozen=True)
class DatasetConfig:
    """
    Future-proof dataset config.

    IMPORTANT:
      - You can load ANY dataset file by changing cfg.path.
      - No code changes required to read a different path.
      - GUI later can choose any file path and build DatasetConfig dynamically.

    Default behavior:
      - If format=None or "auto", we auto-detect from the file extension.
      - CSV is default input type.
      - Supports your EXTENDED layout (same for .csv and .dat):
          row 1:  m,n,c metadata (e.g., 270,13,2)
          rows 2..m+1: n features + label
          last row: n flags (1 quantitative, 0 nominal)
      - .dat files are whitespace-separated (spaces or tabs, decimal point or comma);
        `delimiter` applies to CSV formats only.
    """
    path: str
    format: Optional[str] = None               # None/"auto", "csv_extended", "csv_simple", "dat_matrix", "xlsx", "txt_space"
    delimiter: str = ","                       # CSV only
    has_metadata_header: bool = True           # extended CSV
    has_feature_type_row: bool = True          # extended CSV
    label_col: int = -1                        # -1 means last column
    label_mapping: Optional[Dict[Union[str, int], int]] = None


@dataclass(frozen=True)
class LoadedDataset:
    """Stable dataset bundle used by algorithms and GUI."""
    X: np.ndarray                               # (m, n)
    y: np.ndarray                               # (m,)
    meta: Dict[str, int]                        # {"m":..., "n":..., "c":...} if present
    feature_types: np.ndarray                   # (n,) ints {0,1}; 1=quantitative, 0=nominal
    quantitative_idx: List[int]                 # indices where feature_types==1
    nominal_idx: List[int]                      # indices where feature_types==0


# =========================
# Loader registry (GUI-ready)
# =========================

LoaderFn = Callable[[DatasetConfig], LoadedDataset]


class LoaderRegistry:
    """Registry of loaders so we can extend formats without touching GUI."""
    def __init__(self) -> None:
        self._loaders: Dict[str, LoaderFn] = {}

    def register(self, name: str, fn: LoaderFn) -> None:
        self._loaders[name.lower()] = fn

    def get(self, name: str) -> LoaderFn:
        key = name.lower()
        if key not in self._loaders:
            available = ", ".join(sorted(self._loaders.keys()))
            raise DatasetLoadError(f"Unknown dataset format '{name}'. Available: {available}")
        return self._loaders[key]

    def available(self) -> List[str]:
        return sorted(self._loaders.keys())


_REGISTRY = LoaderRegistry()


def available_formats() -> List[str]:
    """Useful for GUI dropdown."""
    return _REGISTRY.available()


# =========================
# Public entrypoint (GUI calls this)
# =========================

def load_dataset_bundle(cfg: DatasetConfig) -> LoadedDataset:
    """
    Single stable API:
      - GUI can call this
      - Scripts can call this
      - Supports switching dataset path without code changes

    Example (script):
        cfg.path = "datasets/raw/default.csv"

    Example (GUI):
        cfg.path = file_dialog_selected_path
    """
    path = Path(cfg.path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path.resolve()}")

    # 1) Decide format
    fmt = (cfg.format or "auto").lower()
    if fmt == "auto":
        fmt = detect_format(path, cfg)
    _check_format_matches_suffix(fmt, path)

    # 2) Dispatch
    loader = _REGISTRY.get(fmt)
    return loader(cfg)


# =========================
# Format detection (CSV remains default)
# =========================

def detect_format(path: Path, cfg: DatasetConfig) -> str:
    """
    Detection rules (CSV default):
      - If suffix is .csv -> decide between csv_extended vs csv_simple by reading first row
      - If suffix is .dat -> dat_matrix (whitespace-separated; honors has_metadata_header/has_feature_type_row)
      - If suffix is .xlsx -> xlsx (future)
      - If suffix is .txt -> txt_space (future)
      - Else -> csv_extended fallback (because you want CSV default)
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Auto-detect extended vs simple
        return "csv_extended" if looks_like_csv_extended(path, cfg.delimiter) else "csv_simple"

    if suffix == ".dat":
        return "dat_matrix"

    if suffix in (".xlsx", ".xls"):
        return "xlsx"        # placeholder

    if suffix == ".txt":
        return "txt_space"   # placeholder

    # fallback to CSV default behavior
    return "csv_extended"


def _check_format_matches_suffix(fmt: str, path: Path) -> None:
    """
    Catch the easy mistake of switching cfg.path between .csv and .dat
    while leaving an explicit format for the other file type.
    """
    suffix = path.suffix.lower()
    if (fmt.startswith("csv_") and suffix == ".dat") or (fmt == "dat_matrix" and suffix == ".csv"):
        raise DatasetLoadError(
            f"Dataset format '{fmt}' cannot read a '{suffix}' file: {path}. "
            "Set format to 'auto' (detect from the file extension) or to the matching format."
        )


def looks_like_csv_extended(path: Path, delimiter: str = ",") -> bool:
    """
    Heuristic:
      - first non-empty row has at least 3 numeric values (m,n,c)
      - m,n,c are positive
    """
    try:
        rows = _read_csv_rows(path, delimiter)
        if not rows:
            return False
        first = rows[0]
        first_three = [x for x in first[:3] if str(x).strip() != ""]
        if len(first_three) < 3:
            return False
        m = int(float(first_three[0]))
        n = int(float(first_three[1]))
        c = int(float(first_three[2]))
        return (m > 0) and (n > 0) and (c > 0)
    except Exception:
        return False


# =========================
# Built-in loaders (keep here now; later can move into io/formats/*)
# =========================

def load_csv_simple_cfg(cfg: DatasetConfig) -> LoadedDataset:
    X, y = load_csv_simple(
        path=Path(cfg.path),
        delimiter=cfg.delimiter,
        label_col=cfg.label_col,
        label_mapping=_coerce_label_mapping(cfg.label_mapping),
    )
    n = X.shape[1]
    feature_types = np.ones((n,), dtype=int)   # all quantitative by default
    return _bundle(X, y, meta={}, feature_types=feature_types)


def load_csv_extended_cfg(cfg: DatasetConfig) -> LoadedDataset:
    return load_csv_extended(
        path=Path(cfg.path),
        delimiter=cfg.delimiter,
        has_metadata_header=cfg.has_metadata_header,
        has_feature_type_row=cfg.has_feature_type_row,
        label_col=cfg.label_col,
        label_mapping=_coerce_label_mapping(cfg.label_mapping),
    )


def load_dat_matrix_cfg(cfg: DatasetConfig) -> LoadedDataset:
    return load_dat_matrix(
        path=Path(cfg.path),
        has_metadata_header=cfg.has_metadata_header,
        has_feature_type_row=cfg.has_feature_type_row,
        label_col=cfg.label_col,
        label_mapping=_coerce_label_mapping(cfg.label_mapping),
    )


def load_xlsx_cfg(cfg: DatasetConfig) -> LoadedDataset:
    raise DatasetLoadError(
        "Format 'xlsx' is registered but not implemented yet. "
        "Add io/formats/excel_xlsx.py later."
    )


def load_txt_space_cfg(cfg: DatasetConfig) -> LoadedDataset:
    raise DatasetLoadError(
        "Format 'txt_space' is registered but not implemented yet. "
        "Add io/formats/txt_space.py later."
    )


# Register built-ins (CSV default is active immediately)
_REGISTRY.register("csv_simple", load_csv_simple_cfg)
_REGISTRY.register("csv_extended", load_csv_extended_cfg)
_REGISTRY.register("dat_matrix", load_dat_matrix_cfg)

# Future placeholders already registered so GUI can show them:
_REGISTRY.register("xlsx", load_xlsx_cfg)
_REGISTRY.register("txt_space", load_txt_space_cfg)


# =========================
# Core CSV implementations
# =========================

def load_csv_simple(
    path: Path,
    delimiter: str = ",",
    label_col: int = -1,
    label_mapping: Optional[Dict[Union[str, int], int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple CSV: each row = features + label (no meta header, no type footer)."""
    rows = _read_csv_rows(path, delimiter)
    if not rows:
        raise DatasetLoadError(f"Empty CSV: {path}")

    data = _rows_to_float_matrix(rows)

    if data.shape[1] < 2:
        raise DatasetLoadError("CSV must contain at least 1 feature column and 1 label column.")

    lbl_idx = label_col if label_col != -1 else data.shape[1] - 1
    X = data[:, :lbl_idx] if lbl_idx == data.shape[1] - 1 else np.delete(data, lbl_idx, axis=1)
    y_raw = data[:, lbl_idx].astype(int)

    y = _apply_label_mapping(y_raw, label_mapping).astype(int)
    return X.astype(float), y


def load_csv_extended(
    path: Path,
    delimiter: str = ",",
    has_metadata_header: bool = True,
    has_feature_type_row: bool = True,
    label_col: int = -1,
    label_mapping: Optional[Dict[Union[str, int], int]] = None,
) -> LoadedDataset:
    """
    Extended CSV:
      row 1: m,n,c
      rows: m data rows (n features + label)
      last row: n flags (1 quantitative, 0 nominal)
    """
    rows = _read_csv_rows(path, delimiter)
    if not rows:
        raise DatasetLoadError(f"Empty CSV: {path}")

    return _parse_extended_rows(
        rows,
        has_metadata_header=has_metadata_header,
        has_feature_type_row=has_feature_type_row,
        label_col=label_col,
        label_mapping=label_mapping,
    )


# =========================
# Core DAT implementation
# =========================

def load_dat_matrix(
    path: Path,
    has_metadata_header: bool = True,
    has_feature_type_row: bool = True,
    label_col: int = -1,
    label_mapping: Optional[Dict[Union[str, int], int]] = None,
) -> LoadedDataset:
    """
    DAT matrix: same layout as extended CSV, but whitespace-separated:
      row 1: m n c
      rows: m data rows (n features + label)
      last row: n flags (1 quantitative, 0 nominal)

    Accepts spaces or tabs (runs of them count as one separator, padding is ignored)
    and decimal commas (e.g. "-0,91" as written by some locales).
    """
    rows = read_dat_rows(path)
    if not rows:
        raise DatasetLoadError(f"Empty DAT: {path}")

    return _parse_extended_rows(
        rows,
        has_metadata_header=has_metadata_header,
        has_feature_type_row=has_feature_type_row,
        label_col=label_col,
        label_mapping=label_mapping,
    )


# =========================
# Shared layout parser (CSV + DAT)
# =========================

def _parse_extended_rows(
    rows: List[List[str]],
    has_metadata_header: bool,
    has_feature_type_row: bool,
    label_col: int,
    label_mapping: Optional[Dict[Union[str, int], int]],
) -> LoadedDataset:
    """Tokenized rows (header?, data rows, feature-type row?) -> LoadedDataset."""
    meta: Dict[str, int] = {}
    start_idx = 0

    # ---- metadata header ----
    if has_metadata_header:
        header = rows[0]
        first_three = [x for x in header[:3] if str(x).strip() != ""]
        if len(first_three) < 3:
            raise DatasetLoadError(f"Metadata header must contain m,n,c in first row. Got: {rows[0]}")

        try:
            m = int(float(first_three[0]))
            n = int(float(first_three[1]))
            c = int(float(first_three[2]))
        except Exception as e:
            raise DatasetLoadError(f"Failed parsing m,n,c from: {rows[0]}") from e

        meta = {"m": m, "n": n, "c": c}
        start_idx = 1

    # ---- feature type footer ----
    feature_types: Optional[np.ndarray] = None
    end_idx = len(rows)

    if has_feature_type_row:
        footer = rows[-1]
        footer_clean = [x for x in footer if str(x).strip() != ""]

        if "n" in meta:
            n = meta["n"]
            if len(footer_clean) < n:
                raise DatasetLoadError(
                    f"Feature-type row must contain at least n={n} flags. Got {len(footer_clean)}: {rows[-1]}"
                )
            flags = footer_clean[:n]
        else:
            flags = footer_clean

        try:
            ft = np.array([int(float(x)) for x in flags], dtype=int)
        except Exception as e:
            raise DatasetLoadError(f"Feature-type row must contain 0/1 flags. Got: {rows[-1]}") from e

        if not np.all(np.isin(ft, [0, 1])):
            raise DatasetLoadError(f"Feature-type flags must be only 0/1. Got: {ft.tolist()}")

        feature_types = ft
        end_idx = len(rows) - 1

    # ---- data rows ----
    data_rows = rows[start_idx:end_idx]

    if "m" in meta:
        m = meta["m"]
        if len(data_rows) < m:
            raise DatasetLoadError(f"Header says m={m} but only {len(data_rows)} data rows exist.")
        data_rows = data_rows[:m]  # ignore extra safely

    data = _rows_to_float_matrix(data_rows)

    if data.shape[1] < 2:
        raise DatasetLoadError("Data rows must contain at least 1 feature and 1 label column.")

    lbl_idx = label_col if label_col != -1 else data.shape[1] - 1
    X = data[:, :lbl_idx] if lbl_idx == data.shape[1] - 1 else np.delete(data, lbl_idx, axis=1)
    y_raw = data[:, lbl_idx].astype(int)
    y = _apply_label_mapping(y_raw, label_mapping).astype(int)

    if "n" in meta:
        n = meta["n"]
        if X.shape[1] != n:
            raise DatasetLoadError(f"Header says n={n}, but X has {X.shape[1]} features.")

    if feature_types is None:
        feature_types = np.ones((X.shape[1],), dtype=int)

    if feature_types.shape[0] != X.shape[1]:
        raise DatasetLoadError(
            f"Feature-type row length {feature_types.shape[0]} must match feature count {X.shape[1]}."
        )

    return _bundle(X, y, meta=meta, feature_types=feature_types)


# =========================
# Helpers
# =========================

def _bundle(X: np.ndarray, y: np.ndarray, meta: Dict[str, int], feature_types: np.ndarray) -> LoadedDataset:
    feature_types = feature_types.astype(int)
    quantitative_idx = [i for i, t in enumerate(feature_types.tolist()) if t == 1]
    nominal_idx = [i for i, t in enumerate(feature_types.tolist()) if t == 0]
    return LoadedDataset(
        X=X.astype(float),
        y=y.astype(int),
        meta=meta,
        feature_types=feature_types,
        quantitative_idx=quantitative_idx,
        nominal_idx=nominal_idx,
    )


def _coerce_label_mapping(mapping: Optional[Dict[Union[str, int], int]]) -> Dict[Union[str, int], int]:
    if not mapping:
        return {}
    out: Dict[Union[str, int], int] = {}
    for k, v in mapping.items():
        try:
            ik = int(k)
            out[ik] = int(v)
            out[str(ik)] = int(v)
        except Exception:
            out[str(k)] = int(v)
    return out


def _read_csv_rows(path: Path, delimiter: str) -> List[List[str]]:
    rows: List[List[str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row or all(str(x).strip() == "" for x in row):
                    continue
                rows.append(row)
    except Exception as e:
        raise DatasetLoadError(f"Failed reading CSV: {path}") from e
    return rows


def read_dat_rows(path: Path) -> List[List[str]]:
    """Whitespace-tokenized non-empty lines of a .dat file (decimal commas -> points)."""
    rows: List[List[str]] = []
    try:
        # utf-8-sig: tolerate a BOM written by Windows editors
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                tokens = line.split()
                if tokens:
                    rows.append([_dat_token(t) for t in tokens])
    except Exception as e:
        raise DatasetLoadError(f"Failed reading DAT: {path}") from e
    return rows


def _dat_token(token: str) -> str:
    # Decimal comma ("-0,91") -> decimal point; anything else is left for float() to judge
    return token.replace(",", ".") if token.count(",") == 1 and "." not in token else token


def _rows_to_float_matrix(rows: List[List[str]]) -> np.ndarray:
    cleaned = [[str(x).strip() for x in row] for row in rows]
    width = max(len(r) for r in cleaned)
    padded = [r + [""] * (width - len(r)) for r in cleaned]

    numeric_rows: List[List[float]] = []
    for r in padded:
        while r and r[-1] == "":
            r = r[:-1]

        if any(x == "" for x in r):
            raise DatasetLoadError(f"Empty cell inside a data row (not allowed): {r}")

        try:
            numeric_rows.append([float(x) for x in r])
        except Exception as e:
            raise DatasetLoadError(f"Non-numeric value in data row: {r}") from e

    expected = len(numeric_rows[0])
    for i, r in enumerate(numeric_rows, start=1):
        if len(r) != expected:
            raise DatasetLoadError(
                f"Data row {i} has {len(r)} values, but data row 1 has {expected} (missing or extra value?)."
            )

    return np.array(numeric_rows, dtype=float)


def _apply_label_mapping(y: np.ndarray, mapping: Optional[Dict[Union[str, int], int]]) -> np.ndarray:
    if not mapping:
        return y
    out = []
    for val in y.tolist():
        mapped = mapping.get(val, mapping.get(str(val), val))
        out.append(int(mapped))
    return np.array(out, dtype=int)




