from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import (
    QuantitativePipelineResult,
    build_quantitative_nominalization,
)


@dataclass(frozen=True)
class MetaPrepResult:
    """
    Outputs for META training dataset preparation:

    - A: (m, k) ai0..aip values (from original dataset X, but quant features use binary 1/2)
    - D: (m, p) dij latent features (r1..rp), float
    - S: (m, k+p) merged matrix [A | D]
    - y: (m,) class labels
    - headers: column names for CSV
    """
    A: np.ndarray
    D: np.ndarray
    S: np.ndarray
    y: np.ndarray
    headers: List[str]


def _as_int_safely(col: np.ndarray) -> np.ndarray:
    """
    Convert to int when values are essentially integers.
    This keeps nominal columns clean (like Excel tables).
    """
    col = np.asarray(col).reshape(-1)
    if np.allclose(col, np.round(col), atol=1e-9):
        return np.round(col).astype(int)
    return col.astype(float)


def _build_quant_binary_map(
    X: np.ndarray,
    y: np.ndarray,
    quantitative_idx: Sequence[int],
    quantitative_result: Optional[QuantitativePipelineResult] = None,
) -> Dict[int, np.ndarray]:
    """
    Builds a dict:
      feature_index -> binary vector (values {1,2})
    using the same quantitative pipeline already in the project.
    """
    if not quantitative_idx:
        return {}

    q_res = quantitative_result
    if q_res is None:
        q_res = build_quantitative_nominalization(X, y, list(quantitative_idx))

    qmap: Dict[int, np.ndarray] = {}
    for j, fidx in enumerate(q_res.quantitative_idx):
        qmap[int(fidx)] = q_res.binary_X[:, j].astype(int)
    return qmap


def prepare_meta_training_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_types: np.ndarray,
    tuplam: Sequence[int],
    dij: np.ndarray,
    quantitative_result: Optional[QuantitativePipelineResult] = None,
) -> MetaPrepResult:
    """
    PREPARATION FOR META ALGORITHM (Excel-faithful data, but CODE-STYLE HEADERS):

    Given:
      - X: original dataset features (NOT contribution table)
      - feature_types: 0 nominal, 1 quantitative
      - tuplam: ordered selected feature indices (0-based) => ai0..aip
      - dij: latent features from HAG Step-4 outputs (r1..rp) => di1..dip
      - y: class labels

    Build:
      - A: columns from X by tuplam order, BUT:
            if feature is quantitative -> use binary nominalized values {1,2}
            else -> use original values
      - D: dij (float)
      - S: [A | D]
      - headers (IMPORTANT CHANGE REQUESTED):
          ai0(x2), ai1(x5), ...  where x<index> is 0-based (NO +1)
          di1(r1), di2(r2), ...
          Class

    quantitative_result: optional precomputed build_quantitative_nominalization
    output for all quantitative features of X (avoids recomputing it).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    feature_types = np.asarray(feature_types, dtype=int).reshape(-1)
    tuplam = [int(i) for i in tuplam]

    if X.ndim != 2:
        raise ValueError("X must be 2D (m,n)")
    m, n = X.shape
    if y.shape[0] != m:
        raise ValueError("y length must match X rows")
    if feature_types.shape[0] != n:
        raise ValueError("feature_types length must match X columns")
    if len(tuplam) == 0:
        raise ValueError("tuplam must not be empty")

    for idx in tuplam:
        if idx < 0 or idx >= n:
            raise ValueError(f"tuplam index out of range: {idx} for n={n}")

    quantitative_idx = [i for i in range(n) if feature_types[i] == 1]
    qmap = _build_quant_binary_map(X, y, quantitative_idx, quantitative_result)

    # ---- Build A (ai0..aip) in tuplam order ----
    A_cols: List[np.ndarray] = []
    headers: List[str] = []

    for j, fidx in enumerate(tuplam):
        if feature_types[fidx] == 1:
            if fidx not in qmap:
                raise RuntimeError(
                    f"Quantitative feature {fidx} not found in binary map. "
                    "Check feature_types and quantitative pipeline."
                )
            col = qmap[fidx]
        else:
            col = X[:, fidx]

        col = _as_int_safely(col)
        A_cols.append(col.reshape(-1, 1))

        # 0-based x index in label (NO +1)
        headers.append(f"ai{j}(x{fidx})")

    A = np.hstack(A_cols).astype(int)

    # ---- Build D (latent dij) ----
    D = np.asarray(dij, dtype=float)
    if D.size == 0:
        D = np.zeros((m, 0), dtype=float)
    if D.ndim != 2 or D.shape[0] != m:
        raise ValueError("dij must be 2D with same row count as X")

    p = D.shape[1]
    for k in range(p):
        headers.append(f"di{k+1}(r{k+1})")

    # ---- Merge S = [A | D] ----
    S = np.hstack([A.astype(float), D])

    headers.append("Class")

    return MetaPrepResult(A=A, D=D, S=S, y=y, headers=headers)