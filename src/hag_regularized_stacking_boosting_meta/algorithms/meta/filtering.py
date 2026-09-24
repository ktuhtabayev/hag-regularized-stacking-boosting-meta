"""META Algorithm (Algorithm-2) filtering steps.

This module implements the set-filtering part used to classify a new object.

Notation aligned to Excel experiments (code is always 0-based indices):

Training rows:
  S_i = (a_i0, ..., a_ip, d_i1, ..., d_ip, Class)

New object:
  S_new = (a_0, ..., a_p)

Step 1 (j=0):
  B1(a0) = { S_i in K1 | a_0 == a_i0 }
  B2(a0) = { S_i in K2 | a_0 == a_i0 }

Step 2..p:
  For j = 1..p:
    B1(a_j) = { S_i in B1(a_{j-1}) | a_j == a_ij and d_ij > 0 }
    B2(a_j) = { S_i in B2(a_{j-1}) | a_j == a_ij and d_ij < 0 }

The sign constraint uses latent features d_ij (r_j) produced by HAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetaFilteringDebug:
    """Detailed trace for GUI/debug.

    Histories store object ids. If row_ids are provided to run_filtering(),
    those ids are used. Otherwise, row indices 0..n-1 are used.
    """

    b1_history: Dict[int, List[int]]
    b2_history: Dict[int, List[int]]


def _as_int_list(idx: np.ndarray) -> List[int]:
    return [int(i) for i in np.asarray(idx, dtype=int).tolist()]


def _map_ids(idx: np.ndarray, row_ids: np.ndarray | None) -> List[int]:
    idx = np.asarray(idx, dtype=int)
    if row_ids is None:
        return _as_int_list(idx)
    return [int(row_ids[i]) for i in idx.tolist()]


def build_b1_b2_step0(
    A: np.ndarray,
    y: np.ndarray,
    a0: int,
    *,
    k1_label: int,
    k2_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Step 1: build B1(a0), B2(a0) from full training set."""
    if A.ndim != 2:
        raise ValueError("A must be 2-D (n_samples, p+1)")
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if len(y) != A.shape[0]:
        raise ValueError("A and y must have same number of rows")

    a0_col = A[:, 0]
    b1 = np.where((y == k1_label) & (a0_col == a0))[0]
    b2 = np.where((y == k2_label) & (a0_col == a0))[0]
    return b1.astype(int), b2.astype(int)


def filter_step_j(
    A: np.ndarray,
    D: np.ndarray,
    a_j: int,
    *,
    j: int,
    b1_prev: np.ndarray,
    b2_prev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2..p: filter B1/B2 based on a_j match and latent sign constraint."""
    if j <= 0:
        raise ValueError("j must be >= 1 for filter_step_j")
    if A.ndim != 2 or D.ndim != 2:
        raise ValueError("A and D must be 2-D")
    if A.shape[0] != D.shape[0]:
        raise ValueError("A and D must have same number of rows")
    if j >= A.shape[1]:
        raise ValueError(f"j={j} out of range for A with {A.shape[1]} columns")
    if (j - 1) >= D.shape[1]:
        raise ValueError(f"j={j} out of range for D with {D.shape[1]} columns")

    col_a = A[:, j]
    col_d = D[:, j - 1]

    b1_new = b1_prev[(col_a[b1_prev] == a_j) & (col_d[b1_prev] > 0)]
    b2_new = b2_prev[(col_a[b2_prev] == a_j) & (col_d[b2_prev] < 0)]
    return b1_new.astype(int), b2_new.astype(int)


def run_filtering(
    A: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    a_new: Sequence[int],
    *,
    k1_label: int,
    k2_label: int,
    return_debug: bool = False,
    row_ids: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, MetaFilteringDebug | None]:
    """Runs META filtering Steps 1..p for a single new object."""
    a_new_arr = np.asarray(list(a_new), dtype=int)
    if a_new_arr.ndim != 1:
        raise ValueError("a_new must be 1-D")
    if A.shape[1] != a_new_arr.shape[0]:
        raise ValueError(
            f"a_new length must equal number of A columns: {a_new_arr.shape[0]} vs {A.shape[1]}"
        )
    if D.shape[1] != (A.shape[1] - 1):
        raise ValueError("D must have p columns where p = A.shape[1]-1")
    if y.shape[0] != A.shape[0] or D.shape[0] != A.shape[0]:
        raise ValueError("A, D, y must have same row count")

    row_ids_arr: np.ndarray | None = None
    if row_ids is not None:
        row_ids_arr = np.asarray(list(row_ids), dtype=int)
        if row_ids_arr.ndim != 1 or row_ids_arr.shape[0] != A.shape[0]:
            raise ValueError("row_ids must be 1-D with same length as A rows")

    b1_hist: Dict[int, List[int]] = {}
    b2_hist: Dict[int, List[int]] = {}

    # Step 1 (j=0)
    b1, b2 = build_b1_b2_step0(A, y, int(a_new_arr[0]), k1_label=k1_label, k2_label=k2_label)
    if return_debug:
        b1_hist[0] = _map_ids(b1, row_ids_arr)
        b2_hist[0] = _map_ids(b2, row_ids_arr)

    # Step 2..p
    p = D.shape[1]
    for j in range(1, p + 1):
        b1, b2 = filter_step_j(A, D, int(a_new_arr[j]), j=j, b1_prev=b1, b2_prev=b2)
        if return_debug:
            b1_hist[j] = _map_ids(b1, row_ids_arr)
            b2_hist[j] = _map_ids(b2, row_ids_arr)

    dbg = MetaFilteringDebug(b1_history=b1_hist, b2_history=b2_hist) if return_debug else None
    return b1, b2, dbg