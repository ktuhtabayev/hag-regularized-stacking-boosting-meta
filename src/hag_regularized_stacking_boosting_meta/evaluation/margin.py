from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Margin1DResult:
    """
    Margin analysis for ONE latent feature column d (e.g., di1(r1)).
    Mirrors Excel:
      left boundary  = max over class K2
      right boundary = min over class K1
      midpoint       = (L + R)/2
      margin_width   = R - L
      object_margin  = IF(y==K1, +1, -1) * (d - midpoint)
      yhat           = IF(d > midpoint, K1, K2)
    """
    left_boundary: float
    right_boundary: float
    midpoint: float
    width: float
    yhat: np.ndarray
    object_margin: np.ndarray
    left_argmax_idx: int
    right_argmin_idx: int


def margin_analysis_1d(
    d: np.ndarray,
    y: np.ndarray,
    *,
    k1_label: int,
    k2_label: int,
) -> Margin1DResult:
    d = np.asarray(d, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if d.shape[0] != y.shape[0]:
        raise ValueError("d and y must have the same length")

    mask1 = (y == k1_label)
    mask2 = (y == k2_label)
    if not mask1.any() or not mask2.any():
        raise ValueError("Both classes must be present to compute margins")

    # Left boundary = max of class-2 points
    d2 = d[mask2]
    left_boundary = float(d2.max())
    left_argmax_local = int(np.argmax(d2))
    left_argmax_idx = int(np.where(mask2)[0][left_argmax_local])

    # Right boundary = min of class-1 points
    d1 = d[mask1]
    right_boundary = float(d1.min())
    right_argmin_local = int(np.argmin(d1))
    right_argmin_idx = int(np.where(mask1)[0][right_argmin_local])

    midpoint = 0.5 * (left_boundary + right_boundary)
    width = right_boundary - left_boundary

    # Excel: IF(Class==1,1,-1)*(d-mid)
    sign = np.where(y == k1_label, 1.0, -1.0)
    object_margin = sign * (d - midpoint)

    # Excel: IF(d > mid, 1, 2)
    yhat = np.where(d > midpoint, k1_label, k2_label).astype(int)

    return Margin1DResult(
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        midpoint=midpoint,
        width=width,
        yhat=yhat,
        object_margin=object_margin,
        left_argmax_idx=left_argmax_idx,
        right_argmin_idx=right_argmin_idx,
    )


def margin_analysis_latent_matrix(
    dij: np.ndarray,
    y: np.ndarray,
    *,
    k1_label: int,
    k2_label: int,
) -> List[Margin1DResult]:
    dij = np.asarray(dij, dtype=float)
    if dij.ndim != 2:
        raise ValueError("dij must be 2-D")

    results: List[Margin1DResult] = []
    for j in range(dij.shape[1]):
        results.append(margin_analysis_1d(dij[:, j], y, k1_label=k1_label, k2_label=k2_label))
    return results


def build_margin_report_rows(
    results: List[Margin1DResult],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for j, r in enumerate(results, start=1):
        rows.append(
            {
                "latent_index_1based": float(j),
                "left_boundary": float(r.left_boundary),
                "right_boundary": float(r.right_boundary),
                "midpoint": float(r.midpoint),
                "margin_width": float(r.width),
                "left_argmax_idx_0based": float(r.left_argmax_idx),
                "right_argmin_idx_0based": float(r.right_argmin_idx),
            }
        )
    return rows