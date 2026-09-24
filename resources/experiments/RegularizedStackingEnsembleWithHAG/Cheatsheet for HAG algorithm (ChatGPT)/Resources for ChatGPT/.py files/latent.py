from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class LatentBuildResult:
    # dij matrix: shape (m, p) where p = |TUPLAM|-1
    dij: np.ndarray
    p: int


def build_dij_from_R_history(R_step4_history: Sequence[np.ndarray]) -> LatentBuildResult:
    """
    Paper statement:
      "Множество значений {R(St)} ... полученное на шаге 4 алгоритма,
       формируют дополнительные (латентные) признаки..."
    so each Step-4 output becomes one latent feature column. :contentReference[oaicite:7]{index=7}

    If |TUPLAM|=k, then Step-4 is executed (k-1) times => p=k-1.
    """
    if not R_step4_history:
        return LatentBuildResult(dij=np.zeros((0, 0), dtype=float), p=0)

    # Stack columns: (m, p)
    dij = np.column_stack([r.astype(float) for r in R_step4_history])
    return LatentBuildResult(dij=dij, p=dij.shape[1])


def build_meta_matrix(
    X: np.ndarray,
    tuplam: Sequence[int],
    dij: np.ndarray,
) -> np.ndarray:
    """
    Creates matrix for META stage:
      [ y0..yp | r1..rp ]
    where y0..yp are selected original features from TUPLAM
    and r1..rp are latent columns dij.
    """
    if len(tuplam) == 0:
        raise ValueError("tuplam must not be empty")
    base = X[:, list(tuplam)]
    if dij.size == 0:
        return base
    if dij.shape[0] != base.shape[0]:
        raise ValueError("dij row count must match X row count")
    return np.hstack([base, dij])
