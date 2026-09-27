from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
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
    so each Step-4 output becomes one latent feature column.

    If |TUPLAM|=k, then Step-4 is executed (k-1) times => p=k-1.
    """
    if not R_step4_history:
        return LatentBuildResult(dij=np.zeros((0, 0), dtype=float), p=0)

    # Stack columns: (m, p)
    dij = np.column_stack([r.astype(float) for r in R_step4_history])
    return LatentBuildResult(dij=dij, p=dij.shape[1])
