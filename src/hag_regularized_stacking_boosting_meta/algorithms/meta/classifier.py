"""META Algorithm (Algorithm-2) classifier wrapper.

This is a thin OOP facade around:
  - filtering (Steps 1..p)
  - decision (Step 4)

It is intentionally IO-free and GUI-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .decision import MetaDecisionResult, decide_class
from .filtering import MetaFilteringDebug, run_filtering


@dataclass(frozen=True)
class MetaPredictResult:
    predicted_label: int
    decision: MetaDecisionResult
    debug: MetaFilteringDebug | None = None


def _ensure_2d_int(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    return x.astype(int)


def _ensure_1d_int(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return x.astype(int)


class MetaClassifier:
    """Implements META prediction given a prepared training table."""

    def __init__(self, *, k1_label: int = 1, k2_label: int = 2):
        self.k1_label = int(k1_label)
        self.k2_label = int(k2_label)

        self._A: np.ndarray | None = None  # (n, p+1)
        self._D: np.ndarray | None = None  # (n, p)
        self._y: np.ndarray | None = None  # (n,)
        self._row_ids: np.ndarray | None = None  # optional stable ids (Sid, original object ids)

    @property
    def is_fit(self) -> bool:
        return self._A is not None and self._D is not None and self._y is not None

    def fit(
        self,
        *,
        A: np.ndarray,
        D: np.ndarray,
        y: np.ndarray,
        row_ids: np.ndarray | None = None,
    ) -> "MetaClassifier":
        """Stores training arrays (already prepared by META prep stage)."""
        A2 = _ensure_2d_int(A, "A")
        D2 = np.asarray(D, dtype=float)
        y1 = _ensure_1d_int(y, "y")

        if A2.shape[0] != D2.shape[0] or A2.shape[0] != y1.shape[0]:
            raise ValueError("A, D, y must have same number of rows")
        if D2.ndim != 2:
            raise ValueError("D must be 2-D")
        if D2.shape[1] != (A2.shape[1] - 1):
            raise ValueError("D must have p columns where p = A.shape[1]-1")

        if row_ids is not None:
            rid = np.asarray(row_ids, dtype=int)
            if rid.ndim != 1 or rid.shape[0] != A2.shape[0]:
                raise ValueError("row_ids must be 1-D with same length as A rows")
            self._row_ids = rid
        else:
            self._row_ids = None

        self._A = A2
        self._D = D2
        self._y = y1
        return self

    def predict(self, a_new: Sequence[int], *, return_debug: bool = True) -> MetaPredictResult:
        """Predict class for one new object.

        Args:
          a_new: length (p+1) vector, same order as training A columns.
        """
        if not self.is_fit:
            raise RuntimeError("MetaClassifier must be fit() before predict()")

        assert self._A is not None and self._D is not None and self._y is not None

        b1, b2, dbg = run_filtering(
            self._A,
            self._D,
            self._y,
            a_new,
            k1_label=self.k1_label,
            k2_label=self.k2_label,
            return_debug=return_debug,
            row_ids=self._row_ids.tolist() if self._row_ids is not None else None,
        )

        k1_size = int((self._y == self.k1_label).sum())
        k2_size = int((self._y == self.k2_label).sum())

        decision = decide_class(
            b1_size=int(len(b1)),
            b2_size=int(len(b2)),
            k1_size=k1_size,
            k2_size=k2_size,
            k1_label=self.k1_label,
            k2_label=self.k2_label,
        )

        return MetaPredictResult(predicted_label=decision.predicted_label, decision=decision, debug=dbg)

    def predict_batch(self, A_new: np.ndarray) -> List[MetaPredictResult]:
        A_new = _ensure_2d_int(A_new, "A_new")
        out: List[MetaPredictResult] = []
        for i in range(A_new.shape[0]):
            out.append(self.predict(A_new[i, :], return_debug=False))
        return out