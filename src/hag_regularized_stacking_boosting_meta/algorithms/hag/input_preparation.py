from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .weights import build_quantitative_nominalization

# Nominal is optional (project must remain importable if nominal not implemented yet)
try:
    from .weights import build_nominal_contributions  # type: ignore
    _NOMINAL_AVAILABLE = True
except Exception:
    build_nominal_contributions = None  # type: ignore
    _NOMINAL_AVAILABLE = False


@dataclass(frozen=True)
class HAGPrepResult:
    """
    Outputs for HAG stage preparation.

    - X_contrib_full: merged contributions in ORIGINAL feature positions (shape m x n)
    - w_full: merged weights in ORIGINAL feature positions (length n)

    Sorting:
    - weight_sorted_feature_idx: feature indices sorted by weight DESC (ties: left→right)
    - weight_rank_per_feature: rank per original feature index (0 = biggest), matches Excel "rank row" (0-based)
    """
    X_contrib_full: np.ndarray
    w_full: np.ndarray
    weight_sorted_feature_idx: List[int]
    weight_rank_per_feature: List[int]

    # Backward/forward-compatible aliases (some scripts used older names)
    @property
    def weight_sorted_idx(self) -> List[int]:
        return self.weight_sorted_feature_idx

    @property
    def weight_rank_per_feature_index(self) -> List[int]:
        return self.weight_rank_per_feature


def _stable_sort_desc_with_left_to_right_ties(w: np.ndarray) -> np.ndarray:
    """
    Stable descending sort by weight.
    Ties keep original left→right order (index ascending).
    """
    w = np.asarray(w, dtype=float)
    # mergesort is stable -> preserves original order for ties
    return np.argsort(-w, kind="mergesort")


def prepare_hag_inputs(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: np.ndarray,
    quantitative_idx: Optional[List[int]] = None,
    nominal_idx: Optional[List[int]] = None,
    *,
    quantitative_result: Optional[Any] = None,
    nominal_result: Optional[Any] = None,
) -> HAGPrepResult:
    """
    Build:
      1) merged contribution dataset (m x n) aligned to original feature positions
      2) merged weights vector (n) aligned to original feature positions
      3) weights sorted indices (desc, stable ties)
      4) rank-per-feature array (0-based), matching Excel "rank row" concept

    feature_types convention:
      1 = quantitative
      0 = nominal

    quantitative_result / nominal_result: already computed
    build_quantitative_nominalization / build_nominal_contributions outputs for the
    same X, y and indices; passed in to avoid recomputing them.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    feature_types = np.asarray(feature_types).astype(int)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if feature_types.ndim != 1 or feature_types.shape[0] != X.shape[1]:
        raise ValueError(
            f"feature_types must be 1D of length n={X.shape[1]}, got shape={feature_types.shape}"
        )

    m, n = X.shape

    # Derive indices if not provided
    if quantitative_idx is None:
        quantitative_idx = [int(i) for i in np.where(feature_types == 1)[0].tolist()]
    if nominal_idx is None:
        nominal_idx = [int(i) for i in np.where(feature_types == 0)[0].tolist()]

    # Output containers (full size, aligned to ORIGINAL indices)
    X_contrib_full = np.zeros((m, n), dtype=float)
    w_full = np.zeros((n,), dtype=float)

    # -------------------------
    # Quantitative block
    # -------------------------
    if quantitative_idx:
        qres = quantitative_result
        if qres is None:
            qres = build_quantitative_nominalization(X, y, quantitative_idx)

        # contribution_X is (m, len(qidx)) in the SAME ORDER as qres.quantitative_idx
        for j, fidx in enumerate(qres.quantitative_idx):
            X_contrib_full[:, fidx] = qres.contribution_X[:, j]

        # weights_wc is dict[fidx] -> weight
        for fidx in qres.quantitative_idx:
            w_full[fidx] = float(qres.weights_wc[fidx])

    # -------------------------
    # Nominal block
    # -------------------------
    if nominal_idx:
        if not _NOMINAL_AVAILABLE or build_nominal_contributions is None:
            raise RuntimeError(
                "Nominal features exist in dataset, but weights_nominal.py API is not available.\n"
                "Create algorithms/hag/weights_nominal.py with build_nominal_contributions(), "
                "or run on a purely-quantitative dataset."
            )

        nres = nominal_result
        if nres is None:
            nres = build_nominal_contributions(X, y, nominal_idx)

        # contribution_X is (m, len(nidx)) in SAME ORDER as nres.nominal_idx
        for j, fidx in enumerate(nres.nominal_idx):
            X_contrib_full[:, fidx] = nres.contribution_X[:, j]

        # Nominal weights live in nres.w_c dict[fidx] -> weight
        for fidx in nres.nominal_idx:
            w_full[fidx] = float(nres.w_c[fidx])

    # -------------------------
    # Sorting / Ranking
    # -------------------------
    sorted_idx = _stable_sort_desc_with_left_to_right_ties(w_full)  # ndarray
    weight_sorted_feature_idx = [int(i) for i in sorted_idx.tolist()]

    # rank per feature index: rank[feature_index] = position in sorted list (0 = biggest)
    rank = np.empty((n,), dtype=int)
    rank[sorted_idx] = np.arange(n, dtype=int)
    weight_rank_per_feature = [int(r) for r in rank.tolist()]

    return HAGPrepResult(
        X_contrib_full=X_contrib_full,
        w_full=w_full,
        weight_sorted_feature_idx=weight_sorted_feature_idx,
        weight_rank_per_feature=weight_rank_per_feature,
    )
