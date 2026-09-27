from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Dataclasses (GUI-friendly + future-proof)
# ============================================================

@dataclass(frozen=True)
class Criterion1Result:
    """Result of Criterion-1 (Formula 2) for one quantitative feature."""
    weight_wc: float
    pi1: float
    pi2: float
    pi3: float


@dataclass(frozen=True)
class QuantFeatureBinarization:
    """Binarization info for one quantitative feature (Formula 3)."""
    gamma_c: float                     # Γ_c threshold
    b_value: float                     # nearest value to π2 from (π2; π3)
    binary_values: np.ndarray          # shape (m,), values {1,2}


@dataclass(frozen=True)
class GradationCounts:
    """Counts of gradations (1/2) in each class for a feature."""
    g_k1: Dict[int, int]               # e.g. {1: count, 2: count}
    g_k2: Dict[int, int]


@dataclass(frozen=True)
class ContributionResult:
    """Contributions η_c(j) (Formula 5) for j in {1,2}."""
    eta: Dict[int, float]              # e.g. {1: -0.52, 2: +0.52}


@dataclass(frozen=True)
class QuantitativePipelineResult:
    """
    Final output for quantitative features:
      - weights wc (ω)
      - thresholds Γ_c
      - binary nominalized dataset (1/2)
      - contribution dataset (η values)
      - IMPORTANT: original feature indices preserved (for later reunification)
    """
    quantitative_idx: List[int]                 # original indices in full dataset
    weights_wc: Dict[int, float]                # key = original feature index
    pi_table: Dict[int, Tuple[float, float, float]]   # (pi1, pi2, pi3)
    gamma: Dict[int, float]                     # Γ_c per feature index
    binary_X: np.ndarray                        # shape (m, q), values {1,2}
    contribution_X: np.ndarray                  # shape (m, q), float contributions
    contributions_eta: Dict[int, Dict[int, float]]    # feature_idx -> {1:...,2:...}


# ============================================================
# Core math: Criterion-1 (Formula 2)
# ============================================================

def criterion_1(column: List[float], target: List[int]) -> Criterion1Result:
    """
    Criterion-1 (Formula 2): returns weight ω_c and (π1, π2, π3).

    Input:
      column: feature values for all objects
      target: class labels for all objects (1 or 2)
    """
    if len(column) != len(target):
        raise ValueError("column and target must have same length")
    if len(column) == 0:
        raise ValueError("empty column")

    K1 = target.count(1)
    K2 = target.count(2)
    if K1 == 0 or K2 == 0:
        raise ValueError("Both classes must exist (need K1>0 and K2>0).")

    pairs = [[column[i], target[i]] for i in range(len(target))]
    pairs.sort(key=lambda x: x[0])

    column_sorted = [pairs[i][0] for i in range(len(target))]
    target_sorted = [pairs[i][1] for i in range(len(target))]

    best_score = float("-inf")
    best_pi2 = column_sorted[0]

    for boundary in range(1, len(column_sorted) + 1):
        # If boundary doesn't change value -> skip (no new split)
        if boundary != len(column_sorted) and abs(column_sorted[boundary] - column_sorted[boundary - 1]) < 1e-8:
            continue

        left = target_sorted[:boundary]
        right = target_sorted[boundary:]

        left_K1 = sum(1 for el in left if el == 1)
        left_K2 = sum(1 for el in left if el == 2)

        right_K1 = sum(1 for el in right if el == 1)
        right_K2 = sum(1 for el in right if el == 2)

        left_numer = (
            (left_K1**2 - left_K1) + (right_K1**2 - right_K1) +
            (left_K2**2 - left_K2) + (right_K2**2 - right_K2)
        )
        left_denom = (K1**2 - K1) + (K2**2 - K2)
        if left_denom == 0:
            continue

        right_numer = (
            left_K1 * (K2 - left_K2) +
            left_K2 * (K1 - left_K1) +
            right_K1 * (K2 - right_K2) +
            right_K2 * (K1 - right_K1)
        )
        right_denom = 2 * K1 * K2
        if right_denom == 0:
            continue

        score = (left_numer / left_denom) * (right_numer / right_denom)

        if score > best_score:
            best_score = score
            best_pi2 = column_sorted[boundary - 1]

    pi1 = float(min(column_sorted))
    pi2 = float(best_pi2)
    pi3 = float(max(column_sorted))

    return Criterion1Result(weight_wc=float(best_score), pi1=pi1, pi2=pi2, pi3=pi3)


# ============================================================
# Threshold Γ_c (Formula 3)
# ============================================================

def compute_gamma_c(sorted_values: np.ndarray, pi2: float, pi3: float) -> Tuple[float, float]:
    """
    Γ_c = (pi2 + b) / 2
    where b is the nearest value to pi2 from the interval (pi2; pi3).

    If no value exists > pi2, fallback b = pi3.
    """
    right_candidates = sorted_values[sorted_values > pi2]

    if right_candidates.size == 0:
        b = pi3
    else:
        b = float(right_candidates[np.argmin(right_candidates - pi2)])

    gamma_c = (float(pi2) + float(b)) / 2.0
    return gamma_c, float(b)


def binarize_by_gamma(column: np.ndarray, gamma_c: float) -> np.ndarray:
    """
    Convert quantitative column -> nominal scale {1,2}:
      if x <= Γ_c  -> 1
      if x >  Γ_c  -> 2
    """
    return np.where(column <= gamma_c, 1, 2).astype(int)


# ============================================================
# Gradations and Contributions (Formula 5)
# ============================================================

def gradation_counts(binary_column: np.ndarray, target: np.ndarray) -> GradationCounts:
    """Count gradations {1,2} separately inside K1 and K2."""
    mask_k1 = (target == 1)
    mask_k2 = (target == 2)

    g_k1 = {
        1: int(np.sum(binary_column[mask_k1] == 1)),
        2: int(np.sum(binary_column[mask_k1] == 2)),
    }
    g_k2 = {
        1: int(np.sum(binary_column[mask_k2] == 1)),
        2: int(np.sum(binary_column[mask_k2] == 2)),
    }
    return GradationCounts(g_k1=g_k1, g_k2=g_k2)


def contributions_eta(wc: float, g_k1: Dict[int, int], g_k2: Dict[int, int], target: np.ndarray) -> ContributionResult:
    """
    η_c(j) = ω_c * ( g_k1(j)/|K1| - g_k2(j)/|K2| )
    """
    K1 = int(np.sum(target == 1))
    K2 = int(np.sum(target == 2))
    if K1 == 0 or K2 == 0:
        raise ValueError("Both classes must exist to compute contributions.")

    eta = {}
    for j in (1, 2):
        eta[j] = float(wc) * ((g_k1[j] / K1) - (g_k2[j] / K2))

    return ContributionResult(eta=eta)


def apply_contributions(binary_column: np.ndarray, eta: Dict[int, float]) -> np.ndarray:
    """Replace binary values {1,2} with contribution values η_c(1), η_c(2)."""
    out = np.zeros_like(binary_column, dtype=float)
    out[binary_column == 1] = eta[1]
    out[binary_column == 2] = eta[2]
    return out


# ============================================================
# End-to-end quantitative pipeline
# ============================================================

def build_quantitative_nominalization(
    X: np.ndarray,
    y: np.ndarray,
    quantitative_idx: List[int],
) -> QuantitativePipelineResult:
    """
    Full quantitative pipeline:
      1) Criterion-1 => ωc + π1,π2,π3
      2) Γc threshold
      3) binarize to {1,2}
      4) split by classes => gradation counts
      5) contributions ηc(j)
      6) create contribution dataset (replace 1/2 with η)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (m,n)")
    if y.ndim != 1:
        raise ValueError("y must be 1D (m,)")

    m, n = X.shape
    if len(y) != m:
        raise ValueError("X and y size mismatch")

    weights_wc: Dict[int, float] = {}
    pi_table: Dict[int, Tuple[float, float, float]] = {}
    gamma: Dict[int, float] = {}
    contributions_all: Dict[int, Dict[int, float]] = {}

    binary_cols: List[np.ndarray] = []
    contrib_cols: List[np.ndarray] = []

    for feat_idx in quantitative_idx:
        col = X[:, feat_idx].astype(float)
        y_list = y.astype(int).tolist()

        c1 = criterion_1(col.tolist(), y_list)
        weights_wc[feat_idx] = c1.weight_wc
        pi_table[feat_idx] = (c1.pi1, c1.pi2, c1.pi3)

        sorted_vals = np.sort(col)
        gamma_c, b_val = compute_gamma_c(sorted_vals, c1.pi2, c1.pi3)
        gamma[feat_idx] = gamma_c

        bin_col = binarize_by_gamma(col, gamma_c)
        binary_cols.append(bin_col)

        g = gradation_counts(bin_col, y)

        eta_res = contributions_eta(c1.weight_wc, g.g_k1, g.g_k2, y)
        contributions_all[feat_idx] = eta_res.eta

        contrib_col = apply_contributions(bin_col, eta_res.eta)
        contrib_cols.append(contrib_col)

    binary_X = np.column_stack(binary_cols) if binary_cols else np.empty((m, 0), dtype=int)
    contribution_X = np.column_stack(contrib_cols) if contrib_cols else np.empty((m, 0), dtype=float)

    return QuantitativePipelineResult(
        quantitative_idx=list(quantitative_idx),
        weights_wc=weights_wc,
        pi_table=pi_table,
        gamma=gamma,
        binary_X=binary_X,
        contribution_X=contribution_X,
        contributions_eta=contributions_all,
    )
