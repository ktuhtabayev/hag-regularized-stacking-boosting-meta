from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Result container (GUI-friendly + future-proof)
# ============================================================

@dataclass(frozen=True)
class NominalWeightsResult:
    nominal_idx: List[int]

    # Per-feature scalars
    lambda_c: Dict[int, float]         # λ_c
    beta_c: Dict[int, float]           # β_c
    weight_wc: Dict[int, float]        # ω_c = λ_c * β_c   (CANONICAL NAME)

    # Gradations + counts per class (per feature)
    gradations: Dict[int, List[float]]                 # sorted unique values (j=1..μ)
    g_k1: Dict[int, Dict[float, int]]                  # feature -> {gradation_value: count in K1}
    g_k2: Dict[int, Dict[float, int]]                  # feature -> {gradation_value: count in K2}

    # Extra values used in β computation (for audit / Excel-compare)
    p_c: Dict[int, int]           # number of gradations μ
    l1_c: Dict[int, int]          # l1c = number of gradations present in class 1
    l2_c: Dict[int, int]          # l2c = number of gradations present in class 2
    D1_c: Dict[int, int]          # D1c (CANONICAL NAME)
    D2_c: Dict[int, int]          # D2c (CANONICAL NAME)

    # η_c(j) contributions (CANONICAL NAME)
    contributions_eta: Dict[int, Dict[float, float]]  # feature -> {gradation_value: eta}

    # Contribution dataset (same object order as original X!)
    # shape: (m, len(nominal_idx))
    contribution_X: np.ndarray


# ============================================================
# Helpers
# ============================================================

def _validate_binary_classes(y: np.ndarray) -> Tuple[int, int]:
    """Expect y labels to be exactly {1,2}. Return (K1,K2) sizes."""
    vals = sorted(set(y.tolist()))
    if vals != [1, 2]:
        raise ValueError(
            f"Nominal weights expect binary classes labeled {{1,2}}, got {vals}. "
            "Fix label_mapping in DatasetConfig or preprocessing."
        )
    k1 = int(np.sum(y == 1))
    k2 = int(np.sum(y == 2))
    if k1 == 0 or k2 == 0:
        raise ValueError(f"Both classes must be present. Got K1={k1}, K2={k2}.")
    return k1, k2


def _counts_by_class_for_column(
    col: np.ndarray, y: np.ndarray
) -> Tuple[Dict[float, int], Dict[float, int], np.ndarray]:
    """
    Build counts of each gradation value for class K1 and K2 (every gradation appears
    in both dicts, sorted by value). Keys are floats (works for int-valued nominal too).

    Also returns, per object, the position of its value among the sorted gradations.
    """
    gradations, position = np.unique(np.asarray(col, dtype=float), return_inverse=True)
    position = position.reshape(-1)
    in_k1 = np.asarray(y) == 1
    count1 = np.bincount(position[in_k1], minlength=gradations.shape[0])
    count2 = np.bincount(position[~in_k1], minlength=gradations.shape[0])

    keys = [float(v) for v in gradations]
    g1 = {key: int(c) for key, c in zip(keys, count1)}
    g2 = {key: int(c) for key, c in zip(keys, count2)}
    return g1, g2, position


def _lambda_beta_weight(
    g1: Dict[float, int],
    g2: Dict[float, int],
    k1: int,
    k2: int,
    l1: int,
    l2: int,
) -> Tuple[float, float, float, int, int]:
    """
    Implements formulas:

    λ_c = 1 - ( Σ_j g1_j * g2_j ) / ( 2*K1*K2 )

    β_c uses:
      beta_numerator = Σ_j [ g1_j(g1_j-1) + g2_j(g2_j-1) ]
      D1c, D2c depend on p_c (=μ) and l1,l2:
        if p_c > 2:
            Dd = (|Kd| - l_dc + 1)(|Kd| - l_dc)
        else:
            Dd = |Kd|(|Kd|-1)
      β_c = beta_numerator/(D1c + D2c) if D1c+D2c>0 else 0

    ω_c = λ_c * β_c
    """
    lambda_num = 0
    beta_num = 0
    for key in g1.keys():
        a = g1[key]
        b = g2[key]
        lambda_num += a * b
        beta_num += a * (a - 1) + b * (b - 1)

    lam = 1.0 - (lambda_num / (2.0 * k1 * k2))

    p_c = len(g1)  # μ
    if p_c > 2:
        d1 = (k1 - l1 + 1) * (k1 - l1)
        d2 = (k2 - l2 + 1) * (k2 - l2)
    else:
        d1 = k1 * (k1 - 1)
        d2 = k2 * (k2 - 1)

    denom = d1 + d2
    beta = 0.0 if denom == 0 else (beta_num / float(denom))

    w = lam * beta
    return float(lam), float(beta), float(w), int(d1), int(d2)


def _eta_contribution(
    w: float,
    g1: Dict[float, int],
    g2: Dict[float, int],
    k1: int,
    k2: int,
) -> Dict[float, float]:
    """
    Formula (5):
      η_c(j) = ω_c * ( g1_j/K1 - g2_j/K2 )
    """
    eta: Dict[float, float] = {}
    for key in g1.keys():
        eta[key] = float(w) * ((g1[key] / float(k1)) - (g2[key] / float(k2)))
    return eta


# ============================================================
# Public API: build nominal contribution dataset
# ============================================================

def build_nominal_contributions(
    X: np.ndarray,
    y: np.ndarray,
    nominal_idx: List[int],
) -> NominalWeightsResult:
    """
    Build nominal weights (λ, β, ω) and contributions η, then produce
    "Nominal Features with Contribution Values" dataset.

    IMPORTANT:
    - Preserves object order (rows) exactly.
    - Preserves feature indices (nominal_idx are indices in the FULL X).
    - Works if nominal_idx is empty (returns empty contribution_X with shape (m,0)).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    m = int(X.shape[0])
    if not nominal_idx:
        return NominalWeightsResult(
            nominal_idx=[],
            lambda_c={},
            beta_c={},
            weight_wc={},
            gradations={},
            g_k1={},
            g_k2={},
            p_c={},
            l1_c={},
            l2_c={},
            D1_c={},
            D2_c={},
            contributions_eta={},
            contribution_X=np.zeros((m, 0), dtype=float),
        )

    k1, k2 = _validate_binary_classes(y)

    lambda_c: Dict[int, float] = {}
    beta_c: Dict[int, float] = {}
    weight_wc: Dict[int, float] = {}

    gradations: Dict[int, List[float]] = {}
    g_k1: Dict[int, Dict[float, int]] = {}
    g_k2: Dict[int, Dict[float, int]] = {}

    p_c: Dict[int, int] = {}
    l1_c: Dict[int, int] = {}
    l2_c: Dict[int, int] = {}
    D1_c: Dict[int, int] = {}
    D2_c: Dict[int, int] = {}

    contributions_eta: Dict[int, Dict[float, float]] = {}

    contrib = np.zeros((m, len(nominal_idx)), dtype=float)

    for out_j, fidx in enumerate(nominal_idx):
        g1, g2, position = _counts_by_class_for_column(X[:, fidx], y)

        # l1/l2: number of gradations present in each class (non-zero)
        l1 = sum(1 for v in g1.values() if v > 0)
        l2 = sum(1 for v in g2.values() if v > 0)

        lam, beta, w, d1, d2 = _lambda_beta_weight(g1, g2, k1, k2, l1, l2)
        eta = _eta_contribution(w, g1, g2, k1, k2)

        # store
        gradations[fidx] = list(g1.keys())
        g_k1[fidx] = g1
        g_k2[fidx] = g2

        p_c[fidx] = len(g1)
        l1_c[fidx] = l1
        l2_c[fidx] = l2
        D1_c[fidx] = d1
        D2_c[fidx] = d2

        lambda_c[fidx] = lam
        beta_c[fidx] = beta
        weight_wc[fidx] = w
        contributions_eta[fidx] = eta

        # map each row nominal value to eta(value)
        contrib[:, out_j] = np.asarray(list(eta.values()), dtype=float)[position]

    return NominalWeightsResult(
        nominal_idx=list(nominal_idx),
        lambda_c=lambda_c,
        beta_c=beta_c,
        weight_wc=weight_wc,
        gradations=gradations,
        g_k1=g_k1,
        g_k2=g_k2,
        p_c=p_c,
        l1_c=l1_c,
        l2_c=l2_c,
        D1_c=D1_c,
        D2_c=D2_c,
        contributions_eta=contributions_eta,
        contribution_X=contrib,
    )
