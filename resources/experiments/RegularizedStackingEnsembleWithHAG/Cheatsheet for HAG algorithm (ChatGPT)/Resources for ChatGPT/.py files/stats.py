from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class ThetaGammaResult:
    m1: float
    m2: float
    theta: float
    gamma: float
    ratio: float  # theta/gamma (safe)


def safe_ratio(num: float, den: float, *, eps: float = 1e-12) -> float:
    # If gamma is ~0, ratio should become very large (bad candidate).
    if abs(den) < eps:
        return float("inf")
    return num / den


def theta_gamma_from_bt(
    bt: np.ndarray,
    y: np.ndarray,
    *,
    k1_label: int,
    k2_label: int,
) -> ThetaGammaResult:
    """
    Implements the θ and γ computations described in the paper's Step 3 block:
    - M1: mean of bt over class K1
    - M2: mean of bt over class K2
    - θ, γ: accumulations depending on class membership
    :contentReference[oaicite:2]{index=2}
    """
    if bt.ndim != 1:
        raise ValueError("bt must be 1-D")

    mask1 = (y == k1_label)
    mask2 = (y == k2_label)
    if not mask1.any() or not mask2.any():
        raise ValueError("Both classes must be present to compute θ/γ")

    m1 = float(bt[mask1].mean())
    m2 = float(bt[mask2].mean())

    theta = 0.0
    gamma = 0.0

    # Vectorized version of:
    # if St in K1: theta += |bt - M1|; gamma += |bt - M2|
    # else:        theta += |bt - M2|; gamma += |bt - M1|
    d1 = np.abs(bt - m1)
    d2 = np.abs(bt - m2)

    theta = float(d1[mask1].sum() + d2[mask2].sum())
    gamma = float(d2[mask1].sum() + d1[mask2].sum())

    ratio = safe_ratio(theta, gamma)
    return ThetaGammaResult(m1=m1, m2=m2, theta=theta, gamma=gamma, ratio=ratio)
