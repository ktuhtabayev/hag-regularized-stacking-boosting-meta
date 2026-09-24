from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from .update import build_bt_candidate

Majorizer = Callable[[float], float]


@dataclass(frozen=True)
class SelectionResult:
    q: int
    best_ratio: float


def _theta_gamma_ratio(
    bt: np.ndarray,
    y: np.ndarray,
    k1_label: int,
    k2_label: int,
) -> float:
    """
    Compute the selection ratio θ/γ EXACTLY as in Algorithm-1 (Step 3) from the PDF.

    Given candidate vector b_t (built from current R and candidate feature + regularization):

      M1 = (Σ_{S_t ∈ K1} b_t) / |K1|
      M2 = (Σ_{S_t ∈ K2} b_t) / |K2|

      θ = Σ_{t=1..m}  {
            |b_t - M1|  if S_t ∈ K1
            |b_t - M2|  if S_t ∈ K2
          }

      γ = Σ_{t=1..m}  {
            |b_t - M2|  if S_t ∈ K1
            |b_t - M1|  if S_t ∈ K2
          }

      ratio = θ / γ   (if γ == 0 -> +inf)

    Intuition:
      - θ measures within-class scatter around its own class mean.
      - γ measures cross-class scatter to the opposite class mean.
    """
    bt = np.asarray(bt, dtype=float)
    y = np.asarray(y).astype(int)

    mask1 = (y == k1_label)
    mask2 = (y == k2_label)

    if not mask1.any() or not mask2.any():
        return float("inf")

    m1 = float(bt[mask1].mean())
    m2 = float(bt[mask2].mean())

    theta = 0.0
    gamma = 0.0
    for val, cls in zip(bt, y):
        valf = float(val)
        if cls == k1_label:
            theta += abs(valf - m1)
            gamma += abs(valf - m2)
        elif cls == k2_label:
            theta += abs(valf - m2)
            gamma += abs(valf - m1)

    if gamma == 0.0:
        return float("inf")
    return float(theta / gamma)


def choose_next_feature_q(
    X: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    P: List[int],
    *,
    alpha: float,
    majorizer: Majorizer,
    k1_label: int,
    k2_label: int,
    cr1_init: float = 10.0,
) -> SelectionResult:
    """
    Step 3: scan candidates u in P, build b_t, compute θ/γ,
    choose q with minimum ratio (strictly improving over cr1).
    """
    best_q: Optional[int] = None
    best_ratio: float = float(cr1_init)

    for fi in P:
        bt = build_bt_candidate(
            R,
            X[:, fi],
            y,
            alpha=alpha,
            majorizer=majorizer,
            k1_label=k1_label,
            k2_label=k2_label,
        )

        ratio = _theta_gamma_ratio(bt, y, k1_label, k2_label)

        if ratio < best_ratio:
            best_ratio = float(ratio)
            best_q = int(fi)

    if best_q is None:
        # fallback: if nothing improved over cr1, pick the first in P (deterministic)
        best_q = int(P[0])
        bt = build_bt_candidate(
            R,
            X[:, best_q],
            y,
            alpha=alpha,
            majorizer=majorizer,
            k1_label=k1_label,
            k2_label=k2_label,
        )
        best_ratio = float(_theta_gamma_ratio(bt, y, k1_label, k2_label))

    return SelectionResult(q=best_q, best_ratio=best_ratio)
