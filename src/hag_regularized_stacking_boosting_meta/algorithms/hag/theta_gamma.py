from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class ThetaGammaResult:
    m1: float
    m2: float
    theta: float
    gamma: float
    ratio: float  # theta/gamma (safe)


def safe_ratio(num: float, den: float, *, eps: float = 1e-12) -> float:
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
    Paper/global-mean version (kept for reference).
    """
    bt = np.asarray(bt, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)

    mask1 = (y == k1_label)
    mask2 = (y == k2_label)
    if not mask1.any() or not mask2.any():
        raise ValueError("Both classes must be present to compute θ/γ")

    m1 = float(bt[mask1].mean())
    m2 = float(bt[mask2].mean())

    d1 = np.abs(bt - m1)
    d2 = np.abs(bt - m2)

    theta = float(d1[mask1].sum() + d2[mask2].sum())
    gamma = float(d2[mask1].sum() + d1[mask2].sum())

    ratio = safe_ratio(theta, gamma)
    return ThetaGammaResult(m1=m1, m2=m2, theta=theta, gamma=gamma, ratio=ratio)


def excel_step3_trace_from_bt(
    bt_final: np.ndarray,
    y: np.ndarray,
    *,
    k1_label: int,
    k2_label: int,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Excel-faithful running columns (I..O) using bt_final (H) + y.

    I: running sum of H for K1
    J: running sum of H for K2
    K: I / |K1|
    L: J / |K2|
    M: running theta using running means K/L
    N: running gamma using running means K/L
    O: final M_last / N_last
    """
    h = np.asarray(bt_final, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    m = h.shape[0]

    mask1 = (y == int(k1_label))
    mask2 = (y == int(k2_label))
    cnt1 = int(mask1.sum())
    cnt2 = int(mask2.sum())

    if cnt1 == 0 or cnt2 == 0:
        inf = np.array([float("inf")], dtype=float)
        zeros = np.zeros(m, dtype=float)
        return {
            "I_M1_sum": zeros,
            "J_M2_sum": zeros,
            "K_K1_mean": zeros,
            "L_K2_mean": zeros,
            "M_theta_cum": zeros,
            "N_gamma_cum": zeros,
            "O_ratio_final": inf,
        }

    I = np.zeros(m, dtype=float)  # noqa: E741 (Excel column I)
    J = np.zeros(m, dtype=float)
    K = np.zeros(m, dtype=float)
    L = np.zeros(m, dtype=float)
    M_theta = np.zeros(m, dtype=float)
    N_gamma = np.zeros(m, dtype=float)

    s1 = 0.0
    s2 = 0.0
    th = 0.0
    ga = 0.0

    for t in range(m):
        if y[t] == int(k1_label):
            s1 += h[t]
        elif y[t] == int(k2_label):
            s2 += h[t]

        I[t] = s1
        J[t] = s2

        K[t] = s1 / cnt1
        L[t] = s2 / cnt2

        if y[t] == int(k1_label):
            th += abs(h[t] - K[t])
            ga += abs(h[t] - L[t])
        elif y[t] == int(k2_label):
            th += abs(h[t] - L[t])
            ga += abs(h[t] - K[t])

        M_theta[t] = th
        N_gamma[t] = ga

    ratio = float("inf") if abs(ga) < eps else (th / ga)
    return {
        "I_M1_sum": I,
        "J_M2_sum": J,
        "K_K1_mean": K,
        "L_K2_mean": L,
        "M_theta_cum": M_theta,
        "N_gamma_cum": N_gamma,
        "O_ratio_final": np.array([ratio], dtype=float),
    }


def excel_ratio_from_bt(
    bt_final: np.ndarray,
    y: np.ndarray,
    *,
    k1_label: int,
    k2_label: int,
) -> float:
    """
    Returns Excel O value directly: O = M_last / N_last using running means.
    """
    tr = excel_step3_trace_from_bt(
        bt_final, y, k1_label=k1_label, k2_label=k2_label
    )
    return float(tr["O_ratio_final"][0])