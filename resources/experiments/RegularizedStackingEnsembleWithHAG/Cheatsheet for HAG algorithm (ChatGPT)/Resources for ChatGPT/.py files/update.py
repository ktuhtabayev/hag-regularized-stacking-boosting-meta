from __future__ import annotations

from typing import Callable

import numpy as np

from .majorizing import MajorizingFn


def _class_sign(y: np.ndarray, k1_label: int, k2_label: int) -> np.ndarray:
    """
    Map class labels to +1/-1 sign:
      class k1 -> +1
      class k2 -> -1
    """
    y = np.asarray(y).astype(int)
    sign = np.zeros_like(y, dtype=float)
    sign[y == int(k1_label)] = +1.0
    sign[y == int(k2_label)] = -1.0
    return sign


def apply_regularization(
    base: np.ndarray,
    y: np.ndarray,
    alpha: float,
    majorizer: MajorizingFn,
    k1_label: int,
    k2_label: int,
) -> np.ndarray:
    """
    Regularization form (vectorized):
      base + sign(y) * alpha * f(-base)

    This matches the idea in your paper/Excel experiments where the correction uses ±α f(-·).
    """
    base = np.asarray(base, dtype=float)
    sign = _class_sign(y, k1_label, k2_label)
    return base + (sign * float(alpha)) * majorizer(-base)


def build_bt_candidate(
    R: np.ndarray,
    eta_feature: np.ndarray,
    y: np.ndarray,
    alpha: float,
    majorizer: MajorizingFn,
    k1_label: int,
    k2_label: int,
) -> np.ndarray:
    """
    Step-3 candidate:
      bt = R + eta_feature
      bt = bt + sign(y)*alpha*f(-bt)
    """
    base = np.asarray(R, dtype=float) + np.asarray(eta_feature, dtype=float)
    return apply_regularization(base, y, alpha, majorizer, k1_label, k2_label)


def update_R(
    R: np.ndarray,
    eta_feature: np.ndarray,
    y: np.ndarray,
    alpha: float,
    majorizer: MajorizingFn,
    k1_label: int,
    k2_label: int,
) -> np.ndarray:
    """
    Step-4 update:
      R <- R + eta_feature
      R <- R + sign(y)*alpha*f(-R)
    """
    base = np.asarray(R, dtype=float) + np.asarray(eta_feature, dtype=float)
    return apply_regularization(base, y, alpha, majorizer, k1_label, k2_label)
