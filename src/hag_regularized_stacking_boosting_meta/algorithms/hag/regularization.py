from __future__ import annotations

from typing import Dict

import numpy as np

from .majorizing_functions import MajorizingFn


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
    Regularization form:
      base + sign(y) * alpha * f(-base)
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
    Step-3 candidate (Excel columns F -> H):
      F = R + eta_feature
      H = F + sign(y)*alpha*f(-F)
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
    Step-4 update (Excel columns H -> I):

    Excel Step-4 does NOT use H as next organizer.
    It computes I:
      H = (R + eta) + sign*alpha*f(-(R+eta))
      I = H + sign*alpha*f(-H)

    Then organizer for next loop becomes I (r1, r2, ...).

    So we apply regularization TWICE.
    """
    base = np.asarray(R, dtype=float) + np.asarray(eta_feature, dtype=float)

    # First correction -> Excel column H
    H = apply_regularization(base, y, alpha, majorizer, k1_label, k2_label)

    # Second correction -> Excel column I (next organizer)
    I = apply_regularization(H, y, alpha, majorizer, k1_label, k2_label)  # noqa: E741 (Excel column I)
    return I


# -------------------------------------------------------------------
# Debug helpers (optional but useful)
# -------------------------------------------------------------------

def build_bt_candidate_debug(
    R: np.ndarray,
    eta_feature: np.ndarray,
    y: np.ndarray,
    alpha: float,
    majorizer: MajorizingFn,
    k1_label: int,
    k2_label: int,
) -> Dict[str, np.ndarray]:
    """
    Returns vectors matching Excel columns:
      B = R
      D = eta
      F = R + eta
      H = F + sign(y)*alpha*majorizer(-F)
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    eta = np.asarray(eta_feature, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)

    F = R + eta
    sign = _class_sign(y, k1_label, k2_label)
    H = F + (sign * float(alpha)) * majorizer(-F)

    return {
        "B_R": R,
        "D_eta": eta,
        "F_base_bt": F,
        "H_bt_final": H,
    }


def build_step4_debug(
    R: np.ndarray,
    eta_feature: np.ndarray,
    y: np.ndarray,
    alpha: float,
    majorizer: MajorizingFn,
    k1_label: int,
    k2_label: int,
) -> Dict[str, np.ndarray]:
    """
    Excel Step-4 debug:
      H = first corrected vector (bt(sigm))
      I = second corrected vector (R(S^{...})(sigm)) -> organizer for next loop
    """
    R = np.asarray(R, dtype=float).reshape(-1)
    eta = np.asarray(eta_feature, dtype=float).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)

    F = R + eta
    H = apply_regularization(F, y, alpha, majorizer, k1_label, k2_label)
    I = apply_regularization(H, y, alpha, majorizer, k1_label, k2_label)  # noqa: E741 (Excel column I)

    return {"F": F, "H": H, "I": I}

