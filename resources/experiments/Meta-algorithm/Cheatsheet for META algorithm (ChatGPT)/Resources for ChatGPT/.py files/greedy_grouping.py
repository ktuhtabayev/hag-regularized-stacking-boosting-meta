from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from hag_regularized_stacking_boosting_meta.domain.params import HAGParams
from .majorizing import get_majorizing_function
from .selection import choose_next_feature_q
from .update import update_R
from .latent import build_dij_from_R_history, LatentBuildResult


@dataclass(frozen=True)
class HAGResult:
    tuplam: List[int]
    crit_history: List[float]
    r_step4_history: List[np.ndarray]
    dij: np.ndarray
    p: int


def rank_features_by_weight(weights: Sequence[float]) -> List[int]:
    """
    Stable descending rank:
      - larger weight first
      - ties: smaller index first (left-to-right)
    """
    w = list(map(float, weights))
    return sorted(range(len(w)), key=lambda i: (-w[i], i))


def greedy_hag_grouping(
    X: np.ndarray,
    y: np.ndarray,
    weights: Sequence[float],
    params: HAGParams,
) -> HAGResult:
    """
    Algorithm-1 (Greedy HAG + Regularization + Latent).

    Assumption:
      X[:, j] already contains η_j(a_tj) contribution values (merged contrib dataset).

    Stop rule:
      if |TUPLAM| < κ and crit > δ -> continue else stop
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (m,n)")
    m, n = X.shape
    if len(weights) != n:
        raise ValueError("weights length must match X columns")
    if y.shape[0] != m:
        raise ValueError("y length must match X rows")

    majorizer = get_majorizing_function(params.majorizing)

    # Step 1
    P = list(range(n))

    # Step 2
    ranked = rank_features_by_weight(weights)
    u = ranked[0]
    tuplam: List[int] = [u]
    R = X[:, u].astype(float, copy=True)

    cr1 = float(params.cr1)
    P.remove(u)

    r_step4_history: List[np.ndarray] = []
    crit_history: List[float] = []

    while True:
        if not P:
            break

        sel = choose_next_feature_q(
            X=X,
            y=y,
            R=R,
            P=P,
            alpha=float(params.alpha),
            majorizer=majorizer,
            k1_label=int(params.k1_label),
            k2_label=int(params.k2_label),
            cr1_init=cr1,
        )
        q = int(sel.q)
        crit = float(sel.best_ratio)

        # Step 4
        crit_history.append(crit)
        P.remove(q)
        tuplam.append(q)

        # reset cr1 each iteration (Excel-friendly)
        cr1 = float(params.cr1)

        R = update_R(
            R=R,
            eta_feature=X[:, q],
            y=y,
            alpha=float(params.alpha),
            majorizer=majorizer,
            k1_label=int(params.k1_label),
            k2_label=int(params.k2_label),
        )

        r_step4_history.append(R.copy())

        if (len(tuplam) < int(params.kappa)) and (crit > float(params.delta)):
            continue
        break

    latent: LatentBuildResult = build_dij_from_R_history(r_step4_history)
    return HAGResult(
        tuplam=tuplam,
        crit_history=crit_history,
        r_step4_history=r_step4_history,
        dij=latent.dij,
        p=latent.p,
    )
