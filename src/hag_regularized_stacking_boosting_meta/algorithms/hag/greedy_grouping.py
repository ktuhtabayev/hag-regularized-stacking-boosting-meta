from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from hag_regularized_stacking_boosting_meta.domain.params import HAGParams
from .majorizing_functions import get_majorizing_function
from .candidate_selection import choose_next_feature_q
from .regularization import update_R
from .latent_features import build_dij_from_R_history, LatentBuildResult


@dataclass(frozen=True)
class HAGResult:
    tuplam: List[int]
    crit_history: List[float]
    r_step4_history: List[np.ndarray]
    dij: np.ndarray
    p: int
    # organizer u (= tuplam[0]) and, per Step-3 iteration, θ/γ of every scanned candidate
    organizer: int = -1
    candidate_history: List[Dict[int, float]] = field(default_factory=list)


def rank_features_by_weight(weights: Sequence[float]) -> List[int]:
    """
    Stable descending rank:
      - larger weight first
      - ties: smaller index first
    """
    w = list(map(float, weights))
    return sorted(range(len(w)), key=lambda i: (-w[i], i))


def greedy_hag_grouping(
    X: np.ndarray,
    y: np.ndarray,
    weights: Sequence[float],
    params: HAGParams,
    *,
    verbose: bool = True,
) -> HAGResult:
    """
    Algorithm-1 (Greedy HAG + Regularization + Latent).

    verbose=False silences console output, including the Excel Step-3 column trace.

    Excel-faithful rules:
    - Do NOT reorder X columns. Keep original dataset feature order.
    - weights used ONLY ONCE to find organizer u (the feature with the highest weight).
    - Step 3 scans: x0, x1, x3, x4, ... (skip organizer and already-selected).
    - Step 4 organizer update uses Excel I column:
        I = H + sign*alpha*sigmoid(-H)
      so next loop organizer is I (r1, r2, ...).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (m,n)")
    m, n = X.shape
    if len(weights) != n:
        raise ValueError("weights length must match X columns")
    if y.shape[0] != m:
        raise ValueError("y length must match X rows")

    majorizer = get_majorizing_function(params.majorizing)
    if verbose:
        print("MAJOR:", params.majorizing.name, params.majorizing.params)

    # Organizer selection (ONLY ONCE): the feature with the highest weight
    u = int(rank_features_by_weight(weights)[0])

    # Candidate pool in ORIGINAL index order excluding organizer
    P = [i for i in range(n) if i != u]

    # Init
    tuplam: List[int] = [u]
    R = X[:, u].astype(float, copy=True)  # organizer vector

    cr1 = float(params.cr1)

    r_step4_history: List[np.ndarray] = []
    crit_history: List[float] = []
    candidate_history: List[Dict[int, float]] = []

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
            debug=None if verbose else False,
        )
        q = int(sel.q)
        crit = float(sel.best_ratio)

        # Step 4 commit
        crit_history.append(crit)
        candidate_history.append(dict(sel.candidate_ratios))
        P.remove(q)
        tuplam.append(q)

        # reset cr1 for next iteration
        cr1 = float(params.cr1)

        # IMPORTANT: update_R now returns Excel column I (organizer for next loop)
        R = update_R(
            R=R,
            eta_feature=X[:, q],
            y=y,
            alpha=float(params.alpha),
            majorizer=majorizer,
            k1_label=int(params.k1_label),
            k2_label=int(params.k2_label),
        )

        # Unbounded majorizers (exponential, quadratic) can blow up across iterations
        if not np.all(np.isfinite(R)):
            raise ValueError(
                f"HAG diverged at step r{len(r_step4_history) + 1}: the organizer became "
                f"non-finite (majorizing function '{params.majorizing.name}' overflowed). "
                "Lower alpha or the majorizer's parameters, or use a bounded majorizer such as sigmoid."
            )

        # store organizer history (r1, r2, ...) for META + latent build
        r_step4_history.append(R.copy())

        # Stop/continue rule
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
        organizer=u,
        candidate_history=candidate_history,
    )