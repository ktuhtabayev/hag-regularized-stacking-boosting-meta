from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .regularization import build_bt_candidate_debug
from .theta_gamma import excel_step3_trace_from_bt
from .majorizing_functions import MajorizingFn


Majorizer = MajorizingFn  # ndarray -> ndarray


@dataclass(frozen=True)
class SelectionResult:
    q: int
    best_ratio: float
    # θ/γ ratio (Excel column O) of every scanned candidate, keyed by feature index
    candidate_ratios: Dict[int, float] = field(default_factory=dict)


def _print_vec(name: str, v: np.ndarray, *, prec: int = 10) -> None:
    v = np.asarray(v, dtype=float).reshape(-1)
    print(f"{name} = {np.array2string(v, precision=prec, separator=', ')}")


def _print_excel_columns(fi: int, columns: Dict[str, np.ndarray], trace: Dict[str, np.ndarray]) -> None:
    print("\n==============================")
    print(f"STEP 3 CANDIDATE fi = {fi}")
    print("==============================")

    _print_vec("B (R)", columns["B_R"])
    _print_vec("D (eta)", columns["D_eta"])
    _print_vec("F (bt = R+eta)", columns["F_base_bt"])
    _print_vec("H (bt after reg/majorizer)", columns["H_bt_final"])

    _print_vec("I (M1 Σ)", trace["I_M1_sum"])
    _print_vec("J (M2 Σ)", trace["J_M2_sum"])
    _print_vec("K (K1 mean)", trace["K_K1_mean"])
    _print_vec("L (K2 mean)", trace["L_K2_mean"])
    _print_vec("M (theta cum)", trace["M_theta_cum"])
    _print_vec("N (gamma cum)", trace["N_gamma_cum"])
    print("O (theta/gamma final) =", float(trace["O_ratio_final"][0]))


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
    debug: bool = False,
) -> SelectionResult:
    """
    Step 3 (Excel-faithful):
      - scan candidates in original index order: 0,1,3,4,... (skip organizer already removed)
      - build bt_final
      - compute ratio using Excel running columns => O = M_last/N_last
      - choose q with minimum ratio (strictly improving over cr1)
      - if no candidate beats cr1, fall back to the first candidate in P

    debug=True prints the Excel columns B, D, F, H, I-O of every candidate.
    """
    best_q: Optional[int] = None
    best_ratio: float = float(cr1_init)
    candidate_ratios: Dict[int, float] = {}

    for fi in P:  # P is already original order in greedy_grouping.py
        columns = build_bt_candidate_debug(
            R,
            X[:, fi],
            y,
            alpha=alpha,
            majorizer=majorizer,
            k1_label=k1_label,
            k2_label=k2_label,
        )
        trace = excel_step3_trace_from_bt(
            columns["H_bt_final"], y, k1_label=k1_label, k2_label=k2_label
        )

        # Excel ratio (O) for selection
        ratio = float(trace["O_ratio_final"][0])
        candidate_ratios[int(fi)] = ratio

        if debug:
            _print_excel_columns(fi, columns, trace)

        if ratio < best_ratio:
            best_ratio = ratio
            best_q = int(fi)

    if best_q is None:
        # deterministic fallback: first in P
        best_q = int(P[0])
        best_ratio = candidate_ratios[best_q]

    return SelectionResult(q=best_q, best_ratio=best_ratio, candidate_ratios=candidate_ratios)
