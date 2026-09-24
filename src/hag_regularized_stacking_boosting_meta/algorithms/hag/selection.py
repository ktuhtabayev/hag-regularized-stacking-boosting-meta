from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .update import build_bt_candidate, build_bt_candidate_debug
from .stats import excel_step3_trace_from_bt, excel_ratio_from_bt
from .majorizing import MajorizingFn


DEBUG_PRINT_EXCEL_COLUMNS: bool = True
Majorizer = MajorizingFn  # ndarray -> ndarray


@dataclass(frozen=True)
class SelectionResult:
    q: int
    best_ratio: float


def _print_vec(name: str, v: np.ndarray, *, prec: int = 10) -> None:
    v = np.asarray(v, dtype=float).reshape(-1)
    print(f"{name} = {np.array2string(v, precision=prec, separator=', ')}")


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
    Step 3 (Excel-faithful):
      - scan candidates in original index order: 0,1,3,4,... (skip organizer already removed)
      - build bt_final
      - compute ratio using Excel running columns => O = M_last/N_last
      - choose q with minimum ratio (strictly improving over cr1)
    """
    best_q: Optional[int] = None
    best_ratio: float = float(cr1_init)

    for fi in P:  # P is already original order in greedy_grouping.py
        bt_final = build_bt_candidate(
            R,
            X[:, fi],
            y,
            alpha=alpha,
            majorizer=majorizer,
            k1_label=k1_label,
            k2_label=k2_label,
        )

        # Excel ratio (O) for selection
        ratio = excel_ratio_from_bt(
            bt_final, y, k1_label=k1_label, k2_label=k2_label
        )

        if DEBUG_PRINT_EXCEL_COLUMNS:
            dbg = build_bt_candidate_debug(
                R,
                X[:, fi],
                y,
                alpha=alpha,
                majorizer=majorizer,
                k1_label=k1_label,
                k2_label=k2_label,
            )
            trace = excel_step3_trace_from_bt(
                dbg["H_bt_final"],
                y,
                k1_label=k1_label,
                k2_label=k2_label,
            )

            print("\n==============================")
            print(f"STEP 3 CANDIDATE fi = {fi}")
            print("==============================")

            _print_vec("B (R)", dbg["B_R"])
            _print_vec("D (eta)", dbg["D_eta"])
            _print_vec("F (bt = R+eta)", dbg["F_base_bt"])
            _print_vec("H (bt after reg/majorizer)", dbg["H_bt_final"])

            _print_vec("I (M1 Σ)", trace["I_M1_sum"])
            _print_vec("J (M2 Σ)", trace["J_M2_sum"])
            _print_vec("K (K1 mean)", trace["K_K1_mean"])
            _print_vec("L (K2 mean)", trace["L_K2_mean"])
            _print_vec("M (theta cum)", trace["M_theta_cum"])
            _print_vec("N (gamma cum)", trace["N_gamma_cum"])
            print("O (theta/gamma final) =", float(trace["O_ratio_final"][0]))

        if ratio < best_ratio:
            best_ratio = float(ratio)
            best_q = int(fi)

    if best_q is None:
        # deterministic fallback: first in P
        best_q = int(P[0])
        bt_final = build_bt_candidate(
            R,
            X[:, best_q],
            y,
            alpha=alpha,
            majorizer=majorizer,
            k1_label=k1_label,
            k2_label=k2_label,
        )
        best_ratio = float(excel_ratio_from_bt(bt_final, y, k1_label=k1_label, k2_label=k2_label))

    return SelectionResult(q=best_q, best_ratio=best_ratio)