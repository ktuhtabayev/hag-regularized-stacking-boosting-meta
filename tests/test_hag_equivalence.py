"""
The vectorized HAG building blocks reproduce the original element-by-element loops
bit for bit. The loops below are the reference implementations they replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from hag_regularized_stacking_boosting_meta.algorithms.hag.theta_gamma import (
    excel_ratio_from_bt,
    excel_step3_trace_from_bt,
)
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights_nominal import (
    build_nominal_contributions,
)
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights_quantitative import criterion_1


# ------------------------------------------------------------------
# Reference implementations (the former loops)
# ------------------------------------------------------------------

def criterion_1_loop(column, target):
    K1 = target.count(1)
    K2 = target.count(2)
    pairs = sorted([[column[i], target[i]] for i in range(len(target))], key=lambda x: x[0])
    column_sorted = [p[0] for p in pairs]
    target_sorted = [p[1] for p in pairs]

    best_score = float("-inf")
    best_pi2 = column_sorted[0]
    for boundary in range(1, len(column_sorted) + 1):
        if boundary != len(column_sorted) and abs(column_sorted[boundary] - column_sorted[boundary - 1]) < 1e-8:
            continue
        left = target_sorted[:boundary]
        right = target_sorted[boundary:]
        left_K1 = sum(1 for el in left if el == 1)
        left_K2 = sum(1 for el in left if el == 2)
        right_K1 = sum(1 for el in right if el == 1)
        right_K2 = sum(1 for el in right if el == 2)
        left_numer = (
            (left_K1**2 - left_K1) + (right_K1**2 - right_K1) +
            (left_K2**2 - left_K2) + (right_K2**2 - right_K2)
        )
        left_denom = (K1**2 - K1) + (K2**2 - K2)
        if left_denom == 0:
            continue
        right_numer = (
            left_K1 * (K2 - left_K2) + left_K2 * (K1 - left_K1) +
            right_K1 * (K2 - right_K2) + right_K2 * (K1 - right_K1)
        )
        score = (left_numer / left_denom) * (right_numer / (2 * K1 * K2))
        if score > best_score:
            best_score = score
            best_pi2 = column_sorted[boundary - 1]
    return float(best_score), float(min(column_sorted)), float(best_pi2), float(max(column_sorted))


def step3_trace_loop(h, y, k1, k2):
    m = h.shape[0]
    cnt1 = int((y == k1).sum())
    cnt2 = int((y == k2).sum())
    cols = {name: np.zeros(m) for name in ("I", "J", "K", "L", "M", "N")}
    s1 = s2 = th = ga = 0.0
    for t in range(m):
        if y[t] == k1:
            s1 += h[t]
        elif y[t] == k2:
            s2 += h[t]
        cols["I"][t], cols["J"][t] = s1, s2
        cols["K"][t], cols["L"][t] = s1 / cnt1, s2 / cnt2
        if y[t] == k1:
            th += abs(h[t] - cols["K"][t])
            ga += abs(h[t] - cols["L"][t])
        elif y[t] == k2:
            th += abs(h[t] - cols["L"][t])
            ga += abs(h[t] - cols["K"][t])
        cols["M"][t], cols["N"][t] = th, ga
    cols["O"] = float("inf") if abs(ga) < 1e-12 else th / ga
    return cols


def nominal_counts_loop(col, y):
    g1, g2 = {}, {}
    for v, cls in zip(col, y):
        target = g1 if cls == 1 else g2
        target[float(v)] = target.get(float(v), 0) + 1
    keys = sorted(set(g1) | set(g2))
    return {k: g1.get(k, 0) for k in keys}, {k: g2.get(k, 0) for k in keys}


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 120))
    y = rng.integers(1, 3, size=m)
    y[0], y[-1] = 1, 2  # both classes present
    kind = seed % 4
    if kind == 0:
        column = rng.normal(size=m)
    elif kind == 1:
        column = rng.integers(0, 6, size=m).astype(float)  # many ties
    elif kind == 2:
        base = rng.integers(0, 4, size=m).astype(float)
        column = base + rng.choice([0.0, 5e-9, 2e-8], size=m)  # near-duplicates around 1e-8
    else:
        column = np.round(rng.uniform(-3, 3, size=m), 2)
    return column, y


SEEDS = range(60)


@pytest.mark.parametrize("seed", SEEDS)
def test_criterion_1_matches_the_loop_bit_for_bit(seed: int) -> None:
    column, y = _random_case(seed)
    fast = criterion_1(column, y)
    weight, pi1, pi2, pi3 = criterion_1_loop(column.tolist(), y.tolist())
    assert (fast.weight_wc, fast.pi1, fast.pi2, fast.pi3) == (weight, pi1, pi2, pi3)


def test_criterion_1_single_object_per_class_has_no_score() -> None:
    # (|K1|² − |K1|) + (|K2|² − |K2|) = 0: every split is skipped, as in the loop
    fast = criterion_1([3.0, 1.0], [1, 2])
    assert (fast.weight_wc, fast.pi1, fast.pi2, fast.pi3) == criterion_1_loop([3.0, 1.0], [1, 2])
    assert fast.weight_wc == float("-inf") and fast.pi2 == 1.0


@pytest.mark.parametrize("bad", [([], []), ([1.0, 2.0], [1]), ([1.0, 2.0], [1, 1])])
def test_criterion_1_rejects_invalid_input(bad) -> None:
    with pytest.raises(ValueError):
        criterion_1(*bad)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("labels", [(1, 2), (2, 1)])
def test_step3_trace_matches_the_loop_bit_for_bit(seed: int, labels) -> None:
    rng = np.random.default_rng(1000 + seed)
    m = int(rng.integers(2, 150))
    h = rng.normal(scale=rng.choice([1e-3, 1.0, 1e3]), size=m)
    y = rng.integers(1, 4, size=m)  # label 3 belongs to neither class
    y[0], y[-1] = 1, 2
    k1, k2 = labels

    fast = excel_step3_trace_from_bt(h, y, k1_label=k1, k2_label=k2)
    slow = step3_trace_loop(h, y, k1, k2)

    for key, name in [("I_M1_sum", "I"), ("J_M2_sum", "J"), ("K_K1_mean", "K"),
                      ("L_K2_mean", "L"), ("M_theta_cum", "M"), ("N_gamma_cum", "N")]:
        assert fast[key].tobytes() == slow[name].tobytes(), key
    assert fast["O_ratio_final"][0] == slow["O"]
    assert excel_ratio_from_bt(h, y, k1_label=k1, k2_label=k2) == slow["O"]


def test_step3_trace_without_one_class_is_infinite() -> None:
    trace = excel_step3_trace_from_bt(np.ones(3), np.array([1, 1, 1]), k1_label=1, k2_label=2)
    assert trace["O_ratio_final"][0] == float("inf")


@pytest.mark.parametrize("seed", range(20))
def test_nominal_weights_match_the_loop(seed: int) -> None:
    rng = np.random.default_rng(2000 + seed)
    m, n = int(rng.integers(4, 80)), 3
    X = rng.integers(0, int(rng.integers(2, 6)), size=(m, n)).astype(float)
    X[:, 2] = X[:, 2] / 2 - 1  # non-integer and negative gradations
    y = rng.integers(1, 3, size=m)
    y[0], y[-1] = 1, 2

    res = build_nominal_contributions(X, y, [0, 1, 2])

    for fidx in range(n):
        g1, g2 = nominal_counts_loop(X[:, fidx], y)
        assert res.g_k1[fidx] == g1 and res.g_k2[fidx] == g2
        assert list(res.g_k1[fidx]) == list(g1)  # same (sorted) gradation order
        assert res.gradations[fidx] == list(g1)
        eta = res.contributions_eta[fidx]
        expected = np.array([eta[float(v)] for v in X[:, fidx]])
        assert res.contribution_X[:, fidx].tobytes() == expected.tobytes()
