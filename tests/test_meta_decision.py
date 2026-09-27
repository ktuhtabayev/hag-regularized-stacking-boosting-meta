"""META Step 4 decision rule: score1=|B1|/|K1| vs score2=|B2|/|K2|."""

from __future__ import annotations

import pytest

from hag_regularized_stacking_boosting_meta.algorithms.meta.decision import decide_class


K1, K2 = 1, 2


def test_known_case_1_of_4_vs_2_of_6_predicts_k2() -> None:
    res = decide_class(b1_size=1, b2_size=2, k1_size=4, k2_size=6, k1_label=K1, k2_label=K2)

    assert res.score1 == pytest.approx(1 / 4)
    assert res.score2 == pytest.approx(2 / 6)
    assert res.predicted_label == K2


def test_higher_k1_score_predicts_k1() -> None:
    res = decide_class(b1_size=3, b2_size=1, k1_size=4, k2_size=6, k1_label=K1, k2_label=K2)

    assert res.predicted_label == K1


def test_tie_predicts_0() -> None:
    res = decide_class(b1_size=2, b2_size=3, k1_size=4, k2_size=6, k1_label=K1, k2_label=K2)

    assert res.score1 == pytest.approx(res.score2)
    assert res.predicted_label == 0


def test_empty_filtered_sets_tie_to_0() -> None:
    res = decide_class(b1_size=0, b2_size=0, k1_size=4, k2_size=6, k1_label=K1, k2_label=K2)

    assert res.predicted_label == 0


@pytest.mark.parametrize("k1_size, k2_size", [(0, 6), (4, 0)])
def test_missing_class_raises(k1_size: int, k2_size: int) -> None:
    with pytest.raises(ValueError):
        decide_class(b1_size=0, b2_size=0, k1_size=k1_size, k2_size=k2_size, k1_label=K1, k2_label=K2)
