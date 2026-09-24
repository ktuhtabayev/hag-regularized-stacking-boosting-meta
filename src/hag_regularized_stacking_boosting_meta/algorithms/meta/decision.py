"""META Algorithm decision rule (Step 4).

After filtering finishes at j = p, we compute:
  score1 = |B1(a_p)| / |K1|
  score2 = |B2(a_p)| / |K2|

Decision:
  if score1 > score2 -> return K1
  if score1 < score2 -> return K2
  else               -> return 0 (tie/unknown)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaDecisionResult:
    predicted_label: int  # k1_label, k2_label or 0
    score1: float
    score2: float
    b1_size: int
    b2_size: int
    k1_size: int
    k2_size: int


def decide_class(
    *,
    b1_size: int,
    b2_size: int,
    k1_size: int,
    k2_size: int,
    k1_label: int,
    k2_label: int,
) -> MetaDecisionResult:
    if k1_size <= 0 or k2_size <= 0:
        raise ValueError("Both classes must be present in training set")

    score1 = float(b1_size) / float(k1_size)
    score2 = float(b2_size) / float(k2_size)

    if score1 > score2:
        pred = int(k1_label)
    elif score1 < score2:
        pred = int(k2_label)
    else:
        pred = 0

    return MetaDecisionResult(
        predicted_label=pred,
        score1=score1,
        score2=score2,
        b1_size=int(b1_size),
        b2_size=int(b2_size),
        k1_size=int(k1_size),
        k2_size=int(k2_size),
    )


