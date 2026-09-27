from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class MajorizingConfig:
    """
    Majorizing function configuration for HAG regularization.

    Excel experiments are sigmoid-based.
    """
    name: str = "sigmoid"
    params: Dict[str, Any] = field(default_factory=lambda: {"k": 1.0, "x0": 0.0})


@dataclass(frozen=True)
class HAGParams:
    """
    Params for Algorithm-1 (Greedy HAG + Regularization + Latent).

    Important:
    - Excel uses 1-based indices; code uses 0-based.
    - weights are used ONLY ONCE to select the organizer: the feature with the highest weight
      (x3 in Excel -> index 2 on the default dataset).
    """
    alpha: float = 0.3
    delta: float = 0.1
    kappa: int = 5
    cr1: float = 10.0

    k1_label: int = 1
    k2_label: int = 2

    majorizing: MajorizingConfig = field(default_factory=MajorizingConfig)