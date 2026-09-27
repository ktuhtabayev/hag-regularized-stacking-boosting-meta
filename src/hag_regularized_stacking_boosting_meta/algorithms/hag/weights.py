"""
algorithms/hag/weights.py  (STABLE FACADE)

Keep this file stable so the rest of the project (GUI, services, scripts)
always imports "weights" from ONE place.

- Quantitative weights: weights_quantitative.py
- Nominal weights:      weights_nominal.py
"""

from __future__ import annotations

from .weights_nominal import (
    NominalWeightsResult,
    build_nominal_contributions,
)
from .weights_quantitative import (
    # --- data containers ---
    ContributionResult,
    Criterion1Result,
    GradationCounts,
    QuantitativePipelineResult,
    # --- core steps ---
    apply_contributions,
    binarize_by_gamma,
    compute_gamma_c,
    contributions_eta,
    criterion_1,
    gradation_counts,
    # --- full pipeline ---
    build_quantitative_nominalization,
)

__all__ = [
    # Quantitative
    "Criterion1Result",
    "GradationCounts",
    "ContributionResult",
    "QuantitativePipelineResult",
    "criterion_1",
    "compute_gamma_c",
    "binarize_by_gamma",
    "gradation_counts",
    "contributions_eta",
    "apply_contributions",
    "build_quantitative_nominalization",
    # Nominal
    "NominalWeightsResult",
    "build_nominal_contributions",
]
