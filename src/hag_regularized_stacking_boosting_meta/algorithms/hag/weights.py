"""
algorithms/hag/weights.py  (STABLE FACADE)

Keep this file stable so the rest of the project (GUI, services, scripts)
always imports "weights" from ONE place.

- Quantitative weights: weights_quantitative.py
- Nominal weights:      weights_nominal.py
"""

from __future__ import annotations

# ============================================================
# Quantitative exports (CURRENT)
# We keep a "_QUANT_AVAILABLE" flag for symmetry with nominal,
# even though weights_quantitative.py is expected to always exist.
# ============================================================

_QUANT_AVAILABLE = False
try:
    from .weights_quantitative import (  # noqa: F401
        # --- data containers ---
        Criterion1Result,
        GradationCounts,
        ContributionResult,
        QuantitativePipelineResult,
        QuantFeatureBinarization,
        # --- core steps ---
        criterion_1,
        compute_gamma_c,
        binarize_by_gamma,
        gradation_counts,
        contributions_eta,
        apply_contributions,
        # --- full pipeline ---
        build_quantitative_nominalization,
    )

    _QUANT_AVAILABLE = True
except Exception:
    _QUANT_AVAILABLE = False


# ============================================================
# Nominal exports (NOW / FUTURE)
# ============================================================

_NOMINAL_AVAILABLE = False
try:
    from .weights_nominal import (  # noqa: F401
        NominalWeightsResult,
        build_nominal_contributions,
    )

    _NOMINAL_AVAILABLE = True
except Exception:
    _NOMINAL_AVAILABLE = False


__all__ = []

if _QUANT_AVAILABLE:
    __all__ += [
        # Quantitative
        "Criterion1Result",
        "GradationCounts",
        "ContributionResult",
        "QuantitativePipelineResult",
        "QuantFeatureBinarization",
        "criterion_1",
        "compute_gamma_c",
        "binarize_by_gamma",
        "gradation_counts",
        "contributions_eta",
        "apply_contributions",
        "build_quantitative_nominalization",
    ]

if _NOMINAL_AVAILABLE:
    __all__ += [
        # Nominal
        "NominalWeightsResult",
        "build_nominal_contributions",
    ]
