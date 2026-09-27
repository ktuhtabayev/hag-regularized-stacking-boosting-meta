from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import (
    build_quantitative_nominalization,
)


@dataclass(frozen=True)
class MetaNewObject:
    """
    New object Snew=(a0..ap) based on TUPLAM order (0-based indices).

    IMPORTANT:
    - This file uses labels a0(x2), a1(x5), ... (NO 'i') ONLY FOR NEW OBJECT.
    - META training dataset headers remain unchanged elsewhere.

    a_init: initial-format values (nominal original, quantitative raw but in SAME FORMAT as dataset)
    a_bin:  binary-format values (nominal same as init, quantitative -> {1,2})
    """
    a_headers: List[str]
    a_init: np.ndarray
    a_bin: np.ndarray
    tuplam: List[int]
    gamma_map: Dict[int, float]


def _as_int_if_possible(x: float) -> float | int:
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return float(x)


def _compute_gamma_map(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: np.ndarray,
) -> Dict[int, float]:
    """
    Build Γc thresholds for ALL quantitative features using existing quantitative pipeline.
    Returns dict: feature_index -> gamma_c
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    feature_types = np.asarray(feature_types, dtype=int).reshape(-1)

    n = X.shape[1]
    quant_idx = [i for i in range(n) if feature_types[i] == 1]
    if not quant_idx:
        return {}

    q_res = build_quantitative_nominalization(X, y, quant_idx)
    return {int(k): float(v) for k, v in (q_res.gamma or {}).items()}


def quantitative_to_binary(x: float, gamma_c: float) -> int:
    """Γc binarization of one quantitative value: x <= Γc -> 1, else 2."""
    return 1 if float(x) <= float(gamma_c) else 2


def binarize_new_object(
    a_init: Sequence[float],
    *,
    tuplam: Sequence[int],
    feature_types: np.ndarray,
    gamma_map: Dict[int, float],
) -> np.ndarray:
    """
    Initial-format Snew values (TUPLAM order) -> binary format:
      quantitative -> {1,2} by Γc, nominal -> unchanged.
    Used to classify a user-entered new object.
    """
    feature_types = np.asarray(feature_types, dtype=int).reshape(-1)
    values = [float(v) for v in a_init]
    if len(values) != len(tuplam):
        raise ValueError(f"Snew needs {len(tuplam)} values (one per TUPLAM feature), got {len(values)}")

    out: List[float] = []
    for x, fidx in zip(values, tuplam):
        if feature_types[int(fidx)] == 1:
            gamma = gamma_map.get(int(fidx))
            if gamma is None:
                raise RuntimeError(f"No gamma threshold found for quantitative feature {fidx}.")
            out.append(quantitative_to_binary(x, gamma))
        else:
            out.append(float(_as_int_if_possible(x)))
    return np.asarray(out, dtype=float)


def _column_is_integer_valued(col: np.ndarray, *, atol: float = 1e-9) -> bool:
    """
    True if all values in the column are essentially integers.
    """
    col = np.asarray(col, dtype=float).reshape(-1)
    return bool(np.allclose(col, np.round(col), atol=atol))


def _random_nominal_value(col: np.ndarray, rng: np.random.Generator) -> float:
    col = np.asarray(col).reshape(-1)
    uniq = np.unique(col)
    return float(rng.choice(uniq))


def _random_quant_value(col: np.ndarray, rng: np.random.Generator) -> float:
    """
    Quantitative random value in the SAME FORMAT as dataset.

    If the dataset column values are integers (most of your datasets),
    we generate an integer (either sampled from existing values or randint).

    Otherwise we generate a float in [min, max].
    """
    col = np.asarray(col, dtype=float).reshape(-1)

    # If column is integer-valued, keep integer format.
    if _column_is_integer_valued(col):
        # Prefer sampling from existing values (most dataset-faithful)
        uniq = np.unique(np.round(col).astype(int))
        return float(rng.choice(uniq))

        # Alternative (also valid): random int between min/max inclusive
        # mn = int(np.min(uniq))
        # mx = int(np.max(uniq))
        # return float(rng.integers(mn, mx + 1))

    # Otherwise generate float
    mn = float(np.min(col))
    mx = float(np.max(col))
    if abs(mx - mn) < 1e-12:
        return mn
    return round(float(rng.uniform(mn, mx)), 5)


def form_meta_new_object(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_types: np.ndarray,
    tuplam: Sequence[int],
    seed: int | None = None,
    gamma_map: Dict[int, float] | None = None,
) -> MetaNewObject:
    """
    Build a random new object Snew=(a0..ap) following META prep rules:

    - Use TUPLAM order (0-based) for Snew component order.
    - For nominal features: random value from observed unique values.
    - For quantitative features: random raw value in SAME FORMAT as dataset column,
      then binarize by Γc:
        x <= Γc -> 1 else 2

    OUTPUT:
      - a_init: initial-format values (nominal original, quantitative raw)
      - a_bin:  binary-format values (nominal same, quantitative -> {1,2})

    LABELS (ONLY HERE):
      a0(x2), a1(x5), ...  (0-based x index)

    gamma_map: optional precomputed Γc per quantitative feature (avoids recomputing
    the quantitative pipeline).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    feature_types = np.asarray(feature_types, dtype=int).reshape(-1)
    tuplam = [int(i) for i in tuplam]

    if X.ndim != 2:
        raise ValueError("X must be 2D (m,n)")
    m, n = X.shape
    if y.shape[0] != m:
        raise ValueError("y length must match X rows")
    if feature_types.shape[0] != n:
        raise ValueError("feature_types length must match X columns")
    if len(tuplam) == 0:
        raise ValueError("tuplam must not be empty")

    for idx in tuplam:
        if idx < 0 or idx >= n:
            raise ValueError(f"tuplam index out of range: {idx} for n={n}")

    rng = np.random.default_rng(seed)
    if gamma_map is None:
        gamma_map = _compute_gamma_map(X, y, feature_types)

    headers: List[str] = []
    init_vals: List[float] = []
    bin_vals: List[float] = []

    for j, fidx in enumerate(tuplam):
        headers.append(f"a{j}(x{fidx})")  # no 'i'

        if feature_types[fidx] == 1:
            # quantitative: generate dataset-format value (integer if dataset uses ints)
            x_raw = _random_quant_value(X[:, fidx], rng)

            # If column is integer-valued, store as int in init list
            if _column_is_integer_valued(X[:, fidx]):
                init_vals.append(float(int(round(x_raw))))
            else:
                init_vals.append(float(x_raw))

            gamma = gamma_map.get(fidx, None)
            if gamma is None:
                raise RuntimeError(
                    f"No gamma threshold found for quantitative feature {fidx}. "
                    "Check feature_types and quantitative pipeline."
                )
            bin_vals.append(quantitative_to_binary(x_raw, gamma))
        else:
            # nominal
            x_nom = _random_nominal_value(X[:, fidx], rng)
            init_vals.append(float(_as_int_if_possible(float(x_nom))))
            bin_vals.append(float(_as_int_if_possible(float(x_nom))))

    return MetaNewObject(
        a_headers=headers,
        a_init=np.asarray(init_vals, dtype=float),
        a_bin=np.asarray(bin_vals, dtype=float),
        tuplam=tuplam,
        gamma_map=gamma_map,
    )


def format_snew(headers: List[str], values: np.ndarray) -> str:
    """
    Pretty string:
      Snew=(a0(x2)=..., a1(x5)=..., ...)
    """
    vals = np.asarray(values).reshape(-1)
    parts = []
    for h, v in zip(headers, vals):
        parts.append(f"{h}={_as_int_if_possible(float(v))}")
    return "Snew=(" + ", ".join(parts) + ")"