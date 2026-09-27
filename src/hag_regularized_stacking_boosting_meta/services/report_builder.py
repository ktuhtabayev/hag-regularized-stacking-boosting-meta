"""
DataFrame views of a PipelineResult, shared by the GUI tables and the CSV export.

Indexing follows the project rule: everything is 0-based.
  objects  -> S0, S1, ...   (row index; also the ids in META B1/B2 sets)
  features -> x0, x1, ...   (column index in the dataset)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from hag_regularized_stacking_boosting_meta.services.runner import (
    NewObjectClassification,
    PipelineResult,
)


def object_ids(count: int) -> List[str]:
    return [f"S{i}" for i in range(count)]


def feature_name(index: int) -> str:
    return f"x{int(index)}"


def feature_type_name(result: PipelineResult, index: int) -> str:
    return "Quantitative" if int(result.dataset.feature_types[int(index)]) == 1 else "Nominal"


def class_name(label: int, result: PipelineResult) -> str:
    if label == result.k1_label:
        return f"Class {label} (K1)"
    if label == result.k2_label:
        return f"Class {label} (K2)"
    return "0 (undecided)"


def _as_number(value: float) -> float | int:
    value = float(value)
    return int(round(value)) if abs(value - round(value)) < 1e-9 else value


def _object_table(result: PipelineResult, columns: Dict[str, np.ndarray]) -> pd.DataFrame:
    y = result.dataset.y
    frame = pd.DataFrame({"Object": object_ids(len(y))})
    for name, values in columns.items():
        frame[name] = np.asarray(values)
    frame["Class"] = y.astype(int)
    return frame


def _matrix_by_features(
    result: PipelineResult,
    matrix: np.ndarray,
    feature_idx: Sequence[int],
) -> pd.DataFrame:
    return _object_table(
        result,
        {feature_name(fidx): matrix[:, j] for j, fidx in enumerate(feature_idx)},
    )


def _format_ids(ids: Sequence[int], max_ids: Optional[int]) -> str:
    names = [f"S{int(i)}" for i in ids]
    if max_ids is not None and len(names) > max_ids:
        return "{" + ", ".join(names[:max_ids]) + f", … (+{len(names) - max_ids})" + "}"
    return "{" + ", ".join(names) + "}"


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

def dataset_frame(result: PipelineResult) -> pd.DataFrame:
    X = result.dataset.X
    return _object_table(
        result,
        {feature_name(j): [_as_number(v) for v in X[:, j]] for j in range(X.shape[1])},
    )


def feature_summary_frame(result: PipelineResult) -> pd.DataFrame:
    tuplam_position = {fidx: pos for pos, fidx in enumerate(result.hag.tuplam)}
    rows = []
    for j in range(result.dataset.X.shape[1]):
        position = tuplam_position.get(j)
        rows.append(
            {
                "Feature": feature_name(j),
                "Type": feature_type_name(result, j),
                "Weight ω": float(result.prep.w_full[j]),
                "Rank": int(result.prep.weight_rank_per_feature[j]),
                "Γc": result.gamma_map.get(j, np.nan),
                "TUPLAM": "" if position is None else f"a{position}",
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Weights: quantitative (Criterion-1) and nominal (λ, β, ω)
# ------------------------------------------------------------------

def criterion1_frame(result: PipelineResult) -> pd.DataFrame:
    q = result.quantitative
    if q is None:
        return pd.DataFrame()
    rows = []
    for fidx in q.quantitative_idx:
        pi1, pi2, pi3 = q.pi_table[fidx]
        eta = q.contributions_eta[fidx]
        rows.append(
            {
                "Feature": feature_name(fidx),
                "π1 (min)": pi1,
                "π2 (split)": pi2,
                "π3 (max)": pi3,
                "Weight ω": q.weights_wc[fidx],
                "Γc": q.gamma[fidx],
                "η(1)": eta[1],
                "η(2)": eta[2],
            }
        )
    return pd.DataFrame(rows)


def quantitative_binary_frame(result: PipelineResult) -> pd.DataFrame:
    q = result.quantitative
    if q is None:
        return pd.DataFrame()
    return _matrix_by_features(result, q.binary_X.astype(int), q.quantitative_idx)


def quantitative_contrib_frame(result: PipelineResult) -> pd.DataFrame:
    q = result.quantitative
    if q is None:
        return pd.DataFrame()
    return _matrix_by_features(result, q.contribution_X, q.quantitative_idx)


def nominal_weights_frame(result: PipelineResult) -> pd.DataFrame:
    n = result.nominal
    if n is None:
        return pd.DataFrame()
    rows = []
    for fidx in n.nominal_idx:
        rows.append(
            {
                "Feature": feature_name(fidx),
                "Gradations μ": n.p_c[fidx],
                "l1 (in K1)": n.l1_c[fidx],
                "l2 (in K2)": n.l2_c[fidx],
                "D1": n.D1_c[fidx],
                "D2": n.D2_c[fidx],
                "λ": n.lambda_c[fidx],
                "β": n.beta_c[fidx],
                "Weight ω": n.weight_wc[fidx],
            }
        )
    return pd.DataFrame(rows)


def nominal_gradations_frame(result: PipelineResult) -> pd.DataFrame:
    n = result.nominal
    if n is None:
        return pd.DataFrame()
    rows = []
    for fidx in n.nominal_idx:
        for gradation in n.gradations[fidx]:
            rows.append(
                {
                    "Feature": feature_name(fidx),
                    "Gradation": _as_number(gradation),
                    "Count in K1": n.g_k1[fidx][gradation],
                    "Count in K2": n.g_k2[fidx][gradation],
                    "η": n.contributions_eta[fidx][gradation],
                }
            )
    return pd.DataFrame(rows)


def nominal_contrib_frame(result: PipelineResult) -> pd.DataFrame:
    n = result.nominal
    if n is None:
        return pd.DataFrame()
    return _matrix_by_features(result, n.contribution_X, n.nominal_idx)


# ------------------------------------------------------------------
# HAG (Algorithm-1)
# ------------------------------------------------------------------

def merged_contrib_frame(result: PipelineResult) -> pd.DataFrame:
    n_features = result.prep.X_contrib_full.shape[1]
    return _matrix_by_features(result, result.prep.X_contrib_full, range(n_features))


def weight_ranking_frame(result: PipelineResult) -> pd.DataFrame:
    rows = [
        {
            "Rank": rank,
            "Feature": feature_name(fidx),
            "Type": feature_type_name(result, fidx),
            "Weight ω": float(result.prep.w_full[fidx]),
        }
        for rank, fidx in enumerate(result.prep.weight_sorted_feature_idx)
    ]
    return pd.DataFrame(rows)


def hag_iterations_frame(result: PipelineResult) -> pd.DataFrame:
    """Step 1 organizer, then one row per Step 3-4 iteration with the stop rule."""
    hag = result.hag
    params = result.config.hag
    rows = [
        {
            "Step t": 0,
            "Selected": f"u = {feature_name(hag.tuplam[0])}",
            "θ/γ": np.nan,
            "|TUPLAM|": 1,
            "Latent": "",
            "Next": "organizer",
        }
    ]
    last_t = len(hag.crit_history)
    for t, (q, crit) in enumerate(zip(hag.tuplam[1:], hag.crit_history), start=1):
        size = t + 1
        if size >= int(params.kappa):
            decision = f"stop: |TUPLAM| = κ = {params.kappa}"
        elif crit <= float(params.delta):
            decision = f"stop: θ/γ ≤ δ = {params.delta}"
        elif t == last_t:
            decision = "stop: no candidates left"
        else:
            decision = "continue"
        rows.append(
            {
                "Step t": t,
                "Selected": f"q = {feature_name(q)}",
                "θ/γ": crit,
                "|TUPLAM|": size,
                "Latent": f"r{t}",
                "Next": decision,
            }
        )
    return pd.DataFrame(rows)


def step3_scan_frame(result: PipelineResult) -> pd.DataFrame:
    """θ/γ of every candidate per iteration (Excel column O); blank = already selected."""
    hag = result.hag
    candidates = [j for j in range(result.dataset.X.shape[1]) if j != hag.organizer]
    rows = []
    for t, scan in enumerate(hag.candidate_history, start=1):
        row: Dict[str, object] = {"Step t": t, "Selected": feature_name(hag.tuplam[t])}
        for j in candidates:
            row[feature_name(j)] = scan.get(j, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def latent_frame(result: PipelineResult) -> pd.DataFrame:
    dij = result.hag.dij
    return _object_table(result, {f"r{j + 1}": dij[:, j] for j in range(result.hag.p)})


# ------------------------------------------------------------------
# META (Algorithm-2)
# ------------------------------------------------------------------

def meta_training_frame(result: PipelineResult) -> pd.DataFrame:
    meta = result.meta
    columns: Dict[str, np.ndarray] = {}
    for j, header in enumerate(meta.headers[: meta.A.shape[1]]):
        columns[header] = meta.A[:, j]
    for k, header in enumerate(meta.headers[meta.A.shape[1]: -1]):
        columns[header] = meta.D[:, k]
    return _object_table(result, columns)


def new_object_frame(result: PipelineResult, snew: NewObjectClassification) -> pd.DataFrame:
    rows = []
    for header, fidx, init, binary in zip(snew.headers, result.hag.tuplam, snew.a_init, snew.a_bin):
        rows.append(
            {
                "Component": header,
                "Type": feature_type_name(result, fidx),
                "Γc": result.gamma_map.get(int(fidx), np.nan),
                "Initial value": _as_number(init),
                "Binary value": _as_number(binary),
            }
        )
    # object dtype keeps integer-valued entries as ints next to float ones
    return pd.DataFrame(rows, dtype=object)


def filtering_frame(
    result: PipelineResult,
    snew: NewObjectClassification,
    *,
    max_ids: Optional[int] = None,
) -> pd.DataFrame:
    """B1/B2 after each META filtering step j (Step 1: j=0, Step 2: j=1..p)."""
    debug = snew.prediction.debug
    if debug is None:
        return pd.DataFrame()
    rows = []
    for j in sorted(debug.b1_history):
        header = snew.headers[j]
        value = _as_number(snew.a_bin[j])
        if j == 0:
            rule = f"a{j} = ai{j}"
        else:
            rule = f"a{j} = ai{j}; K1: di{j} > 0, K2: di{j} < 0"
        b1 = debug.b1_history[j]
        b2 = debug.b2_history[j]
        rows.append(
            {
                "Step": 1 if j == 0 else 2,
                "j": j,
                "Component": header,
                "Value": value,
                "Rule": rule,
                "|B1|": len(b1),
                "|B2|": len(b2),
                "B1": _format_ids(b1, max_ids),
                "B2": _format_ids(b2, max_ids),
            }
        )
    return pd.DataFrame(rows)


def decision_frame(result: PipelineResult, snew: NewObjectClassification) -> pd.DataFrame:
    d = snew.prediction.decision
    p = result.hag.p
    rows = [
        ("|K1|", d.k1_size),
        ("|K2|", d.k2_size),
        (f"|B1(a{p})|", d.b1_size),
        (f"|B2(a{p})|", d.b2_size),
        ("score1 = |B1| / |K1|", d.score1),
        ("score2 = |B2| / |K2|", d.score2),
        ("Snew ∈", class_name(d.predicted_label, result)),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"]).astype({"Value": object})


# ------------------------------------------------------------------
# Margin analysis
# ------------------------------------------------------------------

def margin_report_frame(result: PipelineResult) -> pd.DataFrame:
    y = result.dataset.y
    rows = []
    for j, margin in enumerate(result.margins, start=1):
        rows.append(
            {
                "Latent": f"r{j}",
                "Left boundary (max K2)": margin.left_boundary,
                "Right boundary (min K1)": margin.right_boundary,
                "Midpoint": margin.midpoint,
                "Width": margin.width,
                "Max K2 object": f"S{margin.left_argmax_idx}",
                "Min K1 object": f"S{margin.right_argmin_idx}",
                "Misclassified": int(np.sum(margin.yhat != y)),
            }
        )
    return pd.DataFrame(rows)


def object_margins_frame(result: PipelineResult, latent_index: int) -> pd.DataFrame:
    """Per-object margins for latent feature r_{latent_index+1} (0-based index)."""
    margin = result.margins[latent_index]
    y = result.dataset.y
    frame = pd.DataFrame(
        {
            "Object": object_ids(len(y)),
            "d": result.hag.dij[:, latent_index],
            "Class": y.astype(int),
            "Object margin": margin.object_margin,
            "ŷ": margin.yhat.astype(int),
        }
    )
    frame["Correct"] = np.where(margin.yhat == y, "yes", "no")
    return frame


# ------------------------------------------------------------------
# Run summary
# ------------------------------------------------------------------

def run_summary(result: PipelineResult) -> Dict[str, object]:
    ds = result.dataset
    params = result.config.hag
    prediction = result.new_object.prediction
    return {
        "run_id": result.run_id,
        "dataset": str(result.dataset_path),
        "objects": int(ds.X.shape[0]),
        "features": int(ds.X.shape[1]),
        "quantitative_idx": [int(i) for i in ds.quantitative_idx],
        "nominal_idx": [int(i) for i in ds.nominal_idx],
        "class_counts": {
            "K1": int(np.sum(ds.y == result.k1_label)),
            "K2": int(np.sum(ds.y == result.k2_label)),
        },
        "hag": {
            "alpha": float(params.alpha),
            "delta": float(params.delta),
            "kappa": int(params.kappa),
            "cr1": float(params.cr1),
            "majorizing": {
                "name": str(params.majorizing.name),
                "params": dict(params.majorizing.params or {}),
            },
        },
        "organizer": int(result.hag.organizer),
        "tuplam": [int(i) for i in result.hag.tuplam],
        "p_latent": int(result.hag.p),
        "crit_history": [float(c) for c in result.hag.crit_history],
        "new_object": {
            "source": result.new_object.source,
            "headers": list(result.new_object.headers),
            "initial": [_as_number(v) for v in result.new_object.a_init],
            "binary": [_as_number(v) for v in result.new_object.a_bin],
            "predicted_label": int(prediction.predicted_label),
            "score1": float(prediction.decision.score1),
            "score2": float(prediction.decision.score2),
        },
        "elapsed_seconds": round(float(result.elapsed_seconds), 3),
    }
