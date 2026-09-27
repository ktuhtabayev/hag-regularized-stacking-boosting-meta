from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from hag_regularized_stacking_boosting_meta.algorithms.hag.greedy_grouping import (
    HAGResult,
    greedy_hag_grouping,
)
from hag_regularized_stacking_boosting_meta.algorithms.hag.input_preparation import (
    HAGPrepResult,
    prepare_hag_inputs,
)
from hag_regularized_stacking_boosting_meta.algorithms.hag.weights import (
    NominalWeightsResult,
    QuantitativePipelineResult,
    build_nominal_contributions,
    build_quantitative_nominalization,
)
from hag_regularized_stacking_boosting_meta.algorithms.meta import MetaClassifier, MetaPredictResult
from hag_regularized_stacking_boosting_meta.algorithms.meta.new_object import (
    binarize_new_object,
    form_meta_new_object,
)
from hag_regularized_stacking_boosting_meta.algorithms.meta.training_set import (
    MetaPrepResult,
    prepare_meta_training_dataset,
)
from hag_regularized_stacking_boosting_meta.evaluation.margin_analysis import (
    Margin1DResult,
    margin_analysis_latent_matrix,
)
from hag_regularized_stacking_boosting_meta.io.configs import RunConfig
from hag_regularized_stacking_boosting_meta.io.loaders import LoadedDataset, load_dataset_bundle
from hag_regularized_stacking_boosting_meta.utils.run_manager import new_run_id


StageCallback = Callable[[str], None]


@dataclass(slots=True)
class NewObjectClassification:
    """One META classification of Snew=(a0..ap), in TUPLAM order."""

    headers: List[str]
    a_init: np.ndarray       # initial format (quantitative raw, nominal original)
    a_bin: np.ndarray        # binary format (quantitative -> {1,2} by Γc)
    source: str              # "random (seed=42)" or "edited"
    prediction: MetaPredictResult


@dataclass(slots=True)
class PipelineResult:
    """Everything one Run produces: HAG (Algorithm-1) -> META (Algorithm-2) -> margins."""

    run_id: str
    config: RunConfig
    dataset_path: Path
    dataset: LoadedDataset
    quantitative: Optional[QuantitativePipelineResult]
    nominal: Optional[NominalWeightsResult]
    prep: HAGPrepResult
    hag: HAGResult
    meta: MetaPrepResult
    gamma_map: Dict[int, float]
    classifier: MetaClassifier
    new_object: NewObjectClassification
    margins: List[Margin1DResult]
    elapsed_seconds: float

    @property
    def k1_label(self) -> int:
        return int(self.config.hag.k1_label)

    @property
    def k2_label(self) -> int:
        return int(self.config.hag.k2_label)


def resolve_dataset_path(path: str | Path, project_root: Path | None) -> Path:
    path = Path(path)
    if project_root is not None and not path.is_absolute():
        return Path(project_root) / path
    return path


def run_pipeline(
    config: RunConfig,
    *,
    project_root: Path | None = None,
    snew_seed: int | None = None,
    on_stage: StageCallback | None = None,
) -> PipelineResult:
    """
    Single entry point for the GUI: the same stage sequence as the scripts
    (weights -> HAG prep -> HAG -> META training set -> Snew -> META -> margins),
    computed once in memory. Criterion-1 / nominal weights are computed a single
    time and reused by the later stages.
    """

    def stage(message: str) -> None:
        if on_stage is not None:
            on_stage(message)

    started = time.perf_counter()
    dataset_path = resolve_dataset_path(config.dataset.path, project_root)

    stage("Loading dataset")
    dataset = load_dataset_bundle(replace(config.dataset, path=str(dataset_path)))

    quantitative: Optional[QuantitativePipelineResult] = None
    if dataset.quantitative_idx:
        stage("Quantitative weights (Criterion-1, Γc, η)")
        quantitative = build_quantitative_nominalization(
            dataset.X, dataset.y, dataset.quantitative_idx
        )

    nominal: Optional[NominalWeightsResult] = None
    if dataset.nominal_idx:
        stage("Nominal weights (λ, β, ω, η)")
        nominal = build_nominal_contributions(dataset.X, dataset.y, dataset.nominal_idx)

    stage("HAG preparation (merged contributions, weight ranking)")
    prep = prepare_hag_inputs(
        X=dataset.X,
        y=dataset.y,
        feature_types=dataset.feature_types,
        quantitative_idx=dataset.quantitative_idx,
        nominal_idx=dataset.nominal_idx,
        quantitative_result=quantitative,
        nominal_result=nominal,
    )

    stage("HAG training (Algorithm-1)")
    hag = greedy_hag_grouping(
        X=prep.X_contrib_full,
        y=dataset.y,
        weights=prep.w_full,
        params=config.hag,
        verbose=False,
    )

    stage("META training set (ai*, di*, Class)")
    meta = prepare_meta_training_dataset(
        X=dataset.X,
        y=dataset.y,
        feature_types=dataset.feature_types,
        tuplam=hag.tuplam,
        dij=hag.dij,
        quantitative_result=quantitative,
    )
    gamma_map = (
        {int(k): float(v) for k, v in quantitative.gamma.items()} if quantitative else {}
    )
    classifier = MetaClassifier(
        k1_label=int(config.hag.k1_label),
        k2_label=int(config.hag.k2_label),
    ).fit(A=meta.A, D=meta.D, y=meta.y)

    stage("META classification of a random new object")
    new_object = _random_classification(
        dataset,
        hag.tuplam,
        gamma_map,
        classifier,
        config.seed if snew_seed is None else snew_seed,
    )

    stage("Margin analysis")
    margins: List[Margin1DResult] = []
    if hag.p > 0:
        margins = margin_analysis_latent_matrix(
            hag.dij,
            dataset.y,
            k1_label=int(config.hag.k1_label),
            k2_label=int(config.hag.k2_label),
        )

    return PipelineResult(
        run_id=new_run_id(),
        config=config,
        dataset_path=dataset_path,
        dataset=dataset,
        quantitative=quantitative,
        nominal=nominal,
        prep=prep,
        hag=hag,
        meta=meta,
        gamma_map=gamma_map,
        classifier=classifier,
        new_object=new_object,
        margins=margins,
        elapsed_seconds=time.perf_counter() - started,
    )


def _random_classification(
    dataset: LoadedDataset,
    tuplam: Sequence[int],
    gamma_map: Dict[int, float],
    classifier: MetaClassifier,
    seed: int,
) -> NewObjectClassification:
    obj = form_meta_new_object(
        X=dataset.X,
        y=dataset.y,
        feature_types=dataset.feature_types,
        tuplam=tuplam,
        seed=int(seed),
        gamma_map=gamma_map,
    )
    return NewObjectClassification(
        headers=list(obj.a_headers),
        a_init=obj.a_init,
        a_bin=obj.a_bin,
        source=f"random (seed={int(seed)})",
        prediction=classifier.predict(obj.a_bin.astype(int), return_debug=True),
    )


def random_new_object(result: PipelineResult, seed: int) -> NewObjectClassification:
    """Random Snew in the dataset's own value format, classified by META."""
    return _random_classification(
        result.dataset, result.hag.tuplam, result.gamma_map, result.classifier, seed
    )


def classify_new_object(
    result: PipelineResult,
    a_init: Sequence[float],
    *,
    source: str = "edited",
) -> NewObjectClassification:
    """Binarize initial-format Snew values by Γc and classify them with META."""
    a_bin = binarize_new_object(
        a_init,
        tuplam=result.hag.tuplam,
        feature_types=result.dataset.feature_types,
        gamma_map=result.gamma_map,
    )
    return NewObjectClassification(
        headers=list(result.new_object.headers),
        a_init=np.asarray([float(v) for v in a_init], dtype=float),
        a_bin=a_bin,
        source=source,
        prediction=result.classifier.predict(a_bin.astype(int), return_debug=True),
    )
