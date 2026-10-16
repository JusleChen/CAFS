"""Leakage-safe V3 BEM model averaging and geometry-only hybrid core.

The model latent is shared by every calibration episode in a head.  A single
strictly-positive Safe-FreeSIGM vector is likewise shared by every BEM model;
this API intentionally has no operator-specific frequency-weight argument.
Model weights are selected from held-out predictive losses and source-map
stability losses only, then frozen for evaluation inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import InverseResult, champagne_sbl
from .complex_sbl import complex_gaussian_posterior
from .v3_uncertain import safe_frequency_weights, stack_freesigm_family


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

BEM_MODEL_COUNT_GRID: tuple[int, ...] = (2, 4)
BEM_MIXTURE_TEMPERATURE_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)
GEOMETRY_HYBRID_RANK_GRID: tuple[int, ...] = (2, 4, 8)


def _readonly_complex(value: ArrayLike) -> ComplexArray:
    out = np.array(value, dtype=np.complex128, copy=True)
    out.setflags(write=False)
    return out


def _readonly_float(value: ArrayLike) -> FloatArray:
    out = np.array(value, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _validate_model_count(count: int) -> int:
    value = int(count)
    if value not in BEM_MODEL_COUNT_GRID:
        raise ValueError(f"BEM mixture model count must be one of {BEM_MODEL_COUNT_GRID}")
    return value


def _validate_temperature(value: float) -> float:
    temperature = float(value)
    if not any(
        np.isclose(temperature, allowed, atol=0.0, rtol=0.0)
        for allowed in BEM_MIXTURE_TEMPERATURE_GRID
    ):
        raise ValueError(
            "temperature must be one of the frozen values "
            f"{BEM_MIXTURE_TEMPERATURE_GRID}"
        )
    return temperature


def _validate_nominal_index(index: int, count: int) -> int:
    nominal = int(index)
    if nominal < 0 or nominal >= int(count):
        raise ValueError("nominal_index is outside the model family")
    return nominal


def _shared_frequency_weights(value: ArrayLike | None, count: int) -> FloatArray:
    if value is not None and np.asarray(value).ndim != 1:
        raise ValueError(
            "operator-specific FreeSIGM weights are forbidden; pass one shared "
            "head frequency vector"
        )
    return safe_frequency_weights(value, int(count))


def _model_priors(
    value: ArrayLike | None,
    count: int,
    nominal_index: int,
) -> FloatArray:
    if value is None:
        priors = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        priors = np.asarray(value, dtype=np.float64).reshape(-1)
        if priors.shape != (count,) or not np.all(np.isfinite(priors)):
            raise ValueError("model_priors must be finite and model aligned")
        if np.any(priors <= 0.0):
            raise ValueError("model_priors must be strictly positive")
        priors = priors / priors.sum()
    other = np.delete(priors, nominal_index)
    if other.size and priors[nominal_index] < float(np.max(other)) - 1e-15:
        raise ValueError("the nominal prior cannot be lower than any single alternative")
    return np.asarray(priors, dtype=np.float64)


def posterior_model_weights(
    model_losses: ArrayLike,
    model_priors: ArrayLike,
    *,
    temperature: float,
) -> FloatArray:
    """Evaluate ``q_k proportional pi_k exp(-(L_k-min L)/T)`` stably."""

    loss = np.asarray(model_losses, dtype=np.float64).reshape(-1)
    prior = np.asarray(model_priors, dtype=np.float64).reshape(-1)
    if loss.shape != prior.shape or loss.size < 1:
        raise ValueError("model losses and priors must align")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(prior)):
        raise ValueError("model losses and priors must be finite")
    if np.any(prior <= 0.0):
        raise ValueError("model priors must be positive")
    temperature_value = _validate_temperature(temperature)
    shifted = -(loss - float(np.min(loss))) / temperature_value
    log_weight = np.log(prior / prior.sum()) + shifted
    log_weight -= float(np.max(log_weight))
    weight = np.exp(log_weight)
    return _readonly_float(weight / weight.sum())


def _predictive_loss_tensor(
    value: ArrayLike,
    frequency_weights: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, int, int, int]:
    """Return flattened ``(heldout_fold, model, frequency)`` losses."""

    loss = np.asarray(value, dtype=np.float64)
    if loss.ndim == 2:
        # Convenience for a single-frequency experiment: fold, model.
        loss = loss[..., None]
    if loss.ndim < 3:
        raise ValueError(
            "predictive_losses must have (..., model, frequency) dimensions"
        )
    models, frequencies = int(loss.shape[-2]), int(loss.shape[-1])
    _validate_model_count(models)
    weights = _shared_frequency_weights(frequency_weights, frequencies)
    flattened = loss.reshape(-1, models, frequencies)
    return (
        np.asarray(flattened, dtype=np.float64),
        weights,
        flattened.shape[0],
        models,
        frequencies,
    )


def _stability_loss_matrix(
    value: ArrayLike,
    *,
    folds: int,
    models: int,
) -> FloatArray:
    stability = np.asarray(value, dtype=np.float64)
    if stability.shape == (models,):
        stability = np.repeat(stability[None, :], folds, axis=0)
    else:
        if stability.ndim < 2 or stability.shape[-1] != models:
            raise ValueError("source_stability_losses must end in the model dimension")
        stability = stability.reshape(-1, models)
        if stability.shape[0] != folds:
            raise ValueError(
                "source_stability_losses must align with all held-out folds/episodes"
            )
    return np.asarray(stability, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BEMModelWeightSelection:
    model_weights: FloatArray
    episode_model_weights: FloatArray
    model_priors: FloatArray
    mean_selection_loss: FloatArray
    mean_predictive_loss: FloatArray
    fold_selection_loss: FloatArray
    fold_model_order: NDArray[np.int64]
    temperature: float
    nominal_index: int
    fallback: bool
    fallback_reason: str | None
    metadata: Mapping[str, object]

    @property
    def weights(self) -> FloatArray:
        return self.model_weights


def select_bem_model_weights(
    predictive_losses: ArrayLike,
    source_stability_losses: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
    model_priors: ArrayLike | None = None,
    temperature: float = 1.0,
    nominal_index: int = 0,
    stability_weight: float = 1.0,
    complexity: ArrayLike | None = None,
    complexity_weight: float = 0.0,
    nominal_safety_margin: float = 0.0,
    ood_detected: bool = False,
) -> BEMModelWeightSelection:
    """Select one shared head-level model posterior without source truth.

    ``predictive_losses`` contains held-out episode/electrode losses with shape
    ``(..., K, F)``.  All leading dimensions are pooled before a *single*
    model latent is inferred.  A two-dimensional ``frequency_weights`` array
    is rejected, preventing operator-specific FreeSIGM fitting from changing
    the BEM ranking.
    """

    predictive, shared_weights, folds, models, frequencies = _predictive_loss_tensor(
        predictive_losses, frequency_weights
    )
    nominal = _validate_nominal_index(nominal_index, models)
    temperature_value = _validate_temperature(temperature)
    priors = _model_priors(model_priors, models, nominal)
    stability = _stability_loss_matrix(
        source_stability_losses, folds=folds, models=models
    )
    stability_weight_value = float(stability_weight)
    complexity_weight_value = float(complexity_weight)
    safety_margin = float(nominal_safety_margin)
    if (
        not np.isfinite(stability_weight_value)
        or stability_weight_value < 0.0
        or not np.isfinite(complexity_weight_value)
        or complexity_weight_value < 0.0
        or not np.isfinite(safety_margin)
        or safety_margin < 0.0
    ):
        raise ValueError("selection weights and safety margin must be finite/nonnegative")
    if complexity is None:
        complexity_array = np.zeros(models, dtype=np.float64)
    else:
        complexity_array = np.asarray(complexity, dtype=np.float64).reshape(-1)
        if complexity_array.shape != (models,):
            raise ValueError("complexity must have one entry per model")

    predictive_by_fold = np.einsum(
        "fmw,w->fm", predictive, shared_weights, optimize=True
    )
    selection_by_fold = (
        predictive_by_fold
        + stability_weight_value * stability
        + complexity_weight_value * complexity_array[None, :]
    )
    finite = bool(
        np.all(np.isfinite(predictive_by_fold))
        and np.all(np.isfinite(selection_by_fold))
    )
    if finite:
        mean_predictive = np.mean(predictive_by_fold, axis=0)
        mean_selection = np.mean(selection_by_fold, axis=0)
        fold_order = np.argsort(selection_by_fold, axis=1, kind="stable")
    else:
        mean_predictive = np.full(models, np.inf, dtype=np.float64)
        mean_selection = np.full(models, np.inf, dtype=np.float64)
        fold_order = np.tile(np.arange(models), (folds, 1))

    def nominal_fallback(reason: str) -> BEMModelWeightSelection:
        model_weight = np.zeros(models, dtype=np.float64)
        model_weight[nominal] = 1.0
        episode_weight = np.repeat(model_weight[None, :], folds, axis=0)
        return BEMModelWeightSelection(
            model_weights=_readonly_float(model_weight),
            episode_model_weights=_readonly_float(episode_weight),
            model_priors=_readonly_float(priors),
            mean_selection_loss=_readonly_float(mean_selection),
            mean_predictive_loss=_readonly_float(mean_predictive),
            fold_selection_loss=_readonly_float(selection_by_fold),
            fold_model_order=np.asarray(fold_order, dtype=np.int64),
            temperature=temperature_value,
            nominal_index=nominal,
            fallback=True,
            fallback_reason=reason,
            metadata=MappingProxyType(
                {
                    "model_latent_scope": "one_shared_latent_per_head",
                    "shared_across_calibration_episodes": True,
                    "shared_freesigm_weights": shared_weights.tolist(),
                    "operator_specific_freesigm_weights_allowed": False,
                    "selection_uses_source_truth_or_emd": False,
                    "selection_inputs": "heldout_episode_electrode_predictive_and_source_stability",
                    "safe_freesigm_preserved_on_fallback": True,
                    "exact_nominal_fallback": True,
                    "frequency_count": frequencies,
                }
            ),
        )

    if ood_detected:
        return nominal_fallback("ood_detected")
    if not finite:
        return nominal_fallback("nonfinite_calibration_loss")
    # The complete ranking, not merely its winner, must agree across folds.
    if not np.all(fold_order == fold_order[0][None, :]):
        return nominal_fallback("inconsistent_fold_model_ranking")
    winner = int(np.argmin(mean_selection))
    if winner == nominal:
        return nominal_fallback("nominal_safety_boundary_not_exceeded")
    selection_gain = float(mean_selection[nominal] - mean_selection[winner])
    predictive_gain = float(mean_predictive[nominal] - mean_predictive[winner])
    if selection_gain <= safety_margin or predictive_gain <= safety_margin:
        return nominal_fallback("nominal_safety_boundary_not_exceeded")

    model_weight = posterior_model_weights(
        mean_selection, priors, temperature=temperature_value
    )
    # One head posterior is copied for auditing only; it is not re-estimated
    # episode by episode.
    episode_weight = np.repeat(model_weight[None, :], folds, axis=0)
    return BEMModelWeightSelection(
        model_weights=model_weight,
        episode_model_weights=_readonly_float(episode_weight),
        model_priors=_readonly_float(priors),
        mean_selection_loss=_readonly_float(mean_selection),
        mean_predictive_loss=_readonly_float(mean_predictive),
        fold_selection_loss=_readonly_float(selection_by_fold),
        fold_model_order=np.asarray(fold_order, dtype=np.int64),
        temperature=temperature_value,
        nominal_index=nominal,
        fallback=False,
        fallback_reason=None,
        metadata=MappingProxyType(
            {
                "model_latent_scope": "one_shared_latent_per_head",
                "shared_across_calibration_episodes": True,
                "shared_freesigm_weights": shared_weights.tolist(),
                "operator_specific_freesigm_weights_allowed": False,
                "selection_uses_source_truth_or_emd": False,
                "selection_inputs": "heldout_episode_electrode_predictive_and_source_stability",
                "safe_freesigm_preserved_on_fallback": True,
                "exact_nominal_fallback": False,
                "frequency_count": frequencies,
            }
        ),
    )


def _one_operator_family(value: ArrayLike) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4:
        # frequency, sensor, group, orientation
        array = array.reshape(array.shape[0], array.shape[1], -1)
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError("each BEM model must have shape (F,C,P) or (F,C,G,O)")
    return np.asarray(array, dtype=np.complex128)


def _operator_models(value: Sequence[ArrayLike] | ArrayLike) -> list[ComplexArray]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim == 3:
            # model, sensor, coefficient (single frequency)
            models = [array[index][None, ...] for index in range(array.shape[0])]
        elif array.ndim == 4:
            # model, frequency, sensor, coefficient
            models = [array[index] for index in range(array.shape[0])]
        elif array.ndim == 5:
            models = [
                array[index].reshape(array.shape[1], array.shape[2], -1)
                for index in range(array.shape[0])
            ]
        else:
            raise ValueError("operator_models must have 3, 4, or 5 dimensions")
    else:
        models = [_one_operator_family(item) for item in value]
    count = _validate_model_count(len(models))
    reference = models[0].shape
    if any(model.shape != reference or not np.all(np.isfinite(model)) for model in models):
        raise ValueError("all BEM models must share frequency/sensor/source dimensions")
    if count != len(models):
        raise AssertionError("validated BEM model count changed")
    return [np.asarray(model, dtype=np.complex128) for model in models]


def _mixture_weights(
    value: BEMModelWeightSelection | ArrayLike,
    *,
    models: int,
    nominal_index: int,
) -> tuple[FloatArray, bool, str | None]:
    if isinstance(value, BEMModelWeightSelection):
        if value.nominal_index != nominal_index:
            raise ValueError("selection nominal_index does not match inference")
        weights = np.asarray(value.model_weights, dtype=np.float64)
        fallback = bool(value.fallback)
        reason = value.fallback_reason
    else:
        weights = np.asarray(value, dtype=np.float64).reshape(-1)
        fallback = False
        reason = None
    if weights.shape != (models,) or not np.all(np.isfinite(weights)):
        raise ValueError("model_weights must be finite and model aligned")
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("model_weights must be nonnegative with positive sum")
    return np.asarray(weights / weights.sum(), dtype=np.float64), fallback, reason


@dataclass(frozen=True, slots=True)
class BEMMixtureResult:
    method: str
    source_estimate: ComplexArray
    source_covariance: ComplexArray
    source_score: FloatArray
    model_weights: FloatArray
    component_results: tuple[InverseResult, ...]
    frequency_weights: FloatArray
    noise: float
    runtime_seconds: float
    fallback: bool
    fallback_reason: str | None
    metadata: Mapping[str, object]

    @property
    def mean(self) -> ComplexArray:
        return self.source_estimate

    @property
    def covariance(self) -> ComplexArray:
        return self.source_covariance

    @property
    def gamma(self) -> FloatArray:
        return self.source_score


def bem_mixture_champagne(
    operator_models: Sequence[ArrayLike] | ArrayLike,
    observations: ArrayLike,
    *,
    model_weights: BEMModelWeightSelection | ArrayLike,
    frequency_weights: ArrayLike | None = None,
    nominal_index: int = 0,
    group_size: int = 3,
    max_iter: int = 80,
    tolerance: float = 1.0e-6,
    prune_threshold: float = 0.0,
    noise0: float | None = None,
) -> BEMMixtureResult:
    """Fit component Champagne posteriors and form a posterior mixture.

    Operators are never linearly averaged.  ``frequency_weights`` is one
    head-level vector used verbatim for every component.  Identical operators
    and an exact nominal point mass take a literal single-model path.
    """

    start = perf_counter()
    models = _operator_models(operator_models)
    count = len(models)
    nominal = _validate_nominal_index(nominal_index, count)
    shared_weights = _shared_frequency_weights(frequency_weights, models[0].shape[0])
    weights, selection_fallback, selection_reason = _mixture_weights(
        model_weights, models=count, nominal_index=nominal
    )
    exact_nominal = bool(
        weights[nominal] == 1.0 and np.count_nonzero(weights) == 1
    )
    identical = all(np.array_equal(models[nominal], model) for model in models)

    def fit_component(index: int) -> tuple[InverseResult, ComplexArray]:
        stacked_operator, stacked_data, _ = stack_freesigm_family(
            models[index],
            observations,
            frequency_weights=shared_weights,
            group_size=group_size,
        )
        result = champagne_sbl(
            stacked_operator,
            stacked_data,
            group_size=group_size,
            max_iter=max_iter,
            tolerance=tolerance,
            prune_threshold=prune_threshold,
            noise0=noise0,
        )
        covariance = complex_gaussian_posterior(
            stacked_operator,
            stacked_data,
            np.asarray(result.source_score, dtype=np.float64),
            float(result.metadata["noise"]),
            group_size=group_size,
        )[1]
        return result, np.asarray(covariance, dtype=np.complex128)

    if exact_nominal or identical:
        nominal_result, nominal_covariance = fit_component(nominal)
        components = tuple(nominal_result for _ in range(count))
        shortcut = "nominal_weight_one" if exact_nominal else "identical_operators"
        return BEMMixtureResult(
            method="V3-BEM-Mixture-Champagne-FreeSIGM",
            source_estimate=_readonly_complex(nominal_result.source_estimate),
            source_covariance=_readonly_complex(nominal_covariance),
            source_score=_readonly_float(nominal_result.source_score),
            model_weights=_readonly_float(weights),
            component_results=components,
            frequency_weights=_readonly_float(shared_weights),
            noise=float(nominal_result.metadata["noise"]),
            runtime_seconds=perf_counter() - start,
            fallback=selection_fallback,
            fallback_reason=selection_reason,
            metadata=MappingProxyType(
                {
                    "posterior_combination": "literal single-model equivalence path",
                    "equivalence_shortcut": shortcut,
                    "nominal_operator_included": True,
                    "shared_freesigm_weights": shared_weights.tolist(),
                    "component_freesigm_weights": [
                        shared_weights.tolist() for _ in range(count)
                    ],
                    "operator_specific_freesigm_weights_allowed": False,
                    "truth_used_for_fit_or_selection": False,
                }
            ),
        )

    fitted: list[InverseResult] = []
    covariances: list[ComplexArray] = []
    for index, weight in enumerate(weights):
        if weight == 0.0:
            # Zero-weight components cannot affect the posterior; reuse the
            # nominal audit object without spending inference time.
            if fitted:
                fitted.append(fitted[nominal] if nominal < len(fitted) else fitted[0])
                covariances.append(covariances[nominal] if nominal < len(covariances) else covariances[0])
            else:
                result, covariance = fit_component(nominal)
                fitted.append(result)
                covariances.append(covariance)
            continue
        result, covariance = fit_component(index)
        fitted.append(result)
        covariances.append(covariance)
    component_mean = np.stack(
        [np.asarray(result.source_estimate, dtype=np.complex128) for result in fitted]
    )
    mixture_mean = np.einsum("k,kpt->pt", weights, component_mean, optimize=True)
    mixture_covariance = np.zeros_like(covariances[0])
    snapshots = mixture_mean.shape[1]
    for weight, mean, covariance in zip(weights, component_mean, covariances):
        difference = mean - mixture_mean
        mixture_covariance += weight * (
            covariance + difference @ difference.conj().T / snapshots
        )
    mixture_gamma = np.einsum(
        "k,kg->g",
        weights,
        np.stack([result.source_score for result in fitted]),
        optimize=True,
    )
    mixture_noise = float(
        np.dot(weights, [float(result.metadata["noise"]) for result in fitted])
    )
    return BEMMixtureResult(
        method="V3-BEM-Mixture-Champagne-FreeSIGM",
        source_estimate=_readonly_complex(mixture_mean),
        source_covariance=_readonly_complex(mixture_covariance),
        source_score=_readonly_float(mixture_gamma),
        model_weights=_readonly_float(weights),
        component_results=tuple(fitted),
        frequency_weights=_readonly_float(shared_weights),
        noise=mixture_noise,
        runtime_seconds=perf_counter() - start,
        fallback=selection_fallback,
        fallback_reason=selection_reason,
        metadata=MappingProxyType(
            {
                "posterior_combination": "proper posterior moment mixture",
                "equivalence_shortcut": None,
                "nominal_operator_included": True,
                "shared_freesigm_weights": shared_weights.tolist(),
                "component_freesigm_weights": [
                    shared_weights.tolist() for _ in range(count)
                ],
                "operator_specific_freesigm_weights_allowed": False,
                "truth_used_for_fit_or_selection": False,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class GeometryHybridResult:
    operator: ComplexArray
    predicted_coefficients: FloatArray
    ood_distance: float
    fallback: bool
    fallback_reason: str | None
    metadata: Mapping[str, object]


def _descriptor_values(descriptor: object) -> FloatArray:
    value = getattr(descriptor, "values", descriptor)
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size < 1 or not np.all(np.isfinite(out)):
        raise ValueError("geometry descriptor must be a finite non-empty vector")
    return out


def _geometry_basis_correction(
    residual_basis: ArrayLike,
    coefficients: FloatArray,
    nominal_shape: tuple[int, int, int],
    rank: int,
) -> ComplexArray:
    frequencies, sensors, source_columns = nominal_shape
    basis = np.asarray(residual_basis, dtype=np.complex128)
    if basis.ndim == 4:
        # rank, frequency, sensor, source
        if basis.shape[1:] != nominal_shape or basis.shape[0] < rank:
            raise ValueError("rank-first residual_basis does not align with nominal operator")
        if coefficients.size not in (rank, frequencies * rank):
            raise ValueError("predicted coefficient count does not match residual rank")
        if coefficients.size == rank:
            return np.einsum("r,rfcp->fcp", coefficients[:rank], basis[:rank])
        coefficient_matrix = coefficients.reshape(frequencies, rank)
        return np.einsum("fr,rfcp->fcp", coefficient_matrix, basis[:rank])
    if basis.ndim == 3:
        # BEM-library convention: frequency, rank, flattened operator.
        if basis.shape[0] != frequencies or basis.shape[2] != sensors * source_columns:
            raise ValueError("frequency-first residual_basis does not align with nominal operator")
        if basis.shape[1] < rank or coefficients.size not in (rank, frequencies * rank):
            raise ValueError("predicted coefficient count does not match residual rank")
        modes = basis[:, :rank].reshape(frequencies, rank, sensors, source_columns)
        coefficient_matrix = (
            np.repeat(coefficients[:rank][None, :], frequencies, axis=0)
            if coefficients.size == rank
            else coefficients.reshape(frequencies, rank)
        )
        return np.einsum("fr,frcp->fcp", coefficient_matrix, modes)
    raise ValueError("residual_basis must be rank-first 4D or frequency-first 3D")


def geometry_conditioned_hybrid(
    nominal_operator: ArrayLike,
    geometry_descriptor: object,
    predictor: object,
    residual_basis: ArrayLike,
    *,
    rank: int,
    ood_threshold: float,
    coefficient_shrinkage: float = 1.0,
    library_gate_passed: bool = True,
) -> GeometryHybridResult:
    """Predict a geometry-only operator correction with an OOD fail-safe.

    The signature contains no EEG, source estimate, source truth, trial
    residual, or localization outcome.  OOD is computed in the predictor's
    frozen standardized descriptor space before ``predict`` is called.
    """

    original = np.asarray(nominal_operator, dtype=np.complex128)
    restore_shape = original.shape
    if original.ndim == 2:
        nominal = original[None, ...]
    elif original.ndim == 3:
        nominal = original
    elif original.ndim == 4:
        nominal = original.reshape(original.shape[0], original.shape[1], -1)
    else:
        raise ValueError("nominal_operator must have 2, 3, or 4 dimensions")
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_operator must be finite")
    rank_value = int(rank)
    if rank_value not in GEOMETRY_HYBRID_RANK_GRID:
        raise ValueError(f"rank must be one of {GEOMETRY_HYBRID_RANK_GRID}")
    threshold = float(ood_threshold)
    shrinkage = float(coefficient_shrinkage)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("ood_threshold must be finite and positive")
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("coefficient_shrinkage must lie in [0, 1]")
    descriptor = _descriptor_values(geometry_descriptor)
    feature_mean = np.asarray(getattr(predictor, "feature_mean"), dtype=np.float64).reshape(-1)
    feature_scale = np.asarray(getattr(predictor, "feature_scale"), dtype=np.float64).reshape(-1)
    if feature_mean.shape != descriptor.shape or feature_scale.shape != descriptor.shape:
        raise ValueError("predictor descriptor normalization does not align")
    if np.any(feature_scale <= 0.0) or not np.all(np.isfinite(feature_scale)):
        raise ValueError("predictor feature_scale must be finite and positive")
    standardized = (descriptor - feature_mean) / feature_scale
    ood_distance = float(np.linalg.norm(standardized) / np.sqrt(descriptor.size))

    def fallback(reason: str) -> GeometryHybridResult:
        return GeometryHybridResult(
            operator=_readonly_complex(original),
            predicted_coefficients=_readonly_float(np.empty(0)),
            ood_distance=ood_distance,
            fallback=True,
            fallback_reason=reason,
            metadata=MappingProxyType(
                {
                    "predictor_inputs": "geometry_descriptor_only",
                    "eeg_source_truth_or_trial_residual_used": False,
                    "safe_freesigm_preserved_on_fallback": True,
                    "exact_nominal_fallback": True,
                    "rank": rank_value,
                    "coefficient_shrinkage": shrinkage,
                }
            ),
        )

    if not library_gate_passed:
        return fallback("heldout_library_gate_failed")
    if ood_distance > threshold:
        return fallback("geometry_descriptor_ood")
    try:
        prediction = np.asarray(
            getattr(predictor, "predict")(geometry_descriptor), dtype=np.float64
        ).reshape(-1)
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return fallback("geometry_predictor_failure")
    if not np.all(np.isfinite(prediction)):
        return fallback("geometry_predictor_nonfinite")
    coefficients = shrinkage * prediction
    try:
        correction = _geometry_basis_correction(
            residual_basis,
            coefficients,
            nominal.shape,
            rank_value,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return fallback("geometry_basis_reconstruction_failure")
    corrected = nominal + correction
    corrected = corrected.reshape(restore_shape)
    return GeometryHybridResult(
        operator=_readonly_complex(corrected),
        predicted_coefficients=_readonly_float(coefficients),
        ood_distance=ood_distance,
        fallback=False,
        fallback_reason=None,
        metadata=MappingProxyType(
            {
                "predictor_inputs": "geometry_descriptor_only",
                "eeg_source_truth_or_trial_residual_used": False,
                "safe_freesigm_preserved_on_fallback": True,
                "exact_nominal_fallback": False,
                "rank": rank_value,
                "coefficient_shrinkage": shrinkage,
            }
        ),
    )


bem_model_average = bem_mixture_champagne
select_bem_weights = select_bem_model_weights
geometry_hybrid_operator = geometry_conditioned_hybrid


__all__ = [
    "BEM_MODEL_COUNT_GRID",
    "BEM_MIXTURE_TEMPERATURE_GRID",
    "GEOMETRY_HYBRID_RANK_GRID",
    "BEMMixtureResult",
    "BEMModelWeightSelection",
    "GeometryHybridResult",
    "bem_mixture_champagne",
    "bem_model_average",
    "geometry_conditioned_hybrid",
    "geometry_hybrid_operator",
    "posterior_model_weights",
    "select_bem_model_weights",
    "select_bem_weights",
]
