"""Leakage-safe, conservative frequency weighting for V3.

This module implements the library-frozen part of Safe-FreeSIGM.  It deliberately
accepts only calibration-derived scores and operator-library statistics.  The
inverse solver remains responsible for refitting every weight-dependent state in
the exact-uniform end-to-end control; :class:`FreeSIGMControlContract` makes that
boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .stability import effective_sample_size, project_to_ess


LOGGER = logging.getLogger(__name__)

ALLOWED_RHO = (0.05, 0.10, 0.25, 0.50)
ALLOWED_ESS_FRACTIONS = (0.80, 0.90, 0.95)
SCORE_NAMES = (
    "evidence",
    "information_trace",
    "spectral_bound",
    "operator_uncertainty",
    "redundancy",
)


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _real_vector(values: object, *, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    if size is not None and result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def project_psd(matrix: np.ndarray, *, relative_floor: float = 0.0) -> np.ndarray:
    """Return the nearest Hermitian PSD matrix by eigenvalue clipping."""

    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError("matrix must be a finite square matrix")
    eigval, eigvec = np.linalg.eigh(_hermitian(value))
    largest = max(float(np.max(eigval, initial=0.0)), 0.0)
    floor = max(float(relative_floor) * largest, 0.0)
    clipped = np.maximum(eigval, floor)
    projected = (eigvec * clipped[None, :]) @ eigvec.conj().T
    projected = _hermitian(projected)
    if np.max(np.abs(projected.imag), initial=0.0) <= 1.0e-13:
        return projected.real.astype(np.float64)
    return projected


@dataclass(frozen=True)
class RobustScale:
    center: float
    scale: float

    def __post_init__(self) -> None:
        if not isfinite(self.center) or not isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("robust center/scale must be finite and scale must be positive")


@dataclass(frozen=True)
class RobustScoreNormalizer:
    """Immutable score scales fitted only on the ``library_train`` tier."""

    statistics: Mapping[str, RobustScale]
    source_split: Literal["library_train"] = "library_train"
    fit_observation_count: int = 0

    def __post_init__(self) -> None:
        if self.source_split != "library_train":
            raise ValueError("Safe-FreeSIGM score normalization must use library_train only")
        missing = set(SCORE_NAMES) - set(self.statistics)
        extra = set(self.statistics) - set(SCORE_NAMES)
        if missing or extra:
            raise ValueError(f"normalizer score fields mismatch: missing={missing}, extra={extra}")
        if int(self.fit_observation_count) < 1:
            raise ValueError("fit_observation_count must be positive")
        frozen = MappingProxyType(dict(self.statistics))
        object.__setattr__(self, "statistics", frozen)

    def transform(self, name: str, values: object, *, center: bool = True) -> np.ndarray:
        if name not in self.statistics:
            raise KeyError(f"unknown score family: {name}")
        value = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        stats = self.statistics[name]
        return (value - stats.center) / stats.scale if center else value / stats.scale


def fit_library_score_normalizer(
    library_scores: Mapping[str, object],
    *,
    source_split: str,
    scale_floor: float = 1.0e-12,
) -> RobustScoreNormalizer:
    """Fit scalar robust scales across library heads and frequencies.

    A scalar scale per score family makes the transformation equivariant to a
    permutation of frequencies.  Location is the median; scale is MAD, with IQR
    and standard-deviation fallbacks for a constant or nearly constant library.
    """

    if source_split != "library_train":
        raise ValueError("score normalization may only be fitted on library_train")
    if set(library_scores) != set(SCORE_NAMES):
        raise ValueError(f"library_scores must contain exactly {SCORE_NAMES}")
    if not isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")

    statistics: dict[str, RobustScale] = {}
    counts: list[int] = []
    for name in SCORE_NAMES:
        flat = np.asarray(library_scores[name], dtype=np.float64).reshape(-1)
        if not flat.size or not np.all(np.isfinite(flat)):
            raise ValueError(f"library score {name} must be finite and nonempty")
        center = float(np.median(flat))
        scale = 1.4826 * float(np.median(np.abs(flat - center)))
        if scale <= scale_floor:
            q25, q75 = np.quantile(flat, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
        if scale <= scale_floor:
            scale = float(np.std(flat))
        if scale <= scale_floor:
            scale = 1.0
        statistics[name] = RobustScale(center=center, scale=max(scale, scale_floor))
        counts.append(int(flat.size))
    return RobustScoreNormalizer(
        statistics=statistics,
        source_split="library_train",
        fit_observation_count=min(counts),
    )


@dataclass(frozen=True)
class SafeFreeSIGMConfig:
    """Frozen global Safe-FreeSIGM hyperparameters."""

    mode: Literal["full", "uniform_ablation"] = "full"
    rho: float = 0.10
    alpha_ess: float = 0.90
    minimum_weight_mass: float = 1.0e-4
    lambda_information: float = 0.10
    lambda_spectral: float = 0.10
    lambda_uncertainty: float = 0.10
    lambda_redundancy: float = 0.05
    lambda_kl: float = 0.25
    information_epsilon: float = 1.0e-8
    optimizer_tolerance: float = 1.0e-11
    feasibility_tolerance: float = 2.0e-8
    kkt_tolerance: float = 2.0e-4
    determinism_tolerance: float = 1.0e-10
    max_iterations: int = 1000
    verify_determinism: bool = True
    fallback_policy: Literal["library_global_then_uniform"] = (
        "library_global_then_uniform"
    )
    formula_version: Literal["safe_freesigm_v3_formula_1"] = (
        "safe_freesigm_v3_formula_1"
    )

    def __post_init__(self) -> None:
        if self.mode == "full":
            if self.rho not in ALLOWED_RHO or self.rho <= 0.0:
                raise ValueError(f"full Safe-FreeSIGM rho must be one of {ALLOWED_RHO}")
        elif self.mode == "uniform_ablation":
            if self.rho != 0.0:
                raise ValueError("exact uniform ablation requires rho=0")
        else:
            raise ValueError("unknown Safe-FreeSIGM mode")
        if self.alpha_ess not in ALLOWED_ESS_FRACTIONS:
            raise ValueError(f"alpha_ess must be one of {ALLOWED_ESS_FRACTIONS}")
        nonnegative = (
            self.lambda_information,
            self.lambda_spectral,
            self.lambda_uncertainty,
            self.lambda_redundancy,
        )
        if any(not isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("information, spectral, uncertainty, and redundancy weights must be nonnegative")
        if not isfinite(self.lambda_kl) or self.lambda_kl <= 0.0:
            raise ValueError("lambda_kl must be finite and strictly positive")
        positive = (
            self.minimum_weight_mass,
            self.information_epsilon,
            self.optimizer_tolerance,
            self.feasibility_tolerance,
            self.kkt_tolerance,
            self.determinism_tolerance,
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Safe-FreeSIGM numerical constants must be finite and positive")
        if self.minimum_weight_mass >= 1.0:
            raise ValueError("minimum_weight_mass must be below one")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive")

    def minimum_weight(self, n_frequencies: int) -> float:
        if n_frequencies < 1:
            raise ValueError("n_frequencies must be positive")
        return self.minimum_weight_mass / float(n_frequencies)


@dataclass(frozen=True)
class SafeFreeSIGMInputs:
    """Per-head calibration/operator-library inputs to the convex solver."""

    evidence: np.ndarray
    information_matrices: np.ndarray
    spectral_bound: np.ndarray
    operator_uncertainty: np.ndarray
    redundancy_matrix: np.ndarray

    @property
    def n_frequencies(self) -> int:
        value = np.asarray(self.evidence)
        if value.ndim != 1 or not value.size:
            raise ValueError("evidence must identify at least one frequency")
        return int(value.size)


@dataclass(frozen=True)
class ConstraintDiagnostics:
    sum_residual: float
    minimum_weight_margin: float
    ess: float
    ess_margin: float
    feasible: bool
    ess_constraint_enforced: bool
    ess_floor: float


@dataclass(frozen=True)
class KKTDiagnostics:
    stationarity_residual: float
    complementarity_residual: float
    primal_residual: float
    equality_multiplier: float
    ess_multiplier: float
    active_lower_bound_count: int


@dataclass(frozen=True)
class ObjectiveTerms:
    evidence: float
    information: float
    spectral: float
    uncertainty: float
    redundancy: float
    kl_to_uniform: float

    @property
    def total(self) -> float:
        return float(
            self.evidence
            + self.information
            + self.spectral
            + self.uncertainty
            + self.redundancy
            + self.kl_to_uniform
        )


@dataclass(frozen=True)
class SafeFreeSIGMResult:
    raw_weights: np.ndarray
    safe_weights: np.ndarray
    effective_likelihood_weights: np.ndarray
    raw_ess: float
    safe_ess: float
    objective_terms: ObjectiveTerms
    constraints: ConstraintDiagnostics
    kkt: KKTDiagnostics
    normalized_scores: Mapping[str, np.ndarray]
    rho: float
    fallback: bool
    fallback_reason: str | None
    numerical_status: str
    optimizer_iterations: int
    deterministic: bool
    runtime_seconds: float

    def __post_init__(self) -> None:
        for name in ("raw_weights", "safe_weights", "effective_likelihood_weights"):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=np.float64))
        frozen_scores = {
            key: _readonly(value) for key, value in dict(self.normalized_scores).items()
        }
        object.__setattr__(self, "normalized_scores", MappingProxyType(frozen_scores))


@dataclass(frozen=True)
class SharedHeadSafeFreeSIGMResult:
    shared: SafeFreeSIGMResult
    weights_by_bem_model: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weights_by_bem_model",
            _readonly(self.weights_by_bem_model, dtype=np.float64),
        )


@dataclass(frozen=True)
class RuntimeOverheadAccounting:
    uniform_pipeline_seconds: float
    safe_pipeline_seconds: float
    freesigm_solver_seconds: float
    overhead_fraction: float
    solver_fraction: float
    target_fraction: float
    within_target: bool


@dataclass(frozen=True)
class FreeSIGMControlContract:
    """Strongly typed hand-off to an end-to-end inverse implementation."""

    name: Literal["full", "exact_uniform", "controlled_direct"]
    rho: float
    adaptive_weighting_bypassed: bool
    refit_all_weight_dependent_state: bool
    reuse_full_downstream_state: bool

    def __post_init__(self) -> None:
        if self.name == "full":
            valid = self.rho in ALLOWED_RHO and not self.adaptive_weighting_bypassed
            valid = valid and self.refit_all_weight_dependent_state
            valid = valid and not self.reuse_full_downstream_state
        elif self.name == "exact_uniform":
            valid = self.rho == 0.0 and self.adaptive_weighting_bypassed
            valid = valid and self.refit_all_weight_dependent_state
            valid = valid and not self.reuse_full_downstream_state
        elif self.name == "controlled_direct":
            valid = self.rho == 0.0 and self.adaptive_weighting_bypassed
            valid = valid and not self.refit_all_weight_dependent_state
            valid = valid and self.reuse_full_downstream_state
        else:
            valid = False
        if not valid:
            raise ValueError("inconsistent matched-control contract")


def matched_control_contract(
    name: Literal["full", "exact_uniform", "controlled_direct"],
    *,
    full_rho: float = 0.10,
) -> FreeSIGMControlContract:
    if name == "full":
        return FreeSIGMControlContract(name, full_rho, False, True, False)
    if name == "exact_uniform":
        return FreeSIGMControlContract(name, 0.0, True, True, False)
    if name == "controlled_direct":
        return FreeSIGMControlContract(name, 0.0, True, False, True)
    raise ValueError("unknown matched-control name")


def proper_complex_negative_evidence(
    sensor_covariances: object,
    sample_covariances: object,
    *,
    sample_counts: object | None = None,
    return_per_sample: bool = True,
) -> np.ndarray:
    r"""Compute ``logdet(Sigma_f) + tr(Sigma_f^-1 C_yf)``.

    ``sample_covariances`` are per-sample cross-spectra.  The default therefore
    returns a per-sample score so unequal trial counts cannot become accidental
    reliability weights.  Set ``return_per_sample=False`` to recover the proper
    complex Gaussian total-data factor ``n_f``.
    """

    covariance = np.asarray(sensor_covariances, dtype=np.complex128)
    sample = np.asarray(sample_covariances, dtype=np.complex128)
    if covariance.ndim == 2:
        covariance = covariance[None, ...]
    if sample.ndim == 2:
        sample = sample[None, ...]
    if covariance.ndim != 3 or covariance.shape != sample.shape:
        raise ValueError("sensor and sample covariances must have shape (F,E,E)")
    if covariance.shape[1] != covariance.shape[2] or not np.all(np.isfinite(covariance)):
        raise ValueError("sensor covariances must be finite square matrices")
    if not np.all(np.isfinite(sample)):
        raise ValueError("sample covariances must be finite")
    counts = np.ones(covariance.shape[0], dtype=np.float64)
    if sample_counts is not None:
        counts = _real_vector(sample_counts, name="sample_counts", size=covariance.shape[0])
        if np.any(counts <= 0.0):
            raise ValueError("sample_counts must be positive")

    result = np.empty(covariance.shape[0], dtype=np.float64)
    for index, (sigma, cy) in enumerate(zip(covariance, sample)):
        sigma = _hermitian(sigma)
        cy = _hermitian(cy)
        chol = np.linalg.cholesky(sigma)
        logdet = 2.0 * np.log(np.diag(chol).real).sum()
        quadratic = np.trace(np.linalg.solve(sigma, cy))
        if abs(float(quadratic.imag)) > 1.0e-8 * max(1.0, abs(float(quadratic.real))):
            raise ValueError("proper-complex evidence has a material imaginary trace")
        score = float(logdet + quadratic.real)
        result[index] = score if return_per_sample else counts[index] * score
    return result


def nuisance_projected_information(
    interest_jacobians: object,
    *,
    nuisance_jacobians: object | None = None,
    noise_precision: object | None = None,
    rank_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Construct PSD information after projecting out nuisance directions."""

    jacobian = np.asarray(interest_jacobians, dtype=np.complex128)
    if jacobian.ndim == 2:
        jacobian = jacobian[None, ...]
    if jacobian.ndim != 3 or not all(jacobian.shape) or not np.all(np.isfinite(jacobian)):
        raise ValueError("interest_jacobians must have shape (F,E,P)")
    n_frequency, n_sensor, _ = jacobian.shape

    if nuisance_jacobians is None:
        nuisance = np.zeros((n_frequency, n_sensor, 0), dtype=np.complex128)
    else:
        nuisance = np.asarray(nuisance_jacobians, dtype=np.complex128)
        if nuisance.ndim == 2:
            nuisance = nuisance[None, ...]
        if nuisance.shape[:2] != (n_frequency, n_sensor) or not np.all(np.isfinite(nuisance)):
            raise ValueError("nuisance_jacobians must align with interest_jacobians")

    if noise_precision is None:
        precision = np.repeat(np.eye(n_sensor, dtype=np.complex128)[None], n_frequency, axis=0)
    else:
        precision = np.asarray(noise_precision, dtype=np.complex128)
        if precision.ndim == 2:
            precision = precision[None, ...]
        if precision.shape != (n_frequency, n_sensor, n_sensor):
            raise ValueError("noise_precision must have shape (F,E,E)")

    output = np.empty((n_frequency, jacobian.shape[2], jacobian.shape[2]), dtype=np.complex128)
    for f in range(n_frequency):
        eigval, eigvec = np.linalg.eigh(_hermitian(precision[f]))
        scale = max(float(np.max(eigval, initial=0.0)), 1.0)
        if float(np.min(eigval, initial=0.0)) < -1.0e-10 * scale:
            raise ValueError("noise precision must be PSD")
        sqrt_precision = (eigvec * np.sqrt(np.maximum(eigval, 0.0))[None, :]) @ eigvec.conj().T
        x = sqrt_precision @ jacobian[f]
        n = sqrt_precision @ nuisance[f]
        if n.shape[1]:
            u, singular, _ = np.linalg.svd(n, full_matrices=False)
            threshold = rank_tolerance * max(float(singular[0]), 1.0) if singular.size else np.inf
            rank = int(np.sum(singular > threshold))
            if rank:
                x = x - u[:, :rank] @ (u[:, :rank].conj().T @ x)
        output[f] = np.asarray(project_psd(x.conj().T @ x), dtype=np.complex128)
    if np.max(np.abs(output.imag), initial=0.0) <= 1.0e-13:
        return output.real.astype(np.float64)
    return output


def information_logdet_and_gradient(
    weights: object,
    information_matrices: object,
    *,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    """Return logdet information and its analytic frequency-weight gradient."""

    w = _real_vector(weights, name="weights")
    matrices = np.asarray(information_matrices, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[0] != w.size or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("information_matrices must have shape (F,P,P)")
    if not np.all(np.isfinite(matrices)) or epsilon <= 0.0:
        raise ValueError("information matrices and epsilon must be finite/positive")
    dimension = matrices.shape[1]
    aggregate = epsilon * np.eye(dimension, dtype=np.complex128)
    for weight, matrix in zip(w, matrices):
        aggregate += weight * _hermitian(matrix)
    aggregate = _hermitian(aggregate)
    chol = np.linalg.cholesky(aggregate)
    value = float(2.0 * np.log(np.diag(chol).real).sum())
    inverse = np.linalg.solve(aggregate, np.eye(dimension, dtype=np.complex128))
    gradient = np.asarray(
        [np.trace(inverse @ _hermitian(matrix)).real for matrix in matrices],
        dtype=np.float64,
    )
    return value, gradient


def spectral_stability_bound(
    unregularized_k: Iterable[np.ndarray],
    whitened_leadfields: Iterable[np.ndarray],
    *,
    noise_scale: object,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Score raw spectral amplification without masking it by regularization."""

    raw_k = [np.asarray(value, dtype=np.complex128) for value in unregularized_k]
    operators = [np.asarray(value, dtype=np.complex128) for value in whitened_leadfields]
    if not raw_k or len(raw_k) != len(operators):
        raise ValueError("unregularized_k and whitened_leadfields must align")
    noise = _real_vector(noise_scale, name="noise_scale", size=len(raw_k))
    if np.any(noise <= 0.0) or epsilon <= 0.0:
        raise ValueError("noise scale and epsilon must be positive")
    result = np.empty(len(raw_k), dtype=np.float64)
    for f, (matrix, operator) in enumerate(zip(raw_k, operators)):
        if matrix.ndim != 2 or operator.ndim != 2 or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(operator)):
            raise ValueError("spectral inputs must be finite matrices")
        singular_k = np.linalg.svd(matrix, compute_uv=False)
        singular_g = np.linalg.svd(operator, compute_uv=False)
        if not singular_k.size or not singular_g.size:
            raise ValueError("spectral inputs must be nonempty")
        largest = float(singular_k[0])
        smallest = float(singular_k[-1])
        condition = largest / max(smallest, epsilon * max(largest, 1.0))
        amplification = largest * float(singular_g[0]) ** 2 / noise[f]
        result[f] = np.log1p(amplification) + np.log1p(condition)
    return result


def operator_uncertainty_penalty(
    library_residual_covariances: object,
    *,
    source_split: str,
    trace_weight: float = 1.0,
    effective_rank_weight: float = 0.0,
) -> np.ndarray:
    """Compute frequency-specific mismatch penalties from library residuals."""

    if source_split != "library_train":
        raise ValueError("operator uncertainty must be estimated from library_train")
    covariance = np.asarray(library_residual_covariances, dtype=np.complex128)
    if covariance.ndim == 4:
        covariance = np.mean(covariance, axis=0)
    if covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]:
        raise ValueError("library residual covariances must have shape (H,F,E,E) or (F,E,E)")
    if not np.all(np.isfinite(covariance)) or trace_weight < 0.0 or effective_rank_weight < 0.0:
        raise ValueError("uncertainty inputs and weights must be finite/nonnegative")
    result = np.empty(covariance.shape[0], dtype=np.float64)
    for f, matrix in enumerate(covariance):
        eigval = np.linalg.eigvalsh(_hermitian(matrix))
        scale = max(float(np.max(np.abs(eigval), initial=0.0)), 1.0)
        if float(np.min(eigval, initial=0.0)) < -1.0e-10 * scale:
            raise ValueError("library residual covariance must be PSD")
        eigval = np.maximum(eigval, 0.0)
        total = float(eigval.sum())
        normalized_trace = total / matrix.shape[0]
        effective_rank = 0.0 if total <= 0.0 else total**2 / max(float(eigval @ eigval), np.finfo(float).tiny)
        result[f] = trace_weight * normalized_trace + effective_rank_weight * effective_rank / matrix.shape[0]
    return result


def redundancy_matrix_from_subspaces(
    whitened_leadfields: Iterable[np.ndarray],
    *,
    signal_rank: int = 3,
) -> np.ndarray:
    """Build a PSD Gram matrix of whitened leadfield projection operators."""

    operators = [np.asarray(value, dtype=np.complex128) for value in whitened_leadfields]
    if not operators or signal_rank < 1:
        raise ValueError("at least one operator and a positive signal_rank are required")
    features: list[np.ndarray] = []
    for operator in operators:
        if operator.ndim != 2 or not np.all(np.isfinite(operator)):
            raise ValueError("whitened leadfields must be finite matrices")
        u, _, _ = np.linalg.svd(operator, full_matrices=False)
        rank = min(int(signal_rank), u.shape[1])
        projection = u[:, :rank] @ u[:, :rank].conj().T
        feature = np.concatenate([projection.real.reshape(-1), projection.imag.reshape(-1)])
        norm = np.linalg.norm(feature)
        if norm <= 0.0:
            raise ValueError("leadfield subspace has zero norm")
        features.append(feature / norm)
    feature_matrix = np.vstack(features)
    gram = feature_matrix @ feature_matrix.T
    gram = np.asarray(project_psd(gram), dtype=np.float64)
    gram[np.abs(gram) < 1.0e-14] = 0.0
    return gram


def effective_evidence_weights(safe_weights: object) -> np.ndarray:
    """Return ``F*w`` so uniform weighting exactly matches an unweighted sum."""

    weights = _real_vector(safe_weights, name="safe_weights")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1.0e-10):
        raise ValueError("safe_weights must be normalized and nonnegative")
    return weights.size * weights


def weighted_evidence_sum(per_frequency_values: object, safe_weights: object) -> np.ndarray:
    values = np.asarray(per_frequency_values)
    weights = effective_evidence_weights(safe_weights)
    if values.shape[0] != weights.size:
        raise ValueError("per-frequency values and weights must align")
    return np.tensordot(weights, values, axes=(0, 0))


def _validate_inputs(
    inputs: SafeFreeSIGMInputs,
    normalizer: RobustScoreNormalizer,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_frequency = inputs.n_frequencies
    evidence = _real_vector(inputs.evidence, name="evidence", size=n_frequency)
    spectral = _real_vector(inputs.spectral_bound, name="spectral_bound", size=n_frequency)
    uncertainty = _real_vector(
        inputs.operator_uncertainty, name="operator_uncertainty", size=n_frequency
    )
    if np.any(spectral < 0.0) or np.any(uncertainty < 0.0):
        raise ValueError("spectral and uncertainty penalties must be nonnegative")

    information = np.asarray(inputs.information_matrices, dtype=np.complex128)
    if information.ndim != 3 or information.shape[0] != n_frequency or information.shape[1] != information.shape[2]:
        raise ValueError("information_matrices must have shape (F,P,P)")
    projected_information = np.empty_like(information)
    for index, matrix in enumerate(information):
        projected_information[index] = np.asarray(project_psd(matrix), dtype=np.complex128)

    redundancy = np.asarray(inputs.redundancy_matrix, dtype=np.complex128)
    if redundancy.shape != (n_frequency, n_frequency):
        raise ValueError("redundancy_matrix must have shape (F,F)")
    redundancy = np.asarray(project_psd(redundancy), dtype=np.complex128)
    if np.max(np.abs(redundancy.imag), initial=0.0) > 1.0e-10:
        raise ValueError("redundancy matrix must be real after PSD projection")
    redundancy = redundancy.real

    normalized: dict[str, np.ndarray] = {
        "evidence": normalizer.transform("evidence", evidence),
        "spectral_bound": normalizer.transform("spectral_bound", spectral),
        "operator_uncertainty": normalizer.transform("operator_uncertainty", uncertainty),
        "information_matrices": projected_information
        / normalizer.statistics["information_trace"].scale,
        "redundancy_matrix": redundancy / normalizer.statistics["redundancy"].scale,
    }
    return normalized, projected_information


def _constraint_diagnostics(
    weights: np.ndarray,
    *,
    minimum_weight: float,
    alpha_ess: float,
    tolerance: float,
    enforce_ess: bool = True,
) -> ConstraintDiagnostics:
    n_frequency = weights.size
    ess = effective_sample_size(weights)
    ess_floor = alpha_ess * n_frequency if enforce_ess else 1.0
    sum_residual = abs(float(weights.sum()) - 1.0)
    minimum_margin = float(np.min(weights) - minimum_weight)
    ess_margin = ess - ess_floor
    feasible = bool(
        np.all(np.isfinite(weights))
        and sum_residual <= tolerance
        and minimum_margin >= -tolerance
        and (not enforce_ess or ess_margin >= -tolerance)
    )
    return ConstraintDiagnostics(
        sum_residual,
        minimum_margin,
        ess,
        ess_margin,
        feasible,
        bool(enforce_ess),
        float(ess_floor),
    )


def _objective_functions(
    normalized: Mapping[str, np.ndarray],
    config: SafeFreeSIGMConfig,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], ObjectiveTerms]]:
    evidence = normalized["evidence"]
    spectral = normalized["spectral_bound"]
    uncertainty = normalized["operator_uncertainty"]
    information = normalized["information_matrices"]
    redundancy = normalized["redundancy_matrix"]
    uniform = np.full(evidence.size, 1.0 / evidence.size)

    def terms(weights: np.ndarray) -> ObjectiveTerms:
        log_information, _ = information_logdet_and_gradient(
            weights, information, epsilon=config.information_epsilon
        )
        kl = float(np.sum(weights * np.log(weights / uniform)))
        return ObjectiveTerms(
            evidence=float(weights @ evidence),
            information=-config.lambda_information * log_information,
            spectral=config.lambda_spectral * float(weights @ spectral),
            uncertainty=config.lambda_uncertainty * float(weights @ uncertainty),
            redundancy=config.lambda_redundancy * float(weights @ redundancy @ weights),
            kl_to_uniform=config.lambda_kl * kl,
        )

    def objective(weights: np.ndarray) -> float:
        return terms(weights).total

    def gradient(weights: np.ndarray) -> np.ndarray:
        _, information_gradient = information_logdet_and_gradient(
            weights, information, epsilon=config.information_epsilon
        )
        return (
            evidence
            - config.lambda_information * information_gradient
            + config.lambda_spectral * spectral
            + config.lambda_uncertainty * uncertainty
            + 2.0 * config.lambda_redundancy * (redundancy @ weights)
            + config.lambda_kl * (np.log(weights / uniform) + 1.0)
        )

    return objective, gradient, terms


def _solve_once(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    *,
    n_frequency: int,
    minimum_weight: float,
    alpha_ess: float,
    config: SafeFreeSIGMConfig,
    enforce_ess: bool = True,
) -> object:
    uniform = np.full(n_frequency, 1.0 / n_frequency)
    squared_limit = 1.0 / (alpha_ess * n_frequency)
    constraints: tuple[dict[str, object], ...] = (
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
    )
    if enforce_ess:
        constraints += (
            {
                "type": "ineq",
                "fun": lambda weights: float(squared_limit - weights @ weights),
                "jac": lambda weights: -2.0 * weights,
            },
        )
    return minimize(
        objective,
        uniform,
        jac=gradient,
        method="SLSQP",
        bounds=[(minimum_weight, 1.0) for _ in range(n_frequency)],
        constraints=constraints,
        options={
            "ftol": config.optimizer_tolerance,
            "maxiter": int(config.max_iterations),
            "disp": False,
        },
    )


def _kkt_diagnostics(
    weights: np.ndarray,
    gradient: np.ndarray,
    constraints: ConstraintDiagnostics,
    *,
    minimum_weight: float,
    alpha_ess: float,
    active_tolerance: float,
    enforce_ess: bool = True,
) -> KKTDiagnostics:
    n_frequency = weights.size
    squared_margin = 1.0 / (alpha_ess * n_frequency) - float(weights @ weights)
    lower_margin = weights - minimum_weight
    active_ess = bool(enforce_ess and squared_margin <= active_tolerance)
    active_lower = np.flatnonzero(lower_margin <= active_tolerance)

    columns = [np.ones(n_frequency)]
    lower_bounds = [-np.inf]
    upper_bounds = [np.inf]
    if active_ess:
        columns.append(2.0 * weights)
        lower_bounds.append(0.0)
        upper_bounds.append(np.inf)
    for index in active_lower:
        column = np.zeros(n_frequency)
        column[index] = -1.0
        columns.append(column)
        lower_bounds.append(0.0)
        upper_bounds.append(np.inf)
    design = np.column_stack(columns)
    multipliers = lsq_linear(
        design,
        -gradient,
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        lsmr_tol="auto",
    ).x
    residual = gradient + design @ multipliers
    cursor = 1
    ess_multiplier = float(multipliers[cursor]) if active_ess else 0.0
    cursor += int(active_ess)
    lower_multipliers = np.zeros(n_frequency)
    if active_lower.size:
        lower_multipliers[active_lower] = multipliers[cursor:]
    complementarity = max(
        abs(ess_multiplier * squared_margin),
        float(np.max(np.abs(lower_multipliers * lower_margin), initial=0.0)),
    )
    primal = max(
        constraints.sum_residual,
        max(-constraints.minimum_weight_margin, 0.0),
        max(-constraints.ess_margin, 0.0),
    )
    return KKTDiagnostics(
        stationarity_residual=float(np.max(np.abs(residual), initial=0.0)),
        complementarity_residual=float(complementarity),
        primal_residual=float(primal),
        equality_multiplier=float(multipliers[0]),
        ess_multiplier=ess_multiplier,
        active_lower_bound_count=int(active_lower.size),
    )


def _zero_terms() -> ObjectiveTerms:
    return ObjectiveTerms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _fallback_result(
    *,
    n_frequency: int,
    config: SafeFreeSIGMConfig,
    reason: str,
    library_global_safe_weights: object | None,
    started: float,
    normalized: Mapping[str, np.ndarray] | None = None,
    terms_function: Callable[[np.ndarray], ObjectiveTerms] | None = None,
    enforce_ess: bool = True,
) -> SafeFreeSIGMResult:
    uniform = np.full(n_frequency, 1.0 / n_frequency)
    minimum_weight = config.minimum_weight(n_frequency)
    selected = uniform
    status = "fallback_uniform"
    if library_global_safe_weights is not None:
        try:
            candidate = _real_vector(
                library_global_safe_weights,
                name="library_global_safe_weights",
                size=n_frequency,
            )
            if np.any(candidate <= 0.0):
                raise ValueError("library-global safe weights must be positive")
            if not np.isclose(candidate.sum(), 1.0, atol=1.0e-12, rtol=0.0):
                raise ValueError(
                    "library-global safe weights must already sum to one"
                )
            candidate_diagnostics = _constraint_diagnostics(
                candidate,
                minimum_weight=minimum_weight,
                alpha_ess=config.alpha_ess,
                tolerance=config.feasibility_tolerance,
                enforce_ess=enforce_ess,
            )
            if not candidate_diagnostics.feasible:
                raise ValueError(
                    "library-global safe weights violate frozen constraints"
                )
            selected = candidate
            status = "fallback_library_global"
        except ValueError:
            status = "fallback_uniform"
    LOGGER.warning("Safe-FreeSIGM fail-closed for head: %s; status=%s", reason, status)
    # ``library_global_safe_weights`` is already the post-rho vector frozen by
    # library training.  Applying the conservative blend again would silently
    # double-shrink the registered fallback toward uniform.
    safe = selected
    constraints = _constraint_diagnostics(
        selected,
        minimum_weight=minimum_weight,
        alpha_ess=config.alpha_ess,
        tolerance=config.feasibility_tolerance,
        enforce_ess=enforce_ess,
    )
    kkt = KKTDiagnostics(np.inf, np.inf, 0.0, 0.0, 0.0, 0)
    objective_terms = terms_function(selected) if terms_function is not None else _zero_terms()
    return SafeFreeSIGMResult(
        raw_weights=selected,
        safe_weights=safe,
        effective_likelihood_weights=effective_evidence_weights(safe),
        raw_ess=effective_sample_size(selected),
        safe_ess=effective_sample_size(safe),
        objective_terms=objective_terms,
        constraints=constraints,
        kkt=kkt,
        normalized_scores={} if normalized is None else normalized,
        rho=config.rho,
        fallback=True,
        fallback_reason=reason,
        numerical_status=status,
        optimizer_iterations=0,
        deterministic=False,
        runtime_seconds=perf_counter() - started,
    )


def _resolve_library_global_safe_weights(
    *,
    library_global_safe_weights: object | None,
    library_global_weights: object | None,
) -> object | None:
    if (
        library_global_safe_weights is not None
        and library_global_weights is not None
    ):
        raise ValueError(
            "pass only library_global_safe_weights; library_global_weights is "
            "a deprecated alias"
        )
    if library_global_weights is not None:
        warnings.warn(
            "library_global_weights is deprecated; it is interpreted as an "
            "already-safe frozen vector. Use library_global_safe_weights.",
            DeprecationWarning,
            stacklevel=3,
        )
        return library_global_weights
    return library_global_safe_weights


def _solve_safe_freesigm(
    inputs: SafeFreeSIGMInputs,
    *,
    config: SafeFreeSIGMConfig,
    normalizer: RobustScoreNormalizer | None,
    library_global_safe_weights: object | None,
    enforce_ess: bool,
) -> SafeFreeSIGMResult:
    started = perf_counter()
    n_frequency = inputs.n_frequencies
    uniform = np.full(n_frequency, 1.0 / n_frequency)

    # Exact control: do not inspect adaptive scores and do not invoke the optimizer.
    if config.mode == "uniform_ablation":
        constraints = _constraint_diagnostics(
            uniform,
            minimum_weight=config.minimum_weight(n_frequency),
            alpha_ess=config.alpha_ess,
            tolerance=config.feasibility_tolerance,
            enforce_ess=enforce_ess,
        )
        return SafeFreeSIGMResult(
            raw_weights=uniform,
            safe_weights=uniform,
            effective_likelihood_weights=np.ones(n_frequency),
            raw_ess=float(n_frequency),
            safe_ess=float(n_frequency),
            objective_terms=_zero_terms(),
            constraints=constraints,
            kkt=KKTDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0),
            normalized_scores={},
            rho=0.0,
            fallback=False,
            fallback_reason=None,
            numerical_status="uniform_ablation_bypass",
            optimizer_iterations=0,
            deterministic=True,
            runtime_seconds=perf_counter() - started,
        )

    if normalizer is None:
        return _fallback_result(
            n_frequency=n_frequency,
            config=config,
            reason="missing_library_train_normalizer",
            library_global_safe_weights=library_global_safe_weights,
            started=started,
            enforce_ess=enforce_ess,
        )

    try:
        normalized, _ = _validate_inputs(inputs, normalizer)
        objective, gradient, terms = _objective_functions(normalized, config)
        minimum_weight = config.minimum_weight(n_frequency)
        first = _solve_once(
            objective,
            gradient,
            n_frequency=n_frequency,
            minimum_weight=minimum_weight,
            alpha_ess=config.alpha_ess,
            config=config,
            enforce_ess=enforce_ess,
        )
        weights = np.asarray(first.x, dtype=np.float64)
        constraints = _constraint_diagnostics(
            weights,
            minimum_weight=minimum_weight,
            alpha_ess=config.alpha_ess,
            tolerance=config.feasibility_tolerance,
            enforce_ess=enforce_ess,
        )
        kkt = _kkt_diagnostics(
            weights,
            gradient(weights),
            constraints,
            minimum_weight=minimum_weight,
            alpha_ess=config.alpha_ess,
            active_tolerance=max(1.0e-6, 10.0 * config.feasibility_tolerance),
            enforce_ess=enforce_ess,
        )
        deterministic = True
        if config.verify_determinism:
            second = _solve_once(
                objective,
                gradient,
                n_frequency=n_frequency,
                minimum_weight=minimum_weight,
                alpha_ess=config.alpha_ess,
                config=config,
                enforce_ess=enforce_ess,
            )
            deterministic = bool(
                second.success
                and np.max(np.abs(np.asarray(second.x) - weights), initial=0.0)
                <= config.determinism_tolerance
            )
        failure_reasons = []
        if not bool(first.success):
            failure_reasons.append(f"optimizer:{first.message}")
        if not constraints.feasible:
            failure_reasons.append("constraint_residual")
        if kkt.stationarity_residual > config.kkt_tolerance:
            failure_reasons.append("kkt_stationarity")
        if kkt.complementarity_residual > config.kkt_tolerance:
            failure_reasons.append("kkt_complementarity")
        if not deterministic:
            failure_reasons.append("nondeterministic_solution")
        if failure_reasons:
            return _fallback_result(
                n_frequency=n_frequency,
                config=config,
                reason=";".join(failure_reasons),
                library_global_safe_weights=library_global_safe_weights,
                started=started,
                normalized=normalized,
                terms_function=terms,
                enforce_ess=enforce_ess,
            )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return _fallback_result(
            n_frequency=n_frequency,
            config=config,
            reason=f"invalid_or_unstable_scores:{type(error).__name__}",
            library_global_safe_weights=library_global_safe_weights,
            started=started,
            enforce_ess=enforce_ess,
        )

    safe = (1.0 - config.rho) * uniform + config.rho * weights
    return SafeFreeSIGMResult(
        raw_weights=weights,
        safe_weights=safe,
        effective_likelihood_weights=effective_evidence_weights(safe),
        raw_ess=effective_sample_size(weights),
        safe_ess=effective_sample_size(safe),
        objective_terms=terms(weights),
        constraints=constraints,
        kkt=kkt,
        normalized_scores=normalized,
        rho=config.rho,
        fallback=False,
        fallback_reason=None,
        numerical_status=(
            "optimal" if enforce_ess else "optimal_diagnostic_no_ess"
        ),
        optimizer_iterations=int(first.nit),
        deterministic=deterministic,
        runtime_seconds=perf_counter() - started,
    )


def solve_safe_freesigm(
    inputs: SafeFreeSIGMInputs,
    *,
    config: SafeFreeSIGMConfig,
    normalizer: RobustScoreNormalizer | None,
    library_global_safe_weights: object | None = None,
    library_global_weights: object | None = None,
) -> SafeFreeSIGMResult:
    """Solve the frozen production Safe-FreeSIGM problem.

    ``library_global_safe_weights`` must be the already blended vector frozen
    by ``library_train``.  The deprecated ``library_global_weights`` keyword is
    retained as an alias with the same corrected, already-safe semantics.
    """

    fallback = _resolve_library_global_safe_weights(
        library_global_safe_weights=library_global_safe_weights,
        library_global_weights=library_global_weights,
    )
    return _solve_safe_freesigm(
        inputs,
        config=config,
        normalizer=normalizer,
        library_global_safe_weights=fallback,
        enforce_ess=True,
    )


def solve_safe_freesigm_no_ess_diagnostic(
    inputs: SafeFreeSIGMInputs,
    *,
    config: SafeFreeSIGMConfig,
    normalizer: RobustScoreNormalizer | None,
    library_global_safe_weights: object | None = None,
    library_global_weights: object | None = None,
) -> SafeFreeSIGMResult:
    """Run the preregistered diagnostic ``no_ESS`` control only.

    The frozen production config remains a positive-rho ``full`` config with
    an allowed ``alpha_ess``.  This separate entry point removes only the ESS
    inequality from the otherwise identical objective, simplex, positivity,
    optimizer, safe blend, and fallback path.  The returned constraint record
    explicitly sets ``ess_constraint_enforced=False`` and ``ess_floor=1``.
    """

    if config.mode != "full" or config.rho <= 0.0:
        raise ValueError(
            "diagnostic no-ESS control requires a frozen positive-rho full config"
        )
    fallback = _resolve_library_global_safe_weights(
        library_global_safe_weights=library_global_safe_weights,
        library_global_weights=library_global_weights,
    )
    return _solve_safe_freesigm(
        inputs,
        config=config,
        normalizer=normalizer,
        library_global_safe_weights=fallback,
        enforce_ess=False,
    )


def solve_shared_head_weights(
    model_inputs: Sequence[SafeFreeSIGMInputs],
    model_probabilities: object,
    *,
    config: SafeFreeSIGMConfig,
    normalizer: RobustScoreNormalizer | None,
    library_global_safe_weights: object | None = None,
    library_global_weights: object | None = None,
) -> SharedHeadSafeFreeSIGMResult:
    """Solve one head-level weight vector and broadcast it to all BEM models."""

    if not model_inputs:
        raise ValueError("at least one BEM model is required")
    probability = _real_vector(
        model_probabilities, name="model_probabilities", size=len(model_inputs)
    )
    if np.any(probability < 0.0) or probability.sum() <= 0.0:
        raise ValueError("model probabilities must be nonnegative with positive mass")
    probability = probability / probability.sum()
    n_frequency = model_inputs[0].n_frequencies
    if any(item.n_frequencies != n_frequency for item in model_inputs):
        raise ValueError("all BEM models must share the same frequency grid")
    aggregate = SafeFreeSIGMInputs(
        evidence=np.tensordot(probability, np.stack([item.evidence for item in model_inputs]), axes=1),
        information_matrices=np.tensordot(
            probability,
            np.stack([item.information_matrices for item in model_inputs]),
            axes=1,
        ),
        spectral_bound=np.tensordot(
            probability, np.stack([item.spectral_bound for item in model_inputs]), axes=1
        ),
        operator_uncertainty=np.tensordot(
            probability,
            np.stack([item.operator_uncertainty for item in model_inputs]),
            axes=1,
        ),
        redundancy_matrix=np.tensordot(
            probability,
            np.stack([item.redundancy_matrix for item in model_inputs]),
            axes=1,
        ),
    )
    shared = solve_safe_freesigm(
        aggregate,
        config=config,
        normalizer=normalizer,
        library_global_safe_weights=library_global_safe_weights,
        library_global_weights=library_global_weights,
    )
    return SharedHeadSafeFreeSIGMResult(
        shared=shared,
        weights_by_bem_model=np.repeat(
            shared.safe_weights[None, :], len(model_inputs), axis=0
        ),
    )


def account_freesigm_runtime(
    *,
    uniform_pipeline_seconds: float,
    safe_pipeline_seconds: float,
    freesigm_solver_seconds: float,
    target_fraction: float = 0.20,
) -> RuntimeOverheadAccounting:
    values = (
        uniform_pipeline_seconds,
        safe_pipeline_seconds,
        freesigm_solver_seconds,
        target_fraction,
    )
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("runtime values must be finite")
    if uniform_pipeline_seconds <= 0.0 or safe_pipeline_seconds < 0.0 or freesigm_solver_seconds < 0.0:
        raise ValueError("runtime values must be nonnegative and baseline must be positive")
    if target_fraction < 0.0:
        raise ValueError("target_fraction must be nonnegative")
    overhead = (safe_pipeline_seconds - uniform_pipeline_seconds) / uniform_pipeline_seconds
    solver_fraction = freesigm_solver_seconds / uniform_pipeline_seconds
    return RuntimeOverheadAccounting(
        uniform_pipeline_seconds=float(uniform_pipeline_seconds),
        safe_pipeline_seconds=float(safe_pipeline_seconds),
        freesigm_solver_seconds=float(freesigm_solver_seconds),
        overhead_fraction=float(overhead),
        solver_fraction=float(solver_fraction),
        target_fraction=float(target_fraction),
        within_target=bool(overhead <= target_fraction),
    )


@dataclass(frozen=True)
class LegacyReproductionConfig:
    """Frozen V2 diagnostic-control semantics, including the missing control."""

    minimum_ess_fraction: float = 0.45
    shuffle_seed: int = 0
    no_information_term_status: Literal["unavailable_noop_in_v2"] = (
        "unavailable_noop_in_v2"
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_ess_fraction <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0,1]")


@dataclass(frozen=True)
class LegacyControlRecord:
    name: str
    weights: np.ndarray
    ess: float
    objective: float
    source_distribution: np.ndarray
    independently_available_in_v2: bool
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _readonly(self.weights, dtype=np.float64))
        object.__setattr__(
            self, "source_distribution", _readonly(self.source_distribution)
        )


def reproduce_legacy_controls(
    *,
    learned_weights: object,
    no_ess_weights: object,
    config: LegacyReproductionConfig,
    evaluate_refit: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> Mapping[str, LegacyControlRecord]:
    """Reproduce V2 control weights and downstream refits on identical input.

    ``no_information_term`` was not an independently implemented V2 control.
    It is therefore returned as an explicitly labelled no-op copy of ``learned``
    rather than fabricating a missing historical result.
    """

    learned = _real_vector(learned_weights, name="learned_weights")
    no_ess = _real_vector(no_ess_weights, name="no_ess_weights", size=learned.size)
    if np.any(learned <= 0.0) or np.any(no_ess <= 0.0):
        raise ValueError("legacy weights must be strictly positive")
    learned = project_to_ess(
        learned / learned.sum(),
        max(1.0, config.minimum_ess_fraction * learned.size),
    )
    no_ess = no_ess / no_ess.sum()
    permutation = np.random.default_rng(config.shuffle_seed).permutation(learned.size)
    if learned.size > 1 and np.array_equal(permutation, np.arange(learned.size)):
        permutation = np.roll(permutation, 1)
    weights = {
        "uniform": np.full(learned.size, 1.0 / learned.size),
        "learned": learned,
        "shuffled": learned[permutation],
        "noESS": no_ess,
    }
    records: dict[str, LegacyControlRecord] = {}
    for name, value in weights.items():
        objective, source_distribution = evaluate_refit(np.array(value, copy=True))
        if not isfinite(float(objective)) or not np.all(np.isfinite(source_distribution)):
            raise ValueError("legacy evaluation callback returned non-finite output")
        records[name] = LegacyControlRecord(
            name=name,
            weights=value,
            ess=effective_sample_size(value),
            objective=float(objective),
            source_distribution=np.asarray(source_distribution),
            independently_available_in_v2=True,
        )
    learned_record = records["learned"]
    records["no_information_term"] = LegacyControlRecord(
        name="no_information_term",
        weights=learned_record.weights,
        ess=learned_record.ess,
        objective=learned_record.objective,
        source_distribution=learned_record.source_distribution,
        independently_available_in_v2=False,
        note=config.no_information_term_status,
    )
    return MappingProxyType(records)


__all__ = [
    "ALLOWED_ESS_FRACTIONS",
    "ALLOWED_RHO",
    "ConstraintDiagnostics",
    "FreeSIGMControlContract",
    "KKTDiagnostics",
    "LegacyControlRecord",
    "LegacyReproductionConfig",
    "ObjectiveTerms",
    "RobustScale",
    "RobustScoreNormalizer",
    "RuntimeOverheadAccounting",
    "SCORE_NAMES",
    "SafeFreeSIGMConfig",
    "SafeFreeSIGMInputs",
    "SafeFreeSIGMResult",
    "SharedHeadSafeFreeSIGMResult",
    "account_freesigm_runtime",
    "effective_evidence_weights",
    "fit_library_score_normalizer",
    "information_logdet_and_gradient",
    "matched_control_contract",
    "nuisance_projected_information",
    "operator_uncertainty_penalty",
    "project_psd",
    "proper_complex_negative_evidence",
    "redundancy_matrix_from_subspaces",
    "reproduce_legacy_controls",
    "solve_safe_freesigm",
    "solve_safe_freesigm_no_ess_diagnostic",
    "solve_shared_head_weights",
    "spectral_stability_bound",
    "weighted_evidence_sum",
]
