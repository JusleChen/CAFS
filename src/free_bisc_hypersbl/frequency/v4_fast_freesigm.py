"""Closed-form, leakage-safe frequency weighting for V4.

The V3 Safe-FreeSIGM implementation deliberately solved a tiny constrained
optimization problem.  V4 retains its library-frozen score construction but
replaces the optimizer with three deterministic operations: softmax, the
analytic uniform-path ESS shrink, and a conservative positive-``rho`` blend.

Missing frequencies are handled on their available sub-simplex.  Returned
vectors retain the caller's full frequency axis, with exact zeros at missing
positions and ``omega = F_available * w_safe`` on the available positions.
The exact-uniform arm bypasses every adaptive score and is therefore a literal
unweighted sum over the available frequency family.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from scipy import linalg

FAST_SCORE_NAMES = (
    "evidence",
    "information",
    "spectral_bound",
    "operator_uncertainty",
    "redundancy",
)
_V3_NORMALIZER_NAMES = {
    "evidence": "evidence",
    "information": "information_trace",
    "spectral_bound": "spectral_bound",
    "operator_uncertainty": "operator_uncertainty",
    "redundancy": "redundancy",
}
FAST_FORMULA_VERSION = "fast_safe_freesigm_v4_closed_form_1"
MINIMUM_WEIGHT_DEPENDENT_FIELDS = frozenset(
    {"gamma", "noise", "source_posterior"}
)


def _readonly(value: object, *, dtype: object = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _vector(
    value: object,
    *,
    name: str,
    size: int | None = None,
    require_finite: bool = True,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 1:
        raise ValueError(f"{name} must be a nonempty vector")
    if size is not None and result.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},)")
    if require_finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _mask(value: object | None, *, size: int) -> np.ndarray:
    if value is None:
        result = np.ones(size, dtype=bool)
    else:
        raw = np.asarray(value)
        if raw.dtype.kind != "b" or raw.shape != (size,):
            raise ValueError("available_mask must be a boolean frequency vector")
        result = np.asarray(raw, dtype=bool)
    if not np.any(result):
        raise ValueError("at least one frequency must be available")
    return result


def _embed(active: np.ndarray, available: np.ndarray) -> np.ndarray:
    result = np.zeros(available.size, dtype=np.float64)
    result[available] = np.asarray(active, dtype=np.float64)
    return result


def _simplex(value: object, *, name: str) -> np.ndarray:
    result = _vector(value, name=name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    total = float(np.sum(result))
    if not isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} has invalid total mass")
    result = result / total
    return result


def _effective_sample_size(value: object) -> float:
    probability = _simplex(value, name="ESS probability")
    return float(1.0 / np.sum(probability**2))


def softmax_simplex(scores: object, *, temperature: float) -> np.ndarray:
    """Return a strictly positive, stable softmax simplex."""

    values = _vector(scores, name="scores")
    scale = float(temperature)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    shifted = values / scale
    shifted -= float(np.max(shifted))
    # Ordinary exp can underflow to an exact zero for a strongly directed
    # score.  Clipping at the smallest normal float retains strict positivity
    # without materially changing any representable probability.
    shifted = np.maximum(shifted, np.log(np.finfo(np.float64).tiny))
    probability = np.exp(shifted)
    probability /= float(np.sum(probability))
    probability = np.maximum(probability, np.finfo(np.float64).tiny)
    probability /= float(np.sum(probability))
    return probability


def closed_form_ess_shrink(
    raw_probability: object,
    *,
    alpha_ess: float,
    epsilon: float = 1.0e-15,
) -> tuple[np.ndarray, float]:
    """Shrink along the uniform path to the ``alpha_ess * F`` ESS floor.

    If ``p_ess = (1-kappa) v + kappa p``, orthogonality to the uniform
    direction gives

    ``sum(p_ess**2) = 1/F + kappa**2 * (sum(p**2) - 1/F)``.

    Solving this inequality gives the closed-form shrinkage coefficient;
    no iterative projection or general optimizer is used here.
    """

    probability = _simplex(raw_probability, name="raw_probability")
    alpha = float(alpha_ess)
    eps = float(epsilon)
    if not isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha_ess must lie in (0, 1]")
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    count = probability.size
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    raw_denom = float(probability @ probability - 1.0 / count)
    if raw_denom <= eps:
        kappa = 1.0
    else:
        denom = max(raw_denom, eps)
        numer = max(1.0 / (alpha * count) - 1.0 / count, 0.0)
        kappa = min(1.0, float(np.sqrt(numer / denom)))
    shrunk = (1.0 - kappa) * uniform + kappa * probability
    # The affine construction already sums to one.  This normalization only
    # removes a final rounding ulp and is permutation equivariant.
    shrunk /= float(np.sum(shrunk))
    floor = alpha * count
    if _effective_sample_size(shrunk) + 2.0e-12 < floor:
        raise FloatingPointError("closed-form ESS shrink missed its analytic floor")
    return shrunk, kappa


@dataclass(frozen=True, slots=True)
class FastSafeFreeSIGMConfig:
    """Frozen V4 closed-form weighting surface."""

    enabled: bool = True
    mode: Literal["fast_safe_closed_form", "exact_uniform_ablation"] = (
        "fast_safe_closed_form"
    )
    rho: float = 0.05
    alpha_ess: float = 0.80
    temperature: float = 1.0
    a_evidence: float = 1.0
    a_information: float = 0.10
    a_spectral: float = 0.10
    a_uncertainty: float = 0.10
    a_redundancy: float = 0.05
    epsilon: float = 1.0e-15
    formula_version: Literal["fast_safe_freesigm_v4_closed_form_1"] = (
        FAST_FORMULA_VERSION
    )

    def __post_init__(self) -> None:
        if self.formula_version != FAST_FORMULA_VERSION:
            raise ValueError("Fast-FreeSIGM formula version changed")
        if self.enabled is not True:
            raise ValueError("V4 Fast-Safe-FreeSIGM must remain enabled")
        if self.mode == "fast_safe_closed_form":
            if not isfinite(float(self.rho)) or not 0.0 < float(self.rho) <= 1.0:
                raise ValueError("full Fast-Safe-FreeSIGM requires rho in (0, 1]")
            if self.a_evidence <= 0.0:
                raise ValueError("the evidence coefficient must be positive")
            if self.a_information <= 0.0 or self.a_spectral <= 0.0:
                raise ValueError(
                    "full Fast-Safe-FreeSIGM requires active information and spectral terms"
                )
        elif self.mode == "exact_uniform_ablation":
            if float(self.rho) != 0.0:
                raise ValueError("exact uniform ablation requires rho=0")
        else:
            raise ValueError("unknown Fast-Safe-FreeSIGM mode")
        if not isfinite(float(self.alpha_ess)) or not 0.0 < self.alpha_ess <= 1.0:
            raise ValueError("alpha_ess must lie in (0, 1]")
        if not isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        coefficients = (
            self.a_evidence,
            self.a_information,
            self.a_spectral,
            self.a_uncertainty,
            self.a_redundancy,
        )
        if any(not isfinite(float(value)) or value < 0.0 for value in coefficients):
            raise ValueError("Fast-FreeSIGM coefficients must be finite and nonnegative")
        if not isfinite(float(self.epsilon)) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")


@dataclass(frozen=True, slots=True)
class FastSafeFreeSIGMInputs:
    """Five scalar V4 score families on a common frequency axis."""

    evidence: np.ndarray
    information: np.ndarray
    spectral_bound: np.ndarray
    operator_uncertainty: np.ndarray
    redundancy: np.ndarray
    available_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        evidence = _vector(
            self.evidence, name="evidence", require_finite=False
        )
        count = evidence.size
        values = {"evidence": evidence}
        for name in FAST_SCORE_NAMES[1:]:
            values[name] = _vector(
                getattr(self, name),
                name=name,
                size=count,
                require_finite=False,
            )
        available = _mask(self.available_mask, size=count)
        for name, value in values.items():
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "available_mask", _readonly(available, dtype=bool))

    @property
    def n_frequencies(self) -> int:
        return int(np.asarray(self.evidence).size)

    @property
    def available_count(self) -> int:
        return int(np.count_nonzero(self.available_mask))


def build_fast_inputs_from_v3_components(
    inputs: object,
    *,
    available_mask: object | None = None,
) -> FastSafeFreeSIGMInputs:
    """Scalarize the already-tested V3 PSD information/redundancy objects.

    Information uses the trace of each nuisance-projected PSD matrix.  The
    redundancy burden is the mean off-diagonal row burden, matching the V4
    ``sum_g R_fg / (F-1)`` definition.
    """

    # Keep the optimizer-bearing V3 module outside V4's per-head import path.
    # This adapter is used only when importing already-tested V3 score objects.
    from free_bisc_hypersbl.freesigm.safe import SafeFreeSIGMInputs

    if not isinstance(inputs, SafeFreeSIGMInputs):
        raise TypeError("inputs must be the V3 SafeFreeSIGMInputs contract")
    evidence = np.asarray(inputs.evidence, dtype=np.float64)
    count = evidence.size
    information_matrix = np.asarray(inputs.information_matrices, dtype=np.complex128)
    if (
        information_matrix.ndim != 3
        or information_matrix.shape[0] != count
        or information_matrix.shape[1] != information_matrix.shape[2]
    ):
        raise ValueError("information matrices must have shape (F,P,P)")
    traces = np.trace(information_matrix, axis1=1, axis2=2)
    if np.max(np.abs(traces.imag), initial=0.0) > 1.0e-10:
        raise ValueError("information traces have material imaginary components")
    redundancy_matrix = np.asarray(inputs.redundancy_matrix, dtype=np.float64)
    if redundancy_matrix.shape != (count, count):
        raise ValueError("redundancy matrix must have shape (F,F)")
    if count == 1:
        redundancy = np.zeros(1, dtype=np.float64)
    else:
        redundancy = (
            np.sum(redundancy_matrix, axis=1) - np.diag(redundancy_matrix)
        ) / (count - 1)
    return FastSafeFreeSIGMInputs(
        evidence=evidence,
        information=traces.real,
        spectral_bound=inputs.spectral_bound,
        operator_uncertainty=inputs.operator_uncertainty,
        redundancy=redundancy,
        available_mask=available_mask,
    )


def fit_fast_library_score_normalizer(
    library_scores: Mapping[str, object],
    *,
    source_split: str,
) -> object:
    """Fit V4 scalar robust scales through the V3 library-only implementation."""

    # Normalizer fitting is an offline library operation.  The lazy import
    # prevents V3's constrained solver from entering the V4 per-head runtime.
    from free_bisc_hypersbl.freesigm.safe import fit_library_score_normalizer

    if set(library_scores) != set(FAST_SCORE_NAMES):
        raise ValueError(f"library_scores must contain exactly {FAST_SCORE_NAMES}")
    translated = {
        _V3_NORMALIZER_NAMES[name]: library_scores[name]
        for name in FAST_SCORE_NAMES
    }
    return fit_library_score_normalizer(
        translated,
        source_split=source_split,
    )


@dataclass(frozen=True, slots=True)
class FastSafeFreeSIGMResult:
    w_raw: np.ndarray
    w_ess: np.ndarray
    w_safe: np.ndarray
    omega: np.ndarray
    score: np.ndarray
    normalized_scores: Mapping[str, np.ndarray]
    available_mask: np.ndarray
    available_count: int
    kappa_ess: float
    ess_raw: float
    ess_after_shrink: float
    ess_safe: float
    rho: float
    mode: str
    score_substitution: str
    runtime_seconds: float
    formula_version: str = FAST_FORMULA_VERSION

    def __post_init__(self) -> None:
        for name in ("w_raw", "w_ess", "w_safe", "omega", "score"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(
            self, "available_mask", _readonly(self.available_mask, dtype=bool)
        )
        frozen = {
            str(name): _readonly(value)
            for name, value in dict(self.normalized_scores).items()
        }
        object.__setattr__(self, "normalized_scores", MappingProxyType(frozen))

    @property
    def raw_weights(self) -> np.ndarray:
        return self.w_raw

    @property
    def ess_weights(self) -> np.ndarray:
        return self.w_ess

    @property
    def safe_weights(self) -> np.ndarray:
        return self.w_safe

    @property
    def effective_likelihood_weights(self) -> np.ndarray:
        return self.omega


def _restricted_prior(value: object, available: np.ndarray) -> np.ndarray:
    prior = _vector(value, name="library_global_prior", size=available.size)
    if np.any(prior <= 0.0):
        raise ValueError("library_global_prior must be strictly positive")
    return _simplex(prior[available], name="available library_global_prior")


def _validate_active_nonnegative(inputs: FastSafeFreeSIGMInputs) -> None:
    available = np.asarray(inputs.available_mask, dtype=bool)
    for name in FAST_SCORE_NAMES[1:]:
        active = np.asarray(getattr(inputs, name), dtype=np.float64)[available]
        finite = active[np.isfinite(active)]
        if np.any(finite < 0.0):
            raise ValueError(f"{name} must be nonnegative when finite")


def solve_fast_safe_freesigm(
    inputs: FastSafeFreeSIGMInputs,
    *,
    config: FastSafeFreeSIGMConfig,
    normalizer: object | None,
    library_global_prior: object | None,
) -> FastSafeFreeSIGMResult:
    """Execute the V4 closed-form weight computation on one head."""

    started = perf_counter()
    if not isinstance(inputs, FastSafeFreeSIGMInputs):
        raise TypeError("inputs must be FastSafeFreeSIGMInputs")
    if not isinstance(config, FastSafeFreeSIGMConfig):
        raise TypeError("config must be FastSafeFreeSIGMConfig")
    available = np.asarray(inputs.available_mask, dtype=bool)
    count = int(np.count_nonzero(available))
    uniform_active = np.full(count, 1.0 / count, dtype=np.float64)

    if config.mode == "exact_uniform_ablation":
        uniform = _embed(uniform_active, available)
        omega = _embed(np.ones(count, dtype=np.float64), available)
        return FastSafeFreeSIGMResult(
            w_raw=uniform,
            w_ess=uniform,
            w_safe=uniform,
            omega=omega,
            score=np.zeros(inputs.n_frequencies, dtype=np.float64),
            normalized_scores={},
            available_mask=available,
            available_count=count,
            kappa_ess=1.0,
            ess_raw=float(count),
            ess_after_shrink=float(count),
            ess_safe=float(count),
            rho=0.0,
            mode=config.mode,
            score_substitution="uniform_ablation_bypass",
            runtime_seconds=perf_counter() - started,
        )

    if (
        normalizer is None
        or not callable(getattr(normalizer, "transform", None))
        or not isinstance(getattr(normalizer, "source_split", None), str)
    ):
        raise TypeError("full mode requires a frozen robust score normalizer")
    if normalizer.source_split != "library_train":
        raise ValueError("score normalization must remain library_train-only")
    if library_global_prior is None:
        raise ValueError("full mode requires a frozen nonuniform library global prior")
    global_prior_active = _restricted_prior(library_global_prior, available)
    _validate_active_nonnegative(inputs)

    active_values = {
        name: np.asarray(getattr(inputs, name), dtype=np.float64)[available]
        for name in FAST_SCORE_NAMES
    }
    if any(not np.all(np.isfinite(value)) for value in active_values.values()):
        if count > 1 and np.array_equal(global_prior_active, uniform_active):
            raise ValueError(
                "nonfinite-score substitution requires a nonuniform library global prior"
            )
        raw_active = global_prior_active
        normalized_active: dict[str, np.ndarray] = {}
        combined_active = np.zeros(count, dtype=np.float64)
        substitution = "library_global_prior_nonfinite_head_score"
    else:
        normalized_active = {
            name: np.asarray(
                normalizer.transform(_V3_NORMALIZER_NAMES[name], value),
                dtype=np.float64,
            )
            for name, value in active_values.items()
        }
        combined_active = (
            -config.a_evidence * normalized_active["evidence"]
            + config.a_information * normalized_active["information"]
            - config.a_spectral * normalized_active["spectral_bound"]
            - config.a_uncertainty * normalized_active["operator_uncertainty"]
            - config.a_redundancy * normalized_active["redundancy"]
        )
        raw_active = softmax_simplex(
            combined_active, temperature=config.temperature
        )
        substitution = "none"

    ess_active, kappa = closed_form_ess_shrink(
        raw_active,
        alpha_ess=config.alpha_ess,
        epsilon=config.epsilon,
    )
    safe_active = (1.0 - config.rho) * uniform_active + config.rho * ess_active
    safe_active /= float(np.sum(safe_active))
    normalized = {
        name: _embed(value, available)
        for name, value in normalized_active.items()
    }
    return FastSafeFreeSIGMResult(
        w_raw=_embed(raw_active, available),
        w_ess=_embed(ess_active, available),
        w_safe=_embed(safe_active, available),
        omega=_embed(count * safe_active, available),
        score=_embed(combined_active, available),
        normalized_scores=normalized,
        available_mask=available,
        available_count=count,
        kappa_ess=float(kappa),
        ess_raw=_effective_sample_size(raw_active),
        ess_after_shrink=_effective_sample_size(ess_active),
        ess_safe=_effective_sample_size(safe_active),
        rho=float(config.rho),
        mode=config.mode,
        score_substitution=substitution,
        runtime_seconds=perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class PilotEvidenceResult:
    values: np.ndarray
    cache_token: str
    reused_factorization_count: int
    new_factorization_count: int
    return_per_sample: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values))


@dataclass(frozen=True, slots=True)
class PilotFactorizationCache:
    """Read-only proper-complex pilot Cholesky factors reused by scoring."""

    cholesky_factors: np.ndarray
    log_determinants: np.ndarray
    cache_token: str
    factorization_count: int

    def __post_init__(self) -> None:
        factors = np.asarray(self.cholesky_factors, dtype=np.complex128)
        logdet = np.asarray(self.log_determinants, dtype=np.float64)
        if (
            factors.ndim != 3
            or factors.shape[1] != factors.shape[2]
            or logdet.shape != (factors.shape[0],)
            or not np.all(np.isfinite(factors))
            or not np.all(np.isfinite(logdet))
        ):
            raise ValueError("pilot factorization cache arrays are invalid")
        if int(self.factorization_count) != factors.shape[0]:
            raise ValueError("pilot factorization count differs from its frequency axis")
        if (
            not isinstance(self.cache_token, str)
            or len(self.cache_token) != 64
            or any(character not in "0123456789abcdef" for character in self.cache_token)
        ):
            raise ValueError("pilot factorization cache token must be SHA-256")
        object.__setattr__(
            self,
            "cholesky_factors",
            _readonly(factors, dtype=np.complex128),
        )
        object.__setattr__(
            self, "log_determinants", _readonly(logdet)
        )

    @property
    def n_frequencies(self) -> int:
        return int(self.cholesky_factors.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.cholesky_factors.shape[1])

    def proper_complex_negative_evidence(
        self,
        sample_covariances: object,
        *,
        sample_counts: object | None = None,
        return_per_sample: bool = True,
    ) -> PilotEvidenceResult:
        sample = np.asarray(sample_covariances, dtype=np.complex128)
        if sample.ndim == 2:
            sample = sample[None, ...]
        expected = (
            self.n_frequencies,
            self.n_sensors,
            self.n_sensors,
        )
        if sample.shape != expected or not np.all(np.isfinite(sample)):
            raise ValueError("sample_covariances must be finite with shape (F,E,E)")
        counts = np.ones(self.n_frequencies, dtype=np.float64)
        if sample_counts is not None:
            counts = _vector(
                sample_counts,
                name="sample_counts",
                size=self.n_frequencies,
            )
            if np.any(counts <= 0.0):
                raise ValueError("sample_counts must be positive")
        values = np.empty(self.n_frequencies, dtype=np.float64)
        for index, (factor, covariance) in enumerate(
            zip(self.cholesky_factors, sample, strict=True)
        ):
            hermitian = 0.5 * (covariance + covariance.conj().T)
            solved = linalg.cho_solve(
                (factor, True), hermitian, check_finite=False
            )
            trace = np.trace(solved)
            if abs(float(trace.imag)) > 1.0e-8 * max(1.0, abs(float(trace.real))):
                raise ValueError("proper-complex evidence has a material imaginary trace")
            value = float(self.log_determinants[index] + trace.real)
            values[index] = value if return_per_sample else counts[index] * value
        return PilotEvidenceResult(
            values=values,
            cache_token=self.cache_token,
            reused_factorization_count=self.n_frequencies,
            new_factorization_count=0,
            return_per_sample=bool(return_per_sample),
        )


def factorize_pilot_sensor_covariances(
    sensor_covariances: object,
) -> PilotFactorizationCache:
    """Factor each pilot covariance exactly once for all downstream scores."""

    covariance = np.asarray(sensor_covariances, dtype=np.complex128)
    if covariance.ndim == 2:
        covariance = covariance[None, ...]
    if (
        covariance.ndim != 3
        or covariance.shape[1] != covariance.shape[2]
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("sensor_covariances must be finite with shape (F,E,E)")
    factors = np.empty_like(covariance)
    logdet = np.empty(covariance.shape[0], dtype=np.float64)
    digest = hashlib.sha256()
    for index, matrix in enumerate(covariance):
        hermitian = 0.5 * (matrix + matrix.conj().T)
        factor = np.linalg.cholesky(hermitian)
        factors[index] = factor
        logdet[index] = 2.0 * float(np.log(np.diag(factor).real).sum())
        digest.update(np.ascontiguousarray(factor).tobytes())
    return PilotFactorizationCache(
        cholesky_factors=factors,
        log_determinants=logdet,
        cache_token=digest.hexdigest(),
        factorization_count=int(covariance.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class FastWeightDependentState:
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        copied = dict(self.values)
        if not copied or any(not isinstance(key, str) or not key for key in copied):
            raise ValueError("weight-dependent state needs named fields")
        object.__setattr__(self, "values", MappingProxyType(copied))

    @property
    def declared_fields(self) -> frozenset[str]:
        return frozenset(self.values)


@dataclass(frozen=True, slots=True)
class FastEndToEndFitRequest:
    control_name: Literal["full", "exact_uniform"]
    weighting: FastSafeFreeSIGMResult
    forbidden_fit_call_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class FastWeightDependentFit:
    fit_call_id: str
    control_name: Literal["full", "exact_uniform"]
    safe_weights_used: np.ndarray
    reestimated_fields: frozenset[str]
    state: FastWeightDependentState

    def __post_init__(self) -> None:
        if not isinstance(self.fit_call_id, str) or not self.fit_call_id:
            raise ValueError("fit_call_id must be nonempty")
        object.__setattr__(
            self, "safe_weights_used", _readonly(self.safe_weights_used)
        )
        object.__setattr__(
            self, "reestimated_fields", frozenset(self.reestimated_fields)
        )


@dataclass(frozen=True, slots=True)
class FastMatchedAblationResult:
    full: FastWeightDependentFit
    exact_uniform: FastWeightDependentFit


def _validate_weight_dependent_fit(
    fit: FastWeightDependentFit,
    request: FastEndToEndFitRequest,
    required_fields: frozenset[str],
) -> None:
    if not isinstance(fit, FastWeightDependentFit):
        raise TypeError("fit_callback must return FastWeightDependentFit")
    if fit.control_name != request.control_name:
        raise ValueError("fit callback returned the wrong control name")
    if fit.fit_call_id in request.forbidden_fit_call_ids:
        raise ValueError("exact-uniform arm reused the full fit call")
    if not np.array_equal(fit.safe_weights_used, request.weighting.w_safe):
        raise ValueError("fit callback used different frequency weights")
    if fit.reestimated_fields != fit.state.declared_fields:
        raise ValueError("fit callback did not re-estimate every declared state")
    missing = required_fields - fit.reestimated_fields
    if missing:
        raise ValueError(
            "fit callback omitted required weight-dependent states: "
            + ",".join(sorted(missing))
        )


def execute_fast_freesigm_matched_ablation(
    full: FastSafeFreeSIGMResult,
    exact_uniform: FastSafeFreeSIGMResult,
    *,
    fit_callback: Callable[[FastEndToEndFitRequest], FastWeightDependentFit],
    required_weight_dependent_fields: Sequence[str] = tuple(
        sorted(MINIMUM_WEIGHT_DEPENDENT_FIELDS)
    ),
) -> FastMatchedAblationResult:
    """Refit full and exact-uniform arms independently end to end."""

    if full.mode != "fast_safe_closed_form" or full.rho <= 0.0:
        raise ValueError("full arm must be positive-rho Fast-Safe-FreeSIGM")
    if exact_uniform.mode != "exact_uniform_ablation" or exact_uniform.rho != 0.0:
        raise ValueError("control arm must be the exact uniform ablation")
    if not np.array_equal(full.available_mask, exact_uniform.available_mask):
        raise ValueError("matched arms changed the available frequency family")
    required = frozenset(str(value) for value in required_weight_dependent_fields)
    if not MINIMUM_WEIGHT_DEPENDENT_FIELDS.issubset(required):
        raise ValueError("required fields omit the minimum V4 dependent state")

    full_request = FastEndToEndFitRequest("full", full)
    full_fit = fit_callback(full_request)
    _validate_weight_dependent_fit(full_fit, full_request, required)
    uniform_request = FastEndToEndFitRequest(
        "exact_uniform",
        exact_uniform,
        forbidden_fit_call_ids=frozenset({full_fit.fit_call_id}),
    )
    uniform_fit = fit_callback(uniform_request)
    _validate_weight_dependent_fit(uniform_fit, uniform_request, required)
    if uniform_fit is full_fit or uniform_fit.state is full_fit.state:
        raise ValueError("exact-uniform arm reused full-arm downstream state")
    return FastMatchedAblationResult(full_fit, uniform_fit)


@dataclass(frozen=True, slots=True)
class PairedRuntimeRepeat:
    repeat_index: int
    order: Literal["full_then_uniform", "uniform_then_full"]
    full_seconds: float
    uniform_seconds: float
    warm_cache: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.repeat_index, (bool, np.bool_))
            or not isinstance(self.repeat_index, (int, np.integer))
            or int(self.repeat_index) < 0
        ):
            raise ValueError("repeat_index must be a nonnegative integer")
        if self.order not in {"full_then_uniform", "uniform_then_full"}:
            raise ValueError("runtime order is invalid")
        if self.warm_cache is not True:
            raise ValueError("paired runtime repeats require a warm cache")
        if (
            not isfinite(float(self.full_seconds))
            or not isfinite(float(self.uniform_seconds))
            or self.full_seconds <= 0.0
            or self.uniform_seconds <= 0.0
        ):
            raise ValueError("paired method-exclusive runtimes must be positive")


@dataclass(frozen=True, slots=True)
class FastFreeSIGMRuntimeSummary:
    paired_ratios: np.ndarray
    repeat_count: int
    median_ratio: float
    p90_ratio: float
    q25_ratio: float
    q75_ratio: float
    iqr_ratio: float
    ratio_limit: float
    within_limit: bool
    one_time_library_cache_build_seconds: float
    order_alternated: bool
    warm_cache_all: bool
    method_exclusive: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "paired_ratios", _readonly(self.paired_ratios)
        )


def summarize_paired_runtime(
    repeats: Sequence[PairedRuntimeRepeat],
    *,
    ratio_limit: float = 1.20,
    one_time_library_cache_build_seconds: float = 0.0,
) -> FastFreeSIGMRuntimeSummary:
    """Summarize the V4 alternating-order, warm-cache paired benchmark."""

    values = tuple(sorted(repeats, key=lambda item: item.repeat_index))
    if len(values) < 3:
        raise ValueError("paired runtime accounting requires at least three repeats")
    indices = tuple(item.repeat_index for item in values)
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ValueError("runtime repeat indices must be unique and contiguous")
    alternated = all(
        left.order != right.order for left, right in zip(values[:-1], values[1:])
    )
    if not alternated:
        raise ValueError("full/uniform runtime order must alternate")
    limit = float(ratio_limit)
    cache_seconds = float(one_time_library_cache_build_seconds)
    if not isfinite(limit) or limit <= 0.0:
        raise ValueError("ratio_limit must be finite and positive")
    if not isfinite(cache_seconds) or cache_seconds < 0.0:
        raise ValueError("one-time cache build time must be finite and nonnegative")
    ratios = np.asarray(
        [item.full_seconds / item.uniform_seconds for item in values],
        dtype=np.float64,
    )
    q25, q75 = np.quantile(ratios, (0.25, 0.75))
    median = float(np.median(ratios))
    return FastFreeSIGMRuntimeSummary(
        paired_ratios=ratios,
        repeat_count=len(values),
        median_ratio=median,
        p90_ratio=float(np.quantile(ratios, 0.90)),
        q25_ratio=float(q25),
        q75_ratio=float(q75),
        iqr_ratio=float(q75 - q25),
        ratio_limit=limit,
        within_limit=bool(median <= limit),
        one_time_library_cache_build_seconds=cache_seconds,
        order_alternated=True,
        warm_cache_all=all(item.warm_cache for item in values),
    )


__all__ = [
    "FAST_FORMULA_VERSION",
    "FAST_SCORE_NAMES",
    "MINIMUM_WEIGHT_DEPENDENT_FIELDS",
    "FastEndToEndFitRequest",
    "FastFreeSIGMRuntimeSummary",
    "FastMatchedAblationResult",
    "FastSafeFreeSIGMConfig",
    "FastSafeFreeSIGMInputs",
    "FastSafeFreeSIGMResult",
    "FastWeightDependentFit",
    "FastWeightDependentState",
    "PairedRuntimeRepeat",
    "PilotEvidenceResult",
    "PilotFactorizationCache",
    "build_fast_inputs_from_v3_components",
    "closed_form_ess_shrink",
    "execute_fast_freesigm_matched_ablation",
    "factorize_pilot_sensor_covariances",
    "fit_fast_library_score_normalizer",
    "softmax_simplex",
    "solve_fast_safe_freesigm",
    "summarize_paired_runtime",
]
