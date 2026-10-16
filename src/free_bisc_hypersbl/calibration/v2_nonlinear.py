"""Nonlinear, leakage-safe multi-episode calibration for the V2 protocol.

The routines in this module deliberately keep the nonlinear forward builder
outside the optimiser.  A caller supplies ``reassemble(theta)`` and therefore
every trial point is evaluated with a newly assembled :class:`LeadfieldFamily`
and the analytic Jacobian returned at that same point.  No truth-valued field
is accepted by this API.

The lower problem is the existing proper-complex Champagne/group-SBL solver.
For each spatial fold and frequency, all calibration episodes are concatenated
along the snapshot axis.  This shares ARD/noise hyperparameters across
episodes while retaining a different posterior source time course for every
episode.  The held-out likelihood includes both sensor noise and propagated
source-posterior uncertainty, ``G_V Sigma_x G_V^H``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from free_bisc_hypersbl.calibration.bilevel import ElectrodeSplit
from free_bisc_hypersbl.forward.global_matrix import LeadfieldFamily
from free_bisc_hypersbl.inverse.complex_sbl import ComplexGroupSBL, SBLResult


ComplexArray = np.ndarray
FloatArray = np.ndarray


@dataclass(frozen=True)
class PredictiveNLPDComponents:
    """Terms of a proper-complex held-out predictive density."""

    total: float
    per_complex_observation: float
    log_determinant: float
    quadratic: float
    n_sensors: int
    n_snapshots: int
    covariance: ComplexArray
    residual: ComplexArray


@dataclass(frozen=True)
class IdentifiableSubspace:
    """Stable SVD selection in prior-whitened parameter coordinates.

    ``whitened_directions`` contains directions in ``z`` coordinates, where
    ``theta = diag(parameter_scales) P^(-1/2) z`` and ``P`` is the precision
    in dimensionless parameter coordinates.  ``physical_directions`` applies
    that map and can therefore be used directly for nonlinear trial steps.
    """

    singular_values: FloatArray
    relative_singular_values: FloatArray
    numerical_rank: int
    selected_rank: int
    whitened_directions: FloatArray
    physical_directions: FloatArray
    bootstrap_selection_frequency: FloatArray
    threshold: float
    audit: Mapping[str, Any]

    @property
    def directions(self) -> FloatArray:
        """Alias for the physical directions used by the optimiser."""

        return self.physical_directions


@dataclass(frozen=True)
class TrustRegionDecision:
    """Auditable Levenberg--Marquardt actual/predicted-ratio decision."""

    ratio: float
    accepted: bool
    radius: float
    damping: float
    reason: str


@dataclass(frozen=True)
class FoldFrequencyPosterior:
    """Lower posterior and held-out prediction for one fold/frequency."""

    fold_index: int
    frequency_index: int
    inner_indices: np.ndarray
    heldout_indices: np.ndarray
    mean: ComplexArray
    covariance: ComplexArray
    noise: float
    predictive_nlpd: float
    predictive_covariance: ComplexArray
    residual: ComplexArray


@dataclass(frozen=True)
class NonlinearCalibrationResult:
    """Result of joint nonlinear calibration across all episodes and folds."""

    theta: FloatArray
    normalized_theta: FloatArray
    leadfield_family: LeadfieldFamily
    fold_predictive_nlpd: FloatArray
    objective_history: tuple[float, ...]
    predictive_nlpd_history: tuple[float, ...]
    trust_region_history: tuple[Mapping[str, Any], ...]
    identifiable_subspace: IdentifiableSubspace
    posteriors: tuple[FoldFrequencyPosterior, ...]
    accepted_steps: int
    fallback_to_zero: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _FamilyEvaluation:
    objective: float
    predictive_nlpd: float
    fold_nlpd: FloatArray
    posteriors: tuple[FoldFrequencyPosterior, ...]


def _hermitian(array: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (array + array.conj().T), dtype=np.complex128)


def _predictive_components(
    validation_operator: ArrayLike,
    validation_observations: ArrayLike,
    posterior_mean: ArrayLike,
    posterior_covariance: ArrayLike | None,
    noise_covariance: ArrayLike | float,
    *,
    include_posterior_covariance: bool,
    jitter: float,
) -> PredictiveNLPDComponents:
    operator = np.asarray(validation_operator, dtype=np.complex128)
    target = np.asarray(validation_observations, dtype=np.complex128)
    mean = np.asarray(posterior_mean, dtype=np.complex128)
    if target.ndim == 1:
        target = target[:, None]
    if mean.ndim == 1:
        mean = mean[:, None]
    if operator.ndim != 2 or target.ndim != 2 or mean.ndim != 2:
        raise ValueError("operator, observations, and posterior mean must be matrices")
    if operator.shape[0] != target.shape[0] or operator.shape[1] != mean.shape[0]:
        raise ValueError("predictive operator/data/posterior dimensions do not align")
    if mean.shape[1] != target.shape[1]:
        raise ValueError("posterior mean and held-out data need equal snapshot counts")
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")

    n_sensor = operator.shape[0]
    noise = np.asarray(noise_covariance)
    if noise.ndim == 0:
        noise_value = float(noise)
        if not np.isfinite(noise_value) or noise_value <= 0.0:
            raise ValueError("scalar noise variance must be finite and positive")
        covariance = noise_value * np.eye(n_sensor, dtype=np.complex128)
    else:
        covariance = np.asarray(noise, dtype=np.complex128)
        if covariance.shape != (n_sensor, n_sensor):
            raise ValueError("noise covariance has the wrong shape")
        covariance = _hermitian(covariance)

    if include_posterior_covariance:
        if posterior_covariance is None:
            raise ValueError("posterior covariance is required for predictive NLPD")
        source_covariance = np.asarray(posterior_covariance, dtype=np.complex128)
        if source_covariance.shape != (operator.shape[1], operator.shape[1]):
            raise ValueError("posterior covariance has the wrong shape")
        covariance = covariance + operator @ _hermitian(source_covariance) @ operator.conj().T
    covariance = _hermitian(covariance)
    if jitter:
        covariance = covariance + float(jitter) * np.eye(n_sensor, dtype=np.complex128)
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("predictive covariance must be positive definite") from error

    residual = target - operator @ mean
    solved = np.linalg.solve(factor.conj().T, np.linalg.solve(factor, residual))
    log_determinant = float(2.0 * np.log(np.diagonal(factor).real).sum())
    quadratic = float(np.sum(np.conj(residual) * solved).real)
    total = float(
        target.shape[1]
        * (n_sensor * np.log(np.pi) + log_determinant)
        + quadratic
    )
    return PredictiveNLPDComponents(
        total=total,
        per_complex_observation=total / target.size,
        log_determinant=log_determinant,
        quadratic=quadratic,
        n_sensors=n_sensor,
        n_snapshots=target.shape[1],
        covariance=np.asarray(covariance, dtype=np.complex128),
        residual=np.asarray(residual, dtype=np.complex128),
    )


def proper_complex_predictive_nlpd(
    validation_operator: ArrayLike,
    validation_observations: ArrayLike,
    posterior_mean: ArrayLike,
    posterior_covariance: ArrayLike | None,
    noise_covariance: ArrayLike | float,
    *,
    include_posterior_covariance: bool = True,
    normalize: bool = True,
    jitter: float = 1.0e-10,
) -> float:
    """Evaluate the proper-complex predictive negative log density.

    The default is the protocol-required density with
    ``G_V Sigma_x G_V^H``.  The switch that omits posterior covariance exists
    only to support a matched diagnostic/unit test; the nonlinear calibrator
    never disables it.
    """

    components = _predictive_components(
        validation_operator,
        validation_observations,
        posterior_mean,
        posterior_covariance,
        noise_covariance,
        include_posterior_covariance=include_posterior_covariance,
        jitter=jitter,
    )
    return (
        components.per_complex_observation if normalize else components.total
    )


# A descriptive alias used by callers that emphasize the spatial split.
heldout_proper_complex_nlpd = proper_complex_predictive_nlpd


def _precision_and_inverse_sqrt(
    prior_precision: ArrayLike | None, n_parameter: int
) -> tuple[np.ndarray, np.ndarray]:
    if prior_precision is None:
        precision = np.eye(n_parameter, dtype=np.float64)
    else:
        raw = np.asarray(prior_precision, dtype=np.float64)
        if raw.ndim == 1:
            if raw.shape != (n_parameter,):
                raise ValueError("prior precision vector has the wrong shape")
            precision = np.diag(raw)
        else:
            if raw.shape != (n_parameter, n_parameter):
                raise ValueError("prior precision matrix has the wrong shape")
            precision = 0.5 * (raw + raw.T)
    if not np.all(np.isfinite(precision)):
        raise ValueError("prior precision contains non-finite values")
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("prior precision must be positive definite")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return precision, np.asarray(inverse_sqrt, dtype=np.float64)


def prior_whitened_identifiable_subspace(
    sensitivities: ArrayLike,
    parameter_scales: ArrayLike,
    *,
    prior_precision: ArrayLike | None = None,
    row_weights: ArrayLike | None = None,
    min_directions: int = 3,
    max_directions: int = 8,
    relative_threshold: float = 1.0e-6,
    absolute_threshold: float = 1.0e-10,
    bootstrap_count: int = 0,
    bootstrap_seed: int = 0,
    minimum_bootstrap_frequency: float = 0.60,
) -> IdentifiableSubspace:
    """Select stable identifiable directions from aggregated sensitivities.

    The final axis of ``sensitivities`` is the physical-parameter axis; all
    preceding axes may represent episodes, frequencies, folds, sensors, and
    snapshots.  Complex sensitivities are converted to their equivalent real
    Fisher design by stacking real and imaginary rows.

    The requested 3--8 range is respected whenever the numerical rank permits
    it.  A truly rank-deficient problem is never padded with null directions;
    the below-minimum condition is instead recorded in ``audit``.
    """

    design_raw = np.asarray(sensitivities)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    if design_raw.ndim < 2:
        raise ValueError("sensitivities need observations and a parameter axis")
    n_parameter = design_raw.shape[-1]
    if scales.shape != (n_parameter,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite, positive, and parameter-aligned")
    if not 1 <= int(min_directions) <= int(max_directions):
        raise ValueError("direction limits must satisfy 1 <= min <= max")
    if relative_threshold < 0.0 or absolute_threshold < 0.0:
        raise ValueError("SVD thresholds must be non-negative")
    if int(bootstrap_count) < 0:
        raise ValueError("bootstrap_count must be non-negative")
    if not 0.0 <= float(minimum_bootstrap_frequency) <= 1.0:
        raise ValueError("minimum_bootstrap_frequency must lie in [0, 1]")

    complex_design = np.asarray(design_raw, dtype=np.complex128).reshape(-1, n_parameter)
    if not np.all(np.isfinite(complex_design)):
        raise ValueError("sensitivities contain non-finite values")
    if complex_design.shape[0] == 0:
        raise ValueError("sensitivities contain no observations")
    if row_weights is not None:
        weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (complex_design.shape[0],):
            raise ValueError("row_weights must align with flattened sensitivity rows")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("row_weights must be finite, non-negative, and nonzero")
        complex_design = complex_design * np.sqrt(weights)[:, None]

    precision, prior_inverse_sqrt = _precision_and_inverse_sqrt(
        prior_precision, n_parameter
    )
    physical_from_whitened = np.diag(scales) @ prior_inverse_sqrt
    whitened_complex = complex_design @ physical_from_whitened
    real_design = np.vstack((whitened_complex.real, whitened_complex.imag))
    _, singular_values, right_h = np.linalg.svd(real_design, full_matrices=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(float(absolute_threshold), float(relative_threshold) * leading)
    numerical_rank = int(np.sum(singular_values > threshold))
    candidate_rank = min(numerical_rank, int(max_directions), n_parameter)
    candidate_directions = np.asarray(
        right_h[:candidate_rank].T, dtype=np.float64
    )
    relative = (
        np.asarray(singular_values / leading, dtype=np.float64)
        if leading > 0.0
        else np.zeros_like(singular_values, dtype=np.float64)
    )

    bootstrap_frequency = np.ones(candidate_rank, dtype=np.float64)
    mean_projector_overlap = np.ones(candidate_rank, dtype=np.float64)
    if bootstrap_count and candidate_rank:
        rng = np.random.default_rng(int(bootstrap_seed))
        selected = np.zeros(candidate_rank, dtype=np.float64)
        overlaps = np.zeros(candidate_rank, dtype=np.float64)
        n_rows = whitened_complex.shape[0]
        for _ in range(int(bootstrap_count)):
            indices = rng.integers(0, n_rows, size=n_rows)
            sampled = whitened_complex[indices]
            sampled_real = np.vstack((sampled.real, sampled.imag))
            _, boot_singular, boot_right_h = np.linalg.svd(
                sampled_real, full_matrices=False
            )
            boot_leading = float(boot_singular[0]) if boot_singular.size else 0.0
            boot_threshold = max(
                float(absolute_threshold), float(relative_threshold) * boot_leading
            )
            boot_rank = min(
                int(np.sum(boot_singular > boot_threshold)),
                int(max_directions),
            )
            if boot_rank == 0:
                continue
            boot_basis = boot_right_h[:boot_rank].T
            direction_overlap = np.sum(
                (boot_basis.T @ candidate_directions) ** 2, axis=0
            )
            overlaps += direction_overlap
            selected += direction_overlap >= 0.5
        bootstrap_frequency = selected / int(bootstrap_count)
        mean_projector_overlap = overlaps / int(bootstrap_count)

    # Bootstrap-unstable directions are disabled instead of silently entering
    # the optimiser.  With bootstrap_count=0 the deterministic SVD selection
    # is retained in full.
    stable = bootstrap_frequency >= float(minimum_bootstrap_frequency)
    whitened_directions = np.asarray(
        candidate_directions[:, stable], dtype=np.float64
    )
    bootstrap_frequency = np.asarray(
        bootstrap_frequency[stable], dtype=np.float64
    )
    mean_projector_overlap = np.asarray(
        mean_projector_overlap[stable], dtype=np.float64
    )
    selected_rank = int(np.sum(stable))
    physical_directions = np.asarray(
        physical_from_whitened @ whitened_directions, dtype=np.float64
    )

    return IdentifiableSubspace(
        singular_values=np.asarray(singular_values, dtype=np.float64),
        relative_singular_values=relative,
        numerical_rank=numerical_rank,
        selected_rank=selected_rank,
        whitened_directions=whitened_directions,
        physical_directions=physical_directions,
        bootstrap_selection_frequency=bootstrap_frequency,
        threshold=threshold,
        audit={
            "coordinate_system": "theta = diag(parameter_scales) @ prior_precision^(-1/2) @ z",
            "aggregation": "all supplied episode/frequency/fold/sensor/snapshot rows",
            "complex_to_real": "stack(real(J), imag(J))",
            "n_complex_rows": int(complex_design.shape[0]),
            "n_parameters": int(n_parameter),
            "min_requested_directions": int(min_directions),
            "max_requested_directions": int(max_directions),
            "candidate_rank_before_bootstrap_stability": int(candidate_rank),
            "minimum_bootstrap_frequency": float(minimum_bootstrap_frequency),
            "bootstrap_unstable_directions_disabled": int(
                candidate_rank - selected_rank
            ),
            "below_minimum_due_to_rank_deficiency": selected_rank < min(
                int(min_directions), n_parameter
            ),
            "prior_precision": precision.tolist(),
            "bootstrap_count": int(bootstrap_count),
            "bootstrap_mean_projector_overlap": mean_projector_overlap.tolist(),
            "truth_used": False,
        },
    )


select_prior_whitened_identifiable_subspace = (
    prior_whitened_identifiable_subspace
)


def update_trust_region(
    actual_reduction: float,
    predicted_reduction: float,
    *,
    radius: float,
    damping: float,
    step_norm: float,
    acceptance_threshold: float = 0.25,
    expansion_threshold: float = 0.75,
    minimum_radius: float = 1.0e-4,
    maximum_radius: float = 4.0,
    minimum_damping: float = 1.0e-8,
    maximum_damping: float = 1.0e8,
) -> TrustRegionDecision:
    """Apply the pre-registered LM ratio thresholds (0.25, 0.75)."""

    values = np.asarray(
        [actual_reduction, predicted_reduction, radius, damping, step_norm],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("trust-region inputs must be finite")
    if radius <= 0.0 or damping <= 0.0 or step_norm < 0.0:
        raise ValueError("radius/damping must be positive and step_norm non-negative")
    if not 0.0 < acceptance_threshold < expansion_threshold:
        raise ValueError("ratio thresholds must satisfy 0 < lower < upper")
    if not 0.0 < minimum_radius <= maximum_radius:
        raise ValueError("invalid radius bounds")
    if not 0.0 < minimum_damping <= maximum_damping:
        raise ValueError("invalid damping bounds")

    if predicted_reduction <= np.finfo(float).eps:
        return TrustRegionDecision(
            ratio=float("-inf"),
            accepted=False,
            radius=max(minimum_radius, 0.25 * radius),
            damping=min(maximum_damping, 4.0 * damping),
            reason="nonpositive_predicted_reduction",
        )
    ratio = float(actual_reduction / predicted_reduction)
    accepted = bool(actual_reduction > 0.0 and ratio >= acceptance_threshold)
    if not accepted:
        return TrustRegionDecision(
            ratio=ratio,
            accepted=False,
            radius=max(minimum_radius, 0.25 * radius),
            damping=min(maximum_damping, 4.0 * damping),
            reason="ratio_below_0.25",
        )
    if ratio > expansion_threshold:
        boundary_step = step_norm >= 0.8 * radius
        new_radius = min(maximum_radius, 2.0 * radius) if boundary_step else radius
        return TrustRegionDecision(
            ratio=ratio,
            accepted=True,
            radius=new_radius,
            damping=max(minimum_damping, 0.5 * damping),
            reason=(
                "ratio_above_0.75_expand" if boundary_step else "ratio_above_0.75"
            ),
        )
    return TrustRegionDecision(
        ratio=ratio,
        accepted=True,
        radius=radius,
        damping=damping,
        reason="ratio_between_0.25_and_0.75",
    )


trust_region_actual_predicted_ratio = update_trust_region


def _coerce_fold(
    raw: ElectrodeSplit | Sequence[ArrayLike] | Mapping[str, ArrayLike],
    n_electrode: int,
) -> ElectrodeSplit:
    if isinstance(raw, ElectrodeSplit):
        inner_raw, heldout_raw = raw.inner, raw.validation
    elif isinstance(raw, Mapping):
        inner_raw = raw.get("inner")
        heldout_raw = raw.get("validation", raw.get("heldout"))
        if inner_raw is None or heldout_raw is None:
            raise ValueError("fold mappings require inner and validation/heldout")
    else:
        if len(raw) != 2:
            raise ValueError("fold sequences must contain (inner, heldout)")
        inner_raw, heldout_raw = raw
    inner = np.asarray(inner_raw, dtype=np.int64).reshape(-1)
    heldout = np.asarray(heldout_raw, dtype=np.int64).reshape(-1)
    if inner.size == 0 or heldout.size == 0:
        raise ValueError("every fold needs non-empty inner and held-out electrodes")
    if len(np.unique(inner)) != inner.size or len(np.unique(heldout)) != heldout.size:
        raise ValueError("fold electrode indices must be unique")
    if np.intersect1d(inner, heldout).size:
        raise ValueError("inner and held-out electrodes must be disjoint")
    if np.any(inner < 0) or np.any(heldout < 0) or np.any(inner >= n_electrode) or np.any(heldout >= n_electrode):
        raise ValueError("fold electrode indices are out of range")
    complement = np.setdiff1d(np.arange(n_electrode), heldout, assume_unique=True)
    if not np.array_equal(np.sort(inner), complement):
        raise ValueError("each inner set must be the exact held-out complement")
    return ElectrodeSplit(inner=np.sort(inner), validation=np.sort(heldout))


def _validated_family(
    family: LeadfieldFamily, theta: np.ndarray, n_parameter: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(family, "leadfields") or not hasattr(family, "jacobians"):
        raise TypeError("reassemble(theta) must return a LeadfieldFamily-like object")
    operators = np.asarray(family.leadfields, dtype=np.complex128)
    jacobians = np.asarray(family.jacobians, dtype=np.complex128)
    if operators.ndim != 3 or jacobians.ndim != 4:
        raise ValueError("family requires G(F,E,P) and J(F,Q,E,P)")
    if jacobians.shape[0] != operators.shape[0] or jacobians.shape[2:] != operators.shape[1:]:
        raise ValueError("leadfields and analytic Jacobians do not align")
    if jacobians.shape[1] != theta.size:
        raise ValueError("theta and analytic Jacobian parameter counts do not align")
    if n_parameter is not None and jacobians.shape[1] != n_parameter:
        raise ValueError("reassembled family changed its parameter dimension")
    if not np.all(np.isfinite(operators)) or not np.all(np.isfinite(jacobians)):
        raise ValueError("reassembled leadfields/Jacobians contain non-finite values")
    names = tuple(getattr(family, "parameter_names", ()))
    if names and len(names) != theta.size:
        raise ValueError("family parameter names do not align with theta")
    return operators, jacobians


def _coerce_observations(
    observations: Sequence[ArrayLike] | ArrayLike,
    expected_frequency: int,
    expected_electrode: int,
) -> tuple[np.ndarray, ...]:
    # Do not eagerly coerce a sequence: calibration episodes are allowed to
    # have different snapshot counts, which is a ragged sequence under NumPy.
    if isinstance(observations, np.ndarray) and observations.ndim == 4:
        episodes = tuple(
            np.asarray(item, dtype=np.complex128) for item in observations
        )
    else:
        episodes = tuple(np.asarray(item, dtype=np.complex128) for item in observations)
    if len(episodes) < 6:
        raise ValueError("V2 nonlinear calibration requires at least six episodes")
    for episode in episodes:
        if episode.ndim != 3 or episode.shape[:2] != (
            expected_frequency,
            expected_electrode,
        ):
            raise ValueError("episodes must align as Y(F,E,T) with the family")
        if episode.shape[2] == 0 or not np.all(np.isfinite(episode)):
            raise ValueError("episodes require finite, non-empty snapshots")
    return episodes


def _fit_posterior(
    solver: Callable[[np.ndarray, np.ndarray], Any] | None,
    operator: np.ndarray,
    target: np.ndarray,
    *,
    group_size: int,
    max_iter: int,
) -> Any:
    if solver is None:
        power = max(float(np.mean(np.abs(target) ** 2)), 1.0e-10)
        fitted = ComplexGroupSBL(
            group_size=group_size,
            max_iter=max_iter,
            tol=1.0e-5,
        ).fit(operator, target, noise0=0.05 * power)
    else:
        fitted = solver(operator, target)
    for attribute in ("mean", "covariance", "noise"):
        if not hasattr(fitted, attribute):
            raise TypeError(f"posterior solver result is missing {attribute}")
    mean = np.asarray(fitted.mean, dtype=np.complex128)
    covariance = np.asarray(fitted.covariance, dtype=np.complex128)
    noise = float(fitted.noise)
    if mean.shape != (operator.shape[1], target.shape[1]):
        raise ValueError("posterior mean has the wrong shape")
    if covariance.shape != (operator.shape[1], operator.shape[1]):
        raise ValueError("posterior covariance has the wrong shape")
    if not np.isfinite(noise) or noise <= 0.0:
        raise ValueError("posterior noise must be finite and positive")
    return fitted


def _evaluate_family(
    operators: np.ndarray,
    episodes: tuple[np.ndarray, ...],
    folds: tuple[ElectrodeSplit, ...],
    theta: np.ndarray,
    prior_mean: np.ndarray,
    prior_hessian: np.ndarray,
    *,
    group_size: int,
    inner_max_iter: int,
    posterior_solver: Callable[[np.ndarray, np.ndarray], Any] | None,
) -> _FamilyEvaluation:
    posteriors: list[FoldFrequencyPosterior] = []
    fold_scores = np.empty(len(folds), dtype=np.float64)
    for fold_index, fold in enumerate(folds):
        fold_total = 0.0
        fold_count = 0
        for frequency_index, operator in enumerate(operators):
            target = np.concatenate(
                [episode[frequency_index] for episode in episodes], axis=1
            )
            a_inner = operator[fold.inner]
            y_inner = target[fold.inner]
            fitted = _fit_posterior(
                posterior_solver,
                a_inner,
                y_inner,
                group_size=group_size,
                max_iter=inner_max_iter,
            )
            a_validation = operator[fold.validation]
            y_validation = target[fold.validation]
            components = _predictive_components(
                a_validation,
                y_validation,
                fitted.mean,
                fitted.covariance,
                fitted.noise,
                include_posterior_covariance=True,
                jitter=1.0e-10,
            )
            fold_total += components.total
            fold_count += y_validation.size
            posteriors.append(
                FoldFrequencyPosterior(
                    fold_index=fold_index,
                    frequency_index=frequency_index,
                    inner_indices=np.asarray(fold.inner, dtype=np.int64),
                    heldout_indices=np.asarray(fold.validation, dtype=np.int64),
                    mean=np.asarray(fitted.mean, dtype=np.complex128),
                    covariance=np.asarray(fitted.covariance, dtype=np.complex128),
                    noise=float(fitted.noise),
                    predictive_nlpd=components.per_complex_observation,
                    predictive_covariance=components.covariance,
                    residual=components.residual,
                )
            )
        fold_scores[fold_index] = fold_total / max(fold_count, 1)
    predictive = float(np.mean(fold_scores))
    displacement = theta - prior_mean
    prior_penalty = float(0.5 * displacement @ prior_hessian @ displacement)
    return _FamilyEvaluation(
        objective=predictive + prior_penalty,
        predictive_nlpd=predictive,
        fold_nlpd=fold_scores,
        posteriors=tuple(posteriors),
    )


def _analytic_local_model(
    jacobians: np.ndarray,
    evaluation: _FamilyEvaluation,
    theta: np.ndarray,
    prior_mean: np.ndarray,
    prior_hessian: np.ndarray,
    n_fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen-posterior Gauss--Newton model from the current analytic J."""

    n_parameter = jacobians.shape[1]
    gradient = np.zeros(n_parameter, dtype=np.float64)
    hessian = np.zeros((n_parameter, n_parameter), dtype=np.float64)
    sensitivity_rows: list[np.ndarray] = []
    fold_counts = np.zeros(n_fold, dtype=np.int64)
    for posterior in evaluation.posteriors:
        fold_counts[posterior.fold_index] += (
            posterior.heldout_indices.size * posterior.mean.shape[1]
        )
    for posterior in evaluation.posteriors:
        # Index the frequency first so NumPy advanced indexing cannot move the
        # held-out electrode axis ahead of the parameter axis.
        derivative = jacobians[posterior.frequency_index][
            :, posterior.heldout_indices, :
        ]
        sensitivity = np.einsum(
            "qep,pt->qet", derivative, posterior.mean, optimize=True
        )
        factor = np.linalg.cholesky(posterior.predictive_covariance)
        residual_white = np.linalg.solve(factor, posterior.residual)
        sensitivity_white = np.empty_like(sensitivity)
        for parameter in range(n_parameter):
            sensitivity_white[parameter] = np.linalg.solve(
                factor, sensitivity[parameter]
            )
        denominator = max(int(fold_counts[posterior.fold_index]), 1) * n_fold
        gradient += (
            -2.0
            * np.real(
                np.einsum(
                    "qet,et->q",
                    np.conj(sensitivity_white),
                    residual_white,
                    optimize=True,
                )
            )
            / denominator
        )
        hessian += (
            2.0
            * np.real(
                np.einsum(
                    "qet,ret->qr",
                    np.conj(sensitivity_white),
                    sensitivity_white,
                    optimize=True,
                )
            )
            / denominator
        )
        sensitivity_rows.append(
            np.moveaxis(sensitivity_white, 0, -1).reshape(-1, n_parameter)
        )
    gradient += prior_hessian @ (theta - prior_mean)
    hessian += prior_hessian
    hessian = 0.5 * (hessian + hessian.T)
    return gradient, hessian, np.vstack(sensitivity_rows)


def calibrate_nonlinear_multi_episode(
    reassemble: Callable[[np.ndarray], LeadfieldFamily],
    calibration_observations: Sequence[ArrayLike] | ArrayLike,
    folds: Sequence[ElectrodeSplit | Sequence[ArrayLike] | Mapping[str, ArrayLike]],
    parameter_scales: ArrayLike,
    prior_precision: ArrayLike | None = None,
    *,
    theta0: ArrayLike | None = None,
    prior_mean: ArrayLike | None = None,
    group_size: int = 3,
    max_outer_iterations: int = 6,
    inner_max_iter: int = 30,
    initial_radius: float = 0.5,
    initial_damping: float = 1.0e-2,
    minimum_radius: float = 1.0e-4,
    maximum_radius: float = 4.0,
    min_active_directions: int = 3,
    max_active_directions: int = 8,
    svd_relative_threshold: float = 1.0e-6,
    bootstrap_count: int = 24,
    bootstrap_seed: int = 0,
    minimum_bootstrap_frequency: float = 0.60,
    posterior_solver: Callable[[np.ndarray, np.ndarray], SBLResult] | None = None,
) -> NonlinearCalibrationResult:
    """Jointly calibrate one physical ``theta`` from six or more episodes.

    Parameters are stepped in a prior-whitened SVD subspace.  The local model
    uses the analytic Jacobian from the current nonlinear assembly, while each
    trial is scored only after exact reassembly and a fresh lower posterior.
    Standard LM actual/predicted thresholds 0.25 and 0.75 decide rejection,
    radius expansion, and damping.  If no spatial fold improves its held-out
    predictive NLPD, the returned model is the uncalibrated ``theta=0`` model.
    """

    if not callable(reassemble):
        raise TypeError("reassemble must be callable")
    scales = np.asarray(parameter_scales, dtype=np.float64).reshape(-1)
    if scales.size == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite and strictly positive")
    n_parameter = scales.size
    theta = (
        np.zeros(n_parameter, dtype=np.float64)
        if theta0 is None
        else np.asarray(theta0, dtype=np.float64).reshape(-1)
    )
    mean_prior = (
        np.zeros(n_parameter, dtype=np.float64)
        if prior_mean is None
        else np.asarray(prior_mean, dtype=np.float64).reshape(-1)
    )
    if theta.shape != (n_parameter,) or mean_prior.shape != (n_parameter,):
        raise ValueError("theta0/prior_mean must align with parameter_scales")
    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(mean_prior)):
        raise ValueError("theta0/prior_mean contain non-finite values")
    if int(group_size) <= 0 or int(max_outer_iterations) < 0 or int(inner_max_iter) <= 0:
        raise ValueError("group_size/iteration counts are invalid")
    if initial_radius <= 0.0 or initial_damping <= 0.0:
        raise ValueError("initial trust radius and damping must be positive")

    precision, prior_inverse_sqrt = _precision_and_inverse_sqrt(
        prior_precision, n_parameter
    )
    inverse_scales = np.diag(1.0 / scales)
    prior_hessian = inverse_scales @ precision @ inverse_scales

    reassembly_evaluations = 0

    def assemble(candidate: np.ndarray) -> tuple[LeadfieldFamily, np.ndarray, np.ndarray]:
        nonlocal reassembly_evaluations
        family = reassemble(np.array(candidate, dtype=np.float64, copy=True))
        reassembly_evaluations += 1
        operators, jacobians = _validated_family(
            family, candidate, n_parameter=n_parameter
        )
        return family, operators, jacobians

    current_family, current_operators, current_jacobians = assemble(theta)
    n_frequency, n_electrode, n_coefficient = current_operators.shape
    if n_coefficient % int(group_size):
        raise ValueError("leadfield columns are not aligned with group_size")
    episodes = _coerce_observations(
        calibration_observations, n_frequency, n_electrode
    )
    if len(folds) != 3:
        raise ValueError("V2 nonlinear calibration requires exactly three spatial folds")
    splits = tuple(_coerce_fold(fold, n_electrode) for fold in folds)
    sizes = np.asarray([split.validation.size for split in splits], dtype=int)
    if int(sizes.max() - sizes.min()) > 1:
        raise ValueError("held-out folds are not size-balanced")
    heldout_sets = [set(split.validation.tolist()) for split in splits]
    if any(heldout_sets[left] & heldout_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("the three held-out folds must form disjoint spatial blocks")

    current_evaluation = _evaluate_family(
        current_operators,
        episodes,
        splits,
        theta,
        mean_prior,
        prior_hessian,
        group_size=int(group_size),
        inner_max_iter=int(inner_max_iter),
        posterior_solver=posterior_solver,
    )
    initial_family = current_family
    initial_operators = current_operators
    initial_jacobians = current_jacobians
    initial_evaluation = current_evaluation
    objective_history = [current_evaluation.objective]
    predictive_history = [current_evaluation.predictive_nlpd]
    trust_history: list[dict[str, Any]] = []
    radius = float(initial_radius)
    damping = float(initial_damping)
    accepted_steps = 0
    last_subspace: IdentifiableSubspace | None = None

    for iteration in range(int(max_outer_iterations)):
        gradient, hessian, sensitivities = _analytic_local_model(
            current_jacobians,
            current_evaluation,
            theta,
            mean_prior,
            prior_hessian,
            len(splits),
        )
        last_subspace = prior_whitened_identifiable_subspace(
            sensitivities,
            scales,
            prior_precision=precision,
            min_directions=int(min_active_directions),
            max_directions=int(max_active_directions),
            relative_threshold=float(svd_relative_threshold),
            bootstrap_count=int(bootstrap_count),
            bootstrap_seed=int(bootstrap_seed) + iteration,
            minimum_bootstrap_frequency=float(minimum_bootstrap_frequency),
        )
        directions = last_subspace.physical_directions
        if last_subspace.selected_rank == 0:
            trust_history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "no_identifiable_direction",
                    "radius": radius,
                    "damping": damping,
                    "jacobian_norm": float(np.linalg.norm(current_jacobians)),
                }
            )
            break
        reduced_gradient = directions.T @ gradient
        reduced_hessian = directions.T @ hessian @ directions
        reduced_hessian = 0.5 * (reduced_hessian + reduced_hessian.T)
        lm_hessian = reduced_hessian + damping * np.eye(
            last_subspace.selected_rank
        )
        try:
            step = -np.linalg.solve(lm_hessian, reduced_gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(lm_hessian, rcond=1.0e-12) @ reduced_gradient
        step_norm = float(np.linalg.norm(step))
        if step_norm > radius:
            step = step * (radius / step_norm)
            step_norm = radius
        predicted_reduction = float(
            -(reduced_gradient @ step + 0.5 * step @ reduced_hessian @ step)
        )
        if step_norm <= 1.0e-10:
            trust_history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "step_below_tolerance",
                    "radius": radius,
                    "damping": damping,
                    "predicted_reduction": predicted_reduction,
                    "jacobian_norm": float(np.linalg.norm(current_jacobians)),
                }
            )
            break

        trial_theta = theta + directions @ step
        trial_family, trial_operators, trial_jacobians = assemble(trial_theta)
        trial_evaluation = _evaluate_family(
            trial_operators,
            episodes,
            splits,
            trial_theta,
            mean_prior,
            prior_hessian,
            group_size=int(group_size),
            inner_max_iter=int(inner_max_iter),
            posterior_solver=posterior_solver,
        )
        actual_reduction = float(
            current_evaluation.objective - trial_evaluation.objective
        )
        decision = update_trust_region(
            actual_reduction,
            predicted_reduction,
            radius=radius,
            damping=damping,
            step_norm=step_norm,
            minimum_radius=float(minimum_radius),
            maximum_radius=float(maximum_radius),
        )
        trust_history.append(
            {
                "iteration": iteration,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "actual_reduction": actual_reduction,
                "predicted_reduction": predicted_reduction,
                "ratio": decision.ratio,
                "step_norm_prior_whitened": step_norm,
                "radius_before": radius,
                "radius_after": decision.radius,
                "damping_before": damping,
                "damping_after": decision.damping,
                "theta_trial": trial_theta.tolist(),
                "trial_jacobian_norm": float(np.linalg.norm(trial_jacobians)),
                "posterior_refit_before_acceptance": True,
                "nonlinear_reassembly_before_acceptance": True,
            }
        )
        radius, damping = decision.radius, decision.damping
        if decision.accepted:
            theta = trial_theta
            current_family = trial_family
            current_operators = trial_operators
            current_jacobians = trial_jacobians
            current_evaluation = trial_evaluation
            accepted_steps += 1
            objective_history.append(current_evaluation.objective)
            predictive_history.append(current_evaluation.predictive_nlpd)
        if radius <= minimum_radius and not decision.accepted:
            break

    if last_subspace is None:
        _, _, initial_sensitivities = _analytic_local_model(
            initial_jacobians,
            initial_evaluation,
            theta,
            mean_prior,
            prior_hessian,
            len(splits),
        )
        last_subspace = prior_whitened_identifiable_subspace(
            initial_sensitivities,
            scales,
            prior_precision=precision,
            min_directions=int(min_active_directions),
            max_directions=int(max_active_directions),
            relative_threshold=float(svd_relative_threshold),
            bootstrap_count=int(bootstrap_count),
            bootstrap_seed=int(bootstrap_seed),
            minimum_bootstrap_frequency=float(minimum_bootstrap_frequency),
        )

    fold_improved = current_evaluation.fold_nlpd < (
        initial_evaluation.fold_nlpd - 1.0e-10
    )
    fallback = not bool(np.any(fold_improved))
    if fallback:
        zero = np.zeros(n_parameter, dtype=np.float64)
        if np.array_equal(np.asarray(theta0 if theta0 is not None else zero), zero):
            theta = zero
            current_family = initial_family
            current_operators = initial_operators
            current_jacobians = initial_jacobians
            current_evaluation = initial_evaluation
        else:
            theta = zero
            current_family, current_operators, current_jacobians = assemble(theta)
            current_evaluation = _evaluate_family(
                current_operators,
                episodes,
                splits,
                theta,
                mean_prior,
                prior_hessian,
                group_size=int(group_size),
                inner_max_iter=int(inner_max_iter),
                posterior_solver=posterior_solver,
            )
        trust_history.append(
            {
                "iteration": len(trust_history),
                "accepted": False,
                "reason": "all_folds_failed_to_improve_fallback_theta_zero",
                "theta_returned": theta.tolist(),
            }
        )

    normalized_theta = np.linalg.solve(prior_inverse_sqrt, theta / scales)
    return NonlinearCalibrationResult(
        theta=np.asarray(theta, dtype=np.float64),
        normalized_theta=np.asarray(normalized_theta, dtype=np.float64),
        leadfield_family=current_family,
        fold_predictive_nlpd=np.asarray(
            current_evaluation.fold_nlpd, dtype=np.float64
        ),
        objective_history=tuple(float(value) for value in objective_history),
        predictive_nlpd_history=tuple(float(value) for value in predictive_history),
        trust_region_history=tuple(trust_history),
        identifiable_subspace=last_subspace,
        posteriors=current_evaluation.posteriors,
        accepted_steps=accepted_steps,
        fallback_to_zero=fallback,
        metadata={
            "solver": "ComplexGroupSBL/Champagne proper-complex EM",
            "shared_theta_across_episodes": True,
            "episode_count": len(episodes),
            "spatial_fold_count": len(splits),
            "folds_jointly_optimized": True,
            "heldout_nlpd_includes_source_posterior_covariance": True,
            "nonlinear_reassembly_at_every_trial": True,
            "analytic_jacobian_from_every_trial_assembly": True,
            "actual_predicted_ratio_thresholds": [0.25, 0.75],
            "reassembly_evaluations": reassembly_evaluations,
            "accepted_steps": accepted_steps,
            "fold_improved_before_fallback": fold_improved.tolist(),
            "fallback_rule": "theta=0 when no fold improves predictive NLPD",
            "truth_used": False,
        },
    )


# Natural-language-order alias retained for experiment configuration files.
nonlinear_multi_episode_calibration = calibrate_nonlinear_multi_episode


__all__ = [
    "FoldFrequencyPosterior",
    "IdentifiableSubspace",
    "NonlinearCalibrationResult",
    "PredictiveNLPDComponents",
    "TrustRegionDecision",
    "calibrate_nonlinear_multi_episode",
    "heldout_proper_complex_nlpd",
    "nonlinear_multi_episode_calibration",
    "prior_whitened_identifiable_subspace",
    "proper_complex_predictive_nlpd",
    "select_prior_whitened_identifiable_subspace",
    "trust_region_actual_predicted_ratio",
    "update_trust_region",
]
