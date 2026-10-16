"""Auditable V4 multi-frequency complex group-SBL primitives.

Each frequency has its own latent complex source matrix and noise variance::

    Y_f = G_f X_f + E_f,
    X_f[:, t] ~ CN(0, Gamma),
    E_f[:, t] ~ CN(0, lambda_f I).

Only the spatial group variances ``gamma`` are shared.  In particular, this
module never stacks frequency operators in a way that would impose a common
source coefficient matrix.  Fast-Safe-FreeSIGM enters through
``omega_f = F_available * w_f`` when sufficient statistics and the Type-II
objective are combined.  Uniform weights therefore give ``omega_f == 1``.

The routines use the repository's validated proper-complex posterior and
sensor-space Cholesky implementation.  They accept no truth-valued input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .complex_sbl import (
    _cho_factor,
)


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _as_frequency_problem(
    leadfields: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int,
) -> tuple[ComplexArray, ComplexArray, int]:
    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if operators.ndim != 3:
        raise ValueError(
            "leadfields must have shape (frequency, sensor, coefficient)"
        )
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or data.shape[:2] != operators.shape[:2]:
        raise ValueError(
            "observations must have shape (frequency, sensor, snapshot)"
        )
    if operators.shape[0] < 1 or operators.shape[1] < 1 or operators.shape[2] < 1:
        raise ValueError("the multi-frequency problem must be nonempty")
    size = int(group_size)
    if size < 1 or operators.shape[2] % size:
        raise ValueError("coefficient count must divide into positive source groups")
    if not np.all(np.isfinite(operators)) or not np.all(np.isfinite(data)):
        raise ValueError("leadfields and observations must be finite")
    return operators, data, operators.shape[2] // size


def effective_frequency_weights(
    frequency_weights: ArrayLike | None,
    n_frequencies: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return probability weights, ``omega``, and the available mask.

    Zero entries are permitted for unavailable frequencies.  The positive
    entries are normalized on their own simplex and their effective weights
    sum to ``F_available``.  Rescaling all supplied positive weights therefore
    changes no result, and uniform ``ones`` and uniform ``ones/F`` are exactly
    equivalent.
    """

    count = int(n_frequencies)
    if count < 1:
        raise ValueError("n_frequencies must be positive")
    raw = (
        np.ones(count, dtype=np.float64)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=np.float64)
    )
    if (
        raw.shape != (count,)
        or not np.all(np.isfinite(raw))
        or np.any(raw < 0.0)
        or not np.any(raw > 0.0)
    ):
        raise ValueError(
            "frequency_weights must be a finite nonnegative vector with positive mass"
        )
    available = raw > 0.0
    probability = np.zeros(count, dtype=np.float64)
    probability[available] = raw[available] / float(np.sum(raw[available]))
    omega = np.zeros(count, dtype=np.float64)
    omega[available] = int(np.count_nonzero(available)) * probability[available]
    return probability, omega, available


def _noise_vector(
    noise: ArrayLike | float,
    n_frequencies: int,
    *,
    floor: float = 0.0,
) -> FloatArray:
    raw = np.asarray(noise, dtype=np.float64)
    if raw.ndim == 0:
        values = np.full(n_frequencies, float(raw), dtype=np.float64)
    else:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if (
        values.shape != (n_frequencies,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("noise must contain one positive finite variance per frequency")
    return np.maximum(values, float(floor)).astype(np.float64)


def _canonical_frequency_order(
    operators: ComplexArray,
    data: ComplexArray,
) -> NDArray[np.int64]:
    """Return a stable content-based order for permutation-equivariant fitting."""

    keys: list[tuple[str, int]] = []
    for index in range(operators.shape[0]):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(operators[index]).view(np.uint8).tobytes())
        digest.update(np.ascontiguousarray(data[index]).view(np.uint8).tobytes())
        keys.append((digest.hexdigest(), index))
    return np.asarray([index for _, index in sorted(keys)], dtype=np.int64)


def _reorder_posterior(
    posterior: "MultiFrequencyPosterior",
    order: NDArray[np.int64],
) -> "MultiFrequencyPosterior":
    factors = np.asarray(posterior.cholesky_factors[order], dtype=np.complex128)
    digest = hashlib.sha256(np.ascontiguousarray(factors).tobytes()).hexdigest()
    return MultiFrequencyPosterior(
        means=posterior.means[order],
        covariances=posterior.covariances[order],
        sensor_covariances=posterior.sensor_covariances[order],
        cholesky_factors=factors,
        log_determinants=posterior.log_determinants[order],
        factorization_token=digest,
    )


@dataclass(frozen=True, slots=True)
class MultiFrequencyPosterior:
    """Per-frequency posterior state at one shared ``gamma``."""

    means: ComplexArray
    covariances: ComplexArray
    sensor_covariances: ComplexArray
    cholesky_factors: ComplexArray
    log_determinants: FloatArray
    factorization_token: str

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=np.complex128)
        covariance = np.asarray(self.covariances, dtype=np.complex128)
        sensor = np.asarray(self.sensor_covariances, dtype=np.complex128)
        factors = np.asarray(self.cholesky_factors, dtype=np.complex128)
        logdet = np.asarray(self.log_determinants, dtype=np.float64)
        if (
            means.ndim != 3
            or covariance.shape
            != (means.shape[0], means.shape[1], means.shape[1])
            or sensor.ndim != 3
            or sensor.shape[0] != means.shape[0]
            or sensor.shape[1] != sensor.shape[2]
            or factors.shape != sensor.shape
            or logdet.shape != (means.shape[0],)
            or not all(
                np.all(np.isfinite(value))
                for value in (means, covariance, sensor, factors, logdet)
            )
        ):
            raise ValueError("multi-frequency posterior/cache arrays are invalid")
        if (
            not isinstance(self.factorization_token, str)
            or len(self.factorization_token) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.factorization_token
            )
        ):
            raise ValueError("posterior factorization token must be SHA-256")
        for name, value in (
            ("means", means),
            ("covariances", covariance),
            ("sensor_covariances", sensor),
            ("cholesky_factors", factors),
            ("log_determinants", logdet),
        ):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def mean(self) -> ComplexArray:
        return self.means

    @property
    def covariance(self) -> ComplexArray:
        return self.covariances


def multifrequency_posterior(
    leadfields: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    noise: ArrayLike | float,
    *,
    group_size: int = 3,
) -> MultiFrequencyPosterior:
    """Compute independent ``(M_f, V_f)`` pairs at shared group ``gamma``."""

    operators, data, groups = _as_frequency_problem(
        leadfields, observations, group_size=group_size
    )
    values = np.asarray(gamma, dtype=np.float64)
    if (
        values.shape != (groups,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("gamma must contain one positive finite value per group")
    noises = _noise_vector(noise, operators.shape[0])
    means: list[ComplexArray] = []
    covariances: list[ComplexArray] = []
    sensor_covariances: list[ComplexArray] = []
    factors: list[ComplexArray] = []
    log_determinants: list[float] = []
    digest = hashlib.sha256()
    gamma_diagonal = np.repeat(values, int(group_size))
    for frequency in range(operators.shape[0]):
        operator = operators[frequency]
        weighted_operator = operator * gamma_diagonal[None, :]
        sensor_covariance = 0.5 * (
            weighted_operator @ operator.conj().T
            + (weighted_operator @ operator.conj().T).conj().T
        )
        sensor_covariance += noises[frequency] * np.eye(
            operator.shape[0], dtype=np.complex128
        )
        packed, lower = _cho_factor(sensor_covariance)
        if lower is not True:
            raise RuntimeError("V4 posterior requires lower Cholesky factors")
        factor = np.tril(np.asarray(packed, dtype=np.complex128))
        solved_y = linalg.cho_solve(
            (factor, True), data[frequency], check_finite=False
        )
        solved_operator = linalg.cho_solve(
            (factor, True), operator, check_finite=False
        )
        mean = gamma_diagonal[:, None] * (operator.conj().T @ solved_y)
        covariance = np.diag(gamma_diagonal) - (
            gamma_diagonal[:, None]
            * (operator.conj().T @ solved_operator)
            * gamma_diagonal[None, :]
        )
        covariance = 0.5 * (covariance + covariance.conj().T)
        means.append(np.asarray(mean, dtype=np.complex128))
        covariances.append(np.asarray(covariance, dtype=np.complex128))
        sensor_covariances.append(
            np.asarray(sensor_covariance, dtype=np.complex128)
        )
        factors.append(factor)
        log_determinants.append(
            2.0
            * float(
                np.log(
                    np.maximum(np.diag(factor).real, np.finfo(float).tiny)
                ).sum()
            )
        )
        digest.update(np.ascontiguousarray(factor).tobytes())
    return MultiFrequencyPosterior(
        means=np.stack(means, axis=0),
        covariances=np.stack(covariances, axis=0),
        sensor_covariances=np.stack(sensor_covariances, axis=0),
        cholesky_factors=np.stack(factors, axis=0),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        factorization_token=digest.hexdigest(),
    )


def multifrequency_type2_terms_from_posterior(
    observations: ArrayLike,
    posterior: MultiFrequencyPosterior,
) -> FloatArray:
    """Return total proper-complex NLL per frequency without refactorization."""

    data = np.asarray(observations, dtype=np.complex128)
    if data.ndim == 2:
        data = data[:, :, None]
    if (
        data.ndim != 3
        or data.shape[0] != posterior.means.shape[0]
        or data.shape[1] != posterior.sensor_covariances.shape[1]
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("observations do not align with the cached posterior")
    values = np.empty(data.shape[0], dtype=np.float64)
    for frequency in range(data.shape[0]):
        solved = linalg.cho_solve(
            (posterior.cholesky_factors[frequency], True),
            data[frequency],
            check_finite=False,
        )
        quadratic = float(np.vdot(data[frequency], solved).real)
        values[frequency] = (
            data.shape[2] * posterior.log_determinants[frequency] + quadratic
        )
    return values


def multifrequency_type2_objective(
    leadfields: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    noise: ArrayLike | float,
    *,
    frequency_weights: ArrayLike | None = None,
    omega: ArrayLike | None = None,
    group_size: int = 3,
) -> float:
    """Weighted sum of proper-complex per-frequency Type-II objectives.

    Pass either probability-like ``frequency_weights`` or already effective
    ``omega`` values.  When ``omega`` is supplied it is used verbatim (up to
    validation); this is useful for dense-reference tests and cached callers.
    """

    operators, data, _ = _as_frequency_problem(
        leadfields, observations, group_size=group_size
    )
    noises = _noise_vector(noise, operators.shape[0])
    if omega is not None and frequency_weights is not None:
        raise ValueError("pass frequency_weights or omega, not both")
    if omega is None:
        _, effective, _ = effective_frequency_weights(
            frequency_weights, operators.shape[0]
        )
    else:
        effective = np.asarray(omega, dtype=np.float64)
        if (
            effective.shape != (operators.shape[0],)
            or not np.all(np.isfinite(effective))
            or np.any(effective < 0.0)
            or not np.any(effective > 0.0)
        ):
            raise ValueError("omega must be finite, nonnegative, and nonempty")
    state = multifrequency_posterior(
        operators, data, gamma, noises, group_size=group_size
    )
    terms = multifrequency_type2_terms_from_posterior(data, state)
    return float(np.dot(effective, terms))


def multifrequency_group_em_update(
    posterior_means: ArrayLike,
    posterior_covariances: ArrayLike,
    omega: ArrayLike,
    *,
    group_size: int = 3,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Weighted grouped EM update from per-frequency second moments."""

    means = np.asarray(posterior_means, dtype=np.complex128)
    covariances = np.asarray(posterior_covariances, dtype=np.complex128)
    if means.ndim != 3:
        raise ValueError("posterior_means must have shape (frequency,coefficient,snapshot)")
    frequencies, coefficients, snapshots = means.shape
    size = int(group_size)
    if size < 1 or coefficients % size or snapshots < 1:
        raise ValueError("posterior means do not contain valid source groups/snapshots")
    if covariances.shape != (frequencies, coefficients, coefficients):
        raise ValueError(
            "posterior_covariances must have shape (frequency,coefficient,coefficient)"
        )
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
        raise ValueError("posterior moments must be finite")
    effective = np.asarray(omega, dtype=np.float64)
    if (
        effective.shape != (frequencies,)
        or not np.all(np.isfinite(effective))
        or np.any(effective < 0.0)
        or float(np.sum(effective)) <= 0.0
    ):
        raise ValueError("omega must be finite, nonnegative, and nonempty")
    minimum = float(floor)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("floor must be finite and positive")
    groups = coefficients // size
    update = np.zeros(groups, dtype=np.float64)
    for frequency in range(frequencies):
        if effective[frequency] == 0.0:
            continue
        for group in range(groups):
            sl = slice(group * size, (group + 1) * size)
            second = (
                covariances[frequency, sl, sl]
                + means[frequency, sl]
                @ means[frequency, sl].conj().T
                / snapshots
            )
            update[group] += float(effective[frequency]) * float(
                np.trace(second).real / size
            )
    update /= float(np.sum(effective))
    return np.maximum(update, minimum).astype(np.float64)


def multifrequency_group_mu_update(
    leadfields: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    noise: ArrayLike | float,
    omega: ArrayLike,
    *,
    group_size: int = 3,
    floor: float = 1.0e-12,
    posterior: MultiFrequencyPosterior | None = None,
) -> FloatArray:
    """Weighted grouped MacKay/MU proposal using per-frequency solves."""

    operators, data, groups = _as_frequency_problem(
        leadfields, observations, group_size=group_size
    )
    values = np.asarray(gamma, dtype=np.float64)
    if (
        values.shape != (groups,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("gamma must contain one positive finite value per group")
    noises = _noise_vector(noise, operators.shape[0])
    effective = np.asarray(omega, dtype=np.float64)
    if (
        effective.shape != (operators.shape[0],)
        or not np.all(np.isfinite(effective))
        or np.any(effective < 0.0)
        or float(np.sum(effective)) <= 0.0
    ):
        raise ValueError("omega must be finite, nonnegative, and nonempty")
    minimum = float(floor)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("floor must be finite and positive")
    size = int(group_size)
    q_total = np.zeros(groups, dtype=np.float64)
    leverage_total = np.zeros(groups, dtype=np.float64)
    state = (
        multifrequency_posterior(
            operators, data, values, noises, group_size=size
        )
        if posterior is None
        else posterior
    )
    if (
        state.means.shape != (operators.shape[0], operators.shape[2], data.shape[2])
        or state.sensor_covariances.shape[:2]
        != (operators.shape[0], operators.shape[1])
    ):
        raise ValueError("cached posterior does not align with the MU update")
    for frequency in range(operators.shape[0]):
        weight = float(effective[frequency])
        if weight == 0.0:
            continue
        covariance_y = state.sensor_covariances[frequency]
        factor = linalg.cho_factor(
            (covariance_y + covariance_y.conj().T) * 0.5,
            lower=True,
            check_finite=False,
        )
        solved_y = linalg.cho_solve(factor, data[frequency], check_finite=False)
        solved_a = linalg.cho_solve(
            factor, operators[frequency], check_finite=False
        )
        for group in range(groups):
            sl = slice(group * size, (group + 1) * size)
            score = operators[frequency, :, sl].conj().T @ solved_y
            q_total[group] += weight * float(
                np.linalg.norm(score, "fro") ** 2
                / (size * data.shape[2])
            )
            leverage_total[group] += weight * float(
                np.trace(
                    operators[frequency, :, sl].conj().T @ solved_a[:, sl]
                ).real
                / size
            )
    proposal = np.sqrt(
        np.maximum(q_total, minimum) / np.maximum(leverage_total, minimum)
    )
    return np.maximum(proposal, minimum).astype(np.float64)


def multifrequency_noise_update(
    leadfields: ArrayLike,
    observations: ArrayLike,
    posterior_means: ArrayLike,
    posterior_covariances: ArrayLike,
    *,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Return one analytic isotropic noise-variance update per frequency."""

    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if data.ndim == 2:
        data = data[:, :, None]
    means = np.asarray(posterior_means, dtype=np.complex128)
    covariances = np.asarray(posterior_covariances, dtype=np.complex128)
    if operators.ndim != 3 or data.ndim != 3 or data.shape[:2] != operators.shape[:2]:
        raise ValueError("leadfields and observations are incompatible")
    if means.shape != (operators.shape[0], operators.shape[2], data.shape[2]):
        raise ValueError("posterior means do not align with the frequency problem")
    if covariances.shape != (
        operators.shape[0],
        operators.shape[2],
        operators.shape[2],
    ):
        raise ValueError("posterior covariances do not align with the frequency problem")
    minimum = float(floor)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("floor must be finite and positive")
    update = np.empty(operators.shape[0], dtype=np.float64)
    for frequency in range(operators.shape[0]):
        residual = data[frequency] - operators[frequency] @ means[frequency]
        uncertainty = float(
            np.trace(
                operators[frequency]
                @ covariances[frequency]
                @ operators[frequency].conj().T
            ).real
        )
        value = (
            float(np.linalg.norm(residual, "fro") ** 2)
            + data.shape[2] * max(uncertainty, 0.0)
        ) / (operators.shape[1] * data.shape[2])
        update[frequency] = max(float(value), minimum)
    return update


def log_geometric_mix(
    first: ArrayLike,
    second: ArrayLike,
    fraction: float,
    *,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Positive elementwise log-geometric interpolation."""

    one = np.maximum(np.asarray(first, dtype=np.float64), float(floor))
    two = np.maximum(np.asarray(second, dtype=np.float64), float(floor))
    amount = float(fraction)
    if one.shape != two.shape or not 0.0 <= amount <= 1.0:
        raise ValueError("arrays must align and fraction must lie in [0,1]")
    if not np.all(np.isfinite(one)) or not np.all(np.isfinite(two)):
        raise ValueError("log-geometric inputs must be finite")
    return np.exp((1.0 - amount) * np.log(one) + amount * np.log(two))


def log_backtrack_multifrequency(
    current_gamma: ArrayLike,
    current_noise: ArrayLike,
    proposed_gamma: ArrayLike,
    proposed_noise: ArrayLike,
    objective_limit: float,
    evaluator,
    *,
    fractions: Sequence[float] = (1.0, 0.5, 0.25),
    tolerance: float = 2.0e-10,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Joint log-space backtracking for shared gamma and frequency noise."""

    gamma = np.asarray(current_gamma, dtype=np.float64)
    noise = np.asarray(current_noise, dtype=np.float64)
    gamma_proposal = np.asarray(proposed_gamma, dtype=np.float64)
    noise_proposal = np.asarray(proposed_noise, dtype=np.float64)
    if gamma.shape != gamma_proposal.shape or noise.shape != noise_proposal.shape:
        raise ValueError("current and proposed states must align")
    if any(
        np.any(value <= 0.0) or not np.all(np.isfinite(value))
        for value in (gamma, noise, gamma_proposal, noise_proposal)
    ):
        raise ValueError("backtracking variances must be finite and positive")
    limit = float(objective_limit)
    allowed = float(tolerance) * max(1.0, abs(limit))
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("backtracking fractions must lie in (0,1]")
        candidate_gamma = log_geometric_mix(gamma, gamma_proposal, fraction)
        candidate_noise = log_geometric_mix(noise, noise_proposal, fraction)
        value = float(evaluator(candidate_gamma, candidate_noise))
        if np.isfinite(value) and value <= limit + allowed:
            return candidate_gamma, candidate_noise, fraction, value
    return gamma.copy(), noise.copy(), 0.0, limit


@dataclass(frozen=True, slots=True)
class MultiFrequencySBLConfig:
    iterations: int = 40
    group_size: int = 3
    update_rule: Literal["em", "em_mu"] = "em"
    em_fraction: float = 0.65
    gamma_floor: float = 1.0e-12
    noise_floor: float = 1.0e-12
    noise_damping: float = 1.0
    objective_tolerance: float = 2.0e-10
    backtracking_fractions: tuple[float, ...] = (1.0, 0.5, 0.25)

    def __post_init__(self) -> None:
        if int(self.iterations) < 0:
            raise ValueError("iterations must be nonnegative")
        if int(self.group_size) < 1:
            raise ValueError("group_size must be positive")
        if self.update_rule not in {"em", "em_mu"}:
            raise ValueError("update_rule must be em or em_mu")
        if not 0.0 <= float(self.em_fraction) <= 1.0:
            raise ValueError("em_fraction must lie in [0,1]")
        if self.gamma_floor <= 0.0 or self.noise_floor <= 0.0:
            raise ValueError("variance floors must be positive")
        if not 0.0 <= float(self.noise_damping) <= 1.0:
            raise ValueError("noise_damping must lie in [0,1]")
        if not self.backtracking_fractions:
            raise ValueError("at least one backtracking fraction is required")


@dataclass(frozen=True, slots=True)
class MultiFrequencySBLResult:
    means: ComplexArray
    covariances: ComplexArray
    gamma: FloatArray
    noise: FloatArray
    frequency_weights: FloatArray
    omega: FloatArray
    objective: tuple[float, ...]
    accepted_scales: tuple[float, ...]
    endpoint_posterior: MultiFrequencyPosterior
    metadata: Mapping[str, object]

    @property
    def mean(self) -> ComplexArray:
        return self.means

    @property
    def covariance(self) -> ComplexArray:
        return self.covariances


def fit_multifrequency_group_sbl(
    leadfields: ArrayLike,
    observations: ArrayLike,
    frequency_weights: ArrayLike | None = None,
    *,
    config: MultiFrequencySBLConfig | None = None,
    gamma0: ArrayLike | None = None,
    noise0: ArrayLike | float | None = None,
    initial_posterior: MultiFrequencyPosterior | None = None,
) -> MultiFrequencySBLResult:
    """Fit a fixed-iteration independent-source/shared-gamma SBL model."""

    settings = config or MultiFrequencySBLConfig()
    operators, data, groups = _as_frequency_problem(
        leadfields, observations, group_size=settings.group_size
    )
    canonical_order = _canonical_frequency_order(operators, data)
    restore_order = np.argsort(canonical_order)
    operators = operators[canonical_order]
    data = data[canonical_order]
    if frequency_weights is not None:
        frequency_weights = np.asarray(frequency_weights, dtype=np.float64).reshape(-1)[
            canonical_order
        ]
    if noise0 is not None:
        raw_noise0 = np.asarray(noise0)
        if raw_noise0.ndim != 0:
            noise0 = np.asarray(raw_noise0, dtype=np.float64).reshape(-1)[canonical_order]
    if initial_posterior is not None:
        initial_posterior = _reorder_posterior(initial_posterior, canonical_order)
    weights, omega, available = effective_frequency_weights(
        frequency_weights, operators.shape[0]
    )
    frequency_power = np.mean(np.abs(data) ** 2, axis=(1, 2)).real
    shared_power = max(
        float(np.dot(omega, frequency_power) / np.sum(omega)),
        settings.gamma_floor,
    )
    gamma = (
        np.full(groups, shared_power, dtype=np.float64)
        if gamma0 is None
        else np.maximum(np.asarray(gamma0, dtype=np.float64), settings.gamma_floor)
    )
    if gamma.shape != (groups,) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma0 must contain one finite value per source group")
    noise = (
        np.maximum(0.1 * frequency_power, settings.noise_floor)
        if noise0 is None
        else _noise_vector(
            noise0,
            operators.shape[0],
            floor=settings.noise_floor,
        )
    )

    evaluated: list[
        tuple[FloatArray, FloatArray, float, MultiFrequencyPosterior]
    ] = []

    def objective_from_state(state: MultiFrequencyPosterior) -> float:
        return float(
            np.dot(omega, multifrequency_type2_terms_from_posterior(data, state))
        )

    def evaluate(candidate_gamma: FloatArray, candidate_noise: FloatArray) -> float:
        state = multifrequency_posterior(
            operators,
            data,
            candidate_gamma,
            candidate_noise,
            group_size=settings.group_size,
        )
        value = objective_from_state(state)
        evaluated.append(
            (
                np.asarray(candidate_gamma, dtype=np.float64).copy(),
                np.asarray(candidate_noise, dtype=np.float64).copy(),
                value,
                state,
            )
        )
        return value

    if initial_posterior is None:
        initial_value = evaluate(gamma, noise)
        current_state = evaluated[-1][3]
    else:
        current_state = initial_posterior
        expected_sensor = np.empty_like(current_state.sensor_covariances)
        gamma_diagonal = np.repeat(gamma, int(settings.group_size))
        for frequency, operator in enumerate(operators):
            expected_sensor[frequency] = (
                (operator * gamma_diagonal[None, :]) @ operator.conj().T
                + noise[frequency]
                * np.eye(operator.shape[0], dtype=np.complex128)
            )
        if (
            current_state.means.shape
            != (operators.shape[0], operators.shape[2], data.shape[2])
            or not np.allclose(
                current_state.sensor_covariances,
                expected_sensor,
                rtol=2.0e-11,
                atol=2.0e-12,
            )
        ):
            raise ValueError(
                "initial_posterior does not match gamma0/noise0 and the problem"
            )
        initial_value = objective_from_state(current_state)

    objective = [initial_value]
    accepted_scales: list[float] = []
    for _ in range(settings.iterations):
        state = current_state
        em = multifrequency_group_em_update(
            state.means,
            state.covariances,
            omega,
            group_size=settings.group_size,
            floor=settings.gamma_floor,
        )
        if settings.update_rule == "em":
            proposed_gamma = em
        else:
            mu = multifrequency_group_mu_update(
                operators,
                data,
                gamma,
                noise,
                omega,
                group_size=settings.group_size,
                floor=settings.gamma_floor,
                posterior=state,
            )
            # em_fraction is the EM coefficient, matching V4's frozen API.
            proposed_gamma = log_geometric_mix(
                mu,
                em,
                settings.em_fraction,
                floor=settings.gamma_floor,
            )
        raw_noise = multifrequency_noise_update(
            operators,
            data,
            state.means,
            state.covariances,
            floor=settings.noise_floor,
        )
        proposed_noise = log_geometric_mix(
            noise,
            raw_noise,
            settings.noise_damping,
            floor=settings.noise_floor,
        )
        gamma, noise, scale, value = log_backtrack_multifrequency(
            gamma,
            noise,
            proposed_gamma,
            proposed_noise,
            objective[-1],
            evaluate,
            fractions=settings.backtracking_fractions,
            tolerance=settings.objective_tolerance,
        )
        accepted_scales.append(float(scale))
        objective.append(float(value))
        if scale > 0.0:
            matches = [
                item
                for item in reversed(evaluated)
                if np.array_equal(item[0], gamma)
                and np.array_equal(item[1], noise)
            ]
            if not matches:  # pragma: no cover - defensive invariant
                raise RuntimeError("accepted SBL state lacks its cached posterior")
            current_state = matches[0][3]

    endpoint = _reorder_posterior(current_state, restore_order)
    output_noise = np.asarray(noise[restore_order], dtype=np.float64)
    output_weights = np.asarray(weights[restore_order], dtype=np.float64)
    output_omega = np.asarray(omega[restore_order], dtype=np.float64)
    if not (
        np.all(np.isfinite(endpoint.means))
        and np.all(np.isfinite(endpoint.covariances))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(output_noise))
    ):
        raise FloatingPointError("non-finite multi-frequency SBL endpoint")
    return MultiFrequencySBLResult(
        means=endpoint.means,
        covariances=endpoint.covariances,
        gamma=np.asarray(gamma, dtype=np.float64),
        noise=output_noise,
        frequency_weights=output_weights,
        omega=output_omega,
        objective=tuple(float(value) for value in objective),
        accepted_scales=tuple(accepted_scales),
        endpoint_posterior=endpoint,
        metadata={
            "latent_source_contract": "independent_X_f_shared_group_gamma",
            "per_frequency_noise": True,
            "frequency_count": int(operators.shape[0]),
            "available_frequency_count": int(np.count_nonzero(available)),
            "iterations_fixed": int(settings.iterations),
            "initial_posterior_reused": bool(initial_posterior is not None),
            "endpoint_factorization_token": endpoint.factorization_token,
            "update_rule": settings.update_rule,
            "truth_inputs_accepted": False,
        },
    )


__all__ = [
    "MultiFrequencyPosterior",
    "MultiFrequencySBLConfig",
    "MultiFrequencySBLResult",
    "effective_frequency_weights",
    "fit_multifrequency_group_sbl",
    "log_backtrack_multifrequency",
    "log_geometric_mix",
    "multifrequency_group_em_update",
    "multifrequency_group_mu_update",
    "multifrequency_noise_update",
    "multifrequency_posterior",
    "multifrequency_type2_objective",
    "multifrequency_type2_terms_from_posterior",
]
