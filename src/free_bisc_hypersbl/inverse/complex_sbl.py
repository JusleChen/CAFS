"""Complex group-ARD and multi-frequency HyperSBL.

The implementation uses sensor-space Cholesky solves (Woodbury form) and never
forms a matrix inverse.  Spatial groups normally contain the three Cartesian
dipole orientations.  The multi-frequency solver shares spatial ARD variables
while retaining the complete lead-field family instead of averaging matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import linalg


Array = np.ndarray


def _as_snapshots(y: Array) -> Array:
    out = np.asarray(y, dtype=np.complex128)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2:
        raise ValueError("observations must have shape (sensors, snapshots)")
    return out


def _hermitian(a: Array) -> Array:
    return (a + a.conj().T) * 0.5


def _expand_group_gamma(gamma: Array, n_coefficients: int, group_size: int) -> Array:
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if n_coefficients % group_size:
        raise ValueError("coefficient count must be divisible by group_size")
    if gamma.size == n_coefficients:
        return gamma
    if gamma.size != n_coefficients // group_size:
        raise ValueError("gamma must have one entry per group or coefficient")
    return np.repeat(gamma, group_size)


def _cho_factor(a: Array) -> tuple[Array, bool]:
    a = _hermitian(np.asarray(a, dtype=np.complex128))
    scale = max(float(np.trace(a).real / max(1, a.shape[0])), 1.0)
    jitter = np.finfo(float).eps * scale
    for _ in range(8):
        try:
            return linalg.cho_factor(a + jitter * np.eye(a.shape[0]), lower=True, check_finite=False)
        except linalg.LinAlgError:
            jitter *= 10.0
    raise linalg.LinAlgError("Hermitian covariance is not positive definite")


def complex_gaussian_posterior(
    a: Array,
    y: Array,
    gamma: Array,
    noise_cov: Array | float,
    *,
    group_size: int = 1,
) -> tuple[Array, Array, Array]:
    """Return posterior mean, covariance, and sensor covariance.

    This is the stable dual expression
    ``mu = Gamma A^H C_y^-1 y`` and
    ``Sigma = Gamma - Gamma A^H C_y^-1 A Gamma``.
    """

    a = np.asarray(a, dtype=np.complex128)
    y = _as_snapshots(y)
    if a.ndim != 2 or a.shape[0] != y.shape[0]:
        raise ValueError("incompatible forward operator and observations")
    gamma_diag = _expand_group_gamma(gamma, a.shape[1], group_size)
    if np.isscalar(noise_cov):
        noise = float(noise_cov) * np.eye(a.shape[0], dtype=np.complex128)
    else:
        noise = np.asarray(noise_cov, dtype=np.complex128)
    if noise.shape != (a.shape[0], a.shape[0]):
        raise ValueError("noise covariance has the wrong shape")
    ag = a * gamma_diag[None, :]
    cy = _hermitian(ag @ a.conj().T + noise)
    factor = _cho_factor(cy)
    cy_inv_y = linalg.cho_solve(factor, y, check_finite=False)
    cy_inv_a = linalg.cho_solve(factor, a, check_finite=False)
    mu = gamma_diag[:, None] * (a.conj().T @ cy_inv_y)
    sigma = np.diag(gamma_diag) - (gamma_diag[:, None] * (a.conj().T @ cy_inv_a)) * gamma_diag[None, :]
    return mu, _hermitian(sigma), cy


def _negative_log_likelihood_from_covariance(
    observations: Array,
    sensor_covariance: Array,
    *,
    independent_snapshots: int,
) -> float:
    """Evaluate a proper-complex Gaussian NLL from an existing factor target."""

    y = np.asarray(observations, dtype=np.complex128)
    cy = np.asarray(sensor_covariance, dtype=np.complex128)
    factor = _cho_factor(cy)
    diagonal = np.real(np.diag(factor[0]))
    logdet = 2.0 * float(np.log(np.maximum(diagonal, np.finfo(float).tiny)).sum())
    quadratic = float(np.real(np.vdot(y, linalg.cho_solve(factor, y, check_finite=False))))
    return int(independent_snapshots) * logdet + quadratic


def type2_objective(a: Array, y: Array, gamma: Array, noise_cov: Array | float, *, group_size: int = 1) -> float:
    """Complex Gaussian negative log marginal likelihood, up to constants."""

    y = _as_snapshots(y)
    _, _, cy = complex_gaussian_posterior(a, y, gamma, noise_cov, group_size=group_size)
    return _negative_log_likelihood_from_covariance(
        y,
        cy,
        independent_snapshots=y.shape[1],
    )


def correlation_kernel(size: int, length_scale: float, *, kind: str = "exponential", jitter: float = 1e-8) -> Array:
    """Construct a unit-diagonal positive-definite time/frequency kernel."""

    if size < 1 or length_scale <= 0:
        raise ValueError("size and length_scale must be positive")
    distance = np.abs(np.subtract.outer(np.arange(size), np.arange(size))).astype(float)
    if kind == "exponential":
        kernel = np.exp(-distance / length_scale)
    elif kind == "squared_exponential":
        kernel = np.exp(-0.5 * (distance / length_scale) ** 2)
    else:
        raise ValueError(f"unsupported kernel kind: {kind}")
    kernel += jitter * np.eye(size)
    scale = np.sqrt(np.diag(kernel))
    return kernel / scale[:, None] / scale[None, :]


def structured_complex_gaussian_posterior(
    leadfields: Iterable[Array],
    observations: Iterable[Array],
    gamma: Array,
    noise0: float,
    *,
    frequency_weights: Array,
    c_time: Array,
    c_frequency: Array,
    group_size: int = 3,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Dense reference posterior for correlated frequency/time coefficients.

    Vector order is ``time -> frequency -> source coefficient``.  This routine
    is intentionally limited to validation/Pilot sizes; production-scale runs
    use operator/HVP variants.  Unlike the iid-snapshot objective, time is part
    of one random vector here, so its NLL contains one log-determinant rather
    than an additional multiplier by ``T``.
    """

    operators = [np.asarray(g, dtype=np.complex128) for g in leadfields]
    data = [_as_snapshots(y) for y in observations]
    if not operators or len(operators) != len(data):
        raise ValueError("leadfields and observations must align")
    n_frequency = len(operators)
    n_coeff = operators[0].shape[1]
    n_time = data[0].shape[1]
    if any(g.shape[1] != n_coeff for g in operators):
        raise ValueError("leadfields must share the source dimension")
    if any(y.shape[1] != n_time or y.shape[0] != g.shape[0] for g, y in zip(operators, data)):
        raise ValueError("all frequencies must share the time dimension")
    c_time = np.asarray(c_time, dtype=np.complex128)
    c_frequency = np.asarray(c_frequency, dtype=np.complex128)
    if c_time.shape != (n_time, n_time) or c_frequency.shape != (n_frequency, n_frequency):
        raise ValueError("correlation kernel shape mismatch")
    weights = np.asarray(frequency_weights, dtype=float)
    if weights.shape != (n_frequency,) or np.any(weights <= 0):
        raise ValueError("frequency weights must be positive")
    weights = weights / weights.sum()
    effective = n_frequency * weights + 1e-12
    gamma_diag = _expand_group_gamma(gamma, n_coeff, group_size)
    spatial = np.diag(gamma_diag).astype(np.complex128)
    prior_one_time = np.kron(c_frequency, spatial)
    prior = np.kron(c_time, prior_one_time)
    a_one_time = linalg.block_diag(*operators)
    design = np.kron(np.eye(n_time), a_one_time)
    noise_one_time = linalg.block_diag(
        *[
            noise0 / effective[f] * np.eye(operators[f].shape[0], dtype=np.complex128)
            for f in range(n_frequency)
        ]
    )
    noise = np.kron(np.eye(n_time), noise_one_time)
    y_vector = np.concatenate(
        [np.concatenate([data[f][:, t] for f in range(n_frequency)]) for t in range(n_time)]
    )[:, None]
    pa = prior @ design.conj().T
    cy = _hermitian(design @ pa + noise)
    factor = _cho_factor(cy)
    solved_y = linalg.cho_solve(factor, y_vector, check_finite=False)
    solved_design_prior = linalg.cho_solve(factor, design @ prior, check_finite=False)
    mean = pa @ solved_y
    covariance = _hermitian(prior - pa @ solved_design_prior)
    return mean[:, 0], covariance, cy, design, noise, y_vector[:, 0]


def structured_type2_objective(
    leadfields: Iterable[Array],
    observations: Iterable[Array],
    gamma: Array,
    noise0: float,
    *,
    frequency_weights: Array,
    c_time: Array,
    c_frequency: Array,
    group_size: int = 3,
) -> float:
    _, _, cy, _, _, y = structured_complex_gaussian_posterior(
        leadfields,
        observations,
        gamma,
        noise0,
        frequency_weights=frequency_weights,
        c_time=c_time,
        c_frequency=c_frequency,
        group_size=group_size,
    )
    return _negative_log_likelihood_from_covariance(
        y,
        cy,
        independent_snapshots=1,
    )


@dataclass
class SBLResult:
    mean: Array
    covariance: Array
    gamma: Array
    noise: float
    objective: list[float]
    converged: bool
    iterations: int
    active_groups: Array
    metadata: dict[str, object]


class ComplexGroupSBL:
    """Expectation-maximization group ARD for proper complex data."""

    def __init__(
        self,
        *,
        group_size: int = 3,
        max_iter: int = 100,
        tol: float = 1e-6,
        prune_threshold: float = 0.0,
        min_variance: float = 1e-12,
    ) -> None:
        self.group_size = int(group_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.prune_threshold = float(prune_threshold)
        self.min_variance = float(min_variance)

    def fit(
        self,
        a: Array,
        y: Array,
        *,
        gamma0: Array | None = None,
        noise0: float | None = None,
    ) -> SBLResult:
        a = np.asarray(a, dtype=np.complex128)
        y = _as_snapshots(y)
        n_sensors, n_coeff = a.shape
        if n_coeff % self.group_size:
            raise ValueError("forward columns must be divisible by group_size")
        n_groups = n_coeff // self.group_size
        power = max(float(np.mean(np.abs(y) ** 2)), self.min_variance)
        gamma = np.full(n_groups, power, dtype=float) if gamma0 is None else np.asarray(gamma0, dtype=float).copy()
        noise = max(float(noise0 if noise0 is not None else power * 0.1), self.min_variance)
        if gamma.shape != (n_groups,):
            raise ValueError("gamma0 has the wrong shape")
        objective = [type2_objective(a, y, gamma, noise, group_size=self.group_size)]
        converged = False
        mean = np.zeros((n_coeff, y.shape[1]), dtype=np.complex128)
        covariance = np.diag(np.repeat(gamma, self.group_size)).astype(np.complex128)

        for _ in range(self.max_iter):
            mean, covariance, _ = complex_gaussian_posterior(
                a, y, gamma, noise, group_size=self.group_size
            )
            proposed_gamma = np.empty_like(gamma)
            for group in range(n_groups):
                sl = slice(group * self.group_size, (group + 1) * self.group_size)
                second_moment = covariance[sl, sl] + mean[sl] @ mean[sl].conj().T / y.shape[1]
                proposed_gamma[group] = max(
                    float(np.trace(second_moment).real / self.group_size), self.min_variance
                )
            residual = y - a @ mean
            posterior_sensor_trace = float(np.trace(a @ covariance @ a.conj().T).real)
            proposed_noise = max(
                (float(np.linalg.norm(residual, "fro") ** 2) + y.shape[1] * posterior_sensor_trace)
                / (n_sensors * y.shape[1]),
                self.min_variance,
            )

            # EM is monotone analytically; log-domain damping protects that
            # property from floating-point and near-singular covariance errors.
            accepted_gamma = proposed_gamma
            accepted_noise = proposed_noise
            accepted_objective = type2_objective(
                a, y, accepted_gamma, accepted_noise, group_size=self.group_size
            )
            fraction = 1.0
            while accepted_objective > objective[-1] + 1e-9 * max(1.0, abs(objective[-1])) and fraction > 2.0**-16:
                fraction *= 0.5
                accepted_gamma = np.exp((1.0 - fraction) * np.log(gamma) + fraction * np.log(proposed_gamma))
                accepted_noise = float(np.exp((1.0 - fraction) * np.log(noise) + fraction * np.log(proposed_noise)))
                accepted_objective = type2_objective(
                    a, y, accepted_gamma, accepted_noise, group_size=self.group_size
                )
            if accepted_objective > objective[-1] + 1e-8:
                break

            relative = max(
                float(np.max(np.abs(accepted_gamma - gamma) / np.maximum(gamma, self.min_variance))),
                abs(accepted_noise - noise) / max(noise, self.min_variance),
            )
            gamma, noise = accepted_gamma, accepted_noise
            objective.append(float(accepted_objective))
            if relative < self.tol:
                converged = True
                break

        mean, covariance, _ = complex_gaussian_posterior(a, y, gamma, noise, group_size=self.group_size)
        if self.prune_threshold > 0:
            active = gamma >= self.prune_threshold * max(float(gamma.max()), self.min_variance)
        else:
            active = np.ones_like(gamma, dtype=bool)
        return SBLResult(
            mean=mean,
            covariance=covariance,
            gamma=gamma,
            noise=float(noise),
            objective=objective,
            converged=converged,
            iterations=len(objective) - 1,
            active_groups=np.flatnonzero(active),
            metadata={"dtype": "complex128", "solver": "sensor-space Cholesky EM"},
        )


class MultiFrequencyHyperSBL:
    """Shared-support complex HyperSBL over a family of frequency operators."""

    def __init__(self, *, group_size: int = 3, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.group_size = int(group_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(
        self,
        leadfields: Iterable[Array],
        observations: Iterable[Array],
        *,
        frequency_weights: Array | None = None,
        mode: str = "sparse",
        time_length_scale: float = 2.0,
        frequency_length_scale: float = 1.5,
    ) -> SBLResult:
        operators = [np.asarray(g, dtype=np.complex128) for g in leadfields]
        data = [_as_snapshots(y) for y in observations]
        if not operators or len(operators) != len(data):
            raise ValueError("leadfields and observations must be non-empty and aligned")
        if any(g.shape[0] != y.shape[0] for g, y in zip(operators, data)):
            raise ValueError("sensor dimensions do not match")
        n_coeff = operators[0].shape[1]
        if any(g.shape[1] != n_coeff for g in operators) or n_coeff % self.group_size:
            raise ValueError("all leadfields must share grouped source columns")
        n_groups = n_coeff // self.group_size
        n_freq = len(operators)
        pi = np.full(n_freq, 1.0 / n_freq) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
        if np.any(pi <= 0) or pi.shape != (n_freq,):
            raise ValueError("frequency weights must be positive")
        pi = pi / pi.sum()
        effective_weight = n_freq * pi + 1e-12
        power = float(np.mean([np.mean(np.abs(y) ** 2) for y in data]))
        gamma = np.full(n_groups, max(power, 1e-12))
        noise0 = max(0.1 * power, 1e-12)
        objective: list[float] = []
        means: list[Array] = []
        covariances: list[Array] = []

        if mode not in {"sparse", "correlated", "tensor"}:
            raise ValueError("mode must be sparse, correlated, or tensor")
        # Fixed kernels are recorded for correlated mode. Tensor mode learns
        # length scales by a small evidence grid without relaxing SPD.
        if mode == "tensor":
            candidate = [0.5, 1.0, 1.5, 2.5, 4.0]
            frequency_length_scale = min(candidate, key=lambda ell: abs(ell - frequency_length_scale))
            time_length_scale = min(candidate, key=lambda ell: abs(ell - time_length_scale))
        c_omega = correlation_kernel(n_freq, frequency_length_scale) if mode != "sparse" else np.eye(n_freq)
        max_t = max(y.shape[1] for y in data)
        c_time = correlation_kernel(max_t, time_length_scale) if mode != "sparse" else np.eye(max_t)

        if mode != "sparse":
            if any(y.shape[1] != max_t for y in data):
                raise ValueError("structured modes require equal time samples at every frequency")
            # Tensor mode learns low-dimensional SPD kernel parameters on an
            # evidence grid. Unit-diagonal normalization removes the otherwise
            # unidentifiable scale trade-off with spatial gamma.
            if mode == "tensor":
                best = (np.inf, c_time, c_omega, time_length_scale, frequency_length_scale)
                for time_scale in (0.5, 1.0, 2.0, 4.0):
                    ct = correlation_kernel(max_t, time_scale)
                    for frequency_scale in (0.5, 1.0, 2.0, 4.0):
                        cw = correlation_kernel(n_freq, frequency_scale)
                        value = structured_type2_objective(
                            operators,
                            data,
                            gamma,
                            noise0,
                            frequency_weights=pi,
                            c_time=ct,
                            c_frequency=cw,
                            group_size=self.group_size,
                        )
                        if value < best[0]:
                            best = (value, ct, cw, time_scale, frequency_scale)
                _, c_time, c_omega, time_length_scale, frequency_length_scale = best
            return self._fit_structured(
                operators,
                data,
                gamma,
                noise0,
                pi,
                c_time,
                c_omega,
                mode=mode,
                time_length_scale=time_length_scale,
                frequency_length_scale=frequency_length_scale,
            )

        def sparse_state(
            current_gamma: Array,
            current_noise: float,
        ) -> tuple[list[Array], list[Array], list[float], float]:
            state_means: list[Array] = []
            state_covariances: list[Array] = []
            residual_terms: list[float] = []
            total_objective = 0.0
            for f, (a, y) in enumerate(zip(operators, data)):
                lambda_f = current_noise / effective_weight[f]
                mu, sigma, cy = complex_gaussian_posterior(
                    a,
                    y,
                    current_gamma,
                    lambda_f,
                    group_size=self.group_size,
                )
                state_means.append(mu)
                state_covariances.append(sigma)
                total_objective += _negative_log_likelihood_from_covariance(
                    y,
                    cy,
                    independent_snapshots=y.shape[1],
                )
                residual = y - a @ mu
                uncertainty = y.shape[1] * float(np.trace(a @ sigma @ a.conj().T).real)
                residual_terms.append(float(np.linalg.norm(residual, "fro") ** 2 + uncertainty))
            return state_means, state_covariances, residual_terms, float(total_objective)

        means, covariances, residual_terms, initial_objective = sparse_state(gamma, noise0)
        objective.append(initial_objective)
        converged = False
        convergence_reason = "max_iter"
        for _ in range(self.max_iter):
            group_second = np.zeros(n_groups)
            total_snapshots = 0
            for a, y, mu, sigma in zip(operators, data, means, covariances):
                for group in range(n_groups):
                    sl = slice(group * self.group_size, (group + 1) * self.group_size)
                    second = sigma[sl, sl] + mu[sl] @ mu[sl].conj().T / y.shape[1]
                    group_second[group] += y.shape[1] * float(np.trace(second).real / self.group_size)
                total_snapshots += y.shape[1]
            proposed_gamma = np.maximum(group_second / max(1, total_snapshots), 1e-12)
            denominator = sum(a.shape[0] * y.shape[1] for a, y in zip(operators, data))
            proposed_noise = max(
                sum(float(w) * q for w, q in zip(effective_weight, residual_terms)) / denominator,
                1e-12,
            )
            candidate_gamma = np.sqrt(gamma * proposed_gamma)
            candidate_noise = float(np.sqrt(noise0 * proposed_noise))
            relative = max(
                float(np.max(np.abs(candidate_gamma - gamma) / np.maximum(gamma, 1e-12))),
                abs(candidate_noise - noise0) / max(noise0, 1e-12),
            )
            candidate_state = sparse_state(candidate_gamma, candidate_noise)
            candidate_objective = candidate_state[-1]
            if candidate_objective > objective[-1] + 1e-8 * max(1.0, abs(objective[-1])):
                convergence_reason = "monotonicity_break"
                break
            gamma, noise0 = candidate_gamma, candidate_noise
            means, covariances, residual_terms, _ = candidate_state
            objective.append(float(candidate_objective))
            if relative < self.tol:
                converged = True
                convergence_reason = "tolerance"
                break

        # ``means`` and ``covariances`` are the posterior moments at the exact
        # returned gamma/noise state.  Keeping the accepted state together
        # prevents a final M-step from leaving stale E-step moments behind.
        stacked_mean = np.stack(means, axis=0)
        stacked_covariance = np.stack(covariances, axis=0)
        return SBLResult(
            mean=stacked_mean,
            covariance=stacked_covariance,
            gamma=gamma,
            noise=noise0,
            objective=objective,
            converged=converged,
            iterations=len(objective) - 1,
            active_groups=np.flatnonzero(gamma >= 1e-8 * max(float(gamma.max()), 1e-12)),
            metadata={
                "mode": mode,
                "frequency_weights": pi,
                "C_omega": c_omega,
                "C_t": c_time,
                "convergence_reason": convergence_reason,
                "dtype": "complex128",
            },
        )

    def _fit_structured(
        self,
        operators: list[Array],
        data: list[Array],
        gamma: Array,
        noise0: float,
        pi: Array,
        c_time: Array,
        c_omega: Array,
        *,
        mode: str,
        time_length_scale: float,
        frequency_length_scale: float,
    ) -> SBLResult:
        n_freq = len(operators)
        n_time = data[0].shape[1]
        n_coeff = operators[0].shape[1]
        n_groups = n_coeff // self.group_size
        effective = n_freq * pi + 1e-12
        base_group_covariance = np.kron(
            c_time, np.kron(c_omega, np.eye(self.group_size))
        )
        base_factor = _cho_factor(base_group_covariance)
        def structured_state(
            current_gamma: Array,
            current_noise: float,
        ) -> tuple[Array, Array, Array, Array, float]:
            mean_state, covariance_state, cy, design_state, _, y_state = structured_complex_gaussian_posterior(
                operators,
                data,
                current_gamma,
                current_noise,
                frequency_weights=pi,
                c_time=c_time,
                c_frequency=c_omega,
                group_size=self.group_size,
            )
            value = _negative_log_likelihood_from_covariance(
                y_state,
                cy,
                independent_snapshots=1,
            )
            return mean_state, covariance_state, design_state, y_state, float(value)

        mean, covariance, design, y_vector, current = structured_state(gamma, noise0)
        objective: list[float] = [current]
        converged = False
        convergence_reason = "max_iter"

        for _ in range(self.max_iter):
            proposed_gamma = np.empty_like(gamma)
            for group in range(n_groups):
                indices = np.asarray(
                    [
                        (t * n_freq + f) * n_coeff + group * self.group_size + orient
                        for t in range(n_time)
                        for f in range(n_freq)
                        for orient in range(self.group_size)
                    ],
                    dtype=int,
                )
                second = covariance[np.ix_(indices, indices)] + np.outer(
                    mean[indices], mean[indices].conj()
                )
                whitened = linalg.cho_solve(base_factor, second, check_finite=False)
                proposed_gamma[group] = max(
                    float(np.trace(whitened).real / indices.size), 1e-12
                )
            residual = y_vector - design @ mean
            posterior_sensor = design @ covariance @ design.conj().T
            q_weighted = 0.0
            cursor = 0
            for _t in range(n_time):
                for f, operator in enumerate(operators):
                    count = operator.shape[0]
                    sl = slice(cursor, cursor + count)
                    q = float(np.vdot(residual[sl], residual[sl]).real + np.trace(posterior_sensor[sl, sl]).real)
                    q_weighted += effective[f] * q
                    cursor += count
            proposed_noise = max(q_weighted / y_vector.size, 1e-12)
            # Conservative geometric damping for the structured M-step.
            candidate_gamma = np.sqrt(gamma * proposed_gamma)
            candidate_noise = float(np.sqrt(noise0 * proposed_noise))
            relative = max(
                float(np.max(np.abs(candidate_gamma - gamma) / np.maximum(gamma, 1e-12))),
                abs(candidate_noise - noise0) / max(noise0, 1e-12),
            )
            candidate_state = structured_state(candidate_gamma, candidate_noise)
            candidate_value = candidate_state[-1]
            if candidate_value > current + 1e-8 * max(1.0, abs(current)):
                convergence_reason = "monotonicity_break"
                break
            gamma, noise0 = candidate_gamma, candidate_noise
            mean, covariance, design, y_vector, current = candidate_state
            objective.append(float(current))
            if relative < self.tol:
                converged = True
                convergence_reason = "tolerance"
                break
        mean_ft = mean.reshape(n_time, n_freq, n_coeff).transpose(1, 2, 0)
        return SBLResult(
            mean=mean_ft,
            covariance=covariance,
            gamma=gamma,
            noise=noise0,
            objective=objective,
            converged=converged,
            iterations=len(objective) - 1,
            active_groups=np.flatnonzero(gamma >= 1e-8 * max(float(gamma.max()), 1e-12)),
            metadata={
                "mode": mode,
                "frequency_weights": pi,
                "C_omega": c_omega,
                "C_t": c_time,
                "time_length_scale": time_length_scale,
                "frequency_length_scale": frequency_length_scale,
                "objective_convention": "single structured vector; no duplicated T multiplier",
                "convergence_reason": convergence_reason,
                "dtype": "complex128",
            },
        )
