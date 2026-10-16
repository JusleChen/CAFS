"""Fast fixed-iteration structured complex SBL for the V4 candidate family.

The implementation deliberately keeps the safety-critical part analytic.  A
proper-complex group posterior supplies EM and MacKay/MU proposals, log-domain
mixing keeps every variance positive, and a sparse row-stochastic 0/1/2-hop
operator supplies the only spatial structure.  Every proposed structured step
is accepted only when the Type-II objective does not increase (up to a fixed
floating-point tolerance).

No truth-valued object is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

from .complex_sbl import complex_gaussian_posterior, type2_objective
from .v4_multifrequency_core import (
    MultiFrequencyPosterior,
    effective_frequency_weights,
    multifrequency_group_em_update,
    multifrequency_group_mu_update,
    multifrequency_noise_update,
    multifrequency_posterior,
    multifrequency_type2_objective,
    multifrequency_type2_terms_from_posterior,
)


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _snapshots(value: ArrayLike) -> ComplexArray:
    out = np.asarray(value, dtype=np.complex128)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or not np.all(np.isfinite(out)):
        raise ValueError("observations must be a finite sensor-by-snapshot array")
    return out


def stack_frequency_problem(
    leadfields: ArrayLike,
    observations: ArrayLike,
    frequency_weights: ArrayLike,
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    """Build the legacy stacked problem used by non-Bayesian baselines.

    This adapter remains for eLORETA/RAP-MUSIC and historical callers.  V4 SBL
    candidates do *not* use it, because stacking would impose one common
    coefficient matrix across frequencies.  They use
    :mod:`v4_multifrequency_core` and share only group ``gamma``.
    """

    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if operators.ndim != 3:
        raise ValueError("leadfields must have shape (frequency, sensor, coefficient)")
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or data.shape[:2] != operators.shape[:2]:
        raise ValueError("observations must align with frequency and sensor axes")
    if not np.all(np.isfinite(operators)) or not np.all(np.isfinite(data)):
        raise ValueError("leadfields and observations must be finite")
    weights = np.asarray(frequency_weights, dtype=np.float64)
    if weights.shape != (operators.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("frequency_weights must be strictly positive and aligned")
    if not np.all(np.isfinite(weights)):
        raise ValueError("frequency_weights must be finite")
    weights = weights / weights.sum()
    root = np.sqrt(weights)
    stacked_operator = np.vstack(
        [root[f] * operators[f] for f in range(operators.shape[0])]
    ).astype(np.complex128)
    stacked_data = np.vstack(
        [root[f] * data[f] for f in range(data.shape[0])]
    ).astype(np.complex128)
    return stacked_operator, stacked_data, weights


def _validate_group_problem(
    operator: ComplexArray,
    mean: ComplexArray,
    covariance: ComplexArray,
    group_size: int,
) -> int:
    size = int(group_size)
    if size < 1 or operator.shape[1] % size:
        raise ValueError("coefficient count must be divisible by group_size")
    if mean.shape[0] != operator.shape[1]:
        raise ValueError("posterior mean does not match the operator")
    if covariance.shape != (operator.shape[1], operator.shape[1]):
        raise ValueError("posterior covariance does not match the operator")
    return operator.shape[1] // size


def group_em_update(
    posterior_mean: ArrayLike,
    posterior_covariance: ArrayLike,
    *,
    group_size: int = 3,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Exact grouped EM variance update from posterior second moments."""

    mean = _snapshots(posterior_mean)
    covariance = np.asarray(posterior_covariance, dtype=np.complex128)
    size = int(group_size)
    if size < 1 or mean.shape[0] % size:
        raise ValueError("posterior coefficient count must divide into groups")
    if covariance.shape != (mean.shape[0], mean.shape[0]):
        raise ValueError("posterior covariance has the wrong shape")
    minimum = float(floor)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("floor must be finite and positive")
    count = mean.shape[0] // size
    result = np.empty(count, dtype=np.float64)
    for group in range(count):
        sl = slice(group * size, (group + 1) * size)
        second = covariance[sl, sl] + mean[sl] @ mean[sl].conj().T / mean.shape[1]
        result[group] = max(float(np.trace(second).real / size), minimum)
    return result


def group_mu_update(
    operator: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    noise: float,
    *,
    group_size: int = 3,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Grouped MacKay multiplicative (MU) update using dense reference terms.

    For group ``g`` the proposal is ``sqrt(q_g / s_g)`` where ``q_g`` is the
    average squared matched residual score and ``s_g`` is the average group
    leverage under ``C_y^-1``.  Both use the same proper-complex sensor
    covariance as the posterior.
    """

    a = np.asarray(operator, dtype=np.complex128)
    y = _snapshots(observations)
    values = np.asarray(gamma, dtype=np.float64)
    size = int(group_size)
    if a.ndim != 2 or a.shape[0] != y.shape[0] or a.shape[1] % size:
        raise ValueError("operator, observations, and group_size are incompatible")
    n_groups = a.shape[1] // size
    if values.shape != (n_groups,) or np.any(values <= 0.0):
        raise ValueError("gamma must contain one positive value per group")
    minimum = float(floor)
    _, _, cy = complex_gaussian_posterior(a, y, values, noise, group_size=size)
    factor = linalg.cho_factor((cy + cy.conj().T) * 0.5, lower=True, check_finite=False)
    solved_y = linalg.cho_solve(factor, y, check_finite=False)
    solved_a = linalg.cho_solve(factor, a, check_finite=False)
    proposal = np.empty(n_groups, dtype=np.float64)
    for group in range(n_groups):
        sl = slice(group * size, (group + 1) * size)
        score = a[:, sl].conj().T @ solved_y
        q_value = float(np.linalg.norm(score, "fro") ** 2 / (size * y.shape[1]))
        leverage = float(np.trace(a[:, sl].conj().T @ solved_a[:, sl]).real / size)
        proposal[group] = max(np.sqrt(max(q_value, minimum) / max(leverage, minimum)), minimum)
    return proposal


def em_mu_anchor(
    em_gamma: ArrayLike,
    mu_gamma: ArrayLike,
    *,
    em_fraction: float = 0.5,
    floor: float = 1.0e-12,
) -> FloatArray:
    """Positive log-geometric EM/MU anchor."""

    em = np.maximum(np.asarray(em_gamma, dtype=np.float64), float(floor))
    mu = np.maximum(np.asarray(mu_gamma, dtype=np.float64), float(floor))
    fraction = float(em_fraction)
    if em.shape != mu.shape or not 0.0 <= fraction <= 1.0:
        raise ValueError("EM/MU arrays must align and em_fraction must lie in [0,1]")
    if not np.all(np.isfinite(em)) or not np.all(np.isfinite(mu)):
        raise ValueError("EM/MU proposals must be finite")
    return np.exp(fraction * np.log(em) + (1.0 - fraction) * np.log(mu))


def row_stochastic_graph(
    coordinates: ArrayLike,
    *,
    neighbours: int = 6,
    bandwidth: float | None = None,
    include_self: bool = True,
) -> sparse.csr_matrix:
    """Construct a symmetric-kNN weighted, row-stochastic sparse graph."""

    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1 or not np.all(np.isfinite(xyz)):
        raise ValueError("coordinates must be a finite n-by-3 array")
    if len(xyz) == 1:
        return sparse.eye(1, format="csr", dtype=np.float64)
    k = min(max(int(neighbours), 1), len(xyz) - 1)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    nearest = np.argpartition(distances, kth=k, axis=1)[:, 1 : k + 1]
    positive = distances[np.arange(len(xyz))[:, None], nearest]
    scale = float(np.median(positive)) if bandwidth is None else float(bandwidth)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    rows = np.repeat(np.arange(len(xyz)), k)
    cols = nearest.reshape(-1)
    vals = np.exp(-0.5 * (distances[rows, cols] / scale) ** 2)
    base = sparse.coo_matrix((vals, (rows, cols)), shape=(len(xyz), len(xyz))).tocsr()
    base = base.maximum(base.T)
    if include_self:
        base = base + sparse.eye(len(xyz), format="csr", dtype=np.float64)
    row_sum = np.asarray(base.sum(axis=1)).reshape(-1)
    if np.any(row_sum <= 0.0):
        raise ValueError("graph contains an empty row")
    return sparse.diags(1.0 / row_sum) @ base


def multiscale_graph_mix(
    values: ArrayLike,
    graph: sparse.spmatrix,
    logits: Sequence[float] = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Mix zero-, one-, and two-hop graph signals with softmax coefficients."""

    vector = np.asarray(values, dtype=np.float64)
    matrix = sparse.csr_matrix(graph, dtype=np.float64)
    if vector.ndim != 1 or matrix.shape != (len(vector), len(vector)):
        raise ValueError("graph and value vector are incompatible")
    raw = np.asarray(tuple(logits), dtype=np.float64)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError("graph logits must contain three finite values")
    probabilities = np.exp(raw - np.max(raw))
    probabilities /= probabilities.sum()
    one_hop = np.asarray(matrix @ vector).reshape(-1)
    two_hop = np.asarray(matrix @ one_hop).reshape(-1)
    return (
        probabilities[0] * vector
        + probabilities[1] * one_hop
        + probabilities[2] * two_hop
    ).astype(np.float64)


def _noise_update(
    operator: ComplexArray,
    observations: ComplexArray,
    mean: ComplexArray,
    covariance: ComplexArray,
    *,
    floor: float,
) -> float:
    residual = observations - operator @ mean
    posterior_trace = float(np.trace(operator @ covariance @ operator.conj().T).real)
    value = (
        float(np.linalg.norm(residual, "fro") ** 2)
        + observations.shape[1] * max(posterior_trace, 0.0)
    ) / (operator.shape[0] * observations.shape[1])
    return max(float(value), float(floor))


def _log_backtrack(
    current: FloatArray,
    proposal: FloatArray,
    objective_limit: float,
    evaluator,
    *,
    tolerance: float,
) -> tuple[FloatArray, float, float]:
    for fraction in (1.0, 0.5, 0.25):
        candidate = np.exp(
            (1.0 - fraction) * np.log(current) + fraction * np.log(proposal)
        )
        value = float(evaluator(candidate))
        if np.isfinite(value) and value <= objective_limit + tolerance * max(1.0, abs(objective_limit)):
            return candidate, fraction, value
    return current.copy(), 0.0, float(objective_limit)


@dataclass(frozen=True, slots=True)
class FastStructuredMMConfig:
    iterations: int = 8
    group_size: int = 3
    em_fraction: float = 0.65
    graph_logits: tuple[float, float, float] = (1.5, 0.0, -0.5)
    graph_damping: float = 0.35
    gamma_floor: float = 1.0e-12
    noise_floor: float = 1.0e-12
    objective_tolerance: float = 2.0e-10

    def __post_init__(self) -> None:
        if self.iterations not in {6, 8}:
            raise ValueError("V4 fixed iterations must be 6 or 8")
        if self.group_size < 1 or not 0.0 <= self.em_fraction <= 1.0:
            raise ValueError("invalid group size or EM fraction")
        if not 0.0 <= self.graph_damping <= 1.0:
            raise ValueError("graph_damping must lie in [0,1]")
        if self.gamma_floor <= 0.0 or self.noise_floor <= 0.0:
            raise ValueError("variance floors must be positive")


@dataclass(frozen=True, slots=True)
class FastStructuredMMResult:
    method: str
    mean: ComplexArray
    covariance: ComplexArray
    gamma: FloatArray
    noise: FloatArray
    frequency_weights: FloatArray
    objective: tuple[float, ...]
    anchor_scales: tuple[float, ...]
    graph_scales: tuple[float, ...]
    runtime_seconds: float
    metadata: Mapping[str, object]


def fit_fast_structured_mm(
    leadfields: ArrayLike,
    observations: ArrayLike,
    frequency_weights: ArrayLike,
    graph: sparse.spmatrix,
    *,
    config: FastStructuredMMConfig | None = None,
    gamma0: ArrayLike | None = None,
    noise0: ArrayLike | float | None = None,
    initial_posterior: MultiFrequencyPosterior | None = None,
    completed_pilot_iterations: int = 0,
) -> FastStructuredMMResult:
    """Fit the analytic V4 fixed-iteration structured-MM candidate."""

    started = perf_counter()
    settings = config or FastStructuredMMConfig()
    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if data.ndim == 2:
        data = data[:, :, None]
    if (
        operators.ndim != 3
        or data.ndim != 3
        or data.shape[:2] != operators.shape[:2]
        or operators.shape[2] % settings.group_size
        or not np.all(np.isfinite(operators))
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("leadfields and observations must form a finite grouped frequency problem")
    weights, omega, available = effective_frequency_weights(
        frequency_weights, operators.shape[0]
    )
    n_groups = operators.shape[2] // settings.group_size
    matrix = sparse.csr_matrix(graph, dtype=np.float64)
    if matrix.shape != (n_groups, n_groups):
        raise ValueError("graph must contain one row per source group")
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if np.max(np.abs(row_sums - 1.0)) > 1.0e-10 or np.any(matrix.data < 0.0):
        raise ValueError("graph must be nonnegative and row stochastic")
    frequency_power = np.mean(np.abs(data) ** 2, axis=(1, 2)).real
    data_power = max(
        float(np.dot(omega, frequency_power) / np.sum(omega)),
        settings.gamma_floor,
    )
    gamma = (
        np.full(n_groups, data_power, dtype=np.float64)
        if gamma0 is None
        else np.maximum(np.asarray(gamma0, dtype=np.float64), settings.gamma_floor)
    )
    if gamma.shape != (n_groups,) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma0 must be finite with one value per group")
    if noise0 is None:
        noise = np.maximum(0.1 * frequency_power, settings.noise_floor)
    else:
        raw_noise = np.asarray(noise0, dtype=np.float64)
        if raw_noise.ndim == 0:
            noise = np.full(operators.shape[0], float(raw_noise), dtype=np.float64)
        else:
            noise = raw_noise.reshape(-1)
        if (
            noise.shape != (operators.shape[0],)
            or not np.all(np.isfinite(noise))
            or np.any(noise <= 0.0)
        ):
            raise ValueError("noise0 must contain one positive value per frequency")
        noise = np.maximum(noise, settings.noise_floor)
    completed = int(completed_pilot_iterations)
    if completed not in {0, 2}:
        raise ValueError("completed pilot iterations must be zero or two")
    if completed and initial_posterior is None:
        raise ValueError("pilot-completed structured MM requires its posterior cache")
    remaining_iterations = settings.iterations - completed
    if remaining_iterations < 0:
        raise ValueError("pilot iterations exceed structured-MM budget")
    current_posterior = initial_posterior
    objective = [
        float(
            np.dot(
                omega,
                multifrequency_type2_terms_from_posterior(
                    data, current_posterior
                ),
            )
        )
        if current_posterior is not None
        else multifrequency_type2_objective(
            operators,
            data,
            gamma,
            noise,
            omega=omega,
            group_size=settings.group_size,
        )
    ]
    anchor_scales: list[float] = []
    graph_scales: list[float] = []

    for _ in range(remaining_iterations):
        posterior = (
            current_posterior
            if current_posterior is not None
            else multifrequency_posterior(
                operators, data, gamma, noise, group_size=settings.group_size
            )
        )
        current_posterior = None
        em = multifrequency_group_em_update(
            posterior.means,
            posterior.covariances,
            omega,
            group_size=settings.group_size,
            floor=settings.gamma_floor,
        )
        mu = multifrequency_group_mu_update(
            operators,
            data,
            gamma,
            noise,
            omega,
            group_size=settings.group_size,
            floor=settings.gamma_floor,
        )
        anchor_proposal = em_mu_anchor(
            em, mu, em_fraction=settings.em_fraction, floor=settings.gamma_floor
        )
        proposed_noise = multifrequency_noise_update(
            operators,
            data,
            posterior.means,
            posterior.covariances,
            floor=settings.noise_floor,
        )

        def anchor_objective(candidate: FloatArray) -> float:
            return multifrequency_type2_objective(
                operators,
                data,
                candidate,
                proposed_noise,
                omega=omega,
                group_size=settings.group_size,
            )

        anchor, anchor_scale, anchor_value = _log_backtrack(
            gamma,
            anchor_proposal,
            objective[-1],
            anchor_objective,
            tolerance=settings.objective_tolerance,
        )
        if anchor_scale == 0.0:
            proposed_noise = noise
        z_anchor = np.log(np.maximum(anchor, settings.gamma_floor))
        z_graph = multiscale_graph_mix(z_anchor, matrix, settings.graph_logits)
        structured = np.exp(
            (1.0 - settings.graph_damping) * z_anchor
            + settings.graph_damping * z_graph
        )

        def graph_objective(candidate: FloatArray) -> float:
            return multifrequency_type2_objective(
                operators,
                data,
                candidate,
                proposed_noise,
                omega=omega,
                group_size=settings.group_size,
            )

        gamma, graph_scale, accepted_value = _log_backtrack(
            anchor,
            structured,
            anchor_value,
            graph_objective,
            tolerance=settings.objective_tolerance,
        )
        noise = proposed_noise
        anchor_scales.append(float(anchor_scale))
        graph_scales.append(float(graph_scale))
        objective.append(float(accepted_value))

    posterior = multifrequency_posterior(
        operators, data, gamma, noise, group_size=settings.group_size
    )
    if not (
        np.all(np.isfinite(posterior.means))
        and np.all(np.isfinite(posterior.covariances))
        and np.all(np.isfinite(gamma))
        and np.all(np.isfinite(noise))
    ):
        raise FloatingPointError("non-finite fixed-iteration structured-MM endpoint")
    return FastStructuredMMResult(
        method="V4-Fast-Structured-MM-FreeSIGM",
        mean=np.asarray(posterior.means, dtype=np.complex128),
        covariance=np.asarray(posterior.covariances, dtype=np.complex128),
        gamma=np.asarray(gamma, dtype=np.float64),
        noise=np.asarray(noise, dtype=np.float64),
        frequency_weights=np.asarray(weights, dtype=np.float64),
        objective=tuple(objective),
        anchor_scales=tuple(anchor_scales),
        graph_scales=tuple(graph_scales),
        runtime_seconds=float(perf_counter() - started),
        metadata={
            "iterations_fixed": settings.iterations,
            "pilot_iterations_reused": completed,
            "main_iterations_executed": remaining_iterations,
            "pilot_factorization_token": (
                None
                if initial_posterior is None
                else initial_posterior.factorization_token
            ),
            "truth_inputs_accepted": False,
            "active_set_pruning": False,
            "graph_sparse": True,
            "graph_nnz": int(matrix.nnz),
            "objective_nonincrease_tolerance": settings.objective_tolerance,
            "latent_source_contract": "independent_X_f_shared_group_gamma",
            "per_frequency_noise": True,
            "effective_frequency_weights": omega.tolist(),
            "available_frequency_count": int(np.count_nonzero(available)),
        },
    )


__all__ = [
    "FastStructuredMMConfig",
    "FastStructuredMMResult",
    "em_mu_anchor",
    "fit_fast_structured_mm",
    "group_em_update",
    "group_mu_update",
    "multiscale_graph_mix",
    "row_stochastic_graph",
    "stack_frequency_problem",
]
