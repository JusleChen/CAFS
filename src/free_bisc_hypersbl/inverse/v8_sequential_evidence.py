"""Sequential marginal-likelihood coordinate solver in harmonic quotient space."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize

from .v8_chol_updates import CholeskyUpdateError, safe_chol_rank1
from .v8_component_projection import ProjectedComponent
from .v8_sufficient_statistics import FrequencySufficientStatistic


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class CoordinateOptimum:
    theta: float
    delta_objective: float
    derivative_at_zero: float
    stationary_points: tuple[float, ...]


def coordinate_delta(
    theta: float,
    p_matrices: Sequence[ArrayLike],
    q_matrices: Sequence[ArrayLike],
    snapshots: Sequence[int],
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    value = float(theta)
    if value < 0.0:
        return float("inf")
    weights = np.ones(len(p_matrices)) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    total = 0.0
    for frequency, (p_value, q_value, count) in enumerate(zip(p_matrices, q_matrices, snapshots)):
        p = np.atleast_2d(np.asarray(p_value, dtype=np.complex128))
        q = np.atleast_2d(np.asarray(q_value, dtype=np.complex128))
        if p.shape[0] != p.shape[1] or q.shape != p.shape:
            raise ValueError("coordinate P/Q matrices must be aligned and square")
        identity = np.eye(p.shape[0], dtype=np.complex128)
        matrix = identity + value * p
        sign, logdet = np.linalg.slogdet(matrix)
        if np.real(sign) <= 0.0:
            return float("inf")
        trace = np.trace(linalg.solve(matrix, q, assume_a="her", check_finite=False)).real
        total += float(weights[frequency]) * int(count) * (float(logdet) - value * float(trace))
    return float(total)


def coordinate_delta_derivative(
    theta: float,
    p_matrices: Sequence[ArrayLike],
    q_matrices: Sequence[ArrayLike],
    snapshots: Sequence[int],
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    value = float(theta)
    weights = np.ones(len(p_matrices)) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    total = 0.0
    for frequency, (p_value, q_value, count) in enumerate(zip(p_matrices, q_matrices, snapshots)):
        p = np.atleast_2d(np.asarray(p_value, dtype=np.complex128))
        q = np.atleast_2d(np.asarray(q_value, dtype=np.complex128))
        if p.shape[0] != p.shape[1] or q.shape != p.shape:
            raise ValueError("coordinate P/Q matrices must be aligned and square")
        matrix = np.eye(p.shape[0], dtype=np.complex128) + value * p
        inv_p = linalg.solve(matrix, p, assume_a="her", check_finite=False)
        inv_q = linalg.solve(matrix, q, assume_a="her", check_finite=False)
        derivative = (
            np.trace(inv_p).real
            - np.trace(inv_q).real
            + value * np.trace(inv_p @ inv_q).real
        )
        total += float(weights[frequency]) * int(count) * float(derivative)
    return float(total)


def rank1_single_frequency_optimum(p: float, q: float) -> float:
    if p <= 0.0 or q < 0.0:
        raise ValueError("p must be positive and q nonnegative")
    return float(max(0.0, (q - p) / (p * p)))


def optimize_coordinate(
    p_matrices: Sequence[ArrayLike],
    q_matrices: Sequence[ArrayLike],
    snapshots: Sequence[int],
    *,
    theta_max: float,
    frequency_weights: ArrayLike | None = None,
    bracket_points: int = 48,
) -> CoordinateOptimum:
    upper = float(theta_max)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("theta_max must be finite and positive")
    scalar_p = [np.asarray(value).reshape(-1) for value in p_matrices]
    scalar_q = [np.asarray(value).reshape(-1) for value in q_matrices]
    if all(left.size == right.size == 1 for left, right in zip(scalar_p, scalar_q)):
        p = np.asarray([float(np.real(value[0])) for value in scalar_p])
        q = np.asarray([float(np.real(value[0])) for value in scalar_q])
        counts = np.asarray(snapshots, dtype=float)
        weights = (
            np.ones(len(p), dtype=float)
            if frequency_weights is None
            else np.asarray(frequency_weights, dtype=float)
        )
        combined = weights * counts

        def scalar_delta(value: float) -> float:
            denominator = 1.0 + float(value) * p
            return float(
                np.sum(combined * (np.log(denominator) - float(value) * q / denominator))
            )

        def scalar_derivative(value: float) -> float:
            denominator = 1.0 + float(value) * p
            return float(np.sum(combined * (p / denominator - q / np.square(denominator))))

        derivative_zero = scalar_derivative(0.0)
        lower_positive = max(np.finfo(float).eps * upper, 1e-15)
        grid = np.concatenate(([0.0], np.geomspace(lower_positive, upper, int(bracket_points))))
        derivative = np.asarray([scalar_derivative(value) for value in grid])
        roots: list[float] = []
        for left, right, d_left, d_right in zip(grid[:-1], grid[1:], derivative[:-1], derivative[1:]):
            if d_left == 0.0 and left > 0.0:
                roots.append(float(left))
            if d_left * d_right < 0.0:
                roots.append(
                    float(optimize.brentq(scalar_derivative, float(left), float(right), xtol=1e-13, rtol=1e-13))
                )
        candidates = sorted(set([0.0, upper, *roots]))
        values = [scalar_delta(value) for value in candidates]
        best = int(np.argmin(values))
        return CoordinateOptimum(
            theta=float(candidates[best]),
            delta_objective=float(values[best]),
            derivative_at_zero=float(derivative_zero),
            stationary_points=tuple(roots),
        )
    derivative_zero = coordinate_delta_derivative(
        0.0, p_matrices, q_matrices, snapshots, frequency_weights=frequency_weights
    )
    lower_positive = max(np.finfo(float).eps * upper, 1e-15)
    grid = np.concatenate(([0.0], np.geomspace(lower_positive, upper, int(bracket_points))))
    derivative = np.asarray(
        [
            coordinate_delta_derivative(
                value, p_matrices, q_matrices, snapshots, frequency_weights=frequency_weights
            )
            for value in grid
        ]
    )
    roots: list[float] = []
    for left, right, d_left, d_right in zip(grid[:-1], grid[1:], derivative[:-1], derivative[1:]):
        if d_left == 0.0 and left > 0.0:
            roots.append(float(left))
        if d_left * d_right < 0.0:
            roots.append(
                float(
                    optimize.brentq(
                        lambda value: coordinate_delta_derivative(
                            value,
                            p_matrices,
                            q_matrices,
                            snapshots,
                            frequency_weights=frequency_weights,
                        ),
                        float(left),
                        float(right),
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                )
            )
    candidates = sorted(set([0.0, upper, *roots]))
    values = [
        coordinate_delta(
            value, p_matrices, q_matrices, snapshots, frequency_weights=frequency_weights
        )
        for value in candidates
    ]
    best = int(np.argmin(values))
    return CoordinateOptimum(
        theta=float(candidates[best]),
        delta_objective=float(values[best]),
        derivative_at_zero=float(derivative_zero),
        stationary_points=tuple(roots),
    )


@dataclass(frozen=True, slots=True)
class ActionRecord:
    iteration: int
    action: str
    component_id: str
    atom_id: str
    parent_id: str | None
    scale: int
    profile: str
    aree_mode: int | None
    old_theta: float
    new_theta: float
    derivative_at_zero: float
    predicted_gain: float
    realized_gain: float
    cholesky_status: str
    numerical_fallback: bool
    active_size: int
    elapsed_seconds: float


class SequentialEvidenceState:
    """Mutable reduced covariance state; production path stores no inverse."""

    def __init__(
        self,
        statistics: Sequence[FrequencySufficientStatistic],
        *,
        noise: ArrayLike,
        frequency_weights: ArrayLike | None = None,
    ) -> None:
        self.statistics = tuple(statistics)
        self.noise = np.asarray(noise, dtype=float).reshape(-1).copy()
        if self.noise.shape != (len(self.statistics),) or np.any(self.noise <= 0.0):
            raise ValueError("noise must be positive and frequency-aligned")
        self.frequency_weights = (
            np.ones(len(self.statistics), dtype=float)
            if frequency_weights is None
            else np.asarray(frequency_weights, dtype=float).reshape(-1)
        )
        if self.frequency_weights.shape != self.noise.shape:
            raise ValueError("frequency_weights must align")
        self.components: dict[str, ProjectedComponent] = {}
        self.theta: dict[str, float] = {}
        self.covariances = [
            self.noise[index] * np.eye(stat.quotient.effective_rank, dtype=np.complex128)
            for index, stat in enumerate(self.statistics)
        ]
        self.cholesky = [np.linalg.cholesky(value) for value in self.covariances]
        self.version = 0
        self.actions: list[ActionRecord] = []
        self.cholesky_rebuilds = 0
        self.downdate_failures = 0
        self.objective_history = [self.objective()]

    def objective(self) -> float:
        total = 0.0
        for frequency, statistic in enumerate(self.statistics):
            lower = self.cholesky[frequency]
            logdet = 2.0 * float(np.log(np.real(np.diag(lower))).sum())
            solved = linalg.cho_solve((lower, True), statistic.Z, check_finite=False)
            quadratic = float(np.real(np.vdot(statistic.Z, solved)))
            orthogonal = (
                statistic.snapshots
                * (statistic.sensor_dimension - statistic.quotient.effective_rank)
                * np.log(self.noise[frequency])
                + statistic.perpendicular_energy / self.noise[frequency]
            )
            total += self.frequency_weights[frequency] * (
                statistic.snapshots * logdet + quadratic + orthogonal
            )
        return float(total)

    def _cavity(self, component: ProjectedComponent) -> tuple[list[np.ndarray], list[np.ndarray]]:
        old = float(self.theta.get(component.component_id, 0.0))
        covariances: list[np.ndarray] = []
        factors: list[np.ndarray] = []
        for frequency, design in enumerate(component.reduced_factors):
            cavity = self.covariances[frequency] - old * (design @ design.conj().T)
            cavity = 0.5 * (cavity + cavity.conj().T)
            try:
                lower = np.linalg.cholesky(cavity)
            except np.linalg.LinAlgError as error:
                raise CholeskyUpdateError("component cavity is not positive definite") from error
            covariances.append(cavity)
            factors.append(lower)
        return covariances, factors

    def coordinate_optimum(self, component: ProjectedComponent, *, theta_max: float) -> CoordinateOptimum:
        _, cavity_factors = self._cavity(component)
        p_values = []
        q_values = []
        snapshots = []
        for frequency, (design, lower, statistic) in enumerate(
            zip(component.reduced_factors, cavity_factors, self.statistics)
        ):
            inv_design = linalg.cho_solve((lower, True), design, check_finite=False)
            p_values.append(design.conj().T @ inv_design)
            q_values.append(inv_design.conj().T @ statistic.S_z @ inv_design)
            snapshots.append(statistic.snapshots)
        return optimize_coordinate(
            p_values,
            q_values,
            snapshots,
            theta_max=theta_max,
            frequency_weights=self.frequency_weights,
        )

    def apply_coordinate(
        self,
        component: ProjectedComponent,
        optimum: CoordinateOptimum,
        *,
        objective_tolerance: float = 1e-9,
        minimum_gain: float = 1e-10,
    ) -> ActionRecord | None:
        started = perf_counter()
        old_theta = float(self.theta.get(component.component_id, 0.0))
        new_theta = float(optimum.theta)
        old_objective = self.objective_history[-1]
        delta = new_theta - old_theta
        if abs(delta) <= np.finfo(float).eps * max(1.0, old_theta, new_theta):
            return None
        proposed_covariances: list[np.ndarray] = []
        proposed_factors: list[np.ndarray] = []
        fallback = False
        downdate_failed = False
        for frequency, design in enumerate(component.reduced_factors):
            covariance = self.covariances[frequency].copy()
            lower = self.cholesky[frequency]
            try:
                for column in range(design.shape[1]):
                    vector = np.sqrt(abs(delta)) * design[:, column]
                    covariance = covariance + np.sign(delta) * np.outer(
                        vector, vector.conj()
                    )
                    covariance = np.asarray(0.5 * (covariance + covariance.conj().T))
                    update = safe_chol_rank1(
                        lower,
                        vector,
                        sign=1 if delta > 0 else -1,
                        covariance_for_rebuild=covariance,
                    )
                    lower = update.factor
                    fallback = fallback or update.used_rebuild
                    downdate_failed = downdate_failed or update.downdate_failed
            except CholeskyUpdateError:
                return None
            proposed_covariances.append(covariance)
            proposed_factors.append(lower)
        previous_covariances = self.covariances
        previous_factors = self.cholesky
        self.covariances = proposed_covariances
        self.cholesky = proposed_factors
        new_objective = self.objective()
        allowed = objective_tolerance * max(1.0, abs(old_objective))
        realized_gain = old_objective - new_objective
        if new_objective > old_objective + allowed or realized_gain < float(minimum_gain):
            self.covariances = previous_covariances
            self.cholesky = previous_factors
            return None
        if new_theta == 0.0:
            action = "delete"
            self.theta.pop(component.component_id, None)
            self.components.pop(component.component_id, None)
        elif old_theta == 0.0:
            action = "add"
            self.theta[component.component_id] = new_theta
            self.components[component.component_id] = component
        else:
            action = "reestimate"
            self.theta[component.component_id] = new_theta
        self.version += 1
        self.cholesky_rebuilds += int(fallback)
        self.downdate_failures += int(downdate_failed)
        self.objective_history.append(float(new_objective))
        record = ActionRecord(
            iteration=self.version,
            action=action,
            component_id=component.component_id,
            atom_id=component.atom_id,
            parent_id=component.parent_id,
            scale=component.scale,
            profile=component.profile_name,
            aree_mode=component.aree_mode,
            old_theta=old_theta,
            new_theta=new_theta,
            derivative_at_zero=optimum.derivative_at_zero,
            predicted_gain=-optimum.delta_objective,
            realized_gain=float(realized_gain),
            cholesky_status="REBUILD" if fallback else "RANK_UPDATE",
            numerical_fallback=fallback,
            active_size=len(self.theta),
            elapsed_seconds=perf_counter() - started,
        )
        self.actions.append(record)
        return record

    def derivative(self, component: ProjectedComponent) -> float:
        total = 0.0
        for frequency, (design, statistic, lower) in enumerate(
            zip(component.reduced_factors, self.statistics, self.cholesky)
        ):
            inv_design = linalg.cho_solve((lower, True), design, check_finite=False)
            inv_data = linalg.cho_solve((lower, True), statistic.Z, check_finite=False)
            first = float(np.trace(design.conj().T @ inv_design).real)
            second = float(
                np.linalg.norm(design.conj().T @ inv_data, "fro") ** 2
                / statistic.snapshots
            )
            total += self.frequency_weights[frequency] * statistic.snapshots * (first - second)
        return float(total)

    def update_noise(self, *, lower_scale: float = 1e-3, upper_scale: float = 1e3) -> bool:
        old_objective = self.objective_history[-1]
        candidates = self.noise.copy()
        for frequency, statistic in enumerate(self.statistics):
            source_covariance = self.covariances[frequency] - self.noise[frequency] * np.eye(
                statistic.quotient.effective_rank
            )

            def objective(log_noise: float) -> float:
                noise = float(np.exp(log_noise))
                covariance = source_covariance + noise * np.eye(source_covariance.shape[0])
                try:
                    lower = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError:
                    return float("inf")
                logdet = 2.0 * float(np.log(np.real(np.diag(lower))).sum())
                solved = linalg.cho_solve((lower, True), statistic.Z, check_finite=False)
                return float(
                    statistic.snapshots * logdet
                    + np.real(np.vdot(statistic.Z, solved))
                    + statistic.snapshots
                    * (statistic.sensor_dimension - statistic.quotient.effective_rank)
                    * np.log(noise)
                    + statistic.perpendicular_energy / noise
                )

            current = self.noise[frequency]
            result = optimize.minimize_scalar(
                objective,
                bounds=(np.log(current * lower_scale), np.log(current * upper_scale)),
                method="bounded",
                options={"xatol": 1e-10},
            )
            candidates[frequency] = float(np.exp(result.x))
        old_noise = self.noise.copy()
        old_covariances = self.covariances
        old_cholesky = self.cholesky
        self.noise = candidates
        self.covariances = [
            old_covariances[index]
            + (candidates[index] - old_noise[index]) * np.eye(old_covariances[index].shape[0])
            for index in range(len(candidates))
        ]
        try:
            self.cholesky = [np.linalg.cholesky(value) for value in self.covariances]
        except np.linalg.LinAlgError:
            self.noise = old_noise
            self.covariances = old_covariances
            self.cholesky = old_cholesky
            return False
        new_objective = self.objective()
        if new_objective > old_objective + 1e-9 * max(1.0, abs(old_objective)):
            self.noise = old_noise
            self.covariances = old_covariances
            self.cholesky = old_cholesky
            return False
        self.objective_history.append(float(new_objective))
        return True


__all__ = [
    "ActionRecord",
    "CoordinateOptimum",
    "SequentialEvidenceState",
    "coordinate_delta",
    "coordinate_delta_derivative",
    "optimize_coordinate",
    "rank1_single_frequency_optimum",
]
