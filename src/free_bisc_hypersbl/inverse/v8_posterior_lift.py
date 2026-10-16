"""Exact matrix-free source posterior lift from the harmonic quotient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .v8_sequential_evidence import SequentialEvidenceState


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class V8SourcePosterior:
    means: tuple[ComplexArray, ...]
    variance_diagonals: tuple[NDArray[np.float64], ...]
    mean_only_power: NDArray[np.float64]
    second_moment_power: NDArray[np.float64]
    normalized_mean_only_power: NDArray[np.float64]
    normalized_second_moment_power: NDArray[np.float64]
    component_mean_relative_error: float
    full_posterior_covariance_allocated: bool = False


def _normalize_power(value: np.ndarray) -> np.ndarray:
    total = float(np.sum(value))
    if total <= 0.0:
        return np.full(len(value), 1.0 / len(value), dtype=float)
    return np.asarray(value / total, dtype=float)


def exact_source_posterior_lift(
    state: SequentialEvidenceState,
    vh: ArrayLike,
    profile_matrix: ArrayLike,
    *,
    group_size: int = 3,
) -> V8SourcePosterior:
    """Compute exact means and marginal variances without a P-by-P matrix."""

    harmonic_map = np.asarray(vh, dtype=np.complex128)
    rho = np.asarray(profile_matrix, dtype=float)
    if harmonic_map.ndim != 2:
        raise ValueError("Vh must be harmonic-by-source")
    if rho.ndim != 2 or rho.shape[1] != len(state.statistics):
        raise ValueError("profile matrix does not align with frequencies")
    source_dimension = int(harmonic_map.shape[1])
    size = int(group_size)
    if size < 1 or source_dimension % size:
        raise ValueError("source dimension must divide into source groups")
    groups = source_dimension // size
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    mean_power = np.zeros(groups, dtype=float)
    second_power = np.zeros(groups, dtype=float)
    component_error = 0.0
    active = tuple(state.components.items())
    for frequency, (statistic, quotient, covariance, lower) in enumerate(
        zip(state.statistics, (s.quotient for s in state.statistics), state.covariances, state.cholesky)
    ):
        j_action = np.zeros(
            (source_dimension, harmonic_map.shape[0]), dtype=np.complex128
        )
        prior_diagonal = np.zeros(source_dimension, dtype=float)
        component_actions: list[tuple[np.ndarray, np.ndarray]] = []
        for component_id, component in active:
            theta = float(state.theta[component_id])
            spectral = float(rho[component.profile_index, frequency])
            basis = component.source_basis
            harmonic_basis = harmonic_map @ basis
            action = theta * spectral * basis @ harmonic_basis.conj().T
            j_action += action
            prior_diagonal += theta * spectral * np.sum(np.abs(basis) ** 2, axis=1)
            component_actions.append((action, basis))
        inverse_data = linalg.cho_solve((lower, True), statistic.Z, check_finite=False)
        source_sensor = j_action @ quotient.R.conj().T
        mean = source_sensor @ inverse_data
        solved_rows = linalg.cho_solve(
            (lower, True), source_sensor.conj().T, check_finite=False
        )
        correction = np.real(np.einsum("ij,ji->i", source_sensor, solved_rows))
        variance = np.asarray(prior_diagonal - correction, dtype=float)
        scale = max(float(np.max(prior_diagonal, initial=0.0)), 1.0)
        if float(np.min(variance, initial=0.0)) < -1e-9 * scale:
            raise FloatingPointError("posterior marginal variance is materially negative")
        variance = np.maximum(variance, 0.0)
        component_sum = np.zeros_like(mean)
        for action, _ in component_actions:
            component_sum += action @ quotient.R.conj().T @ inverse_data
        component_error = max(
            component_error,
            float(np.linalg.norm(mean - component_sum) / max(np.linalg.norm(mean), 1e-300)),
        )
        grouped_mean = np.sum(
            np.sum(np.abs(mean.reshape(groups, size, mean.shape[1])) ** 2, axis=2),
            axis=1,
        )
        grouped_variance = statistic.snapshots * np.sum(
            variance.reshape(groups, size), axis=1
        )
        mean_power += grouped_mean
        second_power += grouped_mean + grouped_variance
        means.append(np.asarray(mean, dtype=np.complex128))
        variances.append(variance)
    return V8SourcePosterior(
        means=tuple(means),
        variance_diagonals=tuple(variances),
        mean_only_power=mean_power,
        second_moment_power=second_power,
        normalized_mean_only_power=_normalize_power(mean_power),
        normalized_second_moment_power=_normalize_power(second_power),
        component_mean_relative_error=float(component_error),
    )


__all__ = ["V8SourcePosterior", "exact_source_posterior_lift"]
