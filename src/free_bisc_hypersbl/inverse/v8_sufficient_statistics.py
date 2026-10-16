"""Exact V8 full/reduced likelihood and sufficient-statistic references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .v8_harmonic_quotient import HarmonicQuotientFactor


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class FrequencySufficientStatistic:
    quotient: HarmonicQuotientFactor
    Z: ComplexArray
    S_z: ComplexArray
    perpendicular_energy: float
    snapshots: int
    sensor_dimension: int


def build_sufficient_statistic(
    quotient: HarmonicQuotientFactor,
    whitened_data: ArrayLike,
) -> FrequencySufficientStatistic:
    data = np.asarray(whitened_data, dtype=np.complex128)
    if data.ndim == 1:
        data = data[:, None]
    z, residual = quotient.project_data(data)
    snapshots = int(data.shape[1])
    s_z = z @ z.conj().T / snapshots
    return FrequencySufficientStatistic(
        quotient=quotient,
        Z=np.asarray(z),
        S_z=np.asarray(0.5 * (s_z + s_z.conj().T)),
        perpendicular_energy=float(np.linalg.norm(residual, "fro") ** 2),
        snapshots=snapshots,
        sensor_dimension=int(data.shape[0]),
    )


def covariance_from_components(
    component_factors: Sequence[ArrayLike],
    theta: ArrayLike,
    noise: float,
) -> ComplexArray:
    values = np.asarray(theta, dtype=float).reshape(-1)
    if len(component_factors) != len(values):
        raise ValueError("component_factors and theta must align")
    if not np.isfinite(noise) or noise <= 0.0 or np.any(values < 0.0):
        raise ValueError("noise must be positive and theta nonnegative")
    dimension = (
        np.asarray(component_factors[0]).shape[0]
        if component_factors
        else None
    )
    if dimension is None:
        raise ValueError("at least one component factor is required")
    covariance = float(noise) * np.eye(dimension, dtype=np.complex128)
    for value, factor_value in zip(values, component_factors):
        factor = np.asarray(factor_value, dtype=np.complex128)
        covariance += float(value) * (factor @ factor.conj().T)
    return np.asarray(0.5 * (covariance + covariance.conj().T))


def _covariance_likelihood(data: np.ndarray, covariance: np.ndarray) -> float:
    factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
    logdet = 2.0 * float(np.log(np.real(np.diag(factor[0]))).sum())
    quadratic = float(
        np.real(np.vdot(data, linalg.cho_solve(factor, data, check_finite=False)))
    )
    return float(data.shape[1] * logdet + quadratic)


def full_negative_log_evidence(
    whitened_data: Sequence[ArrayLike],
    full_component_factors: Sequence[Sequence[ArrayLike]],
    theta: ArrayLike,
    noise: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    values = np.asarray(theta, dtype=float).reshape(-1)
    lambdas = np.asarray(noise, dtype=float).reshape(-1)
    count = len(whitened_data)
    weights = np.ones(count) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    if lambdas.shape != (count,) or weights.shape != (count,):
        raise ValueError("noise/weights must align with frequencies")
    total = 0.0
    for frequency, data_value in enumerate(whitened_data):
        data = np.asarray(data_value, dtype=np.complex128)
        if data.ndim == 1:
            data = data[:, None]
        factors = [component[frequency] for component in full_component_factors]
        covariance = covariance_from_components(factors, values, lambdas[frequency])
        total += float(weights[frequency]) * _covariance_likelihood(data, covariance)
    return float(total)


def reduced_negative_log_evidence(
    statistics: Sequence[FrequencySufficientStatistic],
    reduced_component_factors: Sequence[Sequence[ArrayLike]],
    theta: ArrayLike,
    noise: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    values = np.asarray(theta, dtype=float).reshape(-1)
    lambdas = np.asarray(noise, dtype=float).reshape(-1)
    count = len(statistics)
    weights = np.ones(count) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    if lambdas.shape != (count,) or weights.shape != (count,):
        raise ValueError("noise/weights must align with frequencies")
    total = 0.0
    for frequency, statistic in enumerate(statistics):
        factors = [component[frequency] for component in reduced_component_factors]
        covariance = covariance_from_components(factors, values, lambdas[frequency])
        reduced = _covariance_likelihood(statistic.Z, covariance)
        orthogonal = (
            statistic.snapshots
            * (statistic.sensor_dimension - statistic.quotient.effective_rank)
            * np.log(lambdas[frequency])
            + statistic.perpendicular_energy / lambdas[frequency]
        )
        total += float(weights[frequency]) * (reduced + orthogonal)
    return float(total)


def reduced_directional_derivatives(
    statistics: Sequence[FrequencySufficientStatistic],
    covariances: Sequence[ArrayLike],
    candidate_factors: Sequence[Sequence[ArrayLike]],
    *,
    frequency_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    count = len(statistics)
    weights = np.ones(count) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    derivatives = np.zeros(len(candidate_factors), dtype=float)
    for frequency, (statistic, covariance_value) in enumerate(zip(statistics, covariances)):
        covariance = np.asarray(covariance_value, dtype=np.complex128)
        chol = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inv_z = linalg.cho_solve(chol, statistic.Z, check_finite=False)
        for index, component in enumerate(candidate_factors):
            design = np.asarray(component[frequency], dtype=np.complex128)
            inv_design = linalg.cho_solve(chol, design, check_finite=False)
            first = float(np.trace(design.conj().T @ inv_design).real)
            second = float(
                np.linalg.norm(design.conj().T @ inv_z, "fro") ** 2
                / statistic.snapshots
            )
            derivatives[index] += (
                float(weights[frequency]) * statistic.snapshots * (first - second)
            )
    return derivatives


__all__ = [
    "FrequencySufficientStatistic",
    "build_sufficient_statistic",
    "covariance_from_components",
    "full_negative_log_evidence",
    "reduced_directional_derivatives",
    "reduced_negative_log_evidence",
]
