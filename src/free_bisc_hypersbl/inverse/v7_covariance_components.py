"""Unified V7 sensor covariance components and Type-II objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .v7_operator_normalization import NormalizedAtom
from .v7_spectral_profiles import SpectralProfile


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class CovarianceComponent:
    component_id: str
    source_basis: ComplexArray
    profile: SpectralProfile
    sensor_factors: tuple[ComplexArray, ...]
    spatial_atom_id: str

    @property
    def rank(self) -> int:
        return int(self.source_basis.shape[1])


def build_covariance_component(
    normalized: NormalizedAtom,
    profile: SpectralProfile,
    operator,
    frequencies_hz: Sequence[float],
    *,
    whitening: Sequence[ArrayLike] | None = None,
) -> CovarianceComponent:
    if len(profile.rho) != len(frequencies_hz):
        raise ValueError("profile and frequency count mismatch")
    factors = []
    for index, frequency in enumerate(frequencies_hz):
        block = operator.apply_block(normalized.basis, frequency)
        if whitening is not None:
            block = np.asarray(whitening[index], dtype=np.complex128) @ block
        factors.append(np.asarray(np.sqrt(profile.rho[index]) * block, dtype=np.complex128))
    return CovarianceComponent(
        f"{normalized.atom.atom_id}|{profile.name}", normalized.basis, profile,
        tuple(factors), normalized.atom.atom_id,
    )


def assemble_sensor_covariances(
    components: Sequence[CovarianceComponent],
    theta: ArrayLike,
    noise: ArrayLike,
    *,
    tail_covariances: Sequence[ArrayLike] | None = None,
    eta: ArrayLike | None = None,
) -> tuple[ComplexArray, ...]:
    if not components:
        raise ValueError("at least one component is required")
    values = np.asarray(theta, dtype=float).reshape(-1)
    lambdas = np.asarray(noise, dtype=float).reshape(-1)
    count = len(components[0].sensor_factors)
    if values.shape != (len(components),) or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("theta must be finite, nonnegative and component-aligned")
    if lambdas.shape != (count,) or np.any(lambdas <= 0) or not np.isfinite(lambdas).all():
        raise ValueError("noise must be finite, positive and frequency-aligned")
    tail_values = np.zeros(count) if eta is None else np.asarray(eta, dtype=float)
    if tail_values.shape != (count,) or np.any(tail_values < 0):
        raise ValueError("eta must be nonnegative and frequency-aligned")
    output = []
    for frequency in range(count):
        sensor_count = components[0].sensor_factors[frequency].shape[0]
        sigma = lambdas[frequency] * np.eye(sensor_count, dtype=np.complex128)
        for value, component in zip(values, components):
            factor = component.sensor_factors[frequency]
            sigma += value * (factor @ factor.conj().T)
        if tail_covariances is not None:
            sigma += tail_values[frequency] * np.asarray(tail_covariances[frequency], dtype=np.complex128)
        output.append(np.asarray(0.5 * (sigma + sigma.conj().T), dtype=np.complex128))
    return tuple(output)


def type2_objective_from_sigma(
    observations: Sequence[ArrayLike | None],
    sigmas: Sequence[ArrayLike],
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    if len(observations) != len(sigmas):
        raise ValueError("observations and sigmas must align")
    weights = (
        np.ones(len(observations), dtype=float)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (len(observations),) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("frequency weights must be finite, positive and aligned")
    if not np.isclose(np.mean(weights), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("frequency likelihood weights must have mean one")
    total = 0.0
    used = 0
    for frequency, (y_value, sigma_value) in enumerate(zip(observations, sigmas)):
        if y_value is None:
            continue
        y = np.asarray(y_value, dtype=np.complex128)
        if y.ndim == 1:
            y = y[:, None]
        sigma = np.asarray(sigma_value, dtype=np.complex128)
        factor = linalg.cho_factor(sigma, lower=True, check_finite=False)
        logdet = 2.0 * float(np.log(np.real(np.diag(factor[0]))).sum())
        quadratic = float(np.real(np.vdot(y, linalg.cho_solve(factor, y, check_finite=False))))
        # A missing frequency retains its registered position and weight.
        total += weights[frequency] * (y.shape[1] * logdet + quadratic)
        used += 1
    if used == 0:
        raise ValueError("at least one frequency must be observed")
    return float(total)


def type2_objective(
    observations: Sequence[ArrayLike | None],
    components: Sequence[CovarianceComponent],
    theta: ArrayLike,
    noise: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    return type2_objective_from_sigma(
        observations,
        assemble_sensor_covariances(components, theta, noise),
        frequency_weights=frequency_weights,
    )


__all__ = ["CovarianceComponent", "build_covariance_component", "assemble_sensor_covariances", "type2_objective", "type2_objective_from_sigma"]
