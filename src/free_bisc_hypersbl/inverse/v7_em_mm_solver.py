"""Proper-complex fixed-active-set EM/GEM solver for V7 FSOST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .v7_covariance_components import CovarianceComponent, type2_objective


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FrequencyLatentMoments:
    mean: ComplexArray
    covariance: ComplexArray
    slices: tuple[slice, ...]


@dataclass(frozen=True)
class EMResult:
    theta: NDArray[np.float64]
    noise: NDArray[np.float64]
    objective: tuple[float, ...]
    status: str
    iterations: int
    backtracks: int
    acceleration_accepts: int
    acceleration_fallbacks: int
    moments: tuple[FrequencyLatentMoments | None, ...]

    @property
    def converged(self) -> bool:
        return self.status == "CONVERGED_FIXED_ACTIVE_SET"


def _latent_moments(
    y_value: ArrayLike,
    components: Sequence[CovarianceComponent],
    theta: np.ndarray,
    noise: float,
    frequency: int,
) -> FrequencyLatentMoments:
    y = np.asarray(y_value, dtype=np.complex128)
    if y.ndim == 1:
        y = y[:, None]
    blocks = [component.sensor_factors[frequency] for component in components]
    design = np.column_stack(blocks)
    ranks = [block.shape[1] for block in blocks]
    edges = np.cumsum([0, *ranks])
    slices = tuple(slice(int(edges[i]), int(edges[i + 1])) for i in range(len(ranks)))
    gamma = np.concatenate([np.full(rank, theta[i]) for i, rank in enumerate(ranks)])
    ag = design * gamma[None, :]
    sigma = ag @ design.conj().T + noise * np.eye(design.shape[0])
    factor = linalg.cho_factor(0.5 * (sigma + sigma.conj().T), lower=True, check_finite=False)
    inv_y = linalg.cho_solve(factor, y, check_finite=False)
    inv_a = linalg.cho_solve(factor, design, check_finite=False)
    mean = gamma[:, None] * (design.conj().T @ inv_y)
    covariance = np.diag(gamma) - (gamma[:, None] * (design.conj().T @ inv_a)) * gamma[None, :]
    covariance = 0.5 * (covariance + covariance.conj().T)
    return FrequencyLatentMoments(np.asarray(mean), np.asarray(covariance), slices)


def fixed_active_em(
    observations: Sequence[ArrayLike | None],
    components: Sequence[CovarianceComponent],
    *,
    theta0: ArrayLike | None = None,
    noise0: ArrayLike | None = None,
    max_iter: int = 100,
    tol_obj: float = 1e-7,
    tol_theta: float = 1e-5,
    min_variance: float = 1e-12,
    acceleration: bool = True,
    frequency_weights: ArrayLike | None = None,
) -> EMResult:
    if not components:
        raise ValueError("components cannot be empty")
    frequency_count = len(components[0].sensor_factors)
    if len(observations) != frequency_count:
        raise ValueError("observations must align with component frequencies")
    observed = [i for i, y in enumerate(observations) if y is not None]
    if not observed:
        raise ValueError("at least one frequency is required")
    weights = (
        np.ones(frequency_count, dtype=float)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (frequency_count,) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("frequency weights must be finite, positive and aligned")
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("frequency weights must have mean one (omega=F*w)")
    power = np.mean([np.mean(np.abs(np.asarray(observations[i])) ** 2) for i in observed])
    power = max(float(power), min_variance)
    theta = np.full(len(components), power) if theta0 is None else np.asarray(theta0, dtype=float).copy()
    noise = np.full(frequency_count, 0.1 * power) if noise0 is None else np.asarray(noise0, dtype=float).copy()
    if theta.shape != (len(components),) or noise.shape != (frequency_count,):
        raise ValueError("initial parameter shape mismatch")
    theta = np.maximum(theta, min_variance)
    noise = np.maximum(noise, min_variance)
    objectives = [
        type2_objective(
            observations,
            components,
            theta,
            noise,
            frequency_weights=weights,
        )
    ]
    stable = 0
    backtracks = 0
    acceleration_accepts = 0
    acceleration_fallbacks = 0
    status = "MAX_ITER_NOT_CONVERGED"

    for _ in range(int(max_iter)):
        moments: list[FrequencyLatentMoments | None] = [None] * frequency_count
        theta_numerator = np.zeros(len(components))
        theta_denominator = np.zeros(len(components))
        proposed_noise = noise.copy()
        for f in observed:
            moment = _latent_moments(observations[f], components, theta, noise[f], f)
            moments[f] = moment
            y = np.asarray(observations[f], dtype=np.complex128)
            if y.ndim == 1:
                y = y[:, None]
            design = np.column_stack([component.sensor_factors[f] for component in components])
            for index, sl in enumerate(moment.slices):
                theta_numerator[index] += weights[f] * (
                    float(np.linalg.norm(moment.mean[sl], "fro") ** 2)
                    + y.shape[1] * float(np.trace(moment.covariance[sl, sl]).real)
                )
                theta_denominator[index] += weights[f] * components[index].rank * y.shape[1]
            residual = y - design @ moment.mean
            posterior_trace = float(np.trace(design @ moment.covariance @ design.conj().T).real)
            proposed_noise[f] = max(
                (float(np.linalg.norm(residual, "fro") ** 2) + y.shape[1] * posterior_trace)
                / (design.shape[0] * y.shape[1]), min_variance,
            )
        proposed_theta = np.maximum(theta_numerator / theta_denominator, min_variance)

        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        if acceleration:
            accelerated_theta = np.exp(np.log(proposed_theta) + 0.5 * (np.log(proposed_theta) - np.log(theta)))
            accelerated_noise = np.exp(np.log(proposed_noise) + 0.5 * (np.log(proposed_noise) - np.log(noise)))
            candidates.append(("accelerated", accelerated_theta, accelerated_noise))
        candidates.append(("em", proposed_theta, proposed_noise))
        accepted = None
        for name, theta_candidate, noise_candidate in candidates:
            fraction = 1.0
            while fraction >= 2.0**-16:
                trial_theta = np.exp((1-fraction)*np.log(theta)+fraction*np.log(theta_candidate))
                trial_noise = np.exp((1-fraction)*np.log(noise)+fraction*np.log(noise_candidate))
                value = type2_objective(
                    observations,
                    components,
                    trial_theta,
                    trial_noise,
                    frequency_weights=weights,
                )
                if value <= objectives[-1] + 1e-9 * max(1.0, abs(objectives[-1])):
                    accepted = (name, trial_theta, trial_noise, float(value))
                    break
                fraction *= 0.5
                backtracks += 1
            if accepted is not None:
                break
            if name == "accelerated":
                acceleration_fallbacks += 1
        if accepted is None:
            status = "OBJECTIVE_ACCEPTANCE_FAILED"
            break
        name, new_theta, new_noise, value = accepted
        if name == "accelerated":
            acceleration_accepts += 1
        relative_parameter = max(
            float(np.max(np.abs(new_theta-theta)/np.maximum(theta,min_variance))),
            float(np.max(np.abs(new_noise-noise)/np.maximum(noise,min_variance))),
        )
        relative_objective = abs(value-objectives[-1])/max(1.0,abs(objectives[-1]))
        theta, noise = new_theta, new_noise
        objectives.append(value)
        stable = stable + 1 if relative_parameter < tol_theta and relative_objective < tol_obj else 0
        if stable >= 3:
            status = "CONVERGED_FIXED_ACTIVE_SET"
            break

    final_moments = tuple(
        None if observations[f] is None else _latent_moments(observations[f], components, theta, noise[f], f)
        for f in range(frequency_count)
    )
    return EMResult(theta, noise, tuple(objectives), status, len(objectives)-1, backtracks, acceleration_accepts, acceleration_fallbacks, final_moments)


__all__ = ["FrequencyLatentMoments", "EMResult", "fixed_active_em"]
