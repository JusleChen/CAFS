"""Vectorized harmonic-bank and spectral-profile screening."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg


def vectorized_frequency_base_scores(
    projected_bank: ArrayLike,
    reduced_factor: ArrayLike,
    covariance: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Return zero-point score for every rank-one atom at one frequency."""

    bank = np.asarray(projected_bank, dtype=np.complex128)
    r = np.asarray(reduced_factor, dtype=np.complex128)
    c = np.asarray(covariance, dtype=np.complex128)
    data = np.asarray(z, dtype=np.complex128)
    if bank.ndim != 2 or r.ndim != 2 or r.shape[1] != bank.shape[0]:
        raise ValueError("projected bank and reduced factor do not align")
    if data.ndim == 1:
        data = data[:, None]
    design = r @ bank
    chol = linalg.cho_factor(c, lower=True, check_finite=False)
    inv_design = linalg.cho_solve(chol, design, check_finite=False)
    inv_data = linalg.cho_solve(chol, data, check_finite=False)
    first = np.real(np.sum(np.conjugate(design) * inv_design, axis=0))
    projections = design.conj().T @ inv_data
    second = np.sum(np.abs(projections) ** 2, axis=1) / data.shape[1]
    return np.asarray(data.shape[1] * (first - second), dtype=np.float64)


def profile_derivative_matrix(
    frequency_base_scores: ArrayLike,
    profile_matrix: ArrayLike,
    *,
    observability_denominators: ArrayLike | None = None,
    frequency_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Combine ``atom x frequency`` scores with ``profile x frequency`` rho."""

    scores = np.asarray(frequency_base_scores, dtype=float)
    rho = np.asarray(profile_matrix, dtype=float)
    if scores.ndim != 2 or rho.ndim != 2 or scores.shape[1] != rho.shape[1]:
        raise ValueError("scores and profile_matrix must share frequency dimension")
    weights = np.ones(scores.shape[1]) if frequency_weights is None else np.asarray(frequency_weights, dtype=float)
    if weights.shape != (scores.shape[1],):
        raise ValueError("frequency_weights must align")
    derivatives = (scores * weights[None, :]) @ rho.T
    if observability_denominators is not None:
        denominator = np.asarray(observability_denominators, dtype=float)
        if denominator.shape != derivatives.shape or np.any(denominator <= 0.0):
            raise ValueError("observability denominators must be positive atom x profile")
        derivatives = derivatives / np.square(denominator)
    return np.asarray(derivatives, dtype=np.float64)


def profile_envelope(derivatives: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    values = np.asarray(derivatives, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("derivatives must be atom x profile")
    index = np.argmin(values, axis=1)
    return values[np.arange(len(values)), index], np.asarray(index, dtype=np.int64)


__all__ = [
    "profile_derivative_matrix",
    "profile_envelope",
    "vectorized_frequency_base_scores",
]
