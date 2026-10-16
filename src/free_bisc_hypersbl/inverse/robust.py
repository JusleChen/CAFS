"""Robust complex likelihood and Bayesian approximation error helpers."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .complex_sbl import ComplexGroupSBL, SBLResult


def student_t_irls_weights(residual: np.ndarray, noise: float, degrees_of_freedom: float = 4.0) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.ndim == 1:
        residual = residual[:, None]
    scaled = np.abs(residual) ** 2 / max(float(noise), 1e-12)
    return (degrees_of_freedom + 2.0) / (degrees_of_freedom + scaled)


def fit_student_t_group_sbl(
    a: np.ndarray,
    y: np.ndarray,
    *,
    group_size: int = 3,
    degrees_of_freedom: float = 4.0,
    outer_iterations: int = 8,
) -> SBLResult:
    """IRLS wrapper using proper-complex Student-t residual weights."""

    a = np.asarray(a, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    if y.ndim == 1:
        y = y[:, None]
    weights = np.ones_like(y.real)
    result: SBLResult | None = None
    for _ in range(outer_iterations):
        sensor_weight = np.sqrt(np.maximum(weights.mean(axis=1), 1e-6))
        result = ComplexGroupSBL(group_size=group_size, max_iter=80).fit(
            sensor_weight[:, None] * a,
            sensor_weight[:, None] * y,
            gamma0=None if result is None else result.gamma,
            noise0=None if result is None else result.noise,
        )
        residual = y - a @ result.mean
        updated = student_t_irls_weights(residual, result.noise, degrees_of_freedom)
        if np.max(np.abs(updated - weights)) < 1e-4:
            weights = updated
            break
        weights = 0.5 * weights + 0.5 * updated
    assert result is not None
    result.metadata.update({"likelihood": "proper-complex Student-t IRLS", "nu": degrees_of_freedom})
    return result


def bae_basis(
    residual_samples: np.ndarray,
    forward: np.ndarray,
    *,
    rank: int = 4,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an orthogonalized low-rank BAE subspace and covariance.

    ``residual_samples`` is sensor x samples.  The returned basis is projected
    through ``I - P_A`` before SVD so it cannot trivially absorb the modeled
    source subspace.
    """

    e = np.asarray(residual_samples, dtype=np.complex128)
    a = np.asarray(forward, dtype=np.complex128)
    # Use the numerical column space directly. A ridge-regularized normal-equation
    # projector is not an orthogonal projector when A is ill-conditioned and can
    # leave a BAE basis strongly aligned with the modeled source subspace.
    u_forward, singular, _ = linalg.svd(a, full_matrices=False, check_finite=False)
    tolerance = max(float(ridge), np.finfo(float).eps * max(a.shape) * singular[0])
    numerical_rank = int(np.sum(singular > tolerance))
    source_basis = u_forward[:, :numerical_rank]
    projector_action = source_basis @ (source_basis.conj().T @ e)
    orthogonal = e - projector_action
    u, s, _ = linalg.svd(orthogonal, full_matrices=False, check_finite=False)
    keep = min(rank, u.shape[1])
    basis = u[:, :keep]
    variance = (s[:keep] ** 2) / max(1, e.shape[1] - 1)
    return basis, np.diag(np.maximum(variance.real, 1e-12))
