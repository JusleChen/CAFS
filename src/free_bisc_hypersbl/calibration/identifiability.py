"""Physical-parameter information after nuisance source projection."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def schur_complement(
    information_theta_theta: np.ndarray,
    information_theta_source: np.ndarray,
    information_source_source: np.ndarray,
    *,
    ridge: float = 1e-10,
) -> np.ndarray:
    itt = np.asarray(information_theta_theta, dtype=float)
    itj = np.asarray(information_theta_source, dtype=float)
    ijj = np.asarray(information_source_source, dtype=float)
    solved = linalg.solve(
        ijj + ridge * np.eye(ijj.shape[0]), itj.T, assume_a="sym", check_finite=False
    )
    out = itt - itj @ solved
    return (out + out.T) * 0.5


def effective_information(
    leadfield: np.ndarray,
    sensitivities: np.ndarray,
    *,
    source_precision: np.ndarray | float,
    noise_covariance: np.ndarray,
    prior_precision: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``2 Re[S^H R^-1/2 P_G^perp R^-1/2 S]``.

    ``sensitivities`` has shape sensors x physical_parameters and already
    includes multiplication by the current source estimate.
    """

    g = np.asarray(leadfield, dtype=np.complex128)
    s = np.asarray(sensitivities, dtype=np.complex128)
    r = np.asarray(noise_covariance, dtype=np.complex128)
    chol = linalg.cho_factor((r + r.conj().T) * 0.5, lower=True, check_finite=False)
    gw = linalg.solve_triangular(chol[0], g, lower=True, check_finite=False)
    sw = linalg.solve_triangular(chol[0], s, lower=True, check_finite=False)
    if np.isscalar(source_precision):
        precision = float(source_precision) * np.eye(g.shape[1])
    else:
        precision = np.asarray(source_precision, dtype=np.complex128)
    gram = gw.conj().T @ gw + precision
    fitted = gw @ linalg.solve(gram, gw.conj().T @ sw, assume_a="her", check_finite=False)
    residual = sw - fitted
    info = 2.0 * np.real(residual.conj().T @ residual)
    if prior_precision is not None:
        info = info + np.asarray(prior_precision, dtype=float)
    return (info + info.T) * 0.5


def stabilize_information(
    information: np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regularized information and the eigenmodes that required freezing."""

    matrix = np.asarray(information, dtype=float)
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    frozen = values < minimum_eigenvalue
    adjusted = np.maximum(values, minimum_eigenvalue)
    stabilized = (vectors * adjusted[None, :]) @ vectors.T
    return (stabilized + stabilized.T) * 0.5, frozen

