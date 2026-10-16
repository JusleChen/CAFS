"""Exact Free-SIGM harmonic quotient construction for V8 HQSL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg


ComplexArray = NDArray[np.complex128]


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode("ascii")
        + repr(array.shape).encode("ascii")
        + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class EffectiveWhitener:
    """Whitening map into the covariance's numerical range."""

    matrix: ComplexArray
    effective_rank: int
    eigenvalues: NDArray[np.float64]
    threshold: float
    condition_number: float


def build_effective_whitener(
    noise_covariance: ArrayLike,
    *,
    relative_tolerance: float | None = None,
) -> EffectiveWhitener:
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("noise covariance must be square")
    covariance = 0.5 * (covariance + covariance.conj().T)
    values, vectors = np.linalg.eigh(covariance)
    largest = float(values[-1])
    if not np.isfinite(values).all() or largest <= 0.0:
        raise ValueError("noise covariance has no finite positive eigenspace")
    relative = (
        np.finfo(float).eps * max(covariance.shape)
        if relative_tolerance is None
        else float(relative_tolerance)
    )
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    threshold = largest * relative
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("noise covariance effective rank is zero")
    retained = np.asarray(values[keep], dtype=np.float64)
    matrix = (vectors[:, keep] / np.sqrt(retained)[None, :]).conj().T
    matrix = np.asarray(matrix, dtype=np.complex128)
    return EffectiveWhitener(
        matrix=matrix,
        effective_rank=int(keep.sum()),
        eigenvalues=retained,
        threshold=float(threshold),
        condition_number=float(retained[-1] / retained[0]),
    )


@dataclass(frozen=True, slots=True)
class HarmonicQuotientFactor:
    """Exact factor ``bar_U = Q R`` after numerical null-mode removal."""

    Q: ComplexArray
    R: ComplexArray
    singular_values: NDArray[np.float64]
    effective_rank: int
    sensor_dimension: int
    harmonic_dimension: int
    method: str
    rank_threshold: float
    fingerprint: str

    def project_data(self, whitened_data: ArrayLike) -> tuple[ComplexArray, ComplexArray]:
        data = np.asarray(whitened_data, dtype=np.complex128)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] != self.sensor_dimension:
            raise ValueError("whitened_data has the wrong sensor dimension")
        z = self.Q.conj().T @ data
        residual = data - self.Q @ z
        return np.asarray(z), np.asarray(residual)


def build_harmonic_quotient(
    sensor_factor: ArrayLike,
    *,
    whitening: ArrayLike | None = None,
    method: Literal["svd", "qr"] = "svd",
    relative_rank_tolerance: float | None = None,
) -> HarmonicQuotientFactor:
    u = np.asarray(sensor_factor, dtype=np.complex128)
    if u.ndim != 2 or not np.isfinite(u).all():
        raise ValueError("sensor_factor must be a finite matrix")
    if whitening is not None:
        w = np.asarray(whitening, dtype=np.complex128)
        if w.ndim != 2 or w.shape[1] != u.shape[0]:
            raise ValueError("whitening and sensor_factor are incompatible")
        u = w @ u
    if method == "svd":
        left, singular, right_h = np.linalg.svd(u, full_matrices=False)
    elif method == "qr":
        q_full, r_full, _ = linalg.qr(u, mode="economic", pivoting=True)
        singular = np.linalg.svd(u, compute_uv=False)
        left = q_full
        right_h = r_full  # rank only; R is reconstructed below in original order.
    else:
        raise ValueError("method must be 'svd' or 'qr'")
    if singular.size == 0 or float(singular[0]) <= 0.0:
        raise ValueError("whitened Free-SIGM sensor factor has zero rank")
    relative = (
        np.finfo(float).eps * max(u.shape)
        if relative_rank_tolerance is None
        else float(relative_rank_tolerance)
    )
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("relative_rank_tolerance must be finite and positive")
    threshold = float(singular[0]) * relative
    rank = int(np.sum(singular > threshold))
    if rank < 1:
        raise ValueError("whitened Free-SIGM sensor factor has zero effective rank")
    if method == "svd":
        q = left[:, :rank]
        r = singular[:rank, None] * right_h[:rank]
    else:
        q = left[:, :rank]
        r = q.conj().T @ u
    q = np.asarray(q, dtype=np.complex128)
    r = np.asarray(r, dtype=np.complex128)
    reconstruction_error = np.linalg.norm(u - q @ r) / max(np.linalg.norm(u), 1e-300)
    if reconstruction_error > 1e-10:
        raise RuntimeError("rank-revealing quotient failed exact reconstruction tolerance")
    orthogonality_error = np.linalg.norm(q.conj().T @ q - np.eye(rank))
    if orthogonality_error > 1e-10:
        raise RuntimeError("quotient basis is not orthonormal")
    payload = {
        "Q": _array_digest(q),
        "R": _array_digest(r),
        "singular": [float(value).hex() for value in singular[:rank]],
        "method": method,
        "threshold": float(threshold).hex(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    q.setflags(write=False)
    r.setflags(write=False)
    retained = np.asarray(singular[:rank], dtype=np.float64)
    retained.setflags(write=False)
    return HarmonicQuotientFactor(
        Q=q,
        R=r,
        singular_values=retained,
        effective_rank=rank,
        sensor_dimension=int(u.shape[0]),
        harmonic_dimension=int(u.shape[1]),
        method=method,
        rank_threshold=float(threshold),
        fingerprint=fingerprint,
    )


def quotient_from_freesigm_operator(
    operator,
    frequency: float,
    *,
    whitening: ArrayLike | None = None,
    method: Literal["svd", "qr"] = "svd",
    relative_rank_tolerance: float | None = None,
) -> tuple[HarmonicQuotientFactor, ComplexArray]:
    u, vh, _ = operator.modal_factors(frequency)
    quotient = build_harmonic_quotient(
        u,
        whitening=whitening,
        method=method,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    return quotient, np.asarray(vh, dtype=np.complex128)


__all__ = [
    "EffectiveWhitener",
    "HarmonicQuotientFactor",
    "build_effective_whitener",
    "build_harmonic_quotient",
    "quotient_from_freesigm_operator",
]
