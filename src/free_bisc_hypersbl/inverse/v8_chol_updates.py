"""Complex Hermitian Cholesky rank-one update/downdate primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]


class CholeskyUpdateError(np.linalg.LinAlgError):
    pass


@dataclass(frozen=True, slots=True)
class CholeskyUpdateResult:
    factor: ComplexArray
    used_rebuild: bool
    downdate_failed: bool


def chol_rank1(
    lower_factor: ArrayLike,
    vector: ArrayLike,
    *,
    sign: int = 1,
) -> ComplexArray:
    """Return ``chol(L L^H + sign*x*x^H)`` in O(n^2)."""

    if int(sign) not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    lower = np.array(lower_factor, dtype=np.complex128, copy=True)
    x = np.array(vector, dtype=np.complex128, copy=True).reshape(-1)
    if lower.ndim != 2 or lower.shape[0] != lower.shape[1] or x.shape != (lower.shape[0],):
        raise ValueError("factor/vector shape mismatch")
    if np.linalg.norm(np.triu(lower, 1)) > 1e-10 * max(np.linalg.norm(lower), 1.0):
        raise ValueError("lower_factor must be lower triangular")
    for index in range(len(x)):
        diagonal = float(np.real(lower[index, index]))
        radicand = diagonal * diagonal + int(sign) * float(abs(x[index]) ** 2)
        if not np.isfinite(radicand) or radicand <= np.finfo(float).tiny:
            raise CholeskyUpdateError("rank-one downdate is not positive definite")
        root = float(np.sqrt(radicand))
        c = root / diagonal
        s = x[index] / diagonal
        lower[index, index] = root
        if index + 1 < len(x):
            tail = lower[index + 1 :, index]
            x_tail = x[index + 1 :]
            updated_tail = (tail + int(sign) * x_tail * np.conjugate(s)) / c
            x[index + 1 :] = c * x_tail - s * updated_tail
            lower[index + 1 :, index] = updated_tail
    lower[np.triu_indices_from(lower, 1)] = 0.0
    return np.asarray(lower)


def safe_chol_rank1(
    lower_factor: ArrayLike,
    vector: ArrayLike,
    *,
    sign: int,
    covariance_for_rebuild: ArrayLike | None = None,
) -> CholeskyUpdateResult:
    try:
        factor = chol_rank1(lower_factor, vector, sign=sign)
        return CholeskyUpdateResult(factor, False, False)
    except CholeskyUpdateError:
        if covariance_for_rebuild is None:
            raise
        covariance = np.asarray(covariance_for_rebuild, dtype=np.complex128)
        try:
            factor = np.linalg.cholesky(0.5 * (covariance + covariance.conj().T))
        except np.linalg.LinAlgError as error:
            raise CholeskyUpdateError("downdate and covariance rebuild both failed") from error
        return CholeskyUpdateResult(np.asarray(factor), True, int(sign) < 0)


__all__ = [
    "CholeskyUpdateError",
    "CholeskyUpdateResult",
    "chol_rank1",
    "safe_chol_rank1",
]
