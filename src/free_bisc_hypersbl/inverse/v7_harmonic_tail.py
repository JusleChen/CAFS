"""Conditional high-order harmonic-tail covariance for V7."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def harmonic_tail_covariance(
    base_operator,
    extended_operator,
    frequency_hz: float,
) -> NDArray[np.complex128]:
    if base_operator.shape != extended_operator.shape:
        raise ValueError("base and extended operators must share dimensions")
    if extended_operator.harmonic_rank(frequency_hz) <= base_operator.harmonic_rank(frequency_hz):
        raise ValueError("extended operator must contain legal higher modes")
    base_u, base_vh, base_labels = base_operator.modal_factors(frequency_hz)
    extended_u, extended_vh, extended_labels = extended_operator.modal_factors(
        frequency_hz
    )
    if tuple(extended_labels[: len(base_labels)]) != tuple(base_labels):
        raise ValueError("extended operator does not preserve base modal ordering")
    if not np.allclose(
        extended_u[:, : len(base_labels)], base_u, rtol=2e-12, atol=2e-12
    ) or not np.allclose(
        extended_vh[: len(base_labels)], base_vh, rtol=2e-12, atol=2e-12
    ):
        raise ValueError("extended operator changes the retained base modes")
    tail_u = extended_u[:, len(base_labels) :]
    tail_vh = extended_vh[len(base_labels) :]
    # G_tail G_tail^H in harmonic-rank space; no dense M x P leadfield.
    covariance = tail_u @ (tail_vh @ tail_vh.conj().T) @ tail_u.conj().T
    trace=float(np.trace(covariance).real)
    if trace<=0 or not np.isfinite(trace):
        raise ValueError("harmonic tail has zero or invalid energy")
    return np.asarray(0.5*(covariance+covariance.conj().T)/trace,dtype=np.complex128)


def add_tail_exact(sigma: ArrayLike, tail: ArrayLike, eta: float) -> NDArray[np.complex128]:
    value=float(eta)
    if value<0 or not np.isfinite(value):
        raise ValueError("eta must be finite and nonnegative")
    return np.asarray(sigma,dtype=np.complex128)+value*np.asarray(tail,dtype=np.complex128)


__all__=["harmonic_tail_covariance","add_tail_exact"]
