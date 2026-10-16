"""Operator-observability normalization for V7 covariance atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .v7_spatial_atoms import SpatialAtom


@dataclass(frozen=True)
class NormalizedAtom:
    atom: SpatialAtom
    profile_name: str
    basis: NDArray[np.complex128]
    observability: float
    denominator: float
    capped: bool
    resolvable: bool
    mode: str


@dataclass(frozen=True)
class WhiteningResult:
    matrix: NDArray[np.complex128]
    effective_rank: int
    condition_number: float
    eigenvalue_floor: float


def stable_whitener(noise_covariance: ArrayLike, *, relative_floor: float = 1e-8) -> WhiteningResult:
    covariance=np.asarray(noise_covariance,dtype=np.complex128)
    if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1]:
        raise ValueError("noise covariance must be square")
    covariance=0.5*(covariance+covariance.conj().T)
    values,vectors=np.linalg.eigh(covariance)
    if values[-1]<=0:
        raise ValueError("noise covariance has no positive eigenvalue")
    floor=max(float(values[-1])*float(relative_floor),np.finfo(float).tiny)
    clipped=np.maximum(values,floor)
    matrix=(vectors*(1/np.sqrt(clipped))[None,:])@vectors.conj().T
    rank=int(np.sum(values>floor))
    return WhiteningResult(np.asarray(matrix),rank,float(clipped[-1]/clipped[0]),floor)


def normalize_atom_observability(
    atom: SpatialAtom,
    operator,
    frequencies_hz: Sequence[float],
    rho: ArrayLike,
    *,
    profile_name: str,
    whitening: Sequence[ArrayLike] | None = None,
    mode: str = "freesigm_observability",
    nu_floor: float = 1e-10,
) -> NormalizedAtom:
    weights = np.asarray(rho, dtype=float).reshape(-1)
    if len(weights) != len(frequencies_hz) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("rho must be nonnegative and frequency-aligned")
    floor = float(nu_floor)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("nu_floor must be positive")
    basis = np.asarray(atom.basis, dtype=np.complex128)
    selected = str(mode)
    if selected == "legacy_none":
        nu = float(np.linalg.norm(basis, "fro") ** 2 / atom.effective_rank)
        denominator = 1.0
    elif selected == "column_or_group_norm":
        nu = float(np.linalg.norm(basis, "fro") ** 2 / atom.effective_rank)
        denominator = np.sqrt(max(nu, floor))
    elif selected == "freesigm_observability":
        energy = 0.0
        for index, frequency in enumerate(frequencies_hz):
            block = operator.apply_block(basis, frequency)
            if whitening is not None:
                block = np.asarray(whitening[index], dtype=np.complex128) @ block
            energy += float(weights[index]) * float(np.linalg.norm(block, "fro") ** 2)
        nu = energy / (atom.effective_rank * float(weights.sum()))
        denominator = np.sqrt(max(nu, floor))
    else:
        raise ValueError("unknown normalization mode")
    capped = bool(nu < floor and selected != "legacy_none")
    normalized = np.asarray(basis / denominator, dtype=np.complex128)
    normalized.setflags(write=False)
    return NormalizedAtom(atom, profile_name, normalized, float(nu), float(denominator), capped, not capped, selected)


__all__ = ["NormalizedAtom", "WhiteningResult", "stable_whitener", "normalize_atom_observability"]
