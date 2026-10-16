"""Factorized V7 linear operator for the audited SHA-DGM family.

The historical project used ``Free-SIGM`` both for a forward-family narrative
and for frequency likelihood weights.  V7 keeps the two separate.  This module
wraps only the corrected analytic operator

    G_f = sum_(ell,m) u_(f,ell,m) c_(ell,m)^H,

which is algebraically equivalent to ``P_E S_M K_f^-1 b``.  It contains no
BEM/FEM construction and no empirical Gaussian-distance leadfield.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .global_matrix import (
    FrequencySphericalHarmonicModel,
    _solid_harmonic_and_gradient,
)


ComplexArray = NDArray[np.complex128]


def _readonly_complex(value: ArrayLike, *, ndim: int, name: str) -> ComplexArray:
    out = np.array(value, dtype=np.complex128, copy=True)
    if out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    out.setflags(write=False)
    return out


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode("ascii")
        + repr(array.shape).encode("ascii")
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class FreeSIGMFrequencyFactor:
    """One immutable frequency factor ``G = U @ Vh``."""

    frequency_hz: float
    U: ArrayLike
    Vh: ArrayLike
    l_max: int
    mode_labels: tuple[tuple[int, int], ...]
    reference_fingerprint: str
    physics_fingerprint: str

    def __post_init__(self) -> None:
        frequency = float(self.frequency_hz)
        u = _readonly_complex(self.U, ndim=2, name="U")
        vh = _readonly_complex(self.Vh, ndim=2, name="Vh")
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency_hz must be finite and nonnegative")
        if u.shape[1] != vh.shape[0]:
            raise ValueError("U and Vh have incompatible harmonic dimensions")
        if len(self.mode_labels) != u.shape[1]:
            raise ValueError("mode_labels must match the factor rank")
        expected = int(self.l_max) * (int(self.l_max) + 2)
        if u.shape[1] != expected:
            raise ValueError("factor rank does not match l_max*(l_max+2)")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "U", u)
        object.__setattr__(self, "Vh", vh)
        object.__setattr__(self, "l_max", int(self.l_max))

    @property
    def n_sensors(self) -> int:
        return int(np.asarray(self.U).shape[0])

    @property
    def n_sources(self) -> int:
        return int(np.asarray(self.Vh).shape[1])

    @property
    def harmonic_rank(self) -> int:
        return int(np.asarray(self.U).shape[1])

    def fingerprint(self) -> str:
        payload = {
            "frequency_hz_hex": float(self.frequency_hz).hex(),
            "l_max": self.l_max,
            "U": _hash_array(np.asarray(self.U)),
            "Vh": _hash_array(np.asarray(self.Vh)),
            "reference": self.reference_fingerprint,
            "physics": self.physics_fingerprint,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class FreeSIGMLinearOperator:
    """Frequency-indexed matrix-free operator with explicit small-test path."""

    def __init__(
        self,
        factors: Sequence[FreeSIGMFrequencyFactor],
        *,
        max_dense_entries: int = 2_000_000,
    ) -> None:
        if not factors:
            raise ValueError("at least one frequency factor is required")
        table: dict[str, FreeSIGMFrequencyFactor] = {}
        shape = (factors[0].n_sensors, factors[0].n_sources)
        for factor in factors:
            if (factor.n_sensors, factor.n_sources) != shape:
                raise ValueError("all frequencies must share sensor/source dimensions")
            key = float(factor.frequency_hz).hex()
            if key in table:
                raise ValueError("duplicate frequency factor")
            table[key] = factor
        self._factors = table
        self._max_dense_entries = int(max_dense_entries)
        if self._max_dense_entries <= 0:
            raise ValueError("max_dense_entries must be positive")

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        return tuple(f.frequency_hz for f in self._factors.values())

    @property
    def shape(self) -> tuple[int, int]:
        first = next(iter(self._factors.values()))
        return first.n_sensors, first.n_sources

    def _factor(self, frequency: float) -> FreeSIGMFrequencyFactor:
        key = float(frequency).hex()
        try:
            return self._factors[key]
        except KeyError as exc:
            raise KeyError(f"frequency {frequency!r} is not registered") from exc

    def apply(self, x: ArrayLike, frequency: float) -> ComplexArray:
        factor = self._factor(frequency)
        value = np.asarray(x, dtype=np.complex128)
        if value.ndim not in (1, 2) or value.shape[0] != factor.n_sources:
            raise ValueError("x must have shape (n_sources,) or (n_sources, n_rhs)")
        return np.asarray(factor.U @ (factor.Vh @ value), dtype=np.complex128)

    def adjoint(self, y: ArrayLike, frequency: float) -> ComplexArray:
        factor = self._factor(frequency)
        value = np.asarray(y, dtype=np.complex128)
        if value.ndim not in (1, 2) or value.shape[0] != factor.n_sensors:
            raise ValueError("y must have shape (n_sensors,) or (n_sensors, n_rhs)")
        return np.asarray(
            factor.Vh.conj().T @ (factor.U.conj().T @ value),
            dtype=np.complex128,
        )

    def apply_block(self, B: ArrayLike, frequency: float) -> ComplexArray:
        block = np.asarray(B, dtype=np.complex128)
        if block.ndim != 2:
            raise ValueError("B must have shape (n_sources, rank)")
        return self.apply(block, frequency)

    def materialize(self, frequency: float) -> ComplexArray:
        factor = self._factor(frequency)
        entries = factor.n_sensors * factor.n_sources
        if entries > self._max_dense_entries:
            raise MemoryError(
                f"dense materialization has {entries} entries; limit is "
                f"{self._max_dense_entries}"
            )
        return np.asarray(factor.U @ factor.Vh, dtype=np.complex128)

    def harmonic_rank(self, frequency: float) -> int:
        return self._factor(frequency).harmonic_rank

    def modal_factors(
        self, frequency: float
    ) -> tuple[ComplexArray, ComplexArray, tuple[tuple[int, int], ...]]:
        """Return immutable factors for harmonic-rank covariance algebra."""

        factor = self._factor(frequency)
        return factor.U, factor.Vh, factor.mode_labels

    def frequency_fingerprint(self, frequency: float) -> str:
        return self._factor(frequency).fingerprint()

    def factor_fingerprint(self) -> str:
        payload = [
            (factor.frequency_hz.hex(), factor.fingerprint())
            for factor in self._factors.values()
        ]
        return sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def build_freesigm_linear_operator(
    model: FrequencySphericalHarmonicModel,
    frequencies_hz: ArrayLike,
    source_xyz_m: ArrayLike,
    *,
    l_max: int,
    theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None = None,
    reference_matrix: ArrayLike | None = None,
    max_dense_entries: int = 2_000_000,
) -> FreeSIGMLinearOperator:
    """Build exact modal factors without materializing a leadfield family."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    sources = np.asarray(source_xyz_m, dtype=np.float64)
    if frequencies.size == 0 or not np.isfinite(frequencies).all() or np.any(frequencies < 0):
        raise ValueError("frequencies_hz must be finite and nonnegative")
    if len(np.unique(frequencies)) != frequencies.size:
        raise ValueError("frequencies_hz must be unique")
    if sources.ndim != 2 or sources.shape[1] != 3 or sources.shape[0] == 0:
        raise ValueError("source_xyz_m must have shape (n_sources, 3)")
    if not np.isfinite(sources).all():
        raise ValueError("source_xyz_m contains non-finite values")
    if np.any(np.linalg.norm(sources, axis=1) >= np.asarray(model.head.radii_m)[0]):
        raise ValueError("sources must lie inside the innermost boundary")
    l_max = int(l_max)
    if l_max < 1:
        raise ValueError("l_max must be at least one")

    reference: NDArray[np.float64] | None = None
    if reference_matrix is not None:
        reference = np.asarray(reference_matrix, dtype=np.float64)
        n_sensor = np.asarray(model.head.electrode_xyz_m).shape[0]
        if reference.ndim != 2 or reference.shape[1] != n_sensor:
            raise ValueError("reference_matrix has the wrong sensor dimension")
        reference_fp = _hash_array(reference.astype(np.complex128))
    else:
        reference_fp = "identity"

    source_dimensionless = sources / model.head.length_scale_m
    labels: list[tuple[int, int]] = []
    source_rows: list[ComplexArray] = []
    for ell in range(1, l_max + 1):
        for m in range(-ell, ell + 1):
            _, gradient = _solid_harmonic_and_gradient(
                source_dimensionless,
                ell=ell,
                m=m,
                radial_power=ell,
            )
            labels.append((ell, m))
            source_rows.append(np.conjugate(gradient).reshape(-1))
    vh = np.asarray(source_rows, dtype=np.complex128)

    physics_payload = {
        "radii": _hash_array(np.asarray(model.head.radii_m, dtype=np.complex128)),
        "conductivity": _hash_array(
            np.asarray(model.head.conductivity_s_per_m, dtype=np.complex128)
        ),
        "permittivity": _hash_array(
            np.asarray(model.head.relative_permittivity, dtype=np.complex128)
        ),
        "dispersion": _hash_array(
            np.asarray(model.head.dispersion_alpha, dtype=np.complex128)
        ),
        "electrodes": _hash_array(
            np.asarray(model.head.electrode_xyz_m, dtype=np.complex128)
        ),
        "sources": _hash_array(sources.astype(np.complex128)),
        "parameters": model.parameter_names,
        "theta": None if theta is None else repr(theta),
        "l_max": l_max,
    }
    physics_fp = sha256(
        json.dumps(physics_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    factors: list[FreeSIGMFrequencyFactor] = []
    for frequency in frequencies:
        columns: list[ComplexArray] = []
        for ell, m in labels:
            solution = model.solve(
                float(frequency),
                ell,
                m,
                source_coefficients=[1.0 + 0.0j],
                theta=theta,
            )
            column = solution.G[:, 0]
            columns.append(reference @ column if reference is not None else column)
        factors.append(
            FreeSIGMFrequencyFactor(
                frequency_hz=float(frequency),
                U=np.column_stack(columns),
                Vh=vh,
                l_max=l_max,
                mode_labels=tuple(labels),
                reference_fingerprint=reference_fp,
                physics_fingerprint=physics_fp,
            )
        )
    return FreeSIGMLinearOperator(factors, max_dense_entries=max_dense_entries)


__all__ = [
    "FreeSIGMFrequencyFactor",
    "FreeSIGMLinearOperator",
    "build_freesigm_linear_operator",
]
