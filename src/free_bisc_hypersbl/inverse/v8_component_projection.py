"""Packed V7-Lite atom projection into the V8 harmonic quotient."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .v7_spatial_atoms import SourceGraph, SpatialAtom, build_spatial_atom_bank
from .v7_spectral_profiles import SpectralProfile, build_spectral_profile_bank
from .v8_harmonic_quotient import HarmonicQuotientFactor


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ProjectedComponent:
    component_id: str
    atom_index: int
    profile_index: int
    atom_id: str
    parent_id: str | None
    center: int
    scale: int
    kind: str
    profile_name: str
    profile_kind: str
    source_basis: ComplexArray
    reduced_factors: tuple[ComplexArray, ...]
    support: tuple[int, ...]
    observability: float
    aree_mode: int | None = None

    @property
    def rank(self) -> int:
        return int(self.source_basis.shape[1])


@dataclass(frozen=True, slots=True)
class ProjectionBank:
    atoms: tuple[SpatialAtom, ...]
    profiles: tuple[SpectralProfile, ...]
    source_bases: ComplexArray
    projected_bases: ComplexArray
    profile_matrix: NDArray[np.float64]
    observability: NDArray[np.float64]
    denominators: NDArray[np.float64]
    valid: NDArray[np.bool_]
    quotient_factors: tuple[HarmonicQuotientFactor, ...]
    fingerprint: str

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def profile_count(self) -> int:
        return len(self.profiles)

    def component(self, atom_index: int, profile_index: int) -> ProjectedComponent:
        atom_position = int(atom_index)
        profile_position = int(profile_index)
        if not 0 <= atom_position < self.atom_count or not 0 <= profile_position < self.profile_count:
            raise IndexError("atom/profile index outside projection bank")
        if not bool(self.valid[atom_position, profile_position]):
            raise ValueError("component is below the observability floor")
        atom = self.atoms[atom_position]
        profile = self.profiles[profile_position]
        denominator = float(self.denominators[atom_position, profile_position])
        source_basis = np.asarray(atom.basis / denominator, dtype=np.complex128)
        harmonic = self.projected_bases[:, atom_position : atom_position + 1] / denominator
        factors = tuple(
            np.asarray(
                np.sqrt(profile.rho[frequency]) * quotient.R @ harmonic,
                dtype=np.complex128,
            )
            for frequency, quotient in enumerate(self.quotient_factors)
        )
        return ProjectedComponent(
            component_id=f"{atom.atom_id}|{profile.name}",
            atom_index=atom_position,
            profile_index=profile_position,
            atom_id=atom.atom_id,
            parent_id=None,
            center=atom.center,
            scale=atom.scale,
            kind=atom.kind,
            profile_name=profile.name,
            profile_kind=profile.kind,
            source_basis=source_basis,
            reduced_factors=factors,
            support=atom.support,
            observability=float(self.observability[atom_position, profile_position]),
        )

    def valid_component_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(left), int(right)) for left, right in np.argwhere(self.valid))


def build_projection_bank(
    graph: SourceGraph,
    vh: ArrayLike,
    quotient_factors: Sequence[HarmonicQuotientFactor],
    frequencies_hz: Sequence[float],
    *,
    include_smooth_profiles: bool = False,
    specific_floor: float = 0.05,
    normalization_mode: str = "freesigm_observability",
    observability_floor: float = 1e-12,
    atom_kinds: Sequence[str] | None = None,
    profile_kinds: Sequence[str] | None = None,
) -> ProjectionBank:
    atoms = tuple(build_spatial_atom_bank(graph, include_lowrank=False))
    if atom_kinds is not None:
        allowed_atoms = frozenset(map(str, atom_kinds))
        atoms = tuple(atom for atom in atoms if atom.kind in allowed_atoms)
        if not atoms:
            raise ValueError("atom_kinds removed the complete spatial dictionary")
    if any(atom.effective_rank != 1 for atom in atoms):
        raise RuntimeError("V8 base bank requires rank-one V7-Lite atoms")
    profiles = tuple(
        build_spectral_profile_bank(
            frequencies_hz,
            specific_floor=specific_floor,
            include_smooth=include_smooth_profiles,
        )
    )
    if profile_kinds is not None:
        allowed_profiles = frozenset(map(str, profile_kinds))
        profiles = tuple(profile for profile in profiles if profile.kind in allowed_profiles)
        if not profiles:
            raise ValueError("profile_kinds removed the complete spectral dictionary")
    if len(quotient_factors) != len(frequencies_hz):
        raise ValueError("quotient factors and frequencies must align")
    source_bases = np.column_stack([atom.basis[:, 0] for atom in atoms])
    harmonic_map = np.asarray(vh, dtype=np.complex128)
    if harmonic_map.ndim != 2 or harmonic_map.shape[1] != source_bases.shape[0]:
        raise ValueError("Vh and source atom dimension do not align")
    projected = harmonic_map @ source_bases
    energy = np.column_stack(
        [
            np.sum(np.abs(quotient.R @ projected) ** 2, axis=0)
            for quotient in quotient_factors
        ]
    )
    rho = np.vstack([profile.rho for profile in profiles])
    if normalization_mode == "freesigm_observability":
        observability = energy @ rho.T / rho.sum(axis=1)[None, :]
    elif normalization_mode == "column_or_group_norm":
        source_norm = np.sum(np.abs(source_bases) ** 2, axis=0)
        observability = np.repeat(source_norm[:, None], len(profiles), axis=1)
    elif normalization_mode == "legacy_none":
        observability = np.ones((len(atoms), len(profiles)))
    else:
        raise ValueError("unknown normalization mode")
    valid = observability >= float(observability_floor)
    denominators = (
        np.ones_like(observability)
        if normalization_mode == "legacy_none"
        else np.sqrt(np.maximum(observability, float(observability_floor)))
    )
    payload = {
        "atoms": [atom.atom_id for atom in atoms],
        "profiles": [profile.name for profile in profiles],
        "graph": graph.fingerprint,
        "projected": hashlib.sha256(np.ascontiguousarray(projected).view(np.uint8)).hexdigest(),
        "normalization": normalization_mode,
        "atom_kinds": sorted({atom.kind for atom in atoms}),
        "profile_kinds": sorted({profile.kind for profile in profiles}),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return ProjectionBank(
        atoms=atoms,
        profiles=profiles,
        source_bases=np.asarray(source_bases, dtype=np.complex128),
        projected_bases=np.asarray(projected, dtype=np.complex128),
        profile_matrix=np.asarray(rho, dtype=np.float64),
        observability=np.asarray(observability, dtype=np.float64),
        denominators=np.asarray(denominators, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        quotient_factors=tuple(quotient_factors),
        fingerprint=fingerprint,
    )


__all__ = ["ProjectedComponent", "ProjectionBank", "build_projection_bank"]
