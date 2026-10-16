"""Hash-checked, column-matched access to the V3 full-source BEM library.

The full library is an already-open ``library_train`` artifact.  This module
never accepts a seed-tier path and never reads diagnostic, validation,
confirmation, or formal truth.  It maps a deployed compact source grid to the
same MNE ico1 vertex identities in all 64 training geometries before fitting
source-dependent residual statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from free_bisc_hypersbl.inverse.v3_uncertain import (
    ULUncertaintyModel,
    build_ul_uncertainty_model,
    group_covariance_from_residual_samples,
)


class FullSourceLibraryError(RuntimeError):
    """The full-source artifact or requested column map is not auditable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value: object, *, dtype: object | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FullSourceGeometryRecord:
    ordinal: int
    geometry_id: str
    array_path: Path
    array_sha256: str
    geometry_fingerprint: str


@dataclass(frozen=True, slots=True)
class FullSourceBEMLibrary:
    root: Path
    records: tuple[FullSourceGeometryRecord, ...]
    full_source_operators: np.ndarray
    frequencies_hz: np.ndarray
    channel_names: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    source_vertex_keys: tuple[str, ...]
    orientation_labels: tuple[str, ...]
    source_grid_namespace: str
    manifest_sha256: str
    metadata: Mapping[str, object]

    @property
    def geometry_count(self) -> int:
        return len(self.records)

    @property
    def source_group_count(self) -> int:
        return len(self.source_vertex_keys)


@dataclass(frozen=True, slots=True)
class ColumnMatchedBEMStatistics:
    """Library-only sufficient statistics for one deployed compact grid."""

    source_group_indices: np.ndarray
    source_vertex_keys: tuple[str, ...]
    operators: np.ndarray
    operator_geometry_ids: tuple[str, ...]
    nominal_leadfields: np.ndarray
    residual_mean: np.ndarray
    group_sensor_covariance: np.ndarray
    residual_basis: np.ndarray
    residual_singular_values: np.ndarray
    medoid_indices: tuple[int, ...]
    medoid_geometry_ids: tuple[str, ...]
    frequency_uncertainty: np.ndarray
    source_grid_namespace: str
    library_manifest_sha256: str
    shrinkage: float
    rank_cap: int
    metadata: Mapping[str, object]

    def uncertainty_model(self) -> ULUncertaintyModel:
        return build_ul_uncertainty_model(
            self.nominal_leadfields,
            group_size=3,
            residual_mean=self.residual_mean,
            source_dependent_covariance=self.group_sensor_covariance,
            rank_cap=None,
        )

    def mixture_operator_models(self, *, model_count: int = 4) -> np.ndarray:
        """Return nominal plus ``model_count-1`` library medoids."""

        count = int(model_count)
        if count not in (2, 4):
            raise ValueError("BEM mixture model_count must be 2 or 4")
        needed = count - 1
        if len(self.medoid_indices) < needed:
            raise ValueError("column-matched library has too few medoids")
        selected = self.operators[np.asarray(self.medoid_indices[:needed], dtype=int)]
        return _readonly(
            np.concatenate((self.nominal_leadfields[None, ...], selected), axis=0),
            dtype=np.complex128,
        )

    def mixture_operator_models_by_geometry_ids(
        self,
        geometry_ids: Sequence[str],
    ) -> np.ndarray:
        """Return nominal plus the exact preregistered library geometries.

        Column matching changes which source columns are loaded; it must not
        silently reselect a different candidate-operator set for each head.
        """

        requested = tuple(str(value) for value in geometry_ids)
        if not requested or any(not value for value in requested):
            raise ValueError("at least one nonempty frozen geometry ID is required")
        if len(set(requested)) != len(requested):
            raise ValueError("frozen geometry IDs must be unique")
        lookup = {value: index for index, value in enumerate(self.operator_geometry_ids)}
        missing = [value for value in requested if value not in lookup]
        if missing:
            raise ValueError(
                "frozen BEM geometry ID is absent from the bound library: "
                + ",".join(missing)
            )
        selected = self.operators[
            np.asarray([lookup[value] for value in requested], dtype=np.int64)
        ]
        return _readonly(
            np.concatenate((self.nominal_leadfields[None, ...], selected), axis=0),
            dtype=np.complex128,
        )

    def mixture_operator_models_by_frozen_geometry_ids(
        self,
        frozen_geometry_ids: Sequence[str],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Map frozen pre-extension IDs to the same geometry ordinals.

        The full-source cache intentionally uses a new geometry-ID namespace,
        so its one-way suffixes differ from the earlier selected-12 cache even
        though the authorized draw at each ordinal is unchanged.  The ordinal
        is public, unique, and immutable; no seed value is required.
        """

        frozen = tuple(str(value) for value in frozen_geometry_ids)
        if not frozen or len(set(frozen)) != len(frozen):
            raise ValueError("frozen geometry IDs must be nonempty and unique")
        ordinals: list[int] = []
        pattern = re.compile(r"library_(\d{3})_[0-9a-f]{12}")
        for value in frozen:
            match = pattern.fullmatch(value)
            if match is None:
                raise ValueError("frozen geometry ID has an invalid namespace format")
            ordinal = int(match.group(1))
            if ordinal >= len(self.operator_geometry_ids):
                raise ValueError("frozen geometry ordinal is outside the full library")
            actual = self.operator_geometry_ids[ordinal]
            if not actual.startswith(f"library_{ordinal:03d}_"):
                raise ValueError("full-source geometry ordinal/ID binding changed")
            ordinals.append(ordinal)
        selected = self.operators[np.asarray(ordinals, dtype=np.int64)]
        models = np.concatenate((self.nominal_leadfields[None, ...], selected), axis=0)
        actual_ids = tuple(self.operator_geometry_ids[index] for index in ordinals)
        return _readonly(models, dtype=np.complex128), actual_ids


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FullSourceLibraryError(f"cannot read JSON artifact {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise FullSourceLibraryError(f"JSON artifact {path.name} is not a mapping")
    return payload


def load_full_source_bem_library(
    root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_geometry_count: int = 64,
) -> FullSourceBEMLibrary:
    """Validate the seedless full-source cache and retain its operator stack.

    Every geometry NPZ is hash-checked and opened exactly once here.  The
    validated full-source operators are then kept as one read-only in-memory
    array so per-head column matching cannot repeat file I/O or hashing.
    """

    directory = Path(root).resolve()
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FullSourceLibraryError("full-source library lacks run_manifest.json")
    manifest_hash = _sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != str(
        expected_manifest_sha256
    ):
        raise FullSourceLibraryError("full-source run-manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    expected = int(expected_geometry_count)
    if (
        manifest.get("status") != "SUCCESS"
        or int(manifest.get("successful_geometry_count", -1)) != expected
        or int(manifest.get("failed_geometry_count", -1)) != 0
        or bool(manifest.get("seed_values_recorded", True))
    ):
        raise FullSourceLibraryError("full-source library is incomplete or discloses seeds")

    metadata_paths = sorted((directory / "geometries").glob("geometry_*.json"))
    if len(metadata_paths) != expected:
        raise FullSourceLibraryError("full-source geometry metadata count differs")
    records: list[FullSourceGeometryRecord] = []
    full_source_operators: list[np.ndarray] = []
    reference: dict[str, object] | None = None
    required_arrays = {
        "leadfield",
        "frequencies_hz",
        "channel_names",
        "orientation_labels",
        "full_source_leadfield",
        "full_source_xyz_m",
        "full_source_group_indices",
        "full_source_group_ids",
        "full_source_vertex_keys",
        "full_source_grid_namespace",
        "selected_source_group_indices",
    }
    for ordinal, metadata_path in enumerate(metadata_paths):
        metadata = _load_json(metadata_path)
        if (
            int(metadata.get("geometry_ordinal", -1)) != ordinal
            or metadata.get("status") != "SUCCESS"
            or bool(metadata.get("seed_value_recorded", True))
        ):
            raise FullSourceLibraryError("full-source geometry metadata is invalid")
        raw_array_path = Path(str(metadata.get("array_path", "")))
        array_path = raw_array_path if raw_array_path.is_absolute() else directory.parents[3] / raw_array_path
        # Generated metadata paths are repository relative.  If the above
        # reconstruction is not the exact file (e.g. a copied test fixture),
        # prefer the sibling NPZ with the same stem.
        sibling = metadata_path.with_suffix(".npz")
        if not array_path.is_file() and sibling.is_file():
            array_path = sibling
        expected_hash = str(metadata.get("array_sha256", ""))
        if not array_path.is_file() or _sha256_file(array_path) != expected_hash:
            raise FullSourceLibraryError("full-source geometry array hash mismatch")
        with np.load(array_path, allow_pickle=False) as archive:
            missing = required_arrays - set(archive.files)
            if missing:
                raise FullSourceLibraryError(
                    "full-source geometry lacks arrays: " + ",".join(sorted(missing))
                )
            full = np.asarray(archive["full_source_leadfield"], dtype=np.complex128)
            coordinates = np.asarray(archive["full_source_xyz_m"], dtype=np.float64)
            indices = np.asarray(archive["full_source_group_indices"], dtype=np.int64)
            selected = np.asarray(archive["selected_source_group_indices"], dtype=np.int64)
            historical = np.asarray(archive["leadfield"], dtype=np.complex128)
            if (
                full.ndim != 4
                or coordinates.shape != (full.shape[2], 3)
                or indices.shape != (full.shape[2],)
                or not np.array_equal(indices, np.arange(full.shape[2]))
                or not np.all(np.isfinite(full))
                or not np.array_equal(full[:, :, selected, :], historical)
            ):
                raise FullSourceLibraryError("full-source array/subset contract failed")
            current = {
                "shape": full.shape,
                "frequencies": tuple(float(value) for value in archive["frequencies_hz"]),
                "channels": tuple(str(value) for value in archive["channel_names"]),
                "groups": tuple(str(value) for value in archive["full_source_group_ids"]),
                "vertices": tuple(str(value) for value in archive["full_source_vertex_keys"]),
                "orientations": tuple(str(value) for value in archive["orientation_labels"]),
                "namespace": str(np.asarray(archive["full_source_grid_namespace"]).item()),
            }
            full_source_operators.append(full)
        if reference is None:
            reference = current
        elif current != reference:
            raise FullSourceLibraryError("full-source identity/order differs across geometries")
        records.append(
            FullSourceGeometryRecord(
                ordinal=ordinal,
                geometry_id=str(metadata["geometry_id"]),
                array_path=array_path,
                array_sha256=expected_hash,
                geometry_fingerprint=str(metadata["geometry_fingerprint"]),
            )
        )
    assert reference is not None
    return FullSourceBEMLibrary(
        root=directory,
        records=tuple(records),
        full_source_operators=_readonly(
            np.stack(full_source_operators, axis=0), dtype=np.complex128
        ),
        frequencies_hz=_readonly(reference["frequencies"], dtype=np.float64),
        channel_names=tuple(reference["channels"]),
        source_group_ids=tuple(reference["groups"]),
        source_vertex_keys=tuple(reference["vertices"]),
        orientation_labels=tuple(reference["orientations"]),
        source_grid_namespace=str(reference["namespace"]),
        manifest_sha256=manifest_hash,
        metadata=MappingProxyType(
            {
                "source_split": "library_train",
                "seed_values_recorded": False,
                "lazy_geometry_arrays": False,
                "validated_operator_cache": "readonly_in_memory_stack",
                "fit_reopens_geometry_arrays": False,
                "truth_or_emd_inputs_accepted": False,
            }
        ),
    )


def _select_medoids(vectors: np.ndarray, count: int) -> tuple[int, ...]:
    number = vectors.shape[0]
    retained = min(int(count), number)
    if retained < 1:
        raise ValueError("at least one medoid is required")
    norm = np.sum(np.abs(vectors) ** 2, axis=1)
    distance = np.sqrt(
        np.maximum(
            norm[:, None] + norm[None, :] - 2.0 * np.real(vectors @ vectors.conj().T),
            0.0,
        )
    )
    medoids = [int(np.argmin(np.sum(distance, axis=1)))]
    while len(medoids) < retained:
        nearest = np.min(distance[:, medoids], axis=1)
        nearest[medoids] = -np.inf
        medoids.append(int(np.argmax(nearest)))
    return tuple(medoids)


def fit_column_matched_bem_statistics(
    library: FullSourceBEMLibrary,
    source_group_indices: Sequence[int],
    nominal_leadfields: np.ndarray,
    *,
    shrinkage: float = 0.10,
    rank_cap: int = 8,
    medoid_count: int = 3,
) -> ColumnMatchedBEMStatistics:
    """Fit residual statistics after exact source-identity column selection."""

    indices = np.asarray(source_group_indices, dtype=np.int64).reshape(-1)
    if (
        indices.size < 1
        or np.any(indices < 0)
        or np.any(indices >= library.source_group_count)
        or len(set(int(value) for value in indices)) != indices.size
    ):
        raise ValueError("source_group_indices must be unique in-range identities")
    nominal = np.asarray(nominal_leadfields, dtype=np.complex128)
    expected_shape = (
        library.frequencies_hz.size,
        len(library.channel_names),
        indices.size,
        len(library.orientation_labels),
    )
    if nominal.shape == (expected_shape[0], expected_shape[1], np.prod(expected_shape[2:])):
        nominal = nominal.reshape(expected_shape)
    if nominal.shape != expected_shape or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_leadfields do not match the selected full-source columns")
    operator_array = np.take(library.full_source_operators, indices, axis=3)
    residual = operator_array - nominal[None, ...]
    flat_residual = residual.reshape(
        residual.shape[0], residual.shape[1], residual.shape[2], -1
    )
    mean_flat, covariance = group_covariance_from_residual_samples(
        flat_residual,
        group_size=len(library.orientation_labels),
        shrinkage=float(shrinkage),
        rank_cap=int(rank_cap),
    )
    mean = np.asarray(mean_flat).reshape(expected_shape)
    vector_size = int(np.prod(expected_shape[1:]))
    retained = min(int(rank_cap), residual.shape[0] - 1, vector_size)
    basis = np.empty((expected_shape[0], retained, vector_size), dtype=np.complex128)
    singular = np.empty((expected_shape[0], retained), dtype=np.float64)
    uncertainty = np.empty(expected_shape[0], dtype=np.float64)
    for frequency in range(expected_shape[0]):
        matrix = residual[:, frequency].reshape(residual.shape[0], -1)
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        _, values, right = np.linalg.svd(centered, full_matrices=False)
        singular[frequency] = values[:retained]
        basis[frequency] = right[:retained]
        uncertainty[frequency] = float(
            np.mean(np.sum(np.abs(matrix) ** 2, axis=1))
            / max(float(np.sum(np.abs(nominal[frequency]) ** 2)), np.finfo(float).eps)
        )
    medoids = _select_medoids(
        residual.reshape(residual.shape[0], -1), int(medoid_count)
    )
    keys = tuple(library.source_vertex_keys[int(index)] for index in indices)
    return ColumnMatchedBEMStatistics(
        source_group_indices=_readonly(indices, dtype=np.int64),
        source_vertex_keys=keys,
        operators=_readonly(operator_array, dtype=np.complex128),
        operator_geometry_ids=tuple(record.geometry_id for record in library.records),
        nominal_leadfields=_readonly(nominal, dtype=np.complex128),
        residual_mean=_readonly(mean, dtype=np.complex128),
        group_sensor_covariance=_readonly(covariance, dtype=np.complex128),
        residual_basis=_readonly(basis, dtype=np.complex128),
        residual_singular_values=_readonly(singular, dtype=np.float64),
        medoid_indices=medoids,
        medoid_geometry_ids=tuple(library.records[index].geometry_id for index in medoids),
        frequency_uncertainty=_readonly(uncertainty, dtype=np.float64),
        source_grid_namespace=library.source_grid_namespace,
        library_manifest_sha256=library.manifest_sha256,
        shrinkage=float(shrinkage),
        rank_cap=int(rank_cap),
        metadata=MappingProxyType(
            {
                "source_split": "library_train",
                "geometry_count": library.geometry_count,
                "column_match": "exact_mne_forward_source_identity",
                "current_head_truth_or_emd_used": False,
                "nominal_operator_role": "deployed_inverse_only",
            }
        ),
    )


__all__ = [
    "ColumnMatchedBEMStatistics",
    "FullSourceBEMLibrary",
    "FullSourceGeometryRecord",
    "FullSourceLibraryError",
    "fit_column_matched_bem_statistics",
    "load_full_source_bem_library",
]
