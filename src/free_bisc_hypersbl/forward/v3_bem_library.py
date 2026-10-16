"""Leakage-safe V3 BEM operator-library primitives.

The module deliberately separates offline ``library_train`` fitting from every
held-out experiment tier.  It contains no seed-registry or experiment-runner
code, and it never retries a failed geometry with a different random draw.
Large BEM solves remain an offline caller responsibility; the routines here
record, validate, align, and summarize their resulting operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class GeometrySplitError(ValueError):
    """Raised when geometry identities cross an experiment-tier boundary."""


class LibraryLeakageError(PermissionError):
    """Raised before held-out operator truth can enter an offline fit."""


class SurfaceNestingError(ValueError):
    """Raised when a deformed head is not strictly nested."""


class LeadfieldAlignmentError(ValueError):
    """Raised when lead-field conventions cannot be aligned unambiguously."""


class GeometrySplit(str, Enum):
    """Strongly typed V3 geometry partitions.

    ``HELDOUT`` is provided for isolated library unit tests.  Production
    experiment records should use the more specific diagnostic/validation/
    confirmation/formal members.
    """

    LIBRARY_TRAIN = "library_train"
    DIAGNOSTIC = "diagnostic"
    VALIDATION_ROUND_1 = "validation_round_1"
    VALIDATION_ROUND_2 = "validation_round_2"
    VALIDATION_ROUND_3_FALLBACK = "validation_round_3_fallback"
    CONFIRMATION = "confirmation"
    FORMAL_TEST = "formal_test"
    HELDOUT = "heldout"

    @property
    def is_library_train(self) -> bool:
        return self is GeometrySplit.LIBRARY_TRAIN


def _as_readonly_float(values: ArrayLike, *, ndim: int | None = None) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got {array.ndim}-D")
    if not np.isfinite(array).all():
        raise ValueError("array contains a non-finite value")
    array.setflags(write=False)
    return array


def _as_readonly_complex(values: ArrayLike, *, ndim: int | None = None) -> ComplexArray:
    array = np.array(values, dtype=np.complex128, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got {array.ndim}-D")
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError("array contains a non-finite value")
    array.setflags(write=False)
    return array


def _nonempty_unique(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value.strip() for value in normalized):
        raise ValueError(f"{label} must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class GeometryProvenance:
    """Immutable geometry identity used at every library boundary."""

    geometry_id: str
    split: GeometrySplit
    geometry_fingerprint: str

    def __post_init__(self) -> None:
        geometry_id = str(self.geometry_id)
        fingerprint = str(self.geometry_fingerprint).lower()
        try:
            split = GeometrySplit(self.split)
        except ValueError as error:
            raise GeometrySplitError(f"unknown geometry split: {self.split!r}") from error
        if not geometry_id.strip():
            raise ValueError("geometry_id must not be empty")
        if not fingerprint.strip():
            raise ValueError("geometry_fingerprint must not be empty")
        object.__setattr__(self, "geometry_id", geometry_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "geometry_fingerprint", fingerprint)


def _identity_set(
    records: Sequence[GeometryProvenance], *, label: str
) -> tuple[set[str], set[str]]:
    ids = [record.geometry_id for record in records]
    fingerprints = [record.geometry_fingerprint for record in records]
    if len(ids) != len(set(ids)):
        raise GeometrySplitError(f"duplicate geometry_id within {label}")
    if len(fingerprints) != len(set(fingerprints)):
        raise GeometrySplitError(f"duplicate geometry fingerprint within {label}")
    return set(ids), set(fingerprints)


def assert_geometry_splits_are_disjoint(
    library_train: Sequence[GeometryProvenance],
    heldout: Sequence[GeometryProvenance],
) -> None:
    """Reject ID or content-fingerprint overlap between train and held-out heads."""

    if not library_train:
        raise GeometrySplitError("library_train must contain at least one geometry")
    if any(not record.split.is_library_train for record in library_train):
        raise GeometrySplitError("all library_train records must have library_train split")
    if any(record.split.is_library_train for record in heldout):
        raise GeometrySplitError("heldout records cannot have library_train split")
    train_ids, train_fingerprints = _identity_set(library_train, label="library_train")
    heldout_ids, heldout_fingerprints = _identity_set(heldout, label="heldout")
    overlapping_ids = sorted(train_ids & heldout_ids)
    overlapping_fingerprints = sorted(train_fingerprints & heldout_fingerprints)
    if overlapping_ids or overlapping_fingerprints:
        raise GeometrySplitError(
            "geometry partitions overlap: "
            f"ids={overlapping_ids}, fingerprints={overlapping_fingerprints}"
        )


@dataclass(frozen=True, slots=True)
class GeometrySplitManifest:
    library_train: tuple[GeometryProvenance, ...]
    heldout: tuple[GeometryProvenance, ...]

    def __post_init__(self) -> None:
        train = tuple(self.library_train)
        heldout = tuple(self.heldout)
        assert_geometry_splits_are_disjoint(train, heldout)
        object.__setattr__(self, "library_train", train)
        object.__setattr__(self, "heldout", heldout)


@dataclass(frozen=True, slots=True)
class SurfaceMesh:
    """Triangular surface in metres and a shared head coordinate system."""

    name: str
    vertices_m: FloatArray
    triangles: NDArray[np.int64]

    def __init__(
        self,
        name: str,
        vertices_m: ArrayLike,
        triangles: ArrayLike | None = None,
    ) -> None:
        name = str(name)
        if not name.strip():
            raise ValueError("surface name must not be empty")
        vertices = _as_readonly_float(vertices_m, ndim=2)
        if vertices.shape[1] != 3 or vertices.shape[0] < 4:
            raise ValueError("surface vertices must have shape (n>=4, 3)")
        if np.any(np.linalg.norm(vertices, axis=1) == 0.0):
            raise ValueError("surface contains a vertex at the coordinate origin")
        if triangles is None:
            triangle_array = np.empty((0, 3), dtype=np.int64)
        else:
            triangle_array = np.array(triangles, dtype=np.int64, copy=True)
            if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
                raise ValueError("triangles must have shape (m, 3)")
            if triangle_array.size and (
                triangle_array.min() < 0 or triangle_array.max() >= vertices.shape[0]
            ):
                raise ValueError("triangle index is outside the vertex array")
        triangle_array.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangle_array)


@dataclass(frozen=True, slots=True)
class NestedSurfaces:
    brain: SurfaceMesh
    inner_skull: SurfaceMesh
    outer_skull: SurfaceMesh
    outer_skin: SurfaceMesh

    def ordered(self) -> tuple[SurfaceMesh, ...]:
        return (self.brain, self.inner_skull, self.outer_skull, self.outer_skin)


@dataclass(frozen=True, slots=True)
class SurfacePairThickness:
    inner_name: str
    outer_name: str
    minimum_m: float
    median_m: float
    maximum_m: float
    maximum_angular_gap_rad: float


@dataclass(frozen=True, slots=True)
class NestingReport:
    pairs: tuple[SurfacePairThickness, ...]
    minimum_required_m: float

    @property
    def global_minimum_m(self) -> float:
        return min(pair.minimum_m for pair in self.pairs)


def _directions_and_radii(vertices: FloatArray, origin_m: FloatArray) -> tuple[FloatArray, FloatArray]:
    centered = vertices - origin_m[None, :]
    radii = np.linalg.norm(centered, axis=1)
    if np.any(radii <= 0.0):
        raise SurfaceNestingError("a surface vertex coincides with the nesting origin")
    return centered / radii[:, None], radii


def _radial_thickness(
    inner: SurfaceMesh,
    outer: SurfaceMesh,
    origin_m: FloatArray,
) -> tuple[FloatArray, float]:
    inner_direction, inner_radius = _directions_and_radii(inner.vertices_m, origin_m)
    outer_direction, outer_radius = _directions_and_radii(outer.vertices_m, origin_m)
    inner_tree = cKDTree(inner_direction)
    outer_tree = cKDTree(outer_direction)
    outer_distance, outer_index = outer_tree.query(inner_direction, k=1)
    inner_distance, inner_index = inner_tree.query(outer_direction, k=1)
    thickness = np.concatenate(
        [outer_radius[outer_index] - inner_radius, outer_radius - inner_radius[inner_index]]
    )
    chord = max(float(np.max(outer_distance)), float(np.max(inner_distance)))
    angular_gap = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return thickness, float(angular_gap)


def validate_nested_surfaces(
    surfaces: NestedSurfaces,
    *,
    minimum_thickness_m: float = 1e-4,
    origin_m: ArrayLike = (0.0, 0.0, 0.0),
    maximum_angular_gap_rad: float = 0.40,
) -> NestingReport:
    """Verify strict radial nesting in both surface sampling directions.

    The bidirectional nearest-direction check supports independent fsaverage
    tessellations without pretending unmatched vertices correspond exactly.
    Sparse surfaces with inadequate angular overlap fail explicitly.
    """

    minimum = float(minimum_thickness_m)
    maximum_gap = float(maximum_angular_gap_rad)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_thickness_m must be strictly positive")
    if not np.isfinite(maximum_gap) or not 0.0 < maximum_gap < np.pi:
        raise ValueError("maximum_angular_gap_rad must lie in (0, pi)")
    origin = _as_readonly_float(origin_m, ndim=1)
    if origin.shape != (3,):
        raise ValueError("origin_m must contain three coordinates")
    reports: list[SurfacePairThickness] = []
    ordered = surfaces.ordered()
    for inner, outer in zip(ordered[:-1], ordered[1:], strict=True):
        thickness, angular_gap = _radial_thickness(inner, outer, origin)
        if angular_gap > maximum_gap:
            raise SurfaceNestingError(
                f"{inner.name}->{outer.name} angular sampling gap "
                f"{angular_gap:.6g} exceeds {maximum_gap:.6g} rad"
            )
        observed_minimum = float(np.min(thickness))
        if observed_minimum < minimum:
            raise SurfaceNestingError(
                f"{inner.name}->{outer.name} minimum radial thickness "
                f"{observed_minimum:.9g} m is below {minimum:.9g} m"
            )
        reports.append(
            SurfacePairThickness(
                inner_name=inner.name,
                outer_name=outer.name,
                minimum_m=observed_minimum,
                median_m=float(np.median(thickness)),
                maximum_m=float(np.max(thickness)),
                maximum_angular_gap_rad=angular_gap,
            )
        )
    return NestingReport(tuple(reports), minimum)


@dataclass(frozen=True, slots=True)
class SurfaceDeformation:
    """Dimensionless real spherical-harmonic radial coefficients."""

    l2: FloatArray
    l3: FloatArray | None = None

    def __init__(self, l2: ArrayLike, l3: ArrayLike | None = None) -> None:
        l2_array = _as_readonly_float(l2, ndim=1)
        if l2_array.shape != (5,):
            raise ValueError("l=2 deformation must contain five real coefficients")
        l3_array = None if l3 is None else _as_readonly_float(l3, ndim=1)
        if l3_array is not None and l3_array.shape != (7,):
            raise ValueError("l=3 deformation must contain seven real coefficients")
        object.__setattr__(self, "l2", l2_array)
        object.__setattr__(self, "l3", l3_array)


@dataclass(frozen=True, slots=True)
class NestedSurfaceDeformations:
    brain: SurfaceDeformation
    inner_skull: SurfaceDeformation
    outer_skull: SurfaceDeformation
    outer_skin: SurfaceDeformation

    def ordered(self) -> tuple[SurfaceDeformation, ...]:
        return (self.brain, self.inner_skull, self.outer_skull, self.outer_skin)


@dataclass(frozen=True, slots=True)
class ConductivityParameters:
    brain_s_per_m: float
    skull_s_per_m: float
    scalp_s_per_m: float

    def __post_init__(self) -> None:
        values = np.array(
            [self.brain_s_per_m, self.skull_s_per_m, self.scalp_s_per_m],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("all conductivities must be finite and positive")
        # Wide safety bounds reject unit mistakes while retaining literature ranges.
        lower = np.array([0.05, 0.001, 0.05])
        upper = np.array([1.50, 0.10, 1.50])
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("conductivity is outside the registered physical bounds")
        if self.skull_s_per_m >= min(self.brain_s_per_m, self.scalp_s_per_m):
            raise ValueError("skull conductivity must be below brain and scalp conductivity")


@dataclass(frozen=True, slots=True)
class ElectrodePerturbation:
    rotation_vector_rad: FloatArray
    translation_m: FloatArray
    jitter_m: FloatArray

    def __init__(
        self,
        rotation_vector_rad: ArrayLike,
        translation_m: ArrayLike,
        jitter_m: ArrayLike,
    ) -> None:
        rotation = _as_readonly_float(rotation_vector_rad, ndim=1)
        translation = _as_readonly_float(translation_m, ndim=1)
        jitter = _as_readonly_float(jitter_m, ndim=2)
        if rotation.shape != (3,) or translation.shape != (3,):
            raise ValueError("rotation and translation must contain three values")
        if jitter.shape[1] != 3:
            raise ValueError("electrode jitter must have shape (n_electrodes, 3)")
        if np.linalg.norm(rotation) > np.deg2rad(20.0):
            raise ValueError("electrode rotation exceeds 20 degrees")
        if np.linalg.norm(translation) > 0.03:
            raise ValueError("electrode translation exceeds 30 mm")
        if jitter.size and np.max(np.linalg.norm(jitter, axis=1)) > 0.01:
            raise ValueError("electrode jitter exceeds 10 mm")
        object.__setattr__(self, "rotation_vector_rad", rotation)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "jitter_m", jitter)


@dataclass(frozen=True, slots=True)
class SourceTransportParameters:
    rotation_vector_rad: FloatArray
    translation_m: FloatArray
    radial_scale: float

    def __init__(
        self,
        rotation_vector_rad: ArrayLike = (0.0, 0.0, 0.0),
        translation_m: ArrayLike = (0.0, 0.0, 0.0),
        radial_scale: float = 1.0,
    ) -> None:
        rotation = _as_readonly_float(rotation_vector_rad, ndim=1)
        translation = _as_readonly_float(translation_m, ndim=1)
        scale = float(radial_scale)
        if rotation.shape != (3,) or translation.shape != (3,):
            raise ValueError("source rotation and translation must contain three values")
        if np.linalg.norm(rotation) > np.deg2rad(20.0):
            raise ValueError("source transport rotation exceeds 20 degrees")
        if np.linalg.norm(translation) > 0.03:
            raise ValueError("source transport translation exceeds 30 mm")
        if not np.isfinite(scale) or not 0.75 <= scale <= 1.25:
            raise ValueError("source radial_scale must lie in [0.75, 1.25]")
        object.__setattr__(self, "rotation_vector_rad", rotation)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "radial_scale", scale)


@dataclass(frozen=True, slots=True)
class BEMGeometryParameters:
    deformations: NestedSurfaceDeformations
    conductivity: ConductivityParameters
    electrodes: ElectrodePerturbation
    source_transport: SourceTransportParameters
    origin_m: FloatArray

    def __init__(
        self,
        deformations: NestedSurfaceDeformations,
        conductivity: ConductivityParameters,
        electrodes: ElectrodePerturbation,
        source_transport: SourceTransportParameters,
        origin_m: ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        origin = _as_readonly_float(origin_m, ndim=1)
        if origin.shape != (3,):
            raise ValueError("origin_m must contain three values")
        object.__setattr__(self, "deformations", deformations)
        object.__setattr__(self, "conductivity", conductivity)
        object.__setattr__(self, "electrodes", electrodes)
        object.__setattr__(self, "source_transport", source_transport)
        object.__setattr__(self, "origin_m", origin)


def _real_harmonic_basis(directions: FloatArray, include_l3: bool) -> FloatArray:
    """Return a fixed real, low-order harmonic-polynomial basis on the sphere."""

    x, y, z = directions.T
    l2 = np.column_stack(
        [x * y, y * z, 0.5 * (3.0 * z * z - 1.0), x * z, x * x - y * y]
    )
    if not include_l3:
        return l2
    l3 = np.column_stack(
        [
            y * (3.0 * x * x - y * y),
            x * y * z,
            y * (5.0 * z * z - 1.0),
            z * (5.0 * z * z - 3.0),
            x * (5.0 * z * z - 1.0),
            z * (x * x - y * y),
            x * (x * x - 3.0 * y * y),
        ]
    )
    return np.column_stack([l2, l3])


def deform_surface_radially(
    surface: SurfaceMesh,
    deformation: SurfaceDeformation,
    *,
    origin_m: ArrayLike = (0.0, 0.0, 0.0),
    maximum_fractional_displacement: float = 0.20,
) -> SurfaceMesh:
    """Apply one smooth l=2/(optional l=3) radial deformation without clipping."""

    origin = _as_readonly_float(origin_m, ndim=1)
    if origin.shape != (3,):
        raise ValueError("origin_m must contain three values")
    directions, radii = _directions_and_radii(surface.vertices_m, origin)
    coefficients = deformation.l2
    if deformation.l3 is not None:
        coefficients = np.concatenate([coefficients, deformation.l3])
    displacement = _real_harmonic_basis(
        directions, include_l3=deformation.l3 is not None
    ) @ coefficients
    limit = float(maximum_fractional_displacement)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_fractional_displacement must be positive")
    observed = float(np.max(np.abs(displacement)))
    if observed > limit:
        raise SurfaceNestingError(
            f"{surface.name} fractional radial displacement {observed:.6g} "
            f"exceeds {limit:.6g}; deformation was not clipped or resampled"
        )
    scale = 1.0 + displacement
    if np.any(scale <= 0.0):
        raise SurfaceNestingError(f"{surface.name} deformation produced nonpositive radius")
    vertices = origin[None, :] + directions * (radii * scale)[:, None]
    return SurfaceMesh(surface.name, vertices, surface.triangles)


@dataclass(frozen=True, slots=True)
class GeometryBuildFailure:
    stage: str
    error_type: str
    message: str
    resampled: bool = False


@dataclass(frozen=True, slots=True)
class GeometryBuildResult:
    geometry_id: str
    parameters: BEMGeometryParameters
    surfaces: NestedSurfaces | None
    nesting_report: NestingReport | None
    failure: GeometryBuildFailure | None

    @property
    def success(self) -> bool:
        return self.failure is None


def build_deformed_nested_surfaces(
    geometry_id: str,
    base_surfaces: NestedSurfaces,
    parameters: BEMGeometryParameters,
    *,
    minimum_thickness_m: float = 1e-4,
    maximum_fractional_displacement: float = 0.20,
) -> GeometryBuildResult:
    """Deform and validate exactly one requested geometry.

    Failure returns an audit record.  There is intentionally no retry loop,
    random generator, clipping, or replacement geometry.
    """

    try:
        deformed = tuple(
            deform_surface_radially(
                surface,
                deformation,
                origin_m=parameters.origin_m,
                maximum_fractional_displacement=maximum_fractional_displacement,
            )
            for surface, deformation in zip(
                base_surfaces.ordered(), parameters.deformations.ordered(), strict=True
            )
        )
        nested = NestedSurfaces(*deformed)
        report = validate_nested_surfaces(
            nested,
            minimum_thickness_m=minimum_thickness_m,
            origin_m=parameters.origin_m,
        )
    except (ValueError, SurfaceNestingError) as error:
        return GeometryBuildResult(
            geometry_id=str(geometry_id),
            parameters=parameters,
            surfaces=None,
            nesting_report=None,
            failure=GeometryBuildFailure(
                stage="surface_deformation_or_nesting",
                error_type=type(error).__name__,
                message=str(error),
                resampled=False,
            ),
        )
    return GeometryBuildResult(
        geometry_id=str(geometry_id),
        parameters=parameters,
        surfaces=nested,
        nesting_report=report,
        failure=None,
    )


def _rotation_matrix(rotation_vector_rad: FloatArray) -> FloatArray:
    angle = float(np.linalg.norm(rotation_vector_rad))
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector_rad / angle
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def perturb_electrodes(
    electrode_xyz_m: ArrayLike,
    perturbation: ElectrodePerturbation,
    *,
    origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    electrodes = _as_readonly_float(electrode_xyz_m, ndim=2)
    origin = _as_readonly_float(origin_m, ndim=1)
    if electrodes.shape[1] != 3 or origin.shape != (3,):
        raise ValueError("electrodes must be (n, 3) and origin must be length three")
    if perturbation.jitter_m.shape != electrodes.shape:
        raise ValueError("electrode jitter shape must match electrode coordinates")
    rotation = _rotation_matrix(perturbation.rotation_vector_rad)
    values = (
        (electrodes - origin[None, :]) @ rotation.T
        + origin[None, :]
        + perturbation.translation_m[None, :]
        + perturbation.jitter_m
    )
    return _as_readonly_float(values, ndim=2)


def transport_sources(
    source_xyz_m: ArrayLike,
    transport: SourceTransportParameters,
    *,
    origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    sources = _as_readonly_float(source_xyz_m, ndim=2)
    origin = _as_readonly_float(origin_m, ndim=1)
    if sources.shape[1] != 3 or origin.shape != (3,):
        raise ValueError("sources must be (n, 3) and origin must be length three")
    rotation = _rotation_matrix(transport.rotation_vector_rad)
    values = (
        transport.radial_scale * ((sources - origin[None, :]) @ rotation.T)
        + origin[None, :]
        + transport.translation_m[None, :]
    )
    return _as_readonly_float(values, ndim=2)


def load_fsaverage_nested_surfaces(fsaverage_path: str | Path) -> NestedSurfaces:
    """Load fsaverage ``bem/{brain,inner_skull,outer_skull,outer_skin}.surf``.

    FreeSurfer surface coordinates are converted from millimetres to metres.
    ``fsaverage_path`` may point to either the subject itself or its parent.
    """

    root = Path(fsaverage_path).expanduser().resolve()
    if root.name != "fsaverage" and (root / "fsaverage").is_dir():
        root = root / "fsaverage"
    bem = root / "bem"
    paths = {name: bem / f"{name}.surf" for name in NestedSurfaces.__annotations__}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fsaverage BEM surfaces: {missing}")
    import mne  # Lazy import keeps synthetic unit tests lightweight.

    meshes: dict[str, SurfaceMesh] = {}
    for name, path in paths.items():
        vertices_mm, triangles = mne.read_surface(path, read_metadata=False)
        meshes[name] = SurfaceMesh(name, np.asarray(vertices_mm) * 1e-3, triangles)
    return NestedSurfaces(**meshes)


def geometry_fingerprint(
    surfaces: NestedSurfaces,
    electrode_xyz_m: ArrayLike,
    conductivity: ConductivityParameters,
) -> str:
    """Content hash used in addition to a human-readable geometry ID."""

    digest = hashlib.sha256()
    for surface in surfaces.ordered():
        digest.update(surface.name.encode("utf-8"))
        digest.update(np.ascontiguousarray(surface.vertices_m, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(surface.triangles, dtype="<i8").tobytes())
    electrodes = _as_readonly_float(electrode_xyz_m, ndim=2)
    digest.update(np.ascontiguousarray(electrodes, dtype="<f8").tobytes())
    digest.update(
        np.asarray(
            [
                conductivity.brain_s_per_m,
                conductivity.skull_s_per_m,
                conductivity.scalp_s_per_m,
            ],
            dtype="<f8",
        ).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LeadfieldAlignmentResult:
    leadfield: ComplexArray
    channel_names: tuple[str, ...]
    channel_permutation: tuple[int, ...]
    reference_matrix: FloatArray
    orientation_transforms: FloatArray | None
    scale: float


def least_squares_leadfield_scale(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return the positive real least-squares scale mapping candidate to reference."""

    target = np.asarray(reference, dtype=np.complex128)
    source = np.asarray(candidate, dtype=np.complex128)
    if target.shape != source.shape or target.size == 0:
        raise LeadfieldAlignmentError("reference and candidate must have one non-empty shape")
    denominator = float(np.vdot(source, source).real)
    if denominator <= 0.0:
        raise LeadfieldAlignmentError("candidate leadfield has zero norm")
    scale = float(np.vdot(source, target).real / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise LeadfieldAlignmentError("least-squares scale is not finite and positive")
    return scale


def align_leadfield_reference_channels_orientation_scale(
    leadfield: ArrayLike,
    *,
    raw_channel_names: Sequence[str],
    target_channel_names: Sequence[str],
    reference_matrix: ArrayLike | None = None,
    output_channel_names: Sequence[str] | None = None,
    raw_orientation_frames: ArrayLike | None = None,
    target_orientation_frames: ArrayLike | None = None,
    scale: float = 1.0,
    channel_axis: int | None = None,
) -> LeadfieldAlignmentResult:
    """Align reference, channel order, source frames, and physical scale.

    Oriented operators use canonical trailing axes ``(..., source, 3)``.
    Each orientation frame stores physical basis vectors in its columns.
    """

    array = np.asarray(leadfield, dtype=np.complex128)
    if array.ndim < 2 or array.size == 0:
        raise LeadfieldAlignmentError("leadfield must be a non-empty array with >=2 axes")
    raw_names = _nonempty_unique(raw_channel_names, label="raw_channel_names")
    target_names = _nonempty_unique(target_channel_names, label="target_channel_names")
    if not set(target_names).issubset(raw_names):
        missing = sorted(set(target_names) - set(raw_names))
        raise LeadfieldAlignmentError(f"target channels absent from raw operator: {missing}")
    if channel_axis is None:
        matches = [axis for axis, size in enumerate(array.shape) if size == len(raw_names)]
        if len(matches) != 1:
            raise LeadfieldAlignmentError(
                "channel axis is ambiguous; pass channel_axis explicitly"
            )
        resolved_axis = matches[0]
    else:
        resolved_axis = int(channel_axis)
        if resolved_axis < 0:
            resolved_axis += array.ndim
        if resolved_axis < 0 or resolved_axis >= array.ndim:
            raise LeadfieldAlignmentError("channel_axis is outside the leadfield dimensions")
        if array.shape[resolved_axis] != len(raw_names):
            raise LeadfieldAlignmentError("channel_axis size does not match raw channels")
    permutation = tuple(raw_names.index(name) for name in target_names)
    ordered = np.take(array, permutation, axis=resolved_axis)
    transform: FloatArray | None = None
    if (raw_orientation_frames is None) != (target_orientation_frames is None):
        raise LeadfieldAlignmentError(
            "raw and target orientation frames must be supplied together"
        )
    if raw_orientation_frames is not None:
        raw_frames = np.asarray(raw_orientation_frames, dtype=np.float64)
        target_frames = np.asarray(target_orientation_frames, dtype=np.float64)
        if raw_frames.shape != target_frames.shape or raw_frames.ndim != 3:
            raise LeadfieldAlignmentError("orientation frames must share shape (source,3,3)")
        if raw_frames.shape[1:] != (3, 3):
            raise LeadfieldAlignmentError("orientation frames must have shape (source,3,3)")
        if ordered.shape[-2:] != (raw_frames.shape[0], 3):
            raise LeadfieldAlignmentError(
                "oriented leadfield must end in (source_count, 3)"
            )
        identities = np.eye(3)
        for label, frames in (("raw", raw_frames), ("target", target_frames)):
            gram = np.einsum("gji,gjk->gik", frames, frames)
            if not np.allclose(gram, identities[None, :, :], atol=1e-10, rtol=1e-10):
                raise LeadfieldAlignmentError(f"{label} orientation frames are not orthonormal")
            if np.any(np.linalg.det(frames) <= 0.0):
                raise LeadfieldAlignmentError(f"{label} orientation frames are not right-handed")
        transform = np.einsum("gji,gjk->gik", raw_frames, target_frames)
        ordered = np.einsum("...go,gop->...gp", ordered, transform)
        transform = _as_readonly_float(transform, ndim=3)
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise LeadfieldAlignmentError("scale must be finite and positive")
    if reference_matrix is None:
        reference = np.eye(len(target_names), dtype=np.float64)
    else:
        reference = np.asarray(reference_matrix, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != len(target_names):
            raise LeadfieldAlignmentError(
                "reference_matrix must have one column per target channel"
            )
        if not np.isfinite(reference).all():
            raise LeadfieldAlignmentError("reference matrix contains a non-finite value")
    moved = np.moveaxis(ordered, resolved_axis, 0)
    referenced = np.tensordot(reference, moved, axes=(1, 0)) * scale_value
    referenced = np.moveaxis(referenced, 0, resolved_axis)
    if output_channel_names is None:
        if reference.shape[0] != len(target_names):
            raise LeadfieldAlignmentError(
                "output_channel_names are required when referencing changes channel count"
            )
        output_names = target_names
    else:
        output_names = _nonempty_unique(
            output_channel_names, label="output_channel_names"
        )
        if len(output_names) != reference.shape[0]:
            raise LeadfieldAlignmentError(
                "output_channel_names must match reference-matrix row count"
            )
    return LeadfieldAlignmentResult(
        leadfield=_as_readonly_complex(referenced),
        channel_names=output_names,
        channel_permutation=permutation,
        reference_matrix=_as_readonly_float(reference, ndim=2),
        orientation_transforms=transform,
        scale=scale_value,
    )


@dataclass(frozen=True, slots=True)
class AlignedLeadfield:
    """Canonical complex operator with axes (frequency, channel, group, orientation)."""

    values: ComplexArray
    frequencies_hz: tuple[float, ...]
    channel_names: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    orientation_labels: tuple[str, ...]

    def __init__(
        self,
        values: ArrayLike,
        frequencies_hz: Sequence[float],
        channel_names: Sequence[str],
        source_group_ids: Sequence[str],
        orientation_labels: Sequence[str] = ("x", "y", "z"),
    ) -> None:
        array = _as_readonly_complex(values, ndim=4)
        frequencies = tuple(float(value) for value in frequencies_hz)
        channels = _nonempty_unique(channel_names, label="channel_names")
        groups = _nonempty_unique(source_group_ids, label="source_group_ids")
        orientations = _nonempty_unique(
            orientation_labels, label="orientation_labels"
        )
        expected = (len(frequencies), len(channels), len(groups), len(orientations))
        if array.shape != expected:
            raise ValueError(f"leadfield shape {array.shape} does not match metadata {expected}")
        if not np.isfinite(frequencies).all() or any(value <= 0.0 for value in frequencies):
            raise ValueError("frequencies must be finite and positive")
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "source_group_ids", groups)
        object.__setattr__(self, "orientation_labels", orientations)


@dataclass(frozen=True, slots=True)
class GeometryDescriptor:
    """Geometry-only predictor input; it contains no EEG or source outcome."""

    names: tuple[str, ...]
    values: FloatArray

    def __init__(self, names: Sequence[str], values: ArrayLike) -> None:
        normalized = _nonempty_unique(names, label="geometry descriptor names")
        forbidden_tokens = (
            "eeg",
            "sensor_data",
            "source_truth",
            "source_location",
            "emd",
            "outcome",
            "truth_leadfield",
        )
        forbidden_names = [
            name
            for name in normalized
            if any(token in name.lower() for token in forbidden_tokens)
        ]
        if forbidden_names:
            raise LibraryLeakageError(
                "geometry descriptors cannot encode EEG/source/outcome truth: "
                + ", ".join(forbidden_names)
            )
        array = _as_readonly_float(values, ndim=1)
        if array.shape != (len(normalized),):
            raise ValueError("descriptor names and values have different lengths")
        object.__setattr__(self, "names", normalized)
        object.__setattr__(self, "values", array)


def _rigidly_aligned_coordinates(coordinates: FloatArray) -> FloatArray:
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axes = right.T
    for column in range(3):
        projected = centered @ axes[:, column]
        pivot = int(np.argmax(np.abs(projected)))
        if projected[pivot] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return centered @ axes


def geometry_descriptor(
    surfaces: NestedSurfaces,
    electrode_xyz_m: ArrayLike,
    conductivity: ConductivityParameters,
    *,
    origin_m: ArrayLike = (0.0, 0.0, 0.0),
) -> GeometryDescriptor:
    """Build observable geometry features without EEG or source coordinates."""

    electrodes = _as_readonly_float(electrode_xyz_m, ndim=2)
    origin = _as_readonly_float(origin_m, ndim=1)
    if electrodes.shape[1] != 3 or electrodes.shape[0] < 3 or origin.shape != (3,):
        raise ValueError("electrodes must have shape (n>=3,3) and origin length three")
    nesting = validate_nested_surfaces(surfaces, origin_m=origin)
    aligned = _rigidly_aligned_coordinates(electrodes)
    names: list[str] = []
    values: list[float] = []
    for channel in range(aligned.shape[0]):
        for axis, label in enumerate(("x", "y", "z")):
            names.append(f"electrode_aligned_{channel}_{label}_m")
            values.append(float(aligned[channel, axis]))
    for left in range(electrodes.shape[0]):
        for right in range(left + 1, electrodes.shape[0]):
            names.append(f"electrode_distance_{left}_{right}_m")
            values.append(float(np.linalg.norm(electrodes[left] - electrodes[right])))
    directions, _ = _directions_and_radii(electrodes, origin)
    moments = np.mean(_real_harmonic_basis(directions, include_l3=True), axis=0)
    for order, value in enumerate(moments):
        names.append(f"electrode_spherical_moment_{order}")
        values.append(float(value))
    for surface in surfaces.ordered():
        _, radii = _directions_and_radii(surface.vertices_m, origin)
        for label, value in (
            ("mean", np.mean(radii)),
            ("std", np.std(radii)),
            ("min", np.min(radii)),
            ("max", np.max(radii)),
        ):
            names.append(f"{surface.name}_radius_{label}_m")
            values.append(float(value))
    for pair in nesting.pairs:
        for label, value in (
            ("min", pair.minimum_m),
            ("median", pair.median_m),
            ("max", pair.maximum_m),
        ):
            names.append(f"{pair.inner_name}_to_{pair.outer_name}_{label}_m")
            values.append(float(value))
    for label, value in (
        ("brain", conductivity.brain_s_per_m),
        ("skull", conductivity.skull_s_per_m),
        ("scalp", conductivity.scalp_s_per_m),
    ):
        names.append(f"conductivity_{label}_s_per_m")
        values.append(float(value))
    return GeometryDescriptor(names, values)


@dataclass(frozen=True, slots=True)
class GeometryOperatorSample:
    provenance: GeometryProvenance
    operator: AlignedLeadfield
    descriptor: GeometryDescriptor


def _assert_library_train_provenance(records: Sequence[GeometryProvenance]) -> None:
    forbidden = [
        f"{record.geometry_id}:{record.split.value}"
        for record in records
        if not record.split.is_library_train
    ]
    if forbidden:
        raise LibraryLeakageError(
            "held-out geometry/operator truth cannot enter library fitting: "
            + ", ".join(forbidden)
        )
    _identity_set(records, label="library_train fit")


@dataclass(frozen=True, slots=True)
class GeometryOnlyResidualPredictor:
    """Frozen ridge map from observable geometry descriptors to coefficients."""

    descriptor_names: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    coefficient: FloatArray
    intercept: FloatArray
    training_geometry_ids: tuple[str, ...]
    ridge: float

    def predict(self, descriptor: GeometryDescriptor) -> FloatArray:
        if descriptor.names != self.descriptor_names:
            raise ValueError("geometry descriptor schema differs from the frozen predictor")
        standardized = (descriptor.values - self.feature_mean) / self.feature_scale
        prediction = standardized @ self.coefficient + self.intercept
        return _as_readonly_float(prediction)


def fit_geometry_only_predictor(
    descriptors: Sequence[GeometryDescriptor],
    residual_coefficients: ArrayLike,
    provenance: Sequence[GeometryProvenance],
    *,
    ridge: float = 1e-3,
) -> GeometryOnlyResidualPredictor:
    """Fit only on train geometry metadata and precomputed residual targets."""

    records = tuple(provenance)
    _assert_library_train_provenance(records)
    if len(descriptors) != len(records) or not descriptors:
        raise ValueError("descriptors and provenance must share a non-zero length")
    schema = descriptors[0].names
    if any(descriptor.names != schema for descriptor in descriptors):
        raise ValueError("all geometry descriptors must share one schema")
    x = np.vstack([descriptor.values for descriptor in descriptors])
    y = np.asarray(residual_coefficients, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0] or not np.isfinite(y).all():
        raise ValueError("residual_coefficients must be finite with one row per geometry")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value <= 0.0:
        raise ValueError("ridge must be finite and strictly positive")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    standardized = (x - mean) / scale
    intercept = np.mean(y, axis=0)
    coefficient = np.linalg.solve(
        standardized.T @ standardized + ridge_value * np.eye(x.shape[1]),
        standardized.T @ (y - intercept[None, :]),
    )
    return GeometryOnlyResidualPredictor(
        descriptor_names=schema,
        feature_mean=_as_readonly_float(mean, ndim=1),
        feature_scale=_as_readonly_float(scale, ndim=1),
        coefficient=_as_readonly_float(coefficient, ndim=2),
        intercept=_as_readonly_float(intercept, ndim=1),
        training_geometry_ids=tuple(record.geometry_id for record in records),
        ridge=ridge_value,
    )


@dataclass(frozen=True, slots=True)
class BEMLibraryStatistics:
    training_geometry_ids: tuple[str, ...]
    training_geometry_fingerprints: tuple[str, ...]
    residual_mean: ComplexArray
    source_dependent_residual_covariance: ComplexArray
    residual_basis: ComplexArray
    residual_singular_values: FloatArray
    medoid_indices: tuple[int, ...]
    medoid_geometry_ids: tuple[str, ...]
    operator_cluster_assignment: tuple[int, ...]
    geometry_descriptor_names: tuple[str, ...]
    geometry_descriptor_matrix: FloatArray
    frequency_uncertainty: FloatArray
    frequency_residual_effective_rank: FloatArray
    spectral_condition_number: FloatArray
    subspace_similarity: FloatArray
    principal_angle_mean_rad: FloatArray
    redundancy_matrix_psd: FloatArray


def _check_operator_compatible(reference: AlignedLeadfield, candidate: AlignedLeadfield) -> None:
    fields = ("frequencies_hz", "channel_names", "source_group_ids", "orientation_labels")
    if any(getattr(reference, field) != getattr(candidate, field) for field in fields):
        raise ValueError("library operators do not share canonical metadata")
    if reference.values.shape != candidate.values.shape:
        raise ValueError("library operators do not share one shape")


def _whitening_matrices(noise_covariance: ArrayLike, frequencies: int, channels: int) -> ComplexArray:
    covariance = np.asarray(noise_covariance, dtype=np.complex128)
    if covariance.shape == (channels, channels):
        covariance = np.repeat(covariance[None, :, :], frequencies, axis=0)
    if covariance.shape != (frequencies, channels, channels):
        raise ValueError("noise_covariance must have shape (C,C) or (F,C,C)")
    matrices = np.empty_like(covariance)
    for frequency, value in enumerate(covariance):
        if not np.allclose(value, value.conj().T, atol=1e-10, rtol=1e-10):
            raise ValueError("noise covariance must be Hermitian")
        try:
            chol = np.linalg.cholesky(value)
        except np.linalg.LinAlgError as error:
            raise ValueError("noise covariance must be positive definite") from error
        matrices[frequency] = np.linalg.solve(chol, np.eye(channels))
    return matrices


def _randomized_svd(
    matrix: ComplexArray,
    rank: int,
    rng: np.random.Generator,
    *,
    oversampling: int = 4,
    power_iterations: int = 2,
) -> tuple[FloatArray, ComplexArray]:
    maximum = min(matrix.shape)
    retained = min(int(rank), maximum)
    if retained < 1:
        raise ValueError("residual rank must be at least one")
    sketch_size = min(maximum, retained + max(0, int(oversampling)))
    omega = (
        rng.standard_normal((matrix.shape[1], sketch_size))
        + 1j * rng.standard_normal((matrix.shape[1], sketch_size))
    ) / np.sqrt(2.0)
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(max(0, int(power_iterations))):
        q, _ = np.linalg.qr(matrix @ (matrix.conj().T @ q), mode="reduced")
    compressed = q.conj().T @ matrix
    _, singular, right = np.linalg.svd(compressed, full_matrices=False)
    return singular[:retained].real, right[:retained]


def _select_medoids(vectors: ComplexArray, count: int) -> tuple[int, ...]:
    number = vectors.shape[0]
    retained = min(int(count), number)
    if retained < 1:
        raise ValueError("medoid_count must be at least one")
    squared_norm = np.sum(np.abs(vectors) ** 2, axis=1)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * np.real(vectors @ vectors.conj().T)
    )
    distance = np.sqrt(np.maximum(squared_distance, 0.0))
    medoids = [int(np.argmin(np.sum(distance, axis=1)))]
    while len(medoids) < retained:
        nearest = np.min(distance[:, medoids], axis=1)
        nearest[medoids] = -np.inf
        medoids.append(int(np.argmax(nearest)))
    for _ in range(50):
        assignment = np.argmin(distance[:, medoids], axis=1)
        updated: list[int] = []
        for cluster in range(retained):
            members = np.flatnonzero(assignment == cluster)
            if members.size == 0:
                updated.append(medoids[cluster])
            else:
                within = distance[np.ix_(members, members)]
                updated.append(int(members[np.argmin(np.sum(within, axis=1))]))
        if updated == medoids:
            break
        medoids = updated
    return tuple(medoids)


def _assign_to_medoids(
    vectors: ComplexArray, medoid_indices: Sequence[int]
) -> tuple[int, ...]:
    medoid_vectors = vectors[np.asarray(medoid_indices, dtype=np.int64)]
    vector_norm = np.sum(np.abs(vectors) ** 2, axis=1)
    medoid_norm = np.sum(np.abs(medoid_vectors) ** 2, axis=1)
    squared_distance = (
        vector_norm[:, None]
        + medoid_norm[None, :]
        - 2.0 * np.real(vectors @ medoid_vectors.conj().T)
    )
    return tuple(int(value) for value in np.argmin(squared_distance, axis=1))


def _project_psd_unit_diagonal(matrix: FloatArray) -> FloatArray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    psd = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    diagonal = np.sqrt(np.clip(np.diag(psd), np.finfo(np.float64).eps, None))
    normalized = psd / diagonal[:, None] / diagonal[None, :]
    return 0.5 * (normalized + normalized.T)


def fit_bem_library_statistics(
    nominal_operator: AlignedLeadfield,
    samples: Sequence[GeometryOperatorSample],
    *,
    noise_covariance: ArrayLike,
    residual_rank: int = 8,
    medoid_count: int = 4,
    randomized_svd_seed: int = 0,
) -> BEMLibraryStatistics:
    """Fit residual/operator statistics using ``library_train`` geometries only."""

    sample_tuple = tuple(samples)
    if len(sample_tuple) < 2:
        raise ValueError("at least two library_train geometries are required")
    records = tuple(sample.provenance for sample in sample_tuple)
    # This is intentionally the first sample-dependent operation.
    _assert_library_train_provenance(records)
    for sample in sample_tuple:
        _check_operator_compatible(nominal_operator, sample.operator)
    descriptor_schema = sample_tuple[0].descriptor.names
    if any(sample.descriptor.names != descriptor_schema for sample in sample_tuple):
        raise ValueError("library geometry descriptors do not share one schema")
    operators = np.stack([sample.operator.values for sample in sample_tuple], axis=0)
    residual = operators - nominal_operator.values[None, ...]
    frequencies, channels, groups, orientations = nominal_operator.values.shape
    whitening = _whitening_matrices(noise_covariance, frequencies, channels)
    whitened_nominal = np.einsum("fij,fjgo->figo", whitening, nominal_operator.values)
    whitened_residual = np.einsum("fij,nfjgo->nfigo", whitening, residual)
    residual_mean = np.mean(residual, axis=0)

    covariance = np.empty(
        (frequencies, groups, channels * orientations, channels * orientations),
        dtype=np.complex128,
    )
    for frequency in range(frequencies):
        for group in range(groups):
            values = np.transpose(whitened_residual[:, frequency, :, group, :], (0, 1, 2))
            values = values.reshape(len(sample_tuple), channels * orientations)
            centered = values - np.mean(values, axis=0, keepdims=True)
            covariance[frequency, group] = centered.conj().T @ centered / (
                len(sample_tuple) - 1
            )

    rng = np.random.default_rng(int(randomized_svd_seed))
    effective_rank = min(int(residual_rank), len(sample_tuple), channels * groups * orientations)
    if effective_rank < 1:
        raise ValueError("residual_rank must be at least one")
    bases = np.empty(
        (frequencies, effective_rank, channels * groups * orientations),
        dtype=np.complex128,
    )
    singular_values = np.empty((frequencies, effective_rank), dtype=np.float64)
    uncertainty = np.empty(frequencies, dtype=np.float64)
    residual_effective_rank = np.empty(frequencies, dtype=np.float64)
    condition_number = np.empty(frequencies, dtype=np.float64)
    for frequency in range(frequencies):
        matrix = whitened_residual[:, frequency].reshape(len(sample_tuple), -1)
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
        singular, basis = _randomized_svd(matrix, effective_rank, rng)
        singular_values[frequency] = singular
        bases[frequency] = basis
        total_residual = float(np.mean(np.sum(np.abs(whitened_residual[:, frequency]) ** 2, axis=(1, 2, 3))))
        nominal_energy = float(np.sum(np.abs(whitened_nominal[frequency]) ** 2))
        uncertainty[frequency] = total_residual / max(
            nominal_energy, np.finfo(np.float64).eps
        )
        full_singular = np.linalg.svd(matrix, compute_uv=False)
        energy = full_singular**2
        residual_effective_rank[frequency] = float(
            np.sum(energy) ** 2
            / max(np.sum(energy**2), np.finfo(np.float64).eps)
        )
        nominal_singular = np.linalg.svd(
            whitened_nominal[frequency].reshape(channels, -1), compute_uv=False
        )
        tolerance = np.finfo(np.float64).eps * max(channels, groups * orientations)
        maximum_finite_condition = 1.0 / np.finfo(np.float64).eps
        if (
            nominal_singular.size == 0
            or nominal_singular[0] <= 0.0
            or nominal_singular[-1] <= tolerance * nominal_singular[0]
        ):
            condition_number[frequency] = maximum_finite_condition
        else:
            condition_number[frequency] = float(
                nominal_singular[0] / nominal_singular[-1]
            )

    flat_residual = whitened_residual.reshape(len(sample_tuple), -1)
    medoid_indices = _select_medoids(flat_residual, medoid_count)
    cluster_assignment = _assign_to_medoids(flat_residual, medoid_indices)

    subspace = np.eye(frequencies, dtype=np.float64)
    angle_mean = np.zeros((frequencies, frequencies), dtype=np.float64)
    subspaces: list[ComplexArray] = []
    for frequency in range(frequencies):
        left, singular, _ = np.linalg.svd(
            whitened_nominal[frequency].reshape(channels, -1), full_matrices=False
        )
        retained = max(1, int(np.count_nonzero(singular > singular[0] * 1e-10)))
        subspaces.append(left[:, :retained])
    for left in range(frequencies):
        for right in range(left + 1, frequencies):
            cosine = np.linalg.svd(
                subspaces[left].conj().T @ subspaces[right], compute_uv=False
            )
            cosine = np.clip(cosine.real, 0.0, 1.0)
            similarity = float(np.mean(cosine**2))
            angle = float(np.mean(np.arccos(cosine)))
            subspace[left, right] = subspace[right, left] = similarity
            angle_mean[left, right] = angle_mean[right, left] = angle

    # A squared absolute Gram is entrywise nonnegative and PSD (Schur theorem).
    signatures = np.concatenate(
        [
            whitened_nominal.reshape(frequencies, -1),
            np.moveaxis(whitened_residual, 1, 0).reshape(frequencies, -1),
        ],
        axis=1,
    )
    signature_norm = np.linalg.norm(signatures, axis=1)
    normalized_signature = signatures / np.maximum(
        signature_norm[:, None], np.finfo(np.float64).eps
    )
    complex_gram = normalized_signature @ normalized_signature.conj().T
    raw_redundancy = np.abs(complex_gram) ** 2
    zero_signature = signature_norm <= np.finfo(np.float64).eps
    raw_redundancy[zero_signature, zero_signature] = 1.0
    redundancy = _project_psd_unit_diagonal(raw_redundancy)

    return BEMLibraryStatistics(
        training_geometry_ids=tuple(record.geometry_id for record in records),
        training_geometry_fingerprints=tuple(
            record.geometry_fingerprint for record in records
        ),
        residual_mean=_as_readonly_complex(residual_mean, ndim=4),
        source_dependent_residual_covariance=_as_readonly_complex(covariance, ndim=4),
        residual_basis=_as_readonly_complex(bases, ndim=3),
        residual_singular_values=_as_readonly_float(singular_values, ndim=2),
        medoid_indices=medoid_indices,
        medoid_geometry_ids=tuple(records[index].geometry_id for index in medoid_indices),
        operator_cluster_assignment=cluster_assignment,
        geometry_descriptor_names=descriptor_schema,
        geometry_descriptor_matrix=_as_readonly_float(
            np.vstack([sample.descriptor.values for sample in sample_tuple]), ndim=2
        ),
        frequency_uncertainty=_as_readonly_float(uncertainty, ndim=1),
        frequency_residual_effective_rank=_as_readonly_float(
            residual_effective_rank, ndim=1
        ),
        spectral_condition_number=_as_readonly_float(condition_number, ndim=1),
        subspace_similarity=_as_readonly_float(subspace, ndim=2),
        principal_angle_mean_rad=_as_readonly_float(angle_mean, ndim=2),
        redundancy_matrix_psd=_as_readonly_float(redundancy, ndim=2),
    )


__all__ = [
    "AlignedLeadfield",
    "BEMGeometryParameters",
    "BEMLibraryStatistics",
    "ConductivityParameters",
    "ElectrodePerturbation",
    "GeometryBuildFailure",
    "GeometryBuildResult",
    "GeometryDescriptor",
    "GeometryOnlyResidualPredictor",
    "GeometryOperatorSample",
    "GeometryProvenance",
    "GeometrySplit",
    "GeometrySplitError",
    "GeometrySplitManifest",
    "LeadfieldAlignmentError",
    "LeadfieldAlignmentResult",
    "LibraryLeakageError",
    "NestedSurfaceDeformations",
    "NestedSurfaces",
    "NestingReport",
    "SourceTransportParameters",
    "SurfaceDeformation",
    "SurfaceMesh",
    "SurfaceNestingError",
    "align_leadfield_reference_channels_orientation_scale",
    "assert_geometry_splits_are_disjoint",
    "build_deformed_nested_surfaces",
    "deform_surface_radially",
    "fit_bem_library_statistics",
    "fit_geometry_only_predictor",
    "geometry_descriptor",
    "geometry_fingerprint",
    "least_squares_leadfield_scale",
    "load_fsaverage_nested_surfaces",
    "perturb_electrodes",
    "transport_sources",
    "validate_nested_surfaces",
]
