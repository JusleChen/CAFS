"""Conditional Adaptive Residual Eigenmode Expansion (AREE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .v7_spatial_atoms import SourceGraph
from .v8_component_projection import ProjectedComponent
from .v8_harmonic_quotient import HarmonicQuotientFactor
from .v8_sequential_evidence import CoordinateOptimum, SequentialEvidenceState


def _local_radial_embedding(graph: SourceGraph, support: tuple[int, ...]) -> np.ndarray:
    source_dimension = 3 * graph.n_vertices
    embedding = np.zeros((source_dimension, len(support)), dtype=np.complex128)
    normals = graph.coordinates / np.linalg.norm(graph.coordinates, axis=1, keepdims=True)
    for column, vertex in enumerate(support):
        embedding[3 * vertex : 3 * vertex + 3, column] = normals[vertex]
    return embedding


def _local_laplacian(graph: SourceGraph, support: tuple[int, ...]) -> np.ndarray:
    lookup = {vertex: index for index, vertex in enumerate(support)}
    adjacency = np.zeros((len(support), len(support)), dtype=float)
    for vertex in support:
        for neighbor in graph.neighbors[vertex]:
            if neighbor in lookup:
                adjacency[lookup[vertex], lookup[neighbor]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    return np.diag(adjacency.sum(axis=1)) - adjacency


def _observable_inner_constraints(
    parent: np.ndarray,
    embedding: np.ndarray,
    vh: np.ndarray,
    quotients: Sequence[HarmonicQuotientFactor],
    frequency_weights: np.ndarray,
) -> np.ndarray:
    source_constraint = parent.conj().T @ embedding
    observable = np.zeros((parent.shape[1], embedding.shape[1]), dtype=np.complex128)
    parent_harmonic = vh @ parent
    embedded_harmonic = vh @ embedding
    for weight, quotient in zip(frequency_weights, quotients):
        observable += float(weight) * (
            (quotient.R @ parent_harmonic).conj().T
            @ (quotient.R @ embedded_harmonic)
        )
    return np.vstack((source_constraint, observable))


def build_aree_modes(
    parent: ProjectedComponent,
    graph: SourceGraph,
    vh: ArrayLike,
    quotients: Sequence[HarmonicQuotientFactor],
    profile_rho: ArrayLike,
    *,
    profile_index: int,
    profile_name: str,
    profile_kind: str,
    maximum_modes: int,
    observability_floor: float,
    frequency_weights: ArrayLike | None = None,
    orthogonality_mode: str = "source_and_observable",
) -> tuple[ProjectedComponent, ...]:
    """Construct local rank-one modes satisfying both orthogonality metrics."""

    if parent.kind not in {"small_coherent", "medium_coherent"}:
        return ()
    support = tuple(parent.support)
    if len(support) < 3 or int(maximum_modes) < 1:
        return ()
    harmonic_map = np.asarray(vh, dtype=np.complex128)
    rho = np.asarray(profile_rho, dtype=float).reshape(-1)
    weights = (
        np.ones(len(quotients), dtype=float)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).reshape(-1)
    )
    if rho.shape != (len(quotients),) or weights.shape != rho.shape:
        raise ValueError("AREE frequencies do not align")
    embedding = _local_radial_embedding(graph, support)
    if orthogonality_mode == "source_and_observable":
        constraints = _observable_inner_constraints(
            parent.source_basis, embedding, harmonic_map, quotients, weights
        )
    elif orthogonality_mode == "source_only":
        constraints = parent.source_basis.conj().T @ embedding
    else:
        raise ValueError("unknown AREE orthogonality mode")
    null_basis = linalg.null_space(constraints, rcond=1e-11)
    if null_basis.shape[1] == 0:
        return ()
    laplacian = _local_laplacian(graph, support)
    reduced_laplacian = null_basis.conj().T @ laplacian @ null_basis
    _, eigenvectors = np.linalg.eigh(0.5 * (reduced_laplacian + reduced_laplacian.conj().T))
    local_modes = null_basis @ eigenvectors
    children: list[ProjectedComponent] = []
    for mode_index in range(min(int(maximum_modes), local_modes.shape[1])):
        source = embedding @ local_modes[:, mode_index : mode_index + 1]
        norm = float(np.linalg.norm(source))
        if norm <= np.finfo(float).tiny:
            continue
        source /= norm
        harmonic = harmonic_map @ source
        energies = np.asarray(
            [np.linalg.norm(quotient.R @ harmonic) ** 2 for quotient in quotients],
            dtype=float,
        )
        observability = float(np.dot(rho, energies) / np.sum(rho))
        if observability < float(observability_floor):
            continue
        denominator = float(np.sqrt(observability))
        normalized_source = np.asarray(source / denominator, dtype=np.complex128)
        factors = tuple(
            np.asarray(
                np.sqrt(rho[frequency]) * quotient.R @ harmonic / denominator,
                dtype=np.complex128,
            )
            for frequency, quotient in enumerate(quotients)
        )
        child_number = len(children) + 1
        children.append(
            ProjectedComponent(
                component_id=f"{parent.component_id}|aree{child_number}",
                atom_index=-1,
                profile_index=int(profile_index),
                atom_id=f"{parent.atom_id}:aree{child_number}",
                parent_id=parent.component_id,
                center=parent.center,
                scale=parent.scale,
                kind="aree_child",
                profile_name=profile_name,
                profile_kind=profile_kind,
                source_basis=normalized_source,
                reduced_factors=factors,
                support=support,
                observability=observability,
                aree_mode=child_number,
            )
        )
    return tuple(children)


def component_sensor_coherence(
    left: ProjectedComponent,
    right: ProjectedComponent,
) -> float:
    numerator = 0.0j
    left_norm = 0.0
    right_norm = 0.0
    for left_factor, right_factor in zip(left.reduced_factors, right.reduced_factors):
        left_covariance = left_factor @ left_factor.conj().T
        right_covariance = right_factor @ right_factor.conj().T
        numerator += np.vdot(left_covariance, right_covariance)
        left_norm += float(np.linalg.norm(left_covariance, "fro") ** 2)
        right_norm += float(np.linalg.norm(right_covariance, "fro") ** 2)
    return float(abs(numerator) / max(np.sqrt(left_norm * right_norm), 1e-300))


@dataclass(frozen=True, slots=True)
class QualifiedAREEChild:
    component: ProjectedComponent
    optimum: CoordinateOptimum
    maximum_active_coherence: float


def qualify_aree_children(
    state: SequentialEvidenceState,
    parent: ProjectedComponent,
    children: Sequence[ProjectedComponent],
    *,
    theta_max: float,
    parent_ratio_threshold: float,
    residual_envelope: float | None,
    residual_percentile_threshold: float,
    coherence_threshold: float,
    derivative_margin: float,
    gain_threshold: float,
    child_limit: int,
) -> tuple[QualifiedAREEChild, ...]:
    """Apply the frozen numerical spawn gate without any morphology label."""

    parent_theta = float(state.theta.get(parent.component_id, 0.0))
    parent_gate = parent_theta / float(theta_max) >= float(parent_ratio_threshold)
    residual_gate = (
        residual_envelope is not None
        and float(residual_envelope) >= float(residual_percentile_threshold)
    )
    if parent_theta <= 0.0 or not (parent_gate or residual_gate):
        return ()
    accepted: list[QualifiedAREEChild] = []
    active = tuple(state.components.values())
    for child in children:
        coherence = max(
            (component_sensor_coherence(child, other) for other in active),
            default=0.0,
        )
        if coherence >= float(coherence_threshold):
            continue
        derivative = state.derivative(child)
        if derivative >= -float(derivative_margin):
            continue
        optimum = state.coordinate_optimum(child, theta_max=theta_max)
        if -optimum.delta_objective <= float(gain_threshold):
            continue
        accepted.append(QualifiedAREEChild(child, optimum, coherence))
        if len(accepted) >= int(child_limit):
            break
    return tuple(accepted)


__all__ = [
    "QualifiedAREEChild",
    "build_aree_modes",
    "component_sensor_coherence",
    "qualify_aree_children",
]
