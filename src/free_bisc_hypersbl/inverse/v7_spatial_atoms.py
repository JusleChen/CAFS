"""Truth-independent graph construction and low-rank V7 spatial atoms."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SourceGraph:
    coordinates: NDArray[np.float64]
    neighbors: tuple[tuple[int, ...], ...]
    edge_lengths: tuple[tuple[float, ...], ...]
    fingerprint: str

    @property
    def n_vertices(self) -> int:
        return int(self.coordinates.shape[0])


@dataclass(frozen=True)
class SpatialAtom:
    atom_id: str
    center: int
    kind: str
    scale: int
    basis: ComplexArray
    support: tuple[int, ...]
    effective_rank: int

    def __post_init__(self) -> None:
        basis = np.array(self.basis, dtype=np.complex128, copy=True)
        if basis.ndim != 2 or basis.shape[1] != int(self.effective_rank):
            raise ValueError("basis/effective_rank mismatch")
        if not np.isfinite(basis).all() or basis.shape[1] < 1:
            raise ValueError("atom basis must be finite and nonempty")
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)


def build_source_graph(
    coordinates: ArrayLike,
    edges: Iterable[Sequence[int]],
) -> SourceGraph:
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("coordinates must have shape (n,3) and be finite")
    adjacency = [set() for _ in range(len(xyz))]
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must have two vertices")
        left, right = int(edge[0]), int(edge[1])
        if left == right or min(left, right) < 0 or max(left, right) >= len(xyz):
            raise ValueError("invalid source-graph edge")
        adjacency[left].add(right)
        adjacency[right].add(left)
    if any(not row for row in adjacency):
        raise ValueError("source graph contains an isolated vertex")
    neighbors = tuple(tuple(sorted(row)) for row in adjacency)
    lengths = tuple(
        tuple(float(np.linalg.norm(xyz[i] - xyz[j])) for j in row)
        for i, row in enumerate(neighbors)
    )
    payload = {
        "xyz": np.round(xyz, 15).tolist(),
        "neighbors": neighbors,
        "lengths": [[float(v).hex() for v in row] for row in lengths],
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    xyz = xyz.copy()
    xyz.setflags(write=False)
    return SourceGraph(xyz, neighbors, lengths, fingerprint)


def knn_source_graph(coordinates: ArrayLike, *, neighbors: int = 6) -> SourceGraph:
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n,3)")
    count = int(neighbors)
    if count < 1 or count >= len(xyz):
        raise ValueError("neighbors must be in [1,n_vertices)")
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    edges: set[tuple[int, int]] = set()
    for left in range(len(xyz)):
        for right in np.argsort(distances[left], kind="stable")[:count]:
            edges.add(tuple(sorted((left, int(right)))))
    return build_source_graph(xyz, sorted(edges))


def _hop_neighborhood(graph: SourceGraph, center: int, hops: int) -> tuple[int, ...]:
    visited = {int(center)}
    frontier = {int(center)}
    for _ in range(int(hops)):
        frontier = {j for i in frontier for j in graph.neighbors[i]}.difference(visited)
        visited.update(frontier)
    return tuple(sorted(visited))


def _orthonormal_basis(value: np.ndarray, *, tolerance: float = 1e-10) -> ComplexArray:
    u, singular, _ = np.linalg.svd(np.asarray(value, dtype=np.complex128), full_matrices=False)
    if singular.size == 0:
        raise ValueError("empty atom candidate")
    keep = singular > max(float(singular[0]) * tolerance, np.finfo(float).tiny)
    if not np.any(keep):
        raise ValueError("atom candidate has numerical rank zero")
    return np.asarray(u[:, keep], dtype=np.complex128)


def _embed_radial_modes(
    scalar_modes: np.ndarray,
    normals: np.ndarray,
) -> ComplexArray:
    modes = np.asarray(scalar_modes, dtype=np.float64)
    if modes.ndim == 1:
        modes = modes[:, None]
    embedded = (modes[:, None, :] * normals[:, :, None]).reshape(3 * len(normals), -1)
    return _orthonormal_basis(embedded)


def build_spatial_atom_bank(
    graph: SourceGraph,
    *,
    centers: Sequence[int] | None = None,
    include_lowrank: bool = True,
    lowrank_ranks: Sequence[int] = (2, 4),
) -> list[SpatialAtom]:
    """Build A0-A3 atoms using geometry only.

    The default orientation is the radial/normal direction, giving a rank-one
    point atom and ranks 1/2/4 for coherent/local modes.  Orientation variants
    can be added prospectively without changing graph supports.
    """

    chosen = tuple(range(graph.n_vertices)) if centers is None else tuple(map(int, centers))
    if any(c < 0 or c >= graph.n_vertices for c in chosen):
        raise ValueError("center outside graph")
    normals = np.asarray(graph.coordinates, dtype=float)
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm <= 0):
        raise ValueError("radial orientation requires nonzero source coordinates")
    normals = normals / norm[:, None]
    atoms: list[SpatialAtom] = []
    for center in chosen:
        delta = np.zeros(graph.n_vertices)
        delta[center] = 1.0
        point = _embed_radial_modes(delta, normals)
        atoms.append(SpatialAtom(f"g{center}:point", center, "point", 0, point, (center,), point.shape[1]))

        for hops, kind in ((1, "small_coherent"), (2, "medium_coherent")):
            support = _hop_neighborhood(graph, center, hops)
            distance = np.linalg.norm(graph.coordinates[list(support)] - graph.coordinates[center], axis=1)
            positive = distance[distance > 0]
            scale = float(np.median(positive)) if positive.size else 1.0
            weights = np.zeros(graph.n_vertices)
            weights[list(support)] = np.exp(-distance / max(scale, np.finfo(float).tiny))
            weights /= np.linalg.norm(weights)
            basis = _embed_radial_modes(weights, normals)
            atoms.append(SpatialAtom(f"g{center}:{kind}", center, kind, hops, basis, support, basis.shape[1]))

        if include_lowrank:
            support = _hop_neighborhood(graph, center, 2)
            local = list(support)
            index = {vertex: i for i, vertex in enumerate(local)}
            adjacency = np.zeros((len(local), len(local)), dtype=float)
            for vertex in local:
                for neighbor in graph.neighbors[vertex]:
                    if neighbor in index:
                        adjacency[index[vertex], index[neighbor]] = 1.0
            laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
            _, vectors = np.linalg.eigh(laplacian)
            for requested in lowrank_ranks:
                rank = min(int(requested), len(local))
                if rank < 2:
                    continue
                scalar = np.zeros((graph.n_vertices, rank))
                scalar[local] = vectors[:, :rank]
                basis = _embed_radial_modes(scalar, normals)
                atoms.append(
                    SpatialAtom(
                        f"g{center}:local_lowrank_r{rank}", center, "local_lowrank", 2,
                        basis, support, basis.shape[1]
                    )
                )
    return atoms


def sensor_component_coherence(left: ArrayLike, right: ArrayLike) -> float:
    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    ka = a @ a.conj().T
    kb = b @ b.conj().T
    return float(abs(np.vdot(ka, kb)) / max(np.linalg.norm(ka) * np.linalg.norm(kb), 1e-300))


def deduplicate_atoms_by_sensor_coherence(
    atoms: Sequence[SpatialAtom],
    sensor_blocks: Sequence[ArrayLike],
    *,
    threshold: float = 0.995,
) -> tuple[list[SpatialAtom], list[int]]:
    if len(atoms) != len(sensor_blocks):
        raise ValueError("atoms and sensor_blocks must align")
    order = sorted(range(len(atoms)), key=lambda i: (atoms[i].effective_rank, atoms[i].scale, atoms[i].atom_id))
    kept: list[int] = []
    for index in order:
        duplicate = any(
            atoms[index].center == atoms[other].center
            and sensor_component_coherence(sensor_blocks[index], sensor_blocks[other]) > threshold
            for other in kept
        )
        if not duplicate:
            kept.append(index)
    kept.sort()
    return [atoms[i] for i in kept], kept


__all__ = [
    "SourceGraph", "SpatialAtom", "build_source_graph", "knn_source_graph",
    "build_spatial_atom_bank", "sensor_component_coherence",
    "deduplicate_atoms_by_sensor_coherence",
]
