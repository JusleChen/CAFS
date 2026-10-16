"""Truth-free event-driven candidate hierarchy for V8 HQSL."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .v8_component_projection import ProjectionBank


@dataclass(frozen=True, slots=True)
class CandidatePriority:
    derivative: float
    atom_index: int
    profile_index: int
    state_version: int


class CandidateTree:
    """Open point atoms globally and patches only through their parent chain.

    The class accepts scores and active-set events only.  There is deliberately
    no source-truth, morphology, support-label, or localization argument.
    """

    def __init__(self, bank: ProjectionBank) -> None:
        self.bank = bank
        self._point = np.asarray([atom.kind == "point" for atom in bank.atoms])
        self._small = np.asarray([atom.kind == "small_coherent" for atom in bank.atoms])
        self._medium = np.asarray([atom.kind == "medium_coherent" for atom in bank.atoms])
        self._enabled_atoms = self._point.copy()
        self._index = {
            (atom.center, atom.kind): index for index, atom in enumerate(bank.atoms)
        }
        self._heap: list[tuple[float, int, int, int, int]] = []
        self._serial = 0
        self.refreshes = 0
        self.stale_pops = 0

    def open_from_active(self, active_atom_indices: Iterable[int]) -> None:
        """Advance point -> small -> medium for active centers."""

        for value in active_atom_indices:
            index = int(value)
            atom = self.bank.atoms[index]
            if atom.kind == "point":
                child = self._index.get((atom.center, "small_coherent"))
                if child is None:
                    child = self._index.get((atom.center, "medium_coherent"))
            elif atom.kind == "small_coherent":
                child = self._index.get((atom.center, "medium_coherent"))
            else:
                child = None
            if child is not None:
                self._enabled_atoms[child] = True

    def open_small_for_centers(self, centers: Iterable[int]) -> None:
        """Open small patches selected by a truth-free residual/posterior rule."""

        for value in centers:
            index = self._index.get((int(value), "small_coherent"))
            if index is not None:
                self._enabled_atoms[index] = True

    def eligible_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.bank.valid & self._enabled_atoms[:, None], dtype=bool)

    def push_derivatives(
        self,
        derivatives: ArrayLike,
        *,
        state_version: int,
        top_k: int,
        excluded: Iterable[tuple[int, int]] = (),
    ) -> int:
        values = np.asarray(derivatives, dtype=float)
        if values.shape != self.bank.valid.shape:
            raise ValueError("derivative matrix does not match projection bank")
        allowed = self.eligible_mask()
        for left, right in excluded:
            allowed[int(left), int(right)] = False
        flat = np.flatnonzero(allowed)
        if flat.size == 0:
            return 0
        count = min(max(int(top_k), 0), flat.size)
        if count == 0:
            return 0
        selected = flat[np.argpartition(values.ravel()[flat], count - 1)[:count]]
        for flat_index in selected:
            atom, profile = np.unravel_index(int(flat_index), values.shape)
            self._serial += 1
            heapq.heappush(
                self._heap,
                (float(values[atom, profile]), self._serial, int(atom), int(profile), int(state_version)),
            )
        self.refreshes += 1
        return int(count)

    def pop_fresh(
        self,
        *,
        state_version: int,
        refresh_one: Callable[[int, int], float],
    ) -> CandidatePriority | None:
        """Return a current priority, lazily refreshing stale heap entries."""

        while self._heap:
            derivative, _, atom, profile, version = heapq.heappop(self._heap)
            if not self.eligible_mask()[atom, profile]:
                continue
            if version != int(state_version):
                self.stale_pops += 1
                current = float(refresh_one(atom, profile))
                self._serial += 1
                heapq.heappush(
                    self._heap,
                    (current, self._serial, atom, profile, int(state_version)),
                )
                continue
            return CandidatePriority(float(derivative), atom, profile, version)
        return None

    def full_dictionary_indices(self) -> tuple[tuple[int, int], ...]:
        """The final KKT domain is independent of the currently open subtree."""

        return self.bank.valid_component_indices()


__all__ = ["CandidatePriority", "CandidateTree"]
