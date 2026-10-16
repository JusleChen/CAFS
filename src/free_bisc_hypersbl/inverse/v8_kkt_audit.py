"""Vectorized exact full-dictionary KKT audit and repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .v8_component_projection import ProjectionBank
from .v8_profile_screening import profile_derivative_matrix, vectorized_frequency_base_scores
from .v8_sequential_evidence import SequentialEvidenceState


@dataclass(frozen=True, slots=True)
class KKTAudit:
    derivatives: NDArray[np.float64]
    valid_count: int
    covered_count: int
    inactive_violations: tuple[tuple[int, int], ...]
    active_violations: tuple[str, ...]
    maximum_inactive_violation: float
    maximum_active_residual: float
    tolerance: float

    @property
    def complete(self) -> bool:
        return self.covered_count == self.valid_count

    @property
    def passed(self) -> bool:
        return self.complete and not self.inactive_violations and not self.active_violations


def full_dictionary_derivatives(
    bank: ProjectionBank,
    state: SequentialEvidenceState,
) -> NDArray[np.float64]:
    per_frequency = np.empty((bank.atom_count, len(state.statistics)), dtype=float)
    for frequency, (quotient, statistic, covariance) in enumerate(
        zip(bank.quotient_factors, state.statistics, state.covariances)
    ):
        per_frequency[:, frequency] = vectorized_frequency_base_scores(
            bank.projected_bases,
            quotient.R,
            covariance,
            statistic.Z,
        )
    result = profile_derivative_matrix(
        per_frequency,
        bank.profile_matrix,
        observability_denominators=bank.denominators,
        frequency_weights=state.frequency_weights,
    )
    result = np.asarray(result, dtype=float)
    result[~bank.valid] = np.nan
    return result


def audit_full_dictionary(
    bank: ProjectionBank,
    state: SequentialEvidenceState,
    *,
    tolerance: float,
) -> KKTAudit:
    tol = float(tolerance)
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tolerance must be finite and positive")
    derivatives = full_dictionary_derivatives(bank, state)
    inactive: list[tuple[int, int]] = []
    active: list[str] = []
    max_inactive = 0.0
    max_active = 0.0
    base_ids: set[str] = set()
    for atom_index, profile_index in bank.valid_component_indices():
        component_id = (
            f"{bank.atoms[atom_index].atom_id}|{bank.profiles[profile_index].name}"
        )
        derivative = float(derivatives[atom_index, profile_index])
        if component_id in state.theta:
            base_ids.add(component_id)
            residual = abs(derivative)
            max_active = max(max_active, residual)
            if residual > tol:
                active.append(component_id)
        elif derivative < -tol:
            inactive.append((atom_index, profile_index))
            max_inactive = max(max_inactive, -derivative)
    # Conditional AREE children are outside the frozen base bank but remain
    # active nonnegative coordinates and therefore require stationarity too.
    for component_id, component in state.components.items():
        if component_id in base_ids:
            continue
        residual = abs(state.derivative(component))
        max_active = max(max_active, residual)
        if residual > tol:
            active.append(component_id)
    return KKTAudit(
        derivatives=derivatives,
        valid_count=int(np.count_nonzero(bank.valid)),
        covered_count=int(np.count_nonzero(np.isfinite(derivatives))),
        inactive_violations=tuple(inactive),
        active_violations=tuple(active),
        maximum_inactive_violation=float(max_inactive),
        maximum_active_residual=float(max_active),
        tolerance=tol,
    )


def repair_full_dictionary_kkt(
    bank: ProjectionBank,
    state: SequentialEvidenceState,
    *,
    tolerance: float,
    theta_max: float,
    maximum_repairs: int = 128,
) -> tuple[KKTAudit, int]:
    """Repair the worst exact coordinate violation, then re-audit globally."""

    repairs = 0
    for _ in range(int(maximum_repairs) + 1):
        audit = audit_full_dictionary(bank, state, tolerance=tolerance)
        if audit.passed or repairs >= int(maximum_repairs):
            return audit, repairs
        if audit.inactive_violations:
            atom_index, profile_index = min(
                audit.inactive_violations,
                key=lambda pair: audit.derivatives[pair[0], pair[1]],
            )
            component = bank.component(atom_index, profile_index)
        else:
            component_id = max(
                audit.active_violations,
                key=lambda identifier: abs(state.derivative(state.components[identifier])),
            )
            component = state.components[component_id]
        optimum = state.coordinate_optimum(component, theta_max=theta_max)
        record = state.apply_coordinate(component, optimum, minimum_gain=0.0)
        if record is None:
            return audit, repairs
        repairs += 1
    raise AssertionError("unreachable KKT repair termination")


__all__ = [
    "KKTAudit",
    "audit_full_dictionary",
    "full_dictionary_derivatives",
    "repair_full_dictionary_kkt",
]
