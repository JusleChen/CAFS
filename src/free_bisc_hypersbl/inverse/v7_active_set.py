"""KKT directional derivatives and deterministic V7 active-set decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .v7_covariance_components import CovarianceComponent
from .v7_covariance_components import assemble_sensor_covariances
from .v7_em_mm_solver import EMResult, fixed_active_em


def inactive_directional_derivative(
    observations: Sequence[ArrayLike | None],
    sigmas: Sequence[ArrayLike],
    component: CovarianceComponent,
    *,
    frequency_weights: ArrayLike | None = None,
) -> float:
    weights = (
        np.ones(len(observations), dtype=float)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (len(observations),):
        raise ValueError("frequency weights must align with observations")
    total = 0.0
    for f, y_value in enumerate(observations):
        if y_value is None:
            continue
        y = np.asarray(y_value, dtype=np.complex128)
        if y.ndim == 1: y = y[:,None]
        sigma = np.asarray(sigmas[f], dtype=np.complex128)
        factor = linalg.cho_factor(sigma, lower=True, check_finite=False)
        a = component.sensor_factors[f]
        inv_a = linalg.cho_solve(factor, a, check_finite=False)
        first = float(np.trace(a.conj().T @ inv_a).real)
        inv_y = linalg.cho_solve(factor, y, check_finite=False)
        second = float(np.linalg.norm(a.conj().T @ inv_y, "fro")**2 / y.shape[1])
        total += weights[f] * y.shape[1] * (first - second)
    return float(total)


def inactive_directional_derivatives(
    observations: Sequence[ArrayLike | None],
    sigmas: Sequence[ArrayLike],
    components: Sequence[CovarianceComponent],
    *,
    frequency_weights: ArrayLike | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Screen inactive components in bounded factor batches.

    Sigma factorization and ``Sigma^-1 Y`` are shared by the whole inactive
    dictionary.  Only a bounded ``M x sum(rank_batch)`` factor matrix is
    assembled, so this is not a dense all-candidate optimizer.
    """

    if not components:
        return np.empty(0, dtype=float)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    weights = (
        np.ones(len(observations), dtype=float)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).reshape(-1)
    )
    if weights.shape != (len(observations),):
        raise ValueError("frequency weights must align with observations")
    derivatives = np.zeros(len(components), dtype=float)
    for frequency, y_value in enumerate(observations):
        if y_value is None:
            continue
        y = np.asarray(y_value, dtype=np.complex128)
        if y.ndim == 1:
            y = y[:, None]
        sigma = np.asarray(sigmas[frequency], dtype=np.complex128)
        factor = linalg.cho_factor(sigma, lower=True, check_finite=False)
        inv_y = linalg.cho_solve(factor, y, check_finite=False)
        for start in range(0, len(components), int(batch_size)):
            batch = components[start : start + int(batch_size)]
            blocks = [component.sensor_factors[frequency] for component in batch]
            design = np.column_stack(blocks)
            inv_design = linalg.cho_solve(factor, design, check_finite=False)
            cursor = 0
            for offset, block in enumerate(blocks):
                stop = cursor + block.shape[1]
                inv_block = inv_design[:, cursor:stop]
                first = float(np.trace(block.conj().T @ inv_block).real)
                second = float(
                    np.linalg.norm(block.conj().T @ inv_y, "fro") ** 2 / y.shape[1]
                )
                derivatives[start + offset] += (
                    weights[frequency] * y.shape[1] * (first - second)
                )
                cursor = stop
    return derivatives


@dataclass(frozen=True)
class KKTStatus:
    status: str
    minimum_inactive_derivative: float
    scanned: int
    total_inactive: int
    coverage: float
    active_cap_reached: bool


def select_atoms_to_add(
    derivatives: ArrayLike,
    *,
    b_add: int,
    delta_add: float,
    active_count: int,
    active_max: int,
    eligible_mask: ArrayLike | None = None,
) -> list[int]:
    values = np.asarray(derivatives, dtype=float)
    allowed = (
        np.ones(values.shape, dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    if allowed.shape != values.shape:
        raise ValueError("eligible_mask must align with derivatives")
    eligible = np.flatnonzero((values < -float(delta_add)) & allowed)
    order = eligible[np.argsort(values[eligible], kind="stable")]
    capacity = max(0, int(active_max)-int(active_count))
    return [int(v) for v in order[:min(int(b_add),capacity)]]


def kkt_status(derivatives: ArrayLike, *, tol_kkt: float, total_inactive: int | None = None, active_cap_reached: bool = False) -> KKTStatus:
    values=np.asarray(derivatives,dtype=float).reshape(-1)
    total=len(values) if total_inactive is None else int(total_inactive)
    coverage=len(values)/max(total,1)
    minimum=float(values.min()) if len(values) else float("inf")
    stationary=coverage==1.0 and minimum>=-float(tol_kkt)
    status="KKT_STATIONARY_WITHIN_TOL" if stationary else ("PARTIAL_DICTIONARY_KKT_NOT_CLAIMED" if coverage<1 else "NEGATIVE_INACTIVE_DERIVATIVE_REMAINS")
    return KKTStatus(status,minimum,len(values),total,float(coverage),bool(active_cap_reached))


@dataclass(frozen=True)
class ActiveSetResult:
    fit: EMResult
    active_indices: tuple[int, ...]
    kkt: KKTStatus
    additions: int
    prunes: int
    outer_iterations: int


def update_prune_hysteresis(
    theta: ArrayLike,
    *,
    below_counts: ArrayLike,
    ages: ArrayLike,
    tau_prune: float,
    consecutive_required: int,
    hysteresis_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance deterministic prune counters and return eligible local indices.

    This performs only the mass and hysteresis part of the registered rule.
    ``fit_active_set`` separately accepts a deletion only after refitting and
    checking that the evidence objective does not increase beyond tolerance.
    """

    values = np.asarray(theta, dtype=float).reshape(-1)
    counts = np.asarray(below_counts, dtype=int).reshape(-1).copy()
    atom_ages = np.asarray(ages, dtype=int).reshape(-1)
    if values.shape != counts.shape or values.shape != atom_ages.shape or values.size == 0:
        raise ValueError("theta, below_counts and ages must be aligned non-empty vectors")
    if np.any(values < 0.0) or tau_prune < 0.0:
        raise ValueError("theta and tau_prune must be nonnegative")
    maximum = max(float(values.max()), np.finfo(float).tiny)
    low = values / maximum < float(tau_prune)
    counts = np.where(low, counts + 1, 0)
    eligible = np.flatnonzero(
        (counts >= int(consecutive_required))
        & (atom_ages >= int(hysteresis_period))
    )
    return counts, eligible.astype(int)


def fit_active_set(
    observations: Sequence[ArrayLike | None],
    dictionary: Sequence[CovarianceComponent],
    *,
    initial_active: Sequence[int],
    b_add: int = 4,
    active_max: int = 64,
    delta_add: float = 1e-6,
    tol_kkt: float = 1e-5,
    max_outer: int = 20,
    tau_prune: float = 1e-6,
    prune_consecutive: int = 3,
    hysteresis_period: int = 2,
    prune_objective_tolerance: float = 1e-8,
    delta_specific: float = 0.0,
    em_kwargs: dict | None = None,
) -> ActiveSetResult:
    active=sorted(set(map(int,initial_active)))
    if not active or any(i<0 or i>=len(dictionary) for i in active):
        raise ValueError("initial_active must contain valid dictionary indices")
    additions=0; prunes=0; fit=None; final_kkt=None
    below_count = {index: 0 for index in active}
    age = {index: int(hysteresis_period) for index in active}
    solver_kwargs = dict(em_kwargs or {})
    if "theta0" in solver_kwargs or "noise0" in solver_kwargs:
        raise ValueError("active-set controls warm-start parameters internally")
    frequency_weights = solver_kwargs.get("frequency_weights")
    for outer in range(1,int(max_outer)+1):
        current=[dictionary[i] for i in active]
        fit=fixed_active_em(observations,current,**solver_kwargs)
        for index in active:
            age[index] = age.get(index, 0) + 1
        counts, eligible_prune = update_prune_hysteresis(
            fit.theta,
            below_counts=[below_count.get(index, 0) for index in active],
            ages=[age.get(index, 0) for index in active],
            tau_prune=tau_prune,
            consecutive_required=prune_consecutive,
            hysteresis_period=hysteresis_period,
        )
        below_count.update({index: int(counts[position]) for position, index in enumerate(active)})
        # Try lowest-mass removals first.  Each accepted removal is refitted and
        # evidence checked, so mass alone never decides a scientific deletion.
        for local_index in sorted(eligible_prune, key=lambda value: fit.theta[value]):
            global_index = active[int(local_index)]
            reduced = [index for index in active if index != global_index]
            if not reduced:
                continue
            reduced_theta = np.asarray(
                [fit.theta[position] for position, index in enumerate(active) if index != global_index]
            )
            proposal = fixed_active_em(
                observations,
                [dictionary[index] for index in reduced],
                theta0=reduced_theta,
                noise0=fit.noise,
                **solver_kwargs,
            )
            tolerance = float(prune_objective_tolerance) * max(1.0, abs(fit.objective[-1]))
            if proposal.objective[-1] <= fit.objective[-1] + tolerance:
                active = reduced
                below_count.pop(global_index, None)
                age.pop(global_index, None)
                fit = proposal
                prunes += 1
                break
        current=[dictionary[i] for i in active]
        sigmas=assemble_sensor_covariances(current,fit.theta,fit.noise)
        inactive=[i for i in range(len(dictionary)) if i not in active]
        derivatives=inactive_directional_derivatives(
            observations,
            sigmas,
            [dictionary[i] for i in inactive],
            frequency_weights=frequency_weights,
        )
        cap=len(active)>=int(active_max)
        final_kkt=kkt_status(derivatives,tol_kkt=tol_kkt,total_inactive=len(inactive),active_cap_reached=cap)
        if final_kkt.status=="KKT_STATIONARY_WITHIN_TOL" or not inactive or cap:
            return ActiveSetResult(fit,tuple(active),final_kkt,additions,prunes,outer)
        if outer == int(max_outer):
            # Never return atoms that have not received a fixed-active fit.
            return ActiveSetResult(fit,tuple(active),final_kkt,additions,prunes,outer)
        active_shared = {
            dictionary[index].spatial_atom_id
            for index in active
            if dictionary[index].profile.kind == "shared"
        }
        inactive_shared = {
            dictionary[index].spatial_atom_id: derivatives[position]
            for position, index in enumerate(inactive)
            if dictionary[index].profile.kind == "shared"
        }
        shared_first = np.ones(len(inactive), dtype=bool)
        for position, index in enumerate(inactive):
            component = dictionary[index]
            if component.profile.kind == "specific" and component.spatial_atom_id not in active_shared:
                shared_derivative = inactive_shared.get(component.spatial_atom_id, np.inf)
                shared_first[position] = bool(
                    derivatives[position] < shared_derivative - float(delta_specific)
                )
        local=select_atoms_to_add(
            derivatives,
            b_add=b_add,
            delta_add=delta_add,
            active_count=len(active),
            active_max=active_max,
            eligible_mask=shared_first,
        )
        if not local:
            return ActiveSetResult(fit,tuple(active),final_kkt,additions,prunes,outer)
        added = [inactive[i] for i in local]
        active.extend(added); active.sort(); additions+=len(local)
        for index in added:
            below_count[index] = 0
            age[index] = 0
    assert fit is not None and final_kkt is not None
    return ActiveSetResult(fit,tuple(active),final_kkt,additions,prunes,int(max_outer))


__all__=["inactive_directional_derivative","inactive_directional_derivatives","KKTStatus","select_atoms_to_add","kkt_status","ActiveSetResult","update_prune_hysteresis","fit_active_set"]
