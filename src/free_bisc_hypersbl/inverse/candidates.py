"""Non-surrogate candidate inverse algorithms.

``SM-RAP/FLEX`` aligns complex signal subspaces across frequency by unitary
Procrustes, scans group/patch Grassmann affinity, and performs recursive RAP
residual refits.  ``ST-VSSI-DGM`` solves a complex group-sparse objective with
frequency and time smoothness using monotone proximal gradient iterations.
"""

from __future__ import annotations

from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .baselines import InverseResult


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _as_snapshots(value: ArrayLike) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("observations must have shape (sensors, snapshots)")
    return array


def _family(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    group_size: int,
) -> tuple[list[ComplexArray], list[ComplexArray], int, int]:
    operators = [np.asarray(item, dtype=np.complex128) for item in leadfields]
    data = [_as_snapshots(item) for item in observations]
    group_size = int(group_size)
    if not operators or len(operators) != len(data):
        raise ValueError("leadfields and observations must be non-empty and aligned")
    n_coefficients = operators[0].shape[1]
    n_snapshots = data[0].shape[1]
    for operator, values in zip(operators, data):
        if operator.ndim != 2 or operator.shape[0] != values.shape[0]:
            raise ValueError("each leadfield must align with its observations")
        if operator.shape[1] != n_coefficients or values.shape[1] != n_snapshots:
            raise ValueError("frequencies must share coefficient and snapshot dimensions")
        if not np.all(np.isfinite(operator)):
            raise ValueError("leadfields contain non-finite values")
    if group_size < 1 or n_coefficients % group_size:
        raise ValueError("coefficient count must be divisible by group_size")
    return operators, data, group_size, n_coefficients // group_size


def _weights(value: ArrayLike | None, count: int) -> FloatArray:
    if value is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape != (count,) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("frequency_weights must be finite and positive")
    return weights / weights.sum()


def _basis(matrix: ComplexArray, rank: int | None = None) -> ComplexArray:
    left, singular, _ = linalg.svd(matrix, full_matrices=False, check_finite=False)
    if not singular.size or singular[0] <= 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.complex128)
    keep = singular > 1.0e-10 * singular[0]
    if rank is not None:
        indices = np.flatnonzero(keep)[: max(1, int(rank))]
    else:
        indices = np.flatnonzero(keep)
    return np.asarray(left[:, indices], dtype=np.complex128)


def _procrustes_consensus(
    spaces: Sequence[ComplexArray],
    weights: FloatArray,
) -> tuple[ComplexArray, list[FloatArray]]:
    rank = min(space.shape[1] for space in spaces)
    if rank < 1:
        raise linalg.LinAlgError("residual signal subspace is empty")
    trimmed = [space[:, :rank] for space in spaces]
    reference = trimmed[int(np.argmax(weights))]
    aligned: list[ComplexArray] = []
    cosines: list[FloatArray] = []
    for space in trimmed:
        left, singular, right_h = linalg.svd(
            space.conj().T @ reference,
            full_matrices=False,
            check_finite=False,
        )
        rotation = left @ right_h
        aligned.append(space @ rotation)
        cosines.append(np.asarray(singular.real, dtype=np.float64))
    mean_basis = sum(weight * space for weight, space in zip(weights, aligned))
    consensus = _basis(np.asarray(mean_basis, dtype=np.complex128), rank=rank)
    return consensus, cosines


def _projector_complement(matrix: ComplexArray, sensors: int) -> ComplexArray:
    basis = _basis(matrix)
    identity = np.eye(sensors, dtype=np.complex128)
    return identity if basis.shape[1] == 0 else identity - basis @ basis.conj().T


def _patch(seed: int, n_groups: int, half_width: int) -> tuple[int, ...]:
    return tuple(range(max(0, seed - half_width), min(n_groups, seed + half_width + 1)))


def _columns(groups: Sequence[int], group_size: int) -> list[int]:
    return [
        column
        for group in groups
        for column in range(group * group_size, (group + 1) * group_size)
    ]


def _group_scores(estimate: ComplexArray, group_size: int) -> FloatArray:
    grouped = estimate.reshape(
        estimate.shape[0],
        estimate.shape[1] // group_size,
        group_size,
        estimate.shape[2],
    )
    return np.asarray(
        np.sqrt(np.sum(np.abs(grouped) ** 2, axis=(0, 2, 3))),
        dtype=np.float64,
    )


def sm_rap_flex(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    *,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
    n_sources: int = 1,
    patch_half_width: int = 0,
) -> InverseResult:
    """Cross-frequency Procrustes/Grassmann SM-RAP with optional FLEX patches."""

    start = perf_counter()
    operators, data, group_size, n_groups = _family(
        leadfields,
        observations,
        group_size,
    )
    weights = _weights(frequency_weights, len(operators))
    n_sources = max(1, min(int(n_sources), n_groups))
    patch_half_width = int(patch_half_width)
    if patch_half_width < 0:
        raise ValueError("patch_half_width cannot be negative")

    selected_seeds: list[int] = []
    selected_groups: set[int] = set()
    selected_scores = np.zeros(n_groups, dtype=np.float64)
    estimates = np.zeros(
        (len(operators), operators[0].shape[1], data[0].shape[1]),
        dtype=np.complex128,
    )
    residuals = [values.copy() for values in data]
    residual_history = [
        float(sum(weight * np.linalg.norm(value, "fro") ** 2 for weight, value in zip(weights, residuals)))
    ]
    procrustes_history: list[list[FloatArray]] = []

    for iteration in range(n_sources):
        projected_data: list[ComplexArray] = []
        complements: list[ComplexArray] = []
        selected_columns = _columns(sorted(selected_groups), group_size)
        for operator, values in zip(operators, data):
            selected_operator = (
                operator[:, selected_columns]
                if selected_columns
                else np.empty((operator.shape[0], 0), dtype=np.complex128)
            )
            complement = _projector_complement(selected_operator, operator.shape[0])
            complements.append(complement)
            projected_data.append(complement @ values)

        rank = max(1, n_sources - iteration)
        signal_spaces = [_basis(value, rank=rank) for value in projected_data]
        consensus, cosines = _procrustes_consensus(signal_spaces, weights)
        procrustes_history.append(cosines)
        signal_projector = consensus @ consensus.conj().T

        scan = np.full(n_groups, -np.inf, dtype=np.float64)
        for seed in range(n_groups):
            patch = _patch(seed, n_groups, patch_half_width)
            if selected_groups.intersection(patch):
                continue
            affinities: list[float] = []
            columns = _columns(patch, group_size)
            for operator, complement in zip(operators, complements):
                candidate_basis = _basis(complement @ operator[:, columns])
                if candidate_basis.shape[1] == 0:
                    affinities.append(0.0)
                else:
                    affinities.append(
                        float(
                            np.trace(
                                candidate_basis.conj().T
                                @ signal_projector
                                @ candidate_basis
                            ).real
                            / candidate_basis.shape[1]
                        )
                    )
            scan[seed] = float(np.dot(weights, affinities))

        chosen = int(np.argmax(scan))
        selected_seeds.append(chosen)
        chosen_patch = _patch(chosen, n_groups, patch_half_width)
        selected_groups.update(chosen_patch)
        selected_scores[chosen] = max(float(scan[chosen]), 0.0)
        selected_columns = _columns(sorted(selected_groups), group_size)

        residual_energy = 0.0
        for frequency, (operator, values, weight) in enumerate(zip(operators, data, weights)):
            coefficients, _, _, _ = linalg.lstsq(
                operator[:, selected_columns],
                values,
                cond=1.0e-10,
                check_finite=False,
            )
            estimates[frequency] = 0.0
            estimates[frequency, selected_columns] = coefficients
            residuals[frequency] = values - operator @ estimates[frequency]
            residual_energy += float(weight * np.linalg.norm(residuals[frequency], "fro") ** 2)
        residual_history.append(residual_energy)

    return InverseResult(
        method="SM-RAP/FLEX",
        source_estimate=estimates,
        source_score=_group_scores(estimates, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={
            "dtype": "complex128",
            "frequency_weights": weights,
            "selected_seeds": np.asarray(selected_seeds, dtype=int),
            "selected_groups": np.asarray(sorted(selected_groups), dtype=int),
            "selection_scores": selected_scores,
            "residual_history": residual_history,
            "procrustes_cosines": procrustes_history,
            "patch_half_width": patch_half_width,
            "alignment": "unitary Procrustes + Grassmann trace affinity",
        },
    )


def _smooth_gradient(
    operators: Sequence[ComplexArray],
    data: Sequence[ComplexArray],
    estimate: ComplexArray,
    lambda_frequency: float,
    lambda_time: float,
) -> ComplexArray:
    gradient = np.stack(
        [operator.conj().T @ (operator @ current - values) for operator, values, current in zip(operators, data, estimate)],
        axis=0,
    )
    if estimate.shape[0] > 1 and lambda_frequency > 0.0:
        differences = estimate[1:] - estimate[:-1]
        gradient[:-1] -= lambda_frequency * differences
        gradient[1:] += lambda_frequency * differences
    if estimate.shape[2] > 1 and lambda_time > 0.0:
        differences = estimate[:, :, 1:] - estimate[:, :, :-1]
        gradient[:, :, :-1] -= lambda_time * differences
        gradient[:, :, 1:] += lambda_time * differences
    return np.asarray(gradient, dtype=np.complex128)


def _group_proximal(
    value: ComplexArray,
    threshold: float,
    group_size: int,
) -> ComplexArray:
    output = value.copy()
    n_groups = value.shape[1] // group_size
    for group in range(n_groups):
        sl = slice(group * group_size, (group + 1) * group_size)
        norm = float(np.linalg.norm(value[:, sl, :]))
        factor = max(0.0, 1.0 - threshold / max(norm, np.finfo(np.float64).tiny))
        output[:, sl, :] *= factor
    return output


def _st_vssi_objective(
    operators: Sequence[ComplexArray],
    data: Sequence[ComplexArray],
    estimate: ComplexArray,
    group_size: int,
    lambda_spatial: float,
    lambda_frequency: float,
    lambda_time: float,
) -> tuple[float, float]:
    data_term = 0.5 * sum(
        float(np.linalg.norm(operator @ current - values, "fro") ** 2)
        for operator, values, current in zip(operators, data, estimate)
    )
    spatial = 0.0
    for group in range(estimate.shape[1] // group_size):
        sl = slice(group * group_size, (group + 1) * group_size)
        spatial += float(np.linalg.norm(estimate[:, sl, :]))
    frequency = (
        0.5 * float(np.linalg.norm(estimate[1:] - estimate[:-1]) ** 2)
        if estimate.shape[0] > 1
        else 0.0
    )
    temporal = (
        0.5 * float(np.linalg.norm(estimate[:, :, 1:] - estimate[:, :, :-1]) ** 2)
        if estimate.shape[2] > 1
        else 0.0
    )
    objective = (
        data_term
        + lambda_spatial * spatial
        + lambda_frequency * frequency
        + lambda_time * temporal
    )
    return float(objective), float(data_term)


def st_vssi_dgm(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    *,
    group_size: int = 3,
    lambda_spatial: float = 1.0e-2,
    lambda_frequency: float = 1.0e-2,
    lambda_time: float = 1.0e-2,
    max_iter: int = 250,
    tolerance: float = 1.0e-7,
) -> InverseResult:
    """Complex spatial-group-sparse, frequency/time-smooth proximal solver."""

    start = perf_counter()
    operators, data, group_size, _ = _family(leadfields, observations, group_size)
    penalties = (float(lambda_spatial), float(lambda_frequency), float(lambda_time))
    if any(not np.isfinite(value) or value < 0.0 for value in penalties):
        raise ValueError("regularization parameters must be finite and non-negative")
    if int(max_iter) < 1 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive")

    estimate = np.zeros(
        (len(operators), operators[0].shape[1], data[0].shape[1]),
        dtype=np.complex128,
    )
    spectral_lipschitz = max(float(linalg.svdvals(operator)[0] ** 2) for operator in operators)
    lipschitz = spectral_lipschitz + 4.0 * lambda_frequency + 4.0 * lambda_time
    step = 0.95 / max(lipschitz, np.finfo(np.float64).tiny)
    objective, data_term = _st_vssi_objective(
        operators,
        data,
        estimate,
        group_size,
        lambda_spatial,
        lambda_frequency,
        lambda_time,
    )
    objective_history = [objective]
    residual_history = [data_term]
    step_history: list[float] = []
    converged = False

    for _ in range(int(max_iter)):
        gradient = _smooth_gradient(
            operators,
            data,
            estimate,
            lambda_frequency,
            lambda_time,
        )
        local_step = step
        accepted = False
        for _ in range(24):
            candidate = _group_proximal(
                estimate - local_step * gradient,
                local_step * lambda_spatial,
                group_size,
            )
            candidate_objective, candidate_data = _st_vssi_objective(
                operators,
                data,
                candidate,
                group_size,
                lambda_spatial,
                lambda_frequency,
                lambda_time,
            )
            if candidate_objective <= objective + 1.0e-12 * max(1.0, abs(objective)):
                accepted = True
                break
            local_step *= 0.5
        if not accepted:
            break

        relative = float(
            np.linalg.norm(candidate - estimate)
            / max(np.linalg.norm(estimate), np.linalg.norm(candidate), 1.0e-15)
        )
        estimate = np.asarray(candidate, dtype=np.complex128)
        objective = float(candidate_objective)
        data_term = float(candidate_data)
        objective_history.append(objective)
        residual_history.append(data_term)
        step_history.append(local_step)
        step = min(local_step * 1.05, 0.99 / max(lipschitz, np.finfo(np.float64).tiny))
        if relative < tolerance:
            converged = True
            break

    return InverseResult(
        method="ST-VSSI-DGM",
        source_estimate=estimate,
        source_score=_group_scores(estimate, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={
            "dtype": "complex128",
            "solver": "monotone complex proximal gradient",
            "objective_history": objective_history,
            "data_residual_history": residual_history,
            "step_history": step_history,
            "iterations": len(objective_history) - 1,
            "converged": converged,
            "lambda_spatial": lambda_spatial,
            "lambda_frequency": lambda_frequency,
            "lambda_time": lambda_time,
        },
    )


SM_RAP_FLEX = sm_rap_flex
ST_VSSI_DGM = st_vssi_dgm
sm_rap_flex_inverse = sm_rap_flex
st_vssi_dgm_inverse = st_vssi_dgm


__all__ = [
    "SM_RAP_FLEX",
    "ST_VSSI_DGM",
    "sm_rap_flex",
    "sm_rap_flex_inverse",
    "st_vssi_dgm",
    "st_vssi_dgm_inverse",
]
