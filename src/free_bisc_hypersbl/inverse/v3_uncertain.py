"""V3 uncertain-leadfield Champagne for proper-complex EEG data.

The uncertainty term in this module is deliberately source dependent.  If
``D[f, g]`` is the sensor covariance induced by a unit-variance source group,
then the moment-matched discrepancy covariance is

``Q_f(gamma) = sum_g gamma[g] D[f, g]``.

Consequently, an inactive source cannot contribute an unconditional BAE-like
noise floor.  All covariance inputs are projected onto the Hermitian PSD cone
and may be rank capped before fitting.  The implementation is a dense
reference/validation implementation; production active-set and Woodbury
variants can share the public covariance and result contracts below.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from . import baselines
from .complex_sbl import complex_gaussian_posterior


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# The grid is a protocol contract, rather than an open tuning range.
UL_ETA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
ETA_GRID = UL_ETA_GRID


def _readonly_complex(value: ArrayLike) -> ComplexArray:
    out = np.array(value, dtype=np.complex128, copy=True)
    out.setflags(write=False)
    return out


def _readonly_float(value: ArrayLike) -> FloatArray:
    out = np.array(value, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _as_snapshots(value: ArrayLike) -> ComplexArray:
    out = np.asarray(value, dtype=np.complex128)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or not np.all(np.isfinite(out)):
        raise ValueError("observations must be finite with shape (sensors, snapshots)")
    return out


def _operator_family(value: Iterable[ArrayLike] | ArrayLike) -> list[ComplexArray]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim == 2:
            items = [array]
        elif array.ndim == 3:
            items = [array[index] for index in range(array.shape[0])]
        elif array.ndim == 4:
            # BEM-library convention: frequency, sensor, source-group,
            # orientation.  Champagne uses the flattened source dimension.
            items = [
                array[index].reshape(array.shape[1], -1)
                for index in range(array.shape[0])
            ]
        else:
            raise ValueError("leadfields must have 2, 3, or 4 dimensions")
    else:
        items = [np.asarray(item, dtype=np.complex128) for item in value]
    if not items:
        raise ValueError("at least one leadfield is required")
    if any(item.ndim != 2 or not np.all(np.isfinite(item)) for item in items):
        raise ValueError("leadfields must contain finite two-dimensional matrices")
    return items


def _observation_family(value: Iterable[ArrayLike] | ArrayLike) -> list[ComplexArray]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim in (1, 2):
            items = [_as_snapshots(array)]
        elif array.ndim == 3:
            items = [_as_snapshots(array[index]) for index in range(array.shape[0])]
        else:
            raise ValueError("observations must have 1, 2, or 3 dimensions")
    else:
        items = [_as_snapshots(item) for item in value]
    if not items:
        raise ValueError("at least one observation array is required")
    return items


def _family_problem(
    leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    group_size: int,
) -> tuple[list[ComplexArray], list[ComplexArray], int, int]:
    operators = _operator_family(leadfields)
    data = _observation_family(observations)
    group_size = int(group_size)
    if len(operators) != len(data):
        raise ValueError("leadfields and observations must align by frequency")
    coefficients = operators[0].shape[1]
    snapshots = data[0].shape[1]
    if group_size < 1 or coefficients % group_size:
        raise ValueError("source columns must be divisible by group_size")
    for operator, target in zip(operators, data):
        if operator.shape[0] != target.shape[0]:
            raise ValueError("leadfield and observation sensor dimensions do not match")
        if operator.shape[1] != coefficients or target.shape[1] != snapshots:
            raise ValueError("all frequencies must share source and snapshot dimensions")
    return operators, data, group_size, coefficients // group_size


def safe_frequency_weights(value: ArrayLike | None, count: int) -> FloatArray:
    """Validate and normalize strictly positive Safe-FreeSIGM weights."""

    if value is None:
        weights = np.full(int(count), 1.0 / int(count), dtype=np.float64)
    else:
        weights = np.asarray(value, dtype=np.float64).reshape(-1)
        if weights.shape != (int(count),):
            raise ValueError("frequency_weights must have one entry per frequency")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("Safe-FreeSIGM frequency weights must be finite and positive")
        weights = weights / weights.sum()
    return np.asarray(weights, dtype=np.float64)


def freesigm_effective_weights(value: ArrayLike | None, count: int) -> FloatArray:
    """Return the evidence-scale contract ``F * w_safe`` (sum equals ``F``)."""

    safe = safe_frequency_weights(value, count)
    return np.asarray(int(count) * safe, dtype=np.float64)


def stack_freesigm_family(
    leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    """Stack a family using the exact ``sqrt(F*w_safe)`` interface.

    This helper is the single eta-zero bridge to the historical
    :func:`inverse.baselines.champagne_sbl` implementation.
    """

    operators, data, _, _ = _family_problem(leadfields, observations, group_size)
    effective = freesigm_effective_weights(frequency_weights, len(operators))
    scale = np.sqrt(effective)
    stacked_operator = np.vstack(
        [factor * operator for factor, operator in zip(scale, operators)]
    )
    stacked_data = np.vstack(
        [factor * target for factor, target in zip(scale, data)]
    )
    return (
        np.asarray(stacked_operator, dtype=np.complex128),
        np.asarray(stacked_data, dtype=np.complex128),
        effective,
    )


def _hermitian(value: ArrayLike) -> ComplexArray:
    matrix = np.asarray(value, dtype=np.complex128)
    return np.asarray(0.5 * (matrix + matrix.conj().T), dtype=np.complex128)


def _project_psd(value: ArrayLike, *, rank_cap: int | None = None) -> tuple[ComplexArray, int]:
    matrix = _hermitian(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance matrices must be square")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order].real, 0.0)
    eigenvectors = eigenvectors[:, order]
    if rank_cap is not None:
        retained = int(rank_cap)
        if retained < 1:
            raise ValueError("rank_cap must be positive")
        eigenvalues[retained:] = 0.0
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(eigenvalues[0]) if eigenvalues.size else 1.0
    ) * max(1, matrix.shape[0])
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    projected = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.conj().T
    return _hermitian(projected), rank


def group_covariance_from_residual_samples(
    residual_samples: ArrayLike,
    *,
    group_size: int = 3,
    shrinkage: float = 0.0,
    rank_cap: int | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Estimate a mean residual and source-dependent sensor covariances.

    Parameters
    ----------
    residual_samples
        Array shaped ``(library_sample, frequency, sensor, source_column)`` or
        ``(library_sample, sensor, source_column)``.  Only training-library
        samples belong in this function.
    shrinkage
        Convex shrinkage toward the isotropic sensor covariance for each
        source group.  It is fixed globally, never selected on head truth.
    """

    residual = np.asarray(residual_samples, dtype=np.complex128)
    if residual.ndim == 3:
        residual = residual[:, None, :, :]
    if residual.ndim != 4 or residual.shape[0] < 2:
        raise ValueError(
            "residual_samples must have at least two samples and shape (N,F,C,P)"
        )
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_samples must be finite")
    group_size = int(group_size)
    samples, frequencies, sensors, coefficients = residual.shape
    if group_size < 1 or coefficients % group_size:
        raise ValueError("residual source columns must be divisible by group_size")
    shrinkage = float(shrinkage)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    groups = coefficients // group_size
    mean = np.mean(residual, axis=0)
    centered = residual - mean[None, ...]
    covariance = np.empty((frequencies, groups, sensors, sensors), dtype=np.complex128)
    for frequency in range(frequencies):
        for group in range(groups):
            accumulated = np.zeros((sensors, sensors), dtype=np.complex128)
            block = centered[
                :, frequency, :, group * group_size : (group + 1) * group_size
            ]
            for sample in range(samples):
                # Sum over orientation columns: Delta G_g Delta G_g^H.
                accumulated += block[sample] @ block[sample].conj().T
            empirical = accumulated / (samples - 1)
            isotropic = float(np.trace(empirical).real / sensors) * np.eye(sensors)
            shrunk = (1.0 - shrinkage) * empirical + shrinkage * isotropic
            covariance[frequency, group], _ = _project_psd(shrunk, rank_cap=rank_cap)
    return _readonly_complex(mean), _readonly_complex(covariance)


def group_covariance_from_low_rank_residual_basis(
    residual_basis: ArrayLike,
    residual_singular_values: ArrayLike,
    *,
    n_sensors: int,
    n_groups: int,
    group_size: int = 3,
    library_sample_count: int,
    rank_cap: int | None = None,
) -> ComplexArray:
    """Convert a library SVD of vectorized operators to group covariances."""

    basis = np.asarray(residual_basis, dtype=np.complex128)
    singular = np.asarray(residual_singular_values, dtype=np.float64)
    if basis.ndim == 2:
        basis = basis[None, ...]
    if singular.ndim == 1:
        singular = singular[None, ...]
    frequencies, available_rank, vector_size = basis.shape
    expected = int(n_sensors) * int(n_groups) * int(group_size)
    if vector_size != expected or singular.shape != (frequencies, available_rank):
        raise ValueError("residual basis/singular values do not match operator dimensions")
    if int(library_sample_count) < 2:
        raise ValueError("library_sample_count must be at least two")
    retained = available_rank if rank_cap is None else min(available_rank, int(rank_cap))
    if retained < 1:
        raise ValueError("rank_cap must retain at least one residual mode")
    covariance = np.zeros(
        (frequencies, int(n_groups), int(n_sensors), int(n_sensors)),
        dtype=np.complex128,
    )
    eigenvalues = singular[:, :retained] ** 2 / (int(library_sample_count) - 1)
    for frequency in range(frequencies):
        modes = basis[frequency, :retained].reshape(
            retained, int(n_sensors), int(n_groups), int(group_size)
        )
        for group in range(int(n_groups)):
            for mode, eigenvalue in zip(modes[:, :, group, :], eigenvalues[frequency]):
                covariance[frequency, group] += float(eigenvalue) * mode @ mode.conj().T
            covariance[frequency, group], _ = _project_psd(covariance[frequency, group])
    return _readonly_complex(covariance)


def _coerce_group_sensor_covariance(
    value: ArrayLike,
    *,
    frequencies: int,
    groups: int,
    sensors: int,
    group_size: int,
    rank_cap: int | None,
) -> tuple[ComplexArray, NDArray[np.int64]]:
    covariance = np.asarray(value, dtype=np.complex128)
    if covariance.ndim == 3 and frequencies == 1:
        covariance = covariance[None, ...]
    direct_shape = (frequencies, groups, sensors, sensors)
    expanded = sensors * group_size
    library_shape = (frequencies, groups, expanded, expanded)
    if covariance.shape == direct_shape:
        group_sensor = np.array(covariance, copy=True)
    elif covariance.shape == library_shape:
        # Partial trace over the source orientation.  The BEM library stores
        # vectorized (sensor, orientation) group residual covariances.
        group_sensor = np.zeros(direct_shape, dtype=np.complex128)
        for frequency in range(frequencies):
            for group in range(groups):
                tensor = covariance[frequency, group].reshape(
                    sensors, group_size, sensors, group_size
                )
                for orientation in range(group_size):
                    group_sensor[frequency, group] += tensor[
                        :, orientation, :, orientation
                    ]
    else:
        raise ValueError(
            "source_dependent_covariance must have shape (F,G,C,C) or "
            "(F,G,C*O,C*O)"
        )
    ranks = np.empty((frequencies, groups), dtype=np.int64)
    for frequency in range(frequencies):
        for group in range(groups):
            group_sensor[frequency, group], ranks[frequency, group] = _project_psd(
                group_sensor[frequency, group], rank_cap=rank_cap
            )
    return _readonly_complex(group_sensor), ranks


@dataclass(frozen=True, slots=True)
class ULUncertaintyModel:
    """Library-only mean and unit-group uncertainty sufficient statistics."""

    residual_mean: ComplexArray
    group_sensor_covariance: ComplexArray
    rank_per_frequency_group: NDArray[np.int64]
    metadata: Mapping[str, object]


def build_ul_uncertainty_model(
    nominal_leadfields: Iterable[ArrayLike] | ArrayLike,
    *,
    group_size: int = 3,
    residual_mean: ArrayLike | None = None,
    source_dependent_covariance: ArrayLike | None = None,
    residual_basis: ArrayLike | None = None,
    residual_singular_values: ArrayLike | None = None,
    library_sample_count: int | None = None,
    rank_cap: int | None = None,
) -> ULUncertaintyModel:
    """Build the UL sufficient statistics without accepting EEG or truth."""

    operators = _operator_family(nominal_leadfields)
    frequencies = len(operators)
    sensors, coefficients = operators[0].shape
    if any(operator.shape != (sensors, coefficients) for operator in operators):
        raise ValueError("nominal leadfields must share dimensions")
    if coefficients % int(group_size):
        raise ValueError("source columns must be divisible by group_size")
    groups = coefficients // int(group_size)
    if residual_mean is None:
        mean = np.zeros((frequencies, sensors, coefficients), dtype=np.complex128)
    else:
        raw_mean = np.asarray(residual_mean, dtype=np.complex128)
        if raw_mean.ndim == 4:
            raw_mean = raw_mean.reshape(raw_mean.shape[0], raw_mean.shape[1], -1)
        if raw_mean.ndim == 2 and frequencies == 1:
            raw_mean = raw_mean[None, ...]
        if raw_mean.shape != (frequencies, sensors, coefficients):
            raise ValueError("residual_mean does not align with nominal leadfields")
        if not np.all(np.isfinite(raw_mean)):
            raise ValueError("residual_mean must be finite")
        mean = np.array(raw_mean, copy=True)

    covariance_source = "source_dependent_covariance"
    if source_dependent_covariance is None:
        if residual_basis is None or residual_singular_values is None or library_sample_count is None:
            raise ValueError(
                "provide source_dependent_covariance or a residual SVD and sample count"
            )
        covariance_source = "low_rank_residual_basis"
        covariance_value = group_covariance_from_low_rank_residual_basis(
            residual_basis,
            residual_singular_values,
            n_sensors=sensors,
            n_groups=groups,
            group_size=int(group_size),
            library_sample_count=int(library_sample_count),
            rank_cap=rank_cap,
        )
        # Already rank capped and in direct group-sensor form.
        covariance, ranks = _coerce_group_sensor_covariance(
            covariance_value,
            frequencies=frequencies,
            groups=groups,
            sensors=sensors,
            group_size=int(group_size),
            rank_cap=None,
        )
    else:
        covariance, ranks = _coerce_group_sensor_covariance(
            source_dependent_covariance,
            frequencies=frequencies,
            groups=groups,
            sensors=sensors,
            group_size=int(group_size),
            rank_cap=rank_cap,
        )
    return ULUncertaintyModel(
        residual_mean=_readonly_complex(mean),
        group_sensor_covariance=covariance,
        rank_per_frequency_group=np.asarray(ranks, dtype=np.int64),
        metadata=MappingProxyType(
            {
                "covariance_source": covariance_source,
                "rank_cap": None if rank_cap is None else int(rank_cap),
                "truth_or_eeg_inputs_accepted": False,
                "proper_complex": True,
            }
        ),
    )


def uncertainty_model_from_library_statistics(
    nominal_leadfields: Iterable[ArrayLike] | ArrayLike,
    statistics: object,
    *,
    group_size: int = 3,
    rank_cap: int | None = None,
) -> ULUncertaintyModel:
    """Duck-typed adapter for ``BEMLibraryStatistics`` without a dependency."""

    covariance = getattr(statistics, "source_dependent_residual_covariance")
    mean = getattr(statistics, "residual_mean")
    return build_ul_uncertainty_model(
        nominal_leadfields,
        group_size=group_size,
        residual_mean=mean,
        source_dependent_covariance=covariance,
        rank_cap=rank_cap,
    )


def uncertainty_sensor_covariance(
    group_sensor_covariance: ArrayLike,
    gamma: ArrayLike,
) -> ComplexArray:
    """Return ``Q(gamma)`` and preserve PSD/source dependence by construction."""

    covariance = np.asarray(group_sensor_covariance, dtype=np.complex128)
    gamma_array = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if covariance.ndim != 3 or covariance.shape[0] != gamma_array.size:
        raise ValueError("group covariances and gamma must align")
    if np.any(gamma_array < 0.0) or not np.all(np.isfinite(gamma_array)):
        raise ValueError("gamma must be finite and nonnegative")
    q = np.einsum("g,gij->ij", gamma_array, covariance, optimize=True)
    projected, _ = _project_psd(q)
    return projected


def _expand_gamma(gamma: ArrayLike, coefficients: int, group_size: int) -> FloatArray:
    group_gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if group_gamma.shape != (coefficients // group_size,):
        raise ValueError("gamma must have one value per source group")
    if np.any(group_gamma <= 0.0) or not np.all(np.isfinite(group_gamma)):
        raise ValueError("gamma must be finite and strictly positive")
    return np.repeat(group_gamma, group_size)


def ul_sensor_covariance(
    mean_leadfield: ArrayLike,
    gamma: ArrayLike,
    noise_covariance: ArrayLike | float,
    group_sensor_covariance: ArrayLike,
    *,
    eta: float,
    group_size: int = 3,
) -> tuple[ComplexArray, ComplexArray]:
    """Construct ``Sigma_y`` and its source-dependent ``Q(gamma)`` term."""

    operator = np.asarray(mean_leadfield, dtype=np.complex128)
    if operator.ndim != 2:
        raise ValueError("mean_leadfield must be two-dimensional")
    gamma_diagonal = _expand_gamma(gamma, operator.shape[1], int(group_size))
    if np.isscalar(noise_covariance):
        noise_value = float(noise_covariance)
        if not np.isfinite(noise_value) or noise_value <= 0.0:
            raise ValueError("noise variance must be finite and positive")
        noise = noise_value * np.eye(operator.shape[0], dtype=np.complex128)
    else:
        noise = np.asarray(noise_covariance, dtype=np.complex128)
        if noise.shape != (operator.shape[0], operator.shape[0]):
            raise ValueError("noise covariance has the wrong shape")
        noise, _ = _project_psd(noise)
    eta_value = float(eta)
    if not np.isfinite(eta_value) or eta_value < 0.0:
        raise ValueError("eta must be finite and nonnegative")
    q = uncertainty_sensor_covariance(group_sensor_covariance, gamma)
    sensor = operator * gamma_diagonal[None, :] @ operator.conj().T
    return _hermitian(noise + sensor + eta_value * q), q


def _cho_factor(value: ArrayLike) -> tuple[ComplexArray, bool]:
    matrix = _hermitian(value)
    scale = max(float(np.trace(matrix).real / max(1, matrix.shape[0])), 1.0)
    jitter = np.finfo(np.float64).eps * scale
    for _ in range(9):
        try:
            return linalg.cho_factor(
                matrix + jitter * np.eye(matrix.shape[0]),
                lower=True,
                check_finite=False,
            )
        except linalg.LinAlgError:
            jitter *= 10.0
    raise linalg.LinAlgError("sensor covariance is not positive definite")


def uncertain_leadfield_posterior(
    mean_leadfield: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    noise_covariance: ArrayLike | float,
    group_sensor_covariance: ArrayLike,
    *,
    eta: float,
    group_size: int = 3,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return moment-matched posterior mean/covariance, ``Sigma_y``, and Q.

    The source-to-sensor cross covariance is the nominal/mean
    ``Gamma Gbar^H``.  The posterior covariance is therefore consistent with
    the same joint Gaussian moment model used by the evidence objective.
    """

    operator = np.asarray(mean_leadfield, dtype=np.complex128)
    target = _as_snapshots(observations)
    if operator.ndim != 2 or operator.shape[0] != target.shape[0]:
        raise ValueError("mean leadfield and observations do not align")
    gamma_diagonal = _expand_gamma(gamma, operator.shape[1], int(group_size))
    sensor, q = ul_sensor_covariance(
        operator,
        gamma,
        noise_covariance,
        group_sensor_covariance,
        eta=eta,
        group_size=group_size,
    )
    factor = _cho_factor(sensor)
    solved_y = linalg.cho_solve(factor, target, check_finite=False)
    solved_operator_gamma = linalg.cho_solve(
        factor, operator * gamma_diagonal[None, :], check_finite=False
    )
    cross = gamma_diagonal[:, None] * operator.conj().T
    mean = cross @ solved_y
    covariance = np.diag(gamma_diagonal) - cross @ solved_operator_gamma
    return (
        np.asarray(mean, dtype=np.complex128),
        _hermitian(covariance),
        sensor,
        q,
    )


def _complex_nll(target: ComplexArray, sensor_covariance: ComplexArray) -> float:
    factor = _cho_factor(sensor_covariance)
    diagonal = np.maximum(
        np.real(np.diag(factor[0])), np.finfo(np.float64).tiny
    )
    logdet = 2.0 * float(np.log(diagonal).sum())
    solved = linalg.cho_solve(factor, target, check_finite=False)
    quadratic = float(np.real(np.vdot(target, solved)))
    return target.shape[1] * logdet + quadratic


def ul_type2_objective(
    mean_leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    gamma: ArrayLike,
    noise_variances: ArrayLike | float,
    group_sensor_covariance: ArrayLike,
    *,
    eta: float,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
) -> float:
    """Weighted proper-complex Type-II negative log evidence."""

    operators, data, _, groups = _family_problem(
        mean_leadfields, observations, group_size
    )
    covariance = np.asarray(group_sensor_covariance, dtype=np.complex128)
    if covariance.shape != (
        len(operators),
        groups,
        operators[0].shape[0],
        operators[0].shape[0],
    ):
        raise ValueError("group_sensor_covariance does not align with the family")
    if np.isscalar(noise_variances):
        noise = np.full(len(operators), float(noise_variances), dtype=np.float64)
    else:
        noise = np.asarray(noise_variances, dtype=np.float64).reshape(-1)
    if noise.shape != (len(operators),) or np.any(noise <= 0.0):
        raise ValueError("noise_variances must be positive and frequency aligned")
    effective = freesigm_effective_weights(frequency_weights, len(operators))
    objective = 0.0
    for frequency, (operator, target) in enumerate(zip(operators, data)):
        sensor, _ = ul_sensor_covariance(
            operator,
            gamma,
            noise[frequency],
            covariance[frequency],
            eta=eta,
            group_size=group_size,
        )
        objective += float(effective[frequency]) * _complex_nll(target, sensor)
    return float(objective)


def comparable_heldout_ul_objective(
    result: "ULChampagneResult",
    leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    uncertainty_model: ULUncertaintyModel,
    *,
    group_size: int = 3,
    normalize_per_sensor_snapshot: bool = True,
) -> float:
    """Score every eta arm with one held-out proper-complex objective.

    The literal ``eta=0`` path is intentionally implemented by a vertically
    stacked Champagne call, whereas positive-eta fits use one covariance per
    frequency.  Their returned training-objective histories are consequently
    not comparable.  This adapter maps the stacked noise variance back to the
    unscaled frequency blocks and evaluates :func:`ul_type2_objective` for all
    eta values on the same held-out observations.

    No truth, support, localization score, or evaluation outcome is accepted.
    """

    operators, data, validated_group_size, groups = _family_problem(
        leadfields, observations, group_size
    )
    if validated_group_size != int(group_size):
        raise AssertionError("group-size validation changed")
    frequencies = len(operators)
    sensors = operators[0].shape[0]
    snapshots = data[0].shape[1]
    expected_mean = (frequencies, sensors, operators[0].shape[1])
    expected_covariance = (frequencies, groups, sensors, sensors)
    if uncertainty_model.residual_mean.shape != expected_mean:
        raise ValueError("uncertainty residual mean does not align with held-out data")
    if uncertainty_model.group_sensor_covariance.shape != expected_covariance:
        raise ValueError("uncertainty covariance does not align with held-out data")
    weights = safe_frequency_weights(result.frequency_weights, frequencies)
    effective = freesigm_effective_weights(weights, frequencies)
    raw_noise = np.asarray(result.noise, dtype=np.float64).reshape(-1)
    if raw_noise.size == 1:
        # The eta-zero stack scales each frequency block by sqrt(F*w_f).  A
        # scalar stacked noise therefore corresponds to noise/(F*w_f) after
        # mapping the block back to its unscaled physical coordinates.
        noise = np.full(frequencies, float(raw_noise[0])) / effective
    elif raw_noise.size == frequencies:
        noise = raw_noise
    else:
        raise ValueError("UL result noise is neither scalar nor frequency aligned")
    if np.any(noise <= 0.0) or not np.all(np.isfinite(noise)):
        raise ValueError("UL result contains invalid noise variances")
    mean_operators = [
        np.asarray(
            operator + uncertainty_model.residual_mean[frequency],
            dtype=np.complex128,
        )
        for frequency, operator in enumerate(operators)
    ]
    value = ul_type2_objective(
        mean_operators,
        data,
        result.source_score,
        noise,
        uncertainty_model.group_sensor_covariance,
        eta=result.eta,
        frequency_weights=weights,
        group_size=group_size,
    )
    if not normalize_per_sensor_snapshot:
        return float(value)
    denominator = frequencies * sensors * snapshots
    return float(value / max(1, denominator))


@dataclass(frozen=True, slots=True)
class FixedStateFrequencyFusion:
    """Descriptive frequency fusion with every fitted state held fixed."""

    per_frequency_group_power: FloatArray
    fused_group_power: FloatArray
    frequency_weights: FloatArray
    gamma: FloatArray
    noise_variances: FloatArray
    eta: float
    metadata: Mapping[str, object]


def fixed_state_frequency_posterior_fusion(
    leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    *,
    uncertainty_model: ULUncertaintyModel,
    gamma: ArrayLike,
    noise_variances: ArrayLike | float,
    eta: float,
    frequency_weights: ArrayLike,
    group_size: int = 3,
) -> FixedStateFrequencyFusion:
    """Fuse per-frequency posterior power without refitting downstream state.

    This is the algorithmic definition used by the preregistered
    controlled-direct FreeSIGM ablation.  ``gamma``, noise, eta, the operator
    family, and the uncertainty model are supplied from the already-completed
    full fit.  Only the final convex frequency fusion changes.  The output is
    descriptive and must not be used for candidate or hyperparameter choice.
    """

    operators, data, _, groups = _family_problem(leadfields, observations, group_size)
    frequencies = len(operators)
    sensors = operators[0].shape[0]
    coefficients = operators[0].shape[1]
    expected_mean = (frequencies, sensors, coefficients)
    expected_covariance = (frequencies, groups, sensors, sensors)
    if uncertainty_model.residual_mean.shape != expected_mean:
        raise ValueError("uncertainty residual mean does not align")
    if uncertainty_model.group_sensor_covariance.shape != expected_covariance:
        raise ValueError("uncertainty covariance does not align")
    weights = safe_frequency_weights(frequency_weights, frequencies)
    group_gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if group_gamma.shape != (groups,) or np.any(group_gamma <= 0.0):
        raise ValueError("gamma must be strictly positive and group aligned")
    raw_noise = np.asarray(noise_variances, dtype=np.float64).reshape(-1)
    if raw_noise.size == 1:
        noise = np.full(frequencies, float(raw_noise[0]))
    elif raw_noise.size == frequencies:
        noise = raw_noise
    else:
        raise ValueError("noise variances must be scalar or frequency aligned")
    if np.any(noise <= 0.0) or not np.all(np.isfinite(noise)):
        raise ValueError("noise variances must be finite and positive")

    power = np.empty((frequencies, groups), dtype=np.float64)
    for frequency, (operator, target) in enumerate(zip(operators, data)):
        mean_operator = np.asarray(
            operator + uncertainty_model.residual_mean[frequency],
            dtype=np.complex128,
        )
        mean, covariance, _, _ = uncertain_leadfield_posterior(
            mean_operator,
            target,
            group_gamma,
            noise[frequency],
            uncertainty_model.group_sensor_covariance[frequency],
            eta=float(eta),
            group_size=group_size,
        )
        mean_coefficient_power = np.sum(np.abs(mean) ** 2, axis=1).real
        covariance_coefficient_power = (
            target.shape[1] * np.maximum(np.real(np.diag(covariance)), 0.0)
        )
        power[frequency] = (
            mean_coefficient_power + covariance_coefficient_power
        ).reshape(groups, int(group_size)).sum(axis=1)
    fused = np.einsum("f,fg->g", weights, power, optimize=True)
    fused = np.maximum(np.asarray(fused, dtype=np.float64), 0.0)
    total = float(fused.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("fixed-state fusion produced no finite source mass")
    fused /= total
    return FixedStateFrequencyFusion(
        per_frequency_group_power=_readonly_float(power),
        fused_group_power=_readonly_float(fused),
        frequency_weights=_readonly_float(weights),
        gamma=_readonly_float(group_gamma),
        noise_variances=_readonly_float(noise),
        eta=float(eta),
        metadata=MappingProxyType(
            {
                "role": "controlled_direct_descriptive_secondary",
                "downstream_state_refit": False,
                "truth_or_emd_used": False,
                "fusion_definition": "weighted per-frequency posterior expected group power",
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class ULChampagneResult:
    method: str
    source_estimate: ComplexArray
    source_covariance: ComplexArray
    frequency_source_estimate: ComplexArray
    frequency_source_covariance: ComplexArray
    source_score: FloatArray
    noise: float | FloatArray
    objective: tuple[float, ...]
    converged: bool
    iterations: int
    eta: float
    frequency_weights: FloatArray
    effective_frequency_weights: FloatArray
    runtime_seconds: float
    metadata: Mapping[str, object]

    @property
    def mean(self) -> ComplexArray:
        return self.source_estimate

    @property
    def covariance(self) -> ComplexArray:
        return self.source_covariance

    @property
    def gamma(self) -> FloatArray:
        return self.source_score


def _validate_protocol_eta(value: float) -> float:
    eta = float(value)
    if not np.isfinite(eta) or not any(np.isclose(eta, item, atol=0.0, rtol=0.0) for item in UL_ETA_GRID):
        raise ValueError(f"eta must be one of the frozen grid values {UL_ETA_GRID}")
    return eta


def uncertain_leadfield_champagne(
    leadfields: Iterable[ArrayLike] | ArrayLike,
    observations: Iterable[ArrayLike] | ArrayLike,
    *,
    uncertainty_model: ULUncertaintyModel | None = None,
    residual_mean: ArrayLike | None = None,
    source_dependent_covariance: ArrayLike | None = None,
    residual_basis: ArrayLike | None = None,
    residual_singular_values: ArrayLike | None = None,
    library_sample_count: int | None = None,
    rank_cap: int | None = None,
    frequency_weights: ArrayLike | None = None,
    eta: float = 0.0,
    group_size: int = 3,
    max_iter: int = 80,
    tolerance: float = 1.0e-6,
    prune_threshold: float = 0.0,
    gamma0: ArrayLike | None = None,
    noise0: float | None = None,
    min_variance: float = 1.0e-12,
) -> ULChampagneResult:
    """Fit shared-gamma uncertain-leadfield Champagne.

    ``eta=0`` is a literal call to the historical Champagne solver on the
    deployed nominal leadfields after the common ``sqrt(F*w_safe)`` stack.
    Neither the residual mean nor covariance may alter that branch, which
    makes the fallback and forward ablation exact even when a column-matched
    uncertainty context is supplied.

    For ``eta>0``, per-frequency moment-matched posteriors share one group
    gamma.  EM-style second-moment proposals are accepted only if the exact
    weighted Type-II objective decreases; otherwise log-domain backtracking is
    used and a failed search stops at the last accepted state.
    """

    start = perf_counter()
    eta_value = _validate_protocol_eta(eta)
    operators, data, group_size, groups = _family_problem(
        leadfields, observations, group_size
    )
    weights = safe_frequency_weights(frequency_weights, len(operators))
    effective = freesigm_effective_weights(weights, len(operators))

    if uncertainty_model is not None and any(
        value is not None
        for value in (
            residual_mean,
            source_dependent_covariance,
            residual_basis,
            residual_singular_values,
        )
    ):
        raise ValueError("pass either uncertainty_model or raw uncertainty statistics")
    if uncertainty_model is None:
        if eta_value == 0.0 and source_dependent_covariance is None and residual_basis is None:
            # Validate a caller-supplied mean for contract consistency, but do
            # not deploy it on the exact eta-zero nominal path below.
            mean = (
                np.zeros((len(operators), operators[0].shape[0], operators[0].shape[1]), dtype=np.complex128)
                if residual_mean is None
                else np.asarray(residual_mean, dtype=np.complex128)
            )
            if mean.ndim == 4:
                mean = mean.reshape(mean.shape[0], mean.shape[1], -1)
            if mean.ndim == 2 and len(operators) == 1:
                mean = mean[None, ...]
            if mean.shape != (len(operators), operators[0].shape[0], operators[0].shape[1]):
                raise ValueError("residual_mean does not align with leadfields")
            covariance = np.zeros(
                (len(operators), groups, operators[0].shape[0], operators[0].shape[0]),
                dtype=np.complex128,
            )
            uncertainty_model = ULUncertaintyModel(
                residual_mean=_readonly_complex(mean),
                group_sensor_covariance=_readonly_complex(covariance),
                rank_per_frequency_group=np.zeros((len(operators), groups), dtype=np.int64),
                metadata=MappingProxyType({"covariance_source": "eta_zero_not_required"}),
            )
        else:
            uncertainty_model = build_ul_uncertainty_model(
                operators,
                group_size=group_size,
                residual_mean=residual_mean,
                source_dependent_covariance=source_dependent_covariance,
                residual_basis=residual_basis,
                residual_singular_values=residual_singular_values,
                library_sample_count=library_sample_count,
                rank_cap=rank_cap,
            )
    expected_mean_shape = (
        len(operators),
        operators[0].shape[0],
        operators[0].shape[1],
    )
    expected_covariance_shape = (
        len(operators),
        groups,
        operators[0].shape[0],
        operators[0].shape[0],
    )
    if uncertainty_model.residual_mean.shape != expected_mean_shape:
        raise ValueError("uncertainty model residual mean does not align")
    if uncertainty_model.group_sensor_covariance.shape != expected_covariance_shape:
        raise ValueError("uncertainty model group covariance does not align")
    if eta_value == 0.0:
        mean_operators = [
            np.asarray(operator, dtype=np.complex128) for operator in operators
        ]
    else:
        mean_operators = [
            np.asarray(
                operator + uncertainty_model.residual_mean[frequency],
                dtype=np.complex128,
            )
            for frequency, operator in enumerate(operators)
        ]

    stacked_operator, stacked_data, _ = stack_freesigm_family(
        mean_operators,
        data,
        frequency_weights=weights,
        group_size=group_size,
    )
    baseline = baselines.champagne_sbl(
        stacked_operator,
        stacked_data,
        group_size=group_size,
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        prune_threshold=float(prune_threshold),
        noise0=noise0,
    )
    baseline_gamma = np.asarray(baseline.source_score, dtype=np.float64)
    baseline_noise = float(baseline.metadata["noise"])
    baseline_mean = np.asarray(baseline.source_estimate, dtype=np.complex128)
    baseline_covariance = complex_gaussian_posterior(
        stacked_operator,
        stacked_data,
        baseline_gamma,
        baseline_noise,
        group_size=group_size,
    )[1]

    if eta_value == 0.0:
        repeated_mean = np.repeat(baseline_mean[None, ...], len(operators), axis=0)
        repeated_covariance = np.repeat(
            np.asarray(baseline_covariance)[None, ...], len(operators), axis=0
        )
        return ULChampagneResult(
            method="V3-UL-Champagne-FreeSIGM",
            source_estimate=_readonly_complex(baseline_mean),
            source_covariance=_readonly_complex(baseline_covariance),
            frequency_source_estimate=_readonly_complex(repeated_mean),
            frequency_source_covariance=_readonly_complex(repeated_covariance),
            source_score=_readonly_float(baseline_gamma),
            noise=baseline_noise,
            objective=tuple(float(value) for value in baseline.metadata["objective"]),
            converged=bool(baseline.metadata["converged"]),
            iterations=int(baseline.metadata["iterations"]),
            eta=0.0,
            frequency_weights=_readonly_float(weights),
            effective_frequency_weights=_readonly_float(effective),
            runtime_seconds=perf_counter() - start,
            metadata=MappingProxyType(
                {
                    "solver": "inverse.baselines.champagne_sbl",
                    "eta_zero_exact_baseline_call": True,
                    "eta_zero_exact_nominal_operator": True,
                    "eta_zero_residual_mean_correction_enabled": False,
                    "freesigm_stack_scale": "sqrt(F*w_safe)",
                    "truth_used_for_fit_or_selection": False,
                    "fallback_safe_freesigm_preserved": True,
                    "uncertainty_rank": uncertainty_model.rank_per_frequency_group.tolist(),
                }
            ),
        )

    min_variance = float(min_variance)
    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be finite and positive")
    if gamma0 is None:
        gamma = np.maximum(baseline_gamma, min_variance)
    else:
        gamma = np.asarray(gamma0, dtype=np.float64).reshape(-1)
        if gamma.shape != (groups,) or np.any(gamma <= 0.0) or not np.all(np.isfinite(gamma)):
            raise ValueError("gamma0 must be finite, positive, and group aligned")
    # The eta-zero stack scales frequency f by sqrt(F*w_f), so its scalar
    # noise corresponds to lambda_f = lambda_stack / (F*w_f).
    noise = np.maximum(baseline_noise / effective, min_variance)
    objective = [
        ul_type2_objective(
            mean_operators,
            data,
            gamma,
            noise,
            uncertainty_model.group_sensor_covariance,
            eta=eta_value,
            frequency_weights=weights,
            group_size=group_size,
        )
    ]
    means: list[ComplexArray] = []
    covariances: list[ComplexArray] = []
    line_search_fractions: list[float] = []
    converged = False
    convergence_reason = "max_iter"

    for _ in range(int(max_iter)):
        means = []
        covariances = []
        for frequency, (operator, target) in enumerate(zip(mean_operators, data)):
            mean, covariance, _, _ = uncertain_leadfield_posterior(
                operator,
                target,
                gamma,
                noise[frequency],
                uncertainty_model.group_sensor_covariance[frequency],
                eta=eta_value,
                group_size=group_size,
            )
            means.append(mean)
            covariances.append(covariance)

        proposed_gamma = np.zeros(groups, dtype=np.float64)
        gamma_denominator = 0.0
        proposed_noise = np.empty_like(noise)
        for frequency, (operator, target, mean, covariance) in enumerate(
            zip(mean_operators, data, means, covariances)
        ):
            multiplier = float(effective[frequency] * target.shape[1])
            gamma_denominator += multiplier
            for group in range(groups):
                sl = slice(group * group_size, (group + 1) * group_size)
                second = covariance[sl, sl] + mean[sl] @ mean[sl].conj().T / target.shape[1]
                proposed_gamma[group] += multiplier * float(
                    np.trace(second).real / group_size
                )
            residual = target - operator @ mean
            posterior_trace = target.shape[1] * float(
                np.trace(operator @ covariance @ operator.conj().T).real
            )
            proposed_noise[frequency] = max(
                (float(np.linalg.norm(residual, "fro") ** 2) + posterior_trace)
                / (operator.shape[0] * target.shape[1]),
                min_variance,
            )
        proposed_gamma = np.maximum(
            proposed_gamma / max(gamma_denominator, min_variance), min_variance
        )

        accepted = False
        fraction = 1.0
        accepted_gamma = gamma
        accepted_noise = noise
        accepted_objective = objective[-1]
        objective_tolerance = 1.0e-9 * max(1.0, abs(objective[-1]))
        while fraction >= 2.0**-16:
            candidate_gamma = np.exp(
                (1.0 - fraction) * np.log(gamma)
                + fraction * np.log(proposed_gamma)
            )
            candidate_noise = np.exp(
                (1.0 - fraction) * np.log(noise)
                + fraction * np.log(proposed_noise)
            )
            candidate_objective = ul_type2_objective(
                mean_operators,
                data,
                candidate_gamma,
                candidate_noise,
                uncertainty_model.group_sensor_covariance,
                eta=eta_value,
                frequency_weights=weights,
                group_size=group_size,
            )
            if candidate_objective <= objective[-1] + objective_tolerance:
                accepted = True
                accepted_gamma = candidate_gamma
                accepted_noise = candidate_noise
                accepted_objective = float(candidate_objective)
                break
            fraction *= 0.5
        if not accepted:
            convergence_reason = "line_search_stalled"
            break
        relative = max(
            float(np.max(np.abs(accepted_gamma - gamma) / np.maximum(gamma, min_variance))),
            float(np.max(np.abs(accepted_noise - noise) / np.maximum(noise, min_variance))),
        )
        gamma = np.asarray(accepted_gamma, dtype=np.float64)
        noise = np.asarray(accepted_noise, dtype=np.float64)
        objective.append(accepted_objective)
        line_search_fractions.append(float(fraction))
        if relative < float(tolerance):
            converged = True
            convergence_reason = "tolerance"
            break

    final_means: list[ComplexArray] = []
    final_covariances: list[ComplexArray] = []
    for frequency, (operator, target) in enumerate(zip(mean_operators, data)):
        mean, covariance, _, _ = uncertain_leadfield_posterior(
            operator,
            target,
            gamma,
            noise[frequency],
            uncertainty_model.group_sensor_covariance[frequency],
            eta=eta_value,
            group_size=group_size,
        )
        final_means.append(mean)
        final_covariances.append(covariance)
    frequency_mean = np.stack(final_means, axis=0)
    frequency_covariance = np.stack(final_covariances, axis=0)
    aggregate_mean = np.einsum("f,fpt->pt", weights, frequency_mean)
    aggregate_covariance = np.einsum("f,fpq->pq", weights, frequency_covariance)

    return ULChampagneResult(
        method="V3-UL-Champagne-FreeSIGM",
        source_estimate=_readonly_complex(aggregate_mean),
        source_covariance=_readonly_complex(aggregate_covariance),
        frequency_source_estimate=_readonly_complex(frequency_mean),
        frequency_source_covariance=_readonly_complex(frequency_covariance),
        source_score=_readonly_float(gamma),
        noise=_readonly_float(noise),
        objective=tuple(float(value) for value in objective),
        converged=converged,
        iterations=len(objective) - 1,
        eta=eta_value,
        frequency_weights=_readonly_float(weights),
        effective_frequency_weights=_readonly_float(effective),
        runtime_seconds=perf_counter() - start,
        metadata=MappingProxyType(
            {
                "solver": "shared-gamma proper-complex Type-II with log-domain line search",
                "eta_zero_exact_baseline_call": False,
                "line_search_guard": True,
                "line_search_fractions": tuple(line_search_fractions),
                "convergence_reason": convergence_reason,
                "truth_used_for_fit_or_selection": False,
                "freesigm_stack_scale": "F*w_safe evidence multipliers",
                "uncertainty_rank": uncertainty_model.rank_per_frequency_group.tolist(),
                "active_groups": np.flatnonzero(
                    gamma >= float(prune_threshold) * max(float(gamma.max()), min_variance)
                    if float(prune_threshold) > 0.0
                    else np.ones_like(gamma, dtype=bool)
                ).tolist(),
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class ULCalibrationSelection:
    eta: float
    fallback: bool
    fallback_reason: str | None
    fold_count: int
    mean_objective_by_eta: Mapping[float, float]
    standard_error_by_eta: Mapping[float, float]
    eligible_eta: tuple[float, ...]
    metadata: Mapping[str, object]


def select_ul_eta_from_calibration(
    fold_objectives: Mapping[float, ArrayLike],
    *,
    ood_detected: bool = False,
    numerical_failures: Mapping[float, ArrayLike] | Sequence[float] | None = None,
    minimum_fold_improvement: float = 0.0,
    one_standard_error: bool = True,
) -> ULCalibrationSelection:
    """Select eta from outcome-free calibration objectives and fail closed.

    Every selected positive eta must improve on eta zero in every usable fold.
    The function accepts no source truth, EMD, or localization argument.  OOD,
    non-finite folds, failed positive candidates, or lack of consistent
    improvement return the exact eta-zero path with an auditable reason.
    """

    normalized: dict[float, FloatArray] = {}
    for raw_eta, values in fold_objectives.items():
        eta = _validate_protocol_eta(float(raw_eta))
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if eta in normalized:
            raise ValueError("duplicate eta after numeric normalization")
        normalized[eta] = array
    if 0.0 not in normalized or normalized[0.0].size < 1:
        raise ValueError("eta=0 calibration folds are required")
    fold_count = int(normalized[0.0].size)
    if any(values.size != fold_count for values in normalized.values()):
        raise ValueError("all eta candidates must use the same calibration folds")

    failed: set[float] = set()
    if numerical_failures is not None:
        if isinstance(numerical_failures, Mapping):
            for raw_eta, flags in numerical_failures.items():
                eta = _validate_protocol_eta(float(raw_eta))
                flag_array = np.asarray(flags, dtype=bool).reshape(-1)
                if flag_array.size not in (1, fold_count):
                    raise ValueError("numerical failure flags must be scalar or fold aligned")
                if np.any(flag_array):
                    failed.add(eta)
        else:
            failed.update(_validate_protocol_eta(float(value)) for value in numerical_failures)

    finite = {eta: bool(np.all(np.isfinite(values))) for eta, values in normalized.items()}
    means = {
        eta: float(np.mean(values)) if finite[eta] else float("inf")
        for eta, values in normalized.items()
    }
    standard_errors = {
        eta: (
            float(np.std(values, ddof=1) / np.sqrt(fold_count))
            if finite[eta] and fold_count > 1
            else 0.0
        )
        for eta, values in normalized.items()
    }

    def fallback(reason: str) -> ULCalibrationSelection:
        return ULCalibrationSelection(
            eta=0.0,
            fallback=True,
            fallback_reason=reason,
            fold_count=fold_count,
            mean_objective_by_eta=MappingProxyType(means),
            standard_error_by_eta=MappingProxyType(standard_errors),
            eligible_eta=(),
            metadata=MappingProxyType(
                {
                    "selection_data": "calibration_objective_only",
                    "truth_or_emd_used": False,
                    "safe_freesigm_preserved": True,
                    "exact_eta_zero_fallback": True,
                }
            ),
        )

    if ood_detected:
        return fallback("ood_detected")
    if not finite[0.0] or 0.0 in failed:
        return fallback("eta_zero_calibration_numerical_failure")
    baseline = normalized[0.0]
    eligible: list[float] = []
    for eta in UL_ETA_GRID[1:]:
        if eta not in normalized or eta in failed or not finite.get(eta, False):
            continue
        if np.all(normalized[eta] < baseline - float(minimum_fold_improvement)):
            eligible.append(eta)
    if not eligible:
        has_numerical_problem = any(
            eta in failed or (eta in normalized and not finite.get(eta, False))
            for eta in UL_ETA_GRID[1:]
        )
        return fallback(
            "positive_eta_numerical_failure"
            if has_numerical_problem
            else "no_consistent_fold_improvement"
        )

    best = min(eligible, key=lambda item: (means[item], item))
    if one_standard_error:
        threshold = means[best] + standard_errors[best]
        within = [eta for eta in eligible if means[eta] <= threshold]
        selected = min(within)
    else:
        selected = best
    return ULCalibrationSelection(
        eta=float(selected),
        fallback=False,
        fallback_reason=None,
        fold_count=fold_count,
        mean_objective_by_eta=MappingProxyType(means),
        standard_error_by_eta=MappingProxyType(standard_errors),
        eligible_eta=tuple(eligible),
        metadata=MappingProxyType(
            {
                "selection_data": "calibration_objective_only",
                "truth_or_emd_used": False,
                "safe_freesigm_preserved": True,
                "one_standard_error_rule": bool(one_standard_error),
            }
        ),
    )


# Short aliases make the integration point readable in head-level selectors.
ul_champagne = uncertain_leadfield_champagne
select_ul_eta = select_ul_eta_from_calibration


__all__ = [
    "ETA_GRID",
    "FixedStateFrequencyFusion",
    "UL_ETA_GRID",
    "ULCalibrationSelection",
    "ULChampagneResult",
    "ULUncertaintyModel",
    "build_ul_uncertainty_model",
    "comparable_heldout_ul_objective",
    "freesigm_effective_weights",
    "fixed_state_frequency_posterior_fusion",
    "group_covariance_from_low_rank_residual_basis",
    "group_covariance_from_residual_samples",
    "safe_frequency_weights",
    "select_ul_eta",
    "select_ul_eta_from_calibration",
    "stack_freesigm_family",
    "ul_champagne",
    "ul_sensor_covariance",
    "ul_type2_objective",
    "uncertain_leadfield_champagne",
    "uncertain_leadfield_posterior",
    "uncertainty_model_from_library_statistics",
    "uncertainty_sensor_covariance",
]
