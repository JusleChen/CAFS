"""Auditable V2 depth/orientation wrapper around the existing Champagne solver.

The module deliberately contains no model or hyper-parameter selection.  An
orientation mode and a depth exponent must be supplied from the predeclared
grid (or selected using calibration-side held-out predictive likelihood).  No
source truth is accepted by any public function.

For an orientation matrix ``O`` and diagonal depth matrix ``D``, the solver
coordinate ``z`` and physical free-orientation coefficient ``x`` obey

``x = O D z`` and ``G_tilde = G O D``.

Consequently posterior moments are returned to physical coordinates exactly as
``mu_x = O D mu_z`` and ``Sigma_x = O D Sigma_z D^H O^H``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from . import baselines
from .complex_sbl import complex_gaussian_posterior


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

DEPTH_EXPONENT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ORIENTATION_MODES = ("cortical_normal", "loose_0.2", "free_3d")


def _finite_complex(name: str, value: ArrayLike, ndim: int) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return array


def _family_problem(
    leadfields: ArrayLike, observations: ArrayLike
) -> tuple[ComplexArray, ComplexArray, bool]:
    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    single_frequency = operators.ndim == 2
    if single_frequency:
        operators = operators[None, ...]
    if data.ndim == 2:
        data = data[None, ...]
    if operators.ndim != 3 or data.ndim != 3:
        raise ValueError("leadfields and observations must have (F,E,P)/(F,E,T) shapes")
    if operators.shape[:2] != data.shape[:2]:
        raise ValueError("leadfields and observations must share frequency/sensor axes")
    if not all(operators.shape) or not all(data.shape):
        raise ValueError("leadfields and observations must have non-empty axes")
    if not np.all(np.isfinite(operators)) or not np.all(np.isfinite(data)):
        raise ValueError("leadfields and observations must be finite")
    if single_frequency != (np.asarray(observations).ndim == 2):
        raise ValueError("leadfields and observations must use the same frequency convention")
    return operators, data, single_frequency


def _frequency_weights(values: ArrayLike | None, count: int) -> FloatArray:
    if values is None:
        result = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if (
            result.shape != (count,)
            or np.any(result <= 0.0)
            or not np.all(np.isfinite(result))
        ):
            raise ValueError("frequency weights must be finite, positive, and aligned")
        total = float(result.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("frequency weights have an invalid sum")
        result = result / total
    return result


def _validate_exponent(exponent: float) -> float:
    value = float(exponent)
    if not np.isfinite(value) or value not in DEPTH_EXPONENT_GRID:
        raise ValueError(f"depth exponent must be one of {DEPTH_EXPONENT_GRID}")
    return value


def _normal_frame(normal: np.ndarray) -> np.ndarray:
    """Return deterministic tangential-1/tangential-2/normal columns."""

    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    reference = np.eye(3)[int(np.argmin(np.abs(unit)))]
    tangent_one = np.cross(unit, reference)
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(unit, tangent_one)
    tangent_two /= np.linalg.norm(tangent_two)
    return np.column_stack((tangent_one, tangent_two, unit))


@dataclass(frozen=True)
class OrientationTransform:
    """Block orientation map from solver coordinates to global XYZ dipoles."""

    mode: str
    matrix: ComplexArray
    pseudoinverse: ComplexArray
    n_groups: int
    input_group_size: int
    output_group_size: int
    audit: Mapping[str, Any]

    def transform_leadfields(self, leadfields: ArrayLike) -> ComplexArray:
        operators = np.asarray(leadfields, dtype=np.complex128)
        if operators.ndim not in {2, 3} or operators.shape[-1] != self.matrix.shape[0]:
            raise ValueError("leadfield coefficient axis does not match orientation transform")
        return np.asarray(operators @ self.matrix, dtype=np.complex128)

    def to_physical_mean(self, solver_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(solver_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != self.matrix.shape[1]:
            raise ValueError("solver mean does not match orientation transform")
        return np.asarray(self.matrix @ mean, dtype=np.complex128)

    def to_solver_mean(self, physical_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(physical_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != self.matrix.shape[0]:
            raise ValueError("physical mean does not match orientation transform")
        return np.asarray(self.pseudoinverse @ mean, dtype=np.complex128)

    def to_physical_covariance(self, solver_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("solver covariance", solver_covariance, 2)
        if covariance.shape != (self.matrix.shape[1], self.matrix.shape[1]):
            raise ValueError("solver covariance does not match orientation transform")
        return np.asarray(
            self.matrix @ covariance @ self.matrix.conj().T, dtype=np.complex128
        )

    def to_solver_covariance(self, physical_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("physical covariance", physical_covariance, 2)
        if covariance.shape != (self.matrix.shape[0], self.matrix.shape[0]):
            raise ValueError("physical covariance does not match orientation transform")
        return np.asarray(
            self.pseudoinverse @ covariance @ self.pseudoinverse.conj().T,
            dtype=np.complex128,
        )


def build_orientation_transform(
    n_groups: int,
    *,
    mode: str = "free_3d",
    source_normals: ArrayLike | None = None,
) -> OrientationTransform:
    """Build one of the three preregistered, truth-independent orientations.

    ``loose_0.2`` treats 0.2 as the tangential-to-normal *prior variance*
    ratio.  Its operator transform therefore uses tangential standard
    deviation ``sqrt(0.2)``.  ``cortical_normal`` reduces each three-column XYZ
    block to one anatomically supplied normal column.
    """

    count = int(n_groups)
    if count < 1 or count != n_groups:
        raise ValueError("n_groups must be a positive integer")
    selected = str(mode).strip().lower()
    if selected not in ORIENTATION_MODES:
        raise ValueError(f"orientation mode must be one of {ORIENTATION_MODES}")

    if selected == "free_3d":
        matrix = np.eye(3 * count, dtype=np.complex128)
        output_group_size = 3
        normal_fingerprint = None
    else:
        normals = np.asarray(source_normals, dtype=np.float64)
        if normals.shape != (count, 3) or not np.all(np.isfinite(normals)):
            raise ValueError("source_normals must be finite with shape (n_groups, 3)")
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("source normals must be non-zero")
        normals = normals / norms[:, None]
        output_group_size = 1 if selected == "cortical_normal" else 3
        matrix = np.zeros(
            (3 * count, output_group_size * count), dtype=np.complex128
        )
        for group, normal in enumerate(normals):
            physical = slice(3 * group, 3 * (group + 1))
            solver = slice(
                output_group_size * group, output_group_size * (group + 1)
            )
            if selected == "cortical_normal":
                block = normal[:, None]
            else:
                block = _normal_frame(normal) @ np.diag(
                    [np.sqrt(0.2), np.sqrt(0.2), 1.0]
                )
            matrix[physical, solver] = block
        normal_fingerprint = np.round(normals, decimals=15).tolist()

    pseudoinverse = np.linalg.pinv(matrix, rcond=1.0e-14)
    audit = {
        "orientation_mode": selected,
        "input_basis": "global_XYZ_three_columns_per_group",
        "output_group_size": output_group_size,
        "loose_tangential_variance_ratio": 0.2 if selected == "loose_0.2" else None,
        "source_normals_unit": normal_fingerprint,
        "truth_used": False,
        "allowed_selection_boundary": "calibration-heldout-predictive-NLL-only",
    }
    return OrientationTransform(
        mode=selected,
        matrix=np.asarray(matrix, dtype=np.complex128),
        pseudoinverse=np.asarray(pseudoinverse, dtype=np.complex128),
        n_groups=count,
        input_group_size=3,
        output_group_size=output_group_size,
        audit=audit,
    )


def _covariance_family(
    value: ArrayLike | float | None,
    *,
    n_frequency: int,
    n_sensor: int,
    name: str,
    identity_if_none: bool,
) -> ComplexArray:
    if value is None:
        base = 1.0 if identity_if_none else 0.0
        return np.repeat(
            (base * np.eye(n_sensor, dtype=np.complex128))[None, ...],
            n_frequency,
            axis=0,
        )
    if np.isscalar(value):
        scalar = complex(value)
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag) or scalar.imag != 0.0:
            raise ValueError(f"{name} scalar must be finite and real")
        return np.repeat(
            (scalar.real * np.eye(n_sensor, dtype=np.complex128))[None, ...],
            n_frequency,
            axis=0,
        )
    covariance = np.asarray(value, dtype=np.complex128)
    if covariance.shape == (n_sensor, n_sensor):
        covariance = np.repeat(covariance[None, ...], n_frequency, axis=0)
    if covariance.shape != (n_frequency, n_sensor, n_sensor):
        raise ValueError(
            f"{name} must be scalar, (E,E), or (F,E,E) and align with leadfields"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(covariance, dtype=np.complex128)


def _hermitian_covariances(
    covariances: np.ndarray, *, name: str, positive_definite: bool
) -> ComplexArray:
    checked = np.empty_like(covariances, dtype=np.complex128)
    for frequency, covariance in enumerate(covariances):
        scale = max(float(np.linalg.norm(covariance, ord=2)), 1.0)
        if not np.allclose(
            covariance, covariance.conj().T, rtol=0.0, atol=1.0e-10 * scale
        ):
            raise ValueError(f"{name}[{frequency}] must be Hermitian")
        hermitian = 0.5 * (covariance + covariance.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian)
        tolerance = 1.0e-12 * scale
        if positive_definite and float(eigenvalues[0]) <= tolerance:
            raise ValueError(f"{name}[{frequency}] must be positive definite")
        if not positive_definite and float(eigenvalues[0]) < -tolerance:
            raise ValueError(f"{name}[{frequency}] must be positive semidefinite")
        checked[frequency] = hermitian
    return checked


@dataclass(frozen=True)
class DepthCompensation:
    """Frequency-aware group depth transform in oriented source coordinates."""

    exponent: float
    frequency_weights: FloatArray
    group_energy_by_frequency: FloatArray
    aggregate_group_energy: FloatArray
    group_depth_scale: FloatArray
    reference_depth_scale: float
    normalized_group_depth_scale: FloatArray
    group_multiplier: FloatArray
    coefficient_multiplier: FloatArray
    group_size: int
    audit: Mapping[str, Any]

    @property
    def matrix(self) -> ComplexArray:
        return np.diag(self.coefficient_multiplier).astype(np.complex128)

    def transform_leadfields(self, leadfields: ArrayLike) -> ComplexArray:
        operators = np.asarray(leadfields, dtype=np.complex128)
        if operators.ndim not in {2, 3} or operators.shape[-1] != len(
            self.coefficient_multiplier
        ):
            raise ValueError("leadfield coefficient axis does not match depth transform")
        return np.asarray(
            operators * self.coefficient_multiplier, dtype=np.complex128
        )

    def to_physical_mean(self, solver_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(solver_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != len(self.coefficient_multiplier):
            raise ValueError("solver mean does not match depth transform")
        scale = self.coefficient_multiplier
        return np.asarray(scale * mean if mean.ndim == 1 else scale[:, None] * mean)

    def to_solver_mean(self, physical_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(physical_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != len(self.coefficient_multiplier):
            raise ValueError("physical mean does not match depth transform")
        scale = self.coefficient_multiplier
        return np.asarray(mean / scale if mean.ndim == 1 else mean / scale[:, None])

    def to_physical_covariance(self, solver_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("solver covariance", solver_covariance, 2)
        size = len(self.coefficient_multiplier)
        if covariance.shape != (size, size):
            raise ValueError("solver covariance does not match depth transform")
        scale = self.coefficient_multiplier
        return np.asarray(scale[:, None] * covariance * scale[None, :])

    def to_solver_covariance(self, physical_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("physical covariance", physical_covariance, 2)
        size = len(self.coefficient_multiplier)
        if covariance.shape != (size, size):
            raise ValueError("physical covariance does not match depth transform")
        scale = self.coefficient_multiplier
        return np.asarray(covariance / scale[:, None] / scale[None, :])


def compute_group_depth_compensation(
    leadfields: ArrayLike,
    *,
    noise_covariances: ArrayLike | float | None = None,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
    exponent: float = 0.0,
) -> DepthCompensation:
    r"""Compute the preregistered group depth scale without matrix inversion.

    For frequency ``f`` and source group ``g`` with ``r`` orientations,

    ``q_fg = tr(G_fg^H Cn_f^-1 G_fg) / r``

    ``d_g = sqrt(sum_f pi_f q_fg)``.

    The dimensionless depth is ``d_g / geometric_mean(d)`` and the operator
    multiplier is ``s_g = (d_g / geometric_mean(d)) ** (-p)``.  Each of the
    ``r`` columns in a group receives the same multiplier.
    """

    operators = np.asarray(leadfields, dtype=np.complex128)
    if operators.ndim == 2:
        operators = operators[None, ...]
    if operators.ndim != 3 or not np.all(np.isfinite(operators)) or not all(
        operators.shape
    ):
        raise ValueError("leadfields must be finite and have shape (F,E,P) or (E,P)")
    size = int(group_size)
    if size < 1 or size != group_size or operators.shape[2] % size:
        raise ValueError("leadfield columns must be divisible by a positive group_size")
    depth_exponent = _validate_exponent(exponent)
    weights = _frequency_weights(frequency_weights, operators.shape[0])
    covariance = _covariance_family(
        noise_covariances,
        n_frequency=operators.shape[0],
        n_sensor=operators.shape[1],
        name="noise_covariances",
        identity_if_none=True,
    )
    covariance = _hermitian_covariances(
        covariance, name="noise_covariances", positive_definite=True
    )

    n_groups = operators.shape[2] // size
    energy = np.empty((operators.shape[0], n_groups), dtype=np.float64)
    for frequency, (operator, noise) in enumerate(zip(operators, covariance)):
        factor = linalg.cho_factor(noise, lower=True, check_finite=False)
        solved = linalg.cho_solve(factor, operator, check_finite=False)
        for group in range(n_groups):
            sl = slice(size * group, size * (group + 1))
            value = np.trace(operator[:, sl].conj().T @ solved[:, sl]).real / size
            energy[frequency, group] = max(float(value), 0.0)
    aggregate = np.asarray(weights @ energy, dtype=np.float64)
    maximum = max(float(np.max(aggregate, initial=0.0)), np.finfo(float).tiny)
    floor = max(np.finfo(float).eps * maximum, np.finfo(float).tiny)
    floored = np.maximum(aggregate, floor)
    depth = np.sqrt(floored)
    reference = float(np.exp(np.mean(np.log(depth))))
    normalized = depth / reference
    multiplier = normalized ** (-depth_exponent)
    coefficient = np.repeat(multiplier, size).astype(np.float64)
    audit = {
        "definition": "q_fg=tr(G_fg^H Cn_f^-1 G_fg)/r; d_g=sqrt(sum pi_f q_fg)",
        "reference": "geometric_mean_of_positive_group_depth_scales",
        "operator_transform": "G_tilde[f,g] = G[f,g] * (d_g/d_ref)^(-p)",
        "posterior_mean_back_transform": "mu_physical = D mu_solver",
        "posterior_covariance_back_transform": "Sigma_physical = D Sigma_solver D^H",
        "depth_exponent_grid": list(DEPTH_EXPONENT_GRID),
        "floored_group_count": int(np.sum(aggregate < floor)),
        "truth_used": False,
    }
    return DepthCompensation(
        exponent=depth_exponent,
        frequency_weights=weights,
        group_energy_by_frequency=energy,
        aggregate_group_energy=aggregate,
        group_depth_scale=depth,
        reference_depth_scale=reference,
        normalized_group_depth_scale=normalized,
        group_multiplier=multiplier,
        coefficient_multiplier=coefficient,
        group_size=size,
        audit=audit,
    )


@dataclass(frozen=True)
class SourceTransform:
    """Combined orientation/depth map from Champagne to physical XYZ space."""

    orientation: OrientationTransform
    depth: DepthCompensation
    matrix: ComplexArray
    pseudoinverse: ComplexArray
    audit: Mapping[str, Any]

    def transform_leadfields(self, leadfields: ArrayLike) -> ComplexArray:
        operators = np.asarray(leadfields, dtype=np.complex128)
        if operators.ndim not in {2, 3} or operators.shape[-1] != self.matrix.shape[0]:
            raise ValueError("leadfield coefficient axis does not match source transform")
        return np.asarray(operators @ self.matrix, dtype=np.complex128)

    def to_physical_mean(self, solver_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(solver_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != self.matrix.shape[1]:
            raise ValueError("solver mean does not match source transform")
        return np.asarray(self.matrix @ mean, dtype=np.complex128)

    def to_solver_mean(self, physical_mean: ArrayLike) -> ComplexArray:
        mean = np.asarray(physical_mean, dtype=np.complex128)
        if mean.ndim not in {1, 2} or mean.shape[0] != self.matrix.shape[0]:
            raise ValueError("physical mean does not match source transform")
        return np.asarray(self.pseudoinverse @ mean, dtype=np.complex128)

    def to_physical_covariance(self, solver_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("solver covariance", solver_covariance, 2)
        if covariance.shape != (self.matrix.shape[1], self.matrix.shape[1]):
            raise ValueError("solver covariance does not match source transform")
        return np.asarray(
            self.matrix @ covariance @ self.matrix.conj().T, dtype=np.complex128
        )

    def to_solver_covariance(self, physical_covariance: ArrayLike) -> ComplexArray:
        covariance = _finite_complex("physical covariance", physical_covariance, 2)
        if covariance.shape != (self.matrix.shape[0], self.matrix.shape[0]):
            raise ValueError("physical covariance does not match source transform")
        return np.asarray(
            self.pseudoinverse @ covariance @ self.pseudoinverse.conj().T,
            dtype=np.complex128,
        )


def build_source_transform(
    leadfields: ArrayLike,
    *,
    orientation_mode: str = "free_3d",
    source_normals: ArrayLike | None = None,
    noise_covariances: ArrayLike | float | None = None,
    frequency_weights: ArrayLike | None = None,
    depth_exponent: float = 0.0,
) -> SourceTransform:
    """Construct the exact combined map used by Calibrated-Champagne."""

    operators = np.asarray(leadfields, dtype=np.complex128)
    if operators.ndim == 2:
        operators = operators[None, ...]
    if operators.ndim != 3 or operators.shape[2] % 3:
        raise ValueError("input leadfields must use three global-XYZ columns per group")
    orientation = build_orientation_transform(
        operators.shape[2] // 3,
        mode=orientation_mode,
        source_normals=source_normals,
    )
    oriented = orientation.transform_leadfields(operators)
    depth = compute_group_depth_compensation(
        oriented,
        noise_covariances=noise_covariances,
        frequency_weights=frequency_weights,
        group_size=orientation.output_group_size,
        exponent=depth_exponent,
    )
    matrix = orientation.matrix @ depth.matrix
    pseudoinverse = np.linalg.pinv(matrix, rcond=1.0e-14)
    return SourceTransform(
        orientation=orientation,
        depth=depth,
        matrix=np.asarray(matrix, dtype=np.complex128),
        pseudoinverse=np.asarray(pseudoinverse, dtype=np.complex128),
        audit={
            "orientation": dict(orientation.audit),
            "depth": dict(depth.audit),
            "combined_operator_transform": "G_tilde = G O D",
            "combined_mean_back_transform": "mu_XYZ = O D mu_solver",
            "combined_covariance_back_transform": "Sigma_XYZ = O D Sigma_solver D^H O^H",
            "truth_used": False,
        },
    )


@dataclass(frozen=True)
class CalibratedChampagneResult:
    """Physical-coordinate posterior returned by the existing Champagne core."""

    method: str
    source_estimate: ComplexArray
    source_covariance: ComplexArray
    latent_source_estimate: ComplexArray
    latent_source_covariance: ComplexArray
    source_score: FloatArray
    noise: float
    objective: tuple[float, ...]
    converged: bool
    iterations: int
    transformed_leadfields: ComplexArray
    stacked_operator: ComplexArray
    stacked_observations: ComplexArray
    source_transform: SourceTransform
    metadata: Mapping[str, Any]

    @property
    def mean(self) -> ComplexArray:
        return self.source_estimate

    @property
    def covariance(self) -> ComplexArray:
        return self.source_covariance

    @property
    def gamma(self) -> FloatArray:
        return self.source_score


def calibrated_champagne(
    leadfields: ArrayLike,
    observations: ArrayLike,
    *,
    theta: Sequence[float] | np.ndarray | None = None,
    frequency_weights: ArrayLike | None = None,
    noise_covariances: ArrayLike | float | None = None,
    bae_covariances: ArrayLike | float | None = None,
    depth_exponent: float = 0.0,
    orientation_mode: str = "free_3d",
    source_normals: ArrayLike | None = None,
    champagne_kwargs: Mapping[str, Any] | None = None,
) -> CalibratedChampagneResult:
    """Run the existing Champagne implementation under explicit V2 transforms.

    ``leadfields`` must already be nonlinearly reassembled at ``theta``; theta
    is retained solely for provenance.  The function never estimates theta or
    chooses orientation/depth settings and has no truth-valued argument.

    Noise and optional BAE covariances define ``C_eff``.  Both operator and
    observations are whitened by ``C_eff^-1/2`` before frequency weighting and
    stacking.  With identity/default noise, BAE off, uniform weights,
    ``free_3d`` and ``depth_exponent=0``, this is the exact input passed to the
    historical :func:`inverse.baselines.champagne_sbl` baseline.
    """

    operators, data, _ = _family_problem(leadfields, observations)
    if operators.shape[2] % 3:
        raise ValueError("Calibrated-Champagne requires free-XYZ three-column groups")
    weights = _frequency_weights(frequency_weights, operators.shape[0])
    theta_array = (
        np.empty(0, dtype=np.float64)
        if theta is None
        else np.asarray(theta, dtype=np.float64)
    )
    if theta_array.ndim != 1 or not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must be a finite one-dimensional vector")

    noise = _covariance_family(
        noise_covariances,
        n_frequency=operators.shape[0],
        n_sensor=operators.shape[1],
        name="noise_covariances",
        identity_if_none=True,
    )
    noise = _hermitian_covariances(
        noise, name="noise_covariances", positive_definite=True
    )
    bae_enabled = bae_covariances is not None
    bae = _covariance_family(
        bae_covariances,
        n_frequency=operators.shape[0],
        n_sensor=operators.shape[1],
        name="bae_covariances",
        identity_if_none=False,
    )
    bae = _hermitian_covariances(
        bae, name="bae_covariances", positive_definite=False
    )
    effective_noise = _hermitian_covariances(
        noise + bae, name="effective_noise_covariances", positive_definite=True
    )

    transform = build_source_transform(
        operators,
        orientation_mode=orientation_mode,
        source_normals=source_normals,
        noise_covariances=effective_noise,
        frequency_weights=weights,
        depth_exponent=depth_exponent,
    )
    transformed = transform.transform_leadfields(operators)
    whitened_operators: list[np.ndarray] = []
    whitened_data: list[np.ndarray] = []
    for operator, target, covariance in zip(transformed, data, effective_noise):
        factor = linalg.cholesky(covariance, lower=True, check_finite=False)
        whitened_operators.append(
            linalg.solve_triangular(
                factor, operator, lower=True, check_finite=False
            )
        )
        whitened_data.append(
            linalg.solve_triangular(
                factor, target, lower=True, check_finite=False
            )
        )
    scale = np.sqrt(weights)
    stacked_operator = np.vstack(
        [factor * operator for factor, operator in zip(scale, whitened_operators)]
    ).astype(np.complex128)
    stacked_observations = np.vstack(
        [factor * target for factor, target in zip(scale, whitened_data)]
    ).astype(np.complex128)

    settings = dict(champagne_kwargs or {})
    if "group_size" in settings:
        raise ValueError("group_size is determined by the orientation transform")
    baseline = baselines.champagne_sbl(
        stacked_operator,
        stacked_observations,
        group_size=transform.orientation.output_group_size,
        **settings,
    )
    gamma = np.asarray(baseline.source_score, dtype=np.float64)
    noise_level = float(baseline.metadata["noise"])
    reconstructed_mean, latent_covariance, _ = complex_gaussian_posterior(
        stacked_operator,
        stacked_observations,
        gamma,
        noise_level,
        group_size=transform.orientation.output_group_size,
    )
    latent_mean = np.asarray(baseline.source_estimate, dtype=np.complex128)
    posterior_reconstruction_error = float(
        np.linalg.norm(reconstructed_mean - latent_mean)
        / max(float(np.linalg.norm(latent_mean)), np.finfo(float).tiny)
    )
    physical_mean = transform.to_physical_mean(latent_mean)
    physical_covariance = transform.to_physical_covariance(latent_covariance)
    objective = tuple(float(value) for value in baseline.metadata["objective"])
    metadata = {
        "solver": "inverse.baselines.champagne_sbl",
        "existing_champagne_reused": True,
        "theta": theta_array.tolist(),
        "theta_operator_contract": "input leadfields already reassembled at recorded theta",
        "frequency_weights": weights.tolist(),
        "bae_enabled": bae_enabled,
        "depth_exponent": transform.depth.exponent,
        "orientation_mode": transform.orientation.mode,
        "source_transform": dict(transform.audit),
        "posterior_reconstruction_relative_error": posterior_reconstruction_error,
        "truth_used_for_fit_or_selection": False,
    }
    return CalibratedChampagneResult(
        method="Calibrated-Champagne",
        source_estimate=physical_mean,
        source_covariance=physical_covariance,
        latent_source_estimate=latent_mean,
        latent_source_covariance=np.asarray(latent_covariance, dtype=np.complex128),
        source_score=gamma,
        noise=noise_level,
        objective=objective,
        converged=bool(baseline.metadata["converged"]),
        iterations=int(baseline.metadata["iterations"]),
        transformed_leadfields=np.asarray(transformed, dtype=np.complex128),
        stacked_operator=stacked_operator,
        stacked_observations=stacked_observations,
        source_transform=transform,
        metadata=metadata,
    )


__all__ = [
    "CalibratedChampagneResult",
    "DEPTH_EXPONENT_GRID",
    "DepthCompensation",
    "ORIENTATION_MODES",
    "OrientationTransform",
    "SourceTransform",
    "build_orientation_transform",
    "build_source_transform",
    "calibrated_champagne",
    "compute_group_depth_compensation",
]
