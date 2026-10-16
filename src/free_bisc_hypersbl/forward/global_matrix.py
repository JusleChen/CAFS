"""Frequency-dependent layered-sphere spherical-harmonic forward operator.

The boundary-condition layout in this module is adapted from the auditable
``fdgm/operators/dgm_frequency.py`` implementation in the legacy workspace.
Only the mathematical layered-sphere construction is reused.  In particular,
the legacy Gaussian-distance ``scalar_leadfield`` surrogate is *not* used:
electrode observations here are evaluations of complex spherical harmonics.

For one spherical-harmonic mode ``(ell, m)``, the potential in layer ``i`` is

``Phi_i(r) = a_i r**ell + b_i r**(-ell-1)``.

The global matrix enforces a prescribed inner multipole coefficient, potential
and normal-current continuity at every interface, and zero normal current at
the scalp.  All radii are nondimensionalised by the nominal scalp radius to
avoid the severe, purely unit-induced conditioning caused by powers of metres.
The resulting matrices and solves are always ``complex128``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.special import gammaln, lpmv


EPSILON_0_F_PER_M = 8.8541878128e-12
_AXIS_TOL = 1.0e-12


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _readonly_float_array(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LayeredSphereHead:
    """Geometry and dispersive tissue properties of a concentric-layer head.

    ``radii_m[i]`` is the outer radius of layer ``i``.  The first layer is used
    as the fixed admittivity reference; calibration therefore operates on
    ratios for selected remaining layers, avoiding the global-conductivity
    ambiguity of the quasistatic EEG equations.
    """

    radii_m: ArrayLike
    conductivity_s_per_m: ArrayLike
    relative_permittivity: ArrayLike
    electrode_xyz_m: ArrayLike
    dispersion_alpha: ArrayLike | None = None
    dispersion_reference_hz: float = 20.0
    layer_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        radii = _readonly_float_array(self.radii_m, name="radii_m", ndim=1)
        conductivity = _readonly_float_array(
            self.conductivity_s_per_m,
            name="conductivity_s_per_m",
            ndim=1,
        )
        permittivity = _readonly_float_array(
            self.relative_permittivity,
            name="relative_permittivity",
            ndim=1,
        )
        electrodes = _readonly_float_array(
            self.electrode_xyz_m,
            name="electrode_xyz_m",
            ndim=2,
        )

        if radii.size < 2:
            raise ValueError("at least two concentric layers are required")
        if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("radii_m must be positive and strictly increasing")
        if conductivity.shape != radii.shape or permittivity.shape != radii.shape:
            raise ValueError("one conductivity and permittivity are required per layer")
        if np.any(conductivity <= 0.0) or np.any(permittivity < 0.0):
            raise ValueError("conductivity must be positive and permittivity non-negative")
        if electrodes.shape[1:] != (3,) or electrodes.shape[0] < 1:
            raise ValueError("electrode_xyz_m must have shape (n_electrodes, 3)")
        if np.any(np.linalg.norm(electrodes, axis=1) <= 0.0):
            raise ValueError("electrode positions must not be at the origin")

        if self.dispersion_alpha is None:
            dispersion = np.zeros_like(radii)
            dispersion.setflags(write=False)
        else:
            dispersion = _readonly_float_array(
                self.dispersion_alpha,
                name="dispersion_alpha",
                ndim=1,
            )
            if dispersion.shape != radii.shape:
                raise ValueError("one dispersion_alpha is required per layer")

        reference_hz = float(self.dispersion_reference_hz)
        if not np.isfinite(reference_hz) or reference_hz <= 0.0:
            raise ValueError("dispersion_reference_hz must be positive")

        names = self.layer_names or tuple(f"layer_{i}" for i in range(radii.size))
        if len(names) != radii.size or len(set(names)) != len(names):
            raise ValueError("layer_names must be unique and match the layer count")

        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "conductivity_s_per_m", conductivity)
        object.__setattr__(self, "relative_permittivity", permittivity)
        object.__setattr__(self, "electrode_xyz_m", electrodes)
        object.__setattr__(self, "dispersion_alpha", dispersion)
        object.__setattr__(self, "dispersion_reference_hz", reference_hz)
        object.__setattr__(self, "layer_names", tuple(str(name) for name in names))

    @property
    def n_layers(self) -> int:
        return int(np.asarray(self.radii_m).size)

    @property
    def length_scale_m(self) -> float:
        """Fixed nondimensionalisation length (the nominal scalp radius)."""

        return float(np.asarray(self.radii_m)[-1])

    def admittivity(self, frequency_hz: float) -> ComplexArray:
        """Return ``sigma(f) + i omega epsilon`` for every tissue layer."""

        frequency = float(frequency_hz)
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        dispersion = 1.0 + np.asarray(self.dispersion_alpha) * np.log1p(
            frequency / self.dispersion_reference_hz
        )
        if np.any(dispersion <= 0.0):
            raise ValueError("the selected dispersion parameters give non-positive conductivity")
        real = np.asarray(self.conductivity_s_per_m) * dispersion
        imag = (
            2.0
            * np.pi
            * frequency
            * EPSILON_0_F_PER_M
            * np.asarray(self.relative_permittivity)
        )
        return np.asarray(real + 1j * imag, dtype=np.complex128)


@dataclass(frozen=True)
class CalibrationSpec:
    """Low-dimensional real parameterisation of complex forward physics.

    A log ratio multiplies the complete complex admittivity of its layer, so
    both conductive and dielectric parts remain complex-valued.  Layer zero is
    the fixed absolute reference by default.  Translation parameters are in
    metres and are applied as a small rigid displacement to all electrodes.
    """

    admittivity_ratio_layers: tuple[int, ...] | None = None
    include_log_radius_scale: bool = True
    include_electrode_translation: bool = True
    source_scale_power: float = 0.0

    def resolved_ratio_layers(self, n_layers: int) -> tuple[int, ...]:
        layers = (
            tuple(range(1, n_layers))
            if self.admittivity_ratio_layers is None
            else tuple(int(index) for index in self.admittivity_ratio_layers)
        )
        if len(set(layers)) != len(layers):
            raise ValueError("admittivity_ratio_layers contains duplicate indices")
        if any(index <= 0 or index >= n_layers for index in layers):
            raise ValueError("admittivity ratio layers must exclude reference layer 0")
        return layers

    def parameter_names(self, head: LayeredSphereHead) -> tuple[str, ...]:
        names = [
            f"log_admittivity_ratio_{head.layer_names[index]}"
            for index in self.resolved_ratio_layers(head.n_layers)
        ]
        if self.include_log_radius_scale:
            names.append("log_radius_scale")
        if self.include_electrode_translation:
            names.extend(
                (
                    "electrode_translation_x_m",
                    "electrode_translation_y_m",
                    "electrode_translation_z_m",
                )
            )
        return tuple(names)


@dataclass(frozen=True)
class GlobalMatrixSystem:
    """Assembled ``K X = B``, observation matrix ``H``, and their derivatives."""

    frequency_hz: float
    ell: int
    m: int
    parameter_names: tuple[str, ...]
    K: ComplexArray
    B: ComplexArray
    H: ComplexArray
    K_q: ComplexArray
    B_q: ComplexArray
    H_q: ComplexArray
    admittivity_s_per_m: ComplexArray
    radii_dimensionless: FloatArray
    electrode_xyz_dimensionless: FloatArray


@dataclass(frozen=True)
class ForwardSolution:
    """Solved harmonic lead field and analytic derivatives."""

    system: GlobalMatrixSystem
    X: ComplexArray
    G: ComplexArray
    D: ComplexArray
    relative_residual: float

    @property
    def jacobian_by_name(self) -> dict[str, ComplexArray]:
        return {
            name: self.D[index]
            for index, name in enumerate(self.system.parameter_names)
        }


@dataclass(frozen=True)
class LeadfieldFamily:
    """Complete multi-frequency, three-orientation lead-field family."""

    frequencies_hz: FloatArray
    source_xyz_m: FloatArray
    leadfields: ComplexArray
    jacobians: ComplexArray
    parameter_names: tuple[str, ...]
    l_max: int
    relative_residual_max: float


def _coerce_theta(
    theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None,
    names: tuple[str, ...],
) -> FloatArray:
    if theta is None:
        values = np.zeros(len(names), dtype=np.float64)
    elif isinstance(theta, Mapping):
        unknown = set(theta).difference(names)
        if unknown:
            raise ValueError(f"unknown calibration parameters: {sorted(unknown)}")
        values = np.asarray([float(theta.get(name, 0.0)) for name in names], dtype=np.float64)
    else:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (len(names),):
            raise ValueError(f"theta must have shape ({len(names)},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta contains non-finite values")
    return values


def _complex_spherical_harmonic_and_dtheta(
    ell: int,
    m: int,
    theta: FloatArray,
    phi: FloatArray,
) -> tuple[ComplexArray, ComplexArray]:
    """Complex ``Y_lm`` and its analytic colatitude derivative."""

    order = abs(m)
    x = np.cos(theta)
    p_lm = lpmv(order, ell, x)
    p_previous = np.zeros_like(p_lm) if ell == order else lpmv(order, ell - 1, x)
    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) > _AXIS_TOL, sin_theta, _AXIS_TOL)
    dp_dtheta = (ell * x * p_lm - (ell + order) * p_previous) / safe_sin

    log_norm = 0.5 * (
        np.log((2 * ell + 1) / (4.0 * np.pi))
        + gammaln(ell - order + 1)
        - gammaln(ell + order + 1)
    )
    phase = np.exp(1j * order * phi)
    y_positive = np.exp(log_norm) * p_lm * phase
    dy_positive = np.exp(log_norm) * dp_dtheta * phase

    if m < 0:
        parity = -1.0 if order % 2 else 1.0
        return (
            np.asarray(parity * np.conjugate(y_positive), dtype=np.complex128),
            np.asarray(parity * np.conjugate(dy_positive), dtype=np.complex128),
        )
    return (
        np.asarray(y_positive, dtype=np.complex128),
        np.asarray(dy_positive, dtype=np.complex128),
    )


def _solid_harmonic_and_gradient(
    xyz: FloatArray,
    *,
    ell: int,
    m: int,
    radial_power: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Evaluate ``r**radial_power Y_lm`` and its Cartesian gradient."""

    radius = np.linalg.norm(xyz, axis=1)
    if np.any(radius <= 0.0):
        raise ValueError("translated electrode position reached the origin")
    rho = np.linalg.norm(xyz[:, :2], axis=1)
    theta = np.arctan2(rho, xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    y_lm, dy_dtheta = _complex_spherical_harmonic_and_dtheta(ell, m, theta, phi)

    radial = radius**radial_power
    value = np.asarray(radial * y_lm, dtype=np.complex128)
    dvalue_dr = radial_power * radius ** (radial_power - 1) * y_lm
    dvalue_dtheta = radial * dy_dtheta
    dvalue_dphi = 1j * m * value

    dr_dxyz = xyz / radius[:, None]
    safe_rho = np.where(rho > _AXIS_TOL, rho, _AXIS_TOL)
    dtheta_dxyz = np.column_stack(
        (
            xyz[:, 0] * xyz[:, 2] / (radius**2 * safe_rho),
            xyz[:, 1] * xyz[:, 2] / (radius**2 * safe_rho),
            -rho / radius**2,
        )
    )
    dphi_dxyz = np.column_stack(
        (
            -xyz[:, 1] / safe_rho**2,
            xyz[:, 0] / safe_rho**2,
            np.zeros_like(radius),
        )
    )
    gradient = (
        dvalue_dr[:, None] * dr_dxyz
        + dvalue_dtheta[:, None] * dtheta_dxyz
        + dvalue_dphi[:, None] * dphi_dxyz
    )
    return value, np.asarray(gradient, dtype=np.complex128)


def _power_with_derivatives(
    radius: float,
    power: int,
    radius_q: FloatArray,
) -> tuple[float, FloatArray]:
    value = float(radius**power)
    derivative = power * value * radius_q / radius
    return value, np.asarray(derivative, dtype=np.float64)


def _assemble_k(
    ell: int,
    radii: FloatArray,
    radii_q: FloatArray,
    admittivity: ComplexArray,
    admittivity_q: ComplexArray,
) -> tuple[ComplexArray, ComplexArray]:
    """Assemble layered boundary conditions and their analytic derivatives."""

    n_layers = radii.size
    n_unknown = 2 * n_layers
    n_parameters = radii_q.shape[0]
    k = np.zeros((n_unknown, n_unknown), dtype=np.complex128)
    k_q = np.zeros((n_parameters, n_unknown, n_unknown), dtype=np.complex128)

    def a_index(layer: int) -> int:
        return 2 * layer

    def b_index(layer: int) -> int:
        return 2 * layer + 1

    # The singular coefficient in the innermost layer is the prescribed source
    # multipole.  The source values themselves are stored in the matching B row.
    k[0, b_index(0)] = 1.0

    row = 1
    for interface in range(n_layers - 1):
        radius = float(radii[interface])
        dr = radii_q[:, interface]
        r_l, dr_l = _power_with_derivatives(radius, ell, dr)
        r_minus, dr_minus = _power_with_derivatives(radius, -ell - 1, dr)

        k[row, a_index(interface)] = r_l
        k[row, b_index(interface)] = r_minus
        k[row, a_index(interface + 1)] = -r_l
        k[row, b_index(interface + 1)] = -r_minus
        k_q[:, row, a_index(interface)] = dr_l
        k_q[:, row, b_index(interface)] = dr_minus
        k_q[:, row, a_index(interface + 1)] = -dr_l
        k_q[:, row, b_index(interface + 1)] = -dr_minus
        row += 1

        r_grad, dr_grad = _power_with_derivatives(radius, ell - 1, dr)
        r_decay, dr_decay = _power_with_derivatives(radius, -ell - 2, dr)
        left_gamma = admittivity[interface]
        right_gamma = admittivity[interface + 1]
        left_gamma_q = admittivity_q[:, interface]
        right_gamma_q = admittivity_q[:, interface + 1]

        k[row, a_index(interface)] = left_gamma * ell * r_grad
        k[row, b_index(interface)] = -left_gamma * (ell + 1) * r_decay
        k[row, a_index(interface + 1)] = -right_gamma * ell * r_grad
        k[row, b_index(interface + 1)] = right_gamma * (ell + 1) * r_decay

        k_q[:, row, a_index(interface)] = ell * (
            left_gamma_q * r_grad + left_gamma * dr_grad
        )
        k_q[:, row, b_index(interface)] = -(ell + 1) * (
            left_gamma_q * r_decay + left_gamma * dr_decay
        )
        k_q[:, row, a_index(interface + 1)] = -ell * (
            right_gamma_q * r_grad + right_gamma * dr_grad
        )
        k_q[:, row, b_index(interface + 1)] = (ell + 1) * (
            right_gamma_q * r_decay + right_gamma * dr_decay
        )
        row += 1

    outer_radius = float(radii[-1])
    dr_outer = radii_q[:, -1]
    r_grad, dr_grad = _power_with_derivatives(outer_radius, ell - 1, dr_outer)
    r_decay, dr_decay = _power_with_derivatives(outer_radius, -ell - 2, dr_outer)
    outer_gamma = admittivity[-1]
    outer_gamma_q = admittivity_q[:, -1]
    k[row, a_index(n_layers - 1)] = outer_gamma * ell * r_grad
    k[row, b_index(n_layers - 1)] = -outer_gamma * (ell + 1) * r_decay
    k_q[:, row, a_index(n_layers - 1)] = ell * (
        outer_gamma_q * r_grad + outer_gamma * dr_grad
    )
    k_q[:, row, b_index(n_layers - 1)] = -(ell + 1) * (
        outer_gamma_q * r_decay + outer_gamma * dr_decay
    )
    return k, k_q


class FrequencySphericalHarmonicModel:
    """Frequency-parameterised complex layered-sphere forward model."""

    def __init__(
        self,
        head: LayeredSphereHead,
        calibration: CalibrationSpec | None = None,
    ) -> None:
        self.head = head
        self.calibration = calibration or CalibrationSpec()
        # Resolve once so invalid/duplicate layer selections fail at construction.
        self.calibration.resolved_ratio_layers(head.n_layers)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.calibration.parameter_names(self.head)

    def assemble(
        self,
        frequency_hz: float,
        ell: int,
        m: int = 0,
        *,
        source_coefficients: ArrayLike = 1.0 + 0.0j,
        theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None = None,
    ) -> GlobalMatrixSystem:
        """Assemble ``K``, ``B``, ``H`` and analytic parameter derivatives."""

        ell = int(ell)
        m = int(m)
        if ell < 1:
            raise ValueError("ell must be at least one; the EEG gauge mode ell=0 is excluded")
        if abs(m) > ell:
            raise ValueError("m must satisfy abs(m) <= ell")

        names = self.parameter_names
        values = _coerce_theta(theta, names)
        name_to_index = {name: index for index, name in enumerate(names)}
        n_parameters = len(names)
        n_layers = self.head.n_layers

        admittivity = self.head.admittivity(frequency_hz).copy()
        admittivity_q = np.zeros((n_parameters, n_layers), dtype=np.complex128)
        for layer in self.calibration.resolved_ratio_layers(n_layers):
            parameter = f"log_admittivity_ratio_{self.head.layer_names[layer]}"
            q = name_to_index[parameter]
            admittivity[layer] *= np.exp(values[q])
            admittivity_q[q, layer] = admittivity[layer]

        log_scale = 0.0
        scale_index: int | None = None
        if self.calibration.include_log_radius_scale:
            scale_index = name_to_index["log_radius_scale"]
            log_scale = values[scale_index]
        radius_scale = float(np.exp(log_scale))
        length_scale = self.head.length_scale_m
        radii = np.asarray(self.head.radii_m) * radius_scale / length_scale
        radii_q = np.zeros((n_parameters, n_layers), dtype=np.float64)
        if scale_index is not None:
            radii_q[scale_index] = radii

        translation = np.zeros(3, dtype=np.float64)
        translation_indices: list[int] = []
        if self.calibration.include_electrode_translation:
            for axis, suffix in enumerate(("x", "y", "z")):
                q = name_to_index[f"electrode_translation_{suffix}_m"]
                translation[axis] = values[q]
                translation_indices.append(q)
        electrode_xyz = (
            np.asarray(self.head.electrode_xyz_m) * radius_scale + translation[None, :]
        ) / length_scale
        electrode_xyz_q = np.zeros(
            (n_parameters, electrode_xyz.shape[0], 3),
            dtype=np.float64,
        )
        if scale_index is not None:
            electrode_xyz_q[scale_index] = (
                np.asarray(self.head.electrode_xyz_m) * radius_scale / length_scale
            )
        for axis, q in enumerate(translation_indices):
            electrode_xyz_q[q, :, axis] = 1.0 / length_scale

        k, k_q = _assemble_k(
            ell,
            np.asarray(radii, dtype=np.float64),
            radii_q,
            admittivity,
            admittivity_q,
        )

        sources = np.asarray(source_coefficients, dtype=np.complex128)
        if sources.ndim == 0:
            sources = sources.reshape(1)
        if sources.ndim != 1 or sources.size == 0:
            raise ValueError("source_coefficients must be a scalar or non-empty vector")
        source_factor = radius_scale ** float(self.calibration.source_scale_power)
        b = np.zeros((2 * n_layers, sources.size), dtype=np.complex128)
        b[0] = sources * source_factor
        b_q = np.zeros((n_parameters, *b.shape), dtype=np.complex128)
        if scale_index is not None and self.calibration.source_scale_power != 0.0:
            b_q[scale_index, 0] = float(self.calibration.source_scale_power) * b[0]

        regular, regular_gradient = _solid_harmonic_and_gradient(
            electrode_xyz,
            ell=ell,
            m=m,
            radial_power=ell,
        )
        singular, singular_gradient = _solid_harmonic_and_gradient(
            electrode_xyz,
            ell=ell,
            m=m,
            radial_power=-ell - 1,
        )
        h = np.zeros((electrode_xyz.shape[0], 2 * n_layers), dtype=np.complex128)
        h[:, 2 * (n_layers - 1)] = regular
        h[:, 2 * (n_layers - 1) + 1] = singular
        h_q = np.zeros((n_parameters, *h.shape), dtype=np.complex128)
        for q in range(n_parameters):
            h_q[q, :, 2 * (n_layers - 1)] = np.einsum(
                "ij,ij->i", regular_gradient, electrode_xyz_q[q]
            )
            h_q[q, :, 2 * (n_layers - 1) + 1] = np.einsum(
                "ij,ij->i", singular_gradient, electrode_xyz_q[q]
            )

        return GlobalMatrixSystem(
            frequency_hz=float(frequency_hz),
            ell=ell,
            m=m,
            parameter_names=names,
            K=np.asarray(k, dtype=np.complex128),
            B=np.asarray(b, dtype=np.complex128),
            H=np.asarray(h, dtype=np.complex128),
            K_q=np.asarray(k_q, dtype=np.complex128),
            B_q=np.asarray(b_q, dtype=np.complex128),
            H_q=np.asarray(h_q, dtype=np.complex128),
            admittivity_s_per_m=np.asarray(admittivity, dtype=np.complex128),
            radii_dimensionless=np.asarray(radii, dtype=np.float64),
            electrode_xyz_dimensionless=np.asarray(electrode_xyz, dtype=np.float64),
        )

    def solve(
        self,
        frequency_hz: float,
        ell: int,
        m: int = 0,
        *,
        source_coefficients: ArrayLike = 1.0 + 0.0j,
        theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None = None,
    ) -> ForwardSolution:
        """Solve the forward model and its analytic Jacobian without an inverse.

        The derivative implements exactly

        ``D_q = H_q X + H K^-1 (B_q - K_q X)``.

        One LU factorisation of ``K`` is reused for ``X`` and all derivative
        right-hand sides.
        """

        system = self.assemble(
            frequency_hz,
            ell,
            m,
            source_coefficients=source_coefficients,
            theta=theta,
        )
        lu_and_piv = linalg.lu_factor(system.K, check_finite=False)
        x = linalg.lu_solve(lu_and_piv, system.B, check_finite=False)
        g = system.H @ x

        n_parameters = len(system.parameter_names)
        derivative = np.empty(
            (n_parameters, system.H.shape[0], system.B.shape[1]),
            dtype=np.complex128,
        )
        if n_parameters:
            rhs = system.B_q - np.einsum("qij,jk->qik", system.K_q, x)
            rhs_block = rhs.transpose(1, 0, 2).reshape(system.K.shape[0], -1)
            dx_block = linalg.lu_solve(lu_and_piv, rhs_block, check_finite=False)
            dx = dx_block.reshape(system.K.shape[0], n_parameters, -1).transpose(1, 0, 2)
            derivative = (
                np.einsum("qij,jk->qik", system.H_q, x)
                + np.einsum("ij,qjk->qik", system.H, dx)
            )

        residual = system.K @ x - system.B
        denominator = max(float(linalg.norm(system.B)), np.finfo(np.float64).tiny)
        relative_residual = float(linalg.norm(residual) / denominator)
        return ForwardSolution(
            system=system,
            X=np.asarray(x, dtype=np.complex128),
            G=np.asarray(g, dtype=np.complex128),
            D=np.asarray(derivative, dtype=np.complex128),
            relative_residual=relative_residual,
        )


def solve_frequency_forward(
    head: LayeredSphereHead,
    frequency_hz: float,
    ell: int,
    m: int = 0,
    *,
    source_coefficients: ArrayLike = 1.0 + 0.0j,
    theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None = None,
    calibration: CalibrationSpec | None = None,
) -> ForwardSolution:
    """Functional convenience wrapper around :class:`FrequencySphericalHarmonicModel`."""

    return FrequencySphericalHarmonicModel(head, calibration).solve(
        frequency_hz,
        ell,
        m,
        source_coefficients=source_coefficients,
        theta=theta,
    )


def build_oriented_leadfield_family(
    model: FrequencySphericalHarmonicModel,
    frequencies_hz: ArrayLike,
    source_xyz_m: ArrayLike,
    *,
    l_max: int = 4,
    theta: Mapping[str, float] | Sequence[float] | NDArray[np.float64] | None = None,
    reference_matrix: ArrayLike | None = None,
) -> LeadfieldFamily:
    """Assemble ``{G_f}`` for Cartesian dipoles at specified source points.

    A dipole is expanded in derivatives of the regular solid harmonics at the
    source location.  Each harmonic is propagated through the layered global
    matrix, retaining its full frequency-dependent complex transfer.  This is
    a genuine operator-family construction; no across-frequency matrix average
    is formed.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    sources_m = np.asarray(source_xyz_m, dtype=np.float64)
    if frequencies.size == 0 or np.any(frequencies < 0) or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be finite and nonnegative")
    if sources_m.ndim != 2 or sources_m.shape[1] != 3 or sources_m.shape[0] == 0:
        raise ValueError("source_xyz_m must have shape (n_sources, 3)")
    if np.any(np.linalg.norm(sources_m, axis=1) >= np.asarray(model.head.radii_m)[0]):
        raise ValueError("all sources must lie inside the innermost modeled boundary")
    if int(l_max) < 1:
        raise ValueError("l_max must be at least one")
    source_dimensionless = sources_m / model.head.length_scale_m
    n_electrodes = np.asarray(model.head.electrode_xyz_m).shape[0]
    n_coefficients = 3 * sources_m.shape[0]
    n_parameters = len(model.parameter_names)
    leadfields = np.zeros(
        (frequencies.size, n_electrodes, n_coefficients), dtype=np.complex128
    )
    jacobians = np.zeros(
        (frequencies.size, n_parameters, n_electrodes, n_coefficients),
        dtype=np.complex128,
    )
    maximum_residual = 0.0
    for frequency_index, frequency in enumerate(frequencies):
        for ell in range(1, int(l_max) + 1):
            for m in range(-ell, ell + 1):
                _, gradient = _solid_harmonic_and_gradient(
                    source_dimensionless,
                    ell=ell,
                    m=m,
                    radial_power=ell,
                )
                # Multipole expansion uses the conjugated source basis.
                coefficients = np.conjugate(gradient).reshape(-1)
                solution = model.solve(
                    float(frequency),
                    ell,
                    m,
                    source_coefficients=coefficients,
                    theta=theta,
                )
                leadfields[frequency_index] += solution.G
                jacobians[frequency_index] += solution.D
                maximum_residual = max(maximum_residual, solution.relative_residual)
    if reference_matrix is not None:
        transform = np.asarray(reference_matrix, dtype=np.float64)
        if transform.ndim != 2 or transform.shape[1] != n_electrodes:
            raise ValueError("reference_matrix has the wrong sensor dimension")
        leadfields = np.einsum("re,fep->frp", transform, leadfields)
        jacobians = np.einsum("re,fqep->fqrp", transform, jacobians)
    return LeadfieldFamily(
        frequencies_hz=frequencies,
        source_xyz_m=sources_m,
        leadfields=np.asarray(leadfields, dtype=np.complex128),
        jacobians=np.asarray(jacobians, dtype=np.complex128),
        parameter_names=model.parameter_names,
        l_max=int(l_max),
        relative_residual_max=float(maximum_residual),
    )


__all__ = [
    "EPSILON_0_F_PER_M",
    "CalibrationSpec",
    "ForwardSolution",
    "FrequencySphericalHarmonicModel",
    "GlobalMatrixSystem",
    "LeadfieldFamily",
    "LayeredSphereHead",
    "solve_frequency_forward",
    "build_oriented_leadfield_family",
]
