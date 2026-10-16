"""Production structured proper-complex posterior for V2.

The dense structured reference in :mod:`complex_sbl` is useful for numerical
validation, but it explicitly allocates the full latent prior and posterior.
This module keeps the same ``time -> frequency -> coefficient`` convention
and provides both an exact dense sensor-space backend for small reference
problems and a truly matrix-free iterative backend.  The latter applies the
separable sensor covariance through ``LinearOperator`` Hessian-vector products
and recovers marginal variances in bounded RHS chunks; neither a full
``(T F C) x (T F C)`` source covariance nor a full ``(T F E) x (T F E)``
sensor covariance is materialized.

At fixed hyperparameters the source prior is

``C_time (x) C_frequency (x) (D_gamma C_graph D_gamma (x) C_orientation)``.

All correlation factors are trace-normalized.  Optional shrinkage is applied
after normalization, removing scale exchange with the group ARD variances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg


Array = np.ndarray


def _hermitian(value: Array) -> Array:
    array = np.asarray(value, dtype=np.complex128)
    return 0.5 * (array + array.conj().T)


def _dense_square(value: object, *, name: str) -> Array:
    if hasattr(value, "toarray"):
        value = value.toarray()  # type: ignore[union-attr]
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    relative_skew = np.linalg.norm(array - array.conj().T, "fro") / max(
        np.linalg.norm(array, "fro"), np.finfo(float).tiny
    )
    if relative_skew > 1.0e-8:
        raise ValueError(f"{name} must be Hermitian")
    return _hermitian(array)


def _trace_normalize_and_shrink(
    value: object,
    *,
    name: str,
    shrinkage: float,
) -> Array:
    """Return a trace-``dimension`` Hermitian positive-definite covariance."""

    if not 0.0 <= float(shrinkage) < 1.0:
        raise ValueError("shrinkage must lie in [0, 1)")
    array = _dense_square(value, name=name)
    dimension = array.shape[0]
    mean_diagonal = float(np.trace(array).real / dimension)
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0.0:
        raise ValueError(f"{name} must have positive trace")
    normalized = array / mean_diagonal
    normalized = (1.0 - float(shrinkage)) * normalized + float(
        shrinkage
    ) * np.eye(dimension, dtype=np.complex128)
    normalized = _hermitian(normalized)
    eigenvalues = linalg.eigvalsh(normalized, check_finite=False)
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    if float(eigenvalues[0]) <= tolerance:
        raise ValueError(
            f"{name} is not positive definite; use positive shrinkage for "
            "a singular covariance/precision"
        )
    return normalized


def _cho_factor(value: Array) -> tuple[Array, bool]:
    """Match the dense reference's guarded sensor-space Cholesky."""

    array = _hermitian(value)
    scale = max(float(np.trace(array).real / max(1, array.shape[0])), 1.0)
    jitter = np.finfo(float).eps * scale
    for _ in range(8):
        try:
            return linalg.cho_factor(
                array + jitter * np.eye(array.shape[0]),
                lower=True,
                check_finite=False,
            )
        except linalg.LinAlgError:
            jitter *= 10.0
    raise linalg.LinAlgError("Hermitian sensor covariance is not positive definite")


def _source_graph_covariance(
    n_groups: int,
    *,
    covariance: object | None,
    precision: object | None,
    shrinkage: float,
) -> tuple[Array, str]:
    if covariance is not None and precision is not None:
        raise ValueError(
            "provide at most one of source_graph_covariance and "
            "source_graph_precision"
        )
    if covariance is None and precision is None:
        return np.eye(n_groups, dtype=np.complex128), "identity"
    if covariance is not None:
        graph = _trace_normalize_and_shrink(
            covariance,
            name="source_graph_covariance",
            shrinkage=shrinkage,
        )
        representation = "covariance"
    else:
        normalized_precision = _trace_normalize_and_shrink(
            precision,
            name="source_graph_precision",
            shrinkage=shrinkage,
        )
        factor = _cho_factor(normalized_precision)
        graph = linalg.cho_solve(
            factor,
            np.eye(n_groups, dtype=np.complex128),
            check_finite=False,
        )
        graph = _trace_normalize_and_shrink(
            graph,
            name="source_graph_covariance_from_precision",
            shrinkage=0.0,
        )
        representation = "precision_solve"
    if graph.shape != (n_groups, n_groups):
        raise ValueError("source graph factor has the wrong group dimension")
    return graph, representation


def _as_problem(
    leadfields: Iterable[Array],
    observations: Iterable[Array],
    gamma: Array,
    *,
    group_size: int,
) -> tuple[list[Array], list[Array], Array, int, int, int]:
    operators = [np.asarray(value, dtype=np.complex128) for value in leadfields]
    data = [np.asarray(value, dtype=np.complex128) for value in observations]
    if not operators or len(operators) != len(data):
        raise ValueError("leadfields and observations must be non-empty and aligned")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    n_coefficients = operators[0].shape[1] if operators[0].ndim == 2 else -1
    if n_coefficients < 1 or n_coefficients % group_size:
        raise ValueError("leadfield columns must contain complete source groups")
    n_time = data[0].shape[1] if data[0].ndim == 2 else -1
    for operator, values in zip(operators, data, strict=True):
        if operator.ndim != 2 or values.ndim != 2:
            raise ValueError("leadfields/observations must be 2-D arrays")
        if operator.shape[0] != values.shape[0]:
            raise ValueError("each observation must match its leadfield sensors")
        if operator.shape[1] != n_coefficients or values.shape[1] != n_time:
            raise ValueError("all frequencies must share coefficient/time dimensions")
        if not np.all(np.isfinite(operator)) or not np.all(np.isfinite(values)):
            raise ValueError("leadfields/observations must be finite")
    n_groups = n_coefficients // group_size
    ard = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if ard.shape != (n_groups,) or np.any(~np.isfinite(ard)) or np.any(ard < 0.0):
        raise ValueError("gamma must contain one finite non-negative value per group")
    if not np.any(ard > 0.0):
        raise ValueError("at least one ARD group variance must be positive")
    return operators, data, ard, len(operators), n_time, n_coefficients


def _sensor_layout(
    operators: list[Array], n_time: int
) -> tuple[list[list[slice]], int]:
    per_time = sum(operator.shape[0] for operator in operators)
    layout: list[list[slice]] = []
    for time in range(n_time):
        cursor = time * per_time
        row: list[slice] = []
        for operator in operators:
            row.append(slice(cursor, cursor + operator.shape[0]))
            cursor += operator.shape[0]
        layout.append(row)
    return layout, n_time * per_time


def _sensor_covariance(
    operators: list[Array],
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    noise0: float,
    effective_frequency_weight: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
) -> Array:
    n_frequency = len(operators)
    n_time = len(layout)
    cross = [
        [
            operators[left]
            @ spatial_covariance
            @ operators[right].conj().T
            for right in range(n_frequency)
        ]
        for left in range(n_frequency)
    ]
    covariance = np.zeros(
        (sensor_dimension, sensor_dimension), dtype=np.complex128
    )
    for left_time in range(n_time):
        for right_time in range(n_time):
            time_scale = c_time[left_time, right_time]
            for left_frequency in range(n_frequency):
                left_slice = layout[left_time][left_frequency]
                for right_frequency in range(n_frequency):
                    covariance[left_slice, layout[right_time][right_frequency]] = (
                        time_scale
                        * c_frequency[left_frequency, right_frequency]
                        * cross[left_frequency][right_frequency]
                    )
    for time in range(n_time):
        for frequency, operator in enumerate(operators):
            block = layout[time][frequency]
            covariance[block, block] += (
                noise0 / effective_frequency_weight[frequency]
            ) * np.eye(operator.shape[0], dtype=np.complex128)
    return _hermitian(covariance)


def _sensor_covariance_linear_operator(
    operators: list[Array],
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    noise0: float,
    effective_frequency_weight: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
) -> LinearOperator:
    """Return ``A C_x A^H + C_noise`` without forming its dense matrix.

    The latent tensor convention is ``time -> frequency -> coefficient``.
    Applying the separable prior is therefore a tensor contraction rather
    than multiplication by a ``(T F C) x (T F C)`` Kronecker product.
    """

    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = spatial_covariance.shape[0]
    noise_variance = noise0 / effective_frequency_weight

    def matvec(value: Array) -> Array:
        vector = np.asarray(value, dtype=np.complex128).reshape(sensor_dimension)
        adjoint = np.empty(
            (n_time, n_frequency, n_coefficients), dtype=np.complex128
        )
        for time, row in enumerate(layout):
            for frequency, block in enumerate(row):
                adjoint[time, frequency] = (
                    operators[frequency].conj().T @ vector[block]
                )
        prior_applied = np.einsum(
            "tu,fh,cd,uhd->tfc",
            c_time,
            c_frequency,
            spatial_covariance,
            adjoint,
            optimize=True,
        )
        result = np.empty(sensor_dimension, dtype=np.complex128)
        for time, row in enumerate(layout):
            for frequency, block in enumerate(row):
                result[block] = (
                    operators[frequency] @ prior_applied[time, frequency]
                    + noise_variance[frequency] * vector[block]
                )
        return result

    return LinearOperator(
        shape=(sensor_dimension, sensor_dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.dtype(np.complex128),
    )


def _sensor_covariance_diagonal(
    operators: list[Array],
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    noise0: float,
    effective_frequency_weight: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
) -> Array:
    """Exact diagonal used by the deterministic Jacobi preconditioner."""

    diagonal = np.empty(sensor_dimension, dtype=np.float64)
    noise_variance = noise0 / effective_frequency_weight
    spatial_diagonal = []
    for operator in operators:
        projected = operator @ spatial_covariance
        values = np.real(np.sum(projected * operator.conj(), axis=1))
        spatial_diagonal.append(np.maximum(values, 0.0))
    for time, row in enumerate(layout):
        for frequency, block in enumerate(row):
            diagonal[block] = (
                float(np.real(c_time[time, time]))
                * float(np.real(c_frequency[frequency, frequency]))
                * spatial_diagonal[frequency]
                + noise_variance[frequency]
            )
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise FloatingPointError("sensor covariance diagonal must be positive")
    return diagonal


def _jacobi_preconditioner(diagonal: Array) -> LinearOperator:
    inverse = 1.0 / np.asarray(diagonal, dtype=np.float64)

    def matvec(value: Array) -> Array:
        return inverse * np.asarray(value, dtype=np.complex128)

    return LinearOperator(
        shape=(inverse.size, inverse.size),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.dtype(np.complex128),
    )


def _cg_solve_audited(
    operator: LinearOperator,
    rhs: Array,
    *,
    preconditioner: LinearOperator,
    rtol: float,
    atol: float,
    maxiter: int,
    role: str,
    rhs_index: int | None,
) -> tuple[Array, dict[str, object]]:
    """Run deterministic CG and return a compact convergence audit record."""

    iterations = 0

    def callback(_: Array) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = cg(
        operator,
        np.asarray(rhs, dtype=np.complex128),
        M=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        callback=callback,
    )
    residual = np.asarray(rhs, dtype=np.complex128) - operator @ solution
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / max(rhs_norm, np.finfo(float).tiny)
    tolerance = float(atol + rtol * rhs_norm)
    record: dict[str, object] = {
        "role": role,
        "rhs_index": rhs_index,
        "iterations": int(iterations),
        "scipy_info": int(info),
        "rhs_norm": rhs_norm,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "requested_absolute_tolerance": tolerance,
        "converged": bool(info == 0),
    }
    if info != 0 or not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "iterative sensor solve did not converge "
            f"(role={role}, rhs_index={rhs_index}, info={info}, "
            f"relative_residual={relative_residual:.3e})"
        )
    return np.asarray(solution, dtype=np.complex128), record


def _stochastic_lanczos_logdet(
    operator: LinearOperator,
    *,
    n_probes: int,
    n_steps: int,
    seed: int,
) -> tuple[float, dict[str, object]]:
    """Fixed-seed stochastic Lanczos quadrature estimate of ``logdet``.

    Complex four-phase Rademacher probes make the trace estimator unbiased.
    Full Lanczos reorthogonalization keeps the deterministic small-problem
    reference stable.  The returned value remains a stochastic approximation
    of the trace unless all coordinate directions are explicitly probed.
    """

    dimension = int(operator.shape[0])
    if n_probes < 1:
        raise ValueError("slq_probes must be positive")
    if n_steps < 1:
        raise ValueError("slq_steps must be positive")
    steps = min(int(n_steps), dimension)
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    actual_steps: list[int] = []
    smallest_ritz = np.inf
    largest_ritz = 0.0
    phases = np.asarray([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex128)
    for _ in range(int(n_probes)):
        probe = phases[rng.integers(0, phases.size, size=dimension)]
        probe_norm = float(np.linalg.norm(probe))
        q = probe / probe_norm
        basis: list[Array] = []
        alphas: list[float] = []
        betas: list[float] = []
        for step in range(steps):
            basis.append(np.array(q, copy=True))
            work = operator @ q
            alpha = float(np.real(np.vdot(q, work)))
            work = work - alpha * q
            # Full reorthogonalization is deterministic and avoids ghost Ritz
            # values in the modest Lanczos depth used for objective auditing.
            for previous in basis:
                work = work - previous * np.vdot(previous, work)
            beta = float(np.linalg.norm(work))
            alphas.append(alpha)
            if step + 1 >= steps or beta <= 64.0 * np.finfo(float).eps:
                break
            betas.append(beta)
            q = work / beta
        tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
        if betas:
            off_diagonal = np.asarray(betas, dtype=np.float64)
            tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
        eigenvalues, eigenvectors = linalg.eigh(
            tridiagonal, check_finite=False
        )
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        floor = 128.0 * np.finfo(float).eps * scale
        if float(np.min(eigenvalues)) < -floor:
            raise FloatingPointError("SLQ encountered a negative Ritz value")
        clipped = np.maximum(eigenvalues, floor)
        weights = np.abs(eigenvectors[0, :]) ** 2
        estimates.append(
            probe_norm**2 * float(np.sum(weights * np.log(clipped)))
        )
        actual_steps.append(len(alphas))
        smallest_ritz = min(smallest_ritz, float(np.min(clipped)))
        largest_ritz = max(largest_ritz, float(np.max(clipped)))
    estimate = float(np.mean(estimates))
    standard_error = (
        float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
        if len(estimates) > 1
        else 0.0
    )
    audit: dict[str, object] = {
        "method": "fixed_seed_stochastic_lanczos_quadrature",
        "exact": False,
        "seed": int(seed),
        "probe_distribution": "complex_four_phase_rademacher",
        "probes": int(n_probes),
        "requested_steps": int(n_steps),
        "actual_steps": actual_steps,
        "probe_estimates": estimates,
        "standard_error": standard_error,
        "smallest_ritz_value": float(smallest_ritz),
        "largest_ritz_value": float(largest_ritz),
    }
    return estimate, audit


def _observation_vector(
    data: list[Array], layout: list[list[slice]], sensor_dimension: int
) -> Array:
    vector = np.empty(sensor_dimension, dtype=np.complex128)
    for time, row in enumerate(layout):
        for frequency, block in enumerate(row):
            vector[block] = data[frequency][:, time]
    return vector


def _posterior_mean_vector(
    operators: list[Array],
    solved_observations: Array,
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    layout: list[list[slice]],
) -> Array:
    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = spatial_covariance.shape[0]
    adjoint = np.empty(
        (n_time, n_frequency, n_coefficients), dtype=np.complex128
    )
    for time, row in enumerate(layout):
        for frequency, block in enumerate(row):
            adjoint[time, frequency] = (
                operators[frequency].conj().T @ solved_observations[block]
            )
    return np.einsum(
        "tu,fh,cd,uhd->tfc",
        c_time,
        c_frequency,
        spatial_covariance,
        adjoint,
        optimize=True,
    ).reshape(-1)


def _posterior_covariance_diagonal(
    operators: list[Array],
    factor: tuple[Array, bool],
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
    *,
    chunk_size: int,
) -> tuple[Array, int]:
    """Compute exact marginal variances without allocating latent covariance."""

    if chunk_size < 1:
        raise ValueError("covariance_diagonal_chunk_size must be positive")
    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = spatial_covariance.shape[0]
    latent_dimension = n_time * n_frequency * n_coefficients
    projected_spatial = [operator @ spatial_covariance for operator in operators]
    diagonal = np.empty(latent_dimension, dtype=np.float64)
    largest_chunk = 0
    for start in range(0, latent_dimension, chunk_size):
        stop = min(start + chunk_size, latent_dimension)
        indices = np.arange(start, stop, dtype=np.int64)
        time_index = indices // (n_frequency * n_coefficients)
        remainder = indices % (n_frequency * n_coefficients)
        frequency_index = remainder // n_coefficients
        coefficient_index = remainder % n_coefficients
        rhs = np.empty((sensor_dimension, stop - start), dtype=np.complex128)
        for time, row in enumerate(layout):
            for frequency, block in enumerate(row):
                scales = (
                    c_time[time, time_index]
                    * c_frequency[frequency, frequency_index]
                )
                rhs[block] = (
                    projected_spatial[frequency][:, coefficient_index]
                    * scales[None, :]
                )
        solved = linalg.cho_solve(factor, rhs, check_finite=False)
        correction = np.real(np.sum(rhs.conj() * solved, axis=0))
        prior_diagonal = np.real(
            c_time[time_index, time_index]
            * c_frequency[frequency_index, frequency_index]
            * spatial_covariance[coefficient_index, coefficient_index]
        )
        marginal = prior_diagonal - correction
        scale = max(1.0, float(np.max(np.abs(prior_diagonal))))
        if float(np.min(marginal)) < -1.0e-8 * scale:
            raise FloatingPointError("posterior marginal variance became negative")
        diagonal[start:stop] = np.maximum(marginal, 0.0)
        largest_chunk = max(largest_chunk, stop - start)
    return diagonal, largest_chunk


def _posterior_covariance_diagonal_iterative(
    operators: list[Array],
    sensor_operator: LinearOperator,
    preconditioner: LinearOperator,
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
    *,
    chunk_size: int,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> tuple[Array, int, list[dict[str, object]]]:
    """Compute marginal variances with block-built RHSs and CG solves.

    Each chunk allocates at most ``sensor_dimension x chunk_size``.  The
    sensor covariance is accessed exclusively through Hessian-vector
    products supplied by ``sensor_operator``.
    """

    if chunk_size < 1:
        raise ValueError("covariance_diagonal_chunk_size must be positive")
    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = spatial_covariance.shape[0]
    latent_dimension = n_time * n_frequency * n_coefficients
    projected_spatial = [operator @ spatial_covariance for operator in operators]
    diagonal = np.empty(latent_dimension, dtype=np.float64)
    audit: list[dict[str, object]] = []
    largest_chunk = 0
    for start in range(0, latent_dimension, chunk_size):
        stop = min(start + chunk_size, latent_dimension)
        indices = np.arange(start, stop, dtype=np.int64)
        time_index = indices // (n_frequency * n_coefficients)
        remainder = indices % (n_frequency * n_coefficients)
        frequency_index = remainder // n_coefficients
        coefficient_index = remainder % n_coefficients
        rhs = np.empty((sensor_dimension, stop - start), dtype=np.complex128)
        for time, row in enumerate(layout):
            for frequency, block in enumerate(row):
                scales = (
                    c_time[time, time_index]
                    * c_frequency[frequency, frequency_index]
                )
                rhs[block] = (
                    projected_spatial[frequency][:, coefficient_index]
                    * scales[None, :]
                )
        correction = np.empty(stop - start, dtype=np.float64)
        for local_index, latent_index in enumerate(indices):
            solved, record = _cg_solve_audited(
                sensor_operator,
                rhs[:, local_index],
                preconditioner=preconditioner,
                rtol=cg_rtol,
                atol=cg_atol,
                maxiter=cg_maxiter,
                role="posterior_marginal_diagonal",
                rhs_index=int(latent_index),
            )
            correction[local_index] = float(
                np.real(np.vdot(rhs[:, local_index], solved))
            )
            audit.append(record)
        prior_diagonal = np.real(
            c_time[time_index, time_index]
            * c_frequency[frequency_index, frequency_index]
            * spatial_covariance[coefficient_index, coefficient_index]
        )
        marginal = prior_diagonal - correction
        scale = max(1.0, float(np.max(np.abs(prior_diagonal))))
        if float(np.min(marginal)) < -1.0e-8 * scale:
            raise FloatingPointError("posterior marginal variance became negative")
        diagonal[start:stop] = np.maximum(marginal, 0.0)
        largest_chunk = max(largest_chunk, stop - start)
    return diagonal, largest_chunk, audit


def _latent_prior_matvec(
    value: Array,
    *,
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
) -> Array:
    n_time = c_time.shape[0]
    n_frequency = c_frequency.shape[0]
    n_coefficients = spatial_covariance.shape[0]
    tensor = np.asarray(value, dtype=np.complex128).reshape(
        n_time, n_frequency, n_coefficients
    )
    return np.einsum(
        "tu,fh,cd,uhd->tfc",
        c_time,
        c_frequency,
        spatial_covariance,
        tensor,
        optimize=True,
    ).reshape(-1)


def _latent_forward_vector(
    operators: list[Array],
    value: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
) -> Array:
    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = operators[0].shape[1]
    tensor = np.asarray(value, dtype=np.complex128).reshape(
        n_time, n_frequency, n_coefficients
    )
    result = np.empty(sensor_dimension, dtype=np.complex128)
    for time, row in enumerate(layout):
        for frequency, block in enumerate(row):
            result[block] = operators[frequency] @ tensor[time, frequency]
    return result


def _posterior_covariance_diagonal_hutchinson(
    operators: list[Array],
    sensor_operator: LinearOperator,
    preconditioner: LinearOperator,
    spatial_covariance: Array,
    c_time: Array,
    c_frequency: Array,
    layout: list[list[slice]],
    sensor_dimension: int,
    *,
    probes: int,
    seed: int,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> tuple[Array, list[dict[str, object]], dict[str, object]]:
    """Estimate the posterior diagonal using fixed-seed Hutchinson probes.

    For each latent-space probe ``z`` this evaluates
    ``conj(z) * (C - C A^H C_y^-1 A C) z``.  Thus the number of sensor-space
    solves is the probe count, independent of the latent dimension.
    """

    if probes < 2:
        raise ValueError("hutchinson_probes must be at least two")
    n_time = len(layout)
    n_frequency = len(operators)
    n_coefficients = spatial_covariance.shape[0]
    latent_dimension = n_time * n_frequency * n_coefficients
    prior_diagonal = np.real(
        np.einsum(
            "t,f,c->tfc",
            np.diag(c_time),
            np.diag(c_frequency),
            np.diag(spatial_covariance),
            optimize=True,
        )
    ).reshape(-1)
    if np.any(~np.isfinite(prior_diagonal)) or np.any(prior_diagonal < 0.0):
        raise FloatingPointError("latent prior diagonal must be finite non-negative")

    rng = np.random.default_rng(int(seed))
    phases = np.asarray([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex128)
    running_mean = np.zeros(latent_dimension, dtype=np.float64)
    running_m2 = np.zeros(latent_dimension, dtype=np.float64)
    maximum_imaginary_sample = 0.0
    iteration_log: list[dict[str, object]] = []
    for probe_index in range(int(probes)):
        probe = phases[rng.integers(0, phases.size, size=latent_dimension)]
        prior_applied = _latent_prior_matvec(
            probe,
            spatial_covariance=spatial_covariance,
            c_time=c_time,
            c_frequency=c_frequency,
        )
        sensor_rhs = _latent_forward_vector(
            operators, prior_applied, layout, sensor_dimension
        )
        solved, record = _cg_solve_audited(
            sensor_operator,
            sensor_rhs,
            preconditioner=preconditioner,
            rtol=cg_rtol,
            atol=cg_atol,
            maxiter=cg_maxiter,
            role="posterior_marginal_hutchinson",
            rhs_index=probe_index,
        )
        conditional = _posterior_mean_vector(
            operators,
            solved,
            spatial_covariance,
            c_time,
            c_frequency,
            layout,
        )
        sample_complex = probe.conj() * (prior_applied - conditional)
        maximum_imaginary_sample = max(
            maximum_imaginary_sample,
            float(np.max(np.abs(np.imag(sample_complex)))),
        )
        sample = np.real(sample_complex)
        count = probe_index + 1
        delta = sample - running_mean
        running_mean += delta / count
        running_m2 += delta * (sample - running_mean)
        iteration_log.append(record)

    standard_error = np.sqrt(
        np.maximum(running_m2, 0.0) / (probes - 1) / probes
    )
    scale = max(float(np.max(prior_diagonal)), np.finfo(float).tiny)
    clipping_tolerance = 1.0e-10 * scale
    negative_mask = running_mean < 0.0
    materially_negative = running_mean < -clipping_tolerance
    above_prior_mask = running_mean > prior_diagonal
    materially_above_prior = running_mean > prior_diagonal + clipping_tolerance
    clipped = np.clip(running_mean, 0.0, prior_diagonal)
    audit: dict[str, object] = {
        "method": "fixed_seed_complex_rademacher_hutchinson",
        "approximate": True,
        "seed": int(seed),
        "probes": int(probes),
        "probe_distribution": "complex_four_phase_rademacher",
        "solve_count": int(probes),
        "latent_dimension": latent_dimension,
        "standard_error": {
            "maximum": float(np.max(standard_error)),
            "median": float(np.median(standard_error)),
            "root_mean_square": float(
                np.sqrt(np.mean(np.square(standard_error)))
            ),
            "maximum_relative_to_prior_scale": float(
                np.max(standard_error) / scale
            ),
        },
        "raw_estimate_range": [
            float(np.min(running_mean)),
            float(np.max(running_mean)),
        ],
        "prior_diagonal_range": [
            float(np.min(prior_diagonal)),
            float(np.max(prior_diagonal)),
        ],
        "clipping": {
            "applied": bool(np.any(negative_mask) or np.any(above_prior_mask)),
            "tolerance": clipping_tolerance,
            "negative_count": int(np.count_nonzero(negative_mask)),
            "materially_negative_count": int(
                np.count_nonzero(materially_negative)
            ),
            "above_prior_count": int(np.count_nonzero(above_prior_mask)),
            "materially_above_prior_count": int(
                np.count_nonzero(materially_above_prior)
            ),
            "absolute_negative_mass": float(
                -np.sum(running_mean[negative_mask])
            ),
            "absolute_above_prior_excess": float(
                np.sum(
                    running_mean[above_prior_mask]
                    - prior_diagonal[above_prior_mask]
                )
            ),
        },
        "maximum_absolute_imaginary_probe_sample": (
            maximum_imaginary_sample
        ),
    }
    return clipped, iteration_log, audit


@dataclass(frozen=True)
class StructuredPosteriorResult:
    """Source moments without a full latent posterior covariance.

    ``mean`` and ``covariance_diagonal`` use the package convention
    ``(frequency, coefficient, time)``.  Flattening
    ``mean.transpose(2, 0, 1)`` recovers the dense reference vector order.
    """

    mean: Array
    covariance_diagonal: Array
    group_second_moment: Array
    gamma: Array
    noise: float
    objective: float
    active_groups: Array
    metadata: dict[str, object]


class MatrixFreeStructuredPosterior:
    """Fixed-hyperparameter structured posterior with dense/iterative solves.

    ``backend='dense'`` is the exact small-problem reference and allocates a
    square sensor covariance.  ``backend='iterative'`` uses only a
    :class:`~scipy.sparse.linalg.LinearOperator`; ``backend='auto'`` chooses
    dense only when the stacked sensor dimension does not exceed
    ``dense_sensor_dimension_threshold``.
    """

    def __init__(
        self,
        *,
        group_size: int = 3,
        shrinkage: float = 0.0,
        graph_shrinkage: float = 0.0,
        prune_threshold: float = 0.0,
        covariance_diagonal_chunk_size: int = 32,
        backend: Literal["auto", "dense", "iterative"] = "auto",
        dense_sensor_dimension_threshold: int = 256,
        cg_rtol: float = 1.0e-12,
        cg_atol: float = 0.0,
        cg_maxiter: int | None = None,
        slq_probes: int = 8,
        slq_steps: int = 32,
        slq_seed: int = 20260710,
        marginal_diagonal_mode: Literal[
            "auto", "exact", "hutchinson"
        ] = "auto",
        exact_marginal_latent_threshold: int = 512,
        hutchinson_probes: int = 24,
        hutchinson_seed: int = 20260711,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError("prune_threshold must lie in [0, 1]")
        if backend not in {"auto", "dense", "iterative"}:
            raise ValueError("backend must be 'auto', 'dense', or 'iterative'")
        if dense_sensor_dimension_threshold < 1:
            raise ValueError("dense_sensor_dimension_threshold must be positive")
        if not np.isfinite(cg_rtol) or cg_rtol <= 0.0:
            raise ValueError("cg_rtol must be finite and positive")
        if not np.isfinite(cg_atol) or cg_atol < 0.0:
            raise ValueError("cg_atol must be finite and non-negative")
        if cg_maxiter is not None and cg_maxiter < 1:
            raise ValueError("cg_maxiter must be positive when supplied")
        if slq_probes < 1 or slq_steps < 1:
            raise ValueError("slq_probes and slq_steps must be positive")
        if marginal_diagonal_mode not in {"auto", "exact", "hutchinson"}:
            raise ValueError(
                "marginal_diagonal_mode must be 'auto', 'exact', or "
                "'hutchinson'"
            )
        if exact_marginal_latent_threshold < 1:
            raise ValueError("exact_marginal_latent_threshold must be positive")
        if hutchinson_probes < 2:
            raise ValueError("hutchinson_probes must be at least two")
        self.group_size = int(group_size)
        self.shrinkage = float(shrinkage)
        self.graph_shrinkage = float(graph_shrinkage)
        self.prune_threshold = float(prune_threshold)
        self.covariance_diagonal_chunk_size = int(covariance_diagonal_chunk_size)
        self.backend = backend
        self.dense_sensor_dimension_threshold = int(
            dense_sensor_dimension_threshold
        )
        self.cg_rtol = float(cg_rtol)
        self.cg_atol = float(cg_atol)
        self.cg_maxiter = None if cg_maxiter is None else int(cg_maxiter)
        self.slq_probes = int(slq_probes)
        self.slq_steps = int(slq_steps)
        self.slq_seed = int(slq_seed)
        self.marginal_diagonal_mode = marginal_diagonal_mode
        self.exact_marginal_latent_threshold = int(
            exact_marginal_latent_threshold
        )
        self.hutchinson_probes = int(hutchinson_probes)
        self.hutchinson_seed = int(hutchinson_seed)

    def posterior(
        self,
        leadfields: Iterable[Array],
        observations: Iterable[Array],
        gamma: Array,
        noise0: float,
        *,
        frequency_weights: Array,
        c_time: Array,
        c_frequency: Array,
        orientation_covariance: Array | None = None,
        source_graph_covariance: object | None = None,
        source_graph_precision: object | None = None,
    ) -> StructuredPosteriorResult:
        operators, data, ard, n_frequency, n_time, n_coefficients = _as_problem(
            leadfields,
            observations,
            gamma,
            group_size=self.group_size,
        )
        if not np.isfinite(noise0) or noise0 <= 0.0:
            raise ValueError("noise0 must be finite and positive")
        weights = np.asarray(frequency_weights, dtype=np.float64).reshape(-1)
        if (
            weights.shape != (n_frequency,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("frequency_weights must be positive and aligned")
        weights = weights / weights.sum()
        effective = n_frequency * weights + 1.0e-12
        normalized_time = _trace_normalize_and_shrink(
            c_time,
            name="c_time",
            shrinkage=self.shrinkage,
        )
        normalized_frequency = _trace_normalize_and_shrink(
            c_frequency,
            name="c_frequency",
            shrinkage=self.shrinkage,
        )
        if normalized_time.shape != (n_time, n_time):
            raise ValueError("c_time has the wrong time dimension")
        if normalized_frequency.shape != (n_frequency, n_frequency):
            raise ValueError("c_frequency has the wrong frequency dimension")
        orientation = _trace_normalize_and_shrink(
            np.eye(self.group_size)
            if orientation_covariance is None
            else orientation_covariance,
            name="orientation_covariance",
            shrinkage=self.shrinkage,
        )
        if orientation.shape != (self.group_size, self.group_size):
            raise ValueError("orientation_covariance has the wrong dimension")
        n_groups = n_coefficients // self.group_size
        graph, graph_representation = _source_graph_covariance(
            n_groups,
            covariance=source_graph_covariance,
            precision=source_graph_precision,
            shrinkage=self.graph_shrinkage,
        )

        relative_cutoff = self.prune_threshold * float(np.max(ard))
        active_mask = (ard > 0.0) & (ard >= relative_cutoff)
        active_groups = np.flatnonzero(active_mask)
        active_coefficients = np.concatenate(
            [
                np.arange(
                    group * self.group_size,
                    (group + 1) * self.group_size,
                    dtype=np.int64,
                )
                for group in active_groups
            ]
        )
        active_operators = [operator[:, active_coefficients] for operator in operators]
        active_graph = _trace_normalize_and_shrink(
            graph[np.ix_(active_groups, active_groups)],
            name="active_source_graph_covariance",
            shrinkage=0.0,
        )
        root_gamma = np.sqrt(ard[active_groups])
        group_covariance = (
            root_gamma[:, None] * active_graph * root_gamma[None, :]
        )
        spatial_covariance = np.kron(group_covariance, orientation)
        active_coefficient_count = active_coefficients.size
        active_latent_dimension = (
            n_time * n_frequency * active_coefficient_count
        )

        layout, sensor_dimension = _sensor_layout(active_operators, n_time)
        observations_vector = _observation_vector(data, layout, sensor_dimension)
        selected_backend = (
            self.backend
            if self.backend != "auto"
            else (
                "dense"
                if sensor_dimension <= self.dense_sensor_dimension_threshold
                else "iterative"
            )
        )
        if selected_backend == "dense":
            selected_marginal_mode = "exact"
        elif self.marginal_diagonal_mode == "auto":
            selected_marginal_mode = (
                "exact"
                if active_latent_dimension
                <= self.exact_marginal_latent_threshold
                else "hutchinson"
            )
        else:
            selected_marginal_mode = self.marginal_diagonal_mode
        iteration_log: list[dict[str, object]] = []
        logdet_audit: dict[str, object]
        marginal_diagonal_audit: dict[str, object]
        if selected_backend == "dense":
            sensor_covariance = _sensor_covariance(
                active_operators,
                spatial_covariance,
                normalized_time,
                normalized_frequency,
                float(noise0),
                effective,
                layout,
                sensor_dimension,
            )
            factor = _cho_factor(sensor_covariance)
            solved_observations = linalg.cho_solve(
                factor, observations_vector, check_finite=False
            )
            active_diagonal_vector, largest_chunk = _posterior_covariance_diagonal(
                active_operators,
                factor,
                spatial_covariance,
                normalized_time,
                normalized_frequency,
                layout,
                sensor_dimension,
                chunk_size=self.covariance_diagonal_chunk_size,
            )
            marginal_diagonal_audit = {
                "method": "exact_dense_cholesky_chunked_rhs",
                "approximate": False,
                "solve_count": active_latent_dimension,
            }
            factor_diagonal = np.real(np.diag(factor[0]))
            logdet = 2.0 * float(
                np.log(
                    np.maximum(factor_diagonal, np.finfo(float).tiny)
                ).sum()
            )
            logdet_audit = {
                "method": "dense_cholesky",
                "exact": True,
            }
            solver_name = "sensor_space_cholesky_chunked_marginals"
        else:
            sensor_operator = _sensor_covariance_linear_operator(
                active_operators,
                spatial_covariance,
                normalized_time,
                normalized_frequency,
                float(noise0),
                effective,
                layout,
                sensor_dimension,
            )
            sensor_diagonal = _sensor_covariance_diagonal(
                active_operators,
                spatial_covariance,
                normalized_time,
                normalized_frequency,
                float(noise0),
                effective,
                layout,
                sensor_dimension,
            )
            preconditioner = _jacobi_preconditioner(sensor_diagonal)
            cg_maxiter = (
                self.cg_maxiter
                if self.cg_maxiter is not None
                else max(100, 10 * sensor_dimension)
            )
            solved_observations, mean_record = _cg_solve_audited(
                sensor_operator,
                observations_vector,
                preconditioner=preconditioner,
                rtol=self.cg_rtol,
                atol=self.cg_atol,
                maxiter=cg_maxiter,
                role="posterior_mean",
                rhs_index=None,
            )
            iteration_log.append(mean_record)
            if selected_marginal_mode == "exact":
                (
                    active_diagonal_vector,
                    largest_chunk,
                    diagonal_iteration_log,
                ) = _posterior_covariance_diagonal_iterative(
                    active_operators,
                    sensor_operator,
                    preconditioner,
                    spatial_covariance,
                    normalized_time,
                    normalized_frequency,
                    layout,
                    sensor_dimension,
                    chunk_size=self.covariance_diagonal_chunk_size,
                    cg_rtol=self.cg_rtol,
                    cg_atol=self.cg_atol,
                    cg_maxiter=cg_maxiter,
                )
                marginal_diagonal_audit = {
                    "method": "exact_linear_operator_cg_per_latent",
                    "approximate": False,
                    "solve_count": active_latent_dimension,
                }
                marginal_solver_name = "exact_chunked_marginals"
            else:
                (
                    active_diagonal_vector,
                    diagonal_iteration_log,
                    marginal_diagonal_audit,
                ) = _posterior_covariance_diagonal_hutchinson(
                    active_operators,
                    sensor_operator,
                    preconditioner,
                    spatial_covariance,
                    normalized_time,
                    normalized_frequency,
                    layout,
                    sensor_dimension,
                    probes=self.hutchinson_probes,
                    seed=self.hutchinson_seed,
                    cg_rtol=self.cg_rtol,
                    cg_atol=self.cg_atol,
                    cg_maxiter=cg_maxiter,
                )
                largest_chunk = 1
                marginal_solver_name = "hutchinson_marginals"
            iteration_log.extend(diagonal_iteration_log)
            logdet, logdet_audit = _stochastic_lanczos_logdet(
                sensor_operator,
                n_probes=self.slq_probes,
                n_steps=self.slq_steps,
                seed=self.slq_seed,
            )
            solver_name = (
                "linear_operator_cg_"
                f"{marginal_solver_name}_slq_logdet"
            )
        active_mean_vector = _posterior_mean_vector(
            active_operators,
            solved_observations,
            spatial_covariance,
            normalized_time,
            normalized_frequency,
            layout,
        )

        active_mean = active_mean_vector.reshape(
            n_time, n_frequency, active_coefficient_count
        )
        active_diagonal = active_diagonal_vector.reshape(
            n_time, n_frequency, active_coefficient_count
        )
        mean_tfc = np.zeros(
            (n_time, n_frequency, n_coefficients), dtype=np.complex128
        )
        diagonal_tfc = np.zeros(
            (n_time, n_frequency, n_coefficients), dtype=np.float64
        )
        mean_tfc[:, :, active_coefficients] = active_mean
        diagonal_tfc[:, :, active_coefficients] = active_diagonal
        mean = mean_tfc.transpose(1, 2, 0)
        covariance_diagonal = diagonal_tfc.transpose(1, 2, 0)
        coefficient_second = np.abs(mean) ** 2 + covariance_diagonal
        group_second = coefficient_second.reshape(
            n_frequency, n_groups, self.group_size, n_time
        ).mean(axis=(0, 2, 3))

        quadratic = float(np.real(np.vdot(observations_vector, solved_observations)))
        objective = logdet + quadratic
        latent_dimension = n_time * n_frequency * n_coefficients
        allocated_dense_shapes: dict[str, list[int]] = {
            "coefficient_spatial_covariance": [
                active_coefficient_count,
                active_coefficient_count,
            ],
            "largest_diagonal_rhs": [sensor_dimension, largest_chunk],
        }
        if selected_backend == "dense":
            allocated_dense_shapes["sensor_covariance"] = [
                sensor_dimension,
                sensor_dimension,
            ]
        else:
            allocated_dense_shapes["sensor_covariance_diagonal"] = [
                sensor_dimension
            ]
        convergence_audit: dict[str, object] = {
            "enabled": selected_backend == "iterative",
            "all_converged": all(
                bool(record["converged"]) for record in iteration_log
            ),
            "solve_count": len(iteration_log),
            "maximum_iterations": max(
                (int(record["iterations"]) for record in iteration_log),
                default=0,
            ),
            "maximum_relative_residual": max(
                (
                    float(record["relative_residual"])
                    for record in iteration_log
                ),
                default=0.0,
            ),
            "cg_rtol": self.cg_rtol,
            "cg_atol": self.cg_atol,
            "cg_maxiter": (
                None
                if selected_backend == "dense"
                else (
                    self.cg_maxiter
                    if self.cg_maxiter is not None
                    else max(100, 10 * sensor_dimension)
                )
            ),
            "preconditioner": (
                None if selected_backend == "dense" else "exact_diagonal_jacobi"
            ),
            "iteration_log": iteration_log,
        }
        return StructuredPosteriorResult(
            mean=mean,
            covariance_diagonal=covariance_diagonal,
            group_second_moment=np.asarray(group_second, dtype=np.float64),
            gamma=np.array(ard, copy=True),
            noise=float(noise0),
            objective=float(objective),
            active_groups=active_groups,
            metadata={
                "mode": "v2_structured_fixed_hyperparameters",
                "solver": solver_name,
                "backend_requested": self.backend,
                "backend_selected": selected_backend,
                "dense_sensor_dimension_threshold": (
                    self.dense_sensor_dimension_threshold
                ),
                "sensor_dimension": sensor_dimension,
                "dtype": "complex128",
                "vector_order": "time_frequency_coefficient",
                "mean_shape": list(mean.shape),
                "covariance_diagonal_shape": list(covariance_diagonal.shape),
                "latent_dimension": latent_dimension,
                "active_latent_dimension": active_latent_dimension,
                "full_source_prior_allocated": False,
                "full_posterior_covariance_allocated": False,
                "full_sensor_covariance_allocated": selected_backend == "dense",
                "sensor_covariance_representation": (
                    "dense_complex128"
                    if selected_backend == "dense"
                    else "scipy_linear_operator_complex128"
                ),
                "sensor_covariance_hvp": selected_backend == "iterative",
                "allocated_dense_shapes": allocated_dense_shapes,
                "iterative_convergence_audit": convergence_audit,
                "marginal_diagonal_mode_requested": (
                    self.marginal_diagonal_mode
                ),
                "marginal_diagonal_mode_selected": selected_marginal_mode,
                "exact_marginal_latent_threshold": (
                    self.exact_marginal_latent_threshold
                ),
                "marginal_diagonal_approximate": bool(
                    marginal_diagonal_audit["approximate"]
                ),
                "marginal_diagonal_audit": marginal_diagonal_audit,
                "objective_logdet_audit": logdet_audit,
                "objective_approximate": not bool(logdet_audit["exact"]),
                "objective_components": {
                    "logdet": float(logdet),
                    "quadratic": quadratic,
                },
                "kernel_trace_normalized": True,
                "kernel_shrinkage": self.shrinkage,
                "graph_shrinkage": self.graph_shrinkage,
                "source_graph_representation": graph_representation,
                "prune_threshold_relative_to_max_gamma": self.prune_threshold,
            },
        )

    fit_fixed = posterior


def matrix_free_structured_posterior(
    leadfields: Iterable[Array],
    observations: Iterable[Array],
    gamma: Array,
    noise0: float,
    *,
    frequency_weights: Array,
    c_time: Array,
    c_frequency: Array,
    orientation_covariance: Array | None = None,
    source_graph_covariance: object | None = None,
    source_graph_precision: object | None = None,
    group_size: int = 3,
    shrinkage: float = 0.0,
    graph_shrinkage: float = 0.0,
    prune_threshold: float = 0.0,
    covariance_diagonal_chunk_size: int = 32,
    backend: Literal["auto", "dense", "iterative"] = "auto",
    dense_sensor_dimension_threshold: int = 256,
    cg_rtol: float = 1.0e-12,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    slq_probes: int = 8,
    slq_steps: int = 32,
    slq_seed: int = 20260710,
    marginal_diagonal_mode: Literal[
        "auto", "exact", "hutchinson"
    ] = "auto",
    exact_marginal_latent_threshold: int = 512,
    hutchinson_probes: int = 24,
    hutchinson_seed: int = 20260711,
) -> StructuredPosteriorResult:
    """Functional entry point for the fixed-hyperparameter production engine."""

    engine = MatrixFreeStructuredPosterior(
        group_size=group_size,
        shrinkage=shrinkage,
        graph_shrinkage=graph_shrinkage,
        prune_threshold=prune_threshold,
        covariance_diagonal_chunk_size=covariance_diagonal_chunk_size,
        backend=backend,
        dense_sensor_dimension_threshold=dense_sensor_dimension_threshold,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
        slq_probes=slq_probes,
        slq_steps=slq_steps,
        slq_seed=slq_seed,
        marginal_diagonal_mode=marginal_diagonal_mode,
        exact_marginal_latent_threshold=exact_marginal_latent_threshold,
        hutchinson_probes=hutchinson_probes,
        hutchinson_seed=hutchinson_seed,
    )
    return engine.posterior(
        leadfields,
        observations,
        gamma,
        noise0,
        frequency_weights=frequency_weights,
        c_time=c_time,
        c_frequency=c_frequency,
        orientation_covariance=orientation_covariance,
        source_graph_covariance=source_graph_covariance,
        source_graph_precision=source_graph_precision,
    )


__all__ = [
    "MatrixFreeStructuredPosterior",
    "StructuredPosteriorResult",
    "matrix_free_structured_posterior",
]
