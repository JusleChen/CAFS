"""Auditable complex-valued EEG inverse baselines.

Every method in this module performs its stated numerical optimisation or
subspace calculation.  No deterministic score table or result surrogate is
used.  Inputs and source estimates are promoted to ``complex128`` and spatial
scores aggregate the three dipole orientations when ``group_size=3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .complex_sbl import ComplexGroupSBL


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InverseResult:
    """Common result contract for classical and candidate inverse methods."""

    method: str
    source_estimate: ComplexArray
    source_score: FloatArray
    runtime_seconds: float
    metadata: dict[str, object]

    @property
    def estimate(self) -> ComplexArray:
        return self.source_estimate

    @property
    def score(self) -> FloatArray:
        return self.source_score

    @property
    def runtime(self) -> float:
        return self.runtime_seconds


def _snapshots(observations: ArrayLike) -> ComplexArray:
    y = np.asarray(observations, dtype=np.complex128)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError("observations must be finite with shape (sensors, snapshots)")
    return y


def _problem(
    forward: ArrayLike,
    observations: ArrayLike,
    group_size: int,
) -> tuple[ComplexArray, ComplexArray, int]:
    a = np.asarray(forward, dtype=np.complex128)
    y = _snapshots(observations)
    group_size = int(group_size)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError("forward must be a finite two-dimensional matrix")
    if a.shape[0] != y.shape[0]:
        raise ValueError("forward and observations have incompatible sensor counts")
    if group_size < 1 or a.shape[1] % group_size:
        raise ValueError("forward columns must be divisible by a positive group_size")
    return a, y, group_size


def _source_scores(source_estimate: ArrayLike, group_size: int) -> FloatArray:
    estimate = np.asarray(source_estimate, dtype=np.complex128)
    if estimate.ndim == 2:
        estimate = estimate[None, ...]
    if estimate.ndim != 3 or estimate.shape[1] % group_size:
        raise ValueError("source estimates must have shape (sources, time) or (frequency, sources, time)")
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


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return np.asarray(0.5 * (matrix + matrix.conj().T), dtype=np.complex128)


def _hermitian_solve(matrix: ComplexArray, rhs: ComplexArray) -> ComplexArray:
    """Jittered Cholesky solve; never forms a matrix inverse."""

    matrix = _hermitian(np.asarray(matrix, dtype=np.complex128))
    rhs = np.asarray(rhs, dtype=np.complex128)
    scale = max(float(np.trace(matrix).real / max(1, matrix.shape[0])), 1.0)
    jitter = np.finfo(np.float64).eps * scale
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    for _ in range(10):
        try:
            factor = linalg.cho_factor(
                matrix + jitter * identity,
                lower=True,
                check_finite=False,
            )
            return np.asarray(
                linalg.cho_solve(factor, rhs, check_finite=False),
                dtype=np.complex128,
            )
        except linalg.LinAlgError:
            jitter *= 10.0
    raise linalg.LinAlgError("regularized Hermitian system could not be factorized")


def _regularization_scale(a: ComplexArray, relative_regularization: float) -> float:
    relative = float(relative_regularization)
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("regularization must be finite and positive")
    return max(
        relative * float(np.linalg.norm(a, "fro") ** 2 / max(1, a.shape[0])),
        np.finfo(np.float64).eps,
    )


def _dual_mne(
    a: ComplexArray,
    y: ComplexArray,
    regularization: float,
) -> tuple[ComplexArray, ComplexArray, float]:
    ridge = _regularization_scale(a, regularization)
    covariance = a @ a.conj().T + ridge * np.eye(a.shape[0], dtype=np.complex128)
    solved_y = _hermitian_solve(covariance, y)
    solved_a = _hermitian_solve(covariance, a)
    kernel = solved_a.conj().T
    estimate = a.conj().T @ solved_y
    return (
        np.asarray(estimate, dtype=np.complex128),
        np.asarray(kernel, dtype=np.complex128),
        ridge,
    )


def minimum_norm(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    regularization: float = 1.0e-2,
) -> InverseResult:
    """Classical Tikhonov minimum-norm estimate in sensor-space dual form."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    estimate, _, ridge = _dual_mne(a, y, regularization)
    return InverseResult(
        method="MNE",
        source_estimate=estimate,
        source_score=_source_scores(estimate, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={"dtype": "complex128", "ridge": ridge, "solver": "dual Cholesky"},
    )


def dspm(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    regularization: float = 1.0e-2,
) -> InverseResult:
    """Noise-normalized dynamic statistical parametric mapping."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    estimate, kernel, ridge = _dual_mne(a, y, regularization)
    noise_std = np.sqrt(np.maximum(np.sum(np.abs(kernel) ** 2, axis=1), 1.0e-15))
    normalized = estimate / noise_std[:, None]
    return InverseResult(
        method="dSPM",
        source_estimate=np.asarray(normalized, dtype=np.complex128),
        source_score=_source_scores(normalized, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={"dtype": "complex128", "ridge": ridge, "normalization": "kernel noise gain"},
    )


def sloreta(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    regularization: float = 1.0e-2,
) -> InverseResult:
    """Standardized LORETA using the diagonal resolution variance."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    estimate, kernel, ridge = _dual_mne(a, y, regularization)
    resolution_diagonal = np.maximum(np.real(np.einsum("ij,ji->i", kernel, a)), 1.0e-15)
    standardized = estimate / np.sqrt(resolution_diagonal)[:, None]
    return InverseResult(
        method="sLORETA",
        source_estimate=np.asarray(standardized, dtype=np.complex128),
        source_score=_source_scores(standardized, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={"dtype": "complex128", "ridge": ridge, "normalization": "resolution diagonal"},
    )


def eloreta(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    regularization: float = 1.0e-2,
    max_iter: int = 80,
    tolerance: float = 1.0e-7,
) -> InverseResult:
    """Iterative block-weighted exact-LORETA fixed-point solver.

    A scalar weight is shared by the three orientations in each source group.
    The weights are repeatedly updated from block resolution leverage until the
    eLORETA fixed point is reached.
    """

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    n_groups = a.shape[1] // group_size
    group_prior = np.ones(n_groups, dtype=np.float64)
    ridge = _regularization_scale(a, regularization)
    history: list[float] = []

    for _ in range(int(max_iter)):
        prior = np.repeat(group_prior, group_size)
        covariance = (a * prior[None, :]) @ a.conj().T
        covariance += ridge * np.eye(a.shape[0], dtype=np.complex128)
        solved_a = _hermitian_solve(covariance, a)
        leverage = np.empty(n_groups, dtype=np.float64)
        for group in range(n_groups):
            sl = slice(group * group_size, (group + 1) * group_size)
            leverage[group] = max(
                float(np.trace(a[:, sl].conj().T @ solved_a[:, sl]).real / group_size),
                1.0e-15,
            )
        proposed = 1.0 / np.sqrt(leverage)
        proposed /= max(float(np.median(proposed)), 1.0e-15)
        relative_change = float(
            np.max(np.abs(proposed - group_prior) / np.maximum(group_prior, 1.0e-15))
        )
        history.append(relative_change)
        group_prior = np.sqrt(group_prior * proposed)
        if relative_change < tolerance:
            break

    prior = np.repeat(group_prior, group_size)
    covariance = (a * prior[None, :]) @ a.conj().T
    covariance += ridge * np.eye(a.shape[0], dtype=np.complex128)
    estimate = prior[:, None] * (a.conj().T @ _hermitian_solve(covariance, y))
    return InverseResult(
        method="eLORETA",
        source_estimate=np.asarray(estimate, dtype=np.complex128),
        source_score=_source_scores(estimate, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={
            "dtype": "complex128",
            "ridge": ridge,
            "iterations": len(history),
            "fixed_point_change": history,
            "group_weights": group_prior,
        },
    )


def lcmv(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    regularization: float = 5.0e-2,
) -> InverseResult:
    """Complex linearly constrained minimum-variance vector beamformer."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    covariance = y @ y.conj().T / max(1, y.shape[1])
    loading = max(
        float(regularization) * float(np.trace(covariance).real / max(1, a.shape[0])),
        1.0e-12,
    )
    loaded = covariance + loading * np.eye(a.shape[0], dtype=np.complex128)
    c_inv_a = _hermitian_solve(loaded, a)
    estimate = np.zeros((a.shape[1], y.shape[1]), dtype=np.complex128)
    for group in range(a.shape[1] // group_size):
        sl = slice(group * group_size, (group + 1) * group_size)
        gram = _hermitian(a[:, sl].conj().T @ c_inv_a[:, sl])
        group_rhs = c_inv_a[:, sl].conj().T @ y
        estimate[sl] = _hermitian_solve(gram, group_rhs)
    return InverseResult(
        method="LCMV",
        source_estimate=estimate,
        source_score=_source_scores(estimate, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={"dtype": "complex128", "diagonal_loading": loading, "unit_gain": True},
    )


def _signal_subspace(y: ComplexArray, rank: int) -> ComplexArray:
    rank = max(1, min(int(rank), min(y.shape)))
    left, _, _ = linalg.svd(y, full_matrices=False, check_finite=False)
    return np.asarray(left[:, :rank], dtype=np.complex128)


def _orthonormal_basis(matrix: ComplexArray, tolerance: float = 1.0e-10) -> ComplexArray:
    left, singular, _ = linalg.svd(matrix, full_matrices=False, check_finite=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.complex128)
    keep = singular > tolerance * singular[0]
    return np.asarray(left[:, keep], dtype=np.complex128)


def _music_group_scores(a: ComplexArray, signal_basis: ComplexArray, group_size: int) -> FloatArray:
    projector = signal_basis @ signal_basis.conj().T
    scores = np.zeros(a.shape[1] // group_size, dtype=np.float64)
    for group in range(scores.size):
        sl = slice(group * group_size, (group + 1) * group_size)
        basis = _orthonormal_basis(a[:, sl])
        if basis.shape[1]:
            scores[group] = float(
                np.trace(basis.conj().T @ projector @ basis).real / basis.shape[1]
            )
    return scores


def _selected_least_squares(
    a: ComplexArray,
    y: ComplexArray,
    groups: Sequence[int],
    group_size: int,
) -> tuple[ComplexArray, float]:
    estimate = np.zeros((a.shape[1], y.shape[1]), dtype=np.complex128)
    columns = [
        coefficient
        for group in groups
        for coefficient in range(group * group_size, (group + 1) * group_size)
    ]
    if columns:
        coefficients, _, _, _ = linalg.lstsq(
            a[:, columns],
            y,
            cond=1.0e-10,
            check_finite=False,
        )
        estimate[columns] = coefficients
    residual = float(np.linalg.norm(y - a @ estimate, "fro") ** 2)
    return estimate, residual


def trace_music(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    n_sources: int = 1,
    signal_rank: int | None = None,
) -> InverseResult:
    """Trace-MUSIC group scan followed by joint least-squares amplitudes."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    n_groups = a.shape[1] // group_size
    n_sources = max(1, min(int(n_sources), n_groups))
    basis = _signal_subspace(y, signal_rank or n_sources)
    scores = _music_group_scores(a, basis, group_size)
    selected = np.argsort(scores)[-n_sources:][::-1]
    estimate, residual = _selected_least_squares(a, y, selected.tolist(), group_size)
    return InverseResult(
        method="Trace-MUSIC",
        source_estimate=estimate,
        source_score=scores,
        runtime_seconds=perf_counter() - start,
        metadata={
            "dtype": "complex128",
            "selected_groups": selected,
            "signal_rank": basis.shape[1],
            "residual": residual,
        },
    )


def rap_music(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    n_sources: int = 1,
) -> InverseResult:
    """Recursive-application-and-projection MUSIC with residual refitting."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    n_groups = a.shape[1] // group_size
    n_sources = max(1, min(int(n_sources), n_groups))
    selected: list[int] = []
    aggregate_scores = np.zeros(n_groups, dtype=np.float64)
    residual = y.copy()
    residual_history = [float(np.linalg.norm(residual, "fro") ** 2)]
    estimate = np.zeros((a.shape[1], y.shape[1]), dtype=np.complex128)

    for iteration in range(n_sources):
        basis = _signal_subspace(residual, max(1, n_sources - iteration))
        scores = _music_group_scores(a, basis, group_size)
        if selected:
            scores[selected] = -np.inf
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        aggregate_scores[chosen] = max(float(scores[chosen]), 0.0)
        estimate, residual_energy = _selected_least_squares(a, y, selected, group_size)
        residual = y - a @ estimate
        residual_history.append(residual_energy)

    return InverseResult(
        method="RAP-MUSIC",
        source_estimate=estimate,
        source_score=aggregate_scores,
        runtime_seconds=perf_counter() - start,
        metadata={
            "dtype": "complex128",
            "selected_groups": np.asarray(selected, dtype=int),
            "residual_history": residual_history,
        },
    )


def champagne_sbl(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    max_iter: int = 80,
    tolerance: float = 1.0e-6,
    prune_threshold: float = 0.0,
    noise0: float | None = None,
) -> InverseResult:
    """Wrapper around the proper-complex Champagne/group-SBL EM solver."""

    start = perf_counter()
    a, y, group_size = _problem(forward, observations, group_size)
    fitted = ComplexGroupSBL(
        group_size=group_size,
        max_iter=max_iter,
        tol=tolerance,
        prune_threshold=prune_threshold,
    ).fit(a, y, noise0=noise0)
    return InverseResult(
        method="Champagne/SBL",
        source_estimate=np.asarray(fitted.mean, dtype=np.complex128),
        source_score=np.asarray(fitted.gamma, dtype=np.float64),
        runtime_seconds=perf_counter() - start,
        metadata={
            **fitted.metadata,
            "objective": fitted.objective,
            "noise": fitted.noise,
            "iterations": fitted.iterations,
            "converged": fitted.converged,
            "active_groups": fitted.active_groups,
        },
    )


def _frequency_family(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    group_size: int,
) -> tuple[list[ComplexArray], list[ComplexArray], int]:
    operators = [np.asarray(operator, dtype=np.complex128) for operator in leadfields]
    data = [_snapshots(value) for value in observations]
    if not operators or len(operators) != len(data):
        raise ValueError("leadfields and observations must be non-empty and aligned")
    checked = [_problem(operator, value, group_size) for operator, value in zip(operators, data)]
    n_coefficients = checked[0][0].shape[1]
    snapshots = checked[0][1].shape[1]
    if any(item[0].shape[1] != n_coefficients or item[1].shape[1] != snapshots for item in checked):
        raise ValueError("all frequencies must share coefficient and snapshot dimensions")
    return [item[0] for item in checked], [item[1] for item in checked], checked[0][2]


def _frequency_weights(weights: ArrayLike | None, n_frequencies: int) -> FloatArray:
    if weights is None:
        out = np.full(n_frequencies, 1.0 / n_frequencies, dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64)
        if out.shape != (n_frequencies,) or np.any(out <= 0.0) or not np.all(np.isfinite(out)):
            raise ValueError("frequency weights must be finite, positive, and aligned")
        out = out / out.sum()
    return out


def _weighted_stack(
    leadfields: list[ComplexArray],
    observations: list[ComplexArray],
    weights: FloatArray,
) -> tuple[ComplexArray, ComplexArray]:
    scale = np.sqrt(weights)
    stacked_a = np.vstack([factor * a for factor, a in zip(scale, leadfields)])
    stacked_y = np.vstack([factor * y for factor, y in zip(scale, observations)])
    return np.asarray(stacked_a, dtype=np.complex128), np.asarray(stacked_y, dtype=np.complex128)


def sigm_tikhonov(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    *,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
    regularization: float = 1.0e-2,
) -> InverseResult:
    """SIGM-weighted operator-family Tikhonov inverse."""

    start = perf_counter()
    operators, data, group_size = _frequency_family(leadfields, observations, group_size)
    weights = _frequency_weights(frequency_weights, len(operators))
    stacked_a, stacked_y = _weighted_stack(operators, data, weights)
    estimate, _, ridge = _dual_mne(stacked_a, stacked_y, regularization)
    return InverseResult(
        method="SIGM+Tikhonov",
        source_estimate=estimate,
        source_score=_source_scores(estimate, group_size),
        runtime_seconds=perf_counter() - start,
        metadata={"dtype": "complex128", "frequency_weights": weights, "ridge": ridge},
    )


def sigm_music(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    *,
    frequency_weights: ArrayLike | None = None,
    group_size: int = 3,
    n_sources: int = 1,
) -> InverseResult:
    """SIGM-weighted operator-family Trace-MUSIC inverse."""

    start = perf_counter()
    operators, data, group_size = _frequency_family(leadfields, observations, group_size)
    weights = _frequency_weights(frequency_weights, len(operators))
    stacked_a, stacked_y = _weighted_stack(operators, data, weights)
    result = trace_music(
        stacked_a,
        stacked_y,
        group_size=group_size,
        n_sources=n_sources,
    )
    return InverseResult(
        method="SIGM+MUSIC",
        source_estimate=result.source_estimate,
        source_score=result.source_score,
        runtime_seconds=perf_counter() - start,
        metadata={**result.metadata, "frequency_weights": weights},
    )


def free_sigm_inverse(
    leadfields: Iterable[ArrayLike],
    observations: Iterable[ArrayLike],
    *,
    free_sigm_weights: ArrayLike,
    original_inverse: str = "tikhonov",
    group_size: int = 3,
    regularization: float = 1.0e-2,
    n_sources: int = 1,
) -> InverseResult:
    """Apply learned Free-SIGM weights to an unchanged original inverse."""

    name = original_inverse.lower().replace("-", "_")
    if name in {"tikhonov", "mne"}:
        result = sigm_tikhonov(
            leadfields,
            observations,
            frequency_weights=free_sigm_weights,
            group_size=group_size,
            regularization=regularization,
        )
    elif name in {"music", "trace_music"}:
        result = sigm_music(
            leadfields,
            observations,
            frequency_weights=free_sigm_weights,
            group_size=group_size,
            n_sources=n_sources,
        )
    else:
        raise ValueError("original_inverse must be tikhonov/MNE or MUSIC")
    return InverseResult(
        method=f"Free-SIGM+{original_inverse}",
        source_estimate=result.source_estimate,
        source_score=result.source_score,
        runtime_seconds=result.runtime_seconds,
        metadata={**result.metadata, "weighting": "Free-SIGM", "original_inverse": original_inverse},
    )


MNE = minimum_norm
mne = minimum_norm
dSPM = dspm
sLORETA = sloreta
eLORETA = eloreta
LCMV = lcmv
standard_champagne = champagne_sbl


def run_inverse_baseline(
    method: str,
    forward: ArrayLike,
    observations: ArrayLike,
    **kwargs: object,
) -> InverseResult:
    """Dispatch a single-frequency baseline through the common contract."""

    normalized = method.lower().replace("-", "").replace("_", "")
    methods: dict[str, Callable[..., InverseResult]] = {
        "mne": minimum_norm,
        "dspm": dspm,
        "sloreta": sloreta,
        "eloreta": eloreta,
        "lcmv": lcmv,
        "tracemusic": trace_music,
        "rapmusic": rap_music,
        "champagne": champagne_sbl,
        "sbl": champagne_sbl,
    }
    if normalized not in methods:
        raise ValueError(f"unknown inverse baseline: {method}")
    return methods[normalized](forward, observations, **kwargs)


__all__ = [
    "InverseResult",
    "LCMV",
    "MNE",
    "mne",
    "dSPM",
    "eLORETA",
    "sLORETA",
    "champagne_sbl",
    "dspm",
    "eloreta",
    "free_sigm_inverse",
    "lcmv",
    "minimum_norm",
    "rap_music",
    "run_inverse_baseline",
    "sigm_music",
    "sigm_tikhonov",
    "sloreta",
    "standard_champagne",
    "trace_music",
]
