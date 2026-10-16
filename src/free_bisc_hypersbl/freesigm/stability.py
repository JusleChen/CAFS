"""Physical stability prior and non-collapsing Free-SIGM weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class StabilityProfile:
    condition_number: np.ndarray
    spectral_gap: np.ndarray
    subspace_similarity: np.ndarray
    score: np.ndarray
    prior_weights: np.ndarray
    effective_sample_size: float


def effective_sample_size(weights: np.ndarray) -> float:
    pi = np.asarray(weights, dtype=float)
    if np.any(pi < 0) or not np.isfinite(pi).all() or pi.sum() <= 0:
        raise ValueError("weights must be finite and nonnegative")
    pi = pi / pi.sum()
    return float(1.0 / np.sum(pi**2))


def project_to_ess(weights: np.ndarray, minimum_ess: float, *, iterations: int = 80) -> np.ndarray:
    """Project along the shortest uniform-mixture path satisfying an ESS floor."""

    pi = np.asarray(weights, dtype=float)
    if np.any(pi <= 0) or not np.isfinite(pi).all():
        raise ValueError("Free-SIGM weights must be strictly positive")
    pi = pi / pi.sum()
    n = pi.size
    if not 1.0 <= minimum_ess <= n:
        raise ValueError("minimum_ess must lie in [1, number of frequencies]")
    if effective_sample_size(pi) >= minimum_ess:
        return pi
    uniform = np.full(n, 1.0 / n)
    low, high = 0.0, 1.0
    for _ in range(iterations):
        mix = 0.5 * (low + high)
        candidate = (1.0 - mix) * pi + mix * uniform
        if effective_sample_size(candidate) >= minimum_ess:
            high = mix
        else:
            low = mix
    projected = (1.0 - high) * pi + high * uniform
    projected = np.maximum(projected, np.finfo(float).tiny)
    return projected / projected.sum()


def _softmax(score: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(score, dtype=float) / temperature
    return np.exp(z - logsumexp(z))


def stability_profile(
    leadfields: Iterable[np.ndarray],
    *,
    signal_rank: int = 3,
    alpha_condition: float = 1.0,
    alpha_gap: float = 1.0,
    alpha_subspace: float = 1.0,
    temperature: float = 1.0,
    minimum_ess: float | None = None,
    epsilon: float = 1e-12,
) -> StabilityProfile:
    """Compute condition/gap/subspace stability and the physical weight prior."""

    matrices = [np.asarray(g, dtype=np.complex128) for g in leadfields]
    if not matrices:
        raise ValueError("at least one leadfield is required")
    singular_values: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    condition, gap = [], []
    for g in matrices:
        u, s, _ = np.linalg.svd(g, full_matrices=False)
        singular_values.append(s)
        rank = min(signal_rank, u.shape[1])
        bases.append(u[:, :rank])
        floor = max(epsilon * float(s[0]), epsilon)
        condition.append(float(s[0] / max(float(s[-1]), floor)))
        if len(s) > rank:
            gap.append(float(max(s[rank - 1] - s[rank], 0.0) / max(s[0], floor)))
        else:
            gap.append(float(s[rank - 1] / max(s[0], floor)))
    similarity = np.ones(len(matrices), dtype=float)
    for index in range(1, len(matrices)):
        left, right = bases[index - 1], bases[index]
        rank = min(left.shape[1], right.shape[1])
        overlap = left[:, :rank].conj().T @ right[:, :rank]
        similarity[index] = float(np.linalg.norm(overlap, "fro") ** 2 / max(rank, 1))
    if len(similarity) > 1:
        similarity[0] = similarity[1]
    condition_arr = np.asarray(condition)
    gap_arr = np.asarray(gap)
    score = (
        -alpha_condition * np.log(np.maximum(condition_arr, epsilon))
        + alpha_gap * np.log(np.maximum(gap_arr, epsilon))
        + alpha_subspace * similarity
    )
    prior = _softmax(score, temperature)
    if minimum_ess is not None:
        prior = project_to_ess(prior, minimum_ess)
    return StabilityProfile(
        condition_number=condition_arr,
        spectral_gap=gap_arr,
        subspace_similarity=similarity,
        score=score,
        prior_weights=prior,
        effective_sample_size=effective_sample_size(prior),
    )


def effective_likelihood_weights(
    logits: np.ndarray,
    *,
    minimum_ess: float,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized probabilities ``pi`` and weights ``w_f = F*pi_f + eps``."""

    pi = project_to_ess(_softmax(np.asarray(logits, dtype=float), temperature), minimum_ess)
    return pi, pi.size * pi + epsilon


def free_sigm_regularizer(
    weights: np.ndarray,
    prior: np.ndarray,
    *,
    information_matrix: np.ndarray | None = None,
    kl_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    entropy_weight: float = 0.05,
    information_weight: float = 0.0,
    epsilon: float = 1e-9,
) -> float:
    pi = np.asarray(weights, dtype=float)
    p0 = np.asarray(prior, dtype=float)
    pi, p0 = pi / pi.sum(), p0 / p0.sum()
    kl = float(np.sum(pi * (np.log(pi + epsilon) - np.log(p0 + epsilon))))
    smoothness = float(np.sum(np.diff(pi) ** 2))
    entropy = float(-np.sum(pi * np.log(pi + epsilon)))
    log_information = 0.0
    if information_matrix is not None:
        matrix = np.asarray(information_matrix, dtype=float)
        sign, logdet = np.linalg.slogdet(matrix + epsilon * np.eye(matrix.shape[0]))
        log_information = float(logdet) if sign > 0 else -np.inf
    return (
        kl_weight * kl
        + smoothness_weight * smoothness
        - entropy_weight * entropy
        - information_weight * log_information
    )

