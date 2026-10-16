"""Cross-predictive bilevel calibration and implicit hypergradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg

from free_bisc_hypersbl.inverse.complex_sbl import complex_gaussian_posterior


@dataclass(frozen=True)
class ElectrodeSplit:
    inner: np.ndarray
    validation: np.ndarray


def rotating_electrode_splits(
    n_electrodes: int,
    *,
    holdout_fraction: float = 0.15,
    n_splits: int = 5,
    seed: int = 0,
) -> list[ElectrodeSplit]:
    if n_electrodes < 4 or not 0.0 < holdout_fraction < 0.5:
        raise ValueError("invalid electrode count or holdout fraction")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_electrodes)
    holdout = max(1, int(round(holdout_fraction * n_electrodes)))
    splits: list[ElectrodeSplit] = []
    all_indices = np.arange(n_electrodes)
    for split in range(n_splits):
        start = (split * holdout) % n_electrodes
        positions = np.arange(start, start + holdout) % n_electrodes
        validation = np.sort(order[positions])
        inner = np.setdiff1d(all_indices, validation, assume_unique=True)
        splits.append(ElectrodeSplit(inner=inner, validation=validation))
    return splits


def inner_posterior_from_split(
    forward: np.ndarray,
    observations: np.ndarray,
    split: ElectrodeSplit,
    gamma: np.ndarray,
    noise: float,
    *,
    group_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the lower problem using only ``E_in``."""

    a = np.asarray(forward, dtype=np.complex128)
    y = np.asarray(observations, dtype=np.complex128)
    mu, sigma, _ = complex_gaussian_posterior(
        a[split.inner], y[split.inner], gamma, noise, group_size=group_size
    )
    return mu, sigma


def heldout_predictive_nll(
    forward: np.ndarray,
    observations: np.ndarray,
    split: ElectrodeSplit,
    gamma: np.ndarray,
    noise: float,
    *,
    group_size: int = 1,
) -> float:
    a = np.asarray(forward, dtype=np.complex128)
    y = np.asarray(observations, dtype=np.complex128)
    if y.ndim == 1:
        y = y[:, None]
    mu, sigma = inner_posterior_from_split(a, y, split, gamma, noise, group_size=group_size)
    a_val = a[split.validation]
    residual = y[split.validation] - a_val @ mu
    covariance = noise * np.eye(len(split.validation)) + a_val @ sigma @ a_val.conj().T
    covariance = (covariance + covariance.conj().T) * 0.5
    factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
    logdet = 2.0 * np.log(np.maximum(np.real(np.diag(factor[0])), np.finfo(float).tiny)).sum()
    quadratic = np.real(np.vdot(residual, linalg.cho_solve(factor, residual, check_finite=False)))
    return float(y.shape[1] * logdet + quadratic)


def mean_cross_predictive_nll(
    forward: np.ndarray,
    observations: np.ndarray,
    splits: list[ElectrodeSplit],
    gamma: np.ndarray,
    noise: float,
    *,
    group_size: int = 1,
) -> float:
    return float(
        np.mean(
            [
                heldout_predictive_nll(
                    forward, observations, split, gamma, noise, group_size=group_size
                )
                for split in splits
            ]
        )
    )


def implicit_hypergradient(
    inner_loss: Callable,
    outer_loss: Callable,
    z_star,
    hyperparameters,
    *,
    damping: float = 1e-6,
):
    """Implicit total derivative using an HVP-equivalent linear solve.

    The implementation is intentionally small-problem oriented and is used to
    verify the large-problem truncated/unrolled implementation.  It requires
    PyTorch tensors and does not form a Hessian inverse.
    """

    import torch

    z = z_star.detach().clone().requires_grad_(True)
    u = hyperparameters.detach().clone().requires_grad_(True)

    def inner_z(current_z):
        return inner_loss(current_z, u)

    hzz = torch.autograd.functional.hessian(inner_z, z, create_graph=True)
    grad_z_inner = torch.autograd.grad(inner_loss(z, u), z, create_graph=True)[0]
    rows = []
    for component in grad_z_inner.reshape(-1):
        mixed = torch.autograd.grad(component, u, retain_graph=True, create_graph=True)[0]
        rows.append(mixed.reshape(-1))
    hzu = torch.stack(rows, dim=0)
    loss_out = outer_loss(z, u)
    grad_z_out, direct_u = torch.autograd.grad(loss_out, (z, u), retain_graph=True)
    identity = torch.eye(hzz.shape[0], dtype=hzz.dtype, device=hzz.device)
    vector = torch.linalg.solve((hzz + damping * identity).T, grad_z_out.reshape(-1))
    total = direct_u.reshape(-1) - hzu.T @ vector
    return total.reshape_as(u)


def unrolled_hypergradient(
    inner_loss: Callable,
    outer_loss: Callable,
    z0,
    hyperparameters,
    *,
    steps: int = 100,
    step_size: float = 0.1,
):
    import torch

    u = hyperparameters.detach().clone().requires_grad_(True)
    z = z0.detach().clone().requires_grad_(True)
    for _ in range(steps):
        gradient = torch.autograd.grad(inner_loss(z, u), z, create_graph=True)[0]
        z = z - step_size * gradient
    return torch.autograd.grad(outer_loss(z, u), u)[0]


def finite_difference_total_gradient(
    solve_inner: Callable[[np.ndarray], np.ndarray],
    outer_loss: Callable[[np.ndarray, np.ndarray], float],
    hyperparameters: np.ndarray,
    *,
    step: float = 1e-5,
) -> np.ndarray:
    u = np.asarray(hyperparameters, dtype=float)
    gradient = np.empty_like(u)
    for index in range(u.size):
        plus, minus = u.copy(), u.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            outer_loss(solve_inner(plus), plus) - outer_loss(solve_inner(minus), minus)
        ) / (2.0 * step)
    return gradient

