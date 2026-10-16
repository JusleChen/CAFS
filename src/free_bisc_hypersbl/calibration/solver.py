"""Trust-region unrolled bilevel solver for affine lead-field linearizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from free_bisc_hypersbl.freesigm.stability import project_to_ess

from .bilevel import ElectrodeSplit, implicit_hypergradient


@dataclass
class BilevelCalibrationResult:
    theta: np.ndarray
    frequency_weights: np.ndarray
    gamma: np.ndarray
    noise: float
    outer_objective: list[float]
    inner_objective: list[float]
    accepted_step_sizes: list[float]
    implicit_unrolled_relative_error: float
    metadata: dict[str, object]


def calibrate_affine_leadfield_family(
    leadfields: np.ndarray,
    jacobians: np.ndarray,
    observations: np.ndarray,
    split: ElectrodeSplit,
    physical_prior_weights: np.ndarray,
    *,
    group_size: int = 3,
    minimum_ess: float | None = None,
    outer_iterations: int = 5,
    inner_steps: int = 20,
    inner_step_size: float = 0.15,
    outer_step_size: float = 0.08,
    theta_trust_radius: float = 0.25,
    theta_scales: np.ndarray | None = None,
    theta_prior_weight: float = 0.05,
    kl_weight: float = 0.2,
    smoothness_weight: float = 0.02,
    entropy_weight: float = 0.01,
    damping: float = 1e-5,
) -> BilevelCalibrationResult:
    """Calibrate physical parameters and Free-SIGM weights without labels.

    ``G(theta)`` is the current analytic trust-region linearization
    ``G0 + sum_q theta_q D_q``.  The lower Type-II problem is unrolled through
    group ARD/noise log-parameters.  The upper objective is held-out electrode
    predictive NLL plus physical and Free-SIGM regularizers.  A backtracking
    trust-region step is used at every outer iteration.  At the final small
    problem, the unrolled gradient is checked against an implicit Hessian
    linear solve; production-sized callers may disable this check externally.
    """

    import torch

    g_np = np.asarray(leadfields, dtype=np.complex128)
    d_np = np.asarray(jacobians, dtype=np.complex128)
    y_np = np.asarray(observations, dtype=np.complex128)
    if g_np.ndim != 3 or d_np.ndim != 4 or y_np.ndim != 3:
        raise ValueError("expected G(F,E,P), D(F,Q,E,P), Y(F,E,T)")
    n_frequency, n_electrodes, n_coefficients = g_np.shape
    if d_np.shape[0] != n_frequency or d_np.shape[2:] != (n_electrodes, n_coefficients):
        raise ValueError("Jacobian shape mismatch")
    if y_np.shape[:2] != (n_frequency, n_electrodes):
        raise ValueError("observation shape mismatch")
    if n_coefficients % group_size:
        raise ValueError("source coefficients are not group-aligned")
    if set(split.inner.tolist()) & set(split.validation.tolist()):
        raise ValueError("electrode split leaks validation sensors")
    n_groups = n_coefficients // group_size
    n_physical = d_np.shape[1]
    scales_np = (
        np.ones(n_physical, dtype=np.float64)
        if theta_scales is None
        else np.asarray(theta_scales, dtype=np.float64)
    )
    if scales_np.shape != (n_physical,):
        raise ValueError("theta_scales must have one entry per physical parameter")
    if np.any(~np.isfinite(scales_np)) or np.any(scales_np <= 0.0):
        raise ValueError("theta_scales must be finite and strictly positive")
    prior_np = np.asarray(physical_prior_weights, dtype=float)
    prior_np = prior_np / prior_np.sum()
    if prior_np.shape != (n_frequency,) or np.any(prior_np <= 0):
        raise ValueError("physical prior weights must be positive")
    minimum_ess = float(minimum_ess or max(1.0, 0.45 * n_frequency))

    device = torch.device("cpu")
    real_dtype, complex_dtype = torch.float64, torch.complex128
    g0 = torch.as_tensor(g_np, dtype=complex_dtype, device=device)
    derivative = torch.as_tensor(d_np, dtype=complex_dtype, device=device)
    data = torch.as_tensor(y_np, dtype=complex_dtype, device=device)
    inner_index = torch.as_tensor(split.inner, dtype=torch.long, device=device)
    validation_index = torch.as_tensor(split.validation, dtype=torch.long, device=device)
    prior = torch.as_tensor(prior_np, dtype=real_dtype, device=device)
    physical_scales = torch.as_tensor(scales_np, dtype=real_dtype, device=device)
    identity_inner = torch.eye(len(split.inner), dtype=complex_dtype, device=device)
    identity_validation = torch.eye(len(split.validation), dtype=complex_dtype, device=device)
    jitter = 1e-9

    def unpack_u(u):
        eta = u[:n_physical]
        theta = eta * physical_scales
        logits = u[n_physical:]
        pi = torch.softmax(logits, dim=0)
        effective = n_frequency * pi + 1e-12
        family = g0 + torch.einsum("q,fqep->fep", theta.to(complex_dtype), derivative)
        return theta, pi, effective, family

    def unpack_z(z):
        gamma = torch.exp(torch.clamp(z[:n_groups], -25.0, 25.0))
        noise = torch.exp(torch.clamp(z[-1], -25.0, 25.0))
        return gamma, noise

    def marginal_nll(operator, target, gamma, noise_f, identity):
        gamma_diag = torch.repeat_interleave(gamma, group_size).to(complex_dtype)
        covariance = (operator * gamma_diag.unsqueeze(0)) @ operator.conj().T
        covariance = 0.5 * (covariance + covariance.conj().T) + (noise_f + jitter) * identity
        chol = torch.linalg.cholesky(covariance)
        logdet = 2.0 * torch.log(torch.clamp(torch.diagonal(chol).real, min=1e-30)).sum()
        solved = torch.cholesky_solve(target, chol)
        quadratic = torch.sum(torch.conj(target) * solved).real
        return target.shape[1] * logdet + quadratic

    def inner_loss(z, u):
        gamma, noise = unpack_z(z)
        _, _, effective, family = unpack_u(u)
        total = torch.zeros((), dtype=real_dtype, device=device)
        for f in range(n_frequency):
            operator = family[f].index_select(0, inner_index)
            target = data[f].index_select(0, inner_index)
            total = total + marginal_nll(operator, target, gamma, noise / effective[f], identity_inner)
        return total

    def posterior(operator, target, gamma, noise_f, identity):
        gamma_diag = torch.repeat_interleave(gamma, group_size).to(complex_dtype)
        covariance_y = (operator * gamma_diag.unsqueeze(0)) @ operator.conj().T
        covariance_y = 0.5 * (covariance_y + covariance_y.conj().T) + (noise_f + jitter) * identity
        chol = torch.linalg.cholesky(covariance_y)
        solved_y = torch.cholesky_solve(target, chol)
        solved_a = torch.cholesky_solve(operator, chol)
        mean = gamma_diag.unsqueeze(1) * (operator.conj().T @ solved_y)
        sigma = torch.diag(gamma_diag) - (
            gamma_diag.unsqueeze(1) * (operator.conj().T @ solved_a)
        ) * gamma_diag.unsqueeze(0)
        return mean, 0.5 * (sigma + sigma.conj().T)

    def outer_loss(z, u):
        gamma, noise = unpack_z(z)
        theta, pi, effective, family = unpack_u(u)
        eta = u[:n_physical]
        total = torch.zeros((), dtype=real_dtype, device=device)
        for f in range(n_frequency):
            a_inner = family[f].index_select(0, inner_index)
            y_inner = data[f].index_select(0, inner_index)
            mean, sigma = posterior(a_inner, y_inner, gamma, noise / effective[f], identity_inner)
            a_validation = family[f].index_select(0, validation_index)
            y_validation = data[f].index_select(0, validation_index)
            predicted = a_validation @ mean
            covariance = (noise / effective[f] + jitter) * identity_validation
            covariance = covariance + a_validation @ sigma @ a_validation.conj().T
            covariance = 0.5 * (covariance + covariance.conj().T)
            chol = torch.linalg.cholesky(covariance)
            residual = y_validation - predicted
            logdet = 2.0 * torch.log(torch.clamp(torch.diagonal(chol).real, min=1e-30)).sum()
            quadratic = torch.sum(torch.conj(residual) * torch.cholesky_solve(residual, chol)).real
            total = total + residual.shape[1] * logdet + quadratic
        kl = torch.sum(pi * (torch.log(pi + 1e-12) - torch.log(prior + 1e-12)))
        smoothness = torch.sum(torch.diff(pi) ** 2) if n_frequency > 1 else torch.zeros_like(kl)
        entropy = -torch.sum(pi * torch.log(pi + 1e-12))
        return (
            total
            + theta_prior_weight * torch.sum(eta**2)
            + kl_weight * kl
            + smoothness_weight * smoothness
            - entropy_weight * entropy
        )

    initial_power = max(float(np.mean(np.abs(y_np[:, split.inner]) ** 2)), 1e-8)
    z = torch.cat(
        [
            torch.full((n_groups,), np.log(initial_power), dtype=real_dtype),
            torch.tensor([np.log(max(0.1 * initial_power, 1e-8))], dtype=real_dtype),
        ]
    )
    u = torch.cat(
        [
            torch.zeros(n_physical, dtype=real_dtype),
            torch.log(prior),
        ]
    )

    def solve_lower(start, current_u, *, create_graph):
        current = start
        history: list[float] = []
        for _ in range(inner_steps):
            current = current.requires_grad_(True)
            loss = inner_loss(current, current_u)
            gradient = torch.autograd.grad(loss, current, create_graph=create_graph)[0]
            scaled = gradient / (1.0 + torch.abs(gradient))
            updated = current - inner_step_size * scaled
            current = updated if create_graph else updated.detach()
            history.append(float(loss.detach()))
        return current, history

    def project_u(candidate):
        array = candidate.detach().cpu().numpy().copy()
        array[:n_physical] = np.clip(array[:n_physical], -theta_trust_radius, theta_trust_radius)
        raw = np.exp(array[n_physical:] - np.max(array[n_physical:]))
        projected = project_to_ess(raw / raw.sum(), minimum_ess)
        array[n_physical:] = np.log(projected)
        return torch.as_tensor(array, dtype=real_dtype)

    outer_history: list[float] = []
    inner_history: list[float] = []
    accepted_steps: list[float] = []
    for _ in range(outer_iterations):
        u_graph = u.detach().clone().requires_grad_(True)
        z_graph, lower_values = solve_lower(z.detach(), u_graph, create_graph=True)
        loss = outer_loss(z_graph, u_graph)
        gradient = torch.autograd.grad(loss, u_graph)[0]
        outer_history.append(float(loss.detach()))
        inner_history.extend(lower_values)
        direction = gradient / (1.0 + torch.abs(gradient))
        accepted = False
        step = outer_step_size
        while step >= outer_step_size * 2.0**-8:
            candidate_u = project_u(u_graph.detach() - step * direction.detach())
            candidate_z, _ = solve_lower(z.detach(), candidate_u, create_graph=False)
            candidate_loss = float(outer_loss(candidate_z, candidate_u).detach())
            if candidate_loss <= float(loss.detach()) + 1e-9:
                u, z = candidate_u.detach(), candidate_z.detach()
                accepted_steps.append(step)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            accepted_steps.append(0.0)
            z = z_graph.detach()
            break

    u_final = u.detach().clone().requires_grad_(True)
    z_final, _ = solve_lower(z.detach(), u_final, create_graph=True)
    final_outer = outer_loss(z_final, u_final)
    unrolled_gradient = torch.autograd.grad(final_outer, u_final, retain_graph=True)[0]
    try:
        implicit_gradient = implicit_hypergradient(
            inner_loss, outer_loss, z_final.detach(), u_final.detach(), damping=damping
        )
        relative_error = float(
            torch.linalg.norm(implicit_gradient - unrolled_gradient).detach()
            / torch.clamp(torch.linalg.norm(unrolled_gradient).detach(), min=1e-12)
        )
    except (RuntimeError, torch.linalg.LinAlgError):
        relative_error = float("nan")
    gamma_final, noise_final = unpack_z(z_final.detach())
    theta_final, pi_final, _, _ = unpack_u(u_final.detach())
    eta_final = u_final.detach()[:n_physical]
    return BilevelCalibrationResult(
        theta=theta_final.cpu().numpy(),
        frequency_weights=pi_final.cpu().numpy(),
        gamma=gamma_final.cpu().numpy(),
        noise=float(noise_final),
        outer_objective=outer_history + [float(final_outer.detach())],
        inner_objective=inner_history,
        accepted_step_sizes=accepted_steps,
        implicit_unrolled_relative_error=relative_error,
        metadata={
            "gradient": "unrolled total derivative; implicit Hessian solve audit",
            "trust_radius": theta_trust_radius,
            "theta_coordinate_system": "theta = theta_scales * eta",
            "theta_scales": scales_np.tolist(),
            "normalized_eta": eta_final.cpu().numpy().tolist(),
            "minimum_ess": minimum_ess,
            "lower_uses_validation_electrodes": False,
            "dtype": "complex128",
        },
    )
