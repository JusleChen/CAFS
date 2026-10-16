"""Public V8 HQSL-HyperSBL solver and exact Champagne nesting path."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .baselines import champagne_sbl
from .v7_spatial_atoms import SourceGraph
from .v8_aree import build_aree_modes, qualify_aree_children
from .v8_candidate_tree import CandidateTree
from .v8_component_projection import ProjectionBank
from .v8_kkt_audit import KKTAudit, full_dictionary_derivatives, repair_full_dictionary_kkt
from .v8_posterior_lift import V8SourcePosterior, exact_source_posterior_lift
from .v8_sequential_evidence import SequentialEvidenceState
from .v8_sufficient_statistics import FrequencySufficientStatistic


def fit_v8_pointonly(
    forward: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    max_iter: int = 80,
    tolerance: float = 1.0e-6,
    prune_threshold: float = 0.0,
    noise0: float | None = None,
):
    """Exact nested model: call the repository Champagne implementation."""

    return champagne_sbl(
        forward,
        observations,
        group_size=group_size,
        max_iter=max_iter,
        tolerance=tolerance,
        prune_threshold=prune_threshold,
        noise0=noise0,
    )


@dataclass(frozen=True, slots=True)
class HQSLConfig:
    theta_max: float = 1.0e4
    top_k_exact_gain: int = 8
    maximum_actions: int = 128
    objective_gain_floor: float = 1.0e-9
    kkt_tolerance: float = 1.0e-5
    maximum_kkt_repairs: int = 128
    noise_update_interval: int = 0


@dataclass(frozen=True, slots=True)
class HQSLFitResult:
    method: str
    state: SequentialEvidenceState
    kkt: KKTAudit
    posterior: V8SourcePosterior | None
    candidate_refreshes: int
    candidate_stale_pops: int
    kkt_repairs: int
    initializer_artifact: Any
    runtime_seconds: float
    runtime_breakdown: Mapping[str, float]
    converged: bool


@dataclass(frozen=True, slots=True)
class AREEConfig:
    maximum_modes: int = 2
    parent_ratio_threshold: float = 1.0e-5
    residual_percentile_threshold: float = 0.9
    coherence_threshold: float = 0.995
    derivative_margin: float = 1.0
    gain_threshold: float = 1.0e-7
    observability_floor: float = 1.0e-12
    orthogonality_mode: str = "source_and_observable"


def fit_v8_hqsl_lite(
    statistics: Sequence[FrequencySufficientStatistic],
    bank: ProjectionBank,
    vh: ArrayLike,
    *,
    noise0: ArrayLike,
    config: HQSLConfig = HQSLConfig(),
    frequency_weights: ArrayLike | None = None,
    champagne_initializer: Callable[[], Any] | None = None,
) -> HQSLFitResult:
    """Fit the V8-Lite covariance family using exact one-coordinate updates.

    If supplied, ``champagne_initializer`` is executed inside the standalone
    timer.  Its artifact is retained for audit even when no safe deterministic
    map into the multiscale coordinates is available.
    """

    started = perf_counter()
    initializer_started = perf_counter()
    initializer_artifact = (
        None if champagne_initializer is None else champagne_initializer()
    )
    initializer_seconds = perf_counter() - initializer_started
    state = SequentialEvidenceState(
        statistics, noise=noise0, frequency_weights=frequency_weights
    )
    tree = CandidateTree(bank)
    screening_seconds = 0.0
    coordinate_seconds = 0.0
    noise_seconds = 0.0
    for action_number in range(int(config.maximum_actions)):
        active_base = [
            component.atom_index
            for component in state.components.values()
            if component.atom_index >= 0
        ]
        tree.open_from_active(active_base)
        screen_started = perf_counter()
        derivatives = full_dictionary_derivatives(bank, state)
        excluded = {
            (component.atom_index, component.profile_index)
            for component in state.components.values()
            if component.atom_index >= 0
        }
        tree.push_derivatives(
            derivatives,
            state_version=state.version,
            top_k=config.top_k_exact_gain,
            excluded=excluded,
        )
        screening_seconds += perf_counter() - screen_started
        coordinate_started = perf_counter()
        proposals = []
        for component in state.components.values():
            optimum = state.coordinate_optimum(component, theta_max=config.theta_max)
            proposals.append((float(optimum.delta_objective), component, optimum))
        for _ in range(int(config.top_k_exact_gain)):
            candidate = tree.pop_fresh(
                state_version=state.version,
                refresh_one=lambda atom, profile: float(derivatives[atom, profile]),
            )
            if candidate is None or candidate.derivative >= -config.kkt_tolerance:
                break
            component = bank.component(candidate.atom_index, candidate.profile_index)
            if component.component_id in state.theta:
                continue
            optimum = state.coordinate_optimum(component, theta_max=config.theta_max)
            proposals.append((float(optimum.delta_objective), component, optimum))
        if not proposals:
            coordinate_seconds += perf_counter() - coordinate_started
            break
        _, component, optimum = min(proposals, key=lambda item: item[0])
        if -optimum.delta_objective <= float(config.objective_gain_floor):
            coordinate_seconds += perf_counter() - coordinate_started
            break
        record = state.apply_coordinate(
            component,
            optimum,
            minimum_gain=config.objective_gain_floor,
        )
        coordinate_seconds += perf_counter() - coordinate_started
        if record is None:
            break
        if (
            config.noise_update_interval > 0
            and (action_number + 1) % int(config.noise_update_interval) == 0
        ):
            noise_started = perf_counter()
            state.update_noise()
            noise_seconds += perf_counter() - noise_started
    kkt_started = perf_counter()
    kkt, repairs = repair_full_dictionary_kkt(
        bank,
        state,
        tolerance=config.kkt_tolerance,
        theta_max=config.theta_max,
        maximum_repairs=config.maximum_kkt_repairs,
    )
    kkt_seconds = perf_counter() - kkt_started
    posterior_started = perf_counter()
    posterior = (
        exact_source_posterior_lift(state, vh, bank.profile_matrix)
        if kkt.passed
        else None
    )
    posterior_seconds = perf_counter() - posterior_started
    total = perf_counter() - started
    return HQSLFitResult(
        method="V8-HQSL-Lite",
        state=state,
        kkt=kkt,
        posterior=posterior,
        candidate_refreshes=tree.refreshes,
        candidate_stale_pops=tree.stale_pops,
        kkt_repairs=repairs,
        initializer_artifact=initializer_artifact,
        runtime_seconds=float(total),
        runtime_breakdown={
            "champagne_initializer": float(initializer_seconds),
            "screening": float(screening_seconds),
            "coordinate_update": float(coordinate_seconds),
            "noise": float(noise_seconds),
            "final_kkt": float(kkt_seconds),
            "posterior_lift": float(posterior_seconds),
            "total": float(total),
        },
        converged=bool(kkt.passed and posterior is not None),
    )


def fit_v8_hqsl_aree(
    statistics: Sequence[FrequencySufficientStatistic],
    bank: ProjectionBank,
    vh: ArrayLike,
    graph: SourceGraph,
    *,
    noise0: ArrayLike,
    config: HQSLConfig = HQSLConfig(),
    aree_config: AREEConfig = AREEConfig(),
    frequency_weights: ArrayLike | None = None,
    champagne_initializer: Callable[[], Any] | None = None,
) -> HQSLFitResult:
    """Run Lite, then conditionally add local residual eigenmodes."""

    result = fit_v8_hqsl_lite(
        statistics,
        bank,
        vh,
        noise0=noise0,
        config=config,
        frequency_weights=frequency_weights,
        champagne_initializer=champagne_initializer,
    )
    state = result.state
    aree_started = perf_counter()
    parents = tuple(
        component
        for component in state.components.values()
        if component.kind in {"small_coherent", "medium_coherent"}
    )
    accepted = 0
    for parent in parents:
        profile = bank.profiles[parent.profile_index]
        children = build_aree_modes(
            parent,
            graph,
            vh,
            bank.quotient_factors,
            profile.rho,
            profile_index=parent.profile_index,
            profile_name=profile.name,
            profile_kind=profile.kind,
            maximum_modes=aree_config.maximum_modes,
            observability_floor=aree_config.observability_floor,
            frequency_weights=frequency_weights,
            orthogonality_mode=aree_config.orthogonality_mode,
        )
        qualified = qualify_aree_children(
            state,
            parent,
            children,
            theta_max=config.theta_max,
            parent_ratio_threshold=aree_config.parent_ratio_threshold,
            residual_envelope=None,
            residual_percentile_threshold=aree_config.residual_percentile_threshold,
            coherence_threshold=aree_config.coherence_threshold,
            derivative_margin=aree_config.derivative_margin,
            gain_threshold=aree_config.gain_threshold,
            child_limit=aree_config.maximum_modes,
        )
        for child in qualified:
            if state.apply_coordinate(
                child.component,
                child.optimum,
                minimum_gain=aree_config.gain_threshold,
            ) is not None:
                accepted += 1
    aree_seconds = perf_counter() - aree_started
    if accepted:
        kkt_started = perf_counter()
        kkt, repairs = repair_full_dictionary_kkt(
            bank,
            state,
            tolerance=config.kkt_tolerance,
            theta_max=config.theta_max,
            maximum_repairs=config.maximum_kkt_repairs,
        )
        kkt_seconds = perf_counter() - kkt_started
        posterior_started = perf_counter()
        posterior = (
            exact_source_posterior_lift(state, vh, bank.profile_matrix)
            if kkt.passed
            else None
        )
        posterior_seconds = perf_counter() - posterior_started
    else:
        kkt = result.kkt
        repairs = 0
        kkt_seconds = 0.0
        posterior = result.posterior
        posterior_seconds = 0.0
    breakdown = dict(result.runtime_breakdown)
    breakdown["aree"] = float(aree_seconds)
    breakdown["final_kkt"] = float(breakdown.get("final_kkt", 0.0) + kkt_seconds)
    breakdown["posterior_lift"] = float(
        breakdown.get("posterior_lift", 0.0) + posterior_seconds
    )
    total = float(result.runtime_seconds + aree_seconds + kkt_seconds + posterior_seconds)
    breakdown["total"] = total
    return HQSLFitResult(
        method="V8-HQSL-AREE",
        state=state,
        kkt=kkt,
        posterior=posterior,
        candidate_refreshes=result.candidate_refreshes,
        candidate_stale_pops=result.candidate_stale_pops,
        kkt_repairs=result.kkt_repairs + repairs,
        initializer_artifact=result.initializer_artifact,
        runtime_seconds=total,
        runtime_breakdown=breakdown,
        converged=bool(kkt.passed and posterior is not None),
    )


__all__ = [
    "AREEConfig",
    "HQSLConfig",
    "HQSLFitResult",
    "fit_v8_hqsl_lite",
    "fit_v8_hqsl_aree",
    "fit_v8_pointonly",
]
