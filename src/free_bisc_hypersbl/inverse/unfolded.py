"""Structure-preserving shallow unfolding with objective-safe fallback."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def objective_safe_correction(
    state: np.ndarray,
    classical_state: np.ndarray,
    correction: np.ndarray,
    objective: Callable[[np.ndarray], float],
    *,
    initial_gate: float = 1.0,
    minimum_gate: float = 2.0**-12,
) -> tuple[np.ndarray, float]:
    """Accept a learned correction only when it does not worsen the objective."""

    del state  # Included in the signature for learned-correction interfaces.
    reference = float(objective(classical_state))
    gate = float(initial_gate)
    while gate >= minimum_gate:
        candidate = classical_state + gate * correction
        if np.all(np.isfinite(candidate)) and float(objective(candidate)) <= reference + 1e-12:
            return candidate, gate
        gate *= 0.5
    return np.asarray(classical_state).copy(), 0.0


def shallow_unfold(
    initial_state: np.ndarray,
    classical_update: Callable[[np.ndarray], np.ndarray],
    correction_model: Callable[[int, np.ndarray], np.ndarray],
    objective: Callable[[np.ndarray], float],
    *,
    layers: int = 7,
) -> tuple[np.ndarray, list[float], list[float]]:
    if not 5 <= layers <= 10:
        raise ValueError("the preregistered shallow unfolding depth is 5--10")
    state = np.asarray(initial_state).copy()
    objectives = [float(objective(state))]
    gates: list[float] = []
    for layer in range(layers):
        classical = np.asarray(classical_update(state))
        correction = np.asarray(correction_model(layer, classical))
        state, gate = objective_safe_correction(state, classical, correction, objective)
        objectives.append(float(objective(state)))
        gates.append(gate)
    return state, objectives, gates

