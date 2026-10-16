"""Frozen nonnegative spectral-profile bank and shared-first rule."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SpectralProfile:
    name: str
    rho: NDArray[np.float64]
    kind: str

    def __post_init__(self) -> None:
        rho = np.array(self.rho, dtype=np.float64, copy=True)
        if rho.ndim != 1 or rho.size == 0 or np.any(rho <= 0) or not np.isfinite(rho).all():
            raise ValueError("spectral profile must be finite and strictly positive")
        if not np.isclose(float(np.mean(rho)), 1.0, rtol=0, atol=1e-12):
            raise ValueError("spectral profile must have mean one")
        rho.setflags(write=False)
        object.__setattr__(self, "rho", rho)


def build_spectral_profile_bank(
    frequencies_hz: ArrayLike,
    *,
    specific_floor: float = 0.05,
    include_smooth: bool = True,
) -> list[SpectralProfile]:
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be finite and nonempty")
    floor = float(specific_floor)
    if not 0 < floor < 1:
        raise ValueError("specific_floor must lie in (0,1)")
    count = len(frequencies)
    profiles = [SpectralProfile("shared_broadband", np.ones(count), "shared")]
    for index in range(count):
        values = np.full(count, floor)
        values[index] = 1.0
        values /= np.mean(values)
        profiles.append(SpectralProfile(f"frequency_specific_{index}", values, "specific"))
    if include_smooth and count >= 4:
        grid = np.arange(count, dtype=float)
        for number, center in enumerate((0.33 * (count - 1), 0.67 * (count - 1))):
            width = max(1.0, count / 3.0)
            values = floor + np.maximum(0.0, 1.0 - np.abs(grid - center) / width)
            values /= np.mean(values)
            profiles.append(SpectralProfile(f"smooth_adjacent_{number}", values, "smooth"))
    if len(profiles) > count + 3:
        raise AssertionError("profile bank exceeds Q<=F+3")
    return profiles


def shared_first_allows_specific(
    shared_derivative: float,
    specific_derivative: float,
    *,
    delta_specific: float,
) -> bool:
    margin = float(delta_specific)
    if margin < 0 or not np.isfinite(margin):
        raise ValueError("delta_specific must be finite and nonnegative")
    return bool(float(specific_derivative) < float(shared_derivative) - margin)


def profile_bank_fingerprint(profiles: list[SpectralProfile]) -> str:
    payload = [(p.name, p.kind, [float(v).hex() for v in p.rho]) for p in profiles]
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


__all__ = ["SpectralProfile", "build_spectral_profile_bank", "shared_first_allows_specific", "profile_bank_fingerprint"]
