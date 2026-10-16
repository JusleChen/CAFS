"""Standard-montage layered-sphere templates for data without individual MRI."""

from __future__ import annotations

import numpy as np

from .global_matrix import (
    CalibrationSpec,
    FrequencySphericalHarmonicModel,
    LayeredSphereHead,
    LeadfieldFamily,
    build_oriented_leadfield_family,
)
from .reference import average_reference_matrix


def fibonacci_sources(n_sources: int, radius_m: float = 0.055) -> np.ndarray:
    index = np.arange(int(n_sources), dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n_sources
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    xy = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    return radius_m * np.column_stack([xy * np.cos(angle), xy * np.sin(angle), z])


def montage_positions(channel_names: tuple[str, ...] | list[str], radius_m: float = 0.09) -> np.ndarray:
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    positions = {key.upper(): np.asarray(value, dtype=float) for key, value in montage.get_positions()["ch_pos"].items()}
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    output = []
    missing = []
    for name in channel_names:
        key = aliases.get(str(name).upper(), str(name).upper())
        if key not in positions:
            missing.append(str(name))
            continue
        point = positions[key]
        output.append(point / np.linalg.norm(point) * radius_m)
    if missing:
        raise KeyError(f"standard montage lacks channels: {missing}")
    return np.asarray(output)


def standard_layered_head(channel_names: tuple[str, ...] | list[str]) -> LayeredSphereHead:
    return LayeredSphereHead(
        radii_m=[0.070, 0.076, 0.080, 0.084, 0.090],
        conductivity_s_per_m=[0.14, 0.33, 1.79, 0.0132, 0.43],
        relative_permittivity=[50.0, 80.0, 109.0, 20.0, 65.0],
        electrode_xyz_m=montage_positions(channel_names, 0.090),
        dispersion_alpha=[0.02, 0.03, 0.01, 0.08, 0.04],
        layer_names=("white_matter", "gray_matter", "csf", "skull", "scalp"),
    )


def standard_eeg_leadfield_family(
    channel_names: tuple[str, ...] | list[str],
    frequencies_hz: np.ndarray | list[float],
    *,
    n_sources: int = 32,
    source_radius_m: float = 0.055,
    l_max: int = 4,
    theta=None,
    average_reference: bool = True,
) -> tuple[LeadfieldFamily, FrequencySphericalHarmonicModel]:
    head = standard_layered_head(channel_names)
    calibration = CalibrationSpec(
        admittivity_ratio_layers=(1, 2, 3),
        include_log_radius_scale=True,
        include_electrode_translation=True,
    )
    model = FrequencySphericalHarmonicModel(head, calibration)
    transform = average_reference_matrix(len(channel_names)) if average_reference else None
    family = build_oriented_leadfield_family(
        model,
        frequencies_hz,
        fibonacci_sources(n_sources, source_radius_m),
        l_max=l_max,
        theta=theta,
        reference_matrix=transform,
    )
    return family, model

