"""Linear EEG reference and bipolar montage transformations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _validate_channel_count(n_channels: int) -> int:
    count = int(n_channels)
    if count < 2:
        raise ValueError("at least two channels are required")
    return count


def average_reference_matrix(n_channels: int) -> FloatArray:
    """Return the idempotent average-reference projection ``I - 11^T / E``."""

    count = _validate_channel_count(n_channels)
    return np.eye(count, dtype=np.float64) - np.ones((count, count)) / count


def single_reference_matrix(
    n_channels: int,
    reference: int,
    *,
    drop_reference: bool = False,
) -> FloatArray:
    """Reference every channel to one channel.

    With ``drop_reference=False`` the retained reference row is exactly zero.
    Dropping that row yields the usual full-row-rank ``(E-1) x E`` transform.
    """

    count = _validate_channel_count(n_channels)
    reference = int(reference)
    if reference < 0:
        reference += count
    if reference < 0 or reference >= count:
        raise IndexError("reference channel is out of range")
    transform = np.eye(count, dtype=np.float64)
    transform[:, reference] -= 1.0
    if drop_reference:
        transform = np.delete(transform, reference, axis=0)
    return transform


def _resolve_channel(
    channel: int | str,
    n_channels: int,
    channel_names: Sequence[str] | None,
) -> int:
    if isinstance(channel, str):
        if channel_names is None:
            raise ValueError("channel_names are required when bipolar pairs use names")
        if len(channel_names) != n_channels or len(set(channel_names)) != n_channels:
            raise ValueError("channel_names must be unique and match n_channels")
        try:
            return list(channel_names).index(channel)
        except ValueError as error:
            raise KeyError(f"unknown channel name: {channel}") from error
    index = int(channel)
    if index < 0:
        index += n_channels
    if index < 0 or index >= n_channels:
        raise IndexError("bipolar channel index is out of range")
    return index


def bipolar_reference_matrix(
    n_channels: int,
    pairs: Sequence[tuple[int | str, int | str]],
    *,
    channel_names: Sequence[str] | None = None,
) -> FloatArray:
    """Construct exact ``positive - negative`` rows for a bipolar montage."""

    count = _validate_channel_count(n_channels)
    if not pairs:
        raise ValueError("at least one bipolar pair is required")
    transform = np.zeros((len(pairs), count), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError("each bipolar pair must contain two channels")
        positive = _resolve_channel(pair[0], count, channel_names)
        negative = _resolve_channel(pair[1], count, channel_names)
        if positive == negative:
            raise ValueError("a bipolar pair cannot reference a channel to itself")
        resolved = (positive, negative)
        if resolved in seen:
            raise ValueError(f"duplicate bipolar pair: {pair}")
        seen.add(resolved)
        transform[row, positive] = 1.0
        transform[row, negative] = -1.0
    return transform


def apply_reference(transform: ArrayLike, values: ArrayLike) -> np.ndarray:
    """Apply a channel transform to data or a lead field along axis zero."""

    matrix = np.asarray(transform, dtype=np.float64)
    array = np.asarray(values)
    if matrix.ndim != 2 or array.ndim < 1:
        raise ValueError("transform must be 2-D and values must have a channel axis")
    if matrix.shape[1] != array.shape[0]:
        raise ValueError(
            f"transform expects {matrix.shape[1]} channels, got {array.shape[0]}"
        )
    return matrix @ array


__all__ = [
    "apply_reference",
    "average_reference_matrix",
    "bipolar_reference_matrix",
    "single_reference_matrix",
]
