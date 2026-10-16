"""Auditable head/operator factor cache with complete scientific keys."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Callable, Generic, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class V8CacheKey:
    head_geometry_fingerprint: str
    operator_fingerprint: str
    frequencies_hex: tuple[str, ...]
    harmonic_order: int
    reference_fingerprint: str
    orientation_fingerprint: str
    whitening_fingerprint: str
    atom_bank_version: str
    profile_bank_version: str
    dtype: str

    @property
    def fingerprint(self) -> str:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class V8CacheEvent:
    key_fingerprint: str
    hit: bool
    bytes_estimate: int
    build_seconds: float


def _estimate_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return int(sum(_estimate_bytes(item) for item in value))
    if hasattr(value, "__dataclass_fields__"):
        return int(
            sum(_estimate_bytes(getattr(value, name)) for name in value.__dataclass_fields__)
        )
    return 0


class V8FactorCache(Generic[T]):
    def __init__(self) -> None:
        self._values: dict[str, T] = {}
        self.events: list[V8CacheEvent] = []
        self.hits = 0
        self.misses = 0

    def get_or_build(self, key: V8CacheKey, builder: Callable[[], T]) -> T:
        fingerprint = key.fingerprint
        if fingerprint in self._values:
            self.hits += 1
            value = self._values[fingerprint]
            self.events.append(
                V8CacheEvent(fingerprint, True, _estimate_bytes(value), 0.0)
            )
            return value
        started = perf_counter()
        value = builder()
        elapsed = perf_counter() - started
        self._values[fingerprint] = value
        self.misses += 1
        self.events.append(
            V8CacheEvent(fingerprint, False, _estimate_bytes(value), float(elapsed))
        )
        return value

    def clear(self) -> None:
        self._values.clear()


__all__ = ["V8CacheEvent", "V8CacheKey", "V8FactorCache"]
