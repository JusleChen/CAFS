"""MM-anchored robust unrolled structured SBL (RUST-SBL).

RUST-SBL learns only a small shared residual update on low-dimensional
posterior/evidence features.  It has no source-index embedding and accepts no
truth operator, source truth, EMD, or geometry seed.  Each layer first obtains
an analytic EM/MU anchor, then evaluates a learned graph proposal against the
proper-complex Type-II objective using fixed 1, 1/2, and 1/4 backtracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

from .v4_fast_structured_mm import (
    _log_backtrack,
    multiscale_graph_mix,
)
from .v4_multifrequency_core import (
    MultiFrequencyPosterior,
    MultiFrequencySBLConfig,
    effective_frequency_weights,
    fit_multifrequency_group_sbl,
    multifrequency_group_em_update,
    multifrequency_group_mu_update,
    multifrequency_noise_update,
    multifrequency_posterior,
    multifrequency_type2_objective,
    multifrequency_type2_terms_from_posterior,
)


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
RUST_FEATURE_NAMES_V4 = (
    "log_gamma",
    "log_expected_power",
    "log(EM/gamma)",
    "log(MU/gamma)",
    "relevance",
    "uncertainty",
    "depth",
    "train_library_source_instability",
    "S_mean",
    "S_max",
    "S2_mean",
    "layer_encoding",
)
RUST_FEATURE_NORMALIZER_VERSION = "rust_feature_normalizer_v4_analytic_rollout_2"
RUST_FEATURE_CLIP_V4 = (-8.0, 8.0)
RUST_RAW_FEATURE_CONTRACT_V4 = "raw_truth_free_12d_v2"
RUST_SOURCE_INSTABILITY_FORMULA_V4 = (
    "weighted_rms_train_frequency_mismatch_times_within_group_frequency_sensitivity_v1"
)


def _legacy_standardize(values: ArrayLike) -> FloatArray:
    """Compatibility path for old checkpoints without frozen statistics."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    scale = max(1.4826 * mad, float(np.std(vector)), 1.0e-8)
    return np.clip((vector - median) / scale, -8.0, 8.0)


@dataclass(frozen=True, slots=True)
class RUSTFeatureNormalizer:
    """Frozen library-train robust statistics for the 12 RUST features."""

    center: FloatArray
    scale: FloatArray
    scale_method: tuple[str, ...]
    source_split: str
    fit_episode_count: int
    fit_group_observation_count: int
    clip_lower: float = RUST_FEATURE_CLIP_V4[0]
    clip_upper: float = RUST_FEATURE_CLIP_V4[1]
    formula_version: str = RUST_FEATURE_NORMALIZER_VERSION

    def __post_init__(self) -> None:
        center = np.array(self.center, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if center.shape != (12,) or scale.shape != (12,):
            raise ValueError("RUST feature center/scale must each have shape (12,)")
        if (
            not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("RUST feature center/scale must be finite and positive")
        methods = tuple(str(value) for value in self.scale_method)
        if len(methods) != 12 or set(methods) - {"mad", "iqr", "std", "unit"}:
            raise ValueError("RUST feature scale methods are invalid")
        if not isinstance(self.source_split, str) or not self.source_split:
            raise ValueError("RUST feature normalizer source_split must be nonempty")
        if (
            isinstance(self.fit_episode_count, (bool, np.bool_))
            or int(self.fit_episode_count) < 0
            or isinstance(self.fit_group_observation_count, (bool, np.bool_))
            or int(self.fit_group_observation_count) < 0
        ):
            raise ValueError("RUST feature fit counts must be nonnegative integers")
        if self.source_split == "library_train" and (
            int(self.fit_episode_count) < 1
            or int(self.fit_group_observation_count) < 1
        ):
            raise ValueError("library_train normalizer must record nonempty fit counts")
        if (
            float(self.clip_lower) != RUST_FEATURE_CLIP_V4[0]
            or float(self.clip_upper) != RUST_FEATURE_CLIP_V4[1]
        ):
            raise ValueError("RUST feature clipping is frozen at [-8,8]")
        if self.formula_version != RUST_FEATURE_NORMALIZER_VERSION:
            raise ValueError("unknown RUST feature normalization formula")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "scale_method", methods)
        object.__setattr__(self, "fit_episode_count", int(self.fit_episode_count))
        object.__setattr__(
            self,
            "fit_group_observation_count",
            int(self.fit_group_observation_count),
        )

    @classmethod
    def identity(cls) -> "RUSTFeatureNormalizer":
        """Return a non-production identity adapter for initialization/tests."""

        return cls(
            center=np.zeros(12, dtype=np.float64),
            scale=np.ones(12, dtype=np.float64),
            scale_method=("unit",) * 12,
            source_split="unfitted_identity",
            fit_episode_count=0,
            fit_group_observation_count=0,
        )

    def transform(self, values: ArrayLike) -> FloatArray:
        matrix = np.asarray(values, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != 12
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("raw RUST features must be finite with shape (group,12)")
        return np.asarray(
            np.clip(
                (matrix - np.asarray(self.center)[None, :])
                / np.asarray(self.scale)[None, :],
                self.clip_lower,
                self.clip_upper,
            ),
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "formula_version": self.formula_version,
            "feature_names": list(RUST_FEATURE_NAMES_V4),
            "center": np.asarray(self.center).tolist(),
            "scale": np.asarray(self.scale).tolist(),
            "scale_method": list(self.scale_method),
            "source_split": self.source_split,
            "fit_episode_count": self.fit_episode_count,
            "fit_group_observation_count": self.fit_group_observation_count,
            "clip": [self.clip_lower, self.clip_upper],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RUSTFeatureNormalizer":
        names = tuple(str(item) for item in value.get("feature_names", ()))
        if names and names != RUST_FEATURE_NAMES_V4:
            raise ValueError("RUST feature order changed")
        clip = np.asarray(value.get("clip", RUST_FEATURE_CLIP_V4), dtype=np.float64)
        if clip.shape != (2,):
            raise ValueError("RUST feature clip receipt must contain two values")
        return cls(
            center=np.asarray(value["center"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            scale_method=tuple(str(item) for item in value["scale_method"]),
            source_split=str(value["source_split"]),
            fit_episode_count=int(value["fit_episode_count"]),
            fit_group_observation_count=int(value["fit_group_observation_count"]),
            clip_lower=float(clip[0]),
            clip_upper=float(clip[1]),
            formula_version=str(
                value.get("formula_version", RUST_FEATURE_NORMALIZER_VERSION)
            ),
        )

    @property
    def provenance_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def fit_rust_feature_normalizer(
    feature_matrices: Sequence[ArrayLike],
    *,
    source_split: str = "library_train",
    fit_episode_count: int | None = None,
) -> RUSTFeatureNormalizer:
    """Fit median/MAD statistics over all train episodes and source groups."""

    if source_split != "library_train":
        raise ValueError("RUST feature statistics may be fitted on library_train only")
    matrices = tuple(np.asarray(value, dtype=np.float64) for value in feature_matrices)
    if not matrices or any(
        value.ndim != 2
        or value.shape[1] != 12
        or value.shape[0] < 1
        or not np.all(np.isfinite(value))
        for value in matrices
    ):
        raise ValueError("normalizer fitting requires finite nonempty (*,12) matrices")
    pooled = np.concatenate(matrices, axis=0)
    center = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - center[None, :]), axis=0)
    mad_scale = 1.4826 * mad
    quartiles = np.quantile(pooled, (0.25, 0.75), axis=0, method="linear")
    iqr_scale = (quartiles[1] - quartiles[0]) / 1.349
    std_scale = np.std(pooled, axis=0, ddof=0)
    scale = np.empty(12, dtype=np.float64)
    methods: list[str] = []
    for index in range(12):
        tolerance = max(
            1.0e-12,
            np.finfo(float).eps * max(1.0, abs(float(center[index]))),
        )
        candidates = (
            (float(mad_scale[index]), "mad"),
            (float(iqr_scale[index]), "iqr"),
            (float(std_scale[index]), "std"),
        )
        selected_scale, selected_method = 1.0, "unit"
        for candidate, method in candidates:
            if np.isfinite(candidate) and candidate > tolerance:
                selected_scale, selected_method = candidate, method
                break
        scale[index] = selected_scale
        methods.append(selected_method)
    episode_count = (
        len(matrices) if fit_episode_count is None else int(fit_episode_count)
    )
    if episode_count < 1:
        raise ValueError("normalizer fit_episode_count must be positive")
    return RUSTFeatureNormalizer(
        center=np.asarray(center, dtype=np.float64),
        scale=scale,
        scale_method=tuple(methods),
        source_split=source_split,
        fit_episode_count=episode_count,
        fit_group_observation_count=int(pooled.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class RUSTSBLWeights:
    input_weight: FloatArray
    input_bias: FloatArray
    output_weight: FloatArray
    output_bias: float
    layer_residual_scale: FloatArray
    layer_graph_logits: FloatArray
    layer_graph_damping: FloatArray
    layer_depth_scale: FloatArray
    layer_noise_damping: FloatArray
    layer_alpha: FloatArray | None = None
    train_frequency_mismatch: FloatArray | None = None
    train_frequency_mismatch_receipt: Mapping[str, object] | None = None
    feature_normalizer: RUSTFeatureNormalizer | None = None
    normalization_required: bool = False

    def __post_init__(self) -> None:
        w1 = np.asarray(self.input_weight, dtype=np.float64)
        b1 = np.asarray(self.input_bias, dtype=np.float64)
        w2 = np.asarray(self.output_weight, dtype=np.float64)
        residual = np.asarray(self.layer_residual_scale, dtype=np.float64)
        logits = np.asarray(self.layer_graph_logits, dtype=np.float64)
        damping = np.asarray(self.layer_graph_damping, dtype=np.float64)
        depth = np.asarray(self.layer_depth_scale, dtype=np.float64)
        noise = np.asarray(self.layer_noise_damping, dtype=np.float64)
        alpha = (
            np.full(residual.size, 0.35, dtype=np.float64)
            if self.layer_alpha is None
            else np.asarray(self.layer_alpha, dtype=np.float64)
        )
        if w1.ndim != 2 or b1.shape != (w1.shape[1],) or w2.shape != (w1.shape[1],):
            raise ValueError("shared MLP weight shapes are inconsistent")
        layers = residual.size
        if layers not in {6, 8} or logits.shape != (layers, 3):
            raise ValueError("RUST-SBL supports exactly 6 or 8 layers")
        if (
            damping.shape != (layers,)
            or depth.shape != (layers,)
            or noise.shape != (layers,)
            or alpha.shape != (layers,)
        ):
            raise ValueError("per-layer scalar arrays must align")
        arrays = (w1, b1, w2, residual, logits, damping, depth, noise, alpha)
        if any(not np.all(np.isfinite(value)) for value in arrays) or not np.isfinite(self.output_bias):
            raise ValueError("all learned weights must be finite")
        if np.any(residual < 0.0) or np.any(residual > 1.0):
            raise ValueError("residual scales must lie in [0,1]")
        if np.any(damping < 0.0) or np.any(damping > 1.0):
            raise ValueError("graph damping must lie in [0,1]")
        if np.any(depth < 0.0) or np.any(depth > 1.0):
            raise ValueError("depth gates must lie in [0,1]")
        if np.any(noise < 0.0) or np.any(noise > 1.0):
            raise ValueError("noise damping must lie in [0,1]")
        if np.any(alpha < 0.0) or np.any(alpha > 1.0):
            raise ValueError("EM/MU alpha must lie in [0,1]")
        mismatch = None
        if self.train_frequency_mismatch is not None:
            mismatch = np.array(
                self.train_frequency_mismatch, dtype=np.float64, copy=True
            ).reshape(-1)
            if (
                mismatch.size < 1
                or not np.all(np.isfinite(mismatch))
                or np.any(mismatch < 0.0)
            ):
                raise ValueError(
                    "train frequency mismatch must be a finite nonnegative vector"
                )
            mismatch.setflags(write=False)
        receipt = self.train_frequency_mismatch_receipt
        if (mismatch is None) != (receipt is None):
            raise ValueError(
                "train frequency mismatch and its receipt must be supplied together"
            )
        if receipt is not None:
            receipt = json.loads(
                json.dumps(receipt, sort_keys=True, ensure_ascii=True)
            )
            if (
                not isinstance(receipt, dict)
                or receipt.get("source_split") != "library_train"
                or receipt.get("truth_operator_arrays_serialized") is not False
            ):
                raise ValueError(
                    "source-instability receipt must bind a truth-array-free library_train aggregate"
                )
            if receipt.get("role") != "legacy_static_unit_test_compatibility_only":
                receipt_values = np.asarray(
                    receipt.get("surrogate"), dtype=np.float64
                ).reshape(-1)
                if (
                    mismatch is None
                    or receipt_values.shape != mismatch.shape
                    or not np.array_equal(receipt_values, mismatch)
                ):
                    raise ValueError(
                        "source-instability receipt does not bind the frozen vector"
                    )
        if self.feature_normalizer is not None and not isinstance(
            self.feature_normalizer, RUSTFeatureNormalizer
        ):
            raise TypeError("feature_normalizer must be a RUSTFeatureNormalizer")
        if self.feature_normalizer is not None and w1.shape[0] != 12:
            raise ValueError("frozen RUST feature normalization requires 12 inputs")
        if not isinstance(self.normalization_required, (bool, np.bool_)):
            raise TypeError("normalization_required must be boolean")
        if self.normalization_required and (
            self.feature_normalizer is None
            or self.feature_normalizer.source_split != "library_train"
            or mismatch is None
        ):
            raise ValueError(
                "production RUST weights require library_train normalization and mismatch"
            )
        object.__setattr__(self, "layer_alpha", alpha)
        object.__setattr__(self, "train_frequency_mismatch", mismatch)
        object.__setattr__(self, "train_frequency_mismatch_receipt", receipt)

    @property
    def feature_count(self) -> int:
        return int(np.asarray(self.input_weight).shape[0])

    @property
    def hidden_count(self) -> int:
        return int(np.asarray(self.input_weight).shape[1])

    @property
    def layers(self) -> int:
        return int(np.asarray(self.layer_residual_scale).size)

    @property
    def parameter_count(self) -> int:
        arrays = (
            self.input_weight,
            self.input_bias,
            self.output_weight,
            self.layer_residual_scale,
            self.layer_graph_logits,
            self.layer_graph_damping,
            self.layer_depth_scale,
            self.layer_noise_damping,
            self.layer_alpha,
        )
        return int(sum(np.asarray(value).size for value in arrays) + 1)

    @property
    def source_instability_provenance_sha256(self) -> str | None:
        if self.train_frequency_mismatch is None:
            return None
        payload = {
            "formula": RUST_SOURCE_INSTABILITY_FORMULA_V4,
            "frequency_mismatch": np.asarray(
                self.train_frequency_mismatch, dtype=np.float64
            ).tolist(),
            "receipt": self.train_frequency_mismatch_receipt,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "input_weight": np.asarray(self.input_weight).tolist(),
            "input_bias": np.asarray(self.input_bias).tolist(),
            "output_weight": np.asarray(self.output_weight).tolist(),
            "output_bias": float(self.output_bias),
            "layer_residual_scale": np.asarray(self.layer_residual_scale).tolist(),
            "layer_graph_logits": np.asarray(self.layer_graph_logits).tolist(),
            "layer_graph_damping": np.asarray(self.layer_graph_damping).tolist(),
            "layer_depth_scale": np.asarray(self.layer_depth_scale).tolist(),
            "layer_noise_damping": np.asarray(self.layer_noise_damping).tolist(),
            "layer_alpha": np.asarray(self.layer_alpha).tolist(),
            "train_frequency_mismatch": (
                None
                if self.train_frequency_mismatch is None
                else np.asarray(self.train_frequency_mismatch).tolist()
            ),
            "train_frequency_mismatch_receipt": self.train_frequency_mismatch_receipt,
            "source_instability_formula": RUST_SOURCE_INSTABILITY_FORMULA_V4,
            "feature_normalizer": (
                None
                if self.feature_normalizer is None
                else self.feature_normalizer.to_dict()
            ),
            "normalization_required": bool(self.normalization_required),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RUSTSBLWeights":
        raw_normalizer = value.get("feature_normalizer")
        return cls(
            input_weight=np.asarray(value["input_weight"], dtype=np.float64),
            input_bias=np.asarray(value["input_bias"], dtype=np.float64),
            output_weight=np.asarray(value["output_weight"], dtype=np.float64),
            output_bias=float(value["output_bias"]),
            layer_residual_scale=np.asarray(value["layer_residual_scale"], dtype=np.float64),
            layer_graph_logits=np.asarray(value["layer_graph_logits"], dtype=np.float64),
            layer_graph_damping=np.asarray(value["layer_graph_damping"], dtype=np.float64),
            layer_depth_scale=np.asarray(value["layer_depth_scale"], dtype=np.float64),
            layer_noise_damping=np.asarray(value["layer_noise_damping"], dtype=np.float64),
            layer_alpha=(
                None
                if "layer_alpha" not in value
                else np.asarray(value["layer_alpha"], dtype=np.float64)
            ),
            train_frequency_mismatch=(
                None
                if value.get("train_frequency_mismatch") is None
                else np.asarray(value["train_frequency_mismatch"], dtype=np.float64)
            ),
            train_frequency_mismatch_receipt=(
                None
                if value.get("train_frequency_mismatch_receipt") is None
                else value["train_frequency_mismatch_receipt"]  # type: ignore[arg-type]
            ),
            feature_normalizer=(
                None
                if raw_normalizer is None
                else RUSTFeatureNormalizer.from_dict(raw_normalizer)  # type: ignore[arg-type]
            ),
            normalization_required=bool(value.get("normalization_required", False)),
        )


def initialize_rust_weights(
    seed: int,
    *,
    layers: int = 8,
    feature_count: int = 12,
    hidden_count: int = 8,
    zero_residual: bool = False,
) -> RUSTSBLWeights:
    if layers not in {6, 8} or not 10 <= feature_count <= 16 or hidden_count not in {8, 16}:
        raise ValueError("architecture is outside the preregistered V4 set")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    scale = 0.0 if zero_residual else 0.04
    return RUSTSBLWeights(
        input_weight=rng.normal(scale=scale, size=(feature_count, hidden_count)),
        input_bias=np.zeros(hidden_count, dtype=np.float64),
        output_weight=rng.normal(scale=scale, size=hidden_count),
        output_bias=0.0,
        layer_residual_scale=np.full(layers, 0.20, dtype=np.float64),
        layer_graph_logits=np.tile(np.array([2.0, 0.0, -1.0]), (layers, 1)),
        layer_graph_damping=np.full(layers, 0.16, dtype=np.float64),
        layer_depth_scale=np.full(layers, 0.5, dtype=np.float64),
        layer_noise_damping=np.full(layers, 0.5, dtype=np.float64),
        layer_alpha=np.full(layers, 0.35, dtype=np.float64),
        feature_normalizer=(
            RUSTFeatureNormalizer.identity() if feature_count == 12 else None
        ),
        normalization_required=False,
    )


def shared_mlp_residual(features: ArrayLike, weights: RUSTSBLWeights) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != weights.feature_count:
        raise ValueError("features must have shape (group, frozen feature count)")
    hidden = np.tanh(matrix @ np.asarray(weights.input_weight) + np.asarray(weights.input_bias))
    return np.tanh(hidden @ np.asarray(weights.output_weight) + float(weights.output_bias))


def learned_log_gamma_proposal(
    log_anchor: ArrayLike,
    features: ArrayLike,
    weights: RUSTSBLWeights,
    *,
    layer: int,
) -> FloatArray:
    anchor = np.asarray(log_anchor, dtype=np.float64)
    residual = shared_mlp_residual(features, weights)
    if anchor.shape != residual.shape:
        raise ValueError("anchor and groupwise residual must align")
    return anchor + float(weights.layer_residual_scale[layer]) * residual


def strict_log_gamma_candidate(
    log_gamma_current: ArrayLike,
    z_graph: ArrayLike,
    normalized_depth: ArrayLike,
    *,
    damping: float,
    depth_gate: float,
) -> FloatArray:
    """Frozen Chapter-9 structured candidate, with no hidden extra mixture."""

    current = np.asarray(log_gamma_current, dtype=np.float64)
    graph_value = np.asarray(z_graph, dtype=np.float64)
    depth = np.asarray(normalized_depth, dtype=np.float64)
    amount = float(damping)
    depth_amount = float(depth_gate)
    if (
        current.shape != graph_value.shape
        or current.shape != depth.shape
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(graph_value))
        or not np.all(np.isfinite(depth))
        or not 0.0 <= amount <= 1.0
        or not 0.0 <= depth_amount <= 1.0
    ):
        raise ValueError("strict RUST candidate inputs or gates are invalid")
    return np.asarray(
        (1.0 - amount) * current
        + amount * graph_value
        + depth_amount * depth,
        dtype=np.float64,
    )


def source_instability_from_operator(
    operator: ArrayLike,
    train_frequency_mismatch: ArrayLike,
    *,
    omega: ArrayLike | None = None,
    group_size: int = 3,
) -> FloatArray:
    """Map a frozen frequency mismatch receipt to source groups.

    The mapping only compares frequency sensitivities *within the same source
    group*.  Consequently adding, removing, or permuting other groups cannot
    change a group's value.  A group-wise common operator gain also cancels.
    """

    operators = np.asarray(operator, dtype=np.complex128)
    if operators.ndim == 2:
        operators = operators[None, :, :]
    size = int(group_size)
    if (
        operators.ndim != 3
        or size < 1
        or operators.shape[2] % size
        or not np.all(np.isfinite(operators))
    ):
        raise ValueError("source instability requires finite grouped operators")
    mismatch = np.asarray(train_frequency_mismatch, dtype=np.float64).reshape(-1)
    if (
        mismatch.shape != (operators.shape[0],)
        or not np.all(np.isfinite(mismatch))
        or np.any(mismatch < 0.0)
    ):
        raise ValueError("frozen train mismatch must align with frequencies")
    effective = (
        np.ones(operators.shape[0], dtype=np.float64)
        if omega is None
        else np.asarray(omega, dtype=np.float64)
    )
    if (
        effective.shape != (operators.shape[0],)
        or not np.all(np.isfinite(effective))
        or np.any(effective < 0.0)
        or not np.any(effective > 0.0)
    ):
        raise ValueError("source-instability omega is invalid")
    groups = operators.shape[2] // size
    blocks = operators.reshape(
        operators.shape[0], operators.shape[1], groups, size
    )
    sensitivity = np.sqrt(
        np.sum(np.abs(blocks) ** 2, axis=(1, 3))
        / max(operators.shape[1] * size, 1)
    )
    total = float(np.sum(effective))
    local_rms = np.sqrt(
        np.sum(effective[:, None] * sensitivity**2, axis=0) / total
    )
    relative = sensitivity / np.maximum(local_rms[None, :], 1.0e-300)
    result = np.sqrt(
        np.sum(
            effective[:, None] * (mismatch[:, None] * relative) ** 2,
            axis=0,
        )
        / total
    )
    if result.shape != (groups,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("source-instability feature is invalid")
    return np.asarray(result, dtype=np.float64)


def _group_features(
    operator: ComplexArray,
    observations: ComplexArray,
    gamma: FloatArray,
    expected_power: FloatArray,
    em: FloatArray,
    mu: FloatArray,
    noise: ArrayLike,
    graph: sparse.csr_matrix,
    depth_feature: FloatArray,
    *,
    omega: ArrayLike | None = None,
    train_frequency_mismatch: ArrayLike | None = None,
    layer: int = 0,
    layers: int = 1,
    group_size: int,
) -> FloatArray:
    operators = np.asarray(operator, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if operators.ndim == 2:
        operators = operators[None, :, :]
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError("single-frequency observations must be sensor by snapshot")
        data = data[None, :, :]
    elif operators.ndim == 3:
        if data.ndim == 2:
            data = data[:, :, None]
        if data.ndim != 3 or data.shape[:2] != operators.shape[:2]:
            raise ValueError("multi-frequency observations must align with leadfields")
    else:
        raise ValueError("operator must be single- or multi-frequency")
    noises = np.asarray(noise, dtype=np.float64)
    if noises.ndim == 0:
        noises = np.full(operators.shape[0], float(noises), dtype=np.float64)
    else:
        noises = noises.reshape(-1)
    if noises.shape != (operators.shape[0],) or np.any(noises <= 0.0):
        raise ValueError("features require one positive noise value per frequency")
    effective = (
        np.ones(operators.shape[0], dtype=np.float64)
        if omega is None
        else np.asarray(omega, dtype=np.float64)
    )
    if (
        effective.shape != (operators.shape[0],)
        or np.any(effective < 0.0)
        or not np.any(effective > 0.0)
    ):
        raise ValueError("feature omega must be nonnegative and nonempty")
    state = multifrequency_posterior(
        operators, data, gamma, noises, group_size=group_size
    )
    groups = gamma.size
    posterior_uncertainty = np.zeros(groups)
    leverage = np.zeros(groups)
    for frequency in range(operators.shape[0]):
        weight = float(effective[frequency])
        if weight == 0.0:
            continue
        cy = state.sensor_covariances[frequency]
        factor = linalg.cho_factor(
            (cy + cy.conj().T) * 0.5, lower=True, check_finite=False
        )
        solved_a = linalg.cho_solve(
            factor, operators[frequency], check_finite=False
        )
        for group in range(groups):
            sl = slice(group * group_size, (group + 1) * group_size)
            posterior_uncertainty[group] += weight * (
                np.trace(state.covariances[frequency, sl, sl]).real / group_size
            )
            leverage[group] += weight * (
                np.trace(
                    operators[frequency, :, sl].conj().T @ solved_a[:, sl]
                ).real
                / group_size
            )
    total_weight = float(np.sum(effective))
    posterior_uncertainty /= total_weight
    leverage /= total_weight
    z = np.log(np.maximum(gamma, 1.0e-300))
    log_expected = np.log(np.maximum(expected_power, 1.0e-300))
    one_hop = np.asarray(graph @ log_expected).reshape(-1)
    two_hop = np.asarray(graph @ one_hop).reshape(-1)
    support = graph.toarray() > 0.0
    support |= np.eye(groups, dtype=bool)
    neighbourhood_max = np.asarray(
        [np.max(log_expected[support[group]]) for group in range(groups)],
        dtype=np.float64,
    )
    instability = (
        np.zeros(groups, dtype=np.float64)
        if train_frequency_mismatch is None
        else source_instability_from_operator(
            operators,
            train_frequency_mismatch,
            omega=effective,
            group_size=group_size,
        )
    )
    layer_index = int(layer)
    layer_count = int(layers)
    if layer_count < 1 or not 0 <= layer_index < layer_count:
        raise ValueError("RUST layer encoding is outside the frozen architecture")
    layer_encoding = np.full(
        groups,
        0.0 if layer_count == 1 else -1.0 + 2.0 * layer_index / (layer_count - 1),
        dtype=np.float64,
    )
    raw = np.column_stack(
        [
            z,
            log_expected,
            np.log(np.maximum(em, 1.0e-300) / np.maximum(gamma, 1.0e-300)),
            np.log(np.maximum(mu, 1.0e-300) / np.maximum(gamma, 1.0e-300)),
            np.log(np.maximum(leverage, 1.0e-300)),
            np.log(np.maximum(posterior_uncertainty, 1.0e-300)),
            depth_feature,
            instability,
            one_hop,
            neighbourhood_max,
            two_hop,
            layer_encoding,
        ]
    )
    if raw.shape != (groups, 12) or not np.all(np.isfinite(raw)):
        raise FloatingPointError("raw truth-free RUST features are invalid")
    return np.asarray(raw, dtype=np.float64)


def build_rust_group_features(
    operator: ArrayLike,
    observations: ArrayLike,
    gamma: ArrayLike,
    expected_power: ArrayLike,
    em: ArrayLike,
    mu: ArrayLike,
    noise: ArrayLike,
    graph: sparse.spmatrix,
    depth_feature: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
    train_frequency_mismatch: ArrayLike | None = None,
    layer: int = 0,
    layers: int = 1,
    group_size: int = 3,
    normalizer: RUSTFeatureNormalizer | None = None,
) -> FloatArray:
    """Build raw truth-free features, optionally applying frozen train stats.

    The default deliberately returns raw features for prepared library
    artifacts.  Passing a :class:`RUSTFeatureNormalizer` applies exactly the
    same library-train transform used by frozen inference.
    """

    a = np.asarray(operator, dtype=np.complex128)
    y = np.asarray(observations, dtype=np.complex128)
    if a.ndim == 2 and y.ndim == 1:
        y = y[:, None]
    values = np.asarray(gamma, dtype=np.float64)
    target = np.asarray(expected_power, dtype=np.float64)
    em_values = np.asarray(em, dtype=np.float64)
    mu_values = np.asarray(mu, dtype=np.float64)
    matrix = sparse.csr_matrix(graph, dtype=np.float64)
    depth = np.asarray(depth_feature, dtype=np.float64)
    groups = a.shape[-1] // int(group_size)
    if any(value.shape != (groups,) for value in (values, target, em_values, mu_values, depth)):
        raise ValueError("all group feature inputs must align with the operator")
    if matrix.shape != (groups, groups):
        raise ValueError("feature graph must align with source groups")
    if a.ndim not in {2, 3}:
        raise ValueError("feature operator must be single- or multi-frequency")
    frequency_count = 1 if a.ndim == 2 else a.shape[0]
    _, effective, _ = effective_frequency_weights(
        frequency_weights, frequency_count
    )
    raw = _group_features(
        a,
        y,
        values,
        target,
        em_values,
        mu_values,
        noise,
        matrix,
        depth,
        omega=effective,
        train_frequency_mismatch=train_frequency_mismatch,
        layer=layer,
        layers=layers,
        group_size=int(group_size),
    )
    if normalizer is None:
        return raw
    if not isinstance(normalizer, RUSTFeatureNormalizer):
        raise TypeError("normalizer must be a RUSTFeatureNormalizer")
    return normalizer.transform(raw)


@dataclass(frozen=True, slots=True)
class RUSTSBLConfig:
    group_size: int = 3
    em_fraction: float = 0.65
    gamma_floor: float = 1.0e-12
    noise_floor: float = 1.0e-12
    objective_tolerance: float = 2.0e-10

    def __post_init__(self) -> None:
        if (
            isinstance(self.group_size, (bool, np.bool_))
            or not isinstance(self.group_size, (int, np.integer))
            or int(self.group_size) < 1
            or not np.isfinite(self.em_fraction)
            or not 0.0 <= self.em_fraction <= 1.0
        ):
            raise ValueError("invalid group size or EM fraction")
        if (
            not np.isfinite(self.gamma_floor)
            or not np.isfinite(self.noise_floor)
            or self.gamma_floor <= 0.0
            or self.noise_floor <= 0.0
        ):
            raise ValueError("variance floors must be positive")
        if (
            not np.isfinite(self.objective_tolerance)
            or self.objective_tolerance != 2.0e-10
        ):
            raise ValueError("RUST objective tolerance is frozen at 2e-10")


@dataclass(frozen=True, slots=True)
class RUSTSBLResult:
    method: str
    mean: ComplexArray
    covariance: ComplexArray
    gamma: FloatArray
    noise: FloatArray
    frequency_weights: FloatArray
    objective: tuple[float, ...]
    anchor_scales: tuple[float, ...]
    learned_scales: tuple[float, ...]
    fallback_reason: str | None
    runtime_seconds: float
    metadata: Mapping[str, object]


def _champagne_fallback(
    operator: ComplexArray,
    observations: ComplexArray,
    frequency_weights: ArrayLike,
    *,
    group_size: int,
) -> tuple[ComplexArray, ComplexArray, FloatArray, FloatArray, tuple[float, ...]]:
    fitted = fit_multifrequency_group_sbl(
        operator,
        observations,
        frequency_weights,
        config=MultiFrequencySBLConfig(
            iterations=40,
            group_size=group_size,
            update_rule="em",
        ),
    )
    return (
        fitted.means,
        fitted.covariances,
        fitted.gamma,
        fitted.noise,
        tuple(fitted.objective),
    )


def fit_rust_sbl(
    leadfields: ArrayLike,
    observations: ArrayLike,
    frequency_weights: ArrayLike,
    graph: sparse.spmatrix,
    weights: RUSTSBLWeights,
    *,
    depth_feature: ArrayLike | None = None,
    config: RUSTSBLConfig | None = None,
    gamma0: ArrayLike | None = None,
    noise0: ArrayLike | float | None = None,
    initial_posterior: MultiFrequencyPosterior | None = None,
    completed_pilot_iterations: int = 0,
    fallback_solver: Callable[..., tuple[ComplexArray, ComplexArray, FloatArray, FloatArray, tuple[float, ...]]] | None = None,
) -> RUSTSBLResult:
    """Run deterministic frozen RUST-SBL inference."""

    started = perf_counter()
    settings = config or RUSTSBLConfig()
    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if data.ndim == 2:
        data = data[:, :, None]
    if (
        operators.ndim != 3
        or data.ndim != 3
        or data.shape[:2] != operators.shape[:2]
        or operators.shape[2] % settings.group_size
        or not np.all(np.isfinite(operators))
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("leadfields and observations must form a finite grouped frequency problem")
    frequency, omega, available = effective_frequency_weights(
        frequency_weights, operators.shape[0]
    )
    groups = operators.shape[2] // settings.group_size
    matrix = sparse.csr_matrix(graph, dtype=np.float64)
    if matrix.shape != (groups, groups):
        raise ValueError("graph must align with source groups")
    if np.max(np.abs(np.asarray(matrix.sum(axis=1)).reshape(-1) - 1.0)) > 1e-10:
        raise ValueError("graph must be row stochastic")
    depth = (
        np.zeros(groups, dtype=np.float64)
        if depth_feature is None
        else np.asarray(depth_feature, dtype=np.float64)
    )
    if depth.shape != (groups,) or not np.all(np.isfinite(depth)):
        raise ValueError("depth_feature must contain one finite public feature per group")
    feature_normalizer = weights.feature_normalizer
    frequency_power = np.mean(np.abs(data) ** 2, axis=(1, 2)).real
    power = max(
        float(np.dot(omega, frequency_power) / np.sum(omega)),
        settings.gamma_floor,
    )
    gamma = (
        np.full(groups, power, dtype=np.float64)
        if gamma0 is None
        else np.maximum(np.asarray(gamma0, dtype=np.float64), settings.gamma_floor)
    )
    if gamma.shape != (groups,) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma0 must contain one finite value per group")
    if noise0 is None:
        noise = np.maximum(0.1 * frequency_power, settings.noise_floor)
    else:
        raw_noise = np.asarray(noise0, dtype=np.float64)
        if raw_noise.ndim == 0:
            noise = np.full(operators.shape[0], float(raw_noise), dtype=np.float64)
        else:
            noise = raw_noise.reshape(-1)
        if (
            noise.shape != (operators.shape[0],)
            or not np.all(np.isfinite(noise))
            or np.any(noise <= 0.0)
        ):
            raise ValueError("noise0 must contain one positive value per frequency")
        noise = np.maximum(noise, settings.noise_floor)
    objective: list[float] = []
    anchor_scales: list[float] = []
    learned_scales: list[float] = []
    fallback_reason: str | None = None
    completed = int(completed_pilot_iterations)
    if completed not in {0, 2}:
        raise ValueError("completed pilot iterations must be zero or two")
    if completed and initial_posterior is None:
        raise ValueError("pilot-completed RUST requires its posterior cache")
    current_posterior = initial_posterior

    try:
        initial_objective = (
            float(
                np.dot(
                    omega,
                    multifrequency_type2_terms_from_posterior(
                        data, current_posterior
                    ),
                )
            )
            if current_posterior is not None
            else multifrequency_type2_objective(
                operators,
                data,
                gamma,
                noise,
                omega=omega,
                group_size=settings.group_size,
            )
        )
        if not np.isfinite(initial_objective):
            raise FloatingPointError("non-finite initial Type-II objective")
        objective.append(float(initial_objective))
        for layer in range(weights.layers):
            gamma_current = gamma.copy()
            posterior = (
                current_posterior
                if current_posterior is not None
                else multifrequency_posterior(
                    operators, data, gamma, noise, group_size=settings.group_size
                )
            )
            current_posterior = None
            em = multifrequency_group_em_update(
                posterior.means,
                posterior.covariances,
                omega,
                group_size=settings.group_size,
                floor=settings.gamma_floor,
            )
            mu = multifrequency_group_mu_update(
                operators,
                data,
                gamma,
                noise,
                omega,
                group_size=settings.group_size,
                floor=settings.gamma_floor,
            )
            alpha = float(np.asarray(weights.layer_alpha)[layer])
            anchor_proposal = np.exp(
                (1.0 - alpha) * np.log(np.maximum(em, settings.gamma_floor))
                + alpha * np.log(np.maximum(mu, settings.gamma_floor))
            )
            raw_noise = multifrequency_noise_update(
                operators,
                data,
                posterior.means,
                posterior.covariances,
                floor=settings.noise_floor,
            )
            damping = float(weights.layer_noise_damping[layer])
            proposed_noise = np.exp(
                (1.0 - damping) * np.log(noise) + damping * np.log(raw_noise)
            )

            def objective_at(candidate: FloatArray) -> float:
                return multifrequency_type2_objective(
                    operators,
                    data,
                    candidate,
                    proposed_noise,
                    omega=omega,
                    group_size=settings.group_size,
                )

            anchor, anchor_scale, anchor_value = _log_backtrack(
                gamma,
                anchor_proposal,
                objective[-1],
                objective_at,
                tolerance=settings.objective_tolerance,
            )
            if anchor_scale == 0.0:
                proposed_noise = noise
            raw_features = _group_features(
                operators,
                data,
                gamma_current,
                em,
                em,
                mu,
                noise,
                matrix,
                depth,
                omega=omega,
                train_frequency_mismatch=weights.train_frequency_mismatch,
                layer=layer,
                layers=weights.layers,
                group_size=settings.group_size,
            )
            if feature_normalizer is None:
                # Old checkpoints remain loadable, but their head-wise adapter
                # is explicitly marked legacy and is never written by the new
                # library-only trainer.
                features = np.column_stack(
                    [
                        _legacy_standardize(raw_features[:, column])
                        for column in range(raw_features.shape[1])
                    ]
                )
            else:
                features = feature_normalizer.transform(raw_features)
            if features.shape[1] != weights.feature_count:
                raise ValueError("frozen feature count does not match the checkpoint")
            normalized_depth = features[:, 6]
            z_proposal = learned_log_gamma_proposal(
                np.log(np.maximum(anchor, settings.gamma_floor)),
                features,
                weights,
                layer=layer,
            )
            z_graph = multiscale_graph_mix(
                z_proposal, matrix, tuple(weights.layer_graph_logits[layer])
            )
            graph_damping = float(weights.layer_graph_damping[layer])
            z_candidate = strict_log_gamma_candidate(
                np.log(np.maximum(gamma_current, settings.gamma_floor)),
                z_graph,
                normalized_depth,
                damping=graph_damping,
                depth_gate=float(weights.layer_depth_scale[layer]),
            )
            structured = np.exp(np.clip(z_candidate, -60.0, 60.0))
            gamma, learned_scale, accepted_value = _log_backtrack(
                anchor,
                structured,
                anchor_value,
                objective_at,
                tolerance=settings.objective_tolerance,
            )
            noise = proposed_noise
            anchor_scales.append(float(anchor_scale))
            learned_scales.append(float(learned_scale))
            objective.append(float(accepted_value))
            if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(noise)):
                raise FloatingPointError("non-finite learned state")
        posterior = multifrequency_posterior(
            operators, data, gamma, noise, group_size=settings.group_size
        )
        mean, covariance = posterior.means, posterior.covariances
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise FloatingPointError("non-finite learned posterior")
    except (FloatingPointError, OverflowError, linalg.LinAlgError) as error:
        solver = fallback_solver or _champagne_fallback
        mean, covariance, gamma, noise, fallback_objective = solver(
            operators,
            data,
            frequency,
            group_size=settings.group_size,
        )
        objective = list(fallback_objective)
        fallback_reason = type(error).__name__
        anchor_scales = []
        learned_scales = []

    return RUSTSBLResult(
        method="V4-RUST-SBL-FreeSIGM",
        mean=np.asarray(mean, dtype=np.complex128),
        covariance=np.asarray(covariance, dtype=np.complex128),
        gamma=np.asarray(gamma, dtype=np.float64),
        noise=np.asarray(noise, dtype=np.float64),
        frequency_weights=np.asarray(frequency, dtype=np.float64),
        objective=tuple(float(value) for value in objective),
        anchor_scales=tuple(anchor_scales),
        learned_scales=tuple(learned_scales),
        fallback_reason=fallback_reason,
        runtime_seconds=float(perf_counter() - started),
        metadata={
            "parameter_count": weights.parameter_count,
            "shared_groupwise_mlp": True,
            "source_index_embedding": False,
            "truth_operator_input": False,
            "truth_or_emd_input": False,
            "layers": weights.layers,
            "backtracking_scales": [1.0, 0.5, 0.25],
            "full_fallback": fallback_reason is not None,
            "latent_source_contract": "independent_X_f_shared_group_gamma",
            "per_frequency_noise": True,
            "effective_frequency_weights": omega.tolist(),
            "available_frequency_count": int(np.count_nonzero(available)),
            "feature_normalization_contract": (
                "legacy_headwise_compatibility"
                if feature_normalizer is None
                else feature_normalizer.formula_version
            ),
            "feature_normalization_source_split": (
                None
                if feature_normalizer is None
                else feature_normalizer.source_split
            ),
            "feature_normalizer_provenance_sha256": (
                None
                if feature_normalizer is None
                else feature_normalizer.provenance_sha256
            ),
            "raw_depth_uses_frozen_feature_statistics": bool(
                feature_normalizer is not None
                and feature_normalizer.source_split == "library_train"
            ),
            "analytic_pilot_iterations": completed,
            "learned_unrolled_layer_budget": int(weights.layers),
            "pilot_iterations_reused": completed,
            "main_iterations_executed": int(weights.layers),
            "pilot_factorization_token": (
                None
                if initial_posterior is None
                else initial_posterior.factorization_token
            ),
            "rust_feature_contract": RUST_RAW_FEATURE_CONTRACT_V4,
            "feature_names": list(RUST_FEATURE_NAMES_V4),
            "strict_candidate_formula": (
                "(1-d)*log_gamma_current+d*z_graph+p_depth*depth"
            ),
            "source_instability_formula": RUST_SOURCE_INSTABILITY_FORMULA_V4,
            "source_instability_provenance_sha256": (
                weights.source_instability_provenance_sha256
            ),
        },
    )


try:  # Torch is training-only; frozen production inference above is NumPy.
    import torch
    from torch import nn

    class SharedResidualMLP(nn.Module):
        def __init__(self, feature_count: int = 12, hidden_count: int = 8) -> None:
            super().__init__()
            if not 10 <= feature_count <= 16 or hidden_count not in {8, 16}:
                raise ValueError("architecture is outside the preregistered V4 set")
            self.linear1 = nn.Linear(feature_count, hidden_count)
            self.linear2 = nn.Linear(hidden_count, 1)

        def forward(self, features):
            return torch.tanh(self.linear2(torch.tanh(self.linear1(features))))[..., 0]

except ImportError:  # pragma: no cover - project dependency includes torch
    SharedResidualMLP = None  # type: ignore[assignment]


__all__ = [
    "RUST_FEATURE_CLIP_V4",
    "RUST_FEATURE_NAMES_V4",
    "RUST_FEATURE_NORMALIZER_VERSION",
    "RUST_RAW_FEATURE_CONTRACT_V4",
    "RUST_SOURCE_INSTABILITY_FORMULA_V4",
    "RUSTFeatureNormalizer",
    "RUSTSBLConfig",
    "RUSTSBLResult",
    "RUSTSBLWeights",
    "SharedResidualMLP",
    "build_rust_group_features",
    "fit_rust_sbl",
    "fit_rust_feature_normalizer",
    "initialize_rust_weights",
    "learned_log_gamma_proposal",
    "shared_mlp_residual",
    "source_instability_from_operator",
    "strict_log_gamma_candidate",
]
