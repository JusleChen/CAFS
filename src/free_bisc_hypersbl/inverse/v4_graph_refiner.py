"""Conditional V4 source-space graph refiner.

The refiner is a preregistered fallback, not an always-on component.  Its
learned surface contains only a shared graph filter and a tiny shared MLP; no
parameter is indexed by source.  The learned output is a clipped residual in
log-gamma space.  A proper-complex analytic posterior and one analytic noise
refit restore sensor-data consistency after the source-space update.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from free_bisc_hypersbl.frequency.v4_fast_freesigm import (
    FastSafeFreeSIGMResult,
    softmax_simplex,
)

from .v4_multifrequency_core import (
    effective_frequency_weights,
    multifrequency_noise_update,
    multifrequency_posterior,
    multifrequency_type2_objective,
)


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
GRAPH_REFINER_FEATURE_COUNT = 8
GRAPH_REFINER_FORMULA_VERSION = "v4_graph_refiner_source_physics_1"
GRAPH_REFINER_PARAMETER_LIMIT = 100_000


def _readonly(value: object, *, dtype: object = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _vector(
    value: object,
    *,
    name: str,
    size: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    if size is not None and result.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},)")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.asarray(result, dtype=np.float64)


def _row_stochastic_adjacency(value: object, *, size: int) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    if matrix.shape != (size, size):
        raise ValueError("adjacency must align with the source-group axis")
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
        raise ValueError("adjacency entries must be finite and nonnegative")
    matrix.eliminate_zeros()
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sum, 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("adjacency must be row-stochastic")
    matrix.sort_indices()
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class GraphRefinerPhysicsInputs:
    """Only source-space maps permitted to enter the learned refiner."""

    champagne_gamma: FloatArray
    champagne_power: FloatArray
    champagne_uncertainty: FloatArray
    adjacency: sparse.csr_matrix
    public_depth: FloatArray
    eloreta_power: FloatArray | None = None
    backprojection_power: FloatArray | None = None

    def __post_init__(self) -> None:
        gamma = _vector(
            self.champagne_gamma, name="champagne_gamma", positive=True
        )
        count = gamma.size
        power = _vector(
            self.champagne_power,
            name="champagne_power",
            size=count,
            nonnegative=True,
        )
        uncertainty = _vector(
            self.champagne_uncertainty,
            name="champagne_uncertainty",
            size=count,
            nonnegative=True,
        )
        depth = _vector(
            self.public_depth, name="public_depth", size=count, positive=True
        )
        optional: dict[str, np.ndarray | None] = {}
        for name in ("eloreta_power", "backprojection_power"):
            raw = getattr(self, name)
            optional[name] = (
                None
                if raw is None
                else _vector(raw, name=name, size=count, nonnegative=True)
            )
        object.__setattr__(self, "champagne_gamma", _readonly(gamma))
        object.__setattr__(self, "champagne_power", _readonly(power))
        object.__setattr__(self, "champagne_uncertainty", _readonly(uncertainty))
        object.__setattr__(self, "public_depth", _readonly(depth))
        object.__setattr__(
            self,
            "adjacency",
            _row_stochastic_adjacency(self.adjacency, size=count),
        )
        for name, value in optional.items():
            object.__setattr__(
                self, name, None if value is None else _readonly(value)
            )

    @property
    def group_count(self) -> int:
        return int(self.champagne_gamma.size)


@dataclass(frozen=True, slots=True)
class GraphRefinerConfig:
    blocks: Literal[2, 3] = 2
    hidden_features: Literal[8, 16, 32] = 16
    delta_log_gamma_clip: float = 1.0
    group_size: int = 3
    variance_floor: float = 1.0e-12
    formula_version: Literal["v4_graph_refiner_source_physics_1"] = (
        GRAPH_REFINER_FORMULA_VERSION
    )

    def __post_init__(self) -> None:
        if self.blocks not in {2, 3}:
            raise ValueError("graph refiner must contain two or three residual blocks")
        if self.hidden_features not in {8, 16, 32}:
            raise ValueError("graph refiner hidden width is outside the frozen set")
        if not isfinite(float(self.delta_log_gamma_clip)) or self.delta_log_gamma_clip <= 0.0:
            raise ValueError("delta_log_gamma_clip must be finite and positive")
        if int(self.group_size) < 1:
            raise ValueError("group_size must be positive")
        if not isfinite(float(self.variance_floor)) or self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and positive")
        if self.formula_version != GRAPH_REFINER_FORMULA_VERSION:
            raise ValueError("graph refiner formula version changed")


@dataclass(frozen=True, slots=True)
class GraphRefinerParameters:
    """Library-only, source-count-independent shared parameters."""

    input_weight: FloatArray
    input_bias: FloatArray
    output_weight: FloatArray
    output_bias: float
    block_graph_logits: FloatArray
    block_residual_scale: FloatArray
    source_split: Literal["library_train"] = "library_train"

    def __post_init__(self) -> None:
        w1 = np.asarray(self.input_weight, dtype=np.float64)
        b1 = np.asarray(self.input_bias, dtype=np.float64)
        w2 = np.asarray(self.output_weight, dtype=np.float64)
        logits = np.asarray(self.block_graph_logits, dtype=np.float64)
        scale = np.asarray(self.block_residual_scale, dtype=np.float64)
        if self.source_split != "library_train":
            raise ValueError("graph-refiner parameters may be fitted on library_train only")
        if (
            w1.ndim != 2
            or w1.shape[0] != GRAPH_REFINER_FEATURE_COUNT
            or w1.shape[1] not in {8, 16, 32}
            or b1.shape != (w1.shape[1],)
            or w2.shape != (w1.shape[1],)
        ):
            raise ValueError("shared MLP parameter shapes are outside the frozen surface")
        if logits.ndim != 2 or logits.shape[0] not in {2, 3} or logits.shape[1] != 3:
            raise ValueError("each residual block requires 0/1/2-hop graph logits")
        if scale.shape != (logits.shape[0],) or np.any(scale < 0.0) or np.any(scale > 1.0):
            raise ValueError("block residual scales must align and lie in [0,1]")
        arrays = (w1, b1, w2, logits, scale)
        if any(not np.all(np.isfinite(item)) for item in arrays) or not isfinite(
            float(self.output_bias)
        ):
            raise ValueError("graph-refiner parameters must be finite")
        parameter_count = int(sum(item.size for item in arrays) + 1)
        if parameter_count >= GRAPH_REFINER_PARAMETER_LIMIT:
            raise ValueError("graph-refiner parameter count must remain below 100k")
        object.__setattr__(self, "input_weight", _readonly(w1))
        object.__setattr__(self, "input_bias", _readonly(b1))
        object.__setattr__(self, "output_weight", _readonly(w2))
        object.__setattr__(self, "block_graph_logits", _readonly(logits))
        object.__setattr__(self, "block_residual_scale", _readonly(scale))

    @property
    def blocks(self) -> int:
        return int(self.block_graph_logits.shape[0])

    @property
    def hidden_features(self) -> int:
        return int(self.input_weight.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(
            self.input_weight.size
            + self.input_bias.size
            + self.output_weight.size
            + 1
            + self.block_graph_logits.size
            + self.block_residual_scale.size
        )

    @property
    def contains_source_embedding(self) -> bool:
        return False


def initialize_graph_refiner_parameters(
    seed: int,
    *,
    config: GraphRefinerConfig = GraphRefinerConfig(),
    source_split: str = "library_train",
    zero_residual: bool = False,
) -> GraphRefinerParameters:
    """Initialize the frozen small architecture for library-only training."""

    if source_split != "library_train":
        raise ValueError("graph-refiner initialization is library_train-only")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    weight_scale = 0.0 if zero_residual else 0.05
    return GraphRefinerParameters(
        input_weight=rng.normal(
            scale=weight_scale,
            size=(GRAPH_REFINER_FEATURE_COUNT, config.hidden_features),
        ),
        input_bias=np.zeros(config.hidden_features, dtype=np.float64),
        output_weight=rng.normal(scale=weight_scale, size=config.hidden_features),
        output_bias=0.0,
        block_graph_logits=np.tile(
            np.array([1.5, 0.5, -0.5], dtype=np.float64),
            (config.blocks, 1),
        ),
        block_residual_scale=np.full(config.blocks, 0.25, dtype=np.float64),
        source_split="library_train",
    )


def _robust_standardize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    center = float(np.median(vector))
    mad = 1.4826 * float(np.median(np.abs(vector - center)))
    scale = max(mad, float(np.std(vector)), 1.0e-8)
    return np.clip((vector - center) / scale, -8.0, 8.0)


def _robust_log_map(value: np.ndarray, floor: float) -> np.ndarray:
    logged = np.log(np.maximum(np.asarray(value, dtype=np.float64), float(floor)))
    return _robust_standardize(logged)


def _physics_features(
    inputs: GraphRefinerPhysicsInputs,
    current_log_gamma: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    eloreta_present = inputs.eloreta_power is not None
    backprojection_present = inputs.backprojection_power is not None
    eloreta = (
        np.zeros(inputs.group_count)
        if inputs.eloreta_power is None
        else _robust_log_map(inputs.eloreta_power, floor)
    )
    backprojection = (
        np.zeros(inputs.group_count)
        if inputs.backprojection_power is None
        else _robust_log_map(inputs.backprojection_power, floor)
    )
    return np.column_stack(
        [
            _robust_standardize(current_log_gamma),
            _robust_log_map(inputs.champagne_power, floor),
            _robust_log_map(inputs.champagne_uncertainty, floor),
            _robust_log_map(inputs.public_depth, floor),
            eloreta,
            np.full(inputs.group_count, float(eloreta_present)),
            backprojection,
            np.full(inputs.group_count, float(backprojection_present)),
        ]
    )


@dataclass(frozen=True, slots=True)
class GraphRefinement:
    delta_log_gamma: FloatArray
    gamma_refined: FloatArray
    block_delta_log_gamma: FloatArray
    parameter_count: int
    formula_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_log_gamma", _readonly(self.delta_log_gamma))
        object.__setattr__(self, "gamma_refined", _readonly(self.gamma_refined))
        object.__setattr__(
            self, "block_delta_log_gamma", _readonly(self.block_delta_log_gamma)
        )


def refine_log_gamma(
    inputs: GraphRefinerPhysicsInputs,
    *,
    parameters: GraphRefinerParameters,
    config: GraphRefinerConfig = GraphRefinerConfig(),
) -> GraphRefinement:
    """Run two or three shared residual graph blocks in source space."""

    if not isinstance(inputs, GraphRefinerPhysicsInputs):
        raise TypeError("inputs must be GraphRefinerPhysicsInputs")
    if not isinstance(parameters, GraphRefinerParameters):
        raise TypeError("parameters must be GraphRefinerParameters")
    if parameters.blocks != config.blocks or parameters.hidden_features != config.hidden_features:
        raise ValueError("parameters do not match the frozen graph-refiner config")
    base_log_gamma = np.log(np.asarray(inputs.champagne_gamma, dtype=np.float64))
    delta = np.zeros(inputs.group_count, dtype=np.float64)
    history: list[np.ndarray] = []
    adjacency = inputs.adjacency
    for block in range(config.blocks):
        features = _physics_features(
            inputs,
            base_log_gamma + delta,
            floor=config.variance_floor,
        )
        one_hop = np.asarray(adjacency @ features)
        two_hop = np.asarray(adjacency @ one_hop)
        mix = softmax_simplex(
            parameters.block_graph_logits[block], temperature=1.0
        )
        filtered = mix[0] * features + mix[1] * one_hop + mix[2] * two_hop
        hidden = np.tanh(
            filtered @ parameters.input_weight + parameters.input_bias
        )
        shared_residual = np.tanh(
            hidden @ parameters.output_weight + float(parameters.output_bias)
        )
        delta = np.clip(
            delta + parameters.block_residual_scale[block] * shared_residual,
            -config.delta_log_gamma_clip,
            config.delta_log_gamma_clip,
        )
        history.append(np.array(delta, copy=True))
    refined = np.asarray(inputs.champagne_gamma) * np.exp(delta)
    if not np.all(np.isfinite(refined)) or np.any(refined <= 0.0):
        raise FloatingPointError("refined gamma is nonfinite or nonpositive")
    return GraphRefinement(
        delta_log_gamma=delta,
        gamma_refined=refined,
        block_delta_log_gamma=np.stack(history),
        parameter_count=parameters.parameter_count,
        formula_version=config.formula_version,
    )


def _stack_available_frequency_problem(
    leadfields: ArrayLike,
    observations: ArrayLike,
    frequency_weights: ArrayLike,
) -> tuple[ComplexArray, ComplexArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    operators = np.asarray(leadfields, dtype=np.complex128)
    data = np.asarray(observations, dtype=np.complex128)
    if operators.ndim != 3:
        raise ValueError("leadfields must have shape (frequency,sensor,coefficient)")
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or data.shape[:2] != operators.shape[:2]:
        raise ValueError("observations must align with leadfields")
    if not np.all(np.isfinite(operators)) or not np.all(np.isfinite(data)):
        raise ValueError("leadfields and observations must be finite")
    weights = np.asarray(frequency_weights, dtype=np.float64)
    if (
        weights.shape != (operators.shape[0],)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError("frequency_weights must be a finite nonnegative simplex")
    available = weights > 0.0
    normalized, effective, checked_available = effective_frequency_weights(
        weights, operators.shape[0]
    )
    if not np.array_equal(available, checked_available):
        raise AssertionError("available-frequency masks disagree")
    return (
        np.asarray(operators, dtype=np.complex128),
        np.asarray(data, dtype=np.complex128),
        normalized,
        effective,
        available,
    )


def _group_power(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    if mean.ndim == 2:
        mean = mean[None, :, :]
        covariance = covariance[None, :, :]
    if mean.ndim != 3 or covariance.shape != (
        mean.shape[0],
        mean.shape[1],
        mean.shape[1],
    ):
        raise ValueError("group power requires per-frequency posterior moments")
    groups = mean.shape[1] // group_size
    power = np.empty(groups, dtype=np.float64)
    for group in range(groups):
        sl = slice(group * group_size, (group + 1) * group_size)
        power[group] = float(
            np.sum(np.abs(mean[:, sl]) ** 2).real
            + mean.shape[2]
            * sum(
                np.trace(covariance[frequency, sl, sl]).real
                for frequency in range(mean.shape[0])
            )
        )
    return power


@dataclass(frozen=True, slots=True)
class GraphRefinedPosterior:
    control_name: str
    refinement: GraphRefinement
    posterior_mean: ComplexArray
    posterior_covariance: ComplexArray
    posterior_group_power: FloatArray
    noise_refined: FloatArray
    objective: float
    sensor_covariance: ComplexArray
    frequency_leadfields: ComplexArray
    frequency_observations: ComplexArray
    frequency_weights: FloatArray
    effective_frequency_weights: FloatArray
    available_mask: NDArray[np.bool_]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for name, dtype in (
            ("posterior_mean", np.complex128),
            ("posterior_covariance", np.complex128),
            ("posterior_group_power", np.float64),
            ("noise_refined", np.float64),
            ("sensor_covariance", np.complex128),
            ("frequency_leadfields", np.complex128),
            ("frequency_observations", np.complex128),
            ("frequency_weights", np.float64),
            ("effective_frequency_weights", np.float64),
            ("available_mask", bool),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def refine_graph_posterior(
    inputs: GraphRefinerPhysicsInputs,
    leadfields: ArrayLike,
    observations: ArrayLike,
    *,
    frequency_weights: ArrayLike,
    champagne_noise: ArrayLike,
    parameters: GraphRefinerParameters,
    config: GraphRefinerConfig = GraphRefinerConfig(),
    control_name: str = "full",
) -> GraphRefinedPosterior:
    """Refine gamma, analytically update noise, then recompute the posterior."""

    refinement = refine_log_gamma(inputs, parameters=parameters, config=config)
    operators, data, weights, effective, available = _stack_available_frequency_problem(
        leadfields, observations, frequency_weights
    )
    if operators.shape[2] != inputs.group_count * config.group_size:
        raise ValueError("leadfield source groups do not align with physics maps")
    raw_noise = np.asarray(champagne_noise, dtype=np.float64)
    if raw_noise.ndim == 0:
        noise = np.full(operators.shape[0], float(raw_noise), dtype=np.float64)
    else:
        noise = raw_noise.reshape(-1)
    if (
        noise.shape != (operators.shape[0],)
        or not np.all(np.isfinite(noise))
        or np.any(noise <= 0.0)
    ):
        raise ValueError("champagne_noise must contain one positive value per frequency")
    preliminary = multifrequency_posterior(
        operators,
        data,
        refinement.gamma_refined,
        noise,
        group_size=config.group_size,
    )
    noise_refined = multifrequency_noise_update(
        operators,
        data,
        preliminary.means,
        preliminary.covariances,
        floor=config.variance_floor,
    )
    posterior = multifrequency_posterior(
        operators,
        data,
        refinement.gamma_refined,
        noise_refined,
        group_size=config.group_size,
    )
    objective = multifrequency_type2_objective(
        operators,
        data,
        refinement.gamma_refined,
        noise_refined,
        omega=effective,
        group_size=config.group_size,
    )
    return GraphRefinedPosterior(
        control_name=str(control_name),
        refinement=refinement,
        posterior_mean=posterior.means,
        posterior_covariance=posterior.covariances,
        posterior_group_power=_group_power(
            posterior.means, posterior.covariances, group_size=config.group_size
        ),
        noise_refined=noise_refined,
        objective=float(objective),
        sensor_covariance=posterior.sensor_covariances,
        frequency_leadfields=operators,
        frequency_observations=data,
        frequency_weights=weights,
        effective_frequency_weights=effective,
        available_mask=available,
        metadata={
            "learned_input_domain": "source_space_physics_maps_only",
            "analytic_posterior_refit": True,
            "analytic_noise_refit": True,
            "source_embedding": False,
            "parameter_count": parameters.parameter_count,
            "fast_freesigm_effective_scale": "F_available_times_w_safe",
            "latent_source_contract": "independent_X_f_shared_group_gamma",
            "per_frequency_noise": True,
        },
    )


@dataclass(frozen=True, slots=True)
class GraphRefinerMatchedAblation:
    full: GraphRefinedPosterior
    exact_uniform: GraphRefinedPosterior


def execute_graph_refiner_freesigm_ablation(
    inputs: GraphRefinerPhysicsInputs,
    leadfields: ArrayLike,
    observations: ArrayLike,
    *,
    full_weighting: FastSafeFreeSIGMResult,
    exact_uniform_weighting: FastSafeFreeSIGMResult,
    champagne_noise: ArrayLike,
    parameters: GraphRefinerParameters,
    config: GraphRefinerConfig = GraphRefinerConfig(),
) -> GraphRefinerMatchedAblation:
    """Independently refit positive-rho full and exact-uniform arms."""

    if full_weighting.mode != "fast_safe_closed_form" or full_weighting.rho <= 0.0:
        raise ValueError("full graph-refiner arm requires positive-rho Fast-FreeSIGM")
    if (
        exact_uniform_weighting.mode != "exact_uniform_ablation"
        or exact_uniform_weighting.rho != 0.0
    ):
        raise ValueError("matched control must be the exact uniform arm")
    if not np.array_equal(
        full_weighting.available_mask, exact_uniform_weighting.available_mask
    ):
        raise ValueError("matched arms changed the available frequency family")
    full = refine_graph_posterior(
        inputs,
        leadfields,
        observations,
        frequency_weights=full_weighting.w_safe,
        champagne_noise=champagne_noise,
        parameters=parameters,
        config=config,
        control_name="full",
    )
    uniform = refine_graph_posterior(
        inputs,
        leadfields,
        observations,
        frequency_weights=exact_uniform_weighting.w_safe,
        champagne_noise=champagne_noise,
        parameters=parameters,
        config=config,
        control_name="exact_uniform",
    )
    return GraphRefinerMatchedAblation(full=full, exact_uniform=uniform)


__all__ = [
    "GRAPH_REFINER_FEATURE_COUNT",
    "GRAPH_REFINER_FORMULA_VERSION",
    "GRAPH_REFINER_PARAMETER_LIMIT",
    "GraphRefinedPosterior",
    "GraphRefinement",
    "GraphRefinerConfig",
    "GraphRefinerMatchedAblation",
    "GraphRefinerParameters",
    "GraphRefinerPhysicsInputs",
    "execute_graph_refiner_freesigm_ablation",
    "initialize_graph_refiner_parameters",
    "refine_graph_posterior",
    "refine_log_gamma",
]
