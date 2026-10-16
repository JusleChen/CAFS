"""Bilevel physical calibration and identifiability tools."""

from .identifiability import effective_information, schur_complement, stabilize_information
from .solver import BilevelCalibrationResult, calibrate_affine_leadfield_family
from .v2_nonlinear import (
    FoldFrequencyPosterior,
    IdentifiableSubspace,
    NonlinearCalibrationResult,
    PredictiveNLPDComponents,
    TrustRegionDecision,
    calibrate_nonlinear_multi_episode,
    heldout_proper_complex_nlpd,
    nonlinear_multi_episode_calibration,
    prior_whitened_identifiable_subspace,
    proper_complex_predictive_nlpd,
    select_prior_whitened_identifiable_subspace,
    trust_region_actual_predicted_ratio,
    update_trust_region,
)

__all__ = [
    "BilevelCalibrationResult",
    "FoldFrequencyPosterior",
    "IdentifiableSubspace",
    "NonlinearCalibrationResult",
    "PredictiveNLPDComponents",
    "TrustRegionDecision",
    "calibrate_affine_leadfield_family",
    "calibrate_nonlinear_multi_episode",
    "effective_information",
    "heldout_proper_complex_nlpd",
    "nonlinear_multi_episode_calibration",
    "prior_whitened_identifiable_subspace",
    "proper_complex_predictive_nlpd",
    "schur_complement",
    "select_prior_whitened_identifiable_subspace",
    "stabilize_information",
    "trust_region_actual_predicted_ratio",
    "update_trust_region",
]
