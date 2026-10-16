"""Public V7 FSOST entry points, including the exact Champagne submodel."""

from __future__ import annotations

from numpy.typing import ArrayLike

from .complex_sbl import ComplexGroupSBL, SBLResult
from .v7_covariance_components import CovarianceComponent
from .v7_em_mm_solver import EMResult, fixed_active_em


def fit_point_only_champagne(
    leadfield: ArrayLike,
    observations: ArrayLike,
    *,
    group_size: int = 3,
    max_iter: int = 100,
    tol: float = 1e-6,
    gamma0: ArrayLike | None = None,
    noise0: float | None = None,
) -> SBLResult:
    """Exact nested submodel: delegate to the repository Champagne solver."""

    return ComplexGroupSBL(group_size=group_size,max_iter=max_iter,tol=tol).fit(
        leadfield, observations, gamma0=gamma0, noise0=noise0
    )


def fit_fixed_fsost(
    observations,
    components: list[CovarianceComponent],
    **kwargs,
) -> EMResult:
    return fixed_active_em(observations,components,**kwargs)


__all__=["fit_point_only_champagne","fit_fixed_fsost"]
