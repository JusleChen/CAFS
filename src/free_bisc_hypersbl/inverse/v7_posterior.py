"""Unified latent-component posterior and second-moment source power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .v7_covariance_components import CovarianceComponent
from .v7_em_mm_solver import EMResult


@dataclass(frozen=True)
class SourcePosterior:
    means: tuple[NDArray[np.complex128] | None, ...]
    covariances: tuple[NDArray[np.complex128] | None, ...]
    second_moment_power: NDArray[np.float64]
    mean_only_power: NDArray[np.float64]


def source_posterior(
    observations: Sequence[ArrayLike | None],
    components: Sequence[CovarianceComponent],
    result: EMResult,
    *,
    group_size: int = 3,
) -> SourcePosterior:
    source_dimension = components[0].source_basis.shape[0]
    mean_power = np.zeros(source_dimension)
    second_power = np.zeros(source_dimension)
    means=[]; covariances=[]
    for f,moment in enumerate(result.moments):
        if moment is None:
            means.append(None); covariances.append(None); continue
        source_blocks=[np.sqrt(c.profile.rho[f])*c.source_basis for c in components]
        transform=np.column_stack(source_blocks)
        mean=transform@moment.mean
        covariance=transform@moment.covariance@transform.conj().T
        covariance=0.5*(covariance+covariance.conj().T)
        snapshots=mean.shape[1]
        mean_power += np.sum(np.abs(mean)**2,axis=1)
        second_power += np.sum(np.abs(mean)**2,axis=1)+snapshots*np.maximum(np.real(np.diag(covariance)),0)
        means.append(np.asarray(mean)); covariances.append(np.asarray(covariance))
    if source_dimension%group_size:
        raise ValueError("source dimension must be divisible by group_size")
    mean_group=mean_power.reshape(-1,group_size).sum(axis=1)
    second_group=second_power.reshape(-1,group_size).sum(axis=1)
    return SourcePosterior(tuple(means),tuple(covariances),second_group,mean_group)


__all__=["SourcePosterior","source_posterior"]
