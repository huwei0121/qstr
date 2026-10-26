"""Optimization results and complete paid-observation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .options import SolverOptions


@dataclass(slots=True)
class EvaluationHistory:
    """One row per objective call, in evaluation order."""

    x: NDArray[np.float64]
    fun: NDArray[np.float64]
    elapsed_seconds: NDArray[np.float64]


@dataclass(slots=True)
class ObservationGroup:
    """Repeated measurements at one point, averaged for model feedback.

    ``end_evaluation`` is a one-based cumulative objective-call count.
    A final group may contain fewer than the requested repetitions when
    the stage or total budget is exhausted. A later revisit starts a new group.
    """

    end_evaluation: int
    x: NDArray[np.float64]
    mean: float
    repetitions: int


@dataclass(slots=True)
class StageResult:
    """Settings and outcome of one local trust-region solve.

    Evaluation indices are one-based and inclusive. ``model_evaluations``
    counts averaged feedback calls; ``evaluations`` counts individual
    objective calls. ``initialization_evaluations`` records the first
    interpolation-point count of calls, capped by the stage's actual length.
    """

    index: int
    start_evaluation: int
    end_evaluation: int
    evaluations: int
    dimension: int
    interpolation_points: int
    initialization_evaluations: int
    initial_radius: float
    final_radius: float
    coordinate_map: str
    basis_seed: int
    stop_reason: str
    solver_message: str | None
    solver_recommendation: NDArray[np.float64] | None
    warnings: tuple[str, ...]
    model_evaluations: int
    requested_evaluation_limit: int
    evaluation_limit: int
    restart_reserve: int


@dataclass(slots=True)
class RestartDecision:
    """A budget decision affecting the next or active local solve."""

    stage: int
    action: str
    reserve: int
    remaining_evaluations: int | None = None
    planned_evaluations: int | None = None
    stage_evaluations: int | None = None
    avoided_tail_evaluations: int | None = None


@dataclass(slots=True)
class OptimizationResult:
    """Recommendation and reproducible observation history.

    ``x`` and ``fun`` identify the minimum observed group mean across all
    stages. They are measured values, not an estimate from external validation.
    ``metadata`` is available for problem-specific reporting by adapters.
    """

    x: NDArray[np.float64]
    fun: float
    nfev: int
    max_evaluations: int
    history: EvaluationHistory
    observation_groups: tuple[ObservationGroup, ...]
    stages: tuple[StageResult, ...]
    options: SolverOptions
    elapsed_seconds: float
    seed: int
    restart_decisions: tuple[RestartDecision, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return independent records using JSON-compatible scalar/list types."""

        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))
