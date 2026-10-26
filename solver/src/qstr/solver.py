"""Structured, bounded derivative-free optimization with paid observations."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coordinates import anchored_box_map
from .options import SolverOptions
from .result import (
    EvaluationHistory,
    ObservationGroup,
    OptimizationResult,
    RestartDecision,
    StageResult,
)

Objective = Callable[[NDArray[np.float64]], float]
BasisFactory = Callable[[int, int], ArrayLike | None]


class _StageFinished(Exception):
    """Stop a local solve without consuming another objective evaluation."""


def minimize(
    objective: Objective,
    x0: ArrayLike,
    max_evaluations: int,
    *,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    basis_factory: BasisFactory | None = None,
    seed: int = 0,
    options: SolverOptions | Mapping[str, Any] | None = None,
) -> OptimizationResult:
    """Minimize an expensive scalar objective using structured local models.

    Parameters
    ----------
    objective
        Callable taking one finite parameter vector and returning a finite
        scalar. Each call counts against the evaluation budget. Independent
        measurement noise may be generated inside this callable.
    x0
        Initial parameter vector. It must satisfy the supplied bounds.
    max_evaluations
        Maximum total objective calls, including the initial observations,
        repetitions, and all restarts.
    bounds
        Pair ``(lower, upper)`` of finite vectors with ``lower < upper``.
        Native-coordinate stages permit boundary points. Structured stages
        require an interior center and use a smooth map into the open box.
    basis_factory
        Callable ``basis_factory(stage_index, stage_seed)`` returning a matrix
        with orthonormal columns, or ``None`` for native coordinates. It is
        called once at each attempted stage. The stage seed is
        ``seed + 173 + 1009 * stage_index``. An exact identity matrix selects
        native coordinates; other square matrices still transform coordinates.
    seed
        Seed used for the basis schedule. The objective controls its own
        randomness; this function does not seed a global random generator.
    options
        SolverOptions instance or mapping using its field names.

    Returns
    -------
    OptimizationResult
        The point with the minimum observed group mean, together with all
        paid observations and stage records. The final incomplete repetition
        group, if any, also participates in this selection.

    Notes
    -----
    Each stage uses COBYQA to maintain a quadratic interpolation model. A
    restart retains the best observation group and constructs a new model.
    The budget reserve can extend an active stage to avoid an underfunded
    restart; it does not transfer model coefficients between solves.
    """
    from cobyqa import minimize as minimize_trust_region

    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or x0.size == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a nonempty finite vector")
    if (
        not np.isfinite(max_evaluations)
        or int(max_evaluations) != max_evaluations
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    max_evaluations = int(max_evaluations)
    dimension = x0.size
    bounded = bounds is not None
    lower = upper = None
    if bounded:
        if len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair")
        lower, upper = (np.asarray(bound, dtype=float) for bound in bounds)
        if (
            lower.shape != x0.shape
            or upper.shape != x0.shape
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
            or np.any(x0 < lower)
            or np.any(x0 > upper)
        ):
            raise ValueError("Require finite lower < upper and lower <= x0 <= upper")
    if options is None:
        settings = SolverOptions()
    elif isinstance(options, SolverOptions):
        settings = options
    elif isinstance(options, Mapping):
        settings = SolverOptions.from_mapping(options)
    else:
        raise TypeError("options must be SolverOptions, a mapping, or None")

    history_points: list[list[float]] = []
    history_values: list[float] = []
    evaluation_times: list[float] = []
    observation_groups: list[ObservationGroup] = []
    stages: list[StageResult] = []
    restart_decisions: list[RestartDecision] = []
    start_time = time.perf_counter()
    repetitions = int(settings.repetitions)

    def evaluate(point: ArrayLike, evaluation_limit: int) -> float:
        available_limit = min(evaluation_limit, max_evaluations)
        if len(history_values) >= available_limit:
            raise _StageFinished()
        count = min(repetitions, available_limit - len(history_values))
        point = np.asarray(point, dtype=float)
        if point.shape != x0.shape or not np.all(np.isfinite(point)):
            raise ValueError("Objective proposal must be a finite vector matching x0")
        if bounded and (np.any(point < lower - 1e-12) or np.any(point > upper + 1e-12)):
            raise ValueError("Objective proposal lies outside the supplied bounds")
        values: list[float] = []
        for _ in range(count):
            value = float(objective(point.copy()))
            if not np.isfinite(value):
                raise ValueError("Objective returned a nonfinite scalar")
            history_points.append(point.tolist())
            history_values.append(value)
            evaluation_times.append(time.perf_counter() - start_time)
            values.append(value)
        mean = float(np.mean(values))
        observation_groups.append(
            ObservationGroup(
                end_evaluation=len(history_values),
                x=point.copy(),
                mean=mean,
                repetitions=count,
            )
        )
        return mean

    initial_value = evaluate(x0, max_evaluations)
    stage_start_values: list[float] = []
    for stage_index in range(int(settings.max_stages)):
        if len(history_values) >= max_evaluations:
            break
        incumbent = min(observation_groups, key=lambda group: group.mean)
        center = incumbent.x.copy()
        stage_start_values.append(incumbent.mean)
        stage_seed = int(seed) + 173 + 1009 * stage_index
        basis = None if basis_factory is None else basis_factory(stage_index, stage_seed)
        if basis is not None:
            basis = np.asarray(basis, dtype=float)
            if (
                basis.ndim != 2
                or basis.shape[0] != dimension
                or not 1 <= basis.shape[1] <= dimension
                or not np.all(np.isfinite(basis))
                or not np.allclose(
                    basis.T @ basis,
                    np.eye(basis.shape[1]),
                    atol=1e-10,
                    rtol=1e-10,
                )
            ):
                raise ValueError("A structure basis must have orthonormal columns")
            if basis.shape == (dimension, dimension) and np.array_equal(basis, np.eye(dimension)):
                basis = None

        local_dimension = dimension if basis is None else basis.shape[1]
        if basis is None:
            local_start = center.copy()

            def to_parameters(coordinates: ArrayLike) -> NDArray[np.float64]:
                return np.asarray(coordinates, dtype=float)

            local_bounds = list(zip(lower, upper)) if bounded else None
            coordinate_map = "native"
        else:
            local_start = np.zeros(local_dimension)
            local_bounds = None
            if bounded:
                box_map = anchored_box_map(center, lower, upper)

                def to_parameters(coordinates: ArrayLike) -> NDArray[np.float64]:
                    displacement = basis @ np.asarray(coordinates, dtype=float)
                    return box_map(displacement)

                coordinate_map = "anchored_tanh_subspace"
            else:

                def to_parameters(coordinates: ArrayLike) -> NDArray[np.float64]:
                    return center + basis @ np.asarray(coordinates, dtype=float)

                coordinate_map = "linear_subspace"

        in_warmup = stage_index < int(settings.warmup_stages)
        initial_radius = float(
            settings.warmup_radius
            if in_warmup and settings.warmup_radius is not None
            else settings.initial_radius
        )
        reference_index = int(settings.progress_reference_stage)
        progress_reference = (
            initial_value
            if reference_index < 0
            else stage_start_values[min(stage_index, reference_index)]
        )
        exponent = float(settings.progress_exponent) * (incumbent.mean - progress_reference)
        initial_radius *= float(np.exp(np.clip(exponent, -700, 0)))
        initial_radius = max(float(settings.minimum_radius), initial_radius)
        final_radius = min(float(settings.final_radius), 0.2 * initial_radius)
        interpolation_factor = (
            settings.warmup_interpolation_factor
            if in_warmup and settings.warmup_interpolation_factor is not None
            else settings.interpolation_factor
        )
        interpolation_points = max(
            local_dimension + 1,
            min(
                (local_dimension + 1) * (local_dimension + 2) // 2,
                int(float(interpolation_factor) * local_dimension + 1),
            ),
        )
        stage_budget = int(
            settings.warmup_evaluations
            if in_warmup and settings.warmup_evaluations is not None
            else (
                max_evaluations
                if settings.stage_evaluations is None
                else settings.stage_evaluations
            )
        )
        stage_begin = len(history_values)
        evaluation_limit = min(max_evaluations, stage_begin + stage_budget)
        reserve = max(
            int(settings.min_restart_evaluations),
            int(
                np.ceil(float(settings.restart_reserve_factor) * interpolation_points * repetitions)
            ),
        )
        if stage_index > 0 and max_evaluations - stage_begin < reserve:
            restart_decisions.append(
                RestartDecision(
                    stage=stage_index,
                    action="stop_insufficient_budget",
                    reserve=reserve,
                    remaining_evaluations=max_evaluations - stage_begin,
                )
            )
            break
        requested_limit = evaluation_limit
        # Predict the reserve from this stage's dimension. Calling the basis
        # factory speculatively would change stateful basis schedules.
        if 0 < max_evaluations - evaluation_limit < reserve:
            evaluation_limit = max_evaluations
            restart_decisions.append(
                RestartDecision(
                    stage=stage_index,
                    action="extend_active_stage",
                    reserve=reserve,
                    planned_evaluations=requested_limit - stage_begin,
                    stage_evaluations=evaluation_limit - stage_begin,
                    avoided_tail_evaluations=max_evaluations - requested_limit,
                )
            )

        last_improvement = 0
        improvement_reference = None
        stop_reason = "solver_termination"
        model_evaluations = 0

        def local_objective(coordinates: NDArray[np.float64]) -> float:
            nonlocal last_improvement, improvement_reference, stop_reason, model_evaluations
            local_evaluations = len(history_values) - stage_begin
            minimum_stage_length = max(
                interpolation_points * repetitions + repetitions,
                int(settings.min_stage_evaluations),
            )
            if local_evaluations >= minimum_stage_length and settings.stagnation_patience:
                if (
                    local_evaluations - last_improvement >= int(settings.stagnation_patience)
                    and max_evaluations - len(history_values) > interpolation_points * repetitions
                    and max_evaluations - len(history_values) >= reserve
                ):
                    stop_reason = "stagnation"
                    raise _StageFinished()
            if len(history_values) >= evaluation_limit:
                stop_reason = "evaluation_budget"
                raise _StageFinished()
            value = evaluate(to_parameters(coordinates), evaluation_limit)
            model_evaluations += 1
            if improvement_reference is None or value < improvement_reference - float(
                settings.improvement_tolerance
            ):
                improvement_reference = value
                last_improvement = len(history_values) - stage_begin
            return value

        solver_recommendation = None
        solver_message = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                local_result = minimize_trust_region(
                    local_objective,
                    local_start,
                    bounds=local_bounds,
                    options={
                        "maxfev": max(
                            1, (evaluation_limit - stage_begin + repetitions - 1) // repetitions
                        ),
                        "maxiter": max(
                            1000 * local_dimension, 10 * (evaluation_limit - stage_begin)
                        ),
                        "radius_init": initial_radius,
                        "radius_final": final_radius,
                        "nb_points": interpolation_points,
                        "scale": False,
                        "disp": False,
                        "debug": False,
                    },
                )
                solver_recommendation = to_parameters(local_result.x).copy()
                solver_message = str(local_result.message)
            except _StageFinished:
                pass
            warning_messages = tuple(str(warning.message) for warning in caught)
        used_evaluations = len(history_values) - stage_begin
        if used_evaluations == 0:
            evaluate(center, evaluation_limit)
            used_evaluations = len(history_values) - stage_begin
            stop_reason = "fallback_evaluation"
        stages.append(
            StageResult(
                index=stage_index,
                start_evaluation=stage_begin + 1,
                end_evaluation=len(history_values),
                evaluations=used_evaluations,
                dimension=local_dimension,
                interpolation_points=interpolation_points,
                initialization_evaluations=min(
                    used_evaluations, interpolation_points * repetitions
                ),
                initial_radius=initial_radius,
                final_radius=final_radius,
                coordinate_map=coordinate_map,
                basis_seed=stage_seed,
                stop_reason=stop_reason,
                solver_message=solver_message,
                solver_recommendation=solver_recommendation,
                warnings=warning_messages,
                model_evaluations=model_evaluations,
                requested_evaluation_limit=requested_limit,
                evaluation_limit=evaluation_limit,
                restart_reserve=reserve,
            )
        )

    incumbent = min(observation_groups, key=lambda group: group.mean)
    return OptimizationResult(
        x=incumbent.x.copy(),
        fun=incumbent.mean,
        nfev=len(history_values),
        max_evaluations=max_evaluations,
        history=EvaluationHistory(
            x=np.asarray(history_points, dtype=float),
            fun=np.asarray(history_values, dtype=float),
            elapsed_seconds=np.asarray(evaluation_times, dtype=float),
        ),
        observation_groups=tuple(observation_groups),
        stages=tuple(stages),
        options=settings,
        elapsed_seconds=time.perf_counter() - start_time,
        seed=int(seed),
        restart_decisions=tuple(restart_decisions),
    )
