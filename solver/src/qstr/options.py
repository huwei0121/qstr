"""Numerical settings for structured trust-region optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Control local models, repeated observations, and restart scheduling.

    Evaluation counts refer to individual calls to the objective, including
    repetitions. ``stage_evaluations=None`` gives a stage the full budget.
    A zero stagnation patience disables the stagnation restart criterion.
    A negative progress reference stage uses the initial observation group;
    otherwise, it uses the incumbent at the indicated stage's start.

    The restart reserve is the maximum of ``min_restart_evaluations`` and
    ``ceil(restart_reserve_factor * interpolation_points * repetitions)``.
    It prevents starting a model with an insufficient remaining budget.
    Warmup settings override the corresponding settings in the first
    ``warmup_stages`` stages. All radii are in local model coordinates.
    """

    initial_radius: float = 0.1
    final_radius: float = 1e-6
    interpolation_factor: float = 2.0
    stage_evaluations: int | None = None
    max_stages: int = 1000
    stagnation_patience: int = 0
    min_stage_evaluations: int = 0
    improvement_tolerance: float = 0.0
    progress_exponent: float = 0.0
    minimum_radius: float = 1e-5
    repetitions: int = 1
    warmup_stages: int = 0
    warmup_radius: float | None = None
    warmup_evaluations: int | None = None
    warmup_interpolation_factor: float | None = None
    progress_reference_stage: int = -1
    restart_reserve_factor: float = 2.0
    min_restart_evaluations: int = 0

    def __post_init__(self) -> None:
        positive_scalars = (
            "initial_radius",
            "final_radius",
            "interpolation_factor",
            "minimum_radius",
            "warmup_radius",
            "warmup_interpolation_factor",
        )
        for name in positive_scalars:
            value = getattr(self, name)
            optional = name in ("warmup_radius", "warmup_interpolation_factor")
            if value is None and not optional:
                raise ValueError(f"{name} must be finite and positive")
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.final_radius > self.initial_radius:
            raise ValueError("final_radius must not exceed initial_radius")
        for name in ("improvement_tolerance", "restart_reserve_factor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.progress_exponent):
            raise ValueError("progress_exponent must be finite")
        positive_integers = (
            "stage_evaluations",
            "max_stages",
            "repetitions",
            "warmup_evaluations",
        )
        nonnegative_integers = (
            "stagnation_patience",
            "min_stage_evaluations",
            "warmup_stages",
            "min_restart_evaluations",
        )
        for name in positive_integers + nonnegative_integers:
            value = getattr(self, name)
            minimum = 1 if name in positive_integers else 0
            optional = name in ("stage_evaluations", "warmup_evaluations")
            if value is None and not optional:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if value is not None and (
                not np.isfinite(value) or int(value) != value or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            not np.isfinite(self.progress_reference_stage)
            or int(self.progress_reference_stage) != self.progress_reference_stage
        ):
            raise ValueError("progress_reference_stage must be an integer")

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> SolverOptions:
        """Construct settings, rejecting misspelled or unsupported fields."""
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown solver options: {sorted(unknown)}")
        return cls(**dict(options))

    def to_dict(self) -> dict[str, Any]:
        """Return an independent dictionary of the option values."""
        return asdict(self)
