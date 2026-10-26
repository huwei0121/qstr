"""Structured derivative-free optimization for expensive scalar feedback."""

from .options import SolverOptions
from .presets import get_preset, minimize_with_preset
from .result import EvaluationHistory, ObservationGroup, OptimizationResult, StageResult
from .solver import minimize

__version__ = "0.1.0"

PRESET_NAMES = (
    "pulse_calibration",
    "gate_calibration",
    "qaoa_maxcut",
    "ising_state_transfer",
)

__all__ = [
    "EvaluationHistory",
    "ObservationGroup",
    "OptimizationResult",
    "PRESET_NAMES",
    "SolverOptions",
    "StageResult",
    "get_preset",
    "minimize",
    "minimize_with_preset",
]
