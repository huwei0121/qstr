"""Numerical configurations for four quantum parameter optimization problems."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bases import make_basis_factory
from .options import SolverOptions
from .result import OptimizationResult

_PRESETS: dict[str, dict[str, Any]] = {
    "pulse_calibration": {
        "name": "pulse_calibration",
        "dimension": 82,
        "max_evaluations": 1000,
        "x0": None,
        "bounds": (-0.5, 0.5),
        "options": SolverOptions(
            initial_radius=0.08,
            final_radius=0.001,
            interpolation_factor=1.5,
            stage_evaluations=333,
            max_stages=30,
            stagnation_patience=24,
            min_stage_evaluations=97,
            improvement_tolerance=0.05,
            progress_exponent=0.25,
            minimum_radius=0.01,
            warmup_stages=1,
            warmup_radius=0.08,
            warmup_evaluations=100,
            warmup_interpolation_factor=1.5,
            progress_reference_stage=1,
            restart_reserve_factor=0.0,
        ).to_dict(),
        "coordinates": "82 pulse corrections ordered by in-phase and quadrature channels",
        "feedback": "logarithm of mean failure probability over 40 random circuits",
        "evaluation_cost": {"amount": 40, "unit": "exact_probability_circuits"},
    },
    "gate_calibration": {
        "name": "gate_calibration",
        "dimension": 3,
        "max_evaluations": 100,
        "x0": [-0.054559463003214105, 0.08264154056073997, -0.11956622745212653],
        "bounds": (-1.0, 1.0),
        "options": SolverOptions(
            initial_radius=0.08,
            final_radius=0.0001,
            interpolation_factor=3.0,
            stage_evaluations=50,
        ).to_dict(),
        "coordinates": "3 scaled controls in [-1, 1]",
        "feedback": "mean failure frequency over 25 circuits with 1000 shots each",
        "evaluation_cost": {"amount": 25000, "unit": "shots"},
    },
    "qaoa_maxcut": {
        "name": "qaoa_maxcut",
        "dimension": 10,
        "max_evaluations": 550,
        "x0": [0.0] * 10,
        "bounds": None,
        "options": SolverOptions(
            initial_radius=0.1,
            final_radius=0.003,
            stage_evaluations=550,
            max_stages=10,
            min_stage_evaluations=30,
            minimum_radius=0.001,
        ).to_dict(),
        "coordinates": "10 physical circuit angles: beta layers followed by gamma layers",
        "feedback": "negative mean cut value from 1000 shots",
        "evaluation_cost": {"amount": 1000, "unit": "shots"},
    },
    "ising_state_transfer": {
        "name": "ising_state_transfer",
        "dimension": 100,
        "max_evaluations": 2000,
        "x0": [0.0] * 100,
        "bounds": (-1.0, 1.0),
        "options": SolverOptions(
            initial_radius=0.05,
            stage_evaluations=500,
            max_stages=100,
            stagnation_patience=150,
            repetitions=3,
        ).to_dict(),
        "coordinates": "100 normalized pulse amplitudes; physical amplitudes equal 1000*x",
        "feedback": "state-transfer infidelity from one noisy propagation",
        "evaluation_cost": {"amount": 1, "unit": "physical_propagations"},
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Return an independent copy of a problem's numerical configuration.

    The four supported names are ``pulse_calibration``, ``gate_calibration``,
    ``qaoa_maxcut`` and ``ising_state_transfer``. A preset specifies the
    parameter coordinates, feedback scale, evaluation budget and search
    settings; it does not construct a simulator or an objective function.
    """
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {tuple(_PRESETS)}")
    return deepcopy(_PRESETS[name])


def minimize_with_preset(
    objective: Callable[[NDArray[np.float64]], float],
    *,
    preset: str,
    x0: ArrayLike | None = None,
    max_evaluations: int | None = None,
    seed: int = 0,
    options: Mapping[str, Any] | SolverOptions | None = None,
) -> OptimizationResult:
    """Minimize a supplied objective using one of the problem presets.

    Each call to ``objective(x)`` consumes one evaluation, including repeated
    measurements. The objective owns its random stream independently of
    ``seed``, which controls search coordinates only. It must return feedback
    in the coordinates and scale documented by :func:`get_preset`.

    Pulse calibration requires the instance-specific ``x0``. Other presets
    provide initial points. Circuit-angle rotation and pulse subspaces are
    applied internally, so the objective must accept physical parameters
    without applying the basis a second time.

    A mapping in ``options`` overrides selected preset settings. A complete
    ``SolverOptions`` instance replaces all preset numerical settings.
    ``evaluation_cost`` describes the default measurement protocol; this
    wrapper records objective calls and does not infer hardware expenditure.
    """
    from .solver import minimize

    configuration = get_preset(preset)
    if x0 is None:
        if configuration["x0"] is None:
            raise ValueError("pulse_calibration requires an instance-specific x0")
        x0 = configuration["x0"]
    initial_point = np.asarray(x0, dtype=float)
    dimension = configuration["dimension"]
    if initial_point.shape != (dimension,):
        raise ValueError(f"{preset} requires a vector of dimension {dimension}")
    if max_evaluations is None:
        max_evaluations = configuration["max_evaluations"]
    if isinstance(options, SolverOptions):
        numerical = options
    else:
        numerical = SolverOptions.from_mapping({**configuration["options"], **dict(options or {})})
    interval = configuration["bounds"]
    bounds = (
        None
        if interval is None
        else (
            np.full(dimension, interval[0]),
            np.full(dimension, interval[1]),
        )
    )
    result = minimize(
        objective,
        initial_point,
        max_evaluations,
        bounds=bounds,
        basis_factory=make_basis_factory(preset, warmup_stages=numerical.warmup_stages),
        seed=seed,
        options=numerical,
    )
    result.metadata.update(preset=preset, coordinates=configuration["coordinates"])
    return result
