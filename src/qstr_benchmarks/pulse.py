"""Reproduce the 82-control Duffing/ORBIT pulse calibration comparison."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qstr import minimize_with_preset
from qstr.baselines import minimize_reference

from .pulse_support.baselines import _plain
from .pulse_support.baselines import minimize as minimize_baseline
from .pulse_support.model import PulseModel, sample_circuits
from .pulse_support.oracle import OrbitOracle


def load_model() -> PulseModel:
    """Load the complete reference waveform and physical calibration scales."""
    path = Path(__file__).with_name("pulse_support") / "model.json"
    return PulseModel(**json.loads(path.read_text(encoding="utf-8")))


def run_case(configuration: dict, output: Path) -> dict:
    """Execute a bounded run, freeze native recommendations, then validate them.

    Validation uses independent circuits after optimization. Its propagation
    count is reported separately and never consumes optimizer feedback.
    """
    started = perf_counter()
    output = Path(output)
    model = load_model()
    seed = int(configuration["seed"])
    budget = int(configuration["max_evaluations"])
    method = configuration["method"]
    parameters = dict(configuration.get("parameters", {}))
    initial_point = model.initial(seed)
    oracle = OrbitOracle(model, budget, seed, record_components=True)

    def solve(objective, limit):
        if method == "qstr":
            if parameters not in ({}, {"preset": "pulse_calibration"}):
                raise ValueError("Q-STR requires the pulse_calibration preset")
            return minimize_with_preset(
                objective,
                preset="pulse_calibration",
                x0=initial_point,
                max_evaluations=limit,
                seed=seed,
            )
        if method == "nelder_mead":
            radius = parameters["initial_simplex_radius"]
            orientation = parameters["orientation"]
            if orientation not in ("random", "coordinate"):
                raise ValueError("Unknown simplex orientation")
            directions = np.eye(model.dimension)
            if orientation == "random":
                directions = np.linalg.qr(
                    np.random.default_rng(seed + 173).normal(
                        size=(model.dimension, model.dimension)
                    )
                )[0]
            simplex = np.vstack((initial_point, initial_point + radius * directions.T))
            width = model.upper - model.lower
            simplex = model.lower + width - np.abs(np.mod(simplex - model.lower, 2 * width) - width)
            return minimize_reference(
                objective,
                initial_point,
                limit,
                method=method,
                bounds=(model.lower, model.upper),
                seed=seed,
                options={"initial_simplex": simplex, "adaptive": parameters["adaptive"]},
            )
        return minimize_baseline(
            objective,
            initial_point,
            limit,
            method=method,
            lower=model.lower,
            upper=model.upper,
            seed=seed,
            options=parameters,
        )

    native = solve(oracle, budget)
    optimization_seconds = perf_counter() - started
    result = native.to_dict() if method == "qstr" else native
    if result["nfev"] != len(oracle.loss) or len(oracle.loss) > budget:
        raise RuntimeError("Paid-call accounting mismatch")
    returned = result["x"] if result["x"] is not None else initial_point.tolist()
    interval = int(configuration.get("checkpoint_interval", 50))
    grid = sorted(set(range(interval, budget + 1, interval)) | {budget})
    recommendations = []
    replayed = 0
    for limit in grid:
        if method == "qstr":
            groups = [g for g in result["observation_groups"] if g["end_evaluation"] <= limit]
            group = min(groups, key=lambda g: g["mean"]) if groups else None
            point = group["x"] if group else initial_point.tolist()
            source = "completed_group_minimum" if group else "initial_point"
            used = min(limit, len(oracle.loss))
        elif limit >= len(oracle.loss):
            point = returned
            source = (
                "native_return" if result["x"] is not None else "initial_point_before_native_return"
            )
            used = len(oracle.loss)
        else:
            cursor = 0

            def replay(point):
                nonlocal cursor
                if cursor >= len(oracle.loss) or not np.array_equal(point, oracle.x[cursor]):
                    raise RuntimeError(f"Saved-feedback replay diverged at query {cursor}")
                value = oracle.loss[cursor]
                cursor += 1
                return value

            replay_result = solve(replay, limit)
            point = replay_result["x"]
            source = "native_saved_feedback_replay"
            if point is None:
                point = initial_point.tolist()
                source = "initial_point_before_native_return"
            used = cursor
            replayed += cursor
        recommendations.append(
            {
                "evaluations": limit,
                "actual_evaluations": used,
                "x": _plain(point),
                "source": source,
            }
        )
    # All points are fixed before the first independent physical validation.
    (output / "frozen_recommendations.json").write_text(
        json.dumps(
            _plain(
                {
                    "initial_x": initial_point,
                    "final_x": returned,
                    "recommendations": recommendations,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    holdout_count = int(configuration.get("validation", {}).get("holdout_circuits", 4096))
    if holdout_count < 2:
        raise ValueError("At least two validation circuits are required")
    holdout = sample_circuits(
        np.random.default_rng(np.random.SeedSequence([471931, seed])),
        holdout_count,
        model.config.circuit_length,
        model.config.target_state,
    )
    cache = {}

    def quality(point):
        point = np.asarray(point, float)
        key = point.tobytes()
        if key not in cache:
            values = 1 - model.probabilities(point, holdout)
            cache[key] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))))
        return cache[key][0]

    initial_quality = quality(initial_point)
    final_quality = quality(returned)
    for recommendation in recommendations:
        recommendation["quality"] = quality(recommendation["x"])
    np.savez_compressed(
        output / "trajectory.npz",
        x=oracle.x,
        value=oracle.loss,
        components=oracle.components,
        seconds=oracle.seconds,
    )
    return _plain(
        {
            "problem": "pulse_calibration",
            "method": method,
            "seed": seed,
            "status": "completed",
            "nfev": len(oracle.loss),
            "max_evaluations": budget,
            "history": [{"x": x, "value": y} for x, y in zip(oracle.x, oracle.loss)],
            "recommendations": recommendations,
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "metric": "mean_failure_probability",
            "initial_x": initial_point,
            "final_x": returned,
            "final_standard_error": cache[np.asarray(returned, float).tobytes()][1],
            "cost": {
                "optimization_evaluations": len(oracle.loss),
                "circuits_per_evaluation": model.config.circuits_per_evaluation,
                "optimization_circuits": len(oracle.loss) * model.config.circuits_per_evaluation,
                "optimization_shots": 0,
                "validation_points": len(cache),
                "validation_circuits": len(cache) * len(holdout),
                "saved_feedback_replayed": replayed,
                "replay_new_evaluations": 0,
            },
            "validation": {"holdout_circuits": holdout_count, "seed_sequence": [471931, seed]},
            "result": result,
            "optimization_seconds": optimization_seconds,
            "elapsed_seconds": perf_counter() - started,
        }
    )
