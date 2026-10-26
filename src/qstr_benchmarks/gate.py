"""Reproduce finite-shot gate calibration on the C3 two-qutrit device."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .gate_support.oracle import GateDevice, GateOracle, plain


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf8")


def _baseline_options(parameters: dict) -> dict:
    translations = {
        "population_size": "popsize",
        "initial_spread": "spread",
        "evaluate_initial_point": "init_point",
    }
    return {translations.get(key, key): value for key, value in parameters.items()}


def run_case(configuration: dict, output: Path) -> dict:
    """Optimize with paid counts, then validate frozen native recommendations.

    Each evaluation draws 25 fresh length-100 Clifford circuits and 1000
    independent shots per circuit. The independent validation circuit ensemble
    is evaluated only after all optimization recommendations are frozen.
    """
    from qstr import minimize_with_preset

    from .gate_support.cma import minimize as minimize_cma

    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "configuration.json", configuration)
    method = configuration["method"]
    if method not in ("qstr", "cma_es"):
        raise ValueError("gate_calibration supports qstr and cma_es")
    seed = int(configuration["seed"])
    budget = int(configuration["max_evaluations"])
    if budget < 1:
        raise ValueError("max_evaluations must be positive")
    physical = configuration.get("physical_settings", {})
    target = int(physical.get("target_qubit", 0))
    circuits_per_evaluation = int(physical.get("circuits_per_evaluation", 25))
    if circuits_per_evaluation != 25:
        raise ValueError("The gate_calibration protocol requires 25 circuits per evaluation")
    parameters = dict(configuration.get("parameters", {}))
    parameters.pop("preset", None)
    if method == "cma_es":
        parameters = _baseline_options(parameters)
        if budget < int(parameters.get("popsize", 25)):
            raise ValueError("CMA needs at least one complete population for its native return")
    interval = int(configuration.get("checkpoint_interval", 5))
    if interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    validation = configuration.get("validation", {})
    validation_circuits = int(
        validation.get(
            "circuits",
            validation.get("holdout_circuits", physical.get("validation_circuits", 4096)),
        )
    )
    validation_seed = int(
        validation.get(
            "seed", validation.get("holdout_seed", physical.get("validation_seed", 9431057))
        )
    )
    if validation_circuits < 2:
        raise ValueError("Validation requires at least two circuits")
    device = GateDevice(target, output / "device_setup")
    oracle = GateOracle(
        budget * circuits_per_evaluation * 1000,
        seed,
        output,
        target,
        physical.get("initial_parameters"),
        device=device,
    )
    try:

        def objective(point):
            return oracle.sample(point, circuits_per_evaluation, "optimization")["mean"]

        if method == "qstr":
            result = minimize_with_preset(
                objective,
                preset="gate_calibration",
                x0=oracle.x0,
                max_evaluations=budget,
                seed=seed,
                options=parameters,
            ).to_dict()
        else:
            result = minimize_cma(objective, oracle.x0, budget, seed=seed, options=parameters)
        training_seconds = perf_counter() - started
        count = len(oracle.history)
        if result["nfev"] != count or count > budget:
            raise RuntimeError("Optimizer and physical evaluation accounting disagree")
        if oracle.spent != count * circuits_per_evaluation * 1000:
            raise RuntimeError("Physical shot accounting is inconsistent")
        _write(output / "optimizer.json", result)
        _write(output / "sampling_history.json", oracle.history)

        checkpoints = sorted(set(range(interval, budget + 1, interval)) | {budget})
        recommendations = []
        replay_checks = []
        for checkpoint in checkpoints:
            prefix = min(checkpoint, count)
            if checkpoint >= count:
                point = result["x"]
                source = "native_final_return"
            elif method == "qstr":
                eligible = [
                    group
                    for group in result["observation_groups"]
                    if group["end_evaluation"] <= prefix
                ]
                if not eligible:
                    continue
                point = min(eligible, key=lambda group: group["mean"])["x"]
                source = "minimum_completed_observation_group"
            else:
                cursor = 0
                max_difference = 0.0

                def replay_objective(point):
                    nonlocal cursor, max_difference
                    row = oracle.history[cursor]
                    difference = float(np.max(np.abs(np.asarray(point) - row["x"])))
                    max_difference = max(max_difference, difference)
                    if difference > 1e-12:
                        raise RuntimeError("CMA replay diverged from the paid prefix")
                    cursor += 1
                    return row["mean"]

                replay = minimize_cma(
                    replay_objective,
                    oracle.x0,
                    prefix,
                    seed=seed,
                    options=parameters,
                )
                point = replay["x"]
                replay_checks.append(
                    {
                        "evaluations": checkpoint,
                        "reused_observations": cursor,
                        "max_coordinate_difference": max_difference,
                        "physical_evaluations": 0,
                    }
                )
                source = "native_return_replayed_from_paid_prefix"
            if point is not None:
                recommendations.append(
                    {
                        "evaluations": checkpoint,
                        "actual_evaluations": prefix,
                        "x": plain(point),
                        "source": source,
                    }
                )
        # Recommendations are immutable before any independent quality is read.
        _write(output / "recommendations_before_validation.json", recommendations)
        _write(output / "replay_checks.json", replay_checks)
        points = {}
        aliases = {}

        def add_point(label, point):
            key = tuple(np.asarray(point, dtype=float))
            existing = next((name for name, value in points.items() if tuple(value) == key), label)
            points[existing] = list(key)
            aliases[label] = existing

        add_point("initial", oracle.x0)
        add_point("final", result["x"])
        for row in recommendations:
            add_point(str(row["evaluations"]), row["x"])
        holdout = oracle.holdout(points, n=validation_circuits, seed=validation_seed)
        holdout["point_aliases"] = aliases
        _write(output / "validation.json", holdout)
        values = {row["name"]: row for row in holdout["values"]}
        for row in recommendations:
            estimate = values[aliases[str(row["evaluations"])]]
            row["quality"] = estimate["mean"]
            row["standard_error"] = estimate["circuit_se"]
        history = [
            {
                "x": row["x"],
                "value": row["mean"],
                "cost": row["cost"],
                "cumulative_cost": row["cost_end"],
            }
            for row in oracle.history
        ]
        record = {
            "problem": "gate_calibration",
            "method": method,
            "seed": seed,
            "status": "completed",
            "nfev": count,
            "history": history,
            "recommendations": recommendations,
            "initial_quality": values[aliases["initial"]]["mean"],
            "final_quality": values[aliases["final"]]["mean"],
            "initial_x": plain(oracle.x0),
            "final_x": plain(result["x"]),
            "metric": "ORBIT failure probability",
            "cost": {
                "evaluations": count,
                "shots": oracle.spent,
                "circuits": count * circuits_per_evaluation,
                "validation_circuits": validation_circuits * len(points),
                "validation_operator_evaluations": len(points),
                "validation_shots": 0,
            },
            "validation": {"circuits": validation_circuits, "seed": validation_seed},
            "result": result,
            "training_seconds": training_seconds,
            "elapsed_seconds": perf_counter() - started,
        }
        _write(output / "result.json", record)
        return record
    finally:
        oracle.close()
