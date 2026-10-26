"""Five-qubit Ising state transfer with fresh Gaussian coupling noise."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from qstr import minimize_with_preset

from .ising_support.model import IsingModel
from .pulse_support.baselines import _plain


def run_case(configuration: dict, output: Path) -> dict:
    """Run Q-STR or full QuOCS dCRAB, then apply independent quadrature.

    Each scalar feedback evaluates one propagation with a fresh coupling
    g distributed as Normal(2, 0.1). The variance is 0.1, not its deviation.
    Independent Gauss--Hermite quadrature never affects point selection.
    """
    started = perf_counter()
    output = Path(output).resolve()
    model = IsingModel()
    seed = int(configuration["seed"])
    budget = int(configuration["max_evaluations"])
    method = configuration["method"]
    parameters = dict(configuration.get("parameters", {}))
    initial_point = np.zeros(100)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 71855]))
    history, events = [], []

    def objective(point):
        if len(history) >= budget:
            raise RuntimeError("Physical propagation cap exceeded")
        point = np.asarray(point, float)
        if point.shape != (100,) or np.max(abs(point)) > 1 + 1e-10:
            raise ValueError("Expected 100 pulse amplitudes in [-1, 1]")
        coupling = 2 + np.sqrt(0.1) * rng.normal()
        began = perf_counter()
        value = model.loss(1000 * point, coupling)
        history.append(
            {
                "x": point.tolist(),
                "value": value,
                "coupling": coupling,
                "seconds": perf_counter() - began,
            }
        )
        return value

    with (output / "solver.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if method == "qstr":
                if parameters not in ({}, {"preset": "ising_state_transfer"}):
                    raise ValueError("Q-STR requires the ising_state_transfer preset")
                native = minimize_with_preset(
                    objective,
                    preset="ising_state_transfer",
                    x0=initial_point,
                    max_evaluations=budget,
                    seed=seed,
                )
                result = native.to_dict()
                returned = result["x"]
                best = float("inf")
                for group in result["observation_groups"]:
                    if group["mean"] < best:
                        best = group["mean"]
                        events.append(
                            {
                                "evaluations": group["end_evaluation"],
                                "x": group["x"],
                                "source": "completed_group_minimum",
                            }
                        )
            elif method == "dcrab":
                from quocslib.Optimizer import Optimizer
                from quocslib.utils.AbstractFoM import AbstractFoM

                reported_std = float(
                    configuration.get("physical_settings", {}).get("reported_objective_std", 0.1)
                )

                class FigureOfMerit(AbstractFoM):
                    def get_FoM(self, pulses=None, parameters=None, timegrids=None):
                        return {"FoM": objective(np.asarray(pulses[0]) / 1000), "std": reported_std}

                path = Path(__file__).with_name("ising_support") / "dcrab.json"
                settings = json.loads(path.read_text(encoding="utf-8"))
                settings.update(
                    create_logfile=True,
                    console_info=False,
                    use_microsecond_name=True,
                    optimization_client_name="qo",
                )
                settings["pulses"][0]["amplitude_variation"] = parameters["amplitude_variation"]
                settings["algorithm_settings"].update(
                    max_eval_total=budget,
                    random_number_generator={"seed_number": seed},
                    optimization_direction="minimization",
                )
                previous = Path.cwd()
                os.chdir(output)
                try:
                    optimizer = Optimizer(settings, FigureOfMerit())
                    algorithm = optimizer.get_optimization_algorithm()
                    original = algorithm._inner_routine_call

                    def tracked(*args, **kwargs):
                        value = original(*args, **kwargs)
                        if getattr(algorithm, "best_xx", None) is not None:
                            point = (
                                np.asarray(algorithm._get_controls(algorithm.best_xx)["pulses"][0])
                                / 1000
                            )
                            events.append(
                                {
                                    "evaluations": len(history),
                                    "x": point.copy(),
                                    "source": "quocs_best_controls",
                                }
                            )
                        return value

                    algorithm._inner_routine_call = tracked
                    optimizer.execute()
                    returned = np.asarray(algorithm.get_best_controls()["pulses"][0]) / 1000
                    result = algorithm._get_final_results()
                    result.update(
                        nfev=len(history),
                        superiterations=algorithm.super_it,
                        re_evaluation_steps=algorithm.re_evaluation_steps,
                        reported_objective_std=reported_std,
                        stopping_compatibility={
                            "requested_frtol": 0.001,
                            "frtol_effective": False,
                            "effective_fatol": 1e-100,
                            "effective_xatol": 1e-14,
                        },
                    )
                finally:
                    os.chdir(previous)
                (output / "effective_configuration.json").write_text(
                    json.dumps(settings, indent=2),
                    encoding="utf-8",
                )
            else:
                raise ValueError("Ising state transfer supports qstr and dcrab")
    optimization_seconds = perf_counter() - started
    if result["nfev"] != len(history) or len(history) > budget:
        raise RuntimeError("Paid-call accounting mismatch")
    interval = int(configuration.get("checkpoint_interval", 100))
    grid = sorted(set(range(interval, budget + 1, interval)) | {budget})
    recommendations = []
    for limit in grid:
        if limit >= len(history):
            point, source, used = returned, "native_return", len(history)
        else:
            eligible = [event for event in events if event["evaluations"] <= limit]
            event = (
                eligible[-1]
                if eligible
                else {
                    "x": initial_point,
                    "source": "initial_point_before_native_return",
                    "evaluations": 0,
                }
            )
            point, source, used = event["x"], event["source"], event["evaluations"]
        recommendations.append(
            {
                "evaluations": limit,
                "x": _plain(point),
                "source": source,
                "recommendation_evaluations": used,
            }
        )
    (output / "native_events.json").write_text(
        json.dumps(_plain(events), indent=2), encoding="utf-8"
    )
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
    validation = configuration.get("validation", {})
    quadrature_nodes = int(validation.get("quadrature_nodes", 128))
    convergence_nodes = int(validation.get("convergence_nodes", 64))
    if quadrature_nodes < 1 or convergence_nodes < 0:
        raise ValueError("Invalid independent quadrature orders")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    couplings = 2 + np.sqrt(0.2) * nodes
    cache = {}

    def quality(point):
        point = np.asarray(point, float)
        key = point.tobytes()
        if key not in cache:
            cache[key] = float(
                sum(
                    weight * model.loss(1000 * point, coupling)
                    for weight, coupling in zip(weights, couplings)
                )
                / np.sqrt(np.pi)
            )
        return cache[key]

    initial_quality, final_quality = quality(initial_point), quality(returned)
    for recommendation in recommendations:
        recommendation["quality"] = quality(recommendation["x"])
    convergence = []
    if convergence_nodes:
        nodes, weights = np.polynomial.hermite.hermgauss(convergence_nodes)
        value = float(
            sum(
                weight * model.loss(1000 * np.asarray(returned), 2 + np.sqrt(0.2) * node)
                for weight, node in zip(weights, nodes)
            )
            / np.sqrt(np.pi)
        )
        convergence.append(
            {
                "nodes": convergence_nodes,
                "quality": value,
                "absolute_difference": abs(value - final_quality),
            }
        )
    np.savez_compressed(
        output / "trajectory.npz",
        x=[h["x"] for h in history],
        value=[h["value"] for h in history],
        coupling=[h["coupling"] for h in history],
    )
    return _plain(
        {
            "problem": "ising_state_transfer",
            "method": method,
            "seed": seed,
            "status": "completed",
            "nfev": len(history),
            "max_evaluations": budget,
            "history": history,
            "recommendations": recommendations,
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "initial_x": initial_point,
            "final_x": returned,
            "metric": "expected_state_transfer_infidelity",
            "result": result,
            "cost": {
                "optimization_evaluations": len(history),
                "optimization_propagations": len(history),
                "validation_points": len(cache),
                "validation_propagations": len(cache) * quadrature_nodes + convergence_nodes,
            },
            "validation": {"quadrature_nodes": quadrature_nodes, "convergence": convergence},
            "optimization_seconds": optimization_seconds,
            "elapsed_seconds": perf_counter() - started,
        }
    )
