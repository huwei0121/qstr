"""Reproduce the finite-count QAOA MaxCut comparison with six optimizers."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2) + "\n", encoding="utf-8")


def _anatra(oracle, x0, budget, events):
    from .qaoa_support.model import load_source

    vendor = Path(__file__).parent / "qaoa_support/vendor"
    # The upstream modules use top-level sibling imports. Add only these
    # bundled source directories; no research checkout is consulted.
    for directory in (vendor / "anatra", vendor / "minq"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import anatra
    from ibcdfo.pounders.general_h_funs import identity_combine

    compatibility = load_source("qstr_input_check", vendor / "input_check/checkinputss.py")
    anatra.checkinputss = compatibility.checkinputss

    def observed_pair(x):
        frame = sys._getframe(1)
        state = frame.f_locals
        if frame.f_code.co_name == "anatra" and "X" in state and "xk_in" in state:
            index = int(state["xk_in"])
            points = state["X"]
            if 0 <= index < len(points):
                center = np.asarray(points[index], float).copy()
                if np.all(np.isfinite(center)):
                    events.append(
                        dict(
                            calls=len(oracle.history),
                            x=center,
                            source="anatra_current_center",
                            center_index=index,
                        )
                    )
        del frame, state
        return oracle.pair(x)

    points, values, model_values, flag, index = anatra.anatra(
        observed_pair,
        x0,
        len(x0),
        budget,
        1e-13,
        0.1,
        1,
        np.full(len(x0), -np.inf),
        np.full(len(x0), np.inf),
        Options={"hfun": lambda value: value, "combinemodels": identity_combine, "printf": 0},
    )
    if flag < 0:
        raise RuntimeError(f"ANATRA returned failure flag {flag}")
    returned = np.asarray(points[index]).copy()
    events.append(dict(calls=len(oracle.history), x=returned, source="anatra_native_endpoint"))
    result = dict(
        x=returned,
        values=values,
        model_values=model_values,
        flag=float(flag),
        index=int(index),
        nfev=len(oracle.history),
        return_policy="native current-center index",
    )
    return returned, result


def run_case(configuration: dict, output: Path) -> dict:
    """Run one seed and freeze recommendations before exact diagnostics.

    Each evaluation costs shots_per_evaluation counts (1000 by default).
    Checkpoints observe the ongoing run; they do not restart it. NOMAD's native
    prefix returns are reconstructed by replaying previously paid feedback.
    """
    from .qaoa_support.model import CountOracle, exact_evaluator

    if configuration.get("problem") != "qaoa_maxcut":
        raise ValueError("This backend implements qaoa_maxcut")
    method = configuration["method"]
    if method not in {"qstr", "anatra", "pybobyqa", "imfil", "nomad", "spsa"}:
        raise ValueError(f"Unknown QAOA method: {method}")
    parameters = configuration.get("parameters", {})
    allowed = {"preset", "coordinate_transform", "options"}
    if set(parameters) - allowed:
        raise ValueError(f"Unsupported QAOA parameters: {set(parameters) - allowed}")
    if parameters.get("coordinate_transform", "identity") != "identity":
        raise ValueError("Reference optimizers use the published identity coordinates")
    if parameters.get("preset", "qaoa_maxcut") != "qaoa_maxcut":
        raise ValueError("QAOA requires the qaoa_maxcut preset")
    if "options" in parameters and method != "qstr":
        raise ValueError("Reference settings are fixed; options apply only to Q-STR")
    budget = int(configuration["max_evaluations"])
    seed = int(configuration["seed"])
    observation = configuration.get("observation", {})
    configured_shots = observation.get(
        "shots_per_evaluation", configuration.get("shots_per_evaluation", 1000)
    )
    if (
        "shots_per_evaluation" in configuration
        and configured_shots != configuration["shots_per_evaluation"]
    ):
        raise ValueError("Conflicting top-level and observation shot counts")
    shots = int(configured_shots)
    if observation.get("cost_per_evaluation", shots) != shots:
        raise ValueError("QAOA cost_per_evaluation must equal shots_per_evaluation")
    interval = int(configuration.get("checkpoint_interval", 25))
    if budget < 1 or shots < 1 or interval < 1:
        raise ValueError("Evaluation budget, shots and checkpoint interval must be positive")
    if method == "anatra" and budget <= 11:
        raise ValueError("ANATRA requires more than n+1=11 evaluations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "history.json").exists():
        raise FileExistsError("Keep previous results: select a new case output directory")
    grid = sorted(set(range(interval, budget + 1, interval)) | {budget})
    x0 = np.zeros(10)
    oracle = CountOracle(seed, shots, budget)
    events = []
    np.random.seed(seed)
    started = time.perf_counter()
    with (output / "solver.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if method == "qstr":
                from qstr import minimize_with_preset

                answer = minimize_with_preset(
                    oracle.scalar,
                    preset="qaoa_maxcut",
                    x0=x0,
                    max_evaluations=budget,
                    seed=seed,
                    options=parameters.get("options"),
                )
                returned = answer.x.copy()
                best = float("inf")
                for group in answer.observation_groups:
                    if group.mean < best:
                        best = group.mean
                        events.append(
                            dict(
                                calls=group.end_evaluation,
                                x=group.x.copy(),
                                source="completed_group_minimum",
                            )
                        )
                result = answer.to_dict()
            elif method == "anatra":
                returned, result = _anatra(oracle, x0, budget, events)
            elif method == "nomad":
                from .qaoa_support.nomad import minimize

                result = minimize(
                    oracle.scalar,
                    x0,
                    budget,
                    lower=-100.0,
                    upper=100.0,
                    stats_path=output / "nomad_native_stats.txt",
                    seed=seed,
                    checkpoint_calls=grid,
                )
                returned = np.asarray(result["x"]).copy()
                events = result["recommendation_events"]
            else:
                from .qaoa_support.reference import solve_reference

                returned, result = solve_reference(
                    dict(method=method),
                    oracle.scalar,
                    x0,
                    seed,
                    oracle.history,
                    events,
                    budget,
                    strict_budget=True,
                )
    search_seconds = time.perf_counter() - started
    frozen = []
    for count in grid:
        if count >= len(oracle.history):
            event = dict(
                x=returned,
                source="completed_native_return",
                recommendation_calls=len(oracle.history),
            )
            if result.get("native_status") == "budget_interrupted":
                event["source"] = "budget_interrupted_completed_state"
        else:
            permitted = [event for event in events if event["calls"] <= count]
            event = dict(permitted[-1]) if permitted else dict(x=x0, source="initial_point")
            event["recommendation_calls"] = event.get("calls", 0)
        event.update(evaluations=count, actual_evaluations=min(count, len(oracle.history)))
        frozen.append(event)

    # Persist all selection decisions before constructing the diagnostic.
    _write(output / "history.json", oracle.history)
    _write(output / "recommendation_events.json", events)
    _write(
        output / "frozen_recommendations.json",
        dict(initial_x=x0, final_x=returned, checkpoints=frozen),
    )
    diagnostic = exact_evaluator(oracle.graph)
    cache = {}

    def quality(x):
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if key not in cache:
            cache[key] = max(0.0, 6.0 + diagnostic(x))
        return cache[key]

    initial_quality = quality(x0)
    final_quality = quality(returned)
    recommendations = [
        dict(
            evaluations=0,
            actual_evaluations=0,
            x=x0,
            quality=initial_quality,
            source="initial_point",
        )
    ]
    for event in frozen:
        recommendations.append(dict(event, quality=quality(event["x"])))
    record = dict(
        problem="qaoa_maxcut",
        method=method,
        seed=seed,
        status="completed",
        nfev=len(oracle.history),
        history=oracle.history,
        recommendations=recommendations,
        initial_quality=initial_quality,
        final_quality=final_quality,
        initial_x=x0,
        final_x=returned,
        metric="expected_cut_gap_to_six",
        cost=dict(
            evaluations=len(oracle.history),
            shots=len(oracle.history) * shots,
            shots_per_evaluation=shots,
            search_seconds=search_seconds,
            validation_evaluations=len(cache),
            validation_unit="exact_statevector_points",
        ),
        result=result,
    )
    _write(output / "solver.json", result)
    return _plain(record)
