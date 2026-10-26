"""Reference settings with budget guards and read-only iterate observation."""

from __future__ import annotations

import sys

import numpy as np


class QueryBudgetReached(BaseException):
    """Internal cancellation, caught only by the published-budget adapter.

    SQImFil's line search catches Exception and retries without advancing its
    counter. A RuntimeError budget signal would be swallowed there. This local
    signal bypasses that retry and is handled explicitly in solve_reference.
    """


def solve_reference(v, fun, x0, seed, history, events, common_budget, strict_budget=False):
    """Published settings with read-only native-iterate observation, no added queries."""
    np.random.seed(seed)
    family = v["method"]
    strict = bool(v.get("strict_budget", strict_budget))
    common_budget = int(common_budget)
    if strict and common_budget < 1:
        raise ValueError("The strict published query budget must be positive")
    initial_event_count = len(events)

    def watched(x):
        frame = sys._getframe(1)
        candidates = []
        while frame is not None:
            state = frame.f_locals
            name = frame.f_code.co_name
            module = frame.f_globals.get("__name__", "")
            if family == "pybobyqa" and module == "pybobyqa.solver":
                if name == "solve_main" and "control" in state:
                    model = state["control"].model
                    candidates.append(
                        (float(model.fopt()), model.xopt(abs_coordinates=True).copy())
                    )
                    if model.fsave is not None:
                        candidates.append((float(model.fsave), model.xsave.copy()))
                elif name == "solve" and state.get("xmin") is not None:
                    candidates.append((float(state["fmin"]), np.asarray(state["xmin"]).copy()))
            elif (
                family == "imfil"
                and name == "imfil_core"
                and module.startswith("SQImFil")
                and "x" in state
            ):
                point = np.asarray(state["x"]).reshape(-1) * 200.0 - 100.0
                events.append(
                    dict(
                        calls=len(history),
                        x=point.copy(),
                        source="reference_imfil_current_iterate_before_next_query",
                    )
                )
                break
            elif (
                family == "spsa"
                and name == "minimize"
                and module.endswith(".spsa")
                and "x" in state
            ):
                events.append(
                    dict(
                        calls=len(history),
                        x=np.asarray(state["x"]).copy(),
                        source="reference_spsa_last_completed_iterate_before_next_query",
                        iteration=int(state.get("k", 0)),
                        completed_iterations=max(
                            int(state.get("k", 0)) - (0 if "result" in state else 1), 0
                        ),
                    )
                )
                break
            frame = frame.f_back
        del frame
        if candidates:
            fbest, point = min(candidates, key=lambda pair: pair[0])
            events.append(
                dict(
                    calls=len(history),
                    x=point,
                    source="reference_pybobyqa_saved_or_model_best_before_next_query",
                    observed=fbest,
                )
            )
        # Capture the completed state first: the preceding query may have
        # completed an iteration. Never pay the rejected query or select its x.
        if strict and len(history) >= common_budget:
            raise QueryBudgetReached("Strict common query budget reached before evaluation")
        return fun(np.asarray(x).reshape(-1))

    settings = {}
    try:
        if family == "spsa":
            from qiskit_algorithms.optimizers import SPSA
            from qiskit_algorithms.utils import algorithm_globals

            algorithm_globals.random_seed = seed
            optimizer = SPSA(maxiter=550)
            settings = dict(
                settings=optimizer.settings,
                reference_maxiter=550,
                implementation="qiskit-algorithms0.3.1 maintained API, original maxiter550; other defaults",
                native_nfev_excludes_calibration_and_final_query=True,
            )
            native = optimizer.minimize(watched, x0)
            result = dict(
                native_fun=float(native.fun),
                native_reported_nfev=int(native.nfev),
                native_iterations=int(native.nit),
                actual_calls=len(history),
                **settings,
            )
            returned = np.asarray(native.x).copy()
        else:
            bounds = (np.full(len(x0), -100.0), np.full(len(x0), 100.0))
            if family == "pybobyqa":
                import pybobyqa
                from SQCommon import ObjectiveFunction

                wrapped = ObjectiveFunction(watched, {"simple_function": True})
                settings = dict(
                    requested_budget=common_budget,
                    method="pybobyqa",
                    explicit_parameters={
                        "rhobeg": 0.1,
                        "seek_global_minimum": True,
                        "objfun_has_noise": True,
                    },
                    package="Py-BOBYQA1.5.0",
                )
                native = pybobyqa.solve(
                    wrapped,
                    x0,
                    maxfun=common_budget,
                    bounds=bounds,
                    seek_global_minimum=True,
                    objfun_has_noise=True,
                    rhobeg=0.1,
                )
                returned = np.asarray(native.x).copy()
                result = dict(
                    native_fun=float(native.f),
                    native_flag=int(native.flag),
                    actual_calls=len(history),
                    **settings,
                )
            elif family == "imfil":
                import SQImFil
                import SQImFil._optset as optset

                optset.STANDALONE = False
                settings = dict(
                    requested_budget=common_budget,
                    method="imfil",
                    explicit_parameters={},
                    package="SQImFil0.3.7",
                )
                native, sampled_history = SQImFil.minimize(
                    watched, x0, np.column_stack(bounds), common_budget, None
                )
                returned = np.asarray(native.optpar).reshape(-1).copy()
                result = dict(
                    native_fun=float(native.optval), actual_calls=len(history), **settings
                )
            else:
                raise ValueError(f"Unknown reference method: {family}")

    except QueryBudgetReached:
        # This is an externally interrupted published solver, not a native result.
        # In particular, SPSA's initial calibration need not have completed.
        observed = [e for e in events[initial_event_count:] if e["calls"] <= common_budget]
        event = (
            observed[-1]
            if observed
            else dict(
                x=np.asarray(x0).copy(),
                calls=0,
                source="initial_point_before_first_completed_reference_iterate",
            )
        )
        returned = np.asarray(event["x"]).copy()
        result = dict(
            actual_calls=len(history),
            native_status="budget_interrupted",
            native_result_available=False,
            native_fun=None,
            native_reported_nfev=None,
            native_iterations=None,
            strict_budget=True,
            common_query_budget=common_budget,
            return_source=event["source"],
            recommendation_calls=int(event["calls"]),
            completed_iterations=event.get("completed_iterations"),
            objective_calls_after_budget=0,
            **settings,
        )
        events.append(
            dict(
                calls=len(history),
                x=returned,
                source="budget_interrupted_reference_state",
                state_source=event["source"],
            )
        )
        return returned, result
    if strict:
        result.update(
            native_status="native_return",
            native_result_available=True,
            strict_budget=True,
            common_query_budget=common_budget,
        )
    events.append(dict(calls=len(history), x=returned, source="reference_native_endpoint"))
    return returned, result
