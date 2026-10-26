"""Official NOMAD 4.4 binary with checked SQNomad 0.2.3 settings.

SQNomad 0.2.3 embeds NOMAD 4.0.0 and does not provide a Windows wheel.
This is an explicit official-version substitution, not a claim of identical
4.0/4.4 numerical trajectories. No NOMAD update is implemented in Python.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def minimize(fun, x0, budget, *, lower, upper, stats_path, seed=0, checkpoint_calls=None):
    """Use scalar paid feedback and return native solution + incumbent events.

    ``seed`` labels the problem/noise repetition. SQNomad resets NOMAD's own
    random generator to its default, so the algorithm seed is explicitly zero.
    Recommendation events use native returns from exact paid-prefix replay.
    NOMAD's STATS_FILE is diagnostic only: ties can make its last improvement
    point differ from the native cache-best return. Replays reuse recorded
    scalar values and never invoke the physical objective or exact values.
    ``stats_path`` must be a new per-run file. Separate worker processes are
    required for concurrent NOMAD runs because its C++ state is process-global.
    """
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or not len(x0) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a finite vector")
    lower = np.broadcast_to(np.asarray(lower, float), x0.shape)
    upper = np.broadcast_to(np.asarray(upper, float), x0.shape)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(x0 < lower)
        or np.any(x0 > upper)
    ):
        raise ValueError("finite bounds containing x0 are required")
    destination = Path(stats_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Keep the existing native statistics: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import PyNomad

    # Explicitly match checked 4.0 wrapper choices and changed core defaults.
    # The remaining 4.4 implementation changes are reported, not hidden.
    parameters = [
        "BB_OUTPUT_TYPE OBJ",
        f"MAX_BB_EVAL {budget}",
        f"MAX_EVAL {100 * budget}",
        "GRANULARITY * 0.0000001",
        "SEED 0",
        "DIRECTION_TYPE ORTHO 2N",
        "QUAD_MODEL_SEARCH false",
        "SGTELIB_MODEL_SEARCH false",
        "EVAL_QUEUE_SORT DIR_LAST_SUCCESS",
        "SPECULATIVE_SEARCH true",
        "NM_SEARCH true",
        "EVAL_OPPORTUNISTIC true",
        "BB_MAX_BLOCK_SIZE 1",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "ADD_SEED_TO_FILE_NAMES false",
        f"STATS_FILE {destination.name} BBE OBJ SOL",
    ]
    history_x, history_fun = [], []
    callback_errors = []

    def objective(point):
        try:
            if len(history_fun) >= budget:
                raise RuntimeError("NOMAD attempted a query beyond MAX_BB_EVAL")
            x = np.array([point.get_coord(i) for i in range(point.size())], dtype=float)
            if np.any(x < lower - 1e-12) or np.any(x > upper + 1e-12):
                raise RuntimeError("NOMAD supplied a point outside bounds")
            value = float(fun(x.copy()))
            if not np.isfinite(value):
                raise ValueError("finite scalar feedback is required")
            history_x.append(x.tolist())
            history_fun.append(value)
            point.setBBO(format(value, ".17g").encode("ascii"))
            return 1
        except Exception as exc:
            # Cython callback exception propagation is not guaranteed. Preserve
            # the first cause and reject the whole run after native unwinding.
            callback_errors.append(exc)
            return 0

    began = time.perf_counter()
    # NOMAD's statistics parser splits absolute paths at whitespace. A local
    # filename inside the case directory keeps installation/output paths portable.
    if any(character.isspace() for character in destination.name):
        raise ValueError("NOMAD statistics basename must not contain whitespace")
    previous_directory = Path.cwd()
    try:
        os.chdir(destination.parent)
        PyNomad.resetRandomNumberGenerator()
        native = PyNomad.optimize(
            objective, x0.tolist(), lower.tolist(), upper.tolist(), parameters
        )
    finally:
        os.chdir(previous_directory)
    if callback_errors:
        raise RuntimeError("NOMAD callback failed") from callback_errors[0]
    if int(native["nb_evals"]) != len(history_fun):
        raise RuntimeError("NOMAD native evaluation count differs from paid count")
    if native["run_flag"] < 0 or not native["x_best"]:
        raise RuntimeError(f"NOMAD returned no feasible recommendation: {native}")

    stats_events = []
    for line in destination.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != len(x0) + 2:
            raise RuntimeError(f"Unexpected NOMAD native statistics row: {line}")
        calls = int(fields[0])
        value = float(fields[1])
        x = [float(v) for v in fields[2:]]
        if not 1 <= calls <= len(history_fun):
            raise RuntimeError("NOMAD event cost outside paid history")
        # Event output is full precision. Record which paid point it identifies;
        # do not turn it into a locally selected minimum or reevaluate it.
        matching = [
            i + 1
            for i, (h, y) in enumerate(zip(history_x[:calls], history_fun[:calls]))
            if h == x and y == value
        ]
        if not matching:
            raise RuntimeError("NOMAD native event is not a paid observation")
        stats_events.append(
            dict(
                calls=calls,
                actual_calls=calls,
                x=x,
                fun=value,
                observed_value=value,
                paid_point_call=matching[-1],
                source="NOMAD4.4 native STATS_FILE incumbent event",
            )
        )
    if not stats_events:
        raise RuntimeError("NOMAD produced no native improvement statistics")
    endpoint = dict(
        calls=len(history_fun),
        actual_calls=len(history_fun),
        x=native["x_best"],
        fun=float(native["f_best"]),
        observed_value=float(native["f_best"]),
        source="NOMAD4.4 native cache-best endpoint",
    )
    events, replay_checks = [], []
    for cap in sorted(set(checkpoint_calls or [])):
        if int(cap) != cap or not 1 <= cap <= budget:
            raise ValueError("checkpoint_calls must be integers between1 and budget")
        cap = int(cap)
        if cap >= len(history_fun):
            continue  # The actual native endpoint below carries forward.
        cursor = [0]
        maxdiff = [0.0]

        def replay(x):
            i = cursor[0]
            if i >= len(history_fun):
                raise RuntimeError("Replay requested a new observation")
            expected = np.asarray(history_x[i])
            maxdiff[0] = max(maxdiff[0], float(np.max(np.abs(x - expected))))
            if not np.array_equal(x, expected):
                raise RuntimeError("NOMAD replay diverged from the paid trajectory")
            cursor[0] += 1
            return history_fun[i]

        replay_path = destination.with_name(
            destination.stem + f".prefix_{cap}" + destination.suffix
        )
        answer = minimize(
            replay, x0, cap, lower=lower, upper=upper, stats_path=replay_path, seed=seed
        )
        if answer["nfev"] != cursor[0]:
            raise RuntimeError("NOMAD replay accounting mismatch")
        events.append(
            dict(
                calls=cursor[0],
                actual_calls=cursor[0],
                x=answer["x"],
                fun=answer["fun"],
                observed_value=answer["fun"],
                source="NOMAD4.4 native cache-best from exact paid-prefix replay",
            )
        )
        replay_checks.append(
            dict(
                requested_cap=cap,
                replayed_feedback=cursor[0],
                max_abs_coordinate_difference=maxdiff[0],
                exact_prefix=True,
                additional_physical_queries=0,
            )
        )
    events.append(endpoint)
    events.sort(key=lambda event: event["calls"])
    return dict(
        method="nomad",
        x=native["x_best"],
        fun=float(native["f_best"]),
        nfev=len(history_fun),
        history_x=history_x,
        history_fun=history_fun,
        native_result=native,
        native_recommendation_source="PyNomad.optimize().x_best",
        events=events,
        recommendation_events=events,
        status="native_stop",
        native_stats_events=stats_events,
        stats_last_matches_native=(
            stats_events[-1]["x"] == native["x_best"]
            and stats_events[-1]["fun"] == native["f_best"]
        ),
        replay_checks=replay_checks,
        replayed_feedback_calls=sum(r["replayed_feedback"] for r in replay_checks),
        params=parameters,
        budget=budget,
        seed=seed,
        algorithm_seed=0,
        seconds=time.perf_counter() - began,
        native_stats_path=str(destination),
        implementation="official PyNomadBBO4.4.0",
        published_wrapper_reference="SQNomad0.2.3 embeds NOMAD4.0.0",
        provenance="checked SQNomad settings mapped to official4.4; not identical4.0 trajectories",
        physical_queries_for_diagnostics=0,
    )
