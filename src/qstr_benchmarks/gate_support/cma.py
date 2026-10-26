"""Unmodified C3 CMA wrapper with native return and explicit shot-budget adapter."""

import hashlib
import time
from pathlib import Path

import numpy as np


def canonical_point(z):
    """Quantity.set_opt_value at upstream commit48b7917, vectorized in NumPy."""
    z = np.asarray(z, dtype=float)
    return np.arccos(np.clip(np.cos((z + 1) * np.pi / 2), -1.0, 1.0)) / np.pi * 2 - 1


def minimize(fun, x0, budget, *, seed, options=None):
    """Run the upstream wrapper; all unbounded proposals and its xbest are reflected.

    Hard-cap interruption preserves xbest from completed CMA generations.
    Partial observations are charged and retained, never substituted for xbest.
    """
    import c3.c3objs as quantities
    import c3.libraries.algorithms as algorithms
    import cma.evolution_strategy as cma
    from c3.libraries.algorithms import cmaes

    budget = int(budget)
    if budget < 1:
        raise ValueError("Positive atomic-call budget required")
    x0 = np.asarray(x0, dtype=float)
    opts = dict(popsize=25, spread=0.1, init_point=False)
    opts.update(options or {})
    if opts.get("init_point") is not False:
        raise ValueError("This paper-population/default-wrapper protocol requires init_point=False")
    if "bounds" in opts:
        raise ValueError("Upstream Quantity reflection is used, not CMA bound handling")
    opts["maxfevals"] = min(budget, int(opts.get("maxfevals", budget)))
    opts["seed"] = int(seed) if seed else 1
    opts.setdefault("verbose", -9)
    opts.setdefault("verb_log", 0)
    supplied = dict(opts)
    original = cma.CMAEvolutionStrategy
    holder = {}
    unbounded = []
    physical = []
    values = []

    class BudgetStop(Exception):
        pass

    class CapturedCMA(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["es"] = self

    def paid(z):
        if len(values) >= budget:
            raise BudgetStop()
        z = np.asarray(z, dtype=float)
        x = canonical_point(z)
        value = float(fun(x.copy()))
        if not np.isfinite(value):
            raise ValueError("Nonfinite paid feedback")
        unbounded.append(z.tolist())
        physical.append(x.tolist())
        values.append(value)
        return value

    started = time.perf_counter()
    limited = False
    cma.CMAEvolutionStrategy = CapturedCMA
    try:
        cmaes(x0, fun=paid, options=dict(opts))
    except BudgetStop:
        limited = True
    finally:
        cma.CMAEvolutionStrategy = original
    es = holder["es"]
    raw_best = es.result.xbest
    raw_favorite = es.result.xfavorite
    native = None if raw_best is None else canonical_point(raw_best).tolist()
    finished = int(es.result.evaluations)
    best = None if not values else int(np.argmin(values))
    return dict(
        method="cma_es",
        x=native,
        fun=None if native is None else float(es.result.fbest),
        nfev=len(values),
        history_x=physical,
        history_unbounded_x=unbounded,
        history_fun=values,
        native_unbounded_x=None if raw_best is None else np.asarray(raw_best).tolist(),
        favorite_unbounded_x=None if raw_favorite is None else np.asarray(raw_favorite).tolist(),
        favorite_canonical_x=None
        if raw_favorite is None
        else canonical_point(raw_favorite).tolist(),
        observed_best_x=None if best is None else physical[best],
        observed_best_fun=None if best is None else values[best],
        complete_generations=int(es.result.iterations),
        complete_generation_evaluations=finished,
        partial_population=[
            dict(x=x, unbounded_x=z, fun=y)
            for x, z, y in zip(physical[finished:], unbounded[finished:], values[finished:])
        ],
        native_recommendation_source="Unmodified c3.libraries.algorithms.cmaes / es.result.xbest, mapped by upstream Quantity reflection",
        stop=dict(es.stop()),
        status="strict_hard_cap" if limited else "native_stop",
        supplied_options=supplied,
        budget=budget,
        seed=seed,
        seconds=time.perf_counter() - started,
        upstream_commit="48b7917eb8d84e1471ff9fb911129c354e752dbf",
        upstream_wrapper_sha256=hashlib.sha256(Path(algorithms.__file__).read_bytes()).hexdigest(),
        upstream_quantity_sha256=hashlib.sha256(Path(quantities.__file__).read_bytes()).hexdigest(),
        protocol="Paper Figure2 population25, upstream-wrapper default spread.1/init_pointFalse; no fake additive noise; strict atomic budget; unbounded latent CMA and upstream reflection",
    )
