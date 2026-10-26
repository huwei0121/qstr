"""Pack's six algorithm families with explicit, locally reconstructed settings.

The paper does not publish its PWC optimizer code or complete hyperparameters.
The implementation uses standard library kernels with the recorded
reconstruction parameters, which are supplied through ``options``.
All objective evaluations are recorded. Native recommendations are kept apart
from the minimum observed value; no physical quality oracle is accessed.
"""

from __future__ import annotations

import importlib.metadata
import time

import numpy as np
from qstr.baselines import minimize_reference


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


ALGORITHMS = (
    "cma_es",
    "nelder_mead",
    "powell",
    "differential_evolution",
    "one_plus_one",
    "simulated_annealing",
)


def _reflect(x, lower, upper):
    width = upper - lower
    return lower + width - np.abs((x - lower) % (2 * width) - width)


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def minimize(fun, x0, budget, *, method, lower, upper, seed=0, options=None):
    """Run one of the six families under a strict paid-call cap.

    The normalized, bounded physical coordinates are those passed by the caller.
    Required settings: CMA/NM/1+1 ``sigma``; DE ``sigma,popsize,mutation,
    recombination``; SA ``sigma,temperature,gamma``. CMA additionally accepts
    official CMAOptions; NM/Powell official SciPy options. DE popsize is the
    *absolute* population size, not SciPy's dimension multiplier.

    NM starts at a seeded random orthogonal simplex, DE at a seeded reflected
    Gaussian population containing x0, and 1+1 uses Nevergrad's bouncing bounds.
    These initializations and box treatments are local reconstruction choices.
    SA is Gaussian Metropolis with geometric cooling and the best-so-far return
    of Algorithms for Optimization (2019), Algorithm 8.4. Its chain endpoint is
    returned separately. Native library stopping is honored. No restarts,
    resampling, or free initial objective query are added.

    A complete paid prefix can be deterministically replayed using this same
    entry point and saved objective values. A truncated CMA population remains
    untold, exactly as in standard_baselines; it cannot replace native xbest.
    """
    if method not in ALGORITHMS:
        raise ValueError(f"method must be one of {ALGORITHMS}")
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or not len(x0) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a nonempty finite vector")
    lower = np.broadcast_to(np.asarray(lower, float), x0.shape).copy()
    upper = np.broadcast_to(np.asarray(upper, float), x0.shape).copy()
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(x0 < lower)
        or np.any(x0 > upper)
    ):
        raise ValueError("finite strict bounds containing x0 are required")
    opts = dict(options or {})
    supplied = _plain(opts)
    rng = np.random.default_rng(seed)
    metadata = dict(
        adapter="pulse_calibration",
        provenance="published algorithm families; PWC reconstruction",
        published_complete_pwc_settings_available=False,
        versions={p: _version(p) for p in ("numpy", "scipy", "cma", "nevergrad")},
    )

    if method in ("cma_es", "nelder_mead", "powell"):
        if method == "nelder_mead":
            sigma = float(opts.pop("sigma"))
            if sigma <= 0 or not np.isfinite(sigma):
                raise ValueError("sigma must be positive and finite")
            if "initial_simplex" in opts:
                raise ValueError("sigma already defines the local initial simplex")
            q, _ = np.linalg.qr(rng.normal(size=(len(x0), len(x0))))
            opts["initial_simplex"] = np.vstack([x0, _reflect(x0 + sigma * q, lower, upper)])
        result = minimize_reference(
            fun, x0, budget, method=method, bounds=(lower, upper), seed=seed, options=opts
        )
        result["local_reconstruction_options"] = supplied
        result.update(metadata)
        result["return_kind"] = (
            "no_native_recommendation" if result["x"] is None else "native_solver_recommendation"
        )
        result["native_recommendation"] = result["x"]
        return _plain(result)

    allowed = dict(
        differential_evolution={"sigma", "popsize", "mutation", "recombination"},
        one_plus_one={"sigma"},
        simulated_annealing={"sigma", "temperature", "gamma"},
    )[method]
    if set(opts) != allowed:
        raise ValueError(f"{method} requires exactly {sorted(allowed)}")
    sigma = float(opts["sigma"])
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    history_x, history_fun = [], []
    budget_guard_triggered = False

    def objective(x):
        nonlocal budget_guard_triggered
        if len(history_fun) >= budget:
            # SciPy 1.10's immediate DE tests _nfev > maxfun, allowing one
            # extra callback. StopIteration is the solver's own budget-stop
            # signal and is caught by solve() before any extra physical call.
            budget_guard_triggered = True
            raise StopIteration("strict paid objective cap reached")
        x = np.asarray(x, float)
        if np.any(x < lower - 1e-12) or np.any(x > upper + 1e-12):
            raise ValueError("native solver supplied an infeasible point")
        value = float(fun(x.copy()))
        if not np.isfinite(value):
            raise ValueError("objective must return finite scalar feedback")
        history_x.append(x.tolist())
        history_fun.append(value)
        return value

    started = time.perf_counter()
    native_result, stop, extra = {}, {}, {}
    if method == "differential_evolution":
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

        popsize = int(opts["popsize"])
        if popsize < 5 or popsize != opts["popsize"]:
            raise ValueError("absolute DE popsize must be an integer >= 5")
        population = _reflect(x0 + sigma * rng.normal(size=(popsize, len(x0))), lower, upper)
        population[0] = x0
        # Keep official convergence tolerances. No polishing: that would add a
        # different local optimizer to the paper's DE family and extra queries.
        with DifferentialEvolutionSolver(
            objective,
            list(zip(lower, upper)),
            strategy="rand1bin",
            init=population,
            mutation=opts["mutation"],
            recombination=opts["recombination"],
            seed=int(seed),
            maxfun=budget,
            maxiter=budget,
            polish=False,
            updating="immediate",
            workers=1,
        ) as solver:
            native = solver.solve()
            if int(native.nfev) != len(history_fun):
                raise RuntimeError("DE native evaluation count differs from paid count")
            x, value = native.x, float(native.fun)
            native_result = _plain(dict(native))
            stop = {"success": bool(native.success), "message": str(native.message)}
            extra["population_size"] = int(solver.num_population_members)
        source = "scipy.DifferentialEvolutionSolver.solve().x"
        extra["budget_guard_triggered"] = budget_guard_triggered
        extra["effective_options"] = dict(
            strategy="rand1bin",
            popsize_absolute=popsize,
            init="reflected Gaussian with x0 first",
            maxfun=budget,
            maxiter=budget,
            polish=False,
            updating="immediate",
            workers=1,
            mutation=opts["mutation"],
            recombination=opts["recombination"],
        )
    elif method == "one_plus_one":
        import nevergrad as ng

        parameter = ng.p.Array(init=x0).set_mutation(sigma=sigma)
        parameter.set_bounds(lower, upper, method="bouncing")
        parameter.random_state.seed(int(seed))
        solver = ng.optimizers.OnePlusOne(parametrization=parameter, budget=budget, num_workers=1)
        for _ in range(budget):
            candidate = solver.ask()
            solver.tell(candidate, objective(candidate.value))
        recommendation = solver.provide_recommendation()
        x = np.asarray(recommendation.value, float)
        # Native recommendation loss is an archive estimate. It is not a free
        # oracle evaluation and is not replaced by the minimum noisy feedback.
        native_loss = recommendation.loss
        value = None if native_loss is None else float(native_loss)
        native_result = dict(
            x=x.tolist(), loss=value, num_ask=int(solver.num_ask), num_tell=int(solver.num_tell)
        )
        source = "nevergrad.optimizers.OnePlusOne.provide_recommendation().value"
        extra["effective_options"] = dict(
            sigma=sigma, bounds_method="bouncing", budget=budget, num_workers=1
        )
    else:
        temperature = float(opts["temperature"])
        gamma = float(opts["gamma"])
        if not np.isfinite(temperature) or temperature <= 0 or not 0 < gamma < 1:
            raise ValueError("SA requires temperature > 0 and 0 < gamma < 1")
        chain_x = x0.copy()
        chain_value = objective(chain_x)
        x, value = chain_x.copy(), chain_value
        accepted = 0
        for k in range(budget - 1):
            candidate = _reflect(chain_x + sigma * rng.normal(size=len(x0)), lower, upper)
            candidate_value = objective(candidate)
            delta = candidate_value - chain_value
            current_temperature = max(np.finfo(float).tiny, temperature * gamma**k)
            if delta <= 0 or np.log(rng.uniform()) < -delta / current_temperature:
                chain_x, chain_value = candidate.copy(), candidate_value
                accepted += 1
            if candidate_value < value:
                x, value = candidate.copy(), candidate_value
        source = "AFO2019 Algorithm8.4 best-so-far return (local implementation)"
        native_result = dict(
            x=x.tolist(),
            fun=value,
            chain_x=chain_x.tolist(),
            chain_fun=chain_value,
            accepted_proposals=accepted,
            proposals=budget - 1,
        )
        extra["effective_options"] = dict(
            sigma=sigma,
            temperature=temperature,
            gamma=gamma,
            proposal="isotropic Gaussian; reflected bounds",
            cooling="T(k)=T0*gamma**k, k=0..budget-2",
            return_rule="best-so-far as in Algorithm 8.4",
        )

    best = int(np.argmin(history_fun))
    result = dict(
        method=method,
        x=np.asarray(x).tolist(),
        fun=value,
        native_recommendation=np.asarray(x).tolist(),
        native_recommendation_source=source,
        return_kind="native_solver_recommendation",
        native_result=native_result,
        nfev=len(history_fun),
        history_x=history_x,
        history_fun=history_fun,
        observed_best_x=history_x[best],
        observed_best_fun=history_fun[best],
        status="budget_cap" if len(history_fun) == budget else "native_stop",
        stop=stop,
        supplied_options=supplied,
        options=opts,
        budget=budget,
        seed=seed,
        seconds=time.perf_counter() - started,
    )
    result.update(extra)
    result.update(metadata)
    return _plain(result)
