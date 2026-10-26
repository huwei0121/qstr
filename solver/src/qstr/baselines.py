"""Standard reference solvers with explicit evaluation accounting.

SciPy supplies Nelder--Mead and Powell; the optional ``cma`` dependency supplies
CMA-ES. Native stopping rules, recommendations, and tolerances are preserved.
The common evaluation cap is enforced without replacing a native recommendation
by the smallest value observed during the run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["minimize_reference"]

ReferenceMethod = Literal["nelder_mead", "powell", "cma_es"]


def _serializable(value: Any) -> Any:
    """Convert numerical result metadata to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def minimize_reference(
    objective: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    max_evaluations: int,
    *,
    method: ReferenceMethod,
    bounds: Sequence[ArrayLike | None] | None = None,
    seed: int = 0,
    options: Mapping[str, Any] | None = None,
    scale: ArrayLike | None = None,
) -> dict[str, Any]:
    """Minimize a scalar objective using one standard solver invocation.

    Parameters
    ----------
    objective : callable
        Function receiving a physical parameter vector and returning one scalar
        observation. Each call counts once, including initialization calls.
    x0 : array_like, shape (n,)
        Finite initial parameter vector.
    max_evaluations : int
        Positive upper bound on objective calls.
    method : {'nelder_mead', 'powell', 'cma_es'}
        Underlying reference algorithm.
    bounds : (lower, upper), optional
        Physical bounds. Each side accepts a scalar, an array of length n, or
        None for an unbounded side. Every lower bound must be strictly smaller
        than its upper bound, and x0 must be feasible.
    seed : int, default 0
        CMA-ES random seed, unless options supplies its own seed. The default
        zero maps to one because pycma uses zero for time-dependent seeding.
        This parameter does not alter randomness inside objective.
    options : mapping, optional
        Options passed to scipy.optimize.minimize or CMAEvolutionStrategy.
        CMA-ES requires a positive sigma0 (sigma is also accepted). maxfev or
        maxfevals is capped by max_evaluations. Other stopping settings are
        unchanged. Bounds must be supplied through the bounds argument.
    scale : array_like, optional
        Positive diagonal scaling: x = scale * z. Solver options, including
        initial_simplex and sigma0, are expressed in z coordinates.

    Returns
    -------
    dict
        x and fun contain the native recommendation; nfev, history_x, and
        history_fun describe all paid observations. observed_best_x and
        observed_best_fun are recorded separately. Top-level points and the
        history use physical coordinates; native_result retains the underlying
        solver's metadata (SciPy's nested x remains in solver coordinates).
        The result is JSON serializable.

    Notes
    -----
    No restart or extra initial evaluation is added. A partial final CMA-ES
    population is evaluated and recorded but never passed to tell. If no
    generation completed, x and fun are None even when observations exist.
    CMA-ES xfavorite is retained separately and is not evaluated for free.
    """
    if method not in ("nelder_mead", "powell", "cma_es"):
        raise ValueError("method must be nelder_mead, powell, or cma_es")
    if (
        isinstance(max_evaluations, bool)
        or int(max_evaluations) != max_evaluations
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    max_evaluations = int(max_evaluations)
    initial_point = np.asarray(x0, dtype=float)
    if initial_point.ndim != 1 or not initial_point.size or not np.all(np.isfinite(initial_point)):
        raise ValueError("x0 must be a nonempty finite vector")
    dimension = initial_point.size
    scaling = (
        np.ones(dimension)
        if scale is None
        else np.broadcast_to(np.asarray(scale, dtype=float), (dimension,)).copy()
    )
    if not np.all(np.isfinite(scaling)) or np.any(scaling <= 0):
        raise ValueError("scale must be positive and finite")
    if bounds is None:
        lower_input, upper_input = None, None
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair")
        lower_input, upper_input = bounds
    lower = (
        np.full(dimension, -np.inf)
        if lower_input is None
        else np.broadcast_to(np.asarray(lower_input, dtype=float), (dimension,)).copy()
    )
    upper = (
        np.full(dimension, np.inf)
        if upper_input is None
        else np.broadcast_to(np.asarray(upper_input, dtype=float), (dimension,)).copy()
    )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower >= upper):
        raise ValueError("bounds must satisfy lower < upper")
    if np.any(initial_point < lower) or np.any(initial_point > upper):
        raise ValueError("x0 must satisfy bounds")
    solver_options = dict(options or {})
    supplied_options = _serializable(solver_options)
    if "bounds" in solver_options:
        raise ValueError("Pass bounds via the bounds argument, not options")

    history_x: list[list[float]] = []
    history_fun: list[float] = []

    def evaluate(coordinates: NDArray[np.float64]) -> float:
        if len(history_fun) >= max_evaluations:
            raise RuntimeError("Reference solver attempted to exceed the evaluation cap")
        point = np.asarray(coordinates, dtype=float) * scaling
        value = float(objective(point.copy()))
        history_x.append(point.tolist())
        history_fun.append(value)
        return value

    def physical_point(coordinates: ArrayLike | None) -> list[float] | None:
        if coordinates is None:
            return None
        return (np.asarray(coordinates, dtype=float) * scaling).tolist()

    started = perf_counter()
    native_x, native_fun = None, None
    native_result, stop = {}, {}
    partial_population: list[dict[str, Any]] = []
    complete_generations = 0
    cma_xbest, cma_xfavorite = None, None
    native_source = "scipy.OptimizeResult.x"
    status = "native_stop"

    if method in ("nelder_mead", "powell"):
        from scipy.optimize import minimize as scipy_minimize

        solver_options["maxfev"] = min(
            max_evaluations, solver_options.get("maxfev", max_evaluations)
        )
        solver_bounds = (
            None
            if lower_input is None and upper_input is None
            else list(zip(lower / scaling, upper / scaling))
        )
        result = scipy_minimize(
            evaluate,
            initial_point / scaling,
            method="Nelder-Mead" if method == "nelder_mead" else "Powell",
            bounds=solver_bounds,
            options=solver_options,
        )
        native_x, native_fun = physical_point(result.x), float(result.fun)
        native_result = _serializable(dict(result))
        stop = {
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
        }
        if len(history_fun) >= max_evaluations:
            status = "budget_cap"
    else:
        import cma

        if "sigma0" in solver_options and "sigma" in solver_options:
            raise ValueError("Use one of sigma0 or sigma")
        sigma = solver_options.pop("sigma0", solver_options.pop("sigma", None))
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("CMA-ES requires positive finite options['sigma0']")
        solver_options["maxfevals"] = min(
            max_evaluations, solver_options.get("maxfevals", max_evaluations)
        )
        solver_options.setdefault("seed", int(seed) if seed else 1)
        solver_options.setdefault("verbose", -9)
        solver_options.setdefault("verb_log", 0)
        if lower_input is not None or upper_input is not None:
            solver_options["bounds"] = [
                [None if not np.isfinite(value) else float(value) for value in lower / scaling],
                [None if not np.isfinite(value) else float(value) for value in upper / scaling],
            ]
        strategy = cma.CMAEvolutionStrategy(initial_point / scaling, float(sigma), solver_options)
        while len(history_fun) < max_evaluations and not strategy.stop():
            population = strategy.ask()
            values = [evaluate(point) for point in population[: max_evaluations - len(history_fun)]]
            if len(values) < len(population):
                partial_population = [
                    {"x": physical_point(point), "fun": value}
                    for point, value in zip(population, values)
                ]
                break
            strategy.tell(population, values)
            complete_generations += 1
        cma_xbest = physical_point(strategy.result.xbest)
        cma_xfavorite = physical_point(strategy.result.xfavorite)
        native_x = cma_xbest
        native_fun = None if native_x is None else float(strategy.result.fbest)
        native_source = "cma.CMAEvolutionStrategy.result.xbest"
        stop = _serializable(dict(strategy.stop()))
        native_result = {
            "xbest": cma_xbest,
            "fbest": native_fun,
            "xfavorite": cma_xfavorite,
            "evaluations": int(strategy.result.evaluations),
            "iterations": int(strategy.result.iterations),
            "population": int(strategy.popsize),
        }
        solver_options["sigma0"] = float(sigma)
        if len(history_fun) >= max_evaluations:
            status = "budget_cap_partial_population" if partial_population else "budget_cap"

    best_index = int(np.argmin(history_fun)) if history_fun else None
    return _serializable(
        {
            "method": method,
            "x": native_x,
            "fun": native_fun,
            "nfev": len(history_fun),
            "history_x": history_x,
            "history_fun": history_fun,
            "observed_best_x": None if best_index is None else history_x[best_index],
            "observed_best_fun": None if best_index is None else history_fun[best_index],
            "native_recommendation_source": native_source,
            "native_result": native_result,
            "cma_xbest": cma_xbest,
            "cma_xfavorite": cma_xfavorite,
            "complete_generations": complete_generations,
            "partial_population": partial_population,
            "status": status,
            "stop": stop,
            "options": solver_options,
            "supplied_options": supplied_options,
            "scale": scaling,
            "max_evaluations": max_evaluations,
            "seed": seed,
            "seconds": perf_counter() - started,
        }
    )
