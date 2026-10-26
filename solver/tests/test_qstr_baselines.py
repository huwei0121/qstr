"""Check paid evaluations, native recommendations, and explicit coordinates."""

import json

import numpy as np
import pytest

from qstr.baselines import minimize_reference


@pytest.mark.parametrize(
    "method, options",
    [("nelder_mead", {}), ("powell", {}), ("cma_es", {"sigma0": 0.2, "popsize": 4})],
)
def test_evaluation_cap_and_paid_history(method, options):
    if method == "cma_es":
        pytest.importorskip("cma")
    calls = []

    def objective(point):
        calls.append(point.copy())
        return float(np.sum((point - 0.3) ** 2))

    result = minimize_reference(objective, np.zeros(2), 7, method=method, options=options, seed=91)
    assert 0 < result["nfev"] <= 7
    assert result["nfev"] == len(calls) == len(result["history_fun"])
    assert result["history_x"] == [point.tolist() for point in calls]
    json.dumps(result, allow_nan=False)


def test_partial_population_does_not_replace_native_recommendation():
    pytest.importorskip("cma")
    calls = []

    def objective(point):
        calls.append(point.copy())
        return -float(len(calls))

    result = minimize_reference(
        objective,
        np.zeros(2),
        5,
        method="cma_es",
        seed=92,
        options={"sigma0": 0.2, "popsize": 4},
    )
    assert result["complete_generations"] == 1
    assert len(result["partial_population"]) == 1
    assert result["fun"] == -4 and result["observed_best_fun"] == -5
    assert result["x"] != result["observed_best_x"]
    assert result["status"] == "budget_cap_partial_population"


def test_no_complete_population_has_no_fabricated_native_best():
    pytest.importorskip("cma")
    result = minimize_reference(
        lambda point: float(point @ point),
        np.zeros(2),
        2,
        method="cma_es",
        seed=93,
        options={"sigma0": 0.2, "popsize": 4},
    )
    assert result["nfev"] == 2 and result["complete_generations"] == 0
    assert result["x"] is None and result["fun"] is None
    assert result["observed_best_x"] is not None
    assert len(result["partial_population"]) == 2


@pytest.mark.parametrize(
    "method, options",
    [
        ("nelder_mead", {"initial_simplex": np.array([[0.0, 0.0], [-0.2, 0.0], [0.0, 0.2]])}),
        ("powell", {}),
        ("cma_es", {"sigma0": 0.2, "popsize": 4}),
    ],
)
def test_bounds_and_diagonal_scaling_use_physical_coordinates(method, options):
    if method == "cma_es":
        pytest.importorskip("cma")
    result = minimize_reference(
        lambda point: float((point[0] + 3) ** 2 + (point[1] - 0.4) ** 2),
        np.zeros(2),
        80,
        method=method,
        bounds=([-2, -1], [2, 1]),
        scale=[2, 1],
        options=options,
        seed=94,
    )
    points = np.asarray(result["history_x"])
    assert np.all(points >= [-2, -1]) and np.all(points <= [2, 1])
    assert result["observed_best_fun"] < 2


def test_native_early_stop_preserves_default_tolerances():
    result = minimize_reference(lambda point: 1.0, np.zeros(2), 200, method="nelder_mead")
    assert result["nfev"] < 200 and result["status"] == "native_stop"
    assert "xatol" not in result["options"] and "fatol" not in result["options"]


def test_solver_options_are_not_mutated():
    options = {"maxfev": 100, "xatol": 0.01}
    result = minimize_reference(
        lambda point: float(point @ point),
        np.ones(2),
        6,
        method="nelder_mead",
        options=options,
    )
    assert options == {"maxfev": 100, "xatol": 0.01}
    assert result["options"]["maxfev"] == 6
    assert result["supplied_options"] == options


def test_invalid_bounds_are_rejected_before_evaluation():
    def objective(point):
        pytest.fail("Invalid input must not consume an evaluation")

    with pytest.raises(ValueError, match="x0 must satisfy bounds"):
        minimize_reference(objective, [2.0], 10, method="powell", bounds=(-1, 1))
    with pytest.raises(ValueError, match="lower < upper"):
        minimize_reference(objective, [0.0], 10, method="powell", bounds=(1, 1))


def test_unbounded_side_and_scalar_bounds():
    result = minimize_reference(
        lambda point: float(np.sum((point + 2) ** 2)),
        np.zeros(2),
        30,
        method="powell",
        bounds=(-1, None),
    )
    assert np.all(np.asarray(result["history_x"]) >= -1)
