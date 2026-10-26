"""Budget accounting, recommendation semantics, and structured optimization."""

import json

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from qstr.options import SolverOptions
from qstr.solver import minimize


def test_noisy_bounded_optimization_records_all_paid_observations():
    random = np.random.default_rng(417)
    paid_points = []

    def objective(point):
        assert np.all(point >= -1) and np.all(point <= 1)
        paid_points.append(point.copy())
        return float(np.sum((point - 0.2) ** 2) + random.normal(0, 0.002))

    result = minimize(
        objective,
        np.zeros(3),
        80,
        bounds=(-np.ones(3), np.ones(3)),
        seed=11,
        options=SolverOptions(
            stage_evaluations=25,
            repetitions=2,
            initial_radius=0.1,
            final_radius=0.003,
        ),
    )
    assert result.nfev == len(paid_points) <= 80
    np.testing.assert_array_equal(result.history.x, paid_points)
    assert sum(group.repetitions for group in result.observation_groups) == result.nfev
    previous_end = 0
    for group in result.observation_groups:
        assert group.repetitions == group.end_evaluation - previous_end
        observations = result.history.fun[previous_end : group.end_evaluation]
        assert group.mean == float(np.mean(observations))
        previous_end = group.end_evaluation
    best_group = min(result.observation_groups, key=lambda group: group.mean)
    assert result.fun == best_group.mean
    np.testing.assert_array_equal(result.x, best_group.x)
    assert result.fun < 0.02
    assert json.loads(json.dumps(result.to_dict()))["nfev"] == result.nfev


def test_partial_initial_group_respects_budget_and_uses_group_mean():
    values = iter([3.0, 1.0])
    result = minimize(
        lambda point: next(values),
        [0.0],
        2,
        options={"repetitions": 3},
    )
    assert result.nfev == 2
    assert result.fun == 2.0
    assert len(result.observation_groups) == 1
    assert result.observation_groups[0].repetitions == 2
    assert result.stages == ()


def test_structured_box_search_reduces_objective_without_leaving_bounds():
    basis = np.array([[1.0], [1.0]]) / np.sqrt(2)
    result = minimize(
        lambda point: float(np.sum((point - 0.3) ** 2)),
        np.zeros(2),
        50,
        bounds=(-np.ones(2), np.ones(2)),
        basis_factory=lambda stage, seed: basis,
        options={"initial_radius": 0.2, "restart_reserve_factor": 0},
    )
    assert result.fun < 1e-8
    assert np.all(result.history.x > -1) and np.all(result.history.x < 1)
    np.testing.assert_allclose(result.history.x[:, 0], result.history.x[:, 1])
    assert all(stage.dimension == 1 for stage in result.stages)
    assert all(stage.coordinate_map == "anchored_tanh_subspace" for stage in result.stages)


def test_short_tail_extends_active_stage_without_speculative_basis_call(monkeypatch):
    import cobyqa

    basis_calls = []

    def basis_factory(stage, seed):
        basis_calls.append((stage, seed))
        return None

    def local_solver(objective, x0, bounds=None, options=None):
        for evaluation in range(options["maxfev"]):
            value = objective(x0 + 0.0001 * evaluation)
        return OptimizeResult(x=x0, fun=value, message="evaluation limit")

    monkeypatch.setattr(cobyqa, "minimize", local_solver)
    result = minimize(
        lambda point: float(point @ point),
        np.zeros(3),
        70,
        seed=4,
        basis_factory=basis_factory,
        options={"stage_evaluations": 30},
    )
    assert result.nfev == 70
    assert basis_calls == [(0, 177), (1, 1186)]
    assert [stage.evaluations for stage in result.stages] == [30, 39]
    assert len(result.restart_decisions) == 1
    decision = result.restart_decisions[0]
    assert decision.action == "extend_active_stage"
    assert decision.avoided_tail_evaluations == 9
    assert decision.reserve == 14


def test_insufficient_restart_budget_stops_after_local_convergence(monkeypatch):
    import cobyqa

    def local_solver(objective, x0, bounds=None, options=None):
        for _ in range(9):
            value = objective(x0)
        return OptimizeResult(x=x0, fun=value, message="local convergence")

    monkeypatch.setattr(cobyqa, "minimize", local_solver)
    result = minimize(lambda point: float(point @ point), np.ones(3), 20)
    assert result.nfev == 10
    assert len(result.stages) == 1
    assert result.restart_decisions[-1].action == "stop_insufficient_budget"
    assert result.restart_decisions[-1].remaining_evaluations == 10


def test_stagnation_restart_preserves_budget_and_incumbent(monkeypatch):
    import cobyqa

    def local_solver(objective, x0, bounds=None, options=None):
        for _ in range(options["maxfev"]):
            value = objective(x0)
        return OptimizeResult(x=x0, fun=value, message="evaluation limit")

    monkeypatch.setattr(cobyqa, "minimize", local_solver)
    result = minimize(
        lambda point: float(point @ point),
        np.ones(2),
        31,
        options={"stagnation_patience": 3, "restart_reserve_factor": 0},
    )
    assert result.nfev == 31
    assert result.stages[0].stop_reason == "stagnation"
    assert result.stages[0].evaluations == 6
    assert len(result.stages) > 1
    np.testing.assert_array_equal(result.x, [1, 1])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"radius": 0.1}, "Unknown solver options"),
        ({"repetitions": 0}, "repetitions"),
        ({"restart_reserve_factor": -1}, "restart_reserve_factor"),
        ({"final_radius": 0.2}, "final_radius"),
    ],
)
def test_invalid_options_fail_before_spending_evaluations(options, message):
    def objective(point):
        pytest.fail("Invalid settings must not consume an objective evaluation")

    with pytest.raises(ValueError, match=message):
        minimize(objective, np.zeros(2), 20, options=options)


def test_nonfinite_objective_feedback_is_rejected():
    with pytest.raises(ValueError, match="nonfinite scalar"):
        minimize(lambda point: np.nan, np.zeros(2), 20)
