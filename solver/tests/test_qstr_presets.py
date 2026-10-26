"""Coordinate mathematics and public problem configurations."""

import numpy as np
import pytest

from qstr.bases import (
    dct_basis,
    make_basis_factory,
    pulse_basis,
    qaoa_layer_basis,
    state_transfer_basis,
)
from qstr.coordinates import anchored_box_map
from qstr.options import SolverOptions
from qstr.presets import get_preset, minimize_with_preset


def test_box_map_preserves_anchor_and_local_scale():
    center = np.array([-0.65, 0.2, 1.4])
    lower = np.array([-1.0, -0.5, 0.0])
    upper = np.array([1.0, 2.0, 2.0])
    forward = anchored_box_map(center, lower, upper)
    np.testing.assert_array_equal(forward(np.zeros(3)), center)
    steps = 1e-6 * np.eye(3)
    jacobian = np.column_stack([(forward(step) - forward(-step)) / 2e-6 for step in steps])
    np.testing.assert_allclose(jacobian, np.eye(3), atol=2e-10, rtol=0)
    for displacement in (np.full(3, 1e300), np.full(3, -1e300)):
        point = forward(displacement)
        assert np.all(point > lower)
        assert np.all(point < upper)


def test_box_map_owns_its_anchor_and_checks_displacements():
    center = np.zeros(2)
    lower, upper = -np.ones(2), np.ones(2)
    forward = anchored_box_map(center, lower, upper)
    center[:] = 0.5
    lower[:] = -2.0
    np.testing.assert_array_equal(forward(np.zeros(2)), np.zeros(2))
    with pytest.raises(ValueError, match="displacement"):
        forward(np.zeros(3))
    with pytest.raises(ValueError, match="interior"):
        anchored_box_map([-1.0], [-1.0], [1.0])


@pytest.mark.parametrize("time_slices,modes", [(41, 8), (41, 41), (100, 8), (7, 0)])
def test_cosine_basis_is_orthonormal(time_slices, modes):
    basis = dct_basis(time_slices, modes)
    np.testing.assert_allclose(basis.T @ basis, np.eye(modes), atol=2e-14, rtol=0)
    assert basis.shape == (time_slices, modes)


def test_mixed_pulse_basis_retains_core_and_seed_reproducibility():
    fixed = pulse_basis(8)
    basis = pulse_basis(16, core_modes_per_channel=8, seed=91)
    np.testing.assert_array_equal(basis[:, :16], fixed)
    np.testing.assert_allclose(basis.T @ basis, np.eye(32), atol=2e-14, rtol=0)
    np.testing.assert_array_equal(basis, pulse_basis(16, core_modes_per_channel=8, seed=91))
    changed = pulse_basis(16, core_modes_per_channel=8, seed=92)
    np.testing.assert_array_equal(changed[:, :16], fixed)
    assert not np.array_equal(changed[:, 16:], basis[:, 16:])


def test_qaoa_rotation_diagonalizes_coupled_layer_directions():
    layers = 5
    ordering = np.tril(np.ones((layers, layers)))
    hessian = np.block([[np.zeros_like(ordering), ordering], [ordering.T, np.zeros_like(ordering)]])
    rotation = qaoa_layer_basis(layers)
    singular_values = np.linalg.svd(ordering, compute_uv=False)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(2 * layers), atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        rotation.T @ hessian @ rotation,
        np.diag(np.concatenate((-singular_values, singular_values))),
        atol=2e-14,
        rtol=0,
    )


def test_state_transfer_basis_keeps_smooth_modes_and_independent_complement():
    first, second = state_transfer_basis(17), state_transfer_basis(18)
    np.testing.assert_allclose(first[:, :8], dct_basis(100, 8), atol=2e-15, rtol=0)
    np.testing.assert_array_equal(first[:, :8], second[:, :8])
    np.testing.assert_allclose(first.T @ first, np.eye(12), atol=2e-14, rtol=0)
    assert not np.array_equal(first[:, 8:], second[:, 8:])


def test_basis_factory_changes_dimension_after_warmup():
    factory = make_basis_factory("pulse_calibration", warmup_stages=1)
    assert factory(0, 173).shape == (82, 16)
    assert factory(1, 1182).shape == (82, 32)
    assert make_basis_factory("gate_calibration") is None
    assert make_basis_factory("qaoa_maxcut")(0, 173).shape == (10, 10)
    assert make_basis_factory("ising_state_transfer")(0, 173).shape == (100, 12)


def test_preset_copies_do_not_share_configuration():
    first = get_preset("ising_state_transfer")
    first["x0"][0] = 1.0
    first["options"]["repetitions"] = 9
    first["evaluation_cost"]["amount"] = 100
    second = get_preset("ising_state_transfer")
    assert second["x0"][0] == 0.0
    assert second["options"]["repetitions"] == 3
    assert second["evaluation_cost"]["amount"] == 1
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("missing")


@pytest.mark.parametrize(
    "preset,dimension,budget",
    [
        ("pulse_calibration", 82, 1000),
        ("gate_calibration", 3, 100),
        ("qaoa_maxcut", 10, 550),
        ("ising_state_transfer", 100, 2000),
    ],
)
def test_presets_evaluate_in_physical_coordinates_and_respect_budget(preset, dimension, budget):
    configuration = get_preset(preset)
    assert configuration["dimension"] == dimension
    assert configuration["max_evaluations"] == budget
    initial_point = np.zeros(dimension) if configuration["x0"] is None else configuration["x0"]
    evaluated = []

    def objective(point):
        assert point.shape == (dimension,)
        evaluated.append(point.copy())
        return float(np.sum((point - 0.05) ** 2))

    result = minimize_with_preset(
        objective, preset=preset, x0=initial_point, max_evaluations=20, seed=41
    )
    assert 1 <= result.nfev <= 20
    assert result.metadata["preset"] == preset
    assert len(evaluated) == result.nfev
    np.testing.assert_array_equal(result.history.x[0], initial_point)
    np.testing.assert_array_equal(result.history.x, evaluated)
    if configuration["bounds"] is not None:
        lower, upper = configuration["bounds"]
        assert np.all(result.history.x >= lower)
        assert np.all(result.history.x <= upper)


def test_option_overrides_and_full_settings_have_distinct_meanings():
    def objective(point):
        return float(point @ point)

    partial = minimize_with_preset(
        objective,
        preset="gate_calibration",
        max_evaluations=12,
        options={"initial_radius": 0.04},
    )
    assert partial.options.initial_radius == 0.04
    assert partial.options.interpolation_factor == 3.0
    complete = minimize_with_preset(
        objective,
        preset="gate_calibration",
        max_evaluations=12,
        options=SolverOptions(initial_radius=0.04),
    )
    assert complete.options.initial_radius == 0.04
    assert complete.options.interpolation_factor == 2.0


def test_preset_checks_required_initial_point_and_dimension():
    with pytest.raises(ValueError, match="instance-specific"):
        minimize_with_preset(lambda point: 0.0, preset="pulse_calibration")
    with pytest.raises(ValueError, match="dimension 10"):
        minimize_with_preset(lambda point: 0.0, preset="qaoa_maxcut", x0=[0.0])
