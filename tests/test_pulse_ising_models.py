"""Independent physical checks for the bundled pulse and Ising simulators."""

import numpy as np
import pytest
from scipy.linalg import expm


def test_randomized_clifford_recovery():
    pytest.importorskip("numba")
    from qstr_benchmarks.pulse_support.model import (
        IDEAL_CLIFFORDS,
        circuit_probabilities,
        sample_circuits,
    )

    for target in (0, 1):
        circuits = sample_circuits(np.random.default_rng(4201), 30, 60, target)
        probabilities = circuit_probabilities(IDEAL_CLIFFORDS, circuits, target)
        np.testing.assert_allclose(probabilities, 1.0, atol=4e-14, rtol=0.0)


def test_duffing_fixed_step_agrees_with_independent_adaptive_integrator():
    pytest.importorskip("numba")
    from qstr_benchmarks.pulse import load_model

    model = load_model()
    point = model.initial(7098473)
    np.testing.assert_allclose(
        model.primitives(point),
        model.independent_primitives(point),
        atol=3e-9,
        rtol=0.0,
    )


def test_ising_reduction_is_invariant_and_preserves_full_dynamics():
    pytest.importorskip("quocslib")
    from quocslib.optimalcontrolproblems.IsingModelProblem import (
        get_control_hamiltonian,
        get_static_hamiltonian,
    )

    from qstr_benchmarks.ising_support.model import IsingModel, translation_basis

    basis = translation_basis(5)
    assert basis.shape == (32, 8)
    np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=3e-16)
    control = get_control_hamiltonian(5)
    drift = get_static_hamiltonian(5, 1.0, 2.23)
    for matrix in (control, drift):
        np.testing.assert_allclose(matrix @ basis, basis @ (basis.T @ matrix @ basis), atol=2e-15)
    model = IsingModel()
    waveform = np.full(100, 0.71)
    full_state = expm(-1j * (drift + 0.71 * control))[:, 0]
    full_loss = 1 - abs(full_state[-1]) ** 2
    np.testing.assert_allclose(model.loss(waveform, 2.23), full_loss, atol=5e-14, rtol=0.0)
