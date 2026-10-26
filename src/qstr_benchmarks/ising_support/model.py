"""Exact cyclic-invariant reduction of the periodic five-qubit Ising model."""

import numpy as np
from quocslib.optimalcontrolproblems.IsingModelProblem import (
    get_control_hamiltonian,
    get_static_hamiltonian,
)


def translation_basis(n):
    """Orthonormal basis for states invariant under cyclic site translation."""
    if n < 3:
        raise ValueError("This model family requires at least three qubits")
    unseen = set(range(2**n))
    orbits = []
    mask = 2**n - 1
    while unseen:
        first = min(unseen)
        orbit = set()
        x = first
        while x not in orbit:
            orbit.add(x)
            x = ((x << 1) & mask) | (x >> (n - 1))
        unseen.difference_update(orbit)
        orbits.append(sorted(orbit))
    basis = np.zeros((2**n, len(orbits)))
    for j, orbit in enumerate(orbits):
        basis[orbit, j] = 1 / np.sqrt(len(orbit))
    return basis


class IsingModel:
    """Periodic Ising chain, 100 equal time slices, duration one."""

    def __init__(self):
        basis = translation_basis(5)
        self.nearest_coupling = basis.T @ get_static_hamiltonian(5, 1.0, 0.0) @ basis
        self.next_nearest_coupling = basis.T @ get_static_hamiltonian(5, 0.0, 1.0) @ basis
        self.control = basis.T @ get_control_hamiltonian(5) @ basis
        self.initial = basis[0].astype(complex)
        self.target = basis[-1].astype(complex)

    def loss(self, waveform, coupling=2.0):
        eigenvalues, eigenvectors = np.linalg.eigh(
            self.nearest_coupling[None, :, :]
            + coupling * self.next_nearest_coupling[None, :, :]
            + np.asarray(waveform)[:, None, None] * self.control[None, :, :]
        )
        propagators = (
            eigenvectors * np.exp(-0.01j * eigenvalues)[:, None, :]
        ) @ eigenvectors.transpose(0, 2, 1)
        state = self.initial.copy()
        for propagator in propagators:
            state = propagator @ state
        return float(np.clip(1.0 - abs(np.vdot(self.target, state)) ** 2, 0.0, 1.0))
