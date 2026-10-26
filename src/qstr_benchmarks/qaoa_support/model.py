"""Published depth-five MaxCut circuit and paid finite-count oracle."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer


def load_source(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class CountOracle:
    """One query returns the negative cut mean and estimated mean variance.

    The statevector evaluator is deliberately separate from this class. Fresh
    simulator seeds are drawn for every call, including repeated parameters.
    """

    def __init__(self, seed: int, shots: int, budget: int):
        self.graph = nx.complete_bipartite_graph(3, 2)
        self.depth = 5
        self.shots = int(shots)
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.statistics = load_source(
            "qstr_sampled_circuit", Path(__file__).parent / "vendor/circuit/Sampled.py"
        )
        self.parameters = ParameterVector("theta", 2 * self.depth)
        self.backend = Aer.get_backend("qasm_simulator")
        self.backend.set_options(max_parallel_threads=1, max_parallel_experiments=1)
        circuit = self.statistics.get_qaoa_circuit(
            self.graph, self.parameters[: self.depth], self.parameters[self.depth :]
        )
        self.circuit = transpile(circuit, self.backend, seed_transpiler=173, optimization_level=0)

    def pair(self, x):
        if len(self.history) >= self.budget:
            raise RuntimeError("No additional shots permitted beyond the query cap")
        simulator_seed = int(self.rng.integers(1, 2**31 - 1))
        x = np.asarray(x, dtype=float).copy()
        started = time.perf_counter()
        bound = self.circuit.assign_parameters(dict(zip(self.parameters, x)))
        counts = (
            self.backend.run(bound, shots=self.shots, seed_simulator=simulator_seed)
            .result()
            .get_counts()
        )
        counts = self.statistics.invert_counts(counts)
        if sum(counts.values()) != self.shots:
            raise RuntimeError("Simulator count total differs from requested shots")
        mean = self.statistics.compute_maxcut_energy(counts, self.graph)
        variance = self.statistics.compute_maxcut_variance(counts, self.graph, mean)
        self.history.append(
            dict(
                x=x.tolist(),
                value=mean,
                variance=variance,
                counts=counts,
                simulator_seed=simulator_seed,
                shots=self.shots,
                seconds=time.perf_counter() - started,
            )
        )
        return mean, variance

    def scalar(self, x):
        return self.pair(x)[0]


def exact_evaluator(graph, depth=5):
    """Create a post-search statevector diagnostic, never optimizer feedback."""
    n = len(graph)
    states = np.arange(2**n)
    cuts = sum(((states >> a) & 1) != ((states >> b) & 1) for a, b in graph.edges()).astype(float)
    zz = len(graph.edges()) - 2 * cuts
    indices = [
        (states[(states & (1 << q)) == 0], states[(states & (1 << q)) == 0] + (1 << q))
        for q in range(n)
    ]

    def evaluate(x):
        state = np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)
        for beta, gamma in zip(x[:depth], x[depth:]):
            state *= np.exp(-1j * gamma * zz)
            diagonal, off_diagonal = np.cos(beta), -1j * np.sin(beta)
            for left, right in indices:
                old_left, old_right = state[left].copy(), state[right].copy()
                state[left] = diagonal * old_left + off_diagonal * old_right
                state[right] = off_diagonal * old_left + diagonal * old_right
        return float(-np.dot(abs(state) ** 2, cuts))

    return evaluate
