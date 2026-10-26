"""Physical protocol and bundled-source checks for the QAOA backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import qstr_benchmarks.qaoa_support


def test_bundled_sources_match_recorded_hashes():
    vendor = Path(qstr_benchmarks.qaoa_support.__file__).parent / "vendor"
    records = json.loads((vendor / "SOURCES.json").read_text(encoding="utf-8"))
    for record in records:
        assert (
            hashlib.sha256((vendor / record["path"]).read_bytes()).hexdigest() == record["sha256"]
        )
    for name in ("anatra", "minq", "circuit", "input_check"):
        assert "Permission is hereby granted" in (vendor / name / "LICENSE").read_text()


def test_count_oracle_reproducibility_and_hard_budget():
    pytest.importorskip("qiskit_aer")
    from qstr_benchmarks.qaoa_support.model import CountOracle

    first, second = CountOracle(4173, 1000, 2), CountOracle(4173, 1000, 2)
    for point in (np.zeros(10), np.linspace(-0.4, 0.3, 10)):
        assert first.pair(point) == second.pair(point)
        assert first.history[-1]["counts"] == second.history[-1]["counts"]
        assert sum(first.history[-1]["counts"].values()) == 1000
    assert first.history[0]["simulator_seed"] != first.history[1]["simulator_seed"]
    with pytest.raises(RuntimeError, match="query cap"):
        first.scalar(np.zeros(10))
    assert len(first.history) == 2


def test_offline_evaluator_matches_circuit_statevector():
    pytest.importorskip("qiskit_aer")
    from qiskit.quantum_info import Statevector

    from qstr_benchmarks.qaoa_support.model import CountOracle, exact_evaluator

    oracle = CountOracle(1, 1000, 1)
    evaluate = exact_evaluator(oracle.graph)
    rng = np.random.default_rng(41720)
    for point in [np.zeros(10), *rng.normal(size=(5, 10))]:
        bound = oracle.circuit.assign_parameters(dict(zip(oracle.parameters, point)))
        state = Statevector.from_instruction(bound.remove_final_measurements(inplace=False))
        probabilities = state.probabilities_dict()
        expected = sum(
            probability * oracle.statistics.maxcut_obj(bitstring[::-1], oracle.graph)
            for bitstring, probability in probabilities.items()
        )
        assert abs(evaluate(point) - expected) < 1e-11
    assert not oracle.history


def test_nested_measurement_configuration_controls_paid_counts(tmp_path):
    pytest.importorskip("qiskit_aer")
    from qstr_benchmarks.qaoa import run_case

    record = run_case(
        {
            "problem": "qaoa_maxcut",
            "method": "qstr",
            "seed": 9,
            "max_evaluations": 2,
            "observation": {"shots_per_evaluation": 37, "cost_per_evaluation": 37},
        },
        tmp_path,
    )
    assert record["nfev"] == 2
    assert all(sum(point["counts"].values()) == 37 for point in record["history"])
