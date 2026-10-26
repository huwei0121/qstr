"""Check physical conversions and retain failed or missing experimental runs."""

import json
from pathlib import Path

import numpy as np
import pytest

from qstr_benchmarks.analysis import physical_quality, read_records, summarize


def test_reference_curves_recomputed_from_all_seeds():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/reference_runs.json.gz").exists():
        pytest.skip("Reference data omitted from this installation check")
    records, failures = read_records(root / "data/reference_runs.json.gz")
    summary = summarize(records, failures)
    expected = json.loads((root / "data/reference_curves.json").read_text())
    assert summary["completed"] == 510 and summary["failed_or_missing"] == 0
    assert sum(len(problem["methods"]) for problem in summary["problems"].values()) == 17
    for problem, methods in expected.items():
        for method, reference in methods.items():
            current = summary["problems"][problem]["methods"][method]
            for field in ("calls", "mean", "q25", "q75", "initial_mean", "actual_calls"):
                np.testing.assert_array_equal(current[field], reference[field])
            assert len(current["seeds"]) == 30


def test_physical_units_are_not_relative_errors():
    np.testing.assert_allclose(
        physical_quality("pulse_calibration", [0.1, 0.01]), np.log([0.1, 0.01])
    )
    np.testing.assert_allclose(physical_quality("qaoa_maxcut", [3.0, 0.1]), [-3.0, -5.9])
    np.testing.assert_allclose(physical_quality("ising_state_transfer", [1.0, 0.1]), [0.0, 90.0])
    with pytest.raises(ValueError):
        physical_quality("pulse_calibration", [0.0])


def test_missing_runs_are_kept_in_analysis(tmp_path):
    cases = [{"problem": "qaoa_maxcut", "method": "qstr", "seed": seed} for seed in [1, 2]]
    (tmp_path / "plan.json").write_text(json.dumps({"cases": cases}))
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "failure.json").write_text(
        json.dumps({"status": "failed", "configuration": cases[0]})
    )
    records, failures = read_records(tmp_path)
    assert not records
    assert [row["status"] for row in failures] == ["failed", "missing"]


def test_duplicate_seed_records_are_rejected():
    record = {"problem": "qaoa_maxcut", "method": "qstr", "seed": 1, "status": "completed"}
    with pytest.raises(ValueError, match="Duplicate run"):
        summarize([record, record])
