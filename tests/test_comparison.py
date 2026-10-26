"""Distinguish a recorded early stop from missing intermediate evidence."""

import gzip
import json

import numpy as np

from qstr_benchmarks.comparison import compare


def test_early_native_endpoint_does_not_hide_missing_checkpoints(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    reference = tmp_path / "reference.json.gz"
    np.savez(tmp_path / "trajectory.npz", x=[[0.0], [0.5]], value=[2.0, 1.0])
    record = {
        "problem": "gate_calibration",
        "method": "qstr",
        "seed": 7,
        "status": "completed",
        "nfev": 2,
        "configuration": {"max_evaluations": 3},
        "history": [{"x": [0.0], "value": 2.0}, {"x": [0.5], "value": 1.0}],
        "recommendations": [{"evaluations": 1, "x": [0.0], "quality": 2.0}],
        "initial_quality": 2.0,
        "final_quality": 1.0,
        "result": {"x": [0.5]},
    }
    saved = {
        **record,
        "trajectory": "trajectory.npz",
        "returned_x": [0.5],
        "recommendations": record["recommendations"]
        + [{"evaluations": 2, "x": [0.5], "quality": 1.0}],
    }
    with gzip.open(reference, "wt", encoding="utf-8") as stream:
        json.dump({"records": [saved]}, stream)
    path = output / "record.json"
    path.write_text(json.dumps(record))
    assert compare(output, reference)["matched"] == 1
    record["recommendations"] = []
    path.write_text(json.dumps(record))
    assert not compare(output, reference)["runs"][0]["checkpoints_exact"]
    record["history"][1]["value"] = 1.1
    path.write_text(json.dumps(record))
    assert not compare(output, reference)["runs"][0]["observations_exact"]
