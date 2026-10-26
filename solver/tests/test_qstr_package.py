"""Public imports, CLI plans, serialization, and the executable quantum example."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import qstr
from qstr.cli import build_benchmark_plan, main


def test_public_entry_and_json_result():
    result = qstr.minimize(lambda x: float(x @ x), [0.4, -0.2], 20, seed=1)
    assert isinstance(result, qstr.OptimizationResult)
    record = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert record["nfev"] == len(record["history"]["fun"])
    assert record["x"] == result.x.tolist()


def test_preset_cli_outputs_parseable_configuration(capsys):
    assert main(["presets", "qaoa_maxcut"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["qaoa_maxcut"]["dimension"] == 10
    assert record["qaoa_maxcut"]["options"]["initial_radius"] == 0.1


def test_benchmark_catalog_matches_public_presets():
    catalog = Path(__file__).resolve().parents[1] / "benchmarks/four_problems.json"
    specification = json.loads(catalog.read_text(encoding="utf-8"))
    plan = build_benchmark_plan(specification)
    assert plan["planned_runs"] == 510
    for problem in specification["problems"]:
        preset = qstr.get_preset(problem["name"])
        assert problem["max_evaluations"] == preset["max_evaluations"]
        assert problem["evaluation_cost"] == preset["evaluation_cost"]
    specification["seeds"].append(specification["seeds"][0])
    with pytest.raises(ValueError, match="unique"):
        build_benchmark_plan(specification)


def test_qaoa_example_normalization_axes_and_sampling():
    path = Path(__file__).resolve().parents[1] / "examples/qaoa_maxcut.py"
    specification = importlib.util.spec_from_file_location("qaoa_example", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    objective = module.MaxCutObjective(seed=41, shots=1000)
    zero = np.zeros(10)
    assert objective.expected_value(zero) == pytest.approx(-3.0)
    for coordinate in (0, 4, 5, 9):
        point = zero.copy()
        point[coordinate] = 0.37
        assert objective.expected_value(point) == pytest.approx(-3.0)
    angles = np.linspace(-0.3, 0.4, 10)
    probabilities = objective.probabilities(angles)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities >= 0)
    sample = objective(angles)
    assert -6 <= sample <= 0
    assert objective.calls == 1
    assert sample == pytest.approx(round(sample * 1000) / 1000)
