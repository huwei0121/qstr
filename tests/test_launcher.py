"""Preserve failed attempts and avoid duplicate statistical failures on resume."""

import importlib.util
import json
from pathlib import Path

from qstr_benchmarks.analysis import read_records

SPEC = importlib.util.spec_from_file_location(
    "reproduction_launcher", Path(__file__).resolve().parents[1] / "reproduce.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_partial_backend_outputs_are_preserved_before_retry(tmp_path):
    case = {"problem": "qaoa_maxcut", "method": "qstr", "seed": 3}
    folder = tmp_path / "case"
    folder.mkdir()
    (folder / "configuration.json").write_text(json.dumps(case))
    (folder / "history.json").write_text("[1, 2, 3]")
    (folder / "failure.json").write_text(json.dumps({"status": "failed", "configuration": case}))
    archived = launcher.archive_attempt(folder)
    assert (archived / "history.json").read_text() == "[1, 2, 3]"
    assert not (folder / "history.json").exists()
    (folder / "record.json").write_text(json.dumps({**case, "status": "completed"}))
    records, failures = read_records(tmp_path)
    assert len(records) == 1 and not failures


def test_full_plan_has_all_configured_comparisons():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/full.json").read_text())
    cases = launcher.make_cases(config)
    assert len(cases) == 510
    assert len({(case["problem"], case["method"]) for case in cases}) == 17
    chosen = launcher.make_cases(config, ["qaoa_maxcut"], ["qstr"], [7])
    assert len(chosen) == 1 and chosen[0]["max_evaluations"] == 550
