"""Inspect solver presets and portable benchmark specifications."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import PRESET_NAMES, __version__
from .presets import get_preset


def build_benchmark_plan(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a problem catalog and summarize its evaluation counts.

    Planning reads configuration only. It neither constructs an objective nor
    invokes an optimizer. Measurement units are taken from each problem's
    declared evaluation protocol.
    """
    if specification.get("schema_version") != 1:
        raise ValueError("benchmark specification requires schema_version=1")
    seeds = specification.get("seeds", [])
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    problems = specification.get("problems", [])
    if not problems:
        raise ValueError("the benchmark must contain at least one problem")
    entries = []
    names = set()
    for problem in problems:
        name = problem["name"]
        if name not in PRESET_NAMES or name in names:
            raise ValueError(f"unknown or duplicate problem: {name}")
        names.add(name)
        budget = problem["max_evaluations"]
        if type(budget) is not int or budget < 1:
            raise ValueError("max_evaluations must be a positive integer")
        methods = [method["name"] for method in problem["methods"]]
        if not methods or len(methods) != len(set(methods)):
            raise ValueError("each problem requires distinct method names")
        if any(not isinstance(method, str) or not method for method in methods):
            raise ValueError("method names must be nonempty strings")
        cost = problem["evaluation_cost"]
        if type(cost["amount"]) is not int or cost["amount"] < 1:
            raise ValueError("evaluation_cost.amount must be a positive integer")
        entries.append(
            {
                "problem": name,
                "methods": methods,
                "seeds": len(seeds),
                "runs": len(seeds) * len(methods),
                "max_evaluations_per_run": budget,
                "measurement_limit_per_run": budget * cost["amount"],
                "measurement_unit": cost["unit"],
            }
        )
    return {"problems": entries, "planned_runs": sum(row["runs"] for row in entries)}


def main(argv: list[str] | None = None) -> int:
    """Run inspection commands without starting numerical experiments."""
    parser = argparse.ArgumentParser(prog="qstr", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    preset_parser = commands.add_parser("presets", help="show numerical problem presets")
    preset_parser.add_argument("name", nargs="?", choices=PRESET_NAMES)
    plan_parser = commands.add_parser("benchmark-plan", help="summarize a benchmark catalog")
    plan_parser.add_argument("configuration", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "presets":
        names = (arguments.name,) if arguments.name else PRESET_NAMES
        result = {name: get_preset(name) for name in names}
    else:
        try:
            specification = json.loads(arguments.configuration.read_text(encoding="utf-8-sig"))
            result = build_benchmark_plan(specification)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
