"""Execute a single isolated problem/method/seed combination."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

MODULES = {
    "pulse_calibration": "pulse",
    "qaoa_maxcut": "qaoa",
    "gate_calibration": "gate",
    "ising_state_transfer": "ising",
}


def write_json(path: Path, value: object) -> None:
    """Write a record atomically; never leave a partial completion marker."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_record(record: dict, configuration: dict) -> None:
    """Reject inconsistent identities, missing histories, and budget overruns."""
    for field in ("problem", "method", "seed"):
        if record[field] != configuration[field]:
            raise ValueError(f"Incorrect record {field}")
    if record["status"] != "completed":
        raise ValueError("Backend did not complete")
    if not 0 < record["nfev"] <= configuration["max_evaluations"]:
        raise ValueError("Paid evaluation count exceeds the configured budget")
    if len(record["history"]) != record["nfev"]:
        raise ValueError("History length differs from the paid evaluation count")
    for observation in record["history"]:
        if "x" not in observation or "value" not in observation:
            raise ValueError("An observation lacks coordinates or its paid value")
    for recommendation in record["recommendations"]:
        if not 0 <= recommendation["evaluations"] <= configuration["max_evaluations"]:
            raise ValueError("Recommendation checkpoint exceeds the configured budget")
        for field in ("x", "quality"):
            if field not in recommendation:
                raise ValueError(f"Recommendation lacks {field}")
    for field in ("initial_quality", "final_quality", "metric", "cost", "result"):
        if field not in record:
            raise ValueError(f"Record lacks {field}")
    json.dumps(record, allow_nan=False)


def main() -> int:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration_bytes = args.configuration.read_bytes()
    configuration = json.loads(configuration_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    execution = {
        "configuration_sha256": hashlib.sha256(configuration_bytes).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "distributions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
    }
    try:
        module = importlib.import_module(f"qstr_benchmarks.{MODULES[configuration['problem']]}")
        execution["backend_sha256"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        record = module.run_case(configuration, args.output)
        validate_record(record, configuration)
        record["configuration"] = configuration
        record["execution"] = execution
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "record.json", record)
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "configuration": configuration,
            "execution": execution,
            "exception": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.output / "failure.json", failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
