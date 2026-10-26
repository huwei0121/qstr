"""Compare rerun trajectories with the bundled reference observations."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .analysis import read_records


def compare(input_path: Path, reference_path: Path) -> dict:
    records, failures = read_records(input_path)
    with gzip.open(reference_path, "rt", encoding="utf-8") as stream:
        reference = json.load(stream)["records"]
    lookup = {(row["problem"], row["method"], row["seed"]): row for row in reference}
    results = []
    for record in records:
        identity = (record["problem"], record["method"], record["seed"])
        if identity not in lookup:
            results.append(
                {
                    "problem": identity[0],
                    "method": identity[1],
                    "seed": identity[2],
                    "status": "no_reference",
                }
            )
            continue
        saved = lookup[identity]
        with np.load(reference_path.parent / saved["trajectory"], allow_pickle=False) as arrays:
            coordinates = np.asarray([row["x"] for row in record["history"]])
            observations = np.asarray([row["value"] for row in record["history"]])
            coordinates_exact = np.array_equal(coordinates, arrays["x"])
            observations_exact = np.array_equal(observations, arrays["value"])
            difference = (
                None
                if coordinates.shape != arrays["x"].shape
                else float(np.max(np.abs(coordinates - arrays["x"])))
            )
        final_point = record.get("final_x", record["result"].get("x"))
        actual = {row["evaluations"]: row for row in record["recommendations"]}
        # The reference export also records the actual stopping query when it
        # falls between regular checkpoints. Use the observed native endpoint;
        # do not fill missing intermediate evaluations.
        actual.setdefault(record["nfev"], {"x": final_point, "quality": record["final_quality"]})
        checkpoint_match = all(
            point["evaluations"] in actual
            and np.array_equal(actual[point["evaluations"]]["x"], point["x"])
            and actual[point["evaluations"]]["quality"] == point["quality"]
            for point in saved["recommendations"]
        )
        checks = {
            "budget_equal": record["configuration"]["max_evaluations"]
            == saved["configuration"]["max_evaluations"],
            "coordinates_exact": coordinates_exact,
            "observations_exact": observations_exact,
            "returned_point_exact": np.array_equal(final_point, saved["returned_x"]),
            "initial_quality_exact": record["initial_quality"] == saved["initial_quality"],
            "final_quality_exact": record["final_quality"] == saved["final_quality"],
            "checkpoints_exact": checkpoint_match,
        }
        results.append(
            {
                "problem": identity[0],
                "method": identity[1],
                "seed": identity[2],
                "status": "exact_match" if all(checks.values()) else "different",
                "nfev": record["nfev"],
                "maximum_coordinate_difference": difference,
                **checks,
            }
        )
    return {
        "runs": results,
        "matched": sum(row["status"] == "exact_match" for row in results),
        "compared": len(results),
        "failed_or_missing": len(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.input, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(
        not result["compared"]
        or result["failed_or_missing"]
        or result["matched"] != result["compared"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
