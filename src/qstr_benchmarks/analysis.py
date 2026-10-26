"""Recompute physical-quality curves from individual run records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBLEMS = {
    "pulse_calibration": (
        "Single-qubit PWC pulse calibration",
        r"Log sequence infidelity, $\ln q$  ↓",
    ),
    "qaoa_maxcut": (r"QAOA MaxCut on $K_{3,2}$", r"Negative expected cut, $-\langle C\rangle$  ↓"),
    "gate_calibration": (
        "Single-qubit gate calibration",
        r"ORBIT objective, $f_{\mathrm{ORBIT}}$  ↓",
    ),
    "ising_state_transfer": (
        "5-qubit Ising state transfer",
        r"State-transfer fidelity, $F$ (%)  ↑",
    ),
}
LABELS = {
    "qstr": "Q-STR",
    "cma_es": "CMA-ES",
    "nelder_mead": "Nelder–Mead",
    "powell": "Powell",
    "differential_evolution": "Differential evolution",
    "one_plus_one": "(1+1)-ES",
    "simulated_annealing": "Simulated annealing",
    "anatra": "ANATRA",
    "pybobyqa": "Py-BOBYQA",
    "imfil": "ImFil",
    "nomad": "NOMAD",
    "spsa": "SPSA",
    "dcrab": "dCRAB (NM)",
}
COLORS = ["#bd284d", "#1f77b4", "#e7952a", "#329b75", "#8b6bb1", "#7b6251", "#6c7e8b"]
STYLES = ["-", "--", "-.", ":", "--", "-.", ":"]


def read_records(path: Path) -> tuple[list[dict], list[dict]]:
    if path.is_dir():
        records = [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.rglob("record.json"))
            if "attempts" not in item.relative_to(path).parts
        ]
        failures = []
        for item in sorted(path.rglob("failure.json")):
            if "attempts" in item.relative_to(path).parts:
                continue
            if not (item.parent / "record.json").exists():
                failures.append(json.loads(item.read_text(encoding="utf-8")))
        plan_path = path / "plan.json"
        if plan_path.exists():
            planned = json.loads(plan_path.read_text(encoding="utf-8"))["cases"]
            present = {(record["problem"], record["method"], record["seed"]) for record in records}
            failed = {
                (
                    failure["configuration"]["problem"],
                    failure["configuration"]["method"],
                    failure["configuration"]["seed"],
                )
                for failure in failures
            }
            for case in planned:
                identity = (case["problem"], case["method"], case["seed"])
                if identity not in present | failed:
                    failures.append({"status": "missing", "configuration": case})
        return records, failures
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload["records"], []


def physical_quality(problem: str, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite validation quality")
    if problem == "pulse_calibration":
        if np.any(values <= 0):
            raise ValueError("Logarithmic sequence infidelity requires positive values")
        return np.log(values)
    if problem == "qaoa_maxcut":
        return values - 6.0
    if problem == "ising_state_transfer":
        return 100.0 * (1.0 - values)
    return values


def summarize(records: list[dict], failures=()) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    identities = set()
    for record in records:
        identity = (record["problem"], record["method"], record["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate run: {identity}")
        if record["status"] != "completed":
            raise ValueError(f"Non-completed record in results: {identity}")
        identities.add(identity)
        groups[record["problem"]][record["method"]].append(record)
    result = {
        "schema_version": 1,
        "statistic": "Arithmetic means of independently assessed physical quantities; bands are cross-seed quartiles",
        "completed": len(records),
        "failed_or_missing": len(failures),
        "failures": list(failures),
        "problems": {},
    }
    for problem in PROBLEMS:
        if problem not in groups:
            continue
        task = {
            "methods": {},
            "budget": max(
                record["configuration"]["max_evaluations"]
                for rows in groups[problem].values()
                for record in rows
            ),
        }
        methods = sorted(
            groups[problem], key=lambda method: (method != "qstr", list(LABELS).index(method))
        )
        for method in methods:
            rows = sorted(groups[problem][method], key=lambda row: row["seed"])
            checkpoints = [
                {
                    point["evaluations"]: point["quality"]
                    for point in record["recommendations"]
                    if point["evaluations"] > 0
                }
                for record in rows
            ]
            common = sorted(set.intersection(*(set(row) for row in checkpoints)))
            if not common:
                raise ValueError(f"No common saved checkpoints for {problem}/{method}")
            # Preserve the recorded reduction order as well as the input values.
            matrix = np.array(
                [physical_quality(problem, [row[call] for row in checkpoints]) for call in common]
            ).T
            initial = physical_quality(problem, [row["initial_quality"] for row in rows])
            final = physical_quality(problem, [row["final_quality"] for row in rows])
            task["methods"][method] = {
                "calls": common,
                "mean": np.mean(matrix, axis=0).tolist(),
                "q25": np.quantile(matrix, 0.25, axis=0).tolist(),
                "q75": np.quantile(matrix, 0.75, axis=0).tolist(),
                "initial_mean": float(np.mean(initial)),
                "final_mean": float(np.mean(final)),
                "seeds": [row["seed"] for row in rows],
                "actual_calls": [row["nfev"] for row in rows],
            }
        result["problems"][problem] = task
    return result


def draw_panel(axis, problem: str, task: dict, label: str | None = None) -> None:
    for index, (method, row) in enumerate(task["methods"].items()):
        color, style = COLORS[index % len(COLORS)], STYLES[index % len(STYLES)]
        width = 2.7 if method == "qstr" else 1.65
        axis.plot(
            row["calls"],
            row["mean"],
            color=color,
            linestyle=style,
            linewidth=width,
            label=LABELS[method],
        )
        axis.fill_between(
            row["calls"],
            row["q25"],
            row["q75"],
            color=color,
            alpha=0.12 if method == "qstr" else 0.075,
            linewidth=0,
        )
        axis.plot(
            [0, row["calls"][0]],
            [row["initial_mean"], row["mean"][0]],
            color=color,
            linestyle="--",
            linewidth=1.4,
        )
        axis.scatter(
            [0], [row["initial_mean"]], s=19, facecolors="none", edgecolors=color, clip_on=False
        )
    title, ylabel = PROBLEMS[problem]
    axis.set_title(f"({label}) {title}" if label else title, loc="left", pad=10)
    axis.set_xlabel("Function evaluations")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, task["budget"])
    if problem == "ising_state_transfer":
        axis.set_ylim(0, 100)
    axis.grid(alpha=0.15, linewidth=0.7)
    location = (
        "upper right"
        if problem in ("qaoa_maxcut", "gate_calibration")
        else "lower left"
        if problem == "pulse_calibration"
        else "lower right"
    )
    axis.legend(
        loc=location, fontsize=8.5, ncol=2 if len(task["methods"]) > 3 else 1, framealpha=0.95
    )
    if problem == "qaoa_maxcut" and task["budget"] >= 550:
        inset = axis.inset_axes([0.60, 0.30, 0.38, 0.30])
        for index, row in enumerate(task["methods"].values()):
            inset.plot(
                row["calls"],
                row["mean"],
                color=COLORS[index],
                linestyle=STYLES[index],
                linewidth=1.7 if index == 0 else 1.1,
            )
            inset.fill_between(
                row["calls"], row["q25"], row["q75"], color=COLORS[index], alpha=0.08, linewidth=0
            )
        inset.set_xlim(250, 550)
        inset.set_ylim(-5.97, -5.65)
        inset.set_xticks([275, 550])
        inset.set_yticks([-5.95, -5.8])
        inset.tick_params(labelsize=7.5)
        inset.set_title("Late budget", fontsize=8.5, fontweight="normal", pad=3)
        inset.grid(alpha=0.15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, failures = read_records(args.input)
    if not records:
        raise ValueError("No completed records to plot")
    summary = summarize(records, failures)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    for index, (problem, axis) in enumerate(zip(PROBLEMS, axes.flat)):
        if problem in summary["problems"]:
            draw_panel(axis, problem, summary["problems"][problem], "abcd"[index])
        else:
            axis.set_visible(False)
    figure.tight_layout(pad=1.15, h_pad=2.7, w_pad=2.3)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            args.output / f"four_problems.{extension}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(figure)
    with (args.output / "endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "problem",
                "method",
                "seed",
                "evaluations",
                "initial_quality",
                "final_quality",
                "displayed_final_quality",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["problem"],
                    record["method"],
                    record["seed"],
                    record["nfev"],
                    record["initial_quality"],
                    record["final_quality"],
                    float(physical_quality(record["problem"], record["final_quality"])),
                ]
            )
    print(
        json.dumps(
            {
                "completed": len(records),
                "failed_or_missing": len(failures),
                "figure": str(args.output / "four_problems.png"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
