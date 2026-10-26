"""Install environments, rerun experiments, and regenerate figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIMES = ("analysis", "pulse", "qaoa", "gate")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def archive_attempt(directory: Path) -> Path:
    """Retain an incomplete attempt before creating fresh backend outputs."""
    directory = directory.resolve()
    existing = [path for path in directory.iterdir() if path.name != "attempts"]
    if not (directory / "configuration.json").is_file():
        raise ValueError("Refusing to move files without a saved case configuration")
    history = directory / "attempts"
    history.mkdir(exist_ok=True)
    indices = [int(path.name) for path in history.iterdir() if path.name.isdigit()]
    target = history / f"{max(indices, default=0) + 1:03d}"
    if not target.resolve().is_relative_to(directory):
        raise ValueError("Attempt destination must stay within its case directory")
    for path in existing:
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Attempt files must remain within their case directory")
    target.mkdir()
    for path in existing:
        path.rename(target / path.name)
    return target


def interpreter(runtime: str, override: str | None = None) -> Path:
    if override:
        path = Path(override).resolve()
    else:
        directory = ROOT / ".venvs" / runtime
        path = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {runtime} Python. Run setup --runtimes {runtime}.")
    return path


def solver_source() -> Path:
    bundled = ROOT / "solver"
    if (bundled / "src/qstr").is_dir():
        return bundled
    if (ROOT.parent / "src/qstr").is_dir():
        return ROOT.parent
    raise FileNotFoundError("The reproduction bundle must contain solver/src/qstr")


def process_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(solver_source() / "src")))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    return environment


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for label, directory in (
        ("benchmarks", ROOT / "src"),
        ("solver", solver_source() / "src/qstr"),
        ("third_party", ROOT / "third_party"),
    ):
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in {".py", ".json"}
            ):
                continue
            digest.update(f"{label}/{path.relative_to(directory).as_posix()}\n".encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def setup(args) -> int:
    base = str(Path(args.python).resolve()) if args.python else sys.executable
    probe = subprocess.check_output(
        [base, "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], text=True
    ).strip()
    if probe != "3.10":
        raise ValueError("The recorded quantum stack requires Python 3.10; use --python PATH.")
    for runtime in args.runtimes:
        environment = ROOT / ".venvs" / runtime
        if not environment.exists():
            subprocess.run([base, "-m", "venv", str(environment)], check=True)
        python = interpreter(runtime)
        constraints = []
        lock = ROOT / "requirements/locks" / f"windows-x86_64-{runtime}.txt"
        if os.name == "nt" and lock.is_file():
            constraints = ["-c", str(lock)]
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                "requirements/common.txt",
                "-r",
                f"requirements/{runtime}.txt",
                *constraints,
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(solver_source()), str(ROOT)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
        (environment / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
        print(f"Ready: {runtime}", flush=True)
    return 0


def make_cases(configuration: dict, problems=None, methods=None, seeds=None) -> list[dict]:
    chosen_problems = list(configuration["problems"]) if not problems else problems
    if any(problem not in configuration["problems"] for problem in chosen_problems):
        raise ValueError("Unknown problem")
    labels = configuration["seeds"] if seeds is None else seeds
    if (
        not labels
        or len(labels) != len(set(labels))
        or any(type(seed) is not int or seed < 0 for seed in labels)
    ):
        raise ValueError("Seed labels must be distinct nonnegative integers")
    cases = []
    selected_methods = set()
    for problem in chosen_problems:
        specification = configuration["problems"][problem]
        for method in specification["algorithms"]:
            if methods and method["algorithm"] not in methods:
                continue
            selected_methods.add(method["algorithm"])
            for seed in labels:
                case = {key: value for key, value in specification.items() if key != "algorithms"}
                case.update(
                    problem=problem,
                    method=method["algorithm"],
                    parameters=method["parameters"],
                    seed=seed,
                )
                cases.append(case)
    if not cases or (methods and set(methods) != selected_methods):
        raise ValueError("Selected methods must exist in the selected problems")
    return cases


def run(args) -> int:
    config_path = args.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    cases = make_cases(configuration, args.problems, args.methods, args.seeds)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    plan = {
        "experiment": configuration["name"],
        "runs": len(cases),
        "source_sha256": source_fingerprint(),
        "cases": cases,
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required for a run")
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; choose a new directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Cannot resume with a changed experiment plan")
    write_json(plan_path, plan)
    environment = process_environment()

    def launch(case):
        directory = output / case["problem"] / case["method"] / f"seed_{case['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "configuration.json"
        if path.exists() and json.loads(path.read_text()) != case:
            raise ValueError(f"Stored configuration differs: {directory}")
        if (directory / "record.json").exists():
            saved = json.loads((directory / "record.json").read_text())
            if saved.get("status") != "completed" or saved.get("configuration") != case:
                raise ValueError(f"Invalid completion record: {directory}")
            return {
                "case": str(directory.relative_to(output)),
                "status": "already_completed",
                "returncode": 0,
            }
        if path.exists():
            archive_attempt(directory)
        write_json(path, case)
        python = interpreter(case["runtime"], getattr(args, f"python_{case['runtime']}"))
        command = [
            str(python),
            "-m",
            "qstr_benchmarks.worker",
            "--configuration",
            str(path),
            "--output",
            str(directory),
        ]
        with (directory / "execution.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        return {
            "case": str(directory.relative_to(output)),
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "case": f"{case['problem']}/{case['method']}/{case['seed']}",
                    "status": "failed",
                    "returncode": 1,
                    "message": str(error),
                }
            results.append(result)
            print(f"[{len(results)}/{len(cases)}] {result['case']}: {result['status']}", flush=True)
            write_json(
                output / "execution.json",
                {
                    "planned": len(cases),
                    "completed": sum(item["returncode"] == 0 for item in results),
                    "runs": results,
                },
            )
    return int(any(result["returncode"] for result in results))


def verify_bundle() -> int:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or modified bundled file: {name}")
    print(f"Verified {len(manifest['files'])} bundled files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    installation = commands.add_parser("setup", help="Create pinned Python 3.10 environments")
    installation.add_argument("--runtimes", nargs="+", choices=RUNTIMES, default=list(RUNTIMES))
    installation.add_argument("--python", help="Python 3.10 executable")
    execution = commands.add_parser("run", help="Execute a configured comparison")
    execution.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    execution.add_argument("--output", type=Path)
    execution.add_argument("--problems", nargs="+")
    execution.add_argument("--methods", nargs="+")
    execution.add_argument("--seeds", nargs="+", type=int)
    execution.add_argument("--workers", type=int, default=1)
    execution.add_argument("--plan", action="store_true")
    execution.add_argument("--resume", action="store_true")
    for runtime in RUNTIMES:
        execution.add_argument(f"--python-{runtime}")
    plotting = commands.add_parser("plot", help="Regenerate figures from bundled or new records")
    plotting.add_argument("--input", type=Path, default=ROOT / "data/reference_runs.json.gz")
    plotting.add_argument("--output", type=Path, required=True)
    plotting.add_argument("--python-analysis")
    comparison = commands.add_parser(
        "compare", help="Compare reruns with stored full-budget trajectories"
    )
    comparison.add_argument("--input", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--python-analysis")
    commands.add_parser("verify", help="Verify bundled files against the content manifest")
    args = parser.parse_args()
    if args.command == "setup":
        return setup(args)
    if args.command == "run":
        return run(args)
    if args.command == "verify":
        return verify_bundle()
    python = interpreter("analysis", args.python_analysis)
    if args.command == "compare":
        return subprocess.run(
            [
                str(python),
                "-m",
                "qstr_benchmarks.comparison",
                "--input",
                str(args.input.resolve()),
                "--reference",
                str(ROOT / "data/reference_runs.json.gz"),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            env=process_environment(),
        ).returncode
    return subprocess.run(
        [
            str(python),
            "-m",
            "qstr_benchmarks.analysis",
            "--input",
            str(args.input.resolve()),
            "--output",
            str(args.output.resolve()),
        ],
        cwd=ROOT,
        env=process_environment(),
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
