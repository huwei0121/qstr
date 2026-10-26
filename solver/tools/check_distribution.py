"""Check that built distributions contain only the maintained package."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check_distribution(path: Path) -> dict[str, object]:
    """Inspect archive members without extracting or executing their contents."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            code = {
                name: archive.read(name).decode("utf-8")
                for name in names
                if name.startswith("qstr/") and name.endswith(".py")
            }
        if "qstr/__init__.py" not in names or "qstr/py.typed" not in names:
            raise ValueError("wheel is missing package entry or typing marker")
        unexpected = [
            name for name in names if not name.startswith("qstr/") and ".dist-info/" not in name
        ]
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            names = [
                str(PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0]))
                for member in members
            ]
            code = {}
            for member, name in zip(members, names):
                if name.startswith("src/qstr/") and name.endswith(".py"):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"unreadable source member: {name}")
                    code[name] = stream.read().decode("utf-8")
        allowed_roots = {"src", "docs", "examples", "benchmarks", "tests", "tools"}
        allowed_files = {
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "MANIFEST.in",
            "setup.cfg",
            "PKG-INFO",
            "requirements.txt",
            "requirements-reproducibility.txt",
        }
        unexpected = [
            name
            for name in names
            if PurePosixPath(name).parts[0] not in allowed_roots and name not in allowed_files
        ]
        unexpected += [
            name
            for name in names
            if name.startswith("src/") and not name.startswith(("src/qstr/", "src/qstr.egg-info/"))
        ]
        unexpected += [
            name
            for name in names
            if name.startswith("tests/")
            and PurePosixPath(name).name
            not in {
                "test_qstr_solver.py",
                "test_qstr_presets.py",
                "test_qstr_baselines.py",
                "test_qstr_package.py",
            }
        ]
        if "README.md" not in names or "LICENSE" not in names:
            raise ValueError("source distribution lacks documentation or license")
    if unexpected:
        raise ValueError(f"unexpected archive members: {unexpected}")
    for name in names:
        if ".." in PurePosixPath(name).parts or PurePosixPath(name).is_absolute():
            raise ValueError(f"unsafe archive member: {name}")
    forbidden = ("au" + "thor", "tuned_unified", "orbit_dfo")
    for name, content in code.items():
        if any(term in (name + content).lower() for term in forbidden):
            raise ValueError(f"nonpublic identifier or dependency in {name}")
    return {
        "artifact": path.name,
        "files": len(names),
        "python_modules": len(code),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    arguments = parser.parse_args()
    artifacts = sorted(arguments.directory.glob("qstr-*.whl"))
    artifacts += sorted(arguments.directory.glob("qstr-*.tar.gz"))
    if not artifacts:
        parser.error("no Q-STR wheel or source distribution found")
    print(json.dumps([check_distribution(path) for path in artifacts], indent=2))


if __name__ == "__main__":
    main()
