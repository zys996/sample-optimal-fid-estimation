"""Export the submission files without Git history, caches, or experiment outputs.

Run from any directory: python tools/make_submission_zip.py --output dist/code.zip
Fixed ZIP metadata makes repeated exports of unchanged files identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]

DIRECTORIES = ("src", "experiments", "configs", "reproduce", "tests", "docs", "tools")
TOP_LEVEL = ("README.md", "pyproject.toml", "requirements-report.txt", ".gitignore", ".gitattributes",
             "THIRD_PARTY.md")
EXCLUDED = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "outputs", ".DS_Store"}


def submission_files():
    paths = [ROOT / name for name in TOP_LEVEL if (ROOT / name).is_file()]
    for directory in DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if (path.is_file() and not EXCLUDED.intersection(relative.parts)
                    and path.suffix not in {".pyc", ".pyo", ".log"}
                    and not any(part.endswith(".egg-info") for part in relative.parts)):
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def write_member(archive, name, content):
    info = zipfile.ZipInfo("FD-estimation/" + name, date_time=(2026, 9, 22, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "FD-estimation.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = submission_files()
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            write_member(archive, relative, content)
    print(f"Exported {len(paths)} files to {args.output}")


if __name__ == "__main__":
    main()
