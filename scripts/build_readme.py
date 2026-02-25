#!/usr/bin/env python3
"""Generate PKG_README.md with an auto-generated changelog prepended to README.md.

The changelog is built from git history since the last "Bump <pkg>" commit.
PyPI renders whatever file `readme` points to in pyproject.toml, so the
publish workflows point readme at PKG_README.md (this script's output).

Usage:
    python scripts/build_readme.py [--dir <pkg-dir>]

When --dir is omitted, uses the current working directory.
"""

import argparse
import subprocess
import sys
from datetime import date, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout.strip()


def _find_last_bump_sha(pkg_name: str, repo_root: Path) -> str | None:
    """Return the SHA of the most recent 'Bump <pkg-name>' commit, or None."""
    log = _git(
        "log", "--oneline", "--format=%H %s", f"--grep=Bump {pkg_name}",
        cwd=repo_root,
    )
    for line in log.splitlines():
        if not line:
            continue
        sha, subject = line.split(" ", 1)
        if subject.startswith(f"Bump {pkg_name}"):
            return sha
    return None


def _collect_commits(bump_sha: str | None, pkg_dir: Path, repo_root: Path) -> list[str]:
    """Return one-line commit subjects since bump_sha that touch pkg_dir."""
    # Compute the relative path of pkg_dir from repo_root for git log
    rel_dir = pkg_dir.resolve().relative_to(repo_root.resolve())

    if bump_sha:
        range_spec = f"{bump_sha}..HEAD"
    else:
        range_spec = "HEAD"

    log = _git(
        "log", "--no-merges", "--oneline", "--format=%s", range_spec,
        "--", str(rel_dir),
        cwd=repo_root,
    )
    subjects = []
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        # Filter out bump commits themselves
        if line.startswith("Bump "):
            continue
        subjects.append(line)
    return subjects


def _build_changelog(version: str, commits: list[str]) -> str:
    """Format commits as a markdown changelog section."""
    today = date.today().isoformat()
    lines = [f"# Changelog\n", f"\n## {version} ({today})\n"]
    if commits:
        for subject in commits:
            lines.append(f"\n- {subject}")
    else:
        lines.append(f"\nNo changes since last release.")
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PKG_README.md with changelog")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Package directory (default: cwd)")
    args = parser.parse_args()

    pkg_dir = (args.dir or Path.cwd()).resolve()
    pyproject_path = pkg_dir / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    pkg_name = project["name"]
    version = project["version"]

    # Find repo root
    repo_root = Path(_git("rev-parse", "--show-toplevel", cwd=pkg_dir))

    bump_sha = _find_last_bump_sha(pkg_name, repo_root)
    commits = _collect_commits(bump_sha, pkg_dir, repo_root)
    changelog = _build_changelog(version, commits)

    # Read original README
    readme_path = pkg_dir / "README.md"
    if readme_path.exists():
        original_readme = readme_path.read_text()
    else:
        original_readme = ""

    # Write combined file
    out_path = pkg_dir / "PKG_README.md"
    out_path.write_text(f"{changelog}\n---\n\n{original_readme}")
    print(f"Generated {out_path} ({len(commits)} changelog entries)")


if __name__ == "__main__":
    main()
