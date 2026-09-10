#!/usr/bin/env python3
"""Generate PKG_README.md with an auto-generated changelog prepended to README.md.

The changelog covers commits touching the package since its PREVIOUS
version was set.  PyPI renders whatever file `readme` points to in
pyproject.toml, so the publish workflows point readme at PKG_README.md
(this script's output).

The anchor is read from the `version` line of the package's own
pyproject.toml, walked back through git history.  It used to be the
second-most-recent commit whose subject began "Bump <pkg>" -- a convention
this repository has never once followed (zero matches across 1400+
commits), so the anchor was always None and every published changelog was
the package's ENTIRE history.  jaato-server's PyPI long description was
91k characters and grew by ~400 across a release carrying five merged PRs.

A commit-message convention is a promise someone has to keep on the day
they release.  A version change is not a convention: it is the definition
of a release, it is already enforced (the publish workflow refuses a
version that exists on PyPI), and it survives squash merges, which is what
made the "Bump <pkg>" scheme unkeepable here -- one squashed PR has one
subject line, and a release that bumps two packages needs two.

Usage:
    python scripts/build_readme.py [--dir <pkg-dir>]

When --dir is omitted, uses the current working directory.
"""

import argparse
import subprocess
import sys
from datetime import date
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


def _version_at(sha: str, rel_pyproject: str, repo_root: Path) -> str | None:
    """The version this package's pyproject declared at ``sha``.

    None when the file did not exist there, could not be parsed, or the
    commit is unreachable -- which is what a shallow clone looks like from
    the inside.  ``_git`` returns "" rather than raising on a failed
    command, so all three arrive here the same way and are treated the
    same: as "history ends here", never as a version.
    """
    blob = _git("show", f"{sha}:{rel_pyproject}", cwd=repo_root)
    if not blob:
        return None
    try:
        return tomllib.loads(blob)["project"]["version"]
    except Exception:       # noqa: BLE001 - malformed/renamed file, same answer
        return None


def _find_release_anchors(pkg_dir: Path, repo_root: Path) -> tuple[str | None, str | None]:
    """Return ``(current_sha, previous_sha)`` for this package.

    Both name the commit that *set* a version -- the OLDEST commit of a run
    declaring it, since later commits touching pyproject.toml (a new
    dependency, an extra) carry the same version forward without being the
    release.

    ``current_sha`` is the commit that set the version being built; the
    caller excludes it so a release's own bump is not an entry in its own
    changelog.  ``previous_sha`` is the changelog anchor: the commit that
    set the version before it.

    Either may be None, and both mean "include everything":

      * a first release, where only one version has ever existed;
      * a shallow clone, where the walk hits an unreadable commit before it
        has seen two versions.  CI passes ``fetch-depth: 0`` on the publish
        jobs precisely so this does not happen there.
    """
    rel = (pkg_dir.resolve().relative_to(repo_root.resolve())
           / "pyproject.toml").as_posix()
    shas = _git("log", "--format=%H", "--", rel, cwd=repo_root).splitlines()

    current_sha: str | None = None
    previous_sha: str | None = None
    seen: str | None = None
    groups = 0

    for sha in shas:                        # newest first
        version = _version_at(sha, rel, repo_root)
        if version is None:                 # end of readable history
            if previous_sha is None:
                print(f"warning: cannot read {rel} at {sha[:12]} - history may be "
                      f"shallow; changelog will cover everything reachable",
                      file=sys.stderr)
            break
        if seen is None:
            seen = version
        elif version != seen:
            groups += 1
            seen = version
            if groups == 2:                 # two transitions is all we need
                break
        if groups == 0:
            current_sha = sha
        elif groups == 1:
            previous_sha = sha

    return current_sha, previous_sha


def _collect_commits(previous_sha: str | None, current_sha: str | None,
                     pkg_dir: Path, repo_root: Path) -> list[str]:
    """Commit subjects touching pkg_dir since ``previous_sha`` (exclusive).

    Two kinds of entry are dropped, and the rule for both is "carries no
    news", never "matches a convention":

    * a subject beginning "Bump " -- the pre-existing filter, kept because
      it catches an intermediate release commit anywhere in the range;
    * ``current_sha`` -- this release's own version bump -- but ONLY when
      the sole file it touched inside this package was pyproject.toml.

    That qualifier is load-bearing in both directions.  Dropping
    ``current_sha`` unconditionally loses a real entry on a FIRST release,
    where the commit that "set" the current version is the initial import
    and carries the whole package; and this repository has more than once
    folded a version bump into a feature commit (efdec11c set jaato-server
    0.7.0 *and* shipped the convenience facade), which must still be
    reported.  ``--name-only`` is scoped by the pathspec, so a bump that
    also edited CLAUDE.md still reads as pyproject-only from this
    package's point of view -- which is the right answer for a changelog
    about this package.
    """
    rel_dir = pkg_dir.resolve().relative_to(repo_root.resolve())
    range_spec = f"{previous_sha}..HEAD" if previous_sha else "HEAD"

    # NUL-delimited records so a subject containing a newline cannot be
    # mistaken for a filename.
    raw = _git(
        "log", "--no-merges", "--format=%x00%H %s", "--name-only", range_spec,
        "--", str(rel_dir),
        cwd=repo_root,
    )

    subjects = []
    for record in raw.split("\0"):
        lines = [ln for ln in record.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        sha, _, subject = lines[0].partition(" ")
        files = lines[1:]
        if subject.startswith("Bump "):
            continue
        if (current_sha and sha == current_sha
                and files and all(f.endswith("pyproject.toml") for f in files)):
            continue
        subjects.append(subject)
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
    version = project["version"]

    # Find repo root
    repo_root = Path(_git("rev-parse", "--show-toplevel", cwd=pkg_dir))

    current_sha, previous_sha = _find_release_anchors(pkg_dir, repo_root)
    commits = _collect_commits(previous_sha, current_sha, pkg_dir, repo_root)
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
    anchor = previous_sha[:12] if previous_sha else "(none - full history)"
    print(f"Generated {out_path} ({len(commits)} changelog entries "
          f"since {anchor})")


if __name__ == "__main__":
    main()
