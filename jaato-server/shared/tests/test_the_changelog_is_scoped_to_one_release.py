"""``scripts/build_readme.py`` must scope a changelog to ONE release.

The changelog this script writes is not a developer convenience: it is
prepended to the package README and becomes the long description PyPI
renders, so a wrong answer is published and permanent.

It was wrong for the whole life of the repository.  The anchor was the
second-most-recent commit whose subject began ``Bump <pkg>``, a convention
never once followed here -- zero matches across 1400+ commits -- so
``_find_previous_bump_sha`` always returned None, the range fell back to
``HEAD``, and every "Changelog" section was the package's ENTIRE history.
jaato-server's first PyPI long description was 91k characters and grew by
about 400 across a release carrying five merged PRs.  Nothing failed; the
artifact was simply wrong, which is why only a test can hold this.

The anchor is now the package's own ``version`` line walked back through
git history.  These tests are hermetic -- each builds a throwaway git repo
-- so they assert the RULE rather than this repository's history, which
would drift under them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# jaato-server/shared/tests/<this file> -> repo root
BUILD_README = Path(__file__).resolve().parents[3] / "scripts" / "build_readme.py"


pytestmark = pytest.mark.skipif(
    not BUILD_README.exists(),
    reason=f"{BUILD_README} not present (package installed without the repo)",
)


def _run(*args: str, cwd: Path) -> str:
    done = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, f"{args} failed:\n{done.stderr}"
    return done.stdout


def _pyproject(version: str, extra: str = "") -> str:
    return (
        "[project]\n"
        'name = "demo-pkg"\n'
        f'version = "{version}"\n'
        f"{extra}"
    )


class _Repo:
    """A throwaway git repo holding one package directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pkg = root / "demo-pkg"
        self.pkg.mkdir(parents=True)
        _run("git", "init", "-q", "-b", "main", cwd=root)
        _run("git", "config", "user.email", "t@example.invalid", cwd=root)
        _run("git", "config", "user.name", "Test", cwd=root)
        (self.pkg / "README.md").write_text("# demo\n")

    def commit(self, subject: str, *, version: str | None = None,
               extra: str = "", touch: bool = True) -> str:
        """Commit, optionally rewriting pyproject to declare ``version``."""
        if version is not None:
            (self.pkg / "pyproject.toml").write_text(_pyproject(version, extra))
        if touch:
            marker = self.pkg / "src.py"
            marker.write_text((marker.read_text() if marker.exists() else "") + subject + "\n")
        _run("git", "add", "-A", cwd=self.root)
        _run("git", "commit", "-q", "-m", subject, cwd=self.root)
        return _run("git", "rev-parse", "HEAD", cwd=self.root).strip()

    def set_version_uncommitted(self, version: str) -> None:
        """Bump pyproject in the WORKING TREE only, as a human previewing a
        release does before committing anything."""
        (self.pkg / "pyproject.toml").write_text(_pyproject(version))

    def changelog(self) -> list[str]:
        """Run the real script and return its changelog bullet lines."""
        done = subprocess.run(
            [sys.executable, str(BUILD_README)],
            cwd=self.pkg, capture_output=True, text=True,
        )
        assert done.returncode == 0, done.stderr
        body = (self.pkg / "PKG_README.md").read_text()
        head = body.split("\n---\n", 1)[0]
        return [ln[2:].strip() for ln in head.splitlines() if ln.startswith("- ")]


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    return _Repo(tmp_path)


def test_the_changelog_starts_at_the_previous_version(repo: _Repo) -> None:
    """Only work done since the PREVIOUS version was set may appear.

    The defect this replaces published everything, forever.
    """
    repo.commit("ancient: before 1.0.0 even existed", version="0.9.0")
    repo.commit("old: shipped in 1.0.0", version="0.9.0")
    repo.commit("Bump demo-pkg 1.0.0", version="1.0.0", touch=False)
    repo.commit("new: the first thing after 1.0.0", version="1.0.0")
    repo.commit("newer: the second thing after 1.0.0", version="1.0.0")
    repo.commit("Bump demo-pkg 1.1.0", version="1.1.0", touch=False)

    entries = repo.changelog()

    assert entries == [
        "newer: the second thing after 1.0.0",
        "new: the first thing after 1.0.0",
    ], entries
    # The two that predate 1.0.0 shipped IN 1.0.0 and must not reappear.
    assert not any("ancient" in e or "old" in e for e in entries)


def test_a_release_does_not_list_its_own_bump(repo: _Repo) -> None:
    """The commit that set the version being built is not news about it.

    Identified by SHA and confirmed by content -- the only files it touched
    inside this package are pyproject.toml -- rather than by its subject.
    A release bumping two packages is ONE commit whose subject cannot name
    only one of them, so a subject rule cannot be relied on here.
    """
    repo.commit("feature: something", version="0.9.0")
    repo.commit("Bump demo-pkg 1.0.0", version="1.0.0", touch=False)
    repo.commit("feature: something else", version="1.0.0")
    # A realistic multi-package bump subject, naming another package first.
    repo.commit("Release other-pkg 3.2.1, demo-pkg 1.1.0",
                version="1.1.0", touch=False)

    entries = repo.changelog()

    assert entries == ["feature: something else"], entries


def test_touching_pyproject_without_changing_the_version_is_not_a_release(
    repo: _Repo,
) -> None:
    """Adding a dependency carries the version forward; it does not set it.

    The anchor must be the OLDEST commit of a run declaring a version, or a
    later no-op edit to pyproject.toml would truncate the changelog to
    nothing.
    """
    repo.commit("Bump demo-pkg 1.0.0", version="1.0.0", touch=False)
    repo.commit("deps: declare requests", version="1.0.0",
                extra='dependencies = ["requests"]\n', touch=False)
    repo.commit("feature: after the dependency edit", version="1.0.0")
    repo.commit("Bump demo-pkg 1.1.0", version="1.1.0", touch=False)

    entries = repo.changelog()

    # "deps:" landed after 1.0.0 was set, so it belongs to 1.1.0's changelog.
    # It is also pyproject-only, and is kept: the pyproject-only test applies
    # ONLY to this release's own bump.  Widening it would silently swallow
    # every dependency change, which is real news.
    assert entries == [
        "feature: after the dependency edit",
        "deps: declare requests",
    ], entries


def test_a_first_release_includes_everything(repo: _Repo) -> None:
    """One version has ever existed: there is no previous release to start
    after, so the whole history is the changelog.  This is the pre-existing
    behaviour for a genuinely first release (jaato-eval 0.1.0), and is what
    a shallow clone degrades to."""
    repo.commit("first: initial import", version="0.1.0")
    repo.commit("second: more work", version="0.1.0")

    entries = repo.changelog()

    assert entries == ["second: more work", "first: initial import"], entries


def test_an_uncommitted_bump_still_anchors_on_the_previous_release(
    repo: _Repo,
) -> None:
    """Previewing a release before committing the bump must not anchor one
    release too early.

    The walk reads committed history, so with the bump uncommitted the newest
    commit still declares the PREVIOUS version.  Starting the walk from that
    commit treats it as "current" and anchors on the one before -- observed
    live: previewing jaato-server 0.10.0 anchored on the 0.8.0 commit and
    re-listed everything that had already shipped in 0.9.0.  The walk starts
    from the working tree's declared version instead.
    """
    repo.commit("old: shipped in 1.0.0", version="0.9.0")
    repo.commit("Bump demo-pkg 1.0.0", version="1.0.0", touch=False)
    repo.commit("feature: after 1.0.0", version="1.0.0")
    repo.set_version_uncommitted("1.1.0")        # not committed

    entries = repo.changelog()

    assert entries == ["feature: after 1.0.0"], entries
    assert not any("old:" in e for e in entries)
