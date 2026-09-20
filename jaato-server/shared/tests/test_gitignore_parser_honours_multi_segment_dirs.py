"""Guard: ``GitignoreParser`` answers the way git does.

The daemon's parser is what the workspace monitor's file panel reads, and
since ``jaato-scaffold validate`` judges a workspace ``.gitignore`` by its
EFFECT it is what that verdict rests on too.  It used to approximate git in
three ways that all bite on the one file the scaffold now writes:

* ``*`` crossed ``/`` (``fnmatch``), so ``.jaato/*`` matched every file
  beneath ``.jaato`` outright and a later ``!.jaato/profiles/`` could not
  reach them;
* a multi-segment directory pattern was compared against each path
  SEGMENT on its own, which nothing containing a ``/`` can equal, so
  ``!.jaato/profiles/`` matched nothing at all;
* a directory pattern was read as "the directory and everything beneath"
  for negations too, so ``!.jaato/`` un-ignored ``.jaato/sessions/x``
  where git un-excludes only the directory itself — and, the other way,
  a wholesale ``.jaato/`` followed by ``!.jaato/profiles/`` showed the
  profiles that git hides, because git never descends into an excluded
  directory.

The cases here are git's answers — the emitted block is cross-checked
against ``git check-ignore`` in
``scaffold/tests/test_gitignore_keeps_jaato_assets.py`` — and the
reversion puts the first approximation back.
"""

from __future__ import annotations

from pathlib import Path

from shared.tests.reversion import Reversion
from shared.utils.gitignore import GitignoreParser

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/utils/gitignore.py",
        find='        if c == "*":\n            out.append("[^/]*")',
        replace='        if c == "*":\n            out.append(".*")',
        test="test_star_does_not_cross_a_slash",
        because="`*` crossing `/`, so `.jaato/*` swallowed every file beneath "
                "and a directory re-include could not reach them",
    ),
]


def _parser(tmp_path: Path, text: str) -> GitignoreParser:
    (tmp_path / ".gitignore").write_text(text, encoding="utf-8")
    return GitignoreParser(tmp_path, include_defaults=False)


def test_star_does_not_cross_a_slash(tmp_path):
    """``.jaato/*`` names the direct children; a re-included child
    directory's files are then matched by no rule and are tracked."""
    p = _parser(tmp_path, ".jaato/*\n!.jaato/profiles/\n")
    assert p.is_ignored(tmp_path / ".jaato" / "sessions" / "s.json")
    assert not p.is_ignored(tmp_path / ".jaato" / "profiles" / "a.yaml")
    assert not p.is_ignored(tmp_path / ".jaato" / "profiles" / "set" / "a.yaml")


def test_multi_segment_directory_pattern_ignores_beneath_it(tmp_path):
    p = _parser(tmp_path, "build/out/\n")
    assert p.is_ignored(tmp_path / "build" / "out" / "a.o")
    assert not p.is_ignored(tmp_path / "build" / "a.c")
    # directory-only: a FILE of that name is not matched
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out").write_text("x")
    assert not p.is_ignored(tmp_path / "build" / "out")


def test_single_segment_patterns_match_at_any_depth(tmp_path):
    p = _parser(tmp_path, "__pycache__/\n*.log\n")
    assert p.is_ignored(tmp_path / "a" / "__pycache__" / "m.pyc")
    assert not p.is_ignored(tmp_path / "a" / "m.py")
    assert p.is_ignored(tmp_path / "deep" / "er" / "x.log")


def test_leading_slash_anchors_to_the_root(tmp_path):
    p = _parser(tmp_path, "/build\n")
    assert p.is_ignored(tmp_path / "build")
    assert not p.is_ignored(tmp_path / "src" / "build")


def test_nothing_is_reincluded_beneath_an_excluded_directory(tmp_path):
    """A wholesale ``.jaato/`` beats every later ``!`` beneath it — which is
    why the scaffold's block opens with ``!.jaato/`` rather than relying on
    its re-includes alone."""
    p = _parser(tmp_path, ".jaato/\n!.jaato/profiles/\n")
    (tmp_path / ".jaato").mkdir()
    assert p.is_ignored(tmp_path / ".jaato" / "profiles" / "a.yaml")
    p = _parser(tmp_path, ".jaato/\n!.jaato/\n.jaato/*\n!.jaato/profiles/\n")
    assert not p.is_ignored(tmp_path / ".jaato")
    assert not p.is_ignored(tmp_path / ".jaato" / "profiles" / "a.yaml")
    assert p.is_ignored(tmp_path / ".jaato" / "logs" / "x.log")


def test_a_negated_directory_pattern_unexcludes_only_the_directory(tmp_path):
    """``!.jaato/`` after ``.jaato/*`` leaves the children excluded: the
    negation matches the directory, not the paths beneath it."""
    p = _parser(tmp_path, ".jaato/*\n!.jaato/\n")
    assert p.is_ignored(tmp_path / ".jaato" / "sessions" / "s.json")


def test_double_star_spans_directories(tmp_path):
    p = _parser(tmp_path, ".jaato/**/__pycache__/\n")
    assert p.is_ignored(tmp_path / ".jaato" / "scripts" / "__pycache__" / "m.pyc")
    assert p.is_ignored(tmp_path / ".jaato" / "__pycache__" / "m.pyc")
    assert not p.is_ignored(tmp_path / ".jaato" / "scripts" / "m.py")


def test_extra_patterns_and_defaults_still_apply(tmp_path):
    p = GitignoreParser(tmp_path, extra_patterns=["*.tmp", "!keep.tmp"])
    assert p.is_ignored(tmp_path / ".git" / "HEAD")
    assert p.is_ignored(tmp_path / "a.tmp")
    assert not p.is_ignored(tmp_path / "keep.tmp")
