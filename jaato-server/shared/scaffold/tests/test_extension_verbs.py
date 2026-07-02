"""The scaffold verb-extension seam: facade surface + entry-point discovery.

The `compile` verb (and any other external verb) lives outside this repo and
mounts via the `jaato.scaffold_verbs` entry-point group. These tests pin the two
public halves of that contract: the stable :mod:`shared.scaffold.api` facade, and
the CLI discovery loop that mounts a discovered verb and dispatches it. No
external package is required — a fake verb / fake entry point stands in.
"""

import argparse

import pytest

from shared.scaffold import api
from shared.scaffold import __main__ as cli


# --------------------------------------------------------------------------- #
# The stable facade
# --------------------------------------------------------------------------- #

def test_facade_exports_the_documented_surface():
    assert isinstance(api.SCAFFOLD_EXTENSION_API, str)
    assert api.VERB_ENTRY_POINT_GROUP == "jaato.scaffold_verbs"
    # names an external verb is allowed to import
    for name in ("GeneratedFile", "write_files", "emit_then_validate",
                 "validate_workspace", "Diagnostic", "introspect", "ScaffoldVerb"):
        assert hasattr(api, name), name


def test_write_files_and_force(tmp_path):
    files = [api.GeneratedFile(path="a/b.txt", content="hi")]
    written = api.write_files(files, tmp_path)
    assert written == ["a/b.txt"]
    assert (tmp_path / "a/b.txt").read_text() == "hi"
    with pytest.raises(FileExistsError):
        api.write_files(files, tmp_path)               # exists, no force
    api.write_files([api.GeneratedFile("a/b.txt", "bye")], tmp_path, force=True)
    assert (tmp_path / "a/b.txt").read_text() == "bye"


def test_scaffoldverb_protocol_is_runtime_checkable():
    class Ok:
        name = "x"
        help = "h"
        def configure(self, p): ...
        def run(self, a): return 0

    assert isinstance(Ok(), api.ScaffoldVerb)

    class Missing:
        name = "y"
    assert not isinstance(Missing(), api.ScaffoldVerb)


# --------------------------------------------------------------------------- #
# The discovery loop
# --------------------------------------------------------------------------- #

class _FakeVerb:
    name = "faketest"
    help = "a fake external verb"

    def configure(self, parser):
        parser.add_argument("--value", required=True)

    def run(self, args):
        # signal success via a sentinel exit code derived from the arg
        return 7 if args.value == "ok" else 3


class _FakeEP:
    name = "faketest"

    def __init__(self, obj):
        self._obj = obj

    def load(self):
        return self._obj


def _patch_eps(monkeypatch, eps):
    # patch the symbol used inside _discover_external_verbs
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda *a, **k: list(eps),
    )


def test_discovery_mounts_and_dispatches_external_verb(monkeypatch):
    _patch_eps(monkeypatch, [_FakeEP(_FakeVerb)])       # class → instantiated
    rc = cli.main(["faketest", "--value", "ok"])
    assert rc == 7
    rc = cli.main(["faketest", "--value", "no"])
    assert rc == 3


def test_discovery_accepts_instance_entry_point(monkeypatch):
    _patch_eps(monkeypatch, [_FakeEP(_FakeVerb())])     # already an instance
    assert cli.main(["faketest", "--value", "ok"]) == 7


def test_builtin_verbs_win_on_name_collision(monkeypatch):
    class _Evil:
        name = "validate"
        help = "hijack"
        def configure(self, p): ...
        def run(self, a): return 99

    _patch_eps(monkeypatch, [_FakeEP(_Evil())])
    # `validate` must still be the built-in (which errors 2 on a missing target
    # path, never the hijacker's 99).
    with pytest.raises(SystemExit):
        cli.main(["validate"])  # argparse: missing required 'target'


def test_bad_verb_is_skipped_not_fatal(monkeypatch):
    class _Broken:
        name = "broken"
        # no run() → not a valid ScaffoldVerb

    _patch_eps(monkeypatch, [_FakeEP(_Broken())])
    # An unusable verb is skipped; the CLI still works (prints help, rc 0).
    assert cli.main([]) == 0


def test_entry_point_load_failure_is_skipped(monkeypatch):
    class _Raises:
        name = "kaboom"

        def load(self):
            raise RuntimeError("boom")

    _patch_eps(monkeypatch, [_Raises()])
    assert cli.main([]) == 0  # discovery swallows the failure
