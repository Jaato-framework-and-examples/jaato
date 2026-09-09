"""The `dependencies` facet — derived, never declared.

The tests that matter are the ones asserting this reports what it MEASURED:
a hardcoded provider-to-package table would pass a naive test and be wrong the
first time someone adds an import.
"""
from pathlib import Path

import pytest

from shared.scaffold import dependencies as D


def test_the_word_is_taken_from_any_position():
    from shared.scaffold.__main__ import _take_deps_word
    assert _take_deps_word("dependencies", None, None) == (None, None, True)
    assert _take_deps_word("provider", "openrouter", "dependencies") == \
        ("provider", "openrouter", True)
    assert _take_deps_word("dependencies", "plugin", "cli") == ("plugin", "cli", True)
    assert _take_deps_word("provider", "openrouter", "deps") == \
        ("provider", "openrouter", True)


def test_a_query_without_the_word_is_untouched():
    from shared.scaffold.__main__ import _take_deps_word
    assert _take_deps_word("provider", "openrouter", None) == \
        ("provider", "openrouter", False)


def test_imports_are_parsed_from_source_not_declared(tmp_path):
    """The whole design: read the code, don't keep a table."""
    f = tmp_path / "m.py"
    f.write_text("import httpx\nfrom openai import OpenAI\n"
                 "import os, json\nfrom . import sibling\nfrom shared import x\n")
    got = D.third_party_imports([f])
    assert got == ["httpx", "openai"]          # stdlib, relative and first-party dropped


def test_a_missing_package_is_reported_not_hidden():
    health = D._health(["sys", "definitely_not_a_real_package_xyz"])
    assert health["sys"] == "importable"
    assert health["definitely_not_a_real_package_xyz"] == "MISSING"


def test_provider_facet_finds_the_implementation():
    d = D.for_provider("openrouter")
    assert d["implementation"] and Path(d["implementation"]).is_dir()
    assert d["files_parsed"] > 0
    # openrouter is an OpenAI-compatible provider; `openai` is also what
    # jaato-server[openrouter] declares, so the derived answer and the declared
    # one agree — which is the point of deriving rather than trusting either.
    assert "openai" in d["imports"]


def test_unknown_plugin_is_an_error_not_an_empty_answer():
    data, text = D.render("plugin", "no-such-plugin-xyz")
    assert "unknown plugin" in text


def test_skew_is_detected_from_the_source_tree(tmp_path, monkeypatch):
    """An editable install whose tree has moved on is the case that matters."""
    src = tmp_path / "pkg"; src.mkdir()
    (src / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "9.9.9"\n')

    class _D:
        @staticmethod
        def read_text(_):
            return '{"dir_info": {"editable": true}, "url": "file://%s"}' % src

    monkeypatch.setattr(D, "version", lambda n: "1.0.0")
    monkeypatch.setattr(D, "distribution", lambda n: _D())
    st = D.dist_state("anything")
    assert st["skew"] and st["installed"] == "1.0.0" and st["source_version"] == "9.9.9"


def test_no_skew_when_metadata_matches_source(tmp_path, monkeypatch):
    src = tmp_path / "pkg"; src.mkdir()
    (src / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n')

    class _D:
        @staticmethod
        def read_text(_):
            return '{"dir_info": {"editable": true}, "url": "file://%s"}' % src

    monkeypatch.setattr(D, "version", lambda n: "1.0.0")
    monkeypatch.setattr(D, "distribution", lambda n: _D())
    assert D.dist_state("anything")["skew"] is False


def test_environment_reports_shadowing(monkeypatch):
    """A PYTHONPATH pointing at a checkout changes which distribution answers,
    so the report has to say the answer may not be the daemon's."""
    monkeypatch.setenv("PYTHONPATH", "/some/checkout")
    assert D.environment()["shadowed"] is True
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert D.environment()["shadowed"] is False


def test_framework_picture_lists_extras_from_metadata():
    d = D.framework_picture()
    assert d["distributions"], "no jaato distributions found"
    assert any("[" in k for k in d["extras"]), "extras should be read from metadata"
