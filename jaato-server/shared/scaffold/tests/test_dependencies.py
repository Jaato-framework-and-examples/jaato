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


# --------------------------------------------------- the closure, and why it exists
#
# Most providers own no `import openai` — they inherit the transport that does.
# A scan of `<provider>/*.py` therefore reported `openai` only for the handful
# that own their streaming loop and stayed silent for every other one, including
# the provider named `openai`, while the runtime died on exactly that import.

_PROVIDERS = Path(__file__).resolve().parents[2] / "plugins" / "model_provider"


def test_a_provider_reports_what_the_transport_it_inherits_imports():
    d = D.for_provider("azure_openai")
    assert "openai" in d["imports"], \
        "azure_openai's runtime failure is `openai`; the report must name it"
    assert d["via"]["openai"], "and must say the import comes from shared machinery"
    assert any("_openai_compat" in p for p in d["via"]["openai"])


def test_the_provider_named_openai_reports_openai():
    """The narrowest case: it owns no `import openai` either."""
    assert "openai" in D.for_provider("openai")["imports"]


def test_every_openai_compatible_provider_reports_the_sdk():
    """Derived, so a provider added tomorrow is covered without editing this.

    Inheriting `_openai_compat` IS the dependency on `openai`; a provider that
    inherits it and does not report the package is under-reporting.
    """
    inheritors = []
    for pkg in sorted(p for p in _PROVIDERS.iterdir() if p.is_dir()):
        if pkg.name.startswith("_") or pkg.name == "tests":
            continue
        src = "\n".join(f.read_text(encoding="utf-8", errors="replace")
                        for f in pkg.glob("*.py"))
        if "OpenAICompatProvider" in src:
            inheritors.append(pkg.name)
    assert len(inheritors) > 5, "expected the OpenAI-compatible fleet, found %r" % inheritors
    silent = [n for n in inheritors if "openai" not in D.for_provider(n)["imports"]]
    assert not silent, f"providers under-reporting the openai SDK: {silent}"


def test_an_import_the_unit_owns_is_not_marked_via():
    """`via` separates "your source imports this" from "what you inherit does"."""
    d = D.for_provider("bedrock")
    assert "boto3" in d["imports"] and "boto3" not in d["via"]


def test_the_closure_is_bounded_to_the_family_directory():
    """Unbounded, `subagent` reached 84 files and claimed `anthropic` and
    `google` — true of the framework, useless as an answer about the plugin."""
    d = D.for_plugin("subagent", "shared.plugins.subagent")
    assert d["files_reached"] < 40
    assert "anthropic" not in d["imports"] and "google" not in d["imports"]


def test_a_plugin_is_located_without_importing_it(monkeypatch):
    """The facet must work when a dependency is missing — that is when it is
    asked.  Importing to find the file fails in exactly that case."""
    monkeypatch.setattr(D, "_module_file",
                        lambda _: pytest.fail("an in-tree plugin must not be imported"))
    d = D.for_plugin("memory", "create_plugin (shared.plugins.memory)")
    assert d["files_parsed"] > 0 and d["note"] is None


def test_single_file_scan_still_follows_nothing(tmp_path):
    """`third_party_imports` is the honest per-file answer the framework
    picture asks for; only the unit facets take the closure."""
    (tmp_path / "a.py").write_text("from .b import thing\n")
    (tmp_path / "b.py").write_text("import httpx\n")
    assert D.third_party_imports([tmp_path / "a.py"]) == []
    names, walked = D.import_closure([tmp_path / "a.py"], tmp_path)
    assert list(names) == ["httpx"] and len(walked) == 2


def test_the_closure_does_not_leave_its_scope(tmp_path):
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir(); outside.mkdir()
    (inside / "a.py").write_text("from shared.plugins.x import y\nimport httpx\n")
    names, walked = D.import_closure([inside / "a.py"], inside)
    assert list(names) == ["httpx"] and len(walked) == 1


# ------------------------------------------------------ "why wasn't it installed?"


def test_a_missing_import_names_the_extra_that_declares_it():
    d = D.for_provider("azure_openai")
    if not D._extras_index():
        pytest.skip("jaato-server metadata is not installed in this environment")
    assert "jaato-server[azure-openai]" in d["extras"]["openai"]


def test_the_install_hint_prefers_the_extra_named_after_the_unit():
    """`openai` is declared by five extras; only one belongs to this provider."""
    if not D._extras_index():
        pytest.skip("jaato-server metadata is not installed in this environment")
    hint = D._install_hint(["openai", "azure"], "azure_openai")
    assert hint["named_after_unit"] is True
    assert hint["commands"] == ["pip install 'jaato-server[azure-openai]'"]


def test_no_extra_named_after_the_unit_is_said_so_not_implied():
    """Ten providers have no extra of their own; recommending `[nim]` for
    `minimax` without saying why would read as a claim about ownership."""
    if not D._extras_index():
        pytest.skip("jaato-server metadata is not installed in this environment")
    hint = D._install_hint(["openai"], "minimax")
    assert hint["named_after_unit"] is False and hint["commands"]
    text = D._render_unit(D.for_provider("minimax"))
    assert "openai" in text


def test_a_core_requirement_is_an_incomplete_install_not_a_missing_extra():
    if not D._core_index():
        pytest.skip("jaato-server metadata is not installed in this environment")
    hint = D._install_hint(["httpx"], "anything")
    assert hint["incomplete_install"] == ["httpx"]
    assert not any("httpx" in c for c in hint["commands"])


def test_requires_survives_a_distribution_that_is_not_installed():
    assert D._requires("definitely-not-installed-xyz") == []


def test_a_plugin_discovery_skipped_is_still_reported(monkeypatch):
    """Discovery imports, so a plugin with a missing dependency is skipped —
    and that is precisely the plugin this facet is asked about.  Answering
    "unknown plugin" there withholds the report at the only useful moment."""
    from shared.scaffold import introspect
    monkeypatch.setattr(introspect, "plugins", lambda *a, **k: {})
    data, text = D.render("plugin", "memory")
    assert "error" not in data and data["files_parsed"] > 0
    assert "discovery skipped" in text


def test_a_name_that_is_no_plugin_at_all_is_still_an_error(monkeypatch):
    from shared.scaffold import introspect
    monkeypatch.setattr(introspect, "plugins", lambda *a, **k: {})
    data, text = D.render("plugin", "no-such-plugin-xyz")
    assert "unknown plugin" in text and "error" in data
